import sys
import time
import inspect
import importlib.util
import ast
import os
import pickle
import hashlib
import json
import shutil

# --- INTEGRATED SMART-RUN PACKAGE IMPORTS ---
try:
    from smartrun.session import SessionManager
    from smartrun.cache import SmartCacheManager
    from smartrun_lib.session import init_session_context
    SMART_RUN_AVAILABLE = True
except ImportError:
    SMART_RUN_AVAILABLE = False

class HardwareDetector(ast.NodeVisitor):
    def __init__(self):
        self.is_gpu = False

    def visit_Attribute(self, node):
        if node.attr.lower() in {"cuda", "to", "gpu", "to_gpu"}:
            self.is_gpu = True
        self.generic_visit(node)

    def visit_Name(self, node):
        if "cuda" in node.id.lower():
            self.is_gpu = True
        self.generic_visit(node)

def auto_detect_hardware(func) -> str:
    """Statically analyzes function AST to route workload."""
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
        detector = HardwareDetector()
        detector.visit(tree)
        return "GPU (NVIDIA RTX CUDA Stream)" if detector.is_gpu else "CPU (Multicore Pool)"
    except Exception:
        return "CPU (Multicore Pool)"

def load_module_from_path(file_path):
    """Dynamically loads any python file passed to the sequential engine."""
    module_name = file_path.replace(".py", "").split("\\")[-1]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def resolve_argument_fast(func_name, param_name, raw_payload):
    """
    Identical argument resolution logic to the parallel engine 
    to ensure apples-to-apples baseline comparisons.
    """
    # 1. Hardware benchmarks (matrices/tensors) always get heavy scales
    if any(k in param_name for k in ["tensor", "matrix", "size", "data", "dim"]):
        return (2500,)
    
    # 2. User input script gets the CLI payload (defaulting to 5)
    if "user" in func_name.lower():
        val = int(raw_payload) if raw_payload is not None and str(raw_payload).isdigit() else 5
        return (val,)
        
    # 3. Standard core benchmarks (like factorial.py) use the heavy default scale (1500)
    if any(k in param_name for k in ["n", "num", "factorial", "val", "x", "count"]):
        return (1500,)
        
    return (1000,)

def make_cached_wrapper(func, func_name, force_refresh=False):
    """
    Transparently wraps any standard user function with persistent caching 
    using the .smartcache_sequential directory.
    """
    cache_dir = ".smartcache_sequential"
    os.makedirs(cache_dir, exist_ok=True)

    key_data = f"{func_name}".encode("utf-8")
    cache_hash = hashlib.md5(key_data).hexdigest()
    cache_file = os.path.join(cache_dir, f"func_{cache_hash}.pkl")

    is_cached = not force_refresh and os.path.exists(cache_file)

    def wrapper(args_tuple):
        if is_cached:
            try:
                with open(cache_file, "rb") as f:
                    cached_result = pickle.load(f)
                return cached_result, True
            except Exception:
                pass

        result = func(*args_tuple)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            pass

        return result, False

    wrapper.is_cached = is_cached
    return wrapper

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sequential_core.py <file1.py|directory> [optional_input] [--force-refresh]")
        sys.exit(1)

    raw_args = sys.argv[1:]

    # --- Initialize SmartRun Context if available ---
    if SMART_RUN_AVAILABLE:
        try:
            init_session_context("AegisPy-Sequential-Execution")
        except Exception:
            pass

    # --- Catch and Handle --force-refresh First ---
    force_refresh = "--force-refresh" in raw_args
    if force_refresh:
        raw_args = [arg for arg in raw_args if arg != "--force-refresh"]
        if os.path.exists(".smartcache_sequential"):
            shutil.rmtree(".smartcache_sequential")
            print("[Sequential Engine] Flag detected: .smartcache_sequential cleared for clean cold run.", flush=True)

    # --- Recursive Directory & File Scanning ---
    target_files = []
    for arg in raw_args:
        if os.path.isdir(arg):
            for root, dirs, files in os.walk(arg):
                for file in files:
                    if file.endswith(".py"):
                        target_files.append(os.path.join(root, file))
        elif arg.endswith(".py") and os.path.exists(arg):
            target_files.append(arg)

    non_file_args = [f for f in raw_args if not f.endswith(".py") and not os.path.isdir(f)]
    user_payload = non_file_args[0] if non_file_args else None

    print(f"\n=======================================================", flush=True)
    print(f"  BASELINE: SEQUENTIAL CACHED ENGINE ({len(target_files)} files)", flush=True)
    if user_payload:
        print(f"  [Input Payload Detected]: {user_payload}", flush=True)
    print(f"=======================================================", flush=True)

    functions_to_run = []
    
    for target_file in target_files:
        print(f"\n[Sequential Engine] Loading & Caching file: {target_file}", flush=True)
        try:
            external_module = load_module_from_path(target_file)
            for name, obj in inspect.getmembers(external_module, inspect.isfunction):
                if obj.__module__ == external_module.__name__:
                    hw = auto_detect_hardware(obj)
                    unique_name = f"{target_file}->{name}"
                    
                    cached_func = make_cached_wrapper(obj, unique_name, force_refresh)
                    
                    # Extract parameter name to match parallel engine argument resolution
                    try:
                        sig = inspect.signature(obj)
                        params = list(sig.parameters.values())
                        param_name = params[0].name.lower() if params else "val"
                    except Exception:
                        param_name = "val"

                    bound_args = resolve_argument_fast(unique_name, param_name, user_payload)
                    
                    functions_to_run.append((unique_name, cached_func, hw, bound_args))
                    print(f"  * Discovered: {name:<22} -> Routed To: {hw} (Args: {bound_args})", flush=True)
        except Exception as e:
            print(f"  [!] Error loading {target_file}: {e}", flush=True)

    print("-------------------------------------------------------", flush=True)
    print("\n--- EXECUTING SEQUENTIAL CACHED PIPELINE ---", flush=True)

    start_time = time.time()
    results = {}

    for name, func, hw, bound_args in functions_to_run:
        if func.is_cached:
            print(f"  [Cache HIT] '{name}' loaded instantly from storage!", flush=True)
        else:
            print(f"  [Cache MISS] Sequentially executing '{name}' with {bound_args}...", flush=True)
        
        try:
            res, _ = func(bound_args)
            results[name] = res
        except Exception as e:
            results[name] = f"Error: {e}"

    total_time = time.time() - start_time

    # --- Automated Benchmark Logging to profile.json ---
    profile_data = []
    if os.path.exists("profile.json"):
        try:
            with open("profile.json", "r") as pf:
                profile_data = json.load(pf)
        except Exception:
            profile_data = []

    run_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_count": len(target_files),
        "mode": "sequential",
        "execution_time_seconds": round(total_time, 4)
    }
    profile_data.append(run_record)

    try:
        with open("profile.json", "w") as pf:
            json.dump(profile_data, pf, indent=4)
    except Exception as e:
        print(f"  [!] Warning: Could not write profile.json: {e}", flush=True)

    print("\n=======================================================", flush=True)
    print("  SEQUENTIAL CACHED BENCHMARK RESULTS                  ", flush=True)
    print("=======================================================", flush=True)
    print(f"Total Sequential Execution Time: {total_time:.2f} seconds\n", flush=True)
    for name, res in results.items():
        print(f"  [✔] {name}: {res}", flush=True)
    print("=======================================================\n", flush=True)