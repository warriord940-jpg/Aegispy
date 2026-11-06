import time
import torch
import concurrent.futures
import numpy as np

def cpu_heavy_computation(data_size):
    """Heavy numerical processing routed to CPU core pool."""
    print(f"  [CPU Engine] Starting heavy matrix multiplication ({data_size}x{data_size}) on Ryzen CPU...", flush=True)
    start_cpu = time.time()
    
    # Perform heavy CPU-bound math using NumPy
    arr = np.random.rand(data_size, data_size)
    result_matrix = np.dot(arr, arr)
    
    elapsed_cpu = time.time() - start_cpu
    print(f"  [CPU Engine] Completed in {elapsed_cpu:.2f} seconds.", flush=True)
    return f"CPU_Matrix_Shape_{result_matrix.shape}"

def gpu_heavy_computation(tensor_size):
    """Parallel tensor calculations routed to the NVIDIA RTX 2050."""
    print(f"  [GPU Engine] Starting tensor processing ({tensor_size}x{tensor_size}) on NVIDIA RTX 2050...", flush=True)
    start_gpu = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        # Utilize an explicit CUDA stream for asynchronous parallel execution
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(tensor_size, tensor_size, device=device)
            y = torch.matmul(x, x)
            torch.cuda.synchronize()  # Wait for GPU kernel completion
        hardware_card = torch.cuda.get_device_name(0)
    else:
        hardware_card = "CPU Fallback"
        
    elapsed_gpu = time.time() - start_gpu
    print(f"  [GPU Engine] Completed in {elapsed_gpu:.2f} seconds.", flush=True)
    return f"GPU_Tensor_Processed [{hardware_card}]"

if __name__ == "__main__":
    print("=======================================================", flush=True)
    print("  AEGISPY: HETEROGENEOUS (CPU + GPU) CONCURRENT ENGINE ", flush=True)
    print("=======================================================", flush=True)
    print("  * Task 1: cpu_heavy_computation -> Routed to CPU Pool", flush=True)
    print("  * Task 2: gpu_heavy_computation -> Routed to RTX 2050", flush=True)
    print("-------------------------------------------------------", flush=True)
    print("\n--- DISPATCHING CONCURRENT PIPELINE ---", flush=True)
    
    pipeline_start = time.time()
    
    # Using ThreadPoolExecutor to run CPU and GPU tasks concurrently.
    # (Python releases the Global Interpreter Lock [GIL] during heavy C-extensions 
    # and PyTorch CUDA kernel execution, allowing true parallel hardware utilization).
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_cpu = executor.submit(cpu_heavy_computation, data_size=2500)
        future_gpu = executor.submit(gpu_heavy_computation, tensor_size=4000)
        
        # Gather results asynchronously as they finish
        cpu_output = future_cpu.result()
        gpu_output = future_gpu.result()
        
    total_elapsed = time.time() - pipeline_start

    print("\n=======================================================", flush=True)
    print("  HETEROGENEOUS EXECUTION BENCHMARK RESULTS            ", flush=True)
    print("=======================================================", flush=True)
    print(f"Total Concurrent Pipeline Execution Time: {total_elapsed:.2f} seconds\n", flush=True)
    print(f"  [✔] CPU Task Result : {cpu_output}", flush=True)
    print(f"  [✔] GPU Task Result : {gpu_output}", flush=True)
    print("=======================================================\n", flush=True)