"""Session: DAG scheduler + cache + CPU(thread/process)/GPU lanes + liveness."""
from __future__ import annotations

import ast
import atexit
import contextlib
import importlib.metadata as md
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait)

from .. import _worker
from .analysis import GPU_LIBS, NUMERIC_LIBS, split_blocks
from ..cache import Cache, Unstorable, cp, h

VERSION = "0.2.0"


class BlockError(RuntimeError):
    """A user block raised. The original exception is chained as __cause__."""

    def __init__(self, block, exc):
        self.block = block
        super().__init__(f"block {block.idx} (line {block.node.lineno}) failed: "
                         f"{type(exc).__name__}: {exc}\n    {block.src.splitlines()[0][:100]}")


class _Fallback(Exception):
    pass


class _Run:
    """State of one Session.run() call."""

    def __init__(self, blocks, ns, last_writer, outputs):
        self.blocks, self.ns, self.final, self.outputs = blocks, ns, last_writer, outputs
        self.lock = threading.RLock()
        self.loaded, self.version = set(), {}        # (block,name) in ns / name -> block
        self.meta = {}                               # cached-but-not-yet-loaded blocks
        self.lazy, self.lazy_done = set(), set()     # defs/imports run only when needed
        self.records = {}
        self.protect = set()
        self.initial = set(ns)
        self.consumers = defaultdict(set)
        for b in blocks:
            for n, s in b.sources.items():
                self.consumers[(s, n)].add(b.idx)
        self._ext = {}

    def keep(self, s, n):
        return self.final.get(n) == s and (self.outputs is None or n in self.outputs)

    def ext_hash(self, n):
        if n not in self._ext:
            try:
                self._ext[n] = h(cp.dumps(self.ns[n]))
            except Exception:
                self._ext[n] = os.urandom(8).hex()
        return self._ext[n]


class Session:
    def __init__(self, cache_dir=".smartcache", workers=None, use_processes=True, cache=True,
                 strict_calls=False, max_cache_mb=2048, process_threshold=0.02,
                 mp_start="spawn", verbose=False):
        self.workers = workers or os.cpu_count() or 2
        self.use_processes, self.strict_calls = use_processes, strict_calls
        self.process_threshold, self.verbose, self.mp_start = process_threshold, verbose, mp_start
        self.cache = Cache(cache_dir, max_cache_mb << 20) if cache else None
        self.cache_dir = cache_dir
        self._procs_pool = None
        self._gpus = None
        self._gpu_q = None
        self._vc, self._pd = {}, None
        self._prof_path = os.path.join(cache_dir, "profile.json")
        self.prof = {}
        if cache:
            try:
                with open(self._prof_path) as f:
                    self.prof = json.load(f)
            except Exception:
                self.prof = {}
        self.stats = {"executed": [], "cached": [], "skipped": []}
        self._last = None
        atexit.register(self.close)

    # ------------------------------------------------------------- public API
    def run(self, code, outputs=None, ns=None):
        """Execute `code`. Returns the resulting namespace (or only `outputs`).

        outputs : names you need afterwards. Everything else is freed as soon as
                  its last consumer has run, and unneeded blocks/imports are skipped
                  entirely when their results are cached (lowest memory + time).
        ns      : optional dict to use as globals (e.g. IPython's user_ns)."""
        blocks, _aliases, last_writer = split_blocks(code, self.strict_calls)
        ns = {} if ns is None else ns
        ns.setdefault("__name__", "__smartrun__")
        run = _Run(blocks, ns, last_writer, set(outputs) if outputs is not None else None)
        for b in blocks:
            b.code = compile(ast.Module([b.node], []), f"<smartrun block {b.idx}>", "exec")
        self._schedule(run)
        with run.lock:                                   # materialize what the caller wants
            for n, s in last_writer.items():
                if run.outputs is None or n in run.outputs:
                    self._materialize(s, n, run)
        if self.cache:
            self.cache.evict(protect=run.protect)
            self._save_profile()
        self._finish(run)
        if run.outputs is not None:
            return {n: ns[n] for n in run.outputs if n in ns}
        return ns if len(run.initial) > 1 else {k: v for k, v in ns.items()
                                                 if not (k.startswith("__") and k.endswith("__"))}

    def plan(self, code):
        """Static plan without executing anything."""
        blocks, _, _ = split_blocks(code, self.strict_calls)
        rows = ["  #  line  kind    pure  lane     deps        code"]
        for b in blocks:
            rows.append(f"{b.idx:3d}  {b.node.lineno:4d}  {b.kind:6s}  {str(b.pure):5s} "
                        f"{self._lane(b, plan=True):8s} {str(sorted(b.deps)):11s} "
                        f"{b.src.splitlines()[0][:50]}")
        return "\n".join(rows)

    def report(self):
        if not self._last:
            return ""
        blocks, records, lazy_done = self._last
        rows = ["  #  line  status    lane     secs   code"]
        for b in blocks:
            st, lane, dt = records.get(b.idx, ("?", "-", 0.0))
            if st == "lazy":
                st = "executed" if b.idx in lazy_done else "skipped"
            rows.append(f"{b.idx:3d}  {b.node.lineno:4d}  {st:8s}  {lane:7s} {dt:6.2f}   "
                        f"{b.src.splitlines()[0][:50]}")
        return "\n".join(rows)

    def clear_cache(self):
        if self.cache:
            self.cache.clear()
        self.prof = {}

    def close(self):
        if self._procs_pool is not None:
            self._procs_pool.shutdown(wait=False, cancel_futures=True)
            self._procs_pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------- scheduling
    def _schedule(self, run):
        blocks = run.blocks
        est = {b.idx: self.prof.get(b.sig, {}).get("t", 0.01) for b in blocks}
        prio = {}
        for b in reversed(blocks):                        # critical-path (bottom-level) priority
            prio[b.idx] = est[b.idx] + max((prio[d] for d in b.dependents), default=0.0)
        indeg = {b.idx: len(b.deps) for b in blocks}
        ready = [i for i, k in indeg.items() if k == 0]
        cpu = ThreadPoolExecutor(max(4, self.workers * 2), thread_name_prefix="smartrun")
        gpu, running, error = None, {}, None
        try:
            while ready or running:
                if error is None:
                    for i in sorted(ready, key=lambda i: -prio[i]):
                        b = blocks[i]
                        lane = self._lane(b)
                        if lane == "gpu":
                            gpu = gpu or ThreadPoolExecutor(max(1, self._gpus[0]))
                            fut = gpu.submit(self._task, b, run, lane)
                        else:
                            fut = cpu.submit(self._task, b, run, lane)
                        running[fut] = i
                ready = []
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for f in done:
                    i = running.pop(f)
                    exc = f.exception()
                    if exc is not None:
                        error = error or exc
                        continue
                    for d in blocks[i].dependents:
                        indeg[d] -= 1
                        if indeg[d] == 0 and error is None:
                            ready.append(d)
        finally:
            cpu.shutdown(wait=True)
            if gpu:
                gpu.shutdown(wait=True)
        if error is not None:
            raise error

    def _lane(self, b, plan=False):
        m = b.markers
        if b.kind != "stmt" or b.is_callable:
            return "thread"
        if "gpu" in m or (b.libs & GPU_LIBS and "cpu" not in m):
            if plan or self._gpu_count():
                return "gpu"
        if "process" in m:
            return "process" if self.use_processes else "thread"
        if "thread" in m or not b.pure or not self.use_processes:
            return "thread"
        if b.loops and not (b.libs & NUMERIC_LIBS):        # pure-Python CPU work: dodge the GIL
            t = self.prof.get(b.sig, {}).get("t")
            return "thread" if (t is not None and t < self.process_threshold) else "process"
        return "thread"

    # ------------------------------------------------------------- one block
    def _task(self, b, run, lane):
        t0 = time.perf_counter()
        try:
            status, used = self._resolve(b, run, lane)
        except BlockError:
            raise
        except Exception as e:
            raise BlockError(b, e) from e
        dt = time.perf_counter() - t0
        run.records[b.idx] = (status, used, dt)
        if status == "executed" and b.kind == "stmt":
            old = self.prof.get(b.sig, {}).get("t")
            self.prof[b.sig] = {"t": dt if old is None else 0.5 * old + 0.5 * dt}
        self._release(b, run)

    def _resolve(self, b, run, lane):
        ns = run.ns
        self._make_key(b, run)
        if b.kind != "stmt":
            b.out_hash = h(b.key, "def", self._versions(b))
            if b.kind == "import" or not (b.now_reads & set(b.sources)):
                run.lazy.add(b.idx)
                run.records[b.idx] = ("lazy", "-", 0.0)
                return "lazy", "-"
            with run.lock:
                self._ensure(b, run, names=b.now_reads)
                exec(b.code, ns)
                self._mark(b, run)
            return "executed", "thread"

        cacheable = (self.cache is not None and b.pure and "nocache" not in b.markers
                     and not b.is_callable)
        if cacheable:
            run.protect.add(b.key)
            meta = self.cache.lookup(b.key)
            if meta is not None and not meta["volatile"]:
                b.out_hash = meta["hash"]
                run.meta[b.idx] = meta
                return "cached", "-"
        used = self._exec(b, run, lane)
        vals = {n: ns[n] for n in b.writes if n in ns}
        if b.is_callable:
            b.out_hash = h(b.key, "def")
        elif cacheable:
            try:
                b.out_hash = self.cache.store(b.key, vals)
            except Unstorable:
                b.out_hash = h(b.key, "volatile")
                self.cache.mark_volatile(b.key, b.out_hash)
        else:
            try:
                b.out_hash = h(*[(n, cp.dumps(v)) for n, v in sorted(vals.items())])
            except Exception:
                b.out_hash = h(b.key, "volatile") if b.pure else h(b.key, os.urandom(8).hex())
        return "executed", used

    def _make_key(self, b, run):
        fparts = [self._stat(p) for p in sorted(b.files)]
        for e in b.file_exprs:
            try:
                names = {x.id for x in ast.walk(e) if isinstance(x, ast.Name)}
                with run.lock:
                    self._ensure(b, run, names=names)
                    v = eval(compile(ast.Expression(e), "<smartrun>", "eval"), run.ns)
                fparts.append(self._stat(os.fspath(v)))
            except Exception:
                b.pure = False
        ext = [(n, run.ext_hash(n)) for n in sorted(b.eff)
               if n not in b.sources and n in run.initial]
        b.key = h(VERSION, sys.version_info[:3], ast.dump(b.node),
                  sorted((n, run.blocks[s].out_hash) for n, s in b.sources.items()), fparts, ext)

    @staticmethod
    def _stat(p):
        try:
            st = os.stat(p)
            return (p, st.st_mtime_ns, st.st_size)
        except OSError:
            return (p, None)

    def _versions(self, b):
        if b.kind != "import":
            return ""
        mods = []
        for n in ast.walk(b.node):
            if isinstance(n, ast.Import):
                mods += [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods.append(n.module.split(".")[0])
        return [(m, self._modver(m)) for m in sorted(set(mods))]

    def _modver(self, mod):
        if mod in self._vc:
            return self._vc[mod]
        v = None
        try:
            if self._pd is None:
                self._pd = md.packages_distributions()
            v = md.version((self._pd.get(mod) or [mod])[0])
        except Exception:
            pass
        if v is None and mod in getattr(sys, "stdlib_module_names", ()):
            v = sys.version
        if v is None:                                     # local module: use file mtime
            try:
                spec = importlib.util.find_spec(mod)
                st = os.stat(spec.origin)
                v = f"file:{st.st_mtime_ns}:{st.st_size}"
            except Exception:
                v = "?"
        self._vc[mod] = v
        return v

    # ------------------------------------------------------------- execution lanes
    def _exec(self, b, run, lane):
        with run.lock:
            self._ensure(b, run)
        if lane == "process":
            try:
                self._exec_process(b, run)
                self._mark(b, run)
                return "process"
            except _Fallback:
                lane = "thread"
        if lane == "gpu":
            with self._gpu_ctx():
                exec(b.code, run.ns)
        else:
            exec(b.code, run.ns)
        self._mark(b, run)
        return lane

    def _exec_process(self, b, run):
        inputs = {n: run.ns[n] for n in b.eff if n in run.ns}
        try:
            payload = cp.dumps((b.src, sorted(b.writes), inputs))
        except Exception:
            raise _Fallback()
        try:
            status, data = self._procs().submit(_worker.run_block, payload).result()
        except Exception:                                  # broken pool etc.
            self.close()
            raise _Fallback()
        if status == "ok":
            run.ns.update(cp.loads(data))
        elif status == "unpicklable":
            raise _Fallback()
        else:
            raise RuntimeError("in worker process:\n" + data)

    def _procs(self):
        if self._procs_pool is None:
            self._procs_pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context(self.mp_start))
        return self._procs_pool

    def _gpu_count(self):
        if self._gpus is None:
            n, backend = 0, None
            try:
                if importlib.util.find_spec("torch"):
                    import torch
                    if torch.cuda.is_available():
                        n, backend = torch.cuda.device_count(), "torch"
                if n == 0 and importlib.util.find_spec("cupy"):
                    import cupy
                    n, backend = cupy.cuda.runtime.getDeviceCount(), "cupy"
            except Exception:
                n = 0
            self._gpus = (n, backend)
            self._gpu_q = queue.Queue()
            for i in range(n):
                self._gpu_q.put(i)
        return self._gpus[0]

    @contextlib.contextmanager
    def _gpu_ctx(self):
        dev = self._gpu_q.get()                            # one in-flight task per device
        try:
            backend = self._gpus[1]
            if backend == "torch":
                import torch
                with torch.cuda.device(dev):
                    yield dev
            elif backend == "cupy":
                import cupy
                with cupy.cuda.Device(dev):
                    yield dev
            else:
                yield dev
        finally:
            self._gpu_q.put(dev)

    # ------------------------------------------------------------- namespace management
    def _mark(self, b, run):
        with run.lock:
            for n in b.writes:
                if n in run.ns:
                    run.loaded.add((b.idx, n))
                    run.version[n] = b.idx

    def _ensure(self, b, run, names=None):
        """Make sure every input of `b` is present in the namespace (lazy loads)."""
        with run.lock:
            for n, s in b.sources.items():
                if names is None or n in names:
                    self._materialize(s, n, run)

    def _materialize(self, s, n, run):
        if (s, n) in run.loaded:
            return
        sb = run.blocks[s]
        if s in run.meta:                                   # cached stmt: load this variable only
            meta = run.meta[s]
            if n in meta["names"]:
                run.ns[n] = self.cache.load(sb.key, meta, n)
                run.version[n] = s
            run.loaded.add((s, n))
        elif s in run.lazy:                                 # def/import: execute on first use
            if s not in run.lazy_done:
                exec(sb.code, run.ns)
                run.lazy_done.add(s)
                self._mark(sb, run)

    def _release(self, b, run):
        """Liveness: drop values nobody will read again."""
        with run.lock:
            for n, s in b.sources.items():
                run.consumers[(s, n)].discard(b.idx)
                self._maybe_free(s, n, run)
            for n in b.writes:
                self._maybe_free(b.idx, n, run)

    def _maybe_free(self, s, n, run):
        if run.consumers.get((s, n)) or run.keep(s, n):
            return
        sb = run.blocks[s]
        if sb.kind != "stmt" or sb.is_callable:
            return
        if run.version.get(n) == s and (s, n) in run.loaded:
            run.ns.pop(n, None)
            run.loaded.discard((s, n))
            run.version.pop(n, None)

    # ------------------------------------------------------------- bookkeeping
    def _save_profile(self):
        try:
            tmp = self._prof_path + f".{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(self.prof, f)
            os.replace(tmp, self._prof_path)
        except Exception:
            pass

    def _finish(self, run):
        self._last = (run.blocks, dict(run.records), set(run.lazy_done))
        ex, ca, sk = [], [], []
        for b in run.blocks:
            st = run.records.get(b.idx, ("?",))[0]
            if st == "lazy":
                (ex if b.idx in run.lazy_done else sk).append(b.idx)
            elif st == "executed":
                ex.append(b.idx)
            elif st == "cached":
                ca.append(b.idx)
        self.stats = {"executed": ex, "cached": ca, "skipped": sk}
        if self.verbose:
            print(self.report())


def run(code, **kw):
    """One-shot helper: smartrun.run(code, cache_dir=..., outputs=[...])."""
    run_kw = {k: kw.pop(k) for k in ("outputs", "ns") if k in kw}
    with Session(**kw) as s:
        return s.run(code, **run_kw)
