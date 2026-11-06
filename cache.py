"""On-disk, per-block, per-variable cache with atomic writes, mmap loading of
large arrays, corruption tolerance and LRU eviction."""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import shutil
import time
import uuid

try:
    import cloudpickle as cp
except ImportError:  # functions/lambdas won't be cacheable, everything else works
    cp = pickle
try:
    import numpy as np
except ImportError:
    np = None

BIG = 1 << 20  # arrays >= 1 MiB are stored as .npy and loaded memory-mapped


def h(*parts) -> str:
    m = hashlib.blake2b(digest_size=20)
    for p in parts:
        m.update(p if isinstance(p, bytes) else repr(p).encode())
        m.update(b"\0")
    return m.hexdigest()


class Unstorable(Exception):
    """Value cannot be cached (unpicklable, or bound to the exec'd namespace)."""


def _hash_file(path):
    m = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


class Cache:
    def __init__(self, root, max_bytes=2 << 30, by_value_markers=(b"__smartrun__", b"__main__")):
        self.root, self.max_bytes = root, max_bytes
        self.markers = tuple(by_value_markers)
        os.makedirs(root, exist_ok=True)

    def _dir(self, key):
        return os.path.join(self.root, key)

    # ---- read side
    def lookup(self, key):
        p = os.path.join(self._dir(key), "meta.pkl")
        try:
            with open(p, "rb") as f:
                meta = pickle.load(f)
            os.utime(p)                      # LRU touch
            return meta
        except Exception:
            return None

    def load(self, key, meta, name):
        p = os.path.join(self._dir(key), meta["names"][name])
        if p.endswith(".npy"):
            return np.load(p, mmap_mode="c")  # copy-on-write: cheap, and mutation-safe
        with open(p, "rb") as f:
            return pickle.load(f)

    # ---- write side
    def store(self, key, values: dict) -> str:
        for v in values.values():
            if inspect.isroutine(v) or inspect.isclass(v) or inspect.ismodule(v):
                raise Unstorable("routine/class/module")
        tmp = os.path.join(self.root, f".tmp-{uuid.uuid4().hex}")
        os.makedirs(tmp)
        try:
            names, hashes = {}, []
            for i, (n, v) in enumerate(sorted(values.items())):
                if np is not None and type(v) is np.ndarray and not v.dtype.hasobject and v.nbytes >= BIG:
                    fname = f"v{i}.npy"
                    np.save(os.path.join(tmp, fname), v, allow_pickle=False)
                else:
                    fname = f"v{i}.pkl"
                    try:
                        data = cp.dumps(v, protocol=5)
                    except Exception as e:
                        raise Unstorable(str(e)) from e
                    if any(m in data for m in self.markers):
                        raise Unstorable("pickled by value from exec namespace")
                    with open(os.path.join(tmp, fname), "wb") as f:
                        f.write(data)
                names[n] = fname
                hashes.append((n, _hash_file(os.path.join(tmp, fname))))
            out = h(*hashes)
            self._commit(tmp, key, {"names": names, "hash": out, "volatile": False})
            return out
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def mark_volatile(self, key, out_hash):
        tmp = os.path.join(self.root, f".tmp-{uuid.uuid4().hex}")
        os.makedirs(tmp)
        self._commit(tmp, key, {"names": {}, "hash": out_hash, "volatile": True})

    def _commit(self, tmp, key, meta):
        with open(os.path.join(tmp, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        try:
            os.rename(tmp, self._dir(key))
        except OSError:                      # someone else committed the same key first
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- housekeeping
    def evict(self, protect=()):
        entries, total, now = [], 0, time.time()
        for name in os.listdir(self.root):
            d = os.path.join(self.root, name)
            if not os.path.isdir(d):
                continue
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(d) for f in fs)
            if name.startswith(".tmp-"):
                if now - os.path.getmtime(d) > 3600:
                    shutil.rmtree(d, ignore_errors=True)
                continue
            try:
                mtime = os.path.getmtime(os.path.join(d, "meta.pkl"))
            except OSError:
                shutil.rmtree(d, ignore_errors=True)   # half-written / corrupt entry
                continue
            entries.append((mtime, name, size))
            total += size
        removed = 0
        for mtime, name, size in sorted(entries):
            if total <= self.max_bytes:
                break
            if name in protect:
                continue
            shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
            total -= size
            removed += 1
        return removed

    def clear(self):
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)
