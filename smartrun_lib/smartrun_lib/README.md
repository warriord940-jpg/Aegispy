# smartrun

Run ordinary Python code as a cached, parallel, incremental dependency graph.

```python
from smartrun import Session

with Session(".smartcache") as s:
    ns = s.run(open("script.py").read())                 # full namespace back
    out = s.run(code, outputs=["result"])                # lowest memory/time
    print(s.report())                                    # per-block status / lane / time
```
CLI: `python -m smartrun script.py [-v] [--explain] [--outputs a,b] [--no-processes] [--strict] [--clear]`

> Scripts that use process workers (default, `spawn`) need the usual
> `if __name__ == "__main__":` guard. Use `Session(use_processes=False)` to avoid it.

## What it does
1. Splits code into top-level statements (blocks) with `ast`.
2. Finds reads / writes / **mutations** (`x.append`, `x[i]=`, `x.a=`, `del`, user functions
   that mutate their parameters, globals mutated inside called functions).
3. Builds a DAG: read-after-write, write-after-read, write-after-write; function bodies'
   hidden global reads are attributed to every caller (transitively).
4. Cache key = hash(block AST, content-hashes of every input, files read, library versions).
   Cache is one entry per block, one file per variable, atomic writes, LRU eviction.
5. **Early cutoff**: a re-run block whose output is byte-identical does not invalidate dependents.
6. Runs ready blocks in parallel, longest critical path first (timings are profiled and stored).
7. Lanes: `thread` (NumPy/pandas/IO, GIL-releasing), `process` (pure-Python CPU loops),
   `gpu` (blocks using torch/cupy/jax/cudf/tensorflow; one task per device).
8. Memory: values are freed after their last consumer; cached values load lazily and only
   the variables actually needed; large arrays are memory-mapped copy-on-write; with
   `outputs=[...]` unneeded imports/defs/blocks never run at all.

## Directives (comment on the line above or on the block)
`# smartrun: pure` | `impure` | `nocache` | `thread` | `process` | `gpu` | `cpu`

## Purity rules
Never cached: print/input/sleep/time/random/os/subprocess/network, file writes, `global`,
`exec/eval`, handle reads. Seeded generators (`np.random.default_rng(42)`) are pure.
Files read through `open("x")`, `np.load`, `pd.read_*` etc. are tracked by path/mtime/size
(literal path or simple variable expression).

## Known limits
- Only top-level statements are blocks (a big `for` loop is one block).
- Mutation through aliases (`b = a; b.append(1)`) or unknown library calls is not seen;
  use `strict_calls=True` (conservative) or `# smartrun: impure`.
- GPU lane routes by library usage and serializes per device; it does not translate NumPy to CuPy.
  Not tested on real GPU hardware in development.
- Values pickled by value from the executed code (instances of classes defined in the script,
  lambdas) are re-computed each run instead of cached.
