"""Code that runs inside worker processes (kept import-light for fast spawn)."""
import pickle
import traceback

try:
    import cloudpickle as cp
except ImportError:  # pragma: no cover
    cp = pickle


def run_block(payload: bytes):
    src, writes, inputs = cp.loads(payload)
    g = {"__name__": "__smartrun__"}
    g.update(inputs)
    try:
        exec(compile(src, "<smartrun block>", "exec"), g)
    except BaseException:
        return ("error", traceback.format_exc())
    outs = {n: g[n] for n in writes if n in g}
    try:
        return ("ok", cp.dumps(outs))
    except Exception:
        return ("unpicklable", None)


def ping():
    return True
