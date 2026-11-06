"""python -m smartrun script.py [--explain] [--cache-dir D] [--workers N] ..."""
import argparse
import sys
import time

from .session import Session


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smartrun", description=__doc__)
    ap.add_argument("script", nargs="?")
    ap.add_argument("--cache-dir", default=".smartcache")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--no-processes", action="store_true", help="threads only (no multiprocessing)")
    ap.add_argument("--strict", action="store_true", help="assume unknown calls may mutate their arguments")
    ap.add_argument("--outputs", help="comma-separated names to keep; frees everything else")
    ap.add_argument("--explain", action="store_true", help="show the block plan and exit")
    ap.add_argument("--clear", action="store_true", help="clear the cache and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-block report")
    a = ap.parse_args(argv)
    with Session(a.cache_dir, workers=a.workers, use_processes=not a.no_processes,
                 strict_calls=a.strict, verbose=a.verbose) as s:
        if a.clear:
            s.clear_cache()
            print("cache cleared")
            return 0
        if not a.script:
            ap.error("script required")
        code = open(a.script).read()
        if a.explain:
            print(s.plan(code))
            return 0
        t = time.perf_counter()
        s.run(code, outputs=a.outputs.split(",") if a.outputs else None)
        print(f"\n[smartrun] {time.perf_counter() - t:.2f}s  executed={s.stats['executed']} "
              f"cached={s.stats['cached']} skipped={s.stats['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
