"""Dataset builders. Each module has `build()` (writes a shard under <CACHE>/data/<ds>) and `check()` (verifies it), and a CLI:

    python -m oxydrop_ml.data.<ds> [--build] [--check]

Run from the repo root with `python/` on PYTHONPATH (the ML venv: /home/aravhawk/.t3/scratch/veritas/ml_venv/bin/python).
Row-level data never goes into git: everything is written to the cache directory (OXY_CACHE, default .../ml_cache)."""
import argparse
import os
import sys

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # python/ (for `import oxydrop`)
if _PY not in sys.path:
    sys.path.insert(0, _PY)

VERITAS = os.environ.get('VERITAS', '/home/aravhawk/.t3/scratch/veritas')
OXY_DATA = os.environ.get('OXY_DATA', f'{VERITAS}/oxy_push/data')


def cli(build, check, desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument('--build', action='store_true', help='(re)build the shard')
    ap.add_argument('--check', action='store_true', help='verify the shard against the documented counts and exit non-zero on a mismatch')
    a = ap.parse_args()
    if a.build or not a.check:
        build()
    if a.check:
        ok = check()
        sys.exit(0 if ok else 1)
