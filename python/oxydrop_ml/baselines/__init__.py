"""Honest baselines for the pre-registered tasks P1 (Hoffman low-oxygen), P2 (heart rate on BUT + MTHS) and P3 (MTHS SpO2), evaluated through the
same harness as the CNN: `python -m oxydrop_ml.baselines.run_baselines --all` (run from python/ with the ML venv)."""
import os
import sys

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # python/ (for `import oxydrop`)
if _PY not in sys.path:
    sys.path.insert(0, _PY)
