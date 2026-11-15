#!/usr/bin/env bash
# Reproduce the v2 pipeline.
# Usage: [PYTHON=/path/to/python] [OXY_BUT=<BUT PPG 2.0.0 dir>] [OXY_MTHS=<MTHS Data dir>] scripts/reproduce.sh [path-to-clone-of-ubicomplab/oximetry-phone-cam-data]
# Always: figures + tables from saved results (numpy, scipy, pandas, matplotlib; no dataset needed), then the bundled parity, HR parity, decision and capture tests.
# With the Hoffman repo path: Python reference check, refit of the shipped weights, v1 vs v2 evaluation (eval_v2.py), JS-vs-Python parity on all 12 recordings.
# With OXY_BUT: BUT PPG heart-rate validation and subject-level cross-fit gate selection (validate_hr.py but, hr_crossfit.py).
# With OXY_MTHS: MTHS ratio-of-ratios SpO2 experiment (mths_spo2.py, about 1 minute, under 200 MB).
# Run one heavy step at a time on a small machine; nothing here needs more than about 1 GB.
set -euo pipefail
PY=${PYTHON:-python3}
cd "$(dirname "$0")/.."
"$PY" -I scripts/make_eval.py
"$PY" -I scripts/make_v2_figs.py
echo "== figures and tables regenerated"
if [ $# -ge 1 ]; then
  HOF="$1"
  export OXY_HOF="$HOF"
  PYTHONPATH=python "$PY" -s python/verify_against_reference.py "$HOF" | tail -1
  PYTHONPATH=python "$PY" -s python/fit_and_export.py "$HOF" /tmp/oxydrop_refit | head -1
  diff -q /tmp/oxydrop_refit/weights.json app/data/weights.json && echo "== refit weights identical to shipped"
  "$PY" -s scripts/eval_v2.py "$HOF" results | tail -8
  node tests/parity.mjs v2 tests/fixtures/ref_all_12rec_python_v2.json "$HOF/data/ppg-csv" | tail -2
  node tests/parity.mjs v1 tests/fixtures/ref_all_12rec_python.json "$HOF/data/ppg-csv" | tail -2
else
  node tests/parity.mjs v2 tests/fixtures/ref_2rec_v2.json tests/fixtures/ppg-csv | tail -2
  node tests/parity.mjs v1 tests/fixtures/ref_2rec.json tests/fixtures/ppg-csv | tail -2
fi
if [ -n "${OXY_BUT:-}" ]; then
  "$PY" -s scripts/validate_hr.py but
  "$PY" -s scripts/hr_crossfit.py
fi
if [ -n "${OXY_MTHS:-}" ]; then
  "$PY" -s scripts/mths_spo2.py
fi
node tests/hr_parity.mjs | tail -1
node tests/decision.mjs | tail -1
node tests/capture_sim.mjs | tail -1
