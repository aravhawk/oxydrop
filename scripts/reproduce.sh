#!/usr/bin/env bash
# Reproduce everything. Usage: [PYTHON=/path/to/python] scripts/reproduce.sh [path-to-clone-of-ubicomplab/oximetry-phone-cam-data]
# 1) figures + tables from the saved out-of-fold scores (numpy, scipy, pandas, matplotlib; no dataset needed)
# 2) with the Hoffman repo path: Python reference check, refit of the shipped weights, JS-vs-Python parity on all 12 recordings
set -euo pipefail
PY=${PYTHON:-python3}
cd "$(dirname "$0")/.."
"$PY" -I scripts/make_eval.py
echo "== figures and tables regenerated"
if [ $# -ge 1 ]; then
  HOF="$1"
  PYTHONPATH=python "$PY" -s python/verify_against_reference.py "$HOF" | tail -1
  PYTHONPATH=python "$PY" -s python/fit_and_export.py "$HOF" /tmp/oxydrop_refit | head -1
  diff -q /tmp/oxydrop_refit/weights.json app/data/weights.json && echo "== refit weights identical to shipped"
  node tests/parity.mjs v2 tests/fixtures/ref_all_12rec_python_v2.json "$HOF/data/ppg-csv" | tail -2
  node tests/parity.mjs v1 tests/fixtures/ref_all_12rec_python.json "$HOF/data/ppg-csv" | tail -2
else
  node tests/parity.mjs v2 tests/fixtures/ref_2rec_v2.json tests/fixtures/ppg-csv | tail -2
  node tests/parity.mjs v1 tests/fixtures/ref_2rec.json tests/fixtures/ppg-csv | tail -2
fi
node tests/hr_parity.mjs | tail -1
node tests/capture_sim.mjs | tail -1
