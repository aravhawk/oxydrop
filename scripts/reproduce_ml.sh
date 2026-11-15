#!/usr/bin/env bash
# Reproduce the ML results of the OxyDrop CNN work, one stage at a time (each stage is a section below, owned by one feature; keep the sections separate).
#   bash scripts/reproduce_ml.sh data|baselines|...        run one stage
# Needs the ML venv (scripts/ml_env.sh sets ML_PY, TMPDIR on disk) and the shards built by the data stage in the cache (OXY_CACHE).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ml_env.sh
PY=${ML_PY:-python3}
STAGE=${1:-}

# ===================================================================================================== FEAT-003: data harness
if [ "$STAGE" = "data" ]; then
  for ds in hoffman mths butppg; do (cd python && "$PY" -m oxydrop_ml.data.$ds --build --check); done
  # (cd python && "$PY" -m oxydrop_ml.data.openox --build --check)   # credentialed OpenOximetry, host only
  (cd python && "$PY" -m pytest oxydrop_ml/tests -q)
fi

# ===================================================================================================== FEAT-005: honest baselines (P1, P2, P3)
# Writes results/ml/baselines_{P1,P2,P3}.csv, *_perperson.csv, baselines_P1_skin.csv, *_info.json, scores/baseline_P1_hoffman.csv, scores/baseline_P2_butppg.csv
# and results/ml/BASELINES.md. About 10 minutes on the Surface host (n_jobs 4, under 1.5 GB); no Daytona needed. MTHS per-window scores go to the cache only (licence).
if [ "$STAGE" = "baselines" ]; then
  (cd python && "$PY" -m pytest oxydrop_ml/tests/test_baselines.py -q)
  (cd python && "$PY" -m oxydrop_ml.baselines.run_baselines --all)
fi
