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

# ===================================================================================================== FEAT-006: core CNN (SSL, stage B, nested selection, LOSO, ablations)
# Heavy stages run on Daytona (4 vCPU sandbox, python scripts/ml_daytona.py up|bg|tail|pull|down) because OpenOximetry-derived tensors may only go there; the same
# commands run on the host with a smaller --shard share. RUNS=<runs root>, CK=<ckpt dir> (outside git: ml_cache/runs, ml_cache/ckpt).
#   core-test    unit tests (leak guarantees on a fake dataset, augmentation, SSL exclusions, fold norm folding)
#   core-ssl     contrastive SSL (2500 steps, 5 min on 4 vCPU): ssl_full.pt (all corpora) and ssl_noox.pt (no OpenOx), loss curves + results/ml/ssl_manifest.json
#   core-stageb  OpenOx supervised warm-up from SSL (stageB_full.pt) and from scratch (stageB_nossl.pt)
#   core-nested  nested selection of the recipe (36+ inner fits), then `loso_run select` writes the recipe
#   core-loso    the queue of LOSO / grouped runs (scripts/ml_jobs_core.txt), sharded: SHARD=i/n
#   core-report  results/ml/ablation_core.csv, cnn_vs_baselines_paired.csv, CORE_RESULTS.md, scores/cnn_*.csv from RUNS
OXY_CACHE_DIR=${OXY_CACHE:-/home/aravhawk/.t3/scratch/veritas/ml_cache}
RUNS=${RUNS:-$OXY_CACHE_DIR/runs/final}; CK=${CK:-$OXY_CACHE_DIR/ckpt}
if [ "$STAGE" = "core-test" ]; then (cd python && "$PY" -m pytest oxydrop_ml/tests/test_train.py -q); fi
if [ "$STAGE" = "core-ssl" ]; then
  (cd python && "$PY" -m oxydrop_ml.ssl_pretrain --steps 2500 --out "$CK/ssl_full.pt" --curve ../results/ml/ssl_loss_curve_full.csv --manifest ../results/ml/ssl_manifest.json --tag ssl_full
   "$PY" -m oxydrop_ml.ssl_pretrain --steps 2500 --no-openox --out "$CK/ssl_noox.pt" --curve ../results/ml/ssl_loss_curve_noox.csv --manifest ../results/ml/ssl_manifest.json --tag ssl_noox
   "$PY" -m oxydrop_ml.ssl_pretrain --probe "$CK/ssl_full.pt")
fi
if [ "$STAGE" = "core-stageb" ]; then
  (cd python && "$PY" -m oxydrop_ml.train --stageb --steps 2500 --init "$CK/ssl_full.pt" --out "$CK/stageB_full.pt"
   "$PY" -m oxydrop_ml.train --stageb --steps 2500 --out "$CK/stageB_nossl.pt")
fi
if [ "$STAGE" = "core-nested" ]; then
  (cd python && "$PY" -m oxydrop_ml.loso_run nested --out "$RUNS/nested" --init "$CK/stageB_full.pt" --hp '{"steps":400,"lr":0.001}' --shard "${SHARD:-0/1}"
   "$PY" -m oxydrop_ml.loso_run select "$RUNS/nested")
fi
if [ "$STAGE" = "core-loso" ]; then
  (cd python && "$PY" -m oxydrop_ml.loso_run queue --jobs ../scripts/ml_jobs_core.txt --runs "$RUNS" --ckpt-dir "$CK" --recipe "$RUNS/recipe.json" --shard "${SHARD:-0/1}")
fi
if [ "$STAGE" = "core-report" ]; then
  (cd python && "$PY" -m oxydrop_ml.core_report --runs "$RUNS" --out ../results/ml --baselines ../results/ml)
fi
if [ "$STAGE" = "core-hybrid" ]; then      # hybrid chain + CNN and BatchNorm adaptation on P1 (needs python/oxydrop for chain19.npy, see stack_run.py docstring)
  (cd python && "$PY" -m oxydrop_ml.stack_run run --out "$RUNS/stack/s0" --init "$CK/stageB_full.pt" --recipe "$RUNS/recipe.json" --shard "${SHARD:-0/1}"
   "$PY" -m oxydrop_ml.stack_run report --runs "$RUNS/stack/s0" --out ../results/ml)
fi
