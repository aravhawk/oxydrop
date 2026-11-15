#!/usr/bin/env bash
# Source this file (do not execute) to set up the OxyDrop ML environment:
#   source scripts/ml_env.sh
# It never prints a credential. ~/.bashrc returns early in non-interactive shells,
# so the credential exports are read from it directly (only the named variables).
VERITAS=${VERITAS:-/home/aravhawk/.t3/scratch/veritas}
export TMPDIR="$VERITAS/ml_cache/tmp"          # /tmp is a 3.4 GB RAM-backed tmpfs
export UV_CACHE_DIR="$VERITAS/ml_cache/uv"
export ML_PY="$VERITAS/ml_venv/bin/python"
export PATH="$VERITAS/ml_venv/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export CHROME="${CHROME:-$(ls -d "$HOME"/.cache/ms-playwright/chromium-1254/chrome-linux*/chrome 2>/dev/null | head -1)}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
if [ -r "$HOME/.bashrc" ]; then
  eval "$(grep -E '^export (DAYTONA_API_KEY|KAGGLE_USERNAME|KAGGLE_KEY|PHYSIONET_USER|PHYSIONET_PASS)=' "$HOME/.bashrc")"
fi
# Kaggle CLI 2.x needs the new-style token in KAGGLE_API_TOKEN (KAGGLE_KEY holds a KGAT_ token).
[ -n "${KAGGLE_KEY:-}" ] && export KAGGLE_API_TOKEN="$KAGGLE_KEY"
