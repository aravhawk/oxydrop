#!/usr/bin/env bash
# Runs scripts/check_claims.py (stale-render check, independent recomputation of every quoted number, app text numbers) with the ML venv if it exists, else with python3. Used by `npm test` so a stale README / MODEL.md / WHITEPAPER.md or a wrong number in the app text fails the build.
# Set OXY_SKIP_DOCS_CHECK=1 to skip explicitly (for example on a machine with no Python and pandas); the skip is printed, never silent.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${OXY_SKIP_DOCS_CHECK:-0}" = "1" ]; then echo "DOCS CHECK SKIPPED (OXY_SKIP_DOCS_CHECK=1)"; exit 0; fi
VENV_PY="${VERITAS:-/home/aravhawk/.t3/scratch/veritas}/ml_venv/bin/python"     # the project's ML venv when it exists (no credentials are loaded for this check)
if [ -n "${ML_PY:-}" ]; then PY="$ML_PY"; elif [ -x "$VENV_PY" ]; then PY="$VENV_PY"; else PY=python3; fi
export PATH="$PATH:/home/linuxbrew/.linuxbrew/bin"
if ! "$PY" -c "import pandas, numpy" >/dev/null 2>&1; then
  echo "check_claims needs pandas and numpy ($PY has none). Install them, run scripts/ml_env.sh, or set OXY_SKIP_DOCS_CHECK=1 to skip explicitly." >&2
  exit 2
fi
exec "$PY" scripts/check_claims.py
