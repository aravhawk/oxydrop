#!/usr/bin/env bash
# Swap a new model with the SAME front end (distill's 39 causal features + nn-runtime-v1 MLP students) into the app as the default engine's model file.
#
#   bash scripts/swap_engine_model.sh results/photon/<model>_weights.json [--engine-id ID] [--onnx] [--bump-sw NAME] [--dry-run]
#
# Input: a weights file in the distill_export.py format (format 'oxydrop-photon-distill-v1': cols = the 39 distill_features.js columns, tau, recipe,
# students[{prep:{med,lo,hi,mu,sd}, w1,b1,w2,b2,wo,bo}], reference{X,score,answer}). A model with another front end or another runtime needs its own JS (see
# app/js/photon_engine.js registerRuntime) and cannot use this script.
# Steps (all under this repo, nothing is committed, pushed or deployed):
#   1. checks the file (format, column list = app/js/distill_features.js COLS, 3-head MLP shapes, tau, reference) and `free -m` >= 1.5 GB available (FORCE=1 skips the memory check)
#   2. backs up app/data/distill.json, tests/fixtures/ref_distill.json (and app/sw.js if --bump-sw) to results/photon/swap_backup/<timestamp>/
#   3. python -m oxydrop_ml.photon.distill_app_export --src <file>  -> app/data/distill.json + tests/fixtures/ref_distill.json
#      (checks that the app file reproduces the weights file's own reference, regenerates the L4 stress CSV bit for bit)
#   4. node tests/distill_parity.mjs (JS vs python: features 1e-6, probability 1e-5, SpO2 1e-4, abstain flags exact) and node tests/nn_runtime.mjs
#   5. optional: --onnx re-exports results/photon/onnx_<model>/ (distill_onnx.py --src <file>); --bump-sw NAME sets `const V` in app/sw.js (needed so installed
#      PWAs fetch the new model file) and the hard-coded build name in app/js/app.js, then runs tests/sw_assets.mjs. --engine-id ID sets the engine id (app file + fixture);
#      tests/distill_parity.mjs, tests/flow.mjs and tests/sw_assets.mjs name the shipped id / cache by hand and need the same edit
# On any failure the backed-up files are restored and the script exits 1. --dry-run only runs step 1 (checks the file) and prints the plan; nothing is written.
# Afterwards (by hand): update the model card / docs numbers (they quote the LOSO results of the model, never the refit), run `npm test` and DRY_RUN=1 bash scripts/deploy_pages.sh.
# The restore command if you change your mind later: cp results/photon/swap_backup/<timestamp>/* back to their original paths (listed in RESTORE.txt there).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SRC="" ; ONNX=0 ; BUMP="" ; DRY=0 ; EID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --onnx) ONNX=1 ;;
    --engine-id) EID="${2:?--engine-id needs an id, e.g. oxydrop-photon-distill2-v1}"; shift ;;
    --bump-sw) BUMP="${2:?--bump-sw needs a cache name, e.g. oxydrop-v3.6-photon-<model>}"; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n 2,23p "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) [ -z "$SRC" ] && SRC="$1" || { echo "one weights file only" >&2; exit 2; } ;;
  esac; shift
done
[ -n "$SRC" ] || { echo "usage: bash scripts/swap_engine_model.sh results/photon/<model>_weights.json [--engine-id ID] [--onnx] [--bump-sw NAME] [--dry-run]" >&2; exit 2; }
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 2; }
SRC=$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")

# shellcheck disable=SC1091
source scripts/ml_env.sh
PY=${ML_PY:-python3}
if [ "${FORCE:-0}" != 1 ]; then
  AVAIL=$(free -m | awk '/^Mem:/ {print $7}')
  [ "${AVAIL:-0}" -ge 1500 ] || { echo "only ${AVAIL} MB available (< 1500): stop other jobs or set FORCE=1" >&2; exit 3; }
fi

echo "== 1. checking $SRC"
COLS_JS=$(node --input-type=module -e "import('./app/js/distill_features.js').then((m) => console.log(JSON.stringify(m.COLS)))")
SRC="$SRC" COLS_JS="$COLS_JS" "$PY" - <<'PY'
import json, os, re, sys
d = json.load(open(os.environ['SRC']))
cols = json.loads(os.environ['COLS_JS'])
err = []
if d.get('format') != 'oxydrop-photon-distill-v1': err.append(f"format {d.get('format')!r}, expected 'oxydrop-photon-distill-v1'")
if d.get('cols') != cols: err.append(f'cols differ from distill_features.js COLS ({len(d.get("cols") or [])} vs {len(cols)}): this is a different front end')
if not isinstance(d.get('tau'), (int, float)): err.append('tau missing')
if not d.get('students'): err.append('students missing')
for i, s in enumerate(d.get('students') or []):
    n = len(d.get('cols') or [])
    if any(len(s.get('prep', {}).get(k, [])) != n for k in ('med', 'lo', 'hi', 'mu', 'sd')): err.append(f'student {i}: prep length != {n}')
    try:
        ok = len(s['w1'][0]) == n and len(s['w2'][0]) == len(s['w1']) and len(s['wo'][0]) == len(s['w2']) and len(s['wo']) == 3
        if not ok: err.append(f'student {i}: not a Linear-ReLU-Linear-ReLU-Linear net with 3 heads (hyp, spo2, abstain)')
    except Exception as e: err.append(f'student {i}: {e}')
if not all(k in d.get('reference', {}) for k in ('X', 'score', 'answer')): err.append('reference {X, score, answer} missing')
if err:
    print('REJECTED:', *err, sep='\n  '); sys.exit(1)
print(f"ok: {len(d['students'])} students, hidden {d.get('hidden')}, tau {d['tau']:.4f}, {len(d.get('recipe') or [])} recipe entries")
PY
if [ "$DRY" = 1 ]; then echo "dry run: the file is acceptable; would back up, export and run parity$([ "$ONNX" = 1 ] && echo ', then the ONNX export')$([ -n "$BUMP" ] && echo ", then set the sw cache name to $BUMP")"; exit 0; fi

TS=$(date +%Y%m%d-%H%M%S); BK=results/photon/swap_backup/$TS; mkdir -p "$BK"
FILES=(app/data/distill.json tests/fixtures/ref_distill.json); [ -n "$BUMP" ] && FILES+=(app/sw.js app/js/app.js)
for f in "${FILES[@]}"; do cp -p "$f" "$BK/$(echo "$f" | tr / __)"; done
{ echo "swap of $SRC at $TS; restore with:"; for f in "${FILES[@]}"; do echo "cp -p $BK/$(echo "$f" | tr / __) $f"; done; } > "$BK/RESTORE.txt"
restore() { echo "FAILED: restoring $BK" >&2; for f in "${FILES[@]}"; do cp -p "$BK/$(echo "$f" | tr / __)" "$f"; done; exit 1; }
echo "== 2. backup in $BK"

echo "== 3. export app file + parity fixture"
( cd python && "$PY" -W ignore -m oxydrop_ml.photon.distill_app_export --src "$SRC" ${EID:+--engine-id "$EID"} ) || restore

echo "== 4. JS parity"
node tests/distill_parity.mjs || restore
node tests/nn_runtime.mjs > /dev/null || restore

if [ -n "$BUMP" ]; then
  echo "== 5a. service worker cache name -> $BUMP"
  sed -i -E "s/(capture build )oxydrop-[A-Za-z0-9._-]+'/\1$BUMP'/" app/js/app.js || restore   # hard-coded copy in the debug view
  grep -q "capture build $BUMP'" app/js/app.js || restore
  sed -i -E "1,5s/^const V = '[^']*';/const V = '$BUMP';/" app/sw.js || restore
  grep -q "^const V = '$BUMP';" app/sw.js || restore
  node tests/sw_assets.mjs || restore
fi
if [ "$ONNX" = 1 ]; then
  echo "== 5b. ONNX export + onnxruntime parity"
  ( cd python && "$PY" -W ignore -m oxydrop_ml.photon.distill_onnx --src "$SRC" --out "$ROOT/results/photon/onnx_$(basename "$SRC" _weights.json)" > /dev/null ) || { echo "onnx export failed (app swap itself is fine)" >&2; }
fi
echo "SWAP OK: $(basename "$SRC") is now app/data/distill.json (backup $BK)."
echo "Next, by hand: model-card / README numbers (LOSO, not the refit), npm test, DRY_RUN=1 bash scripts/deploy_pages.sh."
[ -n "$BUMP" ] || echo "Reminder: bump 'const V' in app/sw.js (or rerun with --bump-sw NAME) so installed apps fetch the new model file."
