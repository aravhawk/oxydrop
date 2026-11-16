# R3-EXPORT check (ONNX export, browser e2e, engine swap)

Repo HEAD 811f922 plus uncommitted working tree. Nothing committed, pushed or deployed.

## 1. ONNX export (alternate artefact; the app still runs nn_runtime.js)

`python/oxydrop_ml/photon/distill_onnx.py` (`cd python && python -m oxydrop_ml.photon.distill_onnx [--src W.json] [--out DIR]`). Opset 17, float32, one file per student.
The graph takes the raw 39 features (NaN/inf allowed) and contains impute, clip, standardise and the MLP; output `o` = (hyp logit, SpO2 head, abstain logit). The ensemble is outside the graph, as in the app.

| set | folder | windows | max abs p, ORT vs torch float64 | max abs logit | abstain-flag mismatches | non-finite test (529 entries) |
|---|---|---|---|---|---|---|
| distill (round 2, 3 x 32) | `results/photon/onnx/` | 1,280 | 1.5e-7 | 6.5e-6 | 0 | 4.1e-6 |
| distill2 (shipped, 32/32/64) | `results/photon/onnx_distill2/` | 1,280 | 1.3e-7 | 6.4e-6 | 0 | 5.3e-6 |

Windows are the 3 fixture recordings of `tests/fixtures/ref_distill.json`, calibration windows included. ORT also matches the app's python reference to the same 1.3e-7. Each folder holds `parity.json`, `distill_manifest.json` (tau, columns, ensemble rule) and the .onnx files. Needs `onnx`, `onnxruntime` (both already in the venv). `onnxscript` is not installed, so the legacy TorchScript exporter is used (`dynamo=False`). Peak RSS about 290 MB. `nn_runtime` was not shown wrong (JS and python agree to 1e-13), so ONNX is not wired into the app.

## 2. Browser end-to-end (Playwright Chromium, fake camera, one process at a time)

Harness fixes in `tests/e2e/fake_camera.mjs` (+ README): `ENGINE=` selects the default engine (unset, no query) or `?engine=logistic|hybrid`; engine-specific assertions (the old ones were hybrid-only and would fail on the distill default); a CSV audit of every decision row; `freeze()` now also pauses the `<video>`. The old freeze hook only blocked `requestVideoFrameCallback`, but `app/js/camera.js` has a rAF fallback loop (frame after 250 ms of native silence), so the stall and calstall cases never stalled anything. That is a harness defect, not an app defect.

Results on the shipped distill2 (default engine, no `?engine=`), last runs after the swap:

| case | default engine (distill2) | `?engine=logistic` |
|---|---|---|
| good (72 bpm, calibration, HR 72, debug view, CSV, copy, no console errors) | PASS | PASS |
| bad (grey frame: placement refused, Unsure, HR en dash) | PASS | PASS |
| replay 100001L and 100004R | PASS | PASS |
| stall (flips to Unsure within 2.6 s, recovers) | PASS | PASS |
| calstall (stall in calibration rejects it) | PASS | PASS |

Not rerun after the final swap for `?engine=hybrid` (not requested). One earlier `good` run on distill2 showed "Unsure, re-measure / The camera dropped frames" for 14 of 16 windows while the box was loaded; the harness now accepts a validity reason there, and the rerun on a quiet box passed. Under high load (load average above 10) the real-time cases flake (frame gaps, 20 s selector timeouts): run them on a quiet box.

What the UI shows (distill2, default engine):
* Three states exactly from the model: abstain -> "Unsure, re-measure", P > 0.5 -> "Possible low oxygen: re-check with a pulse oximeter", else "Normal vs your baseline". Replay of 100001L showed 210 normal, 222 low, 79 unsure; 100004R 154 / 121 / 198 (these people are in the final refit, so it is a demo, never an accuracy).
* Debug view: `engine oxydrop-photon-distill2-v1 (...)`, 2,435 / 2,435 / 6,915 parameters per seed, per-window latency (median about 15 ms in Chromium), and the line `model oxydrop-photon-distill2-v1: raw P(SpO2<90) ..  abstain true|false (abstain head ..) ..  EXPERIMENTAL SpO2 estimate (not validated)`.
* CSV audit (last 450 decisions of each replay): every row is the model's answer or `model_unsure`; 0 rows with a pulse, saturation or SNR reason, 0 label/state mismatches.

No app-side rule overrides the model (read `app/js/decision.js`, `app/js/app.js`, `app/js/photon_engine.js` from FEAT-006): for windows with `ownQuality` decide() only applies camera stalled, calibration failed, no finger, movement (accelerometer), fps < 20 and frame gaps before the model label; saturation, invalid, nolock and low-AC reasons are skipped; `watchWeak` and the live weak-signal override are skipped; checkCalibration keeps motion, finger coverage, gaps, stalls and dirty-placement checks and drops only the pulse checks. Observations for the app owner (no redesign done):
1. Setup still gates the Start button on the v2 pulse lock (`lockRun`/`weakNow`, 3 locked windows) and the "weak signal" placement prompt. That is before the measurement and does not change any model answer, but it means a person whose pulse the v2 chain cannot lock (participant 4, 43 bpm) cannot start without "Start anyway".
2. The HR tile and the "signal" tile keep the v2 pulse lock, so the model can answer while HR shows an en dash (100004R replay: answers shown, `hr: -`).
3. After a camera stall ends, the card shows the stalled Unsure until the next window; this passes e2e.

## 3. Swap script

`scripts/swap_engine_model.sh results/photon/<model>_weights.json [--engine-id ID] [--onnx] [--bump-sw NAME] [--dry-run]`. Checks the file (format, 39 columns equal to `distill_features.js` COLS, 3-head MLP shapes, tau, reference) and `free -m` >= 1.5 GB; backs up `app/data/distill.json`, `tests/fixtures/ref_distill.json` (and `app/sw.js`, `app/js/app.js` with `--bump-sw`) to `results/photon/swap_backup/<ts>/` with RESTORE.txt; runs `distill_app_export --src` (the exporter now takes `--src` and `--engine-id`; defaults reproduce distill: `app/data/distill.json` regenerates byte for byte), `tests/distill_parity.mjs`, `tests/nn_runtime.mjs`, optional `tests/sw_assets.mjs`, optional ONNX. Restores everything on failure (verified once: a stale parity test made the first swap roll back cleanly). Used for the distill2 swap below. A different network width is handled (distill2 mixes 32 and 64). `tests/distill_parity.mjs`, `tests/flow.mjs`, `tests/sw_assets.mjs` and the e2e regexes name the engine id and cache by hand and must be edited with `--engine-id` / `--bump-sw`. The model-card and README numbers are not touched by the script.

## 4. distill2 swap (planner steering 20:23 CT)

Weights `results/photon/distill2_weights.json` (hidden 32/32/64, tau 0.9161). Parity (1,190 windows: 100001L 511, 100004R 473, 100001L stress L4 206): max feature error 1.2e-12 (tol 1e-6), max |p_js - p_py| 9.0e-14 (tol 1e-5), SpO2 1.3e-12, abstain mismatches 0; weights-file reference vs app file 2.4e-9. Answered windows 432 / 275 / 87. Latency per 2 s window in Node: core median 4.9 ms, p95 5.8 ms. Cache `oxydrop-v3.6-photon-distill2`; engine id `oxydrop-photon-distill2-v1`; app file kept as `app/data/distill.json` (152 kB).

Tests at the end: `tests/distill_parity.mjs` OK, `tests/sw_assets.mjs` PASS (33 files), `tests/flow.mjs` PASS (all 7 runs inside `npm test`), `npm run test:docs` CHECK CLAIMS PASS, `npm test` exit 0, `DRY_RUN=1 bash scripts/deploy_pages.sh` ends with DRY RUN OK (cache oxydrop-v3.6-photon-distill2). `python/oxydrop_ml/tests/test_photon_distill_app.py` 2 passed. `tests/fixtures/ref_distill.json` regenerated from distill2 (a regeneration with the distill defaults differs from HEAD's fixture in bytes, while the app file is identical; not investigated).

Files changed or added: `app/data/distill.json`, `app/index.html` (card numbers: L0 0.784 [0.596, 0.834], paired -0.012 [-0.091, 0.012] tie, 69.0% answered at 93.3%, X1 0.682 vs 0.508), `app/js/app.js` (build string, per-seed parameter text), `app/js/distill_engine.js` (info text for mixed widths), `app/sw.js`, `tests/fixtures/ref_distill.json`, `tests/distill_parity.mjs`, `tests/flow.mjs`, `tests/sw_assets.mjs`, `tests/e2e/fake_camera.mjs`, `tests/e2e/README.md`, `python/oxydrop_ml/photon/distill_app_export.py`, `python/oxydrop_ml/photon/distill_onnx.py` (new), `python/oxydrop_ml/tests/test_photon_distill_app.py`, `scripts/swap_engine_model.sh` (new), `scripts/check_claims.py` (check_app_text now checks the distill2 card numbers against results/photon/TABLE.csv), `results/photon/onnx/`, `results/photon/onnx_distill2/`, `results/photon/swap_backup/` (backups), this file.

## Not done
`?engine=hybrid` e2e after the swap; docs that still name the hybrid or the older PhotonDistill numbers (`docs/APP_INTEGRATION.md`, `DEMO_SCRIPT.md`, `docs/photon_wiring_notes.md` section 10, whitepaper text outside the card) were not edited by this task beyond what `test:docs` checks.
