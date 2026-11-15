# App integration: the hybrid model as the on-device main path

The app (`app/`) runs the final hybrid model in the browser with dependency-free JavaScript. The v1 logistic chain and the textbook ratio-of-ratios are baselines in the results tables, and the chain logit and the baseline-relative ratio channels are inputs of the model (a physics-informed hybrid). Research prototype, not a medical device.

## Data flow

```
camera frames -> Resampler (30 Hz R,G,B) -> v2 quality chain (engine.js, main thread)  -> valid / SQI / saturation / HR (spectral) / calibration summary
                                          \-> HybridCore (hybrid_engine.js; Web Worker in the live app)
                                                v1 physics chain (engine_v1.js, weights embedded in hybrid_weights.json)  -> c = logit(p_chain)
                                                CnnInput (cnn_features.js): 10 s x 10-channel tensor + 9 side features, first-minute baseline
                                                3 x OxyNet (cnn.js, hybrid_R_s{0,1,2}.json, 167,588 parameters each)       -> corr (mean lowox_logit), CNN HR (median of 3)
                                                r = c + corr -> meta-learner (hybrid_weights.json pipeline.meta)            -> z (z > 0: possible low oxygen)
                                                conformal rule (conformal.json, default alpha 0.10)                         -> NORMAL if z < -0.965, LOW if z > 0.389, else UNSURE
finishWindow(): window object with the fields decision.js decide() expects ({valid, sat, label, hr, ...}); the quality gates of decision.js stay in front.
```

* **Label.** `normal`, `HYPOXIC`, `unsure` from the conformal rule; `nolock` when the v2 pulse-lock gate fails. `decide()` and the three UI texts are unchanged.
* **Baseline.** The first 60 s are label-free calibration (same checks as before: finger coverage, frame rate, stalls, resting HR range). The baseline embedding of the low-oxygen head is the mean embedding of up to 8 evenly spaced calibration windows rebuilt with the finished baseline (`CnnInput.baselineWindows`, the training convention of `train.group_emb`).
* **Heart rate.** Rule `agree` (declared in Amendment 3): the mean of the CNN estimate and the v2 spectral estimate when they are within 5 bpm, otherwise the CNN estimate. It is shown only when the v2 gate says the window has a pulse lock. On Hoffman alone the spectral estimator is more accurate than this rule (0.883 against 0.826 within 5 bpm); the rule wins on the 100 BUT PPG and MTHS people (0.480 against 0.419). On the synthetic 72 bpm test video the CNN reads 4 to 5 bpm high, so the e2e tolerance is +-6 bpm.
* **SpO₂.** No SpO₂ number in the main UI. The CNN SpO₂ head only appears in the debug view on a line labelled `EXPERIMENTAL ... (not validated, do not use)`. Reason: on the OpenOximetry contact sensor the CNN ties the textbook ratio-of-ratios (SaO₂ MAE 2.94 against 3.07, `results/ml/OPENOX_SPO2.md`), and on MTHS (normoxic) it shows no skill over the training mean.
* **Low-amplitude guard (generic).** lac is the log of the standard deviation of the band-passed log-intensity of the 10 s window (about the AC/DC ratio of the pulse); the guard takes the largest of the three channels. If it is below -6.02 (an AC/DC ratio of 0.24 %), the 0.1th percentile over all windows of the six Hoffman people (`scripts/ood_range.py`, `results/ml/ood_ac_range.csv`), the window gets the label `low_ac`: the answer is *Unsure* with a reason text about a pulse weaker than the model was trained on, and the heart rate shown is the spectral estimate only. This is a range check on the training data, not a phone setting. It exists because the pulse-lock gate is scale free: a clean but very weak pulse (a flash far from the lens) would otherwise reach the network, which has never seen such a signal. On the fixture recordings it refuses no window of person 1 (his lowest window is -5.95), and a person 1 pulse attenuated to 3 % is refused in all 39 windows that still lock. Cost: a phone whose pulse is always weaker than 0.24 % AC/DC will always answer *Unsure*; that is a limitation, not a tuning target.
* **Slow pulse (Hoffman person 4, 43 bpm).** The v2 gate cannot lock below about 51 bpm (spectral floor), so every window is `nolock` and the answer is *Unsure*. That is the existing gate, not the CNN: the CNN does not track this person either (MAE 21 bpm, `results/ml/HR_VOTE.md`).

## Files

| file | role |
|---|---|
| `app/js/hybrid_engine.js` | `HybridCore` (v1 chain + CNN + combiner + conformal), `HybridEngine` (v2 quality chain + core, one thread), `finishWindow`, `conformalState`, `hrAgree`, `loadHybridAssets` |
| `app/js/hybrid_async.js`, `app/js/hybrid_worker.js` | live path: the quality chain stays on the main thread, `HybridCore` runs in a module Web Worker so a CNN pass never delays camera frames; results arrive 50 to 300 ms after a window ends |
| `app/js/cnn.js`, `app/js/cnn_features.js` | forward pass and causal input tensor (parity-tested against PyTorch and numpy) |
| `app/data/hybrid_weights.json` | pipeline definition: embedded v1 chain weights, meta-learner, members, checkpoint hashes |
| `app/data/hybrid_R_s{0,1,2}.json` | the three CNN seeds (0.9 MB each, BatchNorm folded, phone input scale folded in) |
| `app/data/conformal.json` | Mondrian conformal thresholds per alpha (`oxydrop-conformal-v1`) |
| `app/data/oxynet.onnx` | ONNX export of seed 0 (interoperability only; the app does not load it) |
| `app/data/weights.json` | v2 logistic chain: quality gate and the `?engine=logistic` baseline |
| `results/ml/FINAL_MODEL.json` | weights hashes, parameters, seeds, training subjects (`scripts/final_model_json.py`) |
| `results/ml/DEPLOY.md` | latency per window and asset sizes |
| `results/ml/ood_ac_range.csv` | training range of the pulse amplitude behind the low-amplitude guard (`scripts/ood_range.py`) |

The embedded v1 chain differs slightly from `weights.json` (largest coefficient difference 0.58) because the meta-learner was fitted on that refit chain; the app uses the embedded weights for `c`.

## URL switches
* default: hybrid. `?engine=logistic`: the v2 logistic chain alone (the baseline of the comparison, also the fallback if the hybrid files cannot be loaded; the debug `engine` line says which one runs and why).
* `?alpha=0.05` sets another conformal level from `conformal.json` (debug only; `0.20` is past the crossing point and refused). The declared default is 0.10.
* `?rec=100001L|100004R` picks the replay recording.

## Replay demo: in-sample, demo only
The three CNN seeds, the embedded chain, the meta-learner and the conformal thresholds were all fitted with the six Hoffman people, including the two replay recordings (100001L and 100004R). Per-fold held-out weights exist only as prediction tables (`results/ml/scores/`), not as checkpoints, so the replay screen says **"Model trained on this person: demo only, not a test"** and so does the tally line. The honest held-out numbers are in `results/ml/HYBRID.md` and `CONFORMAL.md`. `?engine=logistic` replays with the v2 chain weights fitted without that person, as before.

## Device agnostic by construction
Nothing is fitted to a phone. Per-lens probing, adaptive exposure and the placement, motion, frame-gap, saturation, stall and calibration gates are generic. The three iPhones named in `docs/PHONE_CHECKLIST.md` are a held-out sanity check by the team; untested Androids and other phones are a limitation. Known cost: the model runs on the CPU of the phone. The latency per window is reported in the debug view and in `results/ml/DEPLOY.md` (laptop numbers; phones are slower and unmeasured).

## Tests
`npm test` runs, besides the earlier suites:
* `tests/cnn_parity.mjs`: JS forward pass against PyTorch on the shipped seed 0 (worst 4e-6), streaming features against numpy.
* `tests/hybrid_engine.mjs`: combiner against `tests/fixtures/hybrid_ref.json`, conformal rule against `conformal.json` examples, the streaming engine on the two Hoffman fixture recordings against `python/oxydrop_ml/make_engine_ref.py` (PyTorch checkpoints of the three exported seeds, v1 chain of `python/oxydrop`; chain logit 1e-12, CNN correction 2e-6, CNN HR 1e-5 bpm, z 2e-6), the quality gates in front, the HR rule, and the Web Worker path against the in-thread engine (identical).
* `tests/sw_assets.mjs`: every file the hybrid needs offline is in the service-worker precache and the cache version was bumped.
* `tests/flow.mjs`: the camera-flow regression tests of the review fixes, plus the default engine check.
* Browser e2e (`tests/e2e/fake_camera.mjs`, see `tests/e2e/README.md`): good, bad, replay, replaylog, stall, calstall.

Regenerate the engine reference after any model change: `python python/oxydrop_ml/make_engine_ref.py --ckpt-dir <runs>/hyb_final`, and `python python/oxydrop_ml/export_js.py --ckpt <runs>/hyb_final/R_s0.pt --weights <tmp>.json --onnx app/data/oxynet.onnx --fixture tests/fixtures/cnn_ref.json` for the single-network fixture.
