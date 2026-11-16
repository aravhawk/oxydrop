# PhotonNet (round-2 candidate R2A) log

Code: `python/oxydrop_ml/photon/photonnet.py` (model, nested LOSO, nulls, ablation), `python/oxydrop_ml/photon/photonnet_export.py` (plain-ops JSON + reference),
`python/oxydrop_ml/tests/test_photon_photonnet.py`. Designed 18:26-18:31 CT on Sun Nov 15 2026, after the round-1 outer results were seen (Amendment 7
disclosure). The configuration was fixed before any PhotonNet outer score was computed and was not changed after (one full run, no reruns on outer results).

## Model as run

One jointly trained network per fold and seed (4,937 trainable parameters including the PhotonTwin scalars, tissue-light fractions and noise model):
- Input per 10 s window: the 39 FlashGap window features (`flashgap.COLS`: 19 chain v1 deltas relative to the recording's own label-free first minute,
  15 absolute signal-quality features, 5 baseline-relative respiratory features), all computed causally from the per-frame ROI-mean RGB.
- Trunk 39 -> 48 -> 48 (tanh). Heads: 4 physical latents z = (d_sat, d_ln_fb, d_ln_alpha, d_ln_rho) bounded to 2 prior sd; an 8-dim learned physiological
  embedding feeding a residual in SpO2 points; a sigma head (softplus + 0.5 points).
- Physics layer: the PhotonTwin renderer (`optics.py`, Farrell sum with the sign fix, band-integrated phone channels, 13-point quadrature to 720 nm,
  learned spectral scalars and per-channel tissue-light fractions, via `twin_inverse.Decoder`) renders the six optical changes (d ln DC_R,G,B and
  d ln|AC|_R,G,B) from z; reconstruction loss = heteroscedastic Gaussian NLL vs the observed changes (variance grows as the window pulse SNR falls).
- SpO2 = 100 (sat_b + d_sat) + residual; loss also has the SpO2 Gaussian NLL, class-balanced BCE of score = Phi((90 - SpO2) / sigma), and small
  L2 priors on z and the residual. Weights W_REC = W_BCE = 1, W_PRIOR 0.01, W_R 0.001, Adam lr 4e-3, batch 512 (192 rows through the renderer), 600 steps.
- Abstention is the model's own predictive uncertainty: conf = 1 - min(1, sigma / |SpO2 - 90|), answer = conf >= c*, c* nested.
- No gate, no chain_v1 at inference, no stitched rule; `pred` = score > 0.5 at full coverage.

Deviation from the FEAT-008 plan (time, 75 min for the whole feature): the encoder reads causal per-window features (allowed by ROUND2 rule 3) rather than
a 1D conv over per-frame log RGB, and there is no separate HR soft-bin head or quality head (the sigma head plays the abstain role). The model is
per-window (not temporal), so no blank-observation control was required.

## Augmentation prior (declared)

FlashGap's distant-flash prior (`flashgap.AUG_PRIOR`: d_rho U(0, 12) mm, rho0 U(2.5, 4) mm, path scale U(0.1, 0.3), finger optics, leak U(0, 0.4) with
tint, read and shot noise, AE gain cap and error, quantisation step, G/B channel kill p = 0.15), own RNG namespace (AUG_TAG 'FGAP', never a stress
(level, seed) pair, never X1; `flashgap.assert_disjoint` runs on every load). Model seed s trains on clean training people plus their 4 augmented copies of
augmentation seed s (weight 0.25 each). The augmented features are read from FlashGap's feature cache (same feature code as the test features).
Disclosure: that cache directory is tagged with the crc of the earlier stress.py (`ba26b85a`); the only change since (git diff 3448e0b b48e14b) is a
docstring edit, so the cached degraded test features equal what the current `stress.degrade(rgb, level, 0)` produces.

## Protocol

LOSO by person (both hands). Per outer fold and seed: 5 inner person-grouped fits (each training person held out once, sat_b of the inner validation
person = mean of the inner training recordings), training snapshot chosen from (150, 300, 600) steps by the pooled inner OOF TM bal. acc. on the clean
inner validation windows, c* by `twin_inverse.choose_conf` (answered acc and bal. acc >= 0.90 else best answered bal. acc at coverage >= 0.20) on the pooled
inner OOF windows of the clean and augmented copies of the inner validation people. Then one outer fit on the 5 training people, scored on the held-out
person at L0..L5 and X1 (`stress.degrade(., level, 0)`, `stress.degrade_x(., 'X1', 0)`, same weights at every level). `fit` asserts the held-out
person is not among the training people. sat_b of the held-out person = mean first-minute reference SpO2 of the training recordings (no test labels).
Ensemble: mean SpO2, mixture sigma, mean of the seed scores, c* = mean of the 3 seeds' c*.

### Nested choices per outer fold (results/photon/photonnet_choices.csv)

| held out | seed | steps | c* | inner TM |
|---|---|---|---|---|
| 100001 | 0 / 1 / 2 | 150 / 150 / 300 | 0.150 / 0.125 / 0.425 | 0.651 / 0.655 / 0.629 |
| 100002 | 0 / 1 / 2 | 600 / 150 / 300 | 0.575 / 0.400 / 0.350 | 0.662 / 0.665 / 0.657 |
| 100003 | 0 / 1 / 2 | 150 / 150 / 150 | 0.100 / 0.400 / 0.275 | 0.715 / 0.730 / 0.704 |
| 100004 | 0 / 1 / 2 | 150 / 150 / 150 | 0.125 / 0.250 / 0.375 | 0.711 / 0.724 / 0.721 |
| 100005 | 0 / 1 / 2 | 150 / 150 / 150 | 0.050 / 0.100 / 0.275 | 0.634 / 0.673 / 0.618 |
| 100006 | 0 / 1 / 2 | 150 / 150 / 150 | 0.025 / 0.125 / 0.075 | 0.631 / 0.627 / 0.610 |

The inner TM (0.61-0.73) already predicted a clean-light result well below chain_v1.

## Results (evalkit, results/photon/TABLE.csv; 3-seed ensemble `photonnet`; paired person bootstrap vs chain_v1 at the same level)

| level | TM bal. acc. [95% CI] | diff vs chain_v1 [95% CI] | verdict | people improved | coverage | answered acc |
|---|---|---|---|---|---|---|
| L0 | 0.668 [0.508, 0.785] | -0.128 [-0.208, -0.047] | loss | 2/6 | 0.35 | 0.924 |
| L1 | 0.678 [0.518, 0.794] | -0.126 [-0.232, -0.024] | loss | 2/6 | 0.37 | 0.918 |
| L2 | 0.677 [0.534, 0.783] | -0.066 [-0.176, 0.097] | tie | 3/6 | 0.37 | 0.878 |
| L3 | 0.689 [0.526, 0.812] | +0.125 [-0.019, 0.267] | tie | 4/6 | 0.33 | 0.948 |
| L4 | 0.703 [0.516, 0.829] | +0.105 [-0.025, 0.245] | tie | 5/6 | 0.32 | 0.922 |
| L5 | 0.720 [0.564, 0.818] | +0.185 [0.023, 0.290] | win | 5/6 | 0.16 | 0.933 |
| X1 | 0.672 [0.537, 0.775] | +0.165 [0.020, 0.316] | win | 6/6 | 0.34 | 0.865 |

Single seeds at L0: 0.658 / 0.669 / 0.665 (mean 0.664). Worst person 0.50 at L0 (100004). SpO2 MAE 4.98 points at L0.

PhotonNet does NOT beat or tie chain_v1 at clean light: it loses by 0.128 (CI excludes 0). It is flat under the simulated distant flash (0.67-0.72 from
L0 to L5 and X1) and beats chain_v1 at L5 and X1, where chain_v1 is at chance. It answers about a third of the windows at every level (16 % at L5).

## Null controls (ROUND2 rule 5: global permutation of the training labels, 3 draws, seed 0, whole nested pipeline)

| control | draw 0 | draw 1 | draw 2 | mean | rule (<= 0.55) |
|---|---|---|---|---|---|
| photonnet_null (with physics layer) | 0.614 | 0.617 | 0.624 | **0.618** | **FAILS** (evalkit flags `photonnet_null`) |
| photonnet_nophys_null (same net, physics layer removed) | 0.510 | 0.500 | 0.473 | 0.494 | passes |

Files: `photonnet_null__L0.csv` = draw 0, `photonnet_null_d{0,1,2}__L0.csv`, `photonnet_nophys_null__L0.csv` = draw 0, `photonnet_nophys_null_d{0,1,2}__L0.csv`.
The null models' inner TM against their own permuted labels is 0.50-0.51 in every fold (photonnet_choices.csv), so the label permutation works. The
held-out person is never in the permutation and its labels are used only by evalkit. The 0.618 comes from the physics layer: with labels carrying no
information, the reconstruction loss still forces d_sat to explain the observed DC and pulse-amplitude changes through PhotonTwin, and d_sat enters the
SpO2 head directly, so the network makes a label-free physics inversion (the same baseline-relative DC colour cue round 1 found robust). The ablation
without the physics layer scores 0.494 under the same null. This is a label-free prior, not leakage, but the rule is mechanical: **PhotonNet fails rule 5
and is not eligible under the round-2 winner rule as specified.** The null is not a "no information" reference for a physics-informed model; whoever
applies the winner rule should decide whether a label-free physics null (e.g. physics-only inversion as the reference) is the right comparison, and
that decision would be made after seeing this number.

## Ablation: photonnet_nophys (same net and loss, physics layer removed)

| level | photonnet | photonnet_nophys | nophys coverage |
|---|---|---|---|
| L0 | 0.668 | 0.642 [0.528, 0.749] | 0.47 |
| L3 | 0.689 | 0.650 | 0.29 |
| L4 | 0.703 | 0.657 | 0.12 |
| L5 | 0.720 | 0.662 | 0.04 |
| X1 | 0.672 | 0.663 | 0.23 |

The physics layer adds 0.01-0.06 TM at every level (no paired test between the two was run) and keeps coverage higher under stress.

## Export (ROUND2 rule 7)

`python -m oxydrop_ml.photon.photonnet_export` (ran on Kaggle with the rest): final fit on all 6 people, 3 seeds, steps = modal nested choice per seed
(150 for all three), c* = median of the 18 nested choices. `results/photon/photonnet_weights.json` (321 KB, format `oxydrop-photonnet-v1`: per-seed
input clip bounds and scaler, Linear weights, sat_b, psd, c*) and `results/photon/photonnet_ref.json` (features and expected mu, sigma, score, conf,
answer of 2 windows, t = 60 s and 200 s, of tests/fixtures/ppg-csv/Left/100001.csv; mu 97.79 / 96.75, score 0.020 / 0.114). The torch and plain-numpy
forwards agree (asserted in the export; checked again in the test). Inference is Linear + tanh + softplus + erf only; the PhotonTwin layer is a
training-time head. Not done: the JS port. The network is about 40 lines of JS, but the 39-feature extractor (flashgap.rec_features: 15 quality and 5
respiratory features beyond the chain deltas) has no JS implementation, and app/js/cnn.js does not apply (not OxyNet-shaped). The reference features
of the 2 windows were computed by `flashgap.rec_features` locally (results/photon/photonnet_ref_features.json).

## Bottom line

- L0 TM bal. acc. 0.668 [0.508, 0.785] vs chain_v1 0.796: loss, paired diff -0.128 [-0.208, -0.047], 2/6 people improved. Seed mean 0.664.
- Under the simulated distant flash it stays flat (0.68-0.72 to L5, 0.672 at X1). Paired diff vs chain_v1: L3 +0.125 (tie), L4 +0.105 (tie),
  L5 +0.185 [0.023, 0.290] (win), X1 +0.165 [0.020, 0.316] (win, 6/6 people).
- Coverage 0.35 at L0, 0.16 at L5; answered accuracy 0.92 at L0 and 0.87-0.95 under stress.
- Null with physics layer 0.618 (flagged), without 0.494. Not eligible under rule 5 as written.

## Compute disclosure

Main run local (one process, 2 threads, 18:43-19:01 CT, 108 fits, peak RSS ~0.5 GB). After the planner's OOM warning the nulls, the nophys ablation and
the export fit ran on Kaggle CPU (kernel aravhawk/oxy-photonnet-remote, private dataset aravhawk/oxy-photonnet-bundle = the same feature tables packed by
`photonnet.make_bundle`, Hoffman-derived only). An earlier local attempt (18:31-18:42, forked pool of 2) was killed before it wrote anything.
