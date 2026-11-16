# Control models: is the robustness of the physics inversion just the baseline-relative DC colour?

Code: `python/oxydrop_ml/photon/controls.py` (`cd python && python -m oxydrop_ml.photon.controls`; tables: `... controls --table`). Tests: `python/oxydrop_ml/tests/test_photon_controls.py`.
Score files: `results/photon/scores/ctl_{dc_lr,dc_ror_lr,dc_hr_lr}__L0..L5.csv`, `__X1.csv`, and the null controls `ctl_*_null__L0.csv`. Evaluator: `python -m oxydrop_ml.photon.evalkit` (rows in `TABLE.csv`, `TABLE.md`, `STRESS.csv`).

## Models

All three are class-balanced logistic regressions on the calibration-relative features of the shipped chain (`oxydrop.chain.deltas`: window feature minus the median over the windows ending within the first 60 s of the same recording), computed from the degraded raw RGB. No physics model, no neural network.

| model | inputs |
|---|---|
| `ctl_dc_lr` | dc_R, dc_G, dc_B: mean over the 10 s window of ln(rgb + 1) per channel, minus the first-minute median (3 features) |
| `ctl_dc_ror_lr` | the 3 DC values + ratio-of-ratios ror_RG, ror_RB, ror_GB (ln ratio of the AC standard deviations) + perfusion index pi_R, pi_G, pi_B (ln AC standard deviation per channel) (9 features) |
| `ctl_dc_hr_lr` | the 3 DC values + dHR (heart-rate change vs the first minute, the chain's peak-based estimator) (4 features) |

Protocol (same harness as `chain_v1` in `baselines_stress.py`, whose loaders, degraded-feature hook and score writer are reused): leave-one-person-out over the 5,644 evaluated windows, both hands together. Per fold the standardiser and the logistic regression are fit on the clean training people only. C in {0.001, 0.01, 0.1, 1, 10} is chosen by the inner person-grouped 5-fold CV of the training people (criterion: `metrics.time_matched` TM bal. acc., fixed threshold 0.5). The `answer` flag is `|p - 0.5| >= c` with c from `common.choose_band` (smallest c whose answered inner-OOF windows reach accuracy and bal. acc. >= 0.90). The model is not refit per level; only the held-out person's two recordings are degraded (`stress.degrade`, `stress.degrade_x`, baseline minute included), and the features are recomputed from them. The held-out labels are read only to score. The tests check that scrambling the held-out person's labels, times and features changes no C, threshold, abstain half-width or training score.

Chosen C per fold (6 folds): `ctl_dc_lr` 0.001 in all folds; `ctl_dc_ror_lr` 0.01, 0.1, 0.001 x4; `ctl_dc_hr_lr` 0.01, 0.001 x4, 0.1. Abstain half-width c: 0.14 to 0.21 (`dc_lr`), 0.21 to 0.39 (`dc_ror_lr`), 0.15 to 0.25 (`dc_hr_lr`). Coverage of `answer` is 45 to 57 % (`dc_lr`), 34 to 50 % (`dc_ror_lr`), 54 to 72 % (`dc_hr_lr`); TM below uses the forced decision `pred` at full coverage, as for every model.

Null controls (the held-out person's recordings replaced by a constant colour, same pipeline): TM bal. acc. 0.500 for all three (`null_check` ok).

## TM bal. acc. [95 % person-bootstrap CI] by level

Source: `TABLE.csv` (2000 person resamples, simulated stress, 6 held-out people, so the CIs are wide).

| model | L0 | L1 | L2 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|---|---|
| chain_v1 | 0.796 [0.64, 0.86] | 0.805 [0.65, 0.86] | 0.742 [0.59, 0.82] | 0.565 [0.50, 0.63] | 0.598 [0.53, 0.64] | 0.535 [0.48, 0.60] | 0.508 [0.44, 0.55] |
| twin_inverse_noresid | 0.693 [0.55, 0.76] | 0.694 [0.55, 0.76] | 0.696 [0.55, 0.76] | 0.719 [0.56, 0.77] | 0.715 [0.62, 0.75] | 0.647 [0.56, 0.72] | 0.693 [0.61, 0.75] |
| twin_inverse | 0.752 [0.58, 0.81] | 0.757 [0.58, 0.80] | 0.706 [0.57, 0.78] | 0.652 [0.52, 0.77] | 0.624 [0.55, 0.70] | 0.625 [0.54, 0.72] | 0.615 [0.53, 0.68] |
| ctl_dc_lr | 0.744 [0.54, 0.85] | 0.741 [0.54, 0.84] | 0.722 [0.53, 0.81] | 0.692 [0.53, 0.77] | 0.658 [0.51, 0.75] | 0.599 [0.54, 0.65] | 0.646 [0.51, 0.73] |
| ctl_dc_ror_lr | 0.701 [0.54, 0.78] | 0.684 [0.54, 0.76] | 0.670 [0.54, 0.75] | 0.694 [0.60, 0.76] | 0.639 [0.58, 0.69] | 0.566 [0.50, 0.64] | 0.643 [0.56, 0.70] |
| ctl_dc_hr_lr | 0.694 [0.55, 0.79] | 0.698 [0.56, 0.79] | 0.660 [0.54, 0.76] | 0.613 [0.50, 0.72] | 0.615 [0.54, 0.67] | 0.590 [0.52, 0.68] | 0.580 [0.47, 0.67] |

Paired person-bootstrap difference in TM bal. acc. to `twin_inverse_noresid` at the same level (model minus `twin_inverse_noresid`; [95 % CI]; * = CI excludes 0):

| model | L0 | L1 | L2 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|---|---|
| chain_v1 | +0.103 [+0.01, +0.23]* | +0.111 [+0.01, +0.23]* | +0.046 [-0.03, +0.14] | -0.154 [-0.23, -0.02]* | -0.117 [-0.18, -0.06]* | -0.112 [-0.21, -0.02]* | -0.185 [-0.26, -0.09]* |
| twin_inverse | +0.059 [+0.00, +0.12]* | +0.063 [+0.01, +0.12]* | +0.010 [-0.04, +0.06] | -0.068 [-0.18, +0.03] | -0.091 [-0.16, -0.02]* | -0.022 [-0.09, +0.05] | -0.078 [-0.14, -0.03]* |
| ctl_dc_lr | +0.051 [-0.02, +0.15] | +0.047 [-0.02, +0.14] | +0.026 [-0.03, +0.10] | -0.027 [-0.08, +0.03] | -0.057 [-0.14, +0.01] | -0.048 [-0.10, -0.01]* | -0.047 [-0.14, +0.00] |
| ctl_dc_ror_lr | +0.008 [-0.03, +0.06] | -0.010 [-0.04, +0.03] | -0.026 [-0.06, +0.01] | -0.025 [-0.13, +0.06] | -0.077 [-0.14, -0.00]* | -0.081 [-0.14, -0.02]* | -0.050 [-0.08, -0.03]* |
| ctl_dc_hr_lr | +0.002 [-0.08, +0.10] | +0.004 [-0.08, +0.10] | -0.036 [-0.10, +0.04] | -0.106 [-0.21, -0.01]* | -0.100 [-0.15, -0.04]* | -0.056 [-0.14, +0.02] | -0.113 [-0.19, -0.03]* |

AUROC (`TABLE.csv`), L0 / L3 / L4 / L5: chain_v1 0.947 / 0.557 / 0.621 / 0.542; twin_inverse_noresid 0.859 / 0.894 / 0.908 / 0.871; ctl_dc_lr 0.874 / 0.872 / 0.872 / 0.858.

## Reading

* A 3-number logistic regression on the baseline-relative DC colour (`ctl_dc_lr`) is as robust to the simulated distant-flash stress as `twin_inverse_noresid`. It falls from 0.744 (L0) to 0.692 (L3), 0.658 (L4), 0.599 (L5); `twin_inverse_noresid` stays at 0.69 to 0.72 through L4 and 0.647 at L5. The paired CI of the difference includes 0 at L0 to L4 and X1 (point estimates +0.05 at L0 to L2, -0.03 to -0.06 at L3 to X1), and excludes 0 only at L5 (-0.048 [-0.10, -0.01]). Its AUROC is flat (0.87 at L0, L3, L4; 0.86 at L5), as is that of `twin_inverse_noresid`. The robustness of `twin_inverse_noresid` is therefore mostly the robustness of the DC cue itself; this run gives no evidence that the physics inversion adds accuracy under stress beyond it. The most that can be said for the physics model is a small, not significant, edge at L3 to L5 and X1 (up to +0.05).
* Adding the AC optics features without physics (`ctl_dc_ror_lr`: ratio-of-ratios and perfusion index) does not help: its point estimate is below `dc_lr` at L0 to L2 and L4, and it is below `twin_inverse_noresid` at L4, L5, X1 (CI excludes 0). These AC features lose their signal as pulsatile SNR falls (red-channel pulsatile SNR 13 dB at L3, 0 dB at L4, -13 dB at L5), which is the same mechanism that collapses `chain_v1`.
* Adding dHR (`ctl_dc_hr_lr`) lowers the score at every level against `dc_lr` (0.694 vs 0.744 at L0, 0.613 vs 0.692 at L3, 0.580 vs 0.646 at X1). The heart-rate input is the least stable cue and, with this training set, no more informative than DC on the clean data either. Without HR the model is flatter, consistent with `twin_inverse_noresid` having no HR input.
* Clean data: the controls do not reach `chain_v1` (0.796) at L0. Against `chain_v1` they are ties at L0 to L2 (`ctl_dc_lr` +0.05 below at L0, CI [-0.16, +0.04]); `ctl_dc_ror_lr` and `ctl_dc_hr_lr` lose at L0 and L1. From L3 on the controls are higher than `chain_v1` (point estimates +0.02 to +0.14; CI excludes 0 only for `dc_ror_lr` at L3 and L4, and `dc_hr_lr` at L5; `ctl_dc_ror_lr` X1 +0.136 [+0.05, +0.22]).
* The chosen C is almost always the smallest of the grid (0.001): with 6 training people the inner CV prefers heavy shrinkage, so `ctl_dc_lr` is close to an equal-weight standardised-DC score. A less regularised grid was not tried (grid fixed in advance to match `chain_v1_nestedC`).

## Caveats

* Simulation, 6 held-out people: every CI above is wide and differences of 0.05 are not resolved. `ctl_dc_lr` is not claimed better than `twin_inverse_noresid`; the two cannot be separated here.
* Features are ln window means relative to the first minute of the same (degraded) recording, so any multiplicative attenuation that is constant over the recording cancels by construction. The stress simulation is mostly such an attenuation (auto-exposure restores red, blue falls by a constant factor) plus an additive non-pulsatile leak (2 to 35 %), which dilutes the relative DC change but keeps its sign. A real phone with a different gain control or colour processing could break this cancellation for the controls and for any model that uses the same cue.
* `twin_inverse_noresid` rarely answers at high levels (`coverage` 0.23 at L3, 0.005 at L4, 0 at L5); the TM numbers use the forced decision `pred`, so the comparison above is at full coverage. The controls answer 34 to 72 % at L3 to L5 by the nested abstain rule.
* "Perfusion index" is taken as the three per-channel ln AC standard deviations (pi_R, pi_G, pi_B) of the chain, not a single scalar.
* `evalkit` was re-run for the whole `results/photon/scores` directory, so `TABLE.csv`, `TABLE.md` and `STRESS.csv` were regenerated with the models present at that time.
