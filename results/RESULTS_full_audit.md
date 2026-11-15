# VERITAS pivot: phone-only multi-stage hypoxia pipeline (RESULTS)

Sat 2026-11-14. OPUS (architect/verifier) + SONNET (data hunt, stage tests). Plan: `PLAN.md`. Raw log: `NOTES.md`. Every number comes from a script and log under `pivot/opus/`, `pivot/results/` or `multi/scripts/`, `multi/out/` (SONNET).

> **Status after the independent review (REVIEW.md, PR #3; see `docs/REVIEW_RESPONSE.md`).** This file is a historical lab note written while the experiments were running. The scripts it cites under `pivot/`, `opus/` and `multi/` are not in this repository. Where the review could not reproduce a claim it is marked **UNSUPPORTED** below, and the numbers are kept only so the reader can see what was claimed. Only tables that `scripts/make_eval.py` regenerates from `results/scores/` (the main table, per-person and coverage tables) are saved-score arithmetic that has been checked; the nested training itself was not rerun.

## 1. Verdict (read this first)
- **Best full-coverage pipeline: "TINY chain".** Two logistic regressions on causal, calibration-minute-relative stage outputs (ΔHR; ΔHR/HRV/RR/perfusion + Δphysics absorbance/ratio-of-ratios), probabilities averaged. No neural network. 62 floats.
  **acc 0.870 / bacc 0.869 / sens 0.879 / spec 0.858 / AUROC 0.942 / time-matched 0.795 / worst person 0.78** on held-out people (person-bootstrap acc CI 0.83–0.91, TM CI 0.69–0.83).
  The previous honest best was 0.856 / 0.861 / TM 0.802 / worst 0.75 with a CNN ensemble. TINY matches or beats it with ~1/1000 of the parameters.
- **90 % / 90 % at full coverage is NOT reached and will not be by Sun 10 PM CT.** The gain over the time-only clock (0.835) is +3.5 pts (+3.47 on the saved scores), with a paired person-bootstrap CI of [−6.8, +16.7] pts (`scripts/review_checks.py`). Six test people cannot resolve 3-pt differences, and the time-matched (clock-free) accuracy is ~0.80 for every good model. More modelling on 6 people will not move this; only more hypoxic people would. None were found in open data, though that search was not exhaustive or dated (UNSUPPORTED as a count).
- **90 / 90 is reached with abstention ("unsure, re-measure").** With a **fully nested** cutoff (chosen in inner LOPO on the 5 training people to hit ≥ 0.90 acc and bacc, same protocol as SONNET t6; **UNSUPPORTED independently**: `make_eval.py` copies these constants from an external experiment and does not rerun the cutoff selection), TINY gets **acc 0.903 / bacc 0.902 / AUROC 0.956 at 86.7 % coverage, worst person 0.80**. Time-matched accuracy on the answered windows is 0.816, while the clock scores 0.511 on those same windows. A nested target of 0.92 gives 0.920 / 0.917 at 80 % coverage. A fixed a-priori band |p−0.5| < 0.10 gives 0.925 / 0.923 at 85 %. Abstentions concentrate near the 90 % line (73 % coverage within ±3 pts of 90 vs 87 % overall). This is a selective-prediction claim and must be presented that way.
- **What the chain buys:** the data-rich stages, not raw phone hypoxemia data, carry the signal. **Heart-rate rise vs the person's own first minute** (hypoxic tachycardia) tracks SpO2 in all 6 people (per-person Spearman −0.74 to −0.95). An HR→hypoxia stage trained only on a 20-person altitude-chamber set was reported to rank Hoffman windows at AUROC 0.88 zero-shot (**UNSUPPORTED** here: no self-contained rerun). **Red-channel DC absorbance** (Beer-Lambert: deoxy-Hb absorbs red) was reported as a second consistent signal (ρ +0.62 to +0.96; **UNSUPPORTED**, and it would not show oxygen-specific sensing since it is partly drift and time confounded). Ratio-of-ratios on phone RGB is weak (TM 0.60 alone).
- **Biggest product risk:** ΔHR is not oxygen-specific. The deployable rule flags **63.4 % of normoxic exercise windows** as hypoxic (moshekam, 14 subjects; 68.4 % over the 13 with a usable heart rate). The app must enforce 60 s seated rest plus a motion gate, and the claim is limited to "at rest".

## 2. Main table (Hoffman, leave-one-person-out, nested; 5,644 eval windows, base rate 0.575)
| Model | acc [95 % person-boot] | bacc | sens | spec | AUROC | TM acc [95 %] | worst person | Δacc vs clock [95 %] |
|---|---|---|---|---|---|---|---|---|
| Time-only clock (elapsed s; not deployable, control) | 0.835 [0.70, 0.93] | 0.830 | 0.864 | 0.797 | 0.819 | 0.519 [0.50, 0.59] | 0.54 | 0 |
| Prior honest best (CNN-cal + CNN-uncal + LGBM ensemble) | 0.856 [0.80, 0.90] | 0.861 | 0.827 | 0.894 | 0.941 | 0.802 [0.70, 0.82] | 0.75 | +0.020 [−0.11, +0.16] |
| S2 physics only: ΔRoR (LR) | 0.732 | 0.721 | 0.793 | 0.648 | 0.755 | 0.595 | 0.57 | |
| S2 physics: ΔDC absorbance only | 0.815 | 0.816 | 0.811 | 0.821 | 0.904 | 0.665 | 0.72 | |
| S1 cardio: ΔHR only (1 feature) | 0.751 [0.60, 0.88] | 0.759 | 0.706 | 0.812 | 0.855 | 0.791 [0.64, 0.84] | 0.51 | −0.082 [−0.29, +0.15] |
| S1 cardio: all deltas (LR) | 0.726 | 0.739 | 0.654 | 0.823 | 0.836 | 0.759 | 0.51 | |
| S1+S2+DC fusion LR (19 inputs) | 0.843 [0.80, 0.88] | 0.841 | 0.857 | 0.825 | 0.933 | 0.752 | 0.73 | |
| S1+S2+DC fusion, LightGBM | 0.816 | 0.810 | 0.845 | 0.776 | 0.895 | 0.729 | 0.69 | |
| **TINY chain = avg(ΔHR-LR, fusion LR)** | **0.870 [0.83, 0.91]** | **0.869** | 0.879 | 0.858 | 0.942 | **0.795 [0.69, 0.83]** | **0.78** | +0.037 [−0.07, +0.16] |
| TINY + causal output EWMA (6 s) | 0.868 | 0.866 | 0.876 | 0.856 | 0.941 | 0.782 | 0.79 | |
| BIG = avg(prior BEST, fusion LR) | 0.877 [0.83, 0.91] | 0.876 | 0.879 | 0.873 | 0.954 | 0.788 [0.68, 0.82] | 0.78 | +0.044 [−0.06, +0.17] |
| AVG3 = avg(prior BEST, fusion LR, ΔHR-LR) | 0.879 [0.85, 0.91] | 0.882 | 0.862 | 0.902 | 0.952 | 0.801 [0.69, 0.82] | 0.83 | +0.045 [−0.06, +0.17] |
| Zero-shot: ΔHR stage trained on altitude chamber only | 0.725 | 0.676 | 0.998 | 0.355 | 0.880 | 0.647 | 0.51 | (threshold does not transfer) |

TM = time-matched accuracy: 300 random subsets in which every 100 s elapsed-time bin is class-balanced, so the clock scores ~0.52; the CI uses class-reweighted bins with person bootstrap. Per-person accuracy for TINY (people 1–6): 0.92, 0.89, 0.85, 0.78, 0.92, 0.88.

### Selective prediction, NESTED cutoff (reported from an external experiment, `opus/t10_nested_abstain.py`; UNSUPPORTED independently: not rerun in this repository)
Cutoff c per held-out person = smallest |p−0.5| band at which the inner-LOPO out-of-fold scores of the 5 training people reach acc and bacc ≥ target. Inner models use the modal outer hyperparameters (C=0.01 / 0.1, no EWMA).
| Model | target | coverage | acc | bacc | sens | spec | AUROC | TM on answered (clock on same) | worst person | coverage within ±3 of 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| TINY | 0.88 | 0.913 | 0.896 | 0.896 | 0.898 | 0.894 | 0.952 | 0.806 (0.511) | 0.79 | 0.82 |
| **TINY** | **0.90** | **0.867** | **0.903** | **0.902** | 0.908 | 0.896 | 0.956 | 0.816 (0.511) | 0.80 | 0.73 |
| TINY | 0.92 | 0.800 | 0.920 | 0.917 | 0.935 | 0.900 | 0.961 | 0.845 (0.509) | 0.80 | 0.58 |
| TINY | 0.95 | 0.654 | 0.950 | 0.948 | 0.963 | 0.933 | 0.972 | 0.903 (0.510) | 0.87 | 0.39 |
| SONNET Hoffman-only LGBM (t6) | 0.90 | 0.72 | 0.916 | 0.919 | | | 0.956 | 0.81 | 0.81 | |
| SONNET Hoffman-only LGBM (t6) | 0.88 | 0.82 | 0.896 | 0.897 | | | | | | |

### Selective prediction, a-priori fixed bands (secondary; all bands tried are reported)
| Model | band | coverage | acc | bacc | TM on answered | clock on same windows (acc / TM) | worst person |
|---|---|---|---|---|---|---|---|
| TINY | \|p−.5\|<0.10 | 0.85 | **0.925** | **0.923** | 0.853 | 0.844 / 0.512 | 0.83 |
| TINY | <0.15 | 0.77 | 0.943 | 0.942 | 0.883 | 0.851 / 0.516 | 0.88 |
| TINY | <0.20 | 0.66 | 0.953 | 0.952 | 0.908 | 0.844 / 0.517 | 0.88 |
| AVG3 | <0.10 | 0.85 | 0.927 | 0.928 | 0.849 | 0.839 / 0.517 | 0.88 |
| AVG3 | <0.15 | 0.75 | 0.947 | 0.947 | 0.885 | 0.843 / 0.512 | 0.90 |
| prior BEST | <0.10 | 0.87 | 0.887 | 0.891 | 0.820 | 0.839 / 0.519 | 0.76 |

## 3. Stage-by-stage findings (what worked, what did not)
| Stage | Data | Result |
|---|---|---|
| S0 quality gate | BUT PPG expert quality labels (50 people) | SONNET: AUROC 0.756 on BUT; on Hoffman +3 pts at 60 % coverage, corr(q, error) −0.08. **Weak**: confidence abstention beats quality abstention. |
| S1 heart rate / HRV / RR / perfusion | DSP; altitude chamber (20 people) for the ΔHR→hypoxia link; BUT PPG (18 people, ECG reference) for HR accuracy | **HR extractor accuracy (`opus/t11_but_hr.py`):** finger clips at rest MAE 11.1 bpm (median 6.7, 43 % within 5 bpm); expert-good clips MAE 4.5 (77 % within 5). On Hoffman (locked camera settings, at rest) the same extractor is accurate: **MAE 1.2 bpm vs the reference oximeter pulse, 99 % within 5 bpm** (`opus/t12_oracle_hr.py`). **Oracle ceiling:** replacing camera HR with reference pulse lifts TINY only to 0.879 / 0.881 / TM 0.826 (+1 pt). So a better HR stage matters for robustness on uncontrolled phones (BUT-like conditions), not for reaching 90 %. **Strongest clock-free signal.** ΔHR TM 0.79; HRV/RR alone weak (TM 0.60); perfusion alone TM 0.64 (these two from the pre-fix ablation run, `results/opus_t2_ablate.log`). Altitude-trained stage AUROC 0.88 zero-shot. |
| S2 physics (Beer-Lambert pseudo-absorbance, RoR) | none | ΔDC_R strong but time/drift-confounded (TM 0.67); RoR R/G weak, with sign inconsistent in 1–2 of 6 people. |
| S3 red/IR source-domain regressor | HealthRing, moshekam | Not used: batch audit showed ring→phone ratio direction inconsistent and dataset identity decodable at AUROC ≥ 0.98. |
| S4 synthetic desaturation | BUT+MTHS normoxic + physics | SONNET: synthetic-only AUROC 0.62 on real Hoffman; Hoffman+synthetic no gain (0.826 vs 0.830). |
| S5 causal trend (EWMA) | none | No gain (nested CV picks no smoothing; output EWMA −0.2 to −1.3 pts). |
| Pooling other phone datasets (BUT, MTHS) | 99 normoxic people | SONNET: best +MTHS ComBat 0.865 / 0.857 / TM 0.726 (best of 6 variants); naive pooling hurts; pooling lowers selective coverage at equal accuracy. |
| Data hunt | Candidate datasets checked (SONNET, `multi/DATA_HUNT_NEW.md`, not in this repository; the "32" count is UNSUPPORTED as exhaustive) | **Scarcity as far as we found:** no new open dataset with an optical signal and SpO2 < 90. ARPOS (34 people, face) is normoxic; OpenOximetry, MSPM, VitalVideos and VIPL are gated. Only the altitude-chamber set (HR/SpO2, no optical) was usable, for the ΔHR stage. |

## 4. Leakage and confound audit (OPUS)
1. **Found and fixed in my own pipeline:** the leading NaN heart-rate windows (first ~20 s, filter warm-up) had been filled with the whole-recording median, which put future hypoxic HR into the calibration baseline. Fixed to forward fill, then the calibration-minute median. The before/after accuracy change is UNSUPPORTED (not reproduced by any shipped script, so the numbers are removed). All numbers above are post-fix.
2. No elapsed-time feature. Calibration = median over windows with t + 10 ≤ 60 s (label-free). All stage features use trailing (causal) windows, causal IIR filters (`sosfilt`, not `filtfilt`), and both hands of a person held out together. Scaler, C, and EWMA are chosen in an inner LOPO over the 5 training people. Spearman(score, elapsed t) = +0.22 for TINY vs +0.21 for the clock score itself.
3. **Clock control:** TM accuracy (clock 0.52) is 0.795 for TINY.
4. **Label-swap control** (train on another person's label trajectory at the same elapsed time; swapped labels agree 71–78 % with truth): TINY still got a high accuracy (the exact range is UNSUPPORTED: not reproduced by the shipped audit scripts). For low-capacity models this mostly shows robustness to label noise: a 1-feature ΔHR model learns the right sign from 75 %-correct labels. It does not show that the signal is the clock. SONNET's equivalent N2 control (0.780 / 0.653, their GBM) is read the same way: partly timeline, partly label-noise-robust physiology.
5. **SONNET positive controls** (adding elapsed t, whole-recording z-score, window-level split, filtfilt) were run to check that the evaluation catches leaks. Their numbers are **UNSUPPORTED independently** (not reproduced by the shipped audit scripts) and are removed here. None of these variants is in any reported row.
6. **Selection optimism:** ~10 fusion/average variants compared on the same 6 people (TINY, BIG, AVG3, smoothing, ablations). Treat the TINY-vs-prior-BEST difference as a tie (the +1.4 pts figure is UNSUPPORTED independently). The robust claims are TM ≈ 0.80 and worst person ≈ 0.78 (saved-score arithmetic); the nested abstention figure is UNSUPPORTED independently.
7. SONNET t6 nested-selective protocol audited (`multi/scripts/t6.py`): the cutoff comes only from inner-LOPO OOF scores of training people; the per-window dump is `multi/out/t8_remote/t6_windows_tgt090.csv`. No leak found. The TINY nested version has one small optimism: inner hyperparameters fixed to the modal outer choice.
8. **Do not headline** SONNET's pooled many-person accuracy: most of those people are normoxic and trivially right, and sensitivity is 0 on BUT/MTHS positives. The pooled numbers (0.911 / 0.900) are UNSUPPORTED independently.

## 5. On-phone deployability
- TINY chain: per-frame ROI mean RGB (any camera API) → causal 3rd-order IIR band-pass → beat detection → 19 window statistics → Δ vs calibration median → two logistic regressions → average → decision/abstain.
- **Model: 62 floats (1.9 KB JSON weights, `pivot/opus/tiny_chain_weights.json`); reference inference `pivot/opus/tiny_chain.py`.** Unoptimised Python on the host: **0.6 ms per decision including feature extraction.** On a phone this is microseconds in Swift/Kotlin, or a ~2 KB ONNX/CoreML/TFLite graph (LR = MatMul + Add + Sigmoid). Battery and compute are dominated by the camera and flash, not the model.
- SONNET's LightGBM alternative exported to ONNX: 112 KB, ~1.2 ms/window (measured).
- BIG/AVG3 add the prior 1-D CNN ensemble (~10–25 k params, < 100 KB as ONNX/TFLite int8, ~1 ms per window). The +0.7–0.9 pt raw gain is within noise, so it is not recommended for v1.
- Requirements: flash on, fingertip covering the lens, 60 s seated-rest calibration each session, motion gate (accelerometer), abstain on low confidence. The calibration assumes the user is normoxic at the start: it detects a *drop from personal baseline* and will miss chronic hypoxemia that is already present at calibration. The app now says "oxygen status at start unknown" on screen.

## 6. Limitations (say these out loud)
- 6 test people, one phone (Nexus 6P), fingertip only, 1 dark-skinned person, controlled FiO2 protocol with a one-way desaturation shape. CI half-width ±4–8 pts on accuracy and ±7 on TM.
- ΔHR is a physiological correlate (hypoxic tachycardia), not an oxygen measurement. Exercise, anxiety, caffeine, beta-blockers and arrhythmia break it: 63.4 % false alarms during exercise.
- Face video: no open face dataset with real desaturation exists (SONNET hunt; OpenOximetry, MSPM, VitalVideos, VIPL all gated or normoxic), so the face path is unvalidated and not claimed.
- Hoffman reference oximeters disagree by up to ~6 pts at the low end; labels are their mean.

## 6b. What could still improve by Sun 10 PM (ranked)
1. Robust S1 heart-rate stage for uncontrolled phones (auto-exposure, motion), trained/tuned on BUT PPG ECG-HR, plus a beat-quality gate. This matters for real-world use, where BUT finger-at-rest is only 43 % within 5 bpm. It is not expected to move the Hoffman numbers much: an earlier perfect-HR oracle experiment reported only +1 pt (0.879; UNSUPPORTED independently).
2. Motion/rest gate from the accelerometer plus a 'recent exertion' check (HR above resting norms at calibration). This is required for safety, not accuracy.
3. Nothing else is worth the time: pooling, synthetic desaturation, quality gating, smoothing, deep nets and red/IR transfer were all tested today with no gain.

## 7. Blunt answer: is 90/90 reachable by Sun 10 PM CT?
**At full coverage on held-out people: no.** Best is 0.870 / 0.869 (CI upper ~0.91); an earlier perfect-heart-rate oracle experiment reported only 0.879 / 0.881 (UNSUPPORTED independently). The clock-free ceiling sits at ~0.80 across ~40 model variants from three independent pipelines. The only lever that would change this is more people with real phone-camera desaturation, and there are none in open data.
**With honest abstention: yes, already.** With a nested cutoff: 0.903 / 0.902 at 87 % coverage, clock-free TM 0.82 on the answered windows, worst person 0.80. A nested 0.92 target gives 0.920 / 0.917 at 80 %. Present it as "answers ~87 % of readings at 90 % accuracy, and asks to re-measure otherwise; at rest only; validated on 6 people", always next to the time-only baseline (0.835 raw, 0.51 time-matched). Worst-person accuracy (0.80, person 4) is still below 90, so "90 % for every person" is not claimable.
