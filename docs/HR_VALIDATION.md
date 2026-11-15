# Heart-rate validation (v2 estimator)

The v1 chain read heart rate by counting peaks, and a strong dicrotic notch (the second bump in each pulse) made it count two beats per heartbeat. A person at about 75 bpm read about 150. v2 replaces that with a spectral plus autocorrelation estimator and a validity gate. This page documents the estimator, how its gate was chosen without looking at the subjects it is scored on, and every number, including the ones that miss the target.

Files: `python/oxydrop/hr.py` and `app/js/hr.js` (identical logic, checked by `tests/hr_parity.mjs`), `scripts/validate_hr.py` (benchmarks), `scripts/hr_crossfit.py` (gate selection), outputs in `results/hr_validation/`.

## Estimator

For a trailing 10 s window of the band-passed (0.7 to 4 Hz) log intensity of one colour channel:

1. Hann-windowed 4096-point spectrum. Score each candidate frequency `f` in `[F_LO, 3.0 Hz]` as `P(f) + 0.7 P(2f)`. This prefers the fundamental over its second harmonic.
2. The spectral peak is refined by a parabola only if it is a true local maximum, and the refinement is clamped to half a bin.
3. Autocorrelation of the same window. The fundamental is the smallest-lag peak that reaches 0.85 of the strongest peak (the strongest peak itself if every peak is negative).
4. Accepted only if the spectral and autocorrelation rates agree within `AGREE`, the share of band power at the fundamental and its second harmonic is at least `SQI_MIN`, and the autocorrelation peak is at least `ACF_MIN`. Otherwise the app reports "not locked" and shows no number.
5. If the primary channel (green, else red) is not accepted, the other two are tried in order of amplitude.

Three bugs in the first v2 draft are fixed in both languages: unbounded parabolic refinement (a harmonic-sum maximum on a spectral slope gave rates such as -35 or +30 bpm), a crash or `undefined` when all autocorrelation peaks are negative, and a negative spectral frequency passing the agreement test. The shipped code is the validated code: `validate_hr.py` calls `hr.py` directly.

## Gate selection: subject-level cross-fit

The earlier sweep picked the gate on all BUT PPG clips and then reported those same clips, which is tuning on the test set. That sweep and its outputs were deleted. Replacement protocol (`scripts/hr_crossfit.py`):

* BUT PPG v2.0: 3,840 ten-second smartphone clips from 38 subjects, ECG reference rate, expert quality label (795 good, 3,045 bad).
* Select `(F_LO, SQI_MIN, ACF_MIN, AGREE)` on the even-numbered subjects and score on the odd-numbered ones, then swap. Grid: `F_LO` in {0.65, 0.75, 0.85}, `SQI_MIN` and `ACF_MIN` in {0.15 ... 0.60}, `AGREE` in {0.06, 0.10, 0.15}.
* Objective on the selection half: minimise the wrong-accepted rate (absolute error above 10 bpm among accepted clips) subject to accepting at least 70% of the expert-good clips. The expert label is used only to set this coverage floor and to report subsets. It is not available to the app at run time.
* If the folds disagree, ship the more conservative gate (lower acceptance over all clips).

The folds disagreed slightly (even half chose `SQI_MIN` 0.25, odd half 0.30). The shipped gate is the more conservative one:

| F_LO | SQI_MIN | ACF_MIN | AGREE |
|---|---|---|---|
| 0.85 Hz (51 bpm) | 0.30 | 0.30 | 0.15 |

### Held-out results (each clip scored with a gate chosen without its own subject half)

| Selection | Held-out accepted clips | Accepted MAE (bpm) | Accepted within 5 bpm | Wrong-accepted (>10 bpm) | Doubling | Halving | Expert-good accepted: coverage / MAE / within 5 |
|---|---|---|---|---|---|---|
| Fold A (select even, score odd) | 701 of 1,962 | 10.97 | 48.9% | 38.1% | 0.00% | 2.7% | 70.6% / 3.46 / 85.3% |
| Fold B (select odd, score even) | 658 of 1,878 | 12.88 | 45.0% | 42.4% | 0.15% | 6.4% | 70.5% / 4.09 / 83.5% |
| Pooled held-out | 1,359 of 3,840 | 11.89 | 47.0% | 40.2% | 0.07% | 4.5% | 70.6% / 3.74 / 84.5% |

(`results/hr_validation/crossfit_summary.csv`, full grid in `crossfit_grid.csv`, choice in `crossfit_choice.json`.)

### F_LO trade-off

A lower band edge lets the app track a slower resting heart rate but admits more low-frequency noise, so the gate has to be looser to keep coverage and more wrong answers get through. Pooled held-out, same protocol, F_LO fixed:

| F_LO | Lowest trackable HR | Accepted MAE | Within 5 bpm | Wrong-accepted | Expert-good accepted MAE / within 5 |
|---|---|---|---|---|---|
| 0.65 Hz | 39 bpm | 20.15 | 33.7% | 58.8% | 7.72 / 76.9% |
| 0.75 Hz | 45 bpm | 17.22 | 37.2% | 53.5% | 5.43 / 81.5% |
| 0.85 Hz | 51 bpm | 11.89 | 47.0% | 40.2% | 3.74 / 84.5% |

The cross-fit supports 0.85 Hz, so the app cannot lock on a resting rate below 51 bpm. Consequence on Hoffman: person 4 rests at 43 bpm. With 0.85 Hz his pulse never locks (0% valid windows on both hands), the chain has no calibration baseline, and every window is reported as "not locked" instead of a guess. The other five people lock on 96 to 100% of windows. Without the band edge the unlocked estimate for person 4 is a doubled rate (the second harmonic at 86 bpm): at least 90 of his 330 ten-second windows, none of them accepted by the gate. The app's calibration plausibility range (`MIN_HR` in `app/js/app.js`) now comes from the same constant (`MIN_TRACKABLE_BPM`, 51), so it tells such a user that the prototype cannot track their resting rate. Choosing F_LO from BUT PPG is a choice for BUT's rate distribution; a larger recruited set of athletes with low resting rates would be needed to decide whether a lower edge is worth its cost.

## Benchmark table (BUT PPG, shipped gate)

`results/hr_validation/HR_SUMMARY.csv` has every subset. Estimators: old peak interval (v1 feature), naive FFT peak (largest spectral peak 0.7 to 3.5 Hz), v2 on every clip with no gate, v2 accepted by the gate. "Accepted" rows are in-sample for the gate (its parameters came from the same dataset, in cross-fit halves); the held-out numbers are in the table above and agree within 0.1 bpm (11.87 versus 11.89).

| Subset | Estimator | Clips | Coverage | MAE (bpm) | Median AE | Within 5 | Within 10 | Doubling | Halving |
|---|---|---|---|---|---|---|---|---|---|
| All clips | old peak interval | 3,840 | 98.1% | 15.30 | 10.0 | 33.1% | 50.5% | 3.1% | 1.1% |
| All clips | naive FFT | 3,840 | 100% | 20.73 | 17.5 | 27.1% | 37.3% | 1.0% | 12.3% |
| All clips | v2 no gate | 3,840 | 100% | 16.56 | 13.3 | 30.6% | 42.4% | 0.4% | 6.0% |
| All clips | v2 accepted | 3,840 | 35.2% | 11.87 | 6.2 | 47.2% | 60.0% | 0.07% | 4.5% |
| Expert-good | old peak interval | 795 | 99.0% | 6.49 | 2.7 | 71.2% | 84.2% | 2.0% | 0.1% |
| Expert-good | naive FFT | 795 | 100% | 7.03 | 2.3 | 71.9% | 81.8% | 0.5% | 2.9% |
| Expert-good | v2 no gate | 795 | 100% | 5.95 | 2.0 | 74.6% | 82.9% | 0.0% | 1.5% |
| Expert-good | v2 accepted | 795 | 70.6% | 3.74 | 1.7 | 84.5% | 92.0% | 0.0% | 0.9% |
| Finger at rest, good | v2 accepted | 443 | 76.7% | 3.27 | 1.5 | 85.6% | 92.4% | 0.0% | 0.0% |
| Finger at rest (all quality) | v2 accepted | 1,318 | 48.5% | 9.87 | 3.5 | 56.0% | 69.3% | 0.2% | 3.6% |
| Ear, good | v2 accepted | 207 | 64.3% | 6.00 | 2.3 | 78.9% | 87.2% | 0.0% | 3.8% |
| Motion, good | v2 accepted | 164 | 59.8% | 2.66 | 1.3 | 87.8% | 96.9% | 0.0% | 1.0% |
| Motion (all quality) | v2 accepted | 1,152 | 30.7% | 15.85 | 11.6 | 34.2% | 46.0% | 0.0% | 7.1% |

Coverage is the share of clips for which the estimator outputs a number. Doubling means the estimate is within 12% of 2x the reference, halving within 12% of 0.5x. (BUT's ear/finger coding: label 1 is the flash-lit fingertip class, label 0 the ear class; an earlier internal script had these swapped.)

### Headline, stated plainly

* **Target (MAE below 5 bpm on clean clips): met for expert-good accepted clips, 3.7 bpm held out (84.5% within 5 bpm), and for finger-at-rest expert-good accepted clips, 3.3 bpm.** It is not met for ear clips of good quality (6.0 bpm), and it is not met overall: across all accepted clips the MAE is 11.9 bpm with 47% within 5 bpm.
* The gate is what produces the improvement. Without it v2 is not better than the old peak counter on all clips (MAE 16.6 versus 15.3 bpm, 30.6% versus 33.1% within 5 bpm). The gain is that wrong answers are withheld, and the cost is coverage: only 35% of all BUT clips and 70.6% of the good ones are accepted.
* 40% of accepted clips are still wrong by more than 10 bpm. Most of them are expert-bad clips that pass the gate (in the in-sample table, 790 of the 1,351 accepted clips are expert-bad, MAE 17.6 bpm; held out, 798 of the 1,359). The gate cannot see the expert label, and a better signal quality index is the open problem.
* **Doubling is traded for halving.** On all BUT clips the old peak counter doubles 3.1% and halves 1.1%. v2 without the gate doubles 0.4% and halves 6.0%. v2 accepted doubles 0.07% and halves 4.5% (held out, pooled: 0.07% and 4.5%), on 35% of the clips. The like-for-like pair is v2 ungated against the old counter on all clips: doubling 0.4% against 3.1%, halving 6.0% against 1.1%, and MAE 16.6 against 15.3 bpm. The 0.07% figure applies to the clips the gate accepts and should always be quoted with its 35% coverage. On expert-good clips, accepted: 0.0% doubling, 0.9% halving.
* The engine's own validity rule also requires the beat-interval rate to match the spectral rate within 12% (`BEAT_AGREE`) and at most 10% saturated frames (`SAT_MAX`). These two constants are fixed heuristics. They were not in the cross-fit grid and no result file selects or evaluates them: BUT clips are 10 s long and the engine works on a 30 s buffer. Only `F_LO`, `SQI_MIN`, `ACF_MIN` and `AGREE` come from the BUT PPG cross-fit.

## Other datasets (oximeter pulse as reference)

`results/hr_validation/summary_long.csv`, 10 s windows, v2 accepted versus all:

| Dataset | Estimator | Windows | Coverage | MAE (bpm) | Within 5 | Doubling |
|---|---|---|---|---|---|---|
| Hoffman 2022, resting fingertip, 6 people | old peak interval | 1,964 | 100% | 1.20 | 98.7% | 0.0% |
| Hoffman | v2 no gate | 1,964 | 100% | 4.11 | 87.6% | 5.0% (all person 4) |
| Hoffman | v2 accepted | 1,964 | 82.9% | 1.81 | 93.9% | 0.0% |
| MTHS, phone fingertip video, 62 people | old peak interval | 1,284 | 99.4% | 11.20 | 59.6% | 1.4% |
| MTHS | v2 no gate | 1,284 | 100% | 7.94 | 64.9% | 0.1% |
| MTHS | v2 accepted | 1,284 | 82.5% | 6.17 | 72.2% | 0.0% |

On clean resting Hoffman recordings the old peak counter is already fine (1.2 bpm) and v2 accepted is slightly worse (1.8 bpm) because it drops person 4 and some windows. On MTHS v2 is better (6.2 versus 11.2 bpm) but still above 5 bpm. These datasets did not choose the gate.

## Synthetic stress test

`results/hr_validation/summary_synthetic.csv`: 300 simulated fingertip recordings at 50 to 110 bpm with a dicrotic second harmonic of 0.4 to 1.2 times the fundamental, noise down to 0 dB, motion bursts, drift and 8-bit quantisation. Old peak counter: 44.7% doubled, MAE 36.3 bpm. v2 accepted: 93.7% accepted, MAE 3.7 bpm, 0.0% doubled.

`tests/capture_sim.mjs` runs the whole capture path (jittery frame timestamps, 24/30/60 fps, dropped frames, exposure step) through the resampler and both chains. v1 beat-interval versus v2 gated, per group (doubling = estimate within 6% of 2x the true rate):

| Group | v1 doubled | v2 doubled (of accepted) | v2 accepted |
|---|---|---|---|
| Resting 55 to 100 bpm | 28.8% | 0.00% | 100% |
| Fast 110 to 125 bpm | 0.0% | 0.00% | 100% |
| Fast 130 to 160 bpm | 0.0% | 0.00% | 100% |
| Resting 52 to 75 bpm, dicrotic 2nd harmonic 1.2 to 2.0x the fundamental (the field failure) | 90.0% | 0.00% | 100% |

The request asked for strong-notch 130 to 160 bpm cases. There the v1 estimate is not doubled either, because the second harmonic of a 130+ bpm pulse (4.3 Hz and above) is removed by the 0.7 to 4 Hz band-pass. Doubling is a problem for rates below about 120 bpm, which is where the field failure happened, so the table includes the 75 bpm field case. These synthetic tests pass by construction of the simulator, so they check the code path, not the physiology. The 90% and 28.8% doubling rates for the old counter are simulator rates for a worst-case waveform. They are not field rates; the field-like figure is the 3.1% on BUT PPG.

## Limits of this validation

* BUT PPG clips are 10 s and not from the app's capture path (lens, torch, locked exposure). No real phone was available for this work, so end-to-end behaviour on live devices is untested.
* A reading at the very top of the band can be accepted (a synthetic 3.2 Hz sine reads as 180 bpm). The app's resting range check (51 to 110 bpm at calibration) limits the effect for the intended use.
* BUT subject-level folds are split by subject number parity, not randomised. There are 38 subjects, and clips from one subject are correlated, so intervals on pooled numbers are optimistic.
* The expert quality label is used to set the 70% good-clip coverage floor on the selection half.
