# OxyDrop: three-minute demo

**Runtime:** 3:00. Only the quoted narration is spoken; fewer than 420 words, including the fallback line. Leave short pauses for the screen. Metric values remain **[X]** until checked against the exact model and evaluation used in the recording.

**Recording truth check:** This script distinguishes the existing live logistic-chain app from the CNN research model. The inspected CNN report says it is not yet wired into the app and does **not** demonstrate a win over the logistic chain. Do not present a planned improvement as a result. If integration lands before recording, verify the deployed build and update the corresponding narration and labels. Do not change the evaluation claim unless new held-out evidence supports it.

## Shot list and narration

| Video time | Shot / on-screen content | Spoken narration |
|---|---|---|
| 0:00–0:20 | Presenter, then phone beside an empty space labelled “No pulse oximeter.” Caption: “For a hiker resting at a lodge — design scenario, not tested at altitude.” | “Imagine a hiker resting at a lodge, wondering whether to check their blood oxygen, but without a pulse oximeter. OxyDrop explores whether the phone already in their pocket can help flag a reason to re-check. It is a screening prototype, not a medical device.” |
| 0:20–0:35 | Overhead view: seated user, supported hand, fingertip over rear camera and flash. Cut to actual phone screen: camera permission, selected lens, placement and quality indicators. Label “Live phone capture — logistic-chain build.” | “I sit still, breathe normally, and rest my finger lightly over the camera and flash. The camera sees tiny colour changes as blood pulses through my fingertip. The app checks whether the signal is usable.” |
| 0:35–1:00 | Tap calibration. Show start and end of the same take, with an explicit “60-second calibration — middle shortened” caption and source timer. Keep waveform visible. Then allow the first measurement window to complete. | “First, it learns my own resting baseline for one minute. We have shortened that wait here. In this build, the first result needs another ten seconds, plus the camera setup time. A normal result cannot tell whether my oxygen was already low at the start.” |
| 1:00–1:25 | Show actual heart-rate tile and actual state, large enough to read. Beside it show the three exact UI labels: “Normal vs your baseline”; “Possible low oxygen: re-check with a pulse oximeter”; “Unsure, re-measure.” Label these “Available states,” not live outcomes. | “Here is the heart-rate estimate and the result. The three possible states are normal versus my baseline, possible low oxygen requiring a pulse-oximeter re-check, or unsure: re-measure. There is no oxygen percentage. This live build uses our earlier model; the CNN is evaluated separately.” |
| 1:25–2:05 | Results card: “Recorded study data; six people; both hands held out together.” Show CNN / logistic chain / clock rows, metric and uncertainty; use the fill-in card below. Caption “Full coverage: every test window counted.” | “We test on people the model did not train on, holding out both hands together. On a test that balances normal and low-oxygen examples across elapsed time, the CNN scores [X] percent, the logistic chain [X], and the clock-only baseline [X]. It beats the weaker clock baseline, but has no demonstrated win over the chain. Six people is a small test.” |
| 2:05–2:25 | Lift finger; show actual rejection reason and cleared heart-rate tile. Caption “Signal failure: an unanswered reading is not a successful detection.” | “Now I lift my finger. The signal is lost, so it should say unsure and hide the heart rate. That is an honest failure: no usable reading. An unsure result does not mean normal oxygen.” |
| 2:25–2:50 | Simple graphic: camera colour traces → red/green light-change features + personal baseline → “1D CNN, about 168,000 parameters” → heart rate / screening state / unsure. Caption “CNN research model; live integration pending.” | “Machine learning is core because the model learns pulse patterns and combines them with red-and-green light changes relative to my baseline. Our small neural network is designed to run on the phone. That design alone does not prove reliable oxygen screening.” |
| 2:50–3:00 | End card: app URL, repository URL, “Prototype • Rest only • Re-check with a pulse oximeter.” | “The next step is broader real-phone testing. OxyDrop is a screening aid, never a replacement for a pulse oximeter or medical care.” |

**If the phone does not produce an answer:** keep the real attempt and its reason on screen. Replace “Here is the heart-rate estimate and the result” with “This phone returned unsure; here is why.” Use the built-in recorded demo to explain the remaining flow, permanently labelled **“Recorded study replay — not live phone capture.”** If finger removal does not trigger rejection, show the observed behavior and say “This build failed to reject the lost signal; that needs fixing.” Never manufacture a passing take or imply that replay proves live performance.

### Results card: fill before recording

| Model / measure | Value to display |
|---|---|
| CNN, three-seed ensemble | [X]% time-matched balanced accuracy |
| Logistic chain v1, same held-out people and windows | [X]% time-matched balanced accuracy |
| Clock-only baseline, same held-out people and windows | [X]% time-matched balanced accuracy |
| CNN minus chain, paired person-bootstrap 95% interval | [X] to [X] percentage points; no demonstrated win if it includes zero |
| CNN minus clock, paired person-bootstrap 95% interval | [X] to [X] percentage points; claim “beats” only if entirely positive |

“Time-matched balanced accuracy” means balancing normal and low-oxygen cases across elapsed time, so a clock cannot win just by knowing when the study lowered oxygen. Use full coverage here; do not substitute accuracy on answered readings. If showing selective results elsewhere, display coverage and the share of **all** low-oxygen windows receiving an alert. An abstention is not a detection. These are recorded-data engine results, not live-phone clinical performance.

**Evidence for the editor:** [README](../../README.md), [PR #3 REVIEW.md](https://github.com/aravhawk/oxydrop/blob/7193dcbe74f05c3fb682f85141d24b33ed4c15ed/REVIEW.md), [existing results](../../results/TABLES.md), and [phone retest](../PHONE_RETEST.md). The older results describe the chain, not the CNN. CNN architecture: [MODEL_SPEC on ml-model-cnn](https://github.com/aravhawk/oxydrop/blob/f69ab25/python/oxydrop_ml/MODEL_SPEC.md). CNN evaluation: [CORE_RESULTS.md](https://github.com/aravhawk/oxydrop/blob/ba0538dee028851f7417521b5f25f668eeec4c8f/results/ml/CORE_RESULTS.md) and [paired comparison CSV](https://github.com/aravhawk/oxydrop/blob/ba0538dee028851f7417521b5f25f668eeec4c8f/results/ml/cnn_vs_baselines_paired.csv), inspected at `ba0538d` on `ml-model-train`. In the paired CSV use P1, seed `-1` (ensemble), baseline `chain_v1` and `clock`; do not mix the report's mean-of-seeds values with ensemble differences. Review protocol deviations and uncertainty before filling [X].

---

## Recording checklist — one-page companion

Print this section alone on one portrait page. Capture footage first, then assemble the three-minute edit above.

**Phone and person**

- [ ] Charge phone; clean lens; enable Do Not Disturb; close private notifications. Note phone, OS, browser, app URL, deployed commit and active model.
- [ ] Prefer Android Chrome for flash control. Open <https://aravhawk.github.io/oxydrop/> over HTTPS; allow camera; confirm actual torch, chosen lens and acquisition mode. Disclose ambient/front-camera fallback if used.
- [ ] Sit quietly, support elbow and phone, breathe normally, use light fingertip pressure. Never hold breath, exercise or lower oxygen for the demo. Stop if the flash becomes uncomfortable.
- [ ] Rehearse the full phone retest. The repository does not establish real-phone reliability; record an actual attempt and keep failure evidence.

**Lighting and capture**

- [ ] Use steady room light; avoid direct sun, flickering lamps and changing shadows. Keep finger position and lighting fixed through calibration and measurement.
- [ ] Film an overhead view with a second camera so finger placement and flash are visible. Check focus and glare before starting.
- [ ] Start native screen recording before opening the flow; keep waveform, timer, quality, heart rate and state readable. Test that recording does not stall the camera; keep an unrecorded comparison if it does.
- [ ] Capture narration separately in a quiet room; do not talk during calibration. Clap once to sync overhead footage and screen capture.

**Required screen footage**

- [ ] Show permission/setup, lens and mode, signal quality, calibration start/end, and first actual output. Budget at least 70 seconds after starting calibration, plus setup; capture longer than the final edit.
- [ ] Clearly caption every cut or speed-up. Show actual heart rate/state; show other state labels as examples, not fabricated readings. Do not display a predicted SpO2 percentage.
- [ ] Record finger removal, actual unsure reason and heart-rate clearing. Keep the outcome if rejection fails. Use labelled recorded study replay for a low-oxygen example; never induce it.
- [ ] Keep the live model label visible; identify CNN graphics/results as research until verified integration. Save debug summary, raw trace and deployment identifier privately with the take.

**Results and final export**

- [ ] Fill every [X] from one frozen evaluation commit. Show all three comparison rows, metric, full coverage, six held-out people and paired uncertainty. Retain the chain tie/loss; “beats clock” is a weaker-baseline claim.
- [ ] Show input → features/baseline → model → output, one honest failure, and the screening-only limitation. Add readable captions and app/repository end links.
- [ ] Export exactly 3:00; confirm spoken script stays under 420 words after substitutions. Watch on a phone with sound off for text readability, then with sound on for timing and clean audio. No placeholders in the final video.
