# OxyDrop: three-minute demo

**Runtime:** 3:00. Only the quoted narration is spoken; fewer than 420 words, including the fallback line. Leave short pauses for the screen. Every metric value below is copied from `results/ml/` (see the results card) and must be re-checked against that folder if the evaluation is re-run before recording.

**Recording truth check:** This script describes the **hybrid build** of PR #6: the app's main on-device path is the hybrid (physiological logistic chain plus a residual 1D CNN, run in a Web Worker); the plain logistic chain runs only behind `?engine=logistic` and as the automatic fallback. Before recording, open the deployed site, take one measurement and confirm the debug line reads `engine oxydrop-hybrid-v1` and that the service worker cache is `oxydrop-v3.2-hybrid`. If the debug line shows the logistic chain, the site has not been redeployed from the merged PR (or `?engine=logistic` is set): fix that first, or relabel the shot as the logistic-chain build. The evaluation result is that the pre-registered primary metric was **not beaten**: the hybrid ties the best baseline, the logistic chain. Do not present the CNN as an improvement. Do not change the evaluation claim unless new held-out evidence supports it.

## Shot list and narration

| Video time | Shot / on-screen content | Spoken narration |
|---|---|---|
| 0:00–0:20 | Presenter, then phone beside an empty space labelled “No pulse oximeter.” Caption: “For a hiker resting at a lodge: design scenario, not tested at altitude.” | “Imagine a hiker resting at a lodge, wondering whether to check their blood oxygen, but without a pulse oximeter. OxyDrop explores whether the phone already in their pocket can help flag a reason to re-check. It is a screening prototype, not a medical device.” |
| 0:20–0:35 | Overhead view: seated user, supported hand, fingertip over rear camera and flash. Cut to actual phone screen: camera permission, selected lens, placement and quality indicators. Label “Live phone capture, hybrid CNN build.” | “I sit still, breathe normally, and rest my finger lightly over the camera and flash. The camera sees tiny colour changes as blood pulses through my fingertip. The app checks whether the signal is usable.” |
| 0:35–1:00 | Tap calibration. Show start and end of the same take, with an explicit “60-second calibration: middle shortened” caption and source timer. Keep waveform visible. Then allow the first measurement window to complete. | “First, it learns my own resting baseline for one minute. We have shortened that wait here. In this build, the first result needs another ten seconds, plus the camera setup time. A normal result cannot tell whether my oxygen was already low at the start.” |
| 1:00–1:25 | Show actual heart-rate tile and actual state, large enough to read. Beside it show the three exact UI labels: “Normal vs your baseline”; “Possible low oxygen: re-check with a pulse oximeter”; “Unsure, re-measure.” Label these “Available states,” not live outcomes. | “Here is the heart-rate estimate and the result. The three possible states are normal versus my baseline, possible low oxygen requiring a pulse-oximeter re-check, or unsure: re-measure. There is no oxygen percentage. The model on this phone is a hybrid: a physiological chain plus a small neural network, running in the browser.” |
| 1:25–2:05 | Results card: “Recorded study data; six people; both hands held out together.” Show the hybrid, logistic chain and clock rows with metric and uncertainty from the results card below. Caption “Full coverage: every test window counted. Result: a tie, not a win.” | “We test on people the model did not train on, holding out both hands together. On a test that balances normal and low-oxygen examples across elapsed time, our earlier logistic chain scores 79.6 percent. The hybrid scores 77.9 percent in the unbiased estimate, and 80.9 for the shipped version, which is optimistic. Those are ties. We did not beat the chain. Six people is a small test.” |
| 2:05–2:25 | Lift finger; show actual rejection reason and cleared heart-rate tile. Caption “Signal failure: an unanswered reading is not a successful detection.” | “Now I lift my finger. The signal is lost, so it should say unsure and hide the heart rate. That is an honest failure: no usable reading. An unsure result does not mean normal oxygen.” |
| 2:25–2:50 | Simple graphic: camera colour traces → red/green light-change features + personal baseline → “hybrid: chain score + 1D CNN correction, 167,588 parameters per seed, three seeds” → heart rate / screening state / unsure. Caption “Hybrid model in the browser. No accuracy win over the chain.” | “Machine learning is core because a small neural network learns pulse patterns and corrects the physiological chain, relative to my own baseline. It runs on the phone. That design alone does not prove reliable oxygen screening, and in our test it only tied the simpler chain.” |
| 2:50–3:00 | End card: app URL, repository URL, “Prototype • Rest only • Re-check with a pulse oximeter • Not a medical device • Model weights: non-commercial research and education use.” | “The next step is broader real-phone testing. OxyDrop is a research prototype and a screening aid, never a replacement for a pulse oximeter or medical care.” |

**If the phone does not produce an answer:** keep the real attempt and its reason on screen. Replace “Here is the heart-rate estimate and the result” with “This phone returned unsure; here is why.” Use the built-in recorded demo to explain the remaining flow, permanently labelled **“Recorded study replay, not live phone capture.”** If finger removal does not trigger rejection, show the observed behavior and say “This build failed to reject the lost signal; that needs fixing.” Never manufacture a passing take or imply that replay proves live performance.

### Results card

Source: `results/ml/baselines_P1.csv`, `results/ml/hybrid_p1_final.csv`, `results/ml/hybrid_p1_final_paired.csv`, `results/ml/cnn_vs_baselines_paired.csv` (P1, seed `-1` = three-seed ensemble). Hoffman data, six people, leave-one-person-out, 5,644 windows. Values are time-matched balanced accuracy at full coverage with 95% person-bootstrap intervals.

| Model / measure | Value to display | What it is |
|---|---|---|
| Logistic chain v1 (best baseline) | 0.796 [0.635, 0.856] | the shipped chain, same held-out people and windows |
| Hybrid, nested | 0.779 [0.638, 0.831] | unbiased estimate of the whole selection procedure |
| Deployed pipeline R (what the app runs) | 0.809 [0.644, 0.863] | **optimistic**: chosen after its outer scores were seen |
| CNN alone, three-seed ensemble | 0.754 | the CNN without the chain |
| Clock-only baseline | 0.495 [0.480, 0.514] | near chance by construction: this metric is blind to a clock |
| Hybrid (nested) minus chain, paired person-bootstrap 95% interval | -1.7 [-10.3, +6.9] percentage points | tie: the interval contains zero |
| CNN ensemble minus chain, paired person-bootstrap 95% interval | -4.2 [-13.1, +0.5] percentage points | tie: the interval contains zero |

Do **not** claim the model "beats the clock": the clock row is chance level on this metric by construction, so the comparison is not informative. A chain refit with the same levers and no CNN scores 0.816 [0.674, 0.869], so the simpler model has the higher point estimates. State the tie on screen.

“Time-matched balanced accuracy” means balancing normal and low-oxygen cases across elapsed time, so a clock cannot win just by knowing when the study lowered oxygen. Use full coverage here; do not substitute accuracy on answered readings. If showing selective results elsewhere, display coverage and the share of **all** low-oxygen windows receiving an alert. An abstention is not a detection. These are recorded-data engine results, not live-phone clinical performance.

**Evidence for the editor** (all paths exist on the merged tree): [README](../../README.md), [MODEL.md](../../MODEL.md), [WHITEPAPER.md](../../WHITEPAPER.md), [CORE_RESULTS.md](../../results/ml/CORE_RESULTS.md), [paired comparison CSV](../../results/ml/cnn_vs_baselines_paired.csv), [CNN architecture (MODEL_SPEC)](../../python/oxydrop_ml/MODEL_SPEC.md), [pre-registration](../../python/oxydrop_ml/PREREGISTRATION.md), [per-phone checklist](../PHONE_CHECKLIST.md), [phone retest](../PHONE_RETEST.md), [older chain results](../../results/TABLES.md), [data licences](../../DATA_LICENSES.md), and [PR #3 REVIEW.md](https://github.com/aravhawk/oxydrop/blob/7193dcbe74f05c3fb682f85141d24b33ed4c15ed/REVIEW.md). The pre-registration fixes the evaluation: six Hoffman people, both hands of a person held out together, windows after the first 60 s. Do not mix the report's mean-of-seeds CNN value (0.739) with ensemble differences (seed `-1`).

---

## Recording checklist: one-page companion

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
- [ ] Keep the debug line `engine oxydrop-hybrid-v1` visible at least once in the take, and state that the result is a tie with the chain. Save debug summary, raw trace and deployment identifier privately with the take.

**Results and final export**

- [ ] Use the results card values exactly as written, from one frozen evaluation commit. Show the chain, hybrid and clock rows, metric, full coverage, six held-out people and paired uncertainty. Keep the tie with the chain on screen; do not claim “beats clock” (chance level by construction).
- [ ] Show input → features/baseline → model → output, one honest failure, and the screening-only limitation. Add readable captions and app/repository end links.
- [ ] Export exactly 3:00; confirm spoken script stays under 420 words after any edits. Watch on a phone with sound off for text readability, then with sound on for timing and clean audio. No placeholders in the final video. Do not show an SpO2 percentage or any medical claim.
