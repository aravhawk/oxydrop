# OxyDrop: three-minute demo

**Runtime:** 3:00. Only the quoted narration is spoken; fewer than 420 words, including the fallback line. Leave short pauses for the screen. Every metric value below is copied from `results/ml/` and `results/photon/TABLE.csv` (see the results card) and must be re-checked against that folder if the evaluation is re-run before recording.

**Recording truth check:** This script describes the **round-3 build**: the app's default engine is **PhotonDistill v2**, one small trained network per seed (a multilayer perceptron on 39 baseline-relative window features, three seeds of widths 32, 32 and 64) whose own output and abstain head are shown as they are. The previous hybrid runs behind `?engine=hybrid` and the logistic chain behind `?engine=logistic`. Before recording, open the deployed site, take one measurement and confirm the debug line names `oxydrop-photon-distill2-v1` (model file `data/distill.json`). If it names `oxydrop-photon-distill-v1`, the hybrid or the logistic chain, the site has not been redeployed from the round-3 build (or an `?engine=` parameter is set): fix that first, or relabel the shot. The evaluation result is that PhotonDistill v2 is **on par with** the logistic chain on clean light (1.2 points lower, interval includes zero; it does not beat it; a GPU rerun of the same code scored 0.761, which we report as seed variance) and holds up better only under a **simulated** distant flash. Do not present it as more accurate. Do not change the evaluation claim unless new held-out evidence supports it.

## Shot list and narration

| Video time | Shot / on-screen content | Spoken narration |
|---|---|---|
| 0:00–0:20 | Presenter, then phone beside an empty space labelled “No pulse oximeter.” Caption: “For a hiker resting at a lodge: design scenario, not tested at altitude.” | “Imagine a hiker resting at a lodge, wondering whether to check their blood oxygen, but without a pulse oximeter. OxyDrop explores whether the phone already in their pocket can help flag a reason to re-check. It is a screening prototype, not a medical device.” |
| 0:20–0:35 | Overhead view: seated user, supported hand, fingertip over rear camera and flash. Cut to actual phone screen: camera permission, selected lens, placement and quality indicators. Label “Live phone capture, PhotonDistill v2 build.” | “I sit still, breathe normally, and rest my finger lightly over the camera and flash. The camera sees tiny colour changes as blood pulses through my fingertip. The app checks whether the signal is usable.” |
| 0:35–1:00 | Tap calibration. Show start and end of the same take, with an explicit “60-second calibration: middle shortened” caption and source timer. Keep waveform visible. Then allow the first measurement window to complete. | “First, it learns my own resting baseline for one minute. We have shortened that wait here. In this build, the first result needs another ten seconds, plus the camera setup time. A normal result cannot tell whether my oxygen was already low at the start.” |
| 1:00–1:25 | Show actual heart-rate tile and actual state, large enough to read. Beside it show the three exact UI labels: “Normal vs your baseline”; “Possible low oxygen: re-check with a pulse oximeter”; “Unsure, re-measure.” Label these “Available states,” not live outcomes. | “Here is the heart-rate estimate and the result. The three possible states are normal versus my baseline, possible low oxygen requiring a pulse-oximeter re-check, or unsure: re-measure. There is no oxygen percentage. The model on this phone is one small trained neural network, running in the browser.” |
| 1:25–2:05 | Results card: “Recorded study data; six people; both hands held out together.” Show the PhotonDistill v2 and logistic chain rows (clean light and simulated distant flash X1) from the results card below, then `docs/figures/photon_stress_curve.png`. Caption “Clean light: on par with the chain, not a win. Distant flash: a simulation.” | “We test on people the model did not train on, holding out both hands together. On a test that balances normal and low-oxygen examples across elapsed time, our logistic chain scores 79.6 percent in normal light. The trained model in this app scores 78.4: on par with it, not a win. A rerun on another computer scored 76.1, which is normal variation between training runs. It holds up when we simulate a flash farther from the lens: 68.2 percent where the chain falls to 50.8, chance level. That is a simulation, a simple colour model does nearly as well, and six people is a small test.” |
| 2:05–2:25 | Lift finger; show actual rejection reason and cleared heart-rate tile. Caption “Signal failure: an unanswered reading is not a successful detection.” | “Now I lift my finger. The signal is lost, so it should say unsure and hide the heart rate. That is an honest failure: no usable reading. An unsure result does not mean normal oxygen.” |
| 2:25–2:50 | `docs/figures/photon_architecture.png`, or a simple graphic: camera colour traces → 39 features relative to my first minute → “PhotonDistill v2: one trained network per seed, three seeds” → screening state / unsure. Caption “Trained model in the browser. On par with the chain on clean light.” | “Machine learning is core: one small network learned from two simple teachers, on normal recordings and on physics-simulated distant-flash copies of them. It runs on the phone and says unsure when it cannot read the signal. That does not prove reliable oxygen screening, and in normal light it only matches the simpler chain.” |
| 2:50–3:00 | End card: app URL, repository URL, “Prototype • Rest only • Re-check with a pulse oximeter • Not a medical device • Model weights: non-commercial research and education use.” | “The next step is broader real-phone testing. OxyDrop is a research prototype and a screening aid, never a replacement for a pulse oximeter or medical care.” |

**If the phone does not produce an answer:** keep the real attempt and its reason on screen. Replace “Here is the heart-rate estimate and the result” with “This phone returned unsure; here is why.” Use the built-in recorded demo to explain the remaining flow, permanently labelled **“Recorded study replay, not live phone capture.”** If finger removal does not trigger rejection, show the observed behavior and say “This build failed to reject the lost signal; that needs fixing.” Never manufacture a passing take or imply that replay proves live performance.

### Results card

Source: `results/photon/TABLE.csv` (rounds 1 to 3, the photon harness: `python -m oxydrop_ml.photon.evalkit`; verdict `results/photon/WINNER3.md`, `WINNER3.csv`) and, for the previous engine, `results/ml/baselines_P1.csv`, `results/ml/hybrid_p1_final.csv`, `results/ml/hybrid_p1_final_paired.csv`, `results/ml/cnn_vs_baselines_paired.csv` (P1, seed `-1` = three-seed ensemble). Hoffman data, six people, leave-one-person-out, 5,644 windows. Values are time-matched balanced accuracy at full coverage with 95% person-bootstrap intervals. The photon harness re-runs the bootstrap, so the same chain reads [0.636, 0.857] there and [0.635, 0.856] in `results/ml`.

| Model / measure | Value to display | What it is |
|---|---|---|
| **PhotonDistill v2 (what the app runs), clean light** | 0.784 [0.596, 0.834] | one trained network per seed, three seeds |
| Logistic chain v1 (best baseline), clean light, photon harness | 0.796 [0.636, 0.857] | the same held-out people and windows |
| PhotonDistill v2 minus chain, clean light, paired | -1.2 [-9.1, +1.2] percentage points, 4 of 6 people better | **on par**: the interval contains zero; not a win |
| PhotonDistill v2 minus round-2 PhotonDistill (0.758), clean light, paired | +2.6 [-1.0, +4.0] percentage points | the round-3 replacement rule (Amendment 8) |
| PhotonDistill v2, simulated distant flash X1 (out of family) | 0.682 [0.547, 0.748] against 0.508 for the chain | simulation, not a phone measurement |
| PhotonDistill v2 minus chain at X1, paired | +17.4 [+4.4, +28.4] percentage points, 6 of 6 people | win under the simulation only |
| DC-colour control (three numbers, logistic regression) at X1 | 0.646 [0.511, 0.735] | nearly as robust: a tie with the chain by its interval |
| Logistic chain v1 (best baseline), `results/ml` | 0.796 [0.635, 0.856] | the shipped chain |
| Hybrid, nested (previous engine) | 0.779 [0.638, 0.831] | unbiased estimate of the whole selection procedure |
| Deployed pipeline R (previous engine, `?engine=hybrid`) | 0.809 [0.644, 0.863] | **optimistic**: chosen after its outer scores were seen |
| CNN alone, three-seed ensemble | 0.754 | the CNN without the chain |
| Clock-only baseline | 0.495 [0.480, 0.514] | near chance by construction: this metric is blind to a clock |
| Hybrid (nested) minus chain, paired person-bootstrap 95% interval | -1.7 [-10.3, +6.9] percentage points | tie: the interval contains zero |
| CNN ensemble minus chain, paired person-bootstrap 95% interval | -4.2 [-13.1, +0.5] percentage points | tie: the interval contains zero |

Disclose on screen or in the description: all round-3 models were designed after the round-1 and round-2 results on the same six people were seen (Amendment 8); PhotonDistill v2's null and stress files arrived one minute after the stated 20:15 CT cutoff and were accepted by the planner's lock with that disclosure; the gradient-boosted tree teacher was never chosen and no physics teacher was used; a GPU rerun of PhotonDistill v2 scored 0.761 [0.561, 0.830] against the locked 0.784 and is reported as seed variance. One of the six people is dark-skinned, so there is no skin-tone claim. The pitch is in [PITCH.md](PITCH.md).

**Every candidate of the three rounds** (from `results/photon/TABLE.csv` and `WINNER3.csv`; for the editor, not for the narration):

| Round | Model | Clean light (L0) TM [95% CI] | Paired vs chain [95% CI] | People better | Null mean | X1 paired vs chain |
|---|---|---|---|---|---|---|
| Baseline | Logistic chain v1 | 0.796 [0.636, 0.857] | reference | | | |
| Round 3 | PhotonDistill v2 (**shipped**) | 0.784 [0.596, 0.834] | -0.012 [-0.091, 0.012] on par | 4/6 | 0.504 | +0.174 [0.044, 0.284] win |
| Round 3 | MultiDA | 0.744 [0.553, 0.807] | -0.053 [-0.142, -0.010] loss | 3/6 | 0.522 | +0.170 [0.027, 0.328] win |
| Round 3 | PhotonNet-XL | 0.660 [0.545, 0.762] | -0.136 [-0.210, -0.033] loss | 2/6 | 0.412 | +0.184 [0.072, 0.290] win |
| Round 2 | PhotonDistill (round-2 winner, replaced) | 0.758 [0.603, 0.813] | -0.038 [-0.103, -0.001] loss | 2/6 | 0.492 | +0.169 [0.058, 0.285] win |
| Round 2 | OxySeq | 0.708 [0.452, 0.790] | -0.088 [-0.221, -0.023] loss | 2/6 | 0.475 | +0.001 [-0.047, 0.059] tie |
| Round 2 | PhotonNet | 0.668 [0.508, 0.785] | -0.128 [-0.208, -0.047] loss | 2/6 | 0.618 (fails) | +0.165 [0.020, 0.316] win |
| Round 2 | OxySSL | 0.667 [0.498, 0.741] | -0.130 [-0.274, -0.044] loss | 1/6 | 0.643 (fails) | +0.178 [0.043, 0.214] win |
| Round 1 | PhotonTwin gate (hand-coded, ruled out) | 0.796 [0.636, 0.857] | identical to the chain at L0 |  | 0.584 (fails) | +0.007 [0.000, 0.026] tie |
| Round 1 | OxyTrack | 0.809 [0.656, 0.893] | +0.013 [-0.052, 0.092] tie | 3/6 | n/a | +0.044 [-0.001, 0.104] tie |
| Round 1 | PhotonTwin-Inverse | 0.752 [0.577, 0.810] | -0.044 [-0.127, -0.000] loss | 2/6 | 0.406 | +0.107 [0.023, 0.213] win |
| Round 1 | FlashGap | 0.740 [0.607, 0.831] | -0.057 [-0.127, 0.021] tie | 1/4 | 0.532 | +0.121 [-0.000, 0.240] tie |
| Round 3 | ens3 (stack of three models; landed after the lock, no null controls) | 0.789 [0.621, 0.842] | -0.007 [-0.092, 0.043] | 4/6 | none | candidate only |
| Round 3 | ssl3 | no score file | | | | |

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
- [ ] Keep the debug line naming `oxydrop-photon-distill2-v1` visible at least once in the take, and state that the trained model is on par with the chain on clean light (not a win). Save debug summary, raw trace and deployment identifier privately with the take.

**Results and final export**

- [ ] Use the results card values exactly as written, from one frozen evaluation commit. Show the PhotonDistill v2 and chain rows (clean and X1), metric, full coverage, six held-out people and paired uncertainty. Keep the clean-light "on par" statement on screen and label X1 a simulation; do not claim “beats clock” (chance level by construction).
- [ ] Show input → features/baseline → model → output, one honest failure, and the screening-only limitation. Add readable captions and app/repository end links.
- [ ] Export exactly 3:00; confirm spoken script stays under 420 words after any edits. Watch on a phone with sound off for text readability, then with sound on for timing and clean audio. No placeholders in the final video. Do not show an SpO2 percentage or any medical claim.
