# OxyDrop: a phone-camera check for a drop in blood oxygen

**Track:** Health
**Team:** Arav Jain, [TEAMMATE 2 NAME], [TEAMMATE 3 NAME]  *(Arav: fill in or delete before submitting; see CHECKLIST.md)*
**Live app:** https://aravhawk.github.io/oxydrop/ · **Code:** https://github.com/aravhawk/oxydrop

> **This project is a prototype and is not medical advice.**

## The problem
Low blood oxygen (SpO₂ below 90%) is easy to miss because it often feels like nothing, and the people most exposed to it are often far from a pulse oximeter: a hiker or skier at altitude, someone recovering at home, a parent at night. Our design scenario is **a hiker resting in a lodge without a pulse oximeter**, who wants to know "should I re-check with a real oximeter, or go down?" A phone is already in their pocket. This is a scenario, not evidence: the app has not been tested at altitude, on a real phone, or with any real hiker.

## What we built
A web app (works in a phone browser, nothing to install, video never leaves the phone). The user sits at rest and covers the rear camera and flash with a fingertip. After a 60-second calibration of *their own* normal (the first answer arrives at 70 seconds, once the first 10-second window is complete), every 2 seconds it says **normal**, **possible low-oxygen episode**, or **unsure, re-measure**, and shows heart rate, signal quality and a motion check. It does not output an SpO₂ number. Because it compares you with your own first minute, it cannot know whether you were already low at the start (oxygen status at start unknown). A "recorded demo" mode replays real study recordings so anyone can try it without a camera.

## Prior work, and what we added
Phone-camera low-oxygen detection is **not new**. Hoffman et al. (2022) already demonstrated fingertip camera and flash hypoxemia screening with induced desaturation and held-out testing (81% sensitivity, 79% specificity below 90% SpO₂, 5.00-point SpO₂ MAE; their protocol differs from ours, so this is not a direct comparison). MTHS (Samavati et al., 2022), Sun et al. (2021) and Tian et al. (2022) studied smartphone and camera vital signs and oximetry with ratios, calibration and heart-rate-adaptive filtering. What we added is engineering and reproducibility work: a small personalised classifier, a causal JavaScript port that runs offline in the phone browser (numerically equal to the Python version on all 5,646 test windows, max difference 2.09×10⁻¹³ for v2 and 3.18×10⁻¹³ for v1, which shows implementation agreement and not accuracy), browser capture and replay, abstention, and a confound audit. We do not claim a new sensing principle, a first detector, or superiority over earlier work.

## How it works
**Data.** UW Hoffman 2022 (6 people, 12 hands, MIT licence) is the only open phone-camera dataset with real SpO₂ below 90% that we found. We looked at other candidates and found them normoxic, gated, or without an optical signal, but that was not an exhaustive, dated search. With so little hypoxemia data, a big neural network overfits, so we built a **chain of small stages** where the data-rich parts of the problem do the heavy lifting.
**Model.** Mean colour per camera frame → causal band-pass filter → beat detection (heart rate, variability, breathing rate, pulse strength) and light-absorption features → subtract the person's own first-minute median → two logistic regressions averaged (62 numbers in total, including scaling statistics) → decision, or "unsure" when the probability is near 0.5. The second regression is not heart-rate only: it takes 19 features including ratio-of-ratios, slope and DC-absorption features (four ratio features have nonzero coefficients). Heart-rate extraction was checked on a separate smartphone-PPG dataset with ECG (BUT PPG). An earlier altitude-chamber zero-shot check (AUROC 0.88) has no runnable script here and is not relied on. These datasets guided and validated the design; only the final 62 numbers are fit, on Hoffman.
**Testing.** Leave-one-person-out: the model is always scored on a person it never saw (both hands held out), scaling is fitted on the other five people, and calibration uses only the person's first 60 s. Not all settings came from the other five people: the v2 regularisation strengths and the abstain band are hardcoded values with no independent provenance, and many variants were compared on the same six people, so there is some test-set reuse. We audited for leakage and found and fixed one bug in our own code (full audit in the repo; the review of it is in `docs/REVIEW_RESPONSE.md`).

## Results
Baseline = a clock-only model that knows only how many seconds have passed (the study lowers oxygen steadily, so time alone predicts the label). Always answering "low oxygen" scores 57.48%. The table below is the **v1 research chain**, re-derived from saved out-of-fold scores (arithmetic check, not a rerun of the nested training). The shipped v2 is in the README: 94.68% on the 68% of windows it answers (v1 gets 94.60% on those same windows), only 71.79% of hypoxic windows receive an alert, and the person with a 43 bpm pulse gets no answer in 947 windows.

| | Accuracy | Missed hypoxemia (false negatives) | False alarms (false positives) | Accuracy with time controlled |
|---|---|---|---|---|
| Clock-only baseline | 83.5% | 13.6% | 20.3% | **51.9%** |
| **OxyDrop v1, answers every window** | **87.0%** (95% CI 83–91) | **12.1%** | **14.2%** | **79.5%** |
| **OxyDrop v1, may say "unsure"** (answers 85% of windows; shipped v2 answers 68%, see above) | **92.5%** | **6.6%** | **8.7%** | 85.2% |

- The raw gain over the clock (+3.5 points) is not statistically resolved with 6 people (paired person-bootstrap CI −6.8 to +16.7); the time-controlled score (79.5% vs 51.9%) and the abstention result are the solid claims. 90% accuracy on every window was **not** reached.
- **Where it fails.** Person 4 (20-year-old man, resting heart rate 43 bpm) is the worst case at 77.5%: on one hand the model misses hypoxemia for 107 windows, on the other it raises 79 false alarms between 212 and 888 s. We have not established why; one untested hypothesis is that his very low resting heart rate (43 bpm) makes the heart-rate-rise signal unstable. Exercise is the biggest risk: the same rule flags **63.4%** of normoxic exercise windows as low-oxygen (68.4% counting only the 13 people with a usable heart rate), so the app requires seated rest and pauses when the phone moves.
- **Groups.** The 6 participants are adults aged 20–34, 3 women, 3 men, 1 Black participant (accuracy 88.6%, in line with the others, range 77.5–91.7%). One person per group cannot show fairness; the model is untested on children, older adults and darker skin tones beyond that one person.
- **Not tested:** any phone other than the study's Nexus 6P. iPhone Safari cannot control the flash, so the live camera path there is unvalidated.

## What we'd do next
1. Collect consented phone-camera data from a broader, deliberately diverse group in a safe, supervised hypoxia protocol (e.g. altitude or clinical breath-holding studies), including phones other than the study phone, to test fairness and calibrate per phone.
2. Add a motion/exertion detector trained on exercise data so the app refuses to answer after activity, rather than relying on the user to rest.

## Sources
- Hoffman et al., "Smartphone Camera Oximetry in an Induced Hypoxemia Study", npj Digital Medicine 2022; data https://github.com/ubicomplab/oximetry-phone-cam-data (MIT)
- BUT PPG v2.0, PhysioNet (CC BY 4.0), for heart-rate validation: https://physionet.org/content/butppg/2.0.0/
- OpenOximetry (PhysioNet, credentialed; derived aggregates only, never redistributed) for validation, fairness and later pretraining; 20-person altitude-chamber pilot (figshare, 2025; earlier check not relied on); moshekam PPG-during-exercise data https://github.com/moshekam (MIT); MTHS and HealthRing (experiments with no gain). Full list and licences: https://github.com/aravhawk/oxydrop/blob/main/DATA_LICENSES.md
- Libraries: numpy, scipy, scikit-learn, matplotlib, Node.js.
- AI tools: Claude (Anthropic) was used for code, analysis and drafting. The team reviewed the work and can explain every part.
