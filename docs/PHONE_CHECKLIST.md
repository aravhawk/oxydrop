# Per-phone sanity checklist

**These are held-out sanity checks, never a tuning set.** The app is built to work on every phone, and nothing in it (thresholds, gains, exposure, model weights, conformal levels) is fitted to any particular phone. Per-lens probing, adaptive exposure, quality gating and the heart-rate guards are generic rules that run on whatever camera the phone has. The model was trained on six people filmed with a few phones (UW Hoffman 2022) plus other public data sets, so **untested Androids and other phones are a documented limitation**. A failure on one phone is a bug report to be reproduced and fixed with a generic rule that is also checked on the recorded data and the synthetic camera tests. It is never a reason to change a threshold for that phone.

> **Safety.** This is a research prototype and is not medical advice. Do not hold your breath, exercise, or do anything to lower your oxygen. Sit calmly and breathe normally. If you feel unwell, stop and use a real pulse oximeter or call a clinician.

The generic capture protocol, the reference-pulse method and the pass criteria are in [PHONE_RETEST.md](PHONE_RETEST.md). This file adds what to look for on each phone family with the hybrid model (v1 physics chain + residual CNN + conformal three-state output) and what to record.

## What you need
- The phone, the live site (https://aravhawk.github.io/oxydrop/) opened in Safari (iPhone) or Chrome (Android), and a quiet seat. Phone on a table, elbow supported.
- A reference **pulse** at the same moment (fingertip pulse oximeter, smartwatch, or a 30 s manual count doubled).
- A **fingertip pulse oximeter** SpO₂ reading taken at rest just before the test (the known normal reading, typically 95 to 100 %).
- Optional: a second person to read the phone screen aloud while you keep still.

## Steps that apply to every phone

1. Open the page, tick the box, tap *Measure with my camera*, then *Turn on camera*. During the 3 s countdown lay the fingertip **flat across the whole camera plateau** (every lens and the flash), light pressure.
2. Write down which **lens** the app picked (the line "chosen lens" in the debug view, and the probe scores per lens) and whether the **torch** is on ("Camera on, flash on", debug line `torch on/off/unsupported`).
3. Check the sequence **probe -> setup -> calibration -> result**: the setup screen must appear after the probe, the start button must enable after about 3 locked windows, calibration runs 60 s, and the first answer comes about 10 s later.
4. After the first answer, open the debug view and note the `engine` line: it must read `engine oxydrop-hybrid-v1 ... runs in a Web Worker` (or a visible logistic-chain fallback notice on a browser without module workers) and show the latency per 2 s window. Latency must stay below the 2 s hop; a median above 1 s means the phone is too slow for the on-device model, which is a finding to report.
5. Compare the **heart rate** on screen with the reference pulse at the same moment (pass: within 5 bpm, never about double or half).
6. **Low-oxygen state against the known normal reading.** At rest the app must show *Normal vs your baseline* in most answered windows. *Possible low oxygen* in a healthy seated person is a **false alarm**: write down the time, the debug summary and the trace. *Unsure, re-measure* in some windows is acceptable. Nothing here tests whether the app detects a real drop; do not try to create one.
7. Lift the finger for about 5 s: the state must switch to *Unsure, re-measure* and the HR must disappear.
8. Tap **Copy debug summary** and **Download raw trace (CSV)** (the CSV has the columns `z`, `cnn_hr` and `engine` next to the older ones).

The main screen never shows an SpO₂ number. The debug view has one line labelled `EXPERIMENTAL SpO2 from the CNN head (not validated, do not use)`. Do not compare it with the oximeter as a pass or fail criterion.

## iPhone 13 (post-hoc sanity check)
- What happened before: a seated recording showed about 150 bpm, double the real pulse (the second, dicrotic beat counted as a beat).
- Look at: the HR tile for the first minute of answers. Pass: within 5 bpm of the reference, never about 2x. The debug line `hybrid: ... CNN HR a  spectral HR b` shows both estimates. If the screen shows about double, record both numbers and the trace; the window should have been refused by the beat-agreement gate (state *Unsure*).
- Record: lens chosen, torch (Safari has no torch control on most iPhones: the debug line says `unsupported` and the checks list says so), probe scores, HR vs reference, number of *Unsure* windows.

## iPhone 17 (post-hoc sanity check)
- What happened before: lens and torch selection was the open question.
- Look at: the probe table (one score per back lens). The picked lens should be the one the finger is covering with the strongest pulse; the ring next to the lens name turns green when it looks covered. Check the torch line and that the camera did not silently fall back (the debug view prints `device fallback:` when the exact lens could not be opened).
- Record: all lens names and probe scores, picked lens, torch state, whether switching lens by hand in the menu works, HR vs reference.

## iPhone 18 Pro (post-hoc sanity check)
- What happened before: the flash sits far from the lenses, so a flat finger covers the lens but the light reaching it is weak (low AC amplitude).
- Look at: the placement and quality messages ("Weak signal: re-position", signal quality score) and the adaptive-exposure checklist line ("Exposure for weak or diffuse light"). The expected generic behaviour on a weak or low-AC signal is *Unsure, re-measure* with a reason, not a confident wrong answer: a pulse weaker than the model's training range (AC/DC below about 0.24 %) is refused with the text "The pulse signal is weaker than anything this model was trained on" (the trace CSV column `label` reads `low_ac`). If your phone is always refused this way, record it; it is a documented limitation, and the cut stays where it is. Note whether the probe falls back to ambient light or the front camera.
- Record: probe scores per lens, mode chosen (torch, ambient or front), the adaptive-exposure line, the live quality score, HR vs reference when a pulse locks, and the reason text of any *Unsure*.

## Any Android phone (Chrome)
- Check that torch and exposure/white-balance/focus lock are available (the checks list names what is lockable). Phones with several back lenses (wide, ultra-wide, telephoto, macro) differ widely: the app probes up to four back lenses and picks the best pulse, so the record of the probe scores is the key output. Some Androids expose only one camera id or label lenses oddly; copy the full probe table.
- Record: phone model, Android and Chrome version, lens labels as listed in the lens menu, probe scores, picked lens, torch state, `fps` and `gaps` in the debug view (needs at least 20 fps), HR vs reference, model latency per window.
- Low-end phones: if the median latency per window is near or above 1 s, or the debug view shows many dropped-frame gaps, say so. The CNN runs in a Web Worker so it should not delay camera frames; this is exactly what we cannot verify without phones.

## What to send back for every phone
1. The **Copy debug summary** text, and the **raw trace CSV**.
2. Phone model, OS and browser version, lens picked, mode (torch / ambient / front), torch on or off.
3. Reference pulse and how it was measured, and the reference SpO₂ from the fingertip oximeter taken before the test.
4. The state at rest over the first minute of answers (how many *Normal*, *Possible low oxygen*, *Unsure*), and whether HR matched.
5. Anything surprising (stuck screens, a lens that never opened, the page reloading).

## How results are used
Results are reported in the whitepaper as a small post-hoc check on three phones, not as accuracy. Failures become issues with a reproduction (synthetic camera test or recorded trace) and a **generic** fix, re-checked on the existing tests (`npm test`, the browser e2e cases) and on the held-out numbers. No threshold, gain or model setting may be chosen from these phones' recordings.
