# Phone retest

Purpose: find out whether the v2 capture path works on a **real phone**. The timeline below is a plan; **how long it takes has not been measured** (setup, the lens probe and the 60 second calibration come before the measurement part, so expect more than two minutes). So far it has only been checked with a synthetic fake camera in headless Chromium and a simulator. This test checks heart rate, placement, the lock, and the 3-state output. It does **not** test whether the app detects low oxygen, and you must not try to produce low oxygen.

> **Safety.** This project is a prototype and is not medical advice. **Do not hold your breath, exercise, or do anything to lower your oxygen to test the app.** Sit calmly and breathe normally the whole time. Expect "Normal vs your baseline" at rest. If you feel unwell, stop and use a real pulse oximeter or call a clinician.

## What you need
- One phone: **Android with Chrome** (best case: torch and exposure lock) or **iPhone with Safari** (no torch control, so use a bright steady light or the ambient option).
- A reference pulse taken at the same time: a smartwatch, a fingertip pulse oximeter, or a 30 second manual pulse count (count for 30 s and double it).
- A quiet seat. The phone on a table, elbow supported, finger resting on the lens without pressing hard.

## Timeline

| Time | What to do | What you should see |
|---|---|---|
| Setup and lens probe (planned about 0:00 to 0:30) | Open https://aravhawk.github.io/oxydrop/ and tap *Measure with my camera*, then *Turn on camera* and allow it. After a 3 second countdown the app tests every back camera for about 3 seconds each. **Lay your fingertip flat across the whole camera plateau**, covering the lenses and the flash, light pressure, and keep still. You do not pick the lens: the app picks the one with the strongest pulse and names it. | A progress line "Testing lens 2 of 4, keep your finger flat across the camera plateau" with a score per lens. Then the picked lens name, the mode (torch, ambient or front camera + white screen) and the reason. The ring next to the lens name turns green when the picked lens looks covered. The placement indicator turns to "Fingertip detected". About 3 s later the checklist changes from "Hold still: locking exposure, white balance and focus…" to "Locked: …" (after the adaptive exposure step, which is listed in the checklist). The start button enables after about 3 locked windows. If it never does, note the message and go to *If it fails*. |
| 0:20 to 1:20 **Calibration (60 s)** | Tap *Start 60-second calibration*. Sit still, breathe normally, do not talk. Do not move the finger. | Calibration bar fills. The clock reaches 1:00. The first answer arrives about 10 s later because the first window needs 10 s of data. |
| 1:20 to 2:00 **Measurement (about 40 s)** | Keep still. Write down the reference pulse once, about 10 s after calibration ends. For the last 10 s, **lift your finger off the lens** for about 5 s, then put it back. | HR tile near your reference. State **Normal vs your baseline**. With the finger lifted, the state should change to **Unsure, re-measure** and the HR should disappear. Also write down what the placement indicator says while the finger is lifted (a warm lit surface can read "Fingertip detected"; the pulse lock should still refuse it). |
| 2:00 | Tap **Copy debug summary**, paste it into a note. Tap **Download raw trace (CSV)** and keep the file. | |

The calibration check reports a failed calibration as "Unsure, re-measure" with a reason (pulse found in too few windows, frame rate under 20 fps, too many dropped frames, a camera stall longer than 1.5 s, resting HR outside 51 to 110 bpm, or motion). That is a valid result for this test, so copy the reason.

## What to send back
1. The **Copy debug summary** text.
2. The **raw trace CSV** (about 3,500 lines for a 2-minute run).
3. Phone model, OS and browser version, the lens the app picked, the mode (torch, ambient or front + white screen), and whether the torch actually turned on. The **lens probe scores per lens** are in the debug summary under "lens probe scores"; copy them in full.
4. The reference pulse and how you measured it.
5. Screenshot of the final screen (optional).

## Pass and fail criteria

| # | Check | Pass |
|---|---|---|
| 1 | Heart rate | The HR tile is within **5 bpm** of the reference pulse (taken at the same moment) |
| 2 | No doubling | HR never shows about **2 times** the reference (for example about 150 when the reference is about 75) in any answered window |
| 3 | State at rest | The state is **Normal vs your baseline** in most answered windows at rest. A few "Unsure, re-measure" windows are acceptable. **"Possible low oxygen: re-check with a pulse oximeter" in a healthy seated person is a failure** (note the time) |
| 4 | Finger lifted | The state becomes **Unsure, re-measure** within about 10 s of lifting the finger, with no confident HR |
| 5 | Camera facts | Frame rate at least 20 fps and few dropped-frame warnings in the debug summary |
| 6 | Camera stall | Optional, 10 s: at any time after calibration, switch to another app or lock the screen for about 3 s, then come back. Within about 2 s of the stall the state must read **Unsure, re-measure** ("The camera stopped delivering frames") with no HR, and it must stay unsure for up to about 30 s after the camera resumes, then return to Normal. The 60-second calibration is rejected if the camera stalls for more than 1.5 s inside it |
| 7 | Resting HR floor | If your resting pulse is below 51 bpm, "Unsure, re-measure" is the expected and correct behaviour (the estimator cannot lock below 51 bpm) |

If the reference pulse is below 51 bpm or above about 110 bpm, record it but treat checks 1 and 3 as not applicable.

## If it fails
- Never reaches "Fingertip detected": try the other lens, switch Flash/Ambient, press a little harder or softer, and try a brighter room. Write down what the placement indicator said.
- Torch does not turn on (iPhone Safari always, some Android browsers): use Ambient with a bright lamp and say so in the report.
- HR is about double the reference: this is the failure v2 is meant to remove. Send the CSV.
- Answers are always "Unsure, re-measure": send the debug summary, which lists the reason.

## Fallback modes to try (one extra minute each, optional)
After the normal run, use the *Light source* buttons in setup to try each mode on the same phone and note which gives a green quality meter and "Fingertip detected":
- **Flash on** (torch, best back lens). Try the lens menu override on a second lens too, if you have one, and note which gives the better quality meter.
- **Ambient light** (no flash, bright lamp or window).
- **Front camera + white screen**: cover the front camera with the fingertip, screen brightness at maximum. This is also what the app picks automatically if every back lens scores below 35.
Also check that a **Weak signal** message appears (and the answer becomes "Unsure, re-measure" with no heart rate) if you only half cover the lens for about 5 seconds.

## Result table template

| Phone / OS / browser | Lens picked by app | Probe scores per lens | Mode picked (torch / ambient / front) | Quality meter (good / weak) | Flash or ambient | Torch on? | Reference pulse (method) | App HR (before / after finger lift) | Within 5 bpm? | 2x readings? | State at rest | State with finger lifted | fps / dropped frames | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
