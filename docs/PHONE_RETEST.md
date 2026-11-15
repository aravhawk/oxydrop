# 2-minute phone retest

Purpose: find out whether the v2 capture path works on a **real phone**. So far it has only been checked with a synthetic fake camera in headless Chromium and a simulator. This test checks heart rate, placement, the lock, and the 3-state output. It does **not** test whether the app detects low oxygen, and you must not try to produce low oxygen.

> **Safety.** This project is a prototype and is not medical advice. **Do not hold your breath, exercise, or do anything to lower your oxygen to test the app.** Sit calmly and breathe normally the whole time. Expect "Normal vs your baseline" at rest. If you feel unwell, stop and use a real pulse oximeter or call a clinician.

## What you need
- One phone: **Android with Chrome** (best case: torch and exposure lock) or **iPhone with Safari** (no torch control, so use a bright steady light or the ambient option).
- A reference pulse taken at the same time: a smartwatch, a fingertip pulse oximeter, or a 30 second manual pulse count (count for 30 s and double it).
- A quiet seat. The phone on a table, elbow supported, finger resting on the lens without pressing hard.

## Timeline

| Time | What to do | What you should see |
|---|---|---|
| 0:00 to 0:20 **Setup** | Open https://aravhawk.github.io/oxydrop/ and tap *Measure with my camera*. Tap *Turn on camera* and allow it. Pick the **main (wide) rear lens** (the app avoids ultra-wide and tele). Choose **Flash** (Android) or **Ambient** (iPhone, bright lamp). Cover the lens and light **fully** with a fingertip, light pressure. | The placement indicator turns to "Fingertip detected". The start button enables after about 3 locked windows. If it never does, note the message and go to *If it fails*. |
| 0:20 to 1:20 **Calibration (60 s)** | Tap *Start 60-second calibration*. Sit still, breathe normally, do not talk. Do not move the finger. | Calibration bar fills. The clock reaches 1:00. The first answer arrives about 10 s later because the first window needs 10 s of data. |
| 1:20 to 2:00 **Measurement (about 40 s)** | Keep still. Write down the reference pulse once, about 10 s after calibration ends. For the last 10 s, **lift your finger off the lens** for about 5 s, then put it back. | HR tile near your reference. State **Normal vs your baseline**. With the finger lifted, the state should change to **Unsure, re-measure** and the HR should disappear. |
| 2:00 | Tap **Copy debug summary**, paste it into a note. Tap **Download raw trace (CSV)** and keep the file. | |

The calibration check reports a failed calibration as "Unsure, re-measure" with a reason (pulse found in too few windows, frame rate under 20 fps, too many dropped frames, resting HR outside 51 to 110 bpm, or motion). That is a valid result for this test, so copy the reason.

## What to send back
1. The **Copy debug summary** text.
2. The **raw trace CSV** (about 3,500 lines for a 2-minute run).
3. Phone model, OS and browser version, lens used, Flash or Ambient, and whether the torch actually turned on.
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
| 6 | Resting HR floor | If your resting pulse is below 51 bpm, "Unsure, re-measure" is the expected and correct behaviour (the estimator cannot lock below 51 bpm) |

If the reference pulse is below 51 bpm or above about 110 bpm, record it but treat checks 1 and 3 as not applicable.

## If it fails
- Never reaches "Fingertip detected": try the other lens, switch Flash/Ambient, press a little harder or softer, and try a brighter room. Write down what the placement indicator said.
- Torch does not turn on (iPhone Safari always, some Android browsers): use Ambient with a bright lamp and say so in the report.
- HR is about double the reference: this is the failure v2 is meant to remove. Send the CSV.
- Answers are always "Unsure, re-measure": send the debug summary, which lists the reason.

## Result table template

| Phone / OS / browser | Lens | Flash or ambient | Torch on? | Reference pulse (method) | App HR (before / after finger lift) | Within 5 bpm? | 2x readings? | State at rest | State with finger lifted | fps / dropped frames | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
