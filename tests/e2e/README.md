End-to-end browser test of the **live camera path** with synthetic fingertip videos (no real camera needed), plus the recorded-demo (replay) path.

```bash
python3 tests/e2e/make_fake_video.py good fake_good.y4m     # 72 bpm pulse on a red-dominant frame (pure Python)
python3 tests/e2e/make_fake_video.py bad  fake_bad.y4m      # grey noise, no finger
npm i playwright-core                                       # or point PLAYWRIGHT_CORE at an existing playwright-core/index.mjs
CHROME=/path/to/chromium node tests/e2e/fake_camera.mjs     # serves app/ itself on a random local port
```
Set `LIVE_URL=https://aravhawk.github.io/oxydrop/index.html` to run the cases against the deployed site instead of the local copy (the `good` case passed against the live site after the v2 deploy).

Cases (`CASES=good,bad,replay`, default all):
* **good**: lens select, ambient/flash toggle, placement reaches "Fingertip detected", Start enables after 3 locked windows, 60 s calibration passes, heart rate 72 +-3, state "Normal vs your baseline", debug CSV download and copy work, zero console errors.
* **bad**: the placement check refuses a grey frame, "Start anyway" appears after 20 s, the result is "Unsure, re-measure" and the heart-rate tile shows an en dash.
* **replay**: recorded Hoffman demo through the same decision function; participant 4 (resting 43 bpm, below the 51 bpm band edge) must be "unsure" throughout.

Takes about 6 minutes in real time (the camera runs at real speed). If Chromium crashes at start with "disk I/O error", `/tmp` is full: set `TMPDIR` to a directory on disk.
