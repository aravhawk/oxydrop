End-to-end browser test of the **live camera path** with a synthetic fingertip video (72 bpm pulse), no real camera needed.
```bash
python3 tests/e2e/make_fake_video.py            # writes fake.y4m (needs numpy)
python3 -m http.server 8765 --directory app &
npm i playwright-core
CHROME=/path/to/chromium node tests/e2e/fake_camera.mjs
```
Result when run: signal quality good, calibration passes, heart rate reads 72 bpm, no console errors (about 90 s, real time).
