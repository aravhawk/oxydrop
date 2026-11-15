"""Write synthetic y4m "fingertip" videos for the fake-camera e2e test (pure Python, no numpy).
usage: python3 tests/e2e/make_fake_video.py good fake_good.y4m   # 72 bpm pulse on a red-dominant frame (finger over lens, flash on)
       python3 tests/e2e/make_fake_video.py bad  fake_bad.y4m    # grey frame with random brightness noise, no pulse, no finger
"""
import math, random, sys
kind = sys.argv[1] if len(sys.argv) > 1 else 'good'
path = sys.argv[2] if len(sys.argv) > 2 else f'fake_{kind}.y4m'
W, H, FPS, N = 160, 120, 30, 30 * 95
rnd = random.Random(7)
with open(path, 'wb') as out:
    out.write(f'YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n'.encode())
    for i in range(N):
        t = i / FPS
        if kind == 'good':
            ph = (1.2 * t) % 1.0
            pulse = math.exp(-((ph - 0.15) / 0.08) ** 2) + 0.35 * math.exp(-((ph - 0.45) / 0.1) ** 2)
            y, u, v = 110 + 14 * (pulse - 0.3), 100, 190      # -> mean RGB about 197 / 76 / 60
        else:
            y, u, v = 110 + rnd.gauss(0, 8), 128, 128         # grey: R = G = B, so not a fingertip
        y = max(0, min(255, int(round(y))))
        out.write(b'FRAME\n'); out.write(bytes([y]) * (W * H)); out.write(bytes([u]) * (W * H // 4)); out.write(bytes([v]) * (W * H // 4))
