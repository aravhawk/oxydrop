// Rear camera + torch capture. Produces mean R,G,B per frame, resampled to a fixed 30 Hz grid for the chain.
export class CameraSource {
  constructor(onSample, onInfo) {
    this.onSample = onSample; this.onInfo = onInfo || (() => {});
    this.info = { torch: false, exposureLocked: false, fps: 0, gaps: 0, maxGap: 0 };
    this.prev = null; this.nextT = null; this.frames = 0; this.t0 = 0; this.stopped = false;
  }
  async start(video) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } },
    });
    this.stream = stream; this.video = video; video.srcObject = stream; video.muted = true; video.playsInline = true;
    await video.play();
    this.track = stream.getVideoTracks()[0];
    const caps = this.track.getCapabilities ? this.track.getCapabilities() : {};
    if (caps.torch) {
      try { await this.track.applyConstraints({ advanced: [{ torch: true }] }); this.info.torch = true; } catch (e) { /* ignore */ }
    }
    this.caps = caps;
    this.canvas = document.createElement('canvas'); this.canvas.width = 32; this.canvas.height = 24;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.t0 = performance.now() / 1000; this.lockTimer = setTimeout(() => this.lockExposure(), 2500);
    this.loop();
    this.onInfo(this.info);
  }
  async lockExposure() {
    // Freeze auto exposure / white balance so fingertip colour changes are not cancelled by the camera. Best effort.
    const c = this.caps || {}; let ok = false;
    const tryC = async (adv) => { try { await this.track.applyConstraints({ advanced: [adv] }); return true; } catch (e) { return false; } };
    if (c.exposureMode && c.exposureMode.includes('manual')) ok = (await tryC({ exposureMode: 'manual' })) || ok;
    if (c.whiteBalanceMode && c.whiteBalanceMode.includes('manual')) ok = (await tryC({ whiteBalanceMode: 'manual' })) || ok;
    this.info.exposureLocked = ok; this.onInfo(this.info);
  }
  loop() {
    if (this.stopped) return;
    const cb = (now) => { this.frame(now / 1000); if (!this.stopped) this.loop(); };
    if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback((now) => cb(now));
    else requestAnimationFrame((now) => cb(now));
  }
  frame(t) {
    const v = this.video; if (!v.videoWidth) return;
    this.ctx.drawImage(v, 0, 0, 32, 24);
    const d = this.ctx.getImageData(0, 0, 32, 24).data; let r = 0, g = 0, b = 0; const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
    const cur = [t, r / n, g / n, b / n]; this.frames++;
    this.info.fps = this.frames / Math.max(t - this.t0, 1e-3);
    if (!this.prev) { this.prev = cur; this.nextT = t; this.emit(cur); return; }
    const gap = t - this.prev[0]; if (gap > this.info.maxGap) this.info.maxGap = gap; if (gap > 0.5) this.info.gaps++;
    while (this.nextT <= t) {   // linear resample onto the 30 Hz grid
      const a = (this.nextT - this.prev[0]) / Math.max(t - this.prev[0], 1e-9);
      this.onSample(this.prev[1] + a * (cur[1] - this.prev[1]), this.prev[2] + a * (cur[2] - this.prev[2]), this.prev[3] + a * (cur[3] - this.prev[3]));
      this.nextT += 1 / 30;
    }
    this.prev = cur;
  }
  emit(cur) { this.onSample(cur[1], cur[2], cur[3]); this.nextT += 1 / 30; }
  stop() {
    this.stopped = true; clearTimeout(this.lockTimer);
    if (this.track && this.info.torch) { try { this.track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) { /* ignore */ } }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
  }
}

// Accelerometer-based "are you holding still" check.
export class Motion {
  constructor() { this.buf = []; this.ok = false; this.handler = (e) => this.on(e); }
  async start() {
    try {
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        const r = await DeviceMotionEvent.requestPermission(); if (r !== 'granted') return false;
      }
      if (typeof DeviceMotionEvent === 'undefined') return false;
      window.addEventListener('devicemotion', this.handler); this.listening = true;
      await new Promise((r) => setTimeout(r, 700));
      return (this.ok = this.buf.length > 3);
    } catch (e) { return false; }
  }
  on(e) {
    const a = e.accelerationIncludingGravity; if (!a || a.x == null) return;
    this.buf.push([performance.now(), Math.hypot(a.x, a.y, a.z)]);
    const cut = performance.now() - 12000; while (this.buf.length && this.buf[0][0] < cut) this.buf.shift();
  }
  /** std of |acceleration| (m/s^2) over the last `sec` seconds, or null if no sensor */
  level(sec = 3) {
    if (!this.ok) return null;
    const cut = performance.now() - sec * 1000, v = this.buf.filter((x) => x[0] >= cut).map((x) => x[1]);
    if (v.length < 5) return null;
    const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
  }
  stop() { if (this.listening) window.removeEventListener('devicemotion', this.handler); }
}
