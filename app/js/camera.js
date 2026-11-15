// Rear camera capture: lens selection, optional torch, exposure/white-balance/focus lock, true frame timestamps, per-frame mean RGB.
import { Resampler } from './resample.js';

export async function listBackCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const back = devs.filter((d) => !/front|user|facetime|selfie/i.test(d.label));
    return (back.length ? back : devs).map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` }));
  } catch (e) { return []; }
}
/** Probe targets: every back lens plus the front camera (opened by facingMode; if the device has none, its probe just fails and is skipped). */
export async function listCameras() { return { back: await listBackCameras(), front: true }; }
/** Default lens: the plain main/wide back camera. Ultra-wide and telephoto sit farther from the torch; virtual multi-lens devices may switch lenses mid-measurement. */
export function pickDefault(cams) {
  const score = (l) => (/ultra|0\.5|tele|zoom|macro|depth/i.test(l) ? -5 : 0) + (/back camera$|wide|main|camera2 0,/i.test(l) ? 3 : 0) + (/dual|triple/i.test(l) ? -1 : 0);
  return [...cams].sort((a, b) => score(b.label) - score(a.label))[0] || null;
}

/** Front camera is addressed by facingMode (no deviceId): its geometry is the same on every phone. */
export const FRONT = { id: null, label: 'Front camera' };

/** Probe-camera factory for lensprobe.js: a CameraSource driven through a hidden video element, released with srcObject = null on stop. */
export function probeCamFactory(video) {
  return (onSample) => {
    const c = new CameraSource(onSample);
    return {
      get info() { return c.info; }, get satFrac() { return c.stat.satFrac; }, get stats() { return c.stats; },
      start: (o) => c.start(video, o),
      async stop() { c.stop(); video.srcObject = null; },
    };
  };
}

const LUMA_LO = 90, LUMA_HI = 200, SAT_CAP = 0.10, ADAPT_STEPS = 6;
export class CameraSource {
  constructor(onSample, onStats) {
    this.onSample = onSample; this.onStats = onStats || (() => {}); this.rs = new Resampler((r, g, b) => this.onSample(r, g, b));
    this.info = { torchSupported: false, torch: false, locks: {}, lens: '', width: 0, height: 0, settings: {} };
    this.stopped = false; this.frames = 0; this.t0 = 0; this.stat = { mean: [0, 0, 0], satFrac: 0, spatialCV: 0, luma: 0 }; this.lastPresented = null; this.skipped = 0;
    this.lastFrameAt = 0; this.interrupted = null; this.locked = false;
  }
  async start(video, { deviceId = null, torch = 'auto', facing = 'environment' } = {}) {
    this.video = video; this.torchMode = torch; this.info.facing = facing;
    const v = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
    if (deviceId) v.deviceId = { exact: deviceId }; else v.facingMode = { ideal: facing === 'user' ? 'user' : 'environment' };
    if (facing === 'user') torch = 'off';   // the front camera has no torch; the screen is the light source
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: v });
    this.stream = stream; video.srcObject = stream; video.muted = true; video.playsInline = true; await video.play();
    this.track = stream.getVideoTracks()[0];
    this.caps = this.track.getCapabilities ? this.track.getCapabilities() : {};
    const st = this.track.getSettings ? this.track.getSettings() : {};
    this.info.lens = this.track.label || ''; this.info.deviceId = st.deviceId; this.info.settings = st; this.info.width = st.width; this.info.height = st.height;
    this.info.torchSupported = !!this.caps.torch;
    await this.setTorch(torch !== 'off');
    this.canvas = document.createElement('canvas'); this.canvas.width = 32; this.canvas.height = 24;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.t0 = performance.now() / 1000; this.lastFrameAt = performance.now();
    // Exposure/WB/focus are NOT locked here: the app calls lockExposure() once the fingertip has been placed correctly for a few seconds,
    // so the lock is taken on the covered lens and not on the open scene.
    this.onTrackEnd = () => { this.interrupted = 'The camera track ended.'; };
    this.onTrackMute = () => { this.interrupted = 'The camera was muted by the browser.'; };
    this.onTrackUnmute = () => { if (this.interrupted === 'The camera was muted by the browser.') this.interrupted = null; };
    this.onVis = () => { if (document.hidden) this.interrupted = 'The page was sent to the background.'; else if (this.interrupted === 'The page was sent to the background.') this.interrupted = null; };
    this.track.addEventListener('ended', this.onTrackEnd); this.track.addEventListener('mute', this.onTrackMute); this.track.addEventListener('unmute', this.onTrackUnmute);
    document.addEventListener('visibilitychange', this.onVis);
    this.loop();
  }
  async setTorch(on) {
    if (!this.info.torchSupported) { this.info.torch = false; return false; }
    try { await this.track.applyConstraints({ advanced: [{ torch: !!on }] }); this.info.torch = !!on; } catch (e) { this.info.torch = false; }
    return this.info.torch;
  }
  /** Wall-clock ms since the last camera frame arrived (0 before the first frame). Large when the camera stalls or is interrupted. */
  get staleMs() { return this.lastFrameAt ? performance.now() - this.lastFrameAt : 0; }
  async unlockExposure() {
    // Back to continuous auto modes (used when the light mode changes or the fingertip is lifted), so the next lock is taken on the new scene.
    const c = this.caps || {}, tryC = async (adv) => { try { await this.track.applyConstraints({ advanced: [adv] }); return true; } catch (e) { return false; } };
    if (c.exposureMode && c.exposureMode.includes('continuous')) await tryC({ exposureMode: 'continuous' });
    if (c.whiteBalanceMode && c.whiteBalanceMode.includes('continuous')) await tryC({ whiteBalanceMode: 'continuous' });
    if (c.focusMode && c.focusMode.includes('continuous')) await tryC({ focusMode: 'continuous' });
    this.info.locks = {}; this.locked = false;
  }
  async lockExposure() {
    // Freeze auto-exposure / white balance / focus so the camera does not cancel or add to the pulse signal. Best effort, support varies (Android Chrome yes; iOS Safari no).
    const c = this.caps || {}, ok = {};
    const tryC = async (adv) => { try { await this.track.applyConstraints({ advanced: [adv] }); return true; } catch (e) { return false; } };
    if (c.exposureMode && c.exposureMode.includes('manual')) ok.exposure = await tryC({ exposureMode: 'manual' });
    if (c.whiteBalanceMode && c.whiteBalanceMode.includes('manual')) ok.wb = await tryC({ whiteBalanceMode: 'manual' });
    if (c.focusMode && c.focusMode.includes('manual')) ok.focus = await tryC({ focusMode: 'manual' });
    this.info.locks = ok; this.locked = true;
  }
  /**
   * Adaptive exposure for a weak, diffuse signal (flash far from the lens): step exposure up (or down) until mean luma sits in 90-200 and
   * fewer than 10 % of pixels saturate. Uses exposureCompensation, then brightness, then iso / exposureTime if the track exposes them.
   * Bounded to a few steps; if nothing is adjustable it leaves continuous auto exposure alone. Result is recorded in info.adapt.
   * Call BEFORE lockExposure(): the lock then freezes whatever level was reached.
   */
  async tryAdaptExposure({ sleep = (ms) => new Promise((r) => setTimeout(r, ms)), settleMs = 450, steps = ADAPT_STEPS } = {}) {
    const c = this.caps || {}, knobs = [];
    const rng = (k) => (c[k] && Number.isFinite(c[k].min) && Number.isFinite(c[k].max) ? c[k] : null);
    for (const k of ['exposureCompensation', 'brightness', 'iso', 'exposureTime']) if (rng(k)) knobs.push(k);
    const a = { supported: knobs, applied: [], lumaBefore: this.stat.luma, lumaAfter: this.stat.luma, satAfter: this.stat.satFrac, result: 'none' };
    this.info.adapt = a;
    if (!knobs.length) { a.result = 'unsupported_left_on_auto'; return a; }
    const inBand = () => this.stat.luma >= LUMA_LO && this.stat.luma <= LUMA_HI && this.stat.satFrac < SAT_CAP;
    if (inBand()) { a.result = 'already_in_band'; return a; }
    const cur = {}, set = async (k, val) => {
      try { await this.track.applyConstraints({ advanced: [{ [k]: val }] }); cur[k] = val; a.applied.push({ k, val: Math.round(val * 1000) / 1000 }); return true; } catch (e) { return false; }
    };
    let ki = 0;
    for (let i = 0; i < steps && ki < knobs.length; i++) {
      const up = this.stat.luma < LUMA_LO && this.stat.satFrac < SAT_CAP, k = knobs[ki], r = rng(k), st = r.step || (r.max - r.min) / 8;
      const base = cur[k] ?? (this.track.getSettings ? (this.track.getSettings()[k] ?? (r.min + r.max) / 2) : (r.min + r.max) / 2);
      const stepSize = k === 'iso' || k === 'exposureTime' ? base * 0.5 : Math.max(st, (r.max - r.min) / 6);   // multiplicative for iso/time, additive otherwise
      const next = Math.min(r.max, Math.max(r.min, up ? base + stepSize : base - stepSize));
      if (next === base || !(await set(k, next))) { ki++; continue; }
      await sleep(settleMs);
      if (inBand()) break;
    }
    a.lumaAfter = this.stat.luma; a.satAfter = this.stat.satFrac;
    a.result = inBand() ? 'in_band' : (a.applied.length ? 'bounded_not_in_band' : 'no_change');
    return a;
  }
  loop() {
    if (this.stopped) return;
    if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback((now, meta) => { this.frame(now, meta); this.loop(); });
    else requestAnimationFrame((now) => { this.frame(now, null); this.loop(); });
  }
  frame(now, meta) {
    const v = this.video; if (!v.videoWidth) return;
    // true capture time when the browser provides it (Chrome), else callback time (Safari); the Resampler de-jitters either
    const t = (meta && Number.isFinite(meta.captureTime) ? meta.captureTime : now) / 1000;
    if (meta && Number.isFinite(meta.presentedFrames)) { if (this.lastPresented !== null && meta.presentedFrames - this.lastPresented > 1) this.skipped += meta.presentedFrames - this.lastPresented - 1; this.lastPresented = meta.presentedFrames; }
    this.ctx.drawImage(v, 0, 0, 32, 24);
    const d = this.ctx.getImageData(0, 0, 32, 24).data, n = d.length / 4; let r = 0, g = 0, b = 0, sat = 0, l1 = 0, l2 = 0;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; if (d[i] >= 250 || d[i + 1] >= 250 || d[i + 2] >= 250) sat++; const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; l1 += l; l2 += l * l; }
    const mean = [r / n, g / n, b / n], lm = l1 / n, lsd = Math.sqrt(Math.max(l2 / n - lm * lm, 0));
    this.lastFrameAt = performance.now();
    this.stat = { mean, satFrac: sat / n, spatialCV: lm > 1 ? lsd / lm : 0, luma: lm }; this.frames++;
    this.rs.push(t, mean[0], mean[1], mean[2]);
  }
  get stats() { return { ...this.stat, staleMs: this.staleMs, interrupted: this.interrupted, fps: this.rs.fps, jitterMs: this.rs.jitterMs, gaps: this.rs.gaps, maxGap: this.rs.maxGap, skipped: this.skipped, samples: this.rs.n, frames: this.frames }; }
  stop() {
    this.stopped = true;
    if (this.track && this.onTrackEnd) { this.track.removeEventListener('ended', this.onTrackEnd); this.track.removeEventListener('mute', this.onTrackMute); this.track.removeEventListener('unmute', this.onTrackUnmute); document.removeEventListener('visibilitychange', this.onVis); }
    if (this.track && this.info.torch) { try { this.track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) { /* ignore */ } }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
  }
}

// Accelerometer-based "are you holding still" check.
export class Motion {
  constructor() { this.buf = []; this.ok = false; this.handler = (e) => this.on(e); }
  async start() {
    try {
      if (typeof DeviceMotionEvent === 'undefined') return false;
      if (typeof DeviceMotionEvent.requestPermission === 'function') { const r = await DeviceMotionEvent.requestPermission(); if (r !== 'granted') return false; }
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
  level(sec = 3) {
    if (!this.ok) return null;
    const cut = performance.now() - sec * 1000, v = this.buf.filter((x) => x[0] >= cut).map((x) => x[1]);
    if (v.length < 5) return null;
    const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
  }
  stop() { if (this.listening) window.removeEventListener('devicemotion', this.handler); }
}
