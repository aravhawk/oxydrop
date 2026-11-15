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
  const score = (l) => (/dual|triple|virtual|fusion/i.test(l) ? -100 : 0) + (/ultra|0\.5|tele|zoom|macro|depth/i.test(l) ? -5 : 0) + (/back camera$|wide|main|camera2 0,/i.test(l) ? 3 : 0);
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

/** getUserMedia failures that mean 'this deviceId is not available' (as opposed to a denied permission or a busy camera). */
export const isDeviceGone = (e) => !!e && /^(NotFoundError|OverconstrainedError|DevicesNotFoundError)$/.test(e.name || '');

/** Actual channel means and clipping in a rectangular region; no channel remapping. */
export function regionStats(data, width, roi) {
  const sum = [0, 0, 0], clipped = [0, 0, 0]; let allClip = 0, l1 = 0, l2 = 0, n = 0;
  for (let y = roi.y; y < roi.y + roi.h; y++) for (let x = roi.x; x < roi.x + roi.w; x++) {
    const i = 4 * (y * width + x); let count = 0;
    for (let c = 0; c < 3; c++) { sum[c] += data[i + c]; if (data[i + c] >= 250) { clipped[c]++; count++; } }
    if (count === 3) allClip++;
    const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]; l1 += l; l2 += l * l; n++;
  }
  const luma = l1 / n;
  return { mean: sum.map(v => v / n), channelSat: clipped.map(v => v / n), satFrac: allClip / n,
    spatialCV: luma > 1 ? Math.sqrt(Math.max(0, l2 / n - luma * luma)) / luma : 0, luma };
}

/** Prefer a well-lit region with an unclipped channel. Red may clip while green/blue remain useful for HR. */
export function selectROI(data, width = 32, height = 24) {
  let best = { x: 0, y: 0, w: width, h: height }, bestScore = -1;
  const candidates = [best];
  const w = Math.floor(width / 4), h = Math.floor(height / 3);
  for (let y = 0; y + h <= height; y += h) for (let x = 0; x + w <= width; x += w) candidates.push({ x, y, w, h });
  for (const roi of candidates) {
    const s = regionStats(data, width, roi);
    const usable = Math.max(...s.mean.map((dc, c) => dc > 8 && dc < 245 ? dc * (1 - s.channelSat[c]) : 0));
    const score = usable / (1 + s.spatialCV) * (roi === candidates[0] ? 1.05 : 1);
    if (score > bestScore) { bestScore = score; best = roi; }
  }
  return best;
}

const LUMA_LO = 90, LUMA_HI = 200, SAT_CAP = 0.10, ADAPT_STEPS = 6;
export class CameraSource {
  constructor(onSample, onStats) {
    this.onSample = onSample; this.onStats = onStats || (() => {}); this.rs = new Resampler((r, g, b) => this.onSample(r, g, b));
    this.info = { torchSupported: false, torch: false, torchState: 'off', locks: {}, lens: '', width: 0, height: 0, settings: {} };
    this.stopped = false; this.frames = 0; this.t0 = 0; this.stat = { mean: [0, 0, 0], satFrac: 0, channelSat: [0, 0, 0], spatialCV: 0, luma: 0 }; this.roi = null; this.lastMediaTime = null; this.lastPresented = null; this.skipped = 0;
    this.lastFrameAt = 0; this.interrupted = null; this.locked = false;
    this.nativeCallbacks = 0; this.fallbackCallbacks = 0; this.frameErrors = 0; this.lastFrameError = '';
    this.captureState = 'idle'; this.lastNativeCallbackAt = null; this.lastDecoded = null; this.lastFrameClock = null;
    this.firstFrame = new Promise(resolve => { this._firstFrameResolve = resolve; });
    this.ready = new Promise(resolve => { this._readyResolve = resolve; });

  }
  async start(video, { deviceId = null, torch = 'auto', facing = 'environment', firstFrameTimeoutMs = 8000, startupTimeoutMs, allowFrameTimeout = false, shouldAbort } = {}) {
    this.video = video; this.torchMode = torch; this.info.facing = facing;
    this.shouldAbort = shouldAbort; this.startupTimeoutMs = startupTimeoutMs ?? firstFrameTimeoutMs;
    this.captureState = 'opening'; this.info.readiness = 'opening';
    // WebKit can suspend delivery for detached/display:none videos. Keep a small,
    // rendered inline preview attached throughout capture, including each probe.
    video.muted = true; video.defaultMuted = true; video.playsInline = true; video.autoplay = true; video.hidden = false;
    for (const attr of ['muted', 'playsinline', 'webkit-playsinline', 'autoplay']) video.setAttribute?.(attr, '');
    Object.assign(video.style || {}, { position: 'fixed', right: '8px', top: '8px', bottom: 'auto', width: '32px', height: '24px', display: 'block', visibility: 'visible', opacity: '1', pointerEvents: 'none', zIndex: '10003' });
    if (!video.isConnected) { document.body.appendChild(video); this.attachedVideo = true; }
    try {
    const base = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, facingMode = { ideal: facing === 'user' ? 'user' : 'environment' };
    if (facing === 'user') torch = 'off';   // the front camera has no torch; the screen is the light source
    this.info.fallback = '';
    let stream;
    try { stream = await this._startupWait(navigator.mediaDevices.getUserMedia({ audio: false, video: deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode } }), 'permission', s => s.getTracks().forEach(t => t.stop())); }
    catch (e) {
      // A stale or empty deviceId (ids change after a permission reset, in some browsers per session, and on fake devices) fails with NotFoundError / OverconstrainedError:
      // retry once on the facing mode alone and record why, so the debug view shows that the requested lens was not the one opened.
      if (!deviceId || !isDeviceGone(e)) throw e;
      this.info.fallback = `exact deviceId rejected (${e.name || 'error'}: ${e.message || 'no message'}); reopened by facingMode ${facing === 'user' ? 'user' : 'environment'}`;
      stream = await this._startupWait(navigator.mediaDevices.getUserMedia({ audio: false, video: { ...base, facingMode } }), 'permission', s => s.getTracks().forEach(t => t.stop()));
    }
    this.stream = stream; video.srcObject = stream; this.captureState = 'playing'; this.info.readiness = 'playing'; await this._startupWait(video.play(), 'play');
    this.track = stream.getVideoTracks()[0];
    this.caps = this.track.getCapabilities ? this.track.getCapabilities() : {};
    const st = this.track.getSettings ? this.track.getSettings() : {};
    this.info.lens = this.track.label || ''; this.info.deviceId = st.deviceId; this.info.settings = st; this.info.width = st.width; this.info.height = st.height;
    this.info.torchSupported = !!this.caps.torch;
    await this._startupWait(this.setTorch(torch !== 'off'), 'torch');
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
    this.captureState = 'waiting'; this.info.readiness = 'waiting';
    this.loop();
    try { await this._startupWait(this.ready, 'frames'); }
    catch (e) { if (!(allowFrameTimeout && e.code === 'no_frames')) throw e; this.info.startupError = e.message; }
    return this.info;
    } catch (e) { this.info.startupError = e.message || String(e); this.stop(); this.captureState = 'failed'; this.info.readiness = 'failed'; throw e; }
  }
  _startupWait(promise, stage, disposeLate) {
    return new Promise((resolve, reject) => {
      let finished = false;
      const fail = (code) => {
        if (finished) return; finished = true; clearTimeout(timer); clearInterval(poll);
        const e = new Error(code === 'cancelled' ? 'Camera startup cancelled' : stage === 'frames' ? 'Camera opened but no advancing video frames arrived' : `Camera ${stage} did not become ready`);
        e.name = 'CameraStartupError'; e.code = code; if (this._startupCancel === cancel) this._startupCancel = null; reject(e);
      };
      const cancel = () => fail('cancelled'); this._startupCancel = cancel;
      const timer = setTimeout(() => fail(stage === 'frames' ? 'no_frames' : `${stage}_timeout`), this.startupTimeoutMs);
      const poll = setInterval(() => { if (this.stopped || this.shouldAbort?.()) fail('cancelled'); }, 50);
      Promise.resolve(promise).then(value => {
        if (finished) { if (disposeLate) disposeLate(value); return; }
        if (this.stopped || this.shouldAbort?.()) { fail('cancelled'); if (disposeLate) disposeLate(value); return; }
        finished = true; clearTimeout(timer); clearInterval(poll); if (this._startupCancel === cancel) this._startupCancel = null; resolve(value);
      }, error => { if (!finished) { finished = true; clearTimeout(timer); clearInterval(poll); if (this._startupCancel === cancel) this._startupCancel = null; reject(error); } });
    });
  }
  async setTorch(on) {
    this.info.torchRequested = !!on;
    const cap = this.caps?.torch;
    if (cap === false || (Array.isArray(cap) && !cap.includes(true))) {
      this.info.torch = false; this.info.torchState = 'unsupported'; return false;
    }
    // Some WebKit builds omit torch from capabilities. Still try it; a resolved
    // constraint alone is not proof that the LED turned on (unknown keys may be ignored).
    try {
      await this.track.applyConstraints({ torch: !!on, advanced: [{ torch: !!on }] });
      const settings = this.track.getSettings?.() || {};
      const confirmed = typeof settings.torch === 'boolean';
      this.info.torch = on && (!confirmed || settings.torch);
      this.info.torchSupported = !!cap || (confirmed && settings.torch === !!on);
      this.info.torchState = !on ? 'off' : confirmed ? (settings.torch ? 'confirmed' : 'failed') : 'on-unverified';
    } catch (e) { this.info.torch = false; this.info.torchState = 'failed'; this.info.torchError = e.name || 'constraint_failed'; }
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
    this.info.locks = {}; this.locked = false; this.roi = null;
  }
  async lockExposure() {
    // Freeze auto-exposure / white balance / focus so the camera does not cancel or add to the pulse signal. Best effort; support varies by browser and camera track.
    const c = this.caps || {}, ok = {};
    const tryC = async (adv) => { try { await this.track.applyConstraints({ advanced: [adv] }); return true; } catch (e) { return false; } };
    const st = this.track.getSettings?.() || {};
    if (c.exposureMode && c.exposureMode.includes('manual')) ok.exposure = await tryC({ exposureMode: 'manual', ...(Number.isFinite(st.exposureTime) ? { exposureTime: st.exposureTime } : {}), ...(Number.isFinite(st.iso) ? { iso: st.iso } : {}) });
    if (c.whiteBalanceMode && c.whiteBalanceMode.includes('manual')) ok.wb = await tryC({ whiteBalanceMode: 'manual', ...(Number.isFinite(st.colorTemperature) ? { colorTemperature: st.colorTemperature } : {}) });
    if (c.focusMode && c.focusMode.includes('manual')) ok.focus = await tryC({ focusMode: 'manual', ...(Number.isFinite(st.focusDistance) ? { focusDistance: st.focusDistance } : {}) });
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
    const clipped = () => Math.max(...(this.stat.channelSat || [this.stat.satFrac]));
    const inBand = () => this.stat.luma >= LUMA_LO && this.stat.luma <= LUMA_HI && clipped() < SAT_CAP;
    if (inBand()) { a.result = 'already_in_band'; return a; }
    const cur = {}, set = async (k, val) => {
      try { await this.track.applyConstraints({ advanced: [{ [k]: val }] }); cur[k] = val; a.applied.push({ k, val: Math.round(val * 1000) / 1000 }); return true; } catch (e) { return false; }
    };
    let ki = 0;
    for (let i = 0; i < steps && ki < knobs.length; i++) {
      const up = this.stat.luma < LUMA_LO && clipped() < SAT_CAP, k = knobs[ki], r = rng(k), st = r.step || (r.max - r.min) / 8;
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
    if (this.stopped || this.loopStarted) return;
    this.loopStarted = true; this.lastNativeCallbackAt = performance.now();
    const nativeLoop = () => {
      if (this.stopped || !this.video.requestVideoFrameCallback) return;
      try { this.videoCallbackId = this.video.requestVideoFrameCallback((now, meta) => {
        this.videoCallbackId = null; this.nativeCallbacks++; this.lastNativeCallbackAt = performance.now();
        try { this.frame(now, meta); } finally { nativeLoop(); }
      }); } catch (e) { this.lastFrameError = e.message || String(e); }
    };
    const fallbackLoop = (now) => {
      if (this.stopped) return;
      this.fallbackCallbacks++;
      try {
        if (!this.video.requestVideoFrameCallback || performance.now() - this.lastNativeCallbackAt > 250) this.frame(now, null);
      } finally { if (!this.stopped) this.animationId = requestAnimationFrame(fallbackLoop); }
    };
    nativeLoop(); this.animationId = requestAnimationFrame(fallbackLoop);
  }
  frame(now, meta) {
    const v = this.video;
    if (this.stopped || !v.videoWidth || (v.videoHeight !== undefined && !v.videoHeight) || (v.readyState !== undefined && v.readyState < 2)) return false;
    let decoded = null;
    try { const n = v.getVideoPlaybackQuality?.().totalVideoFrames ?? v.webkitDecodedFrameCount; if (Number.isFinite(n)) decoded = n; } catch (e) { /* optional counter */ }
    const presented = Number.isFinite(meta?.presentedFrames) ? meta.presentedFrames : null;
    const mediaTime = Number.isFinite(meta?.mediaTime) ? meta.mediaTime : Number.isFinite(v.currentTime) ? v.currentTime : null;
    // A native presented-frame counter is authoritative. Some Safari versions
    // repeat mediaTime/currentTime (even zero) while real camera frames advance.
    if (presented !== null) { if (this.lastPresented !== null && presented <= this.lastPresented) return false; }
    else if (decoded !== null) { if (this.lastDecoded !== null && decoded <= this.lastDecoded) return false; }
    else if (mediaTime !== null && mediaTime === this.lastMediaTime) return false;
    // All accepted frames share one monotonic callback clock. captureTime may
    // be finite but frozen at zero, so never let it stop the 30 Hz resampler.
    const t = now / 1000;
    if (this.lastFrameClock !== null && t <= this.lastFrameClock) return false;
    try {
    this.ctx.drawImage(v, 0, 0, 32, 24);
    const d = this.ctx.getImageData(0, 0, 32, 24).data;
    // Pick a region once, after exposure settles. Keep it fixed while measuring:
    // chasing the brightest tile each frame would create an artificial AC signal.
    let roiChanged = false;
    if (!this.roi || (!this.locked && (this.frames === 15 || (this.frames % 60 === 0 && (Math.max(...this.stat.mean) <= 3 || this.stat.satFrac > 0.1))))) {
      this.roi = selectROI(d, 32, 24); roiChanged = true;
    }
    const region = regionStats(d, 32, this.roi), { mean } = region;
    const fullMean = regionStats(d, 32, { x: 0, y: 0, w: 32, h: 24 }).mean;
    if (presented !== null) {
      if (this.lastPresented !== null && presented - this.lastPresented > 1) this.skipped += presented - this.lastPresented - 1;
      this.lastPresented = presented;
    }
    this.lastDecoded = decoded; this.lastMediaTime = mediaTime; this.lastFrameClock = t;
    this.lastFrameAt = performance.now();
    this.stat = { ...region, fullMean, roi: { ...this.roi }, roiChanged }; this.frames++;
    if (this.frames === 1) this._firstFrameResolve?.(this.stats);
    if (this.frames >= 2) { this.captureState = 'ready'; this.info.readiness = 'ready'; this._readyResolve?.(this.stats); }
    this.rs.push(t, mean[0], mean[1], mean[2]); this.onStats(this.stats);
    return true;
    } catch (e) { this.frameErrors++; this.lastFrameError = `${e.name || 'FrameError'}: ${e.message || e}`; return false; }
  }
  get stats() { return { ...this.stat, captureState: this.captureState, readiness: this.captureState, firstFrame: this.frames > 0, nativeCallbacks: this.nativeCallbacks, fallbackCallbacks: this.fallbackCallbacks, frameErrors: this.frameErrors, lastFrameError: this.lastFrameError, videoReadyState: this.video?.readyState ?? 0, videoWidth: this.video?.videoWidth ?? 0, videoHeight: this.video?.videoHeight ?? 0, staleMs: this.staleMs, interrupted: this.interrupted, fps: this.rs.fps, jitterMs: this.rs.jitterMs, gaps: this.rs.gaps, maxGap: this.rs.maxGap, skipped: this.skipped, samples: this.rs.n, frames: this.frames }; }
  stop() {
    this.stopped = true; this.captureState = 'stopped'; this.info.readiness = 'stopped';
    this._startupCancel?.();
    if (this.videoCallbackId != null) { try { this.video?.cancelVideoFrameCallback?.(this.videoCallbackId); } catch (e) { /* already cancelled */ } this.videoCallbackId = null; }
    if (this.animationId != null) { if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this.animationId); this.animationId = null; }
    if (this.track && this.onTrackEnd) { this.track.removeEventListener('ended', this.onTrackEnd); this.track.removeEventListener('mute', this.onTrackMute); this.track.removeEventListener('unmute', this.onTrackUnmute); document.removeEventListener('visibilitychange', this.onVis); }
    if (this.track && this.info.torch) { try { this.track.applyConstraints({ advanced: [{ torch: false }] }).catch(() => {}); } catch (e) { /* ignore */ } }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.video) { try { this.video.pause?.(); this.video.srcObject = null; } catch (e) { /* detached already */ } if (this.attachedVideo) { this.video.remove?.(); this.attachedVideo = false; } }
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
