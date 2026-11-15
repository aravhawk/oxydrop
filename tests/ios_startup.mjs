// Real elapsed startup delay reproduces WebKit's warm-up: play() resolves well
// before the first camera frame. Acquisition time starts at the first sample.
import assert from 'node:assert/strict';
import { probeLens, runProbeLadder, FS, PROBE_SEC, SETTLE_SEC } from '../app/js/lensprobe.js';

function fakeIOSCamera({ firstFrameMs = 1500, neverFrames = false } = {}) {
  const log = { opened: [], closed: [], emitted: 0, active: 0, maxActive: 0 };
  const factory = (onSample) => {
    let warmup, interval, i = 0, stopped = false;
    const cam = {
      info: { torchSupported: true, torch: true, lens: 'Back Camera' }, satFrac: 0,
      get stats() { return { frames: i, samples: i, fps: i ? 30 : 0 }; },
      async start(opts) {
        log.opened.push(opts); log.active++; log.maxActive = Math.max(log.maxActive, log.active);
        if (!neverFrames) warmup = setTimeout(() => {
          interval = setInterval(() => {
            const x = 2 * Math.PI * 1.2 * i++ / FS, w = Math.sin(x) + .35 * Math.sin(2 * x + .3);
            log.emitted++;
            onSample(150 * (1 + .012 * w), 110 * (1 + .018 * w), 80 * (1 + .014 * w));
          }, 1000 / FS);
        }, firstFrameMs);
      },
      async stop() { if (stopped) return; stopped = true; clearTimeout(warmup); clearInterval(interval); log.active--; log.closed.push(true); },
    };
    return cam;
  };
  return { factory, log };
}

const delayed = fakeIOSCamera();
const start = performance.now();
const measured = await probeLens(delayed.factory, { id: 'physical-wide', label: 'Back Camera', mode: 'torch' }, { gapMs: 0 });
assert.equal(measured.samples, Math.round((PROBE_SEC + SETTLE_SEC) * FS), 'late first frame retains the complete settle + body sample budget');
assert.ok(measured.ok && measured.usable, JSON.stringify(measured));
assert.ok(performance.now() - start >= 1500 + (PROBE_SEC + SETTLE_SEC) * 900, 'did not spend capture budget before first frame');
assert.equal(delayed.log.active, 0);
console.log(`IOS STARTUP PASS: first frame delayed 1.5 s; collected ${measured.samples} samples (${PROBE_SEC}s body + ${SETTLE_SEC}s settling)`);

const noFrames = fakeIOSCamera({ neverFrames: true });
const targets = { back: [
  { id: 'triple', label: 'Back Triple Camera' },
  { id: 'physical-wide', label: 'Back Camera' },
  { id: 'ultra', label: 'Back Ultra Wide Camera' },
], front: true };
const out = await runProbeLadder(noFrames.factory, targets, { firstFrameTimeoutMs: 40, startupTimeoutMs: 40, timeoutMs: 40, gapMs: 0 });
assert.equal(out.choice.mode, 'torch', 'no frames must keep a continuous rear-torch path available');
assert.equal(out.choice.deviceId, 'physical-wide', 'fallback retains the default physical rear camera');
assert.equal(out.choice.usable, false, 'a startup fallback never fabricates a pulse score');
assert.ok(!noFrames.log.opened.some((o) => o.facing === 'user'), 'do not retry front after every rear camera produced zero frames');
assert.ok(!noFrames.log.opened.some((o) => o.deviceId === 'triple'), 'virtual triple is excluded while physical rear lenses exist');
assert.equal(noFrames.log.active, 0); assert.equal(noFrames.log.maxActive, 1);
console.log('IOS STARTUP PASS: no frames selects continuous default physical rear-torch fallback; virtual/front not opened');

// Safari may expose requestVideoFrameCallback but never deliver it. The camera
// watchdog must recover through decoded-frame counters without counting 60 Hz RAF
// ticks as new 30 Hz camera images.
const { CameraSource } = await import('../app/js/camera.js');
const rgba = new Uint8ClampedArray(32 * 24 * 4);
for (let i = 0; i < rgba.length; i += 4) rgba.set([180, 80, 40, 255], i);
let videoWasAttachedAtPlay = false, trackStopped = false;
const video = {
  isConnected: false, style: {}, videoWidth: 640, videoHeight: 480, readyState: 4,
  startedAt: 0, srcObject: null, attrs: new Map(),
  setAttribute(k, v) { this.attrs.set(k, v); },
  removeAttribute(k) { this.attrs.delete(k); },
  async play() { videoWasAttachedAtPlay = this.isConnected; this.startedAt = performance.now(); },
  get currentTime() { return Math.floor((performance.now() - this.startedAt) / (1000 / FS)) / FS; },
  getVideoPlaybackQuality() { return { totalVideoFrames: Math.floor((performance.now() - this.startedAt) / (1000 / FS)) }; },
  requestVideoFrameCallback() { return 777; }, // intentionally never invokes callback
  cancelVideoFrameCallback() {},
  addEventListener() {}, removeEventListener() {},
  remove() { this.isConnected = false; },
};
const track = { label: 'Back Camera', getCapabilities: () => ({ torch: true }), getSettings: () => ({ torch: true, width: 640, height: 480 }),
  async applyConstraints() {}, addEventListener() {}, removeEventListener() {}, stop() { trackStopped = true; } };
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: { async getUserMedia() { return { getVideoTracks: () => [track], getTracks: () => [track] }; } } } });
globalThis.document = {
  hidden: false, body: { appendChild(v) { v.isConnected = true; return v; } },
  createElement() { return { getContext: () => ({ drawImage() {}, getImageData: () => ({ data: rgba }) }) }; },
  addEventListener() {}, removeEventListener() {},
};
globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 16);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
let samples = 0;
const camera = new CameraSource(() => samples++);
try {
  await camera.start(video, { firstFrameTimeoutMs: 1200, torch: 'on' });
  assert.ok(videoWasAttachedAtPlay, 'probe video must be attached before play on iOS');
  assert.ok(video.attrs.has('playsinline') && video.attrs.has('muted'), 'inline/muted attributes present before playback');
  assert.ok(camera.frames >= 2, 'ready requires two genuine frames, not play resolution');
  await new Promise((resolve) => setTimeout(resolve, 200));
  assert.ok(samples > 0 && camera.frames >= 4, 'RAF watchdog recovers capture while native callbacks never fire');
  assert.ok(camera.frames < 25, 'watchdog deduplicates RAF using decoded camera counters');
} finally { camera.stop(); }
assert.ok(trackStopped);
console.log('IOS CAMERA WATCHDOG PASS: attached inline video, readiness from real frames, fallback when native callbacks never fire');

// Continuous rear fallback survives an initial no-frames timeout and recovers
// when the existing stream finally begins delivering genuine camera frames.
let liveTrackStopped = false, livePresented = 0;
const liveTrack = { ...track, stop() { liveTrackStopped = true; } };
navigator.mediaDevices.getUserMedia = async () => ({ getVideoTracks: () => [liveTrack], getTracks: () => [liveTrack] });
const lateLiveVideo = {
  isConnected: false, style: {}, videoWidth: 640, videoHeight: 480, readyState: 4, currentTime: 0,
  setAttribute() {}, removeAttribute() {}, async play() {}, remove() { this.isConnected = false; },
  requestVideoFrameCallback(cb) { return setTimeout(() => cb(performance.now(), { captureTime: 0, mediaTime: 0, presentedFrames: ++livePresented }), livePresented ? 33 : 150); },
  cancelVideoFrameCallback(id) { clearTimeout(id); },
};
let lateSamples = 0;
const continuousCamera = new CameraSource(() => lateSamples++);
try {
  await continuousCamera.start(lateLiveVideo, { firstFrameTimeoutMs: 40, allowFrameTimeout: true, torch: 'on' });
  assert.equal(continuousCamera.stats.captureState, 'waiting');
  assert.equal(liveTrackStopped, false, 'continuous fallback keeps its track after initial timeout');
  assert.ok(lateLiveVideo.srcObject, 'stream stays attached while waiting');
  await new Promise(resolve => setTimeout(resolve, 260));
  assert.equal(continuousCamera.stats.captureState, 'ready');
  assert.ok(continuousCamera.frames >= 3 && lateSamples > 0, 'late native frames recover on the same stream with frozen metadata');
} finally { continuousCamera.stop(); }
assert.ok(liveTrackStopped);
console.log('IOS CONTINUOUS RECOVERY PASS: timeout keeps rear stream alive; delayed native frames resume samples without reopening');
