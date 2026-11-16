// Integration test of the live start flow: tap "Turn on camera" -> lens probe -> start camera -> setup screen visible.
// It loads the real app/js/app.js in Node against a minimal fake DOM and a fake navigator.mediaDevices (synthetic 72 bpm fingertip frames),
// so it exercises the session / camera-generation guards that the unit tests for lensprobe.js and the stall logic cannot see.
// Regression this guards: startCamera() bumps the camera generation, and the caller then compared it with the value read before the probe, so the
// setup UI never appeared (REVIEW.md section 3). Cases: (1) cancel during the countdown, (2) full flow, (3) stale deviceId falls back to facingMode.
// Usage: node tests/flow.mjs        (about 25 s of real time: the app's own 3 s get-ready countdown and 3.5 s lens probe are not mocked)
//        (default)                                the default engine slot runs PhotonDistill (app/data/distill.json) in the page thread
//        FLOW_ENGINE=hybrid FLOW_HYBRID_FAIL=1 node tests/flow.mjs   ?engine=hybrid and the hybrid model assets fail to load (HTTP 503): the app falls back to the logistic chain and the measurement screen must say so
//        FLOW_ENGINE=hybrid FLOW_WORKER_FAIL=1 node tests/flow.mjs   the CNN Web Worker loads fine and then errors during the measurement: the app drops to the logistic chain and shows a visible, device-agnostic notice
//        FLOW_ENGINE=logistic node tests/flow.mjs   the live camera path with ?engine=logistic: the logistic chain runs on purpose, no fallback notice
//        FLOW_ENGINE=hybrid node tests/flow.mjs     ?engine=hybrid selects the previous hybrid (CNN worker)
//        FLOW_ENGINE=distill node tests/flow.mjs    ?engine=distill names the default PhotonDistill engine explicitly, no fallback notice
//        FLOW_DISTILL_FAIL=1 node tests/flow.mjs    the default model file data/distill.json fails to load (HTTP 503): logistic chain + visible notice, never the hybrid
//        FLOW_APP=../app/js/app_old.js node tests/flow.mjs    runs another copy of app.js (a path relative to this file, copy placed in app/js/) to show an old version fails
import assert from 'node:assert/strict';
import fs from 'node:fs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let nOk = 0; const ok = (name) => { nOk++; console.log('ok  ', name); };

// ---------------- fake DOM ----------------
const els = new Map(); let lightValue = 'flash';
class El {
  constructor(id = '') {
    this.id = id; this.textContent = ''; this.hidden = false; this.disabled = false; this.value = ''; this.width = 600; this.height = 120; this.children = []; this.style = {};
    this.isConnected = false; this.parentNode = null; this.handlers = {}; this.classList = { toggle() {}, add() {}, remove() {} }; this.className = ''; this.selectedOptions = []; this._html = '';
  }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  removeEventListener() {}
  appendChild(c) { this.children.push(c); c.isConnected = true; c.parentNode = this; return c; }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((c) => c !== this); this.isConnected = false; this.parentNode = null; }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  getContext() { return ctx; }
  set innerHTML(v) { this._html = v; if (v === '') this.children = []; }
  get innerHTML() { return this._html; }
  async click() { const hs = this.handlers.click || []; await Promise.all(hs.map((h) => h({ target: this }))); }
}
const noop = () => {};
let pulseAt = () => [200, 80, 60], fingerOff = false;   // fingerOff: an open grey scene (finger lifted)
const ctx = new Proxy({}, {
  get(_, p) { if (p === 'getImageData') return () => frame(); if (p === 'drawImage') return noop; return noop; },
  set() { return true; },
});
/** 32x24 RGBA frame whose mean is the synthetic pulse value; the fractional part is dithered over the pixels because ImageData holds integers. */
function frame() {
  const n = 32 * 24, d = new Uint8ClampedArray(n * 4), rgb = pulseAt(performance.now() / 1000);
  for (let c = 0; c < 3; c++) { const base = Math.floor(rgb[c]), extra = Math.round((rgb[c] - base) * n); for (let i = 0; i < n; i++) d[i * 4 + c] = base + (i < extra ? 1 : 0); }
  for (let i = 0; i < n; i++) d[i * 4 + 3] = 255;
  return { data: d };
}
pulseAt = (t) => { const x = 2 * Math.PI * 1.2 * t, w = Math.sin(x) + 0.5 * Math.sin(2 * x + 1); if (fingerOff) return [128, 128, 128]; return [200 * (1 + 0.006 * w), 80 * (1 + 0.010 * w), 60 * (1 + 0.008 * w)]; };

class FakeVideo extends El {
  constructor() { super('video'); this.videoWidth = 640; this.videoHeight = 480; this.readyState = 4; this.currentTime = 0; this.srcObject = null; this.n = 0; }
  async play() {}
  cancelVideoFrameCallback(id) { clearTimeout(id); }
  requestVideoFrameCallback(cb) { return setTimeout(() => { const t = performance.now(); this.currentTime += 1 / 30; cb(t, { captureTime: t, mediaTime: this.currentTime, presentedFrames: ++this.n }); }, 33); }
}
const document = {
  hidden: false, body: new El('body'),
  getElementById: (id) => { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); },
  querySelectorAll: () => [],
  querySelector: (sel) => {
    if (sel === 'input[name=light]:checked') return { value: lightValue };
    const m = /^input\[name=light\]\[value=(\w+)\]$/.exec(sel);
    if (m) return { set checked(v) { if (v) lightValue = m[1]; }, get checked() { return lightValue === m[1]; } };
    return null;
  },
  createElement: (tag) => (tag === 'video' ? new FakeVideo() : new El(tag)),
  addEventListener: noop, removeEventListener: noop,
};
const $ = (id) => document.getElementById(id);

// ---------------- fake camera stack ----------------
const log = { gum: [], tracks: [] };
let devices = [{ kind: 'videoinput', deviceId: 'back-main', label: 'camera2 0, facing back' }];
let rejectIds = new Set();          // exact deviceIds that getUserMedia rejects with NotFoundError ("Requested device not found")
function gumError(name, message) { const e = new Error(message); e.name = name; return e; }
const mediaDevices = {
  async enumerateDevices() { return devices; },
  async getUserMedia(c) {
    const v = c.video; log.gum.push(v);
    const exact = v && v.deviceId && v.deviceId.exact;
    const actual = { deviceId: 'actual-back', label: 'camera2 0, facing back' };   // what the browser opens when asked by facingMode only
    if (exact && exact !== actual.deviceId && (rejectIds.has(exact) || !devices.some((d) => d.deviceId === exact))) throw gumError('NotFoundError', 'Requested device not found');
    const dev = exact ? (exact === actual.deviceId ? actual : devices.find((d) => d.deviceId === exact)) : actual;
    const track = {
      kind: 'video', label: dev.label, stopped: false,
      getCapabilities: () => ({ torch: true }), getSettings: () => ({ deviceId: dev.deviceId, width: 640, height: 480 }),
      async applyConstraints() {}, addEventListener: noop, removeEventListener: noop, stop() { this.stopped = true; },
    };
    log.tracks.push(track);
    return { getTracks: () => [track], getVideoTracks: () => [track] };
  },
};
Object.defineProperty(globalThis, 'navigator', { value: { mediaDevices, userAgent: 'node-flow-test' }, configurable: true });
globalThis.window = { scrollTo: noop, addEventListener: noop, removeEventListener: noop };
globalThis.document = document;
globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 16);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
const LOGISTIC_REQ = process.env.FLOW_ENGINE === 'logistic';
const DISTILL_ARG = process.env.FLOW_ENGINE === 'distill';
const HYBRID_ARG = process.env.FLOW_ENGINE === 'hybrid';
const DISTILL_REQ = !LOGISTIC_REQ && !HYBRID_ARG;          // the default engine (no ?engine=) is the PhotonDistill engine slot
globalThis.location = { search: LOGISTIC_REQ ? '?engine=logistic&debug=1' : DISTILL_ARG ? '?engine=distill&debug=1' : HYBRID_ARG ? '?engine=hybrid&debug=1' : '?debug=1' };
const HYBRID_FAIL = process.env.FLOW_HYBRID_FAIL === '1';
const WORKER_FAIL = process.env.FLOW_WORKER_FAIL === '1';
const DISTILL_FAIL = process.env.FLOW_DISTILL_FAIL === '1';
if ((HYBRID_FAIL || WORKER_FAIL) && !HYBRID_ARG) throw new Error('FLOW_HYBRID_FAIL / FLOW_WORKER_FAIL need FLOW_ENGINE=hybrid (the hybrid is no longer the default)');
if (!LOGISTIC_REQ) {
  // A stand-in module Worker: it starts and reports ready (so the app commits to the worker path), never answers a window, and raises an error event 1 s after the measurement's engine is created.
  globalThis.Worker = class {
    constructor() { this.onmessage = null; this.onerror = null; this.terminated = false; globalThis.__fakeWorker = this; }
    postMessage(m) {
      if (m.type === 'init') setTimeout(() => this.onmessage && this.onmessage({ data: { type: 'ready', info: { model: 'OxyNet residual CNN, 3 seeds', params: 167588, paramsTotal: 502764, alpha: 0.1, thresholds: { normal_below: 0, low_above: 0 } } } }), 20);
      else if (m.type === 'win') setTimeout(() => { if (!this.terminated && this.onmessage) this.onmessage({ data: { type: 'res', gen: m.gen, id: m.id, h: { calibrating: true } } }); }, 1);
      else if (WORKER_FAIL && m.type === 'reset') setTimeout(() => { if (!this.terminated && this.onerror) this.onerror({ message: 'simulated worker crash' }); }, 1000);
    }
    terminate() { this.terminated = true; }
  };
}
globalThis.fetch = async (url) => {
  if (HYBRID_FAIL && /hybrid_|conformal/.test(url)) return { ok: false, status: 503, json: async () => { throw new Error('unavailable'); } };
  if (DISTILL_FAIL && /distill/.test(url)) return { ok: false, status: 503, json: async () => { throw new Error('unavailable'); } };
  return { ok: true, json: async () => JSON.parse(fs.readFileSync(new URL('../app/' + url, import.meta.url), 'utf8')) };
};

await import(process.env.FLOW_APP || '../app/js/app.js');
assert.equal(document.body.children.find(e => e.id === 'capture-debug').hidden, false, '?debug=1 opens overlay');
await document.body.children.find(e => e.id === 'capture-debug-toggle').click();
assert.equal(document.body.children.find(e => e.id === 'capture-debug').hidden, true, 'overlay toggle closes it');
await document.body.children.find(e => e.id === 'capture-debug-toggle').click();

const openTracks = () => log.tracks.filter((t) => !t.stopped).length;
const fresh = () => { log.gum = []; log.tracks = []; $('setup-live').hidden = true; };
const lastCamState = () => $('cam-state').textContent;

// ---------------- 1. cancelling during the get-ready countdown ----------------
fresh();
await $('ack').handlers.change[0]({ target: { checked: true } });
await $('btn-live').click();
const cancelled = $('btn-cam').click();          // starts the countdown + probe
await sleep(1200);
await $('wm-stop').click();                       // Stop / Back: cleanup() while the probe is pending
await cancelled;
assert.equal($('setup-live').hidden, true, 'a cancelled start must not reveal the setup UI');
assert.equal(openTracks(), 0, 'a cancelled start must leave no open camera stream');
assert.equal(log.gum.length, 1, 'only the permission grant ran; no lens was opened after cancelling');
ok('cancel during the countdown: no setup UI, no stream left open, no lens opened');

// ---------------- 2. the full flow: probe -> start camera -> setup UI ----------------
fresh();
await $('btn-live').click();
assert.equal($('setup-live').hidden, true);
await $('btn-cam').click();
assert.equal($('setup-live').hidden, false, 'setup UI must be shown after the probe and camera start (the generation-guard blocker)');
assert.match(lastCamState(), /Camera on/, lastCamState());
assert.ok($('lens').children.length >= 1, 'lens list populated');
assert.ok($('checks').innerHTML.includes('<li'), 'checks list rendered');
assert.equal(openTracks(), 1, 'exactly one live camera stream while in setup');
assert.equal(lightValue, 'flash', 'the probe chose the torch');
assert.ok($('probe-scores').textContent.includes('torch'), 'probe scores shown');
await sleep(800);
assert.equal($('place-title').textContent, 'Fingertip detected');
await sleep(1200);
assert.match($('dbg-setup').textContent, /samples [1-9]\d*/, 'frames reach the engine after the transition (the new camera generation is live)');
ok('probe -> start -> setup UI visible, one stream open, frames flowing');

// starting calibration from setup (exposure settles first, then the minute begins) and a steady finger does not void it
await $('btn-begin-anyway').click();
assert.equal($('phase').textContent, 'Calibrating');
assert.equal($('status-main').textContent, 'Calibrating');
await sleep(1500);
assert.equal($('phase').textContent, 'Calibrating', 'a steady, well-placed finger must not restart the calibration minute');
await sleep(600);
if (LOGISTIC_REQ) {
  assert.match($('dbg-measure').textContent, /engine logistic chain v2 \(baseline \/ fallback\) \(\?engine=logistic\)/, 'the live path runs the logistic chain when asked for');
  assert.doesNotMatch($('dbg-measure').textContent, /oxydrop-hybrid-v1/, 'no hybrid engine in this mode');
  assert.equal($('engine-note').hidden, true, 'no fallback notice when the chain was requested on purpose');
  assert.equal($('status-main').textContent, 'Calibrating');
  ok('?engine=logistic on the live camera path: logistic chain, no fallback notice');
} else if (DISTILL_REQ && DISTILL_FAIL) {
  assert.match($('dbg-measure').textContent, /engine logistic chain v2 \(baseline \/ fallback\).*model file data\/distill\.json could not be loaded/, 'the debug text records the load failure of the default model');
  assert.equal($('engine-note').hidden, false, 'the default model load failure must be visible on the measurement screen');
  assert.match($('engine-note').textContent, /selected on-device model could not be loaded.*logistic chain alone.*three result words are unchanged/, $('engine-note').textContent);
  assert.doesNotMatch($('engine-note').textContent, /iPhone|Android|Pixel|Samsung/, 'the notice is device-agnostic');
  assert.doesNotMatch($('dbg-measure').textContent, /oxydrop-hybrid-v1/, 'a failed default model never falls back to the hybrid');
  assert.equal($('status-main').textContent, 'Calibrating', 'the measurement itself still runs on the fallback chain');
  ok('default engine with data/distill.json failing to load (HTTP 503): the logistic chain runs and the screen shows a visible, device-agnostic notice');
} else if (DISTILL_REQ) {
  assert.match($('dbg-measure').textContent, /engine oxydrop-photon-distill2-v1 \(default engine slot, \?engine=distill, model file data\/distill\.json\): PhotonDistill student MLP 39-h-h \(h = 32\/32\/64\)-3, 3 seeds.*runs in the page thread; output shown as the model gives it/, 'the default engine runs PhotonDistill in the page thread');
  assert.doesNotMatch($('dbg-measure').textContent, /oxydrop-hybrid-v1/, 'no hybrid engine in this mode');
  assert.equal(globalThis.__fakeWorker, undefined, 'the default engine starts no CNN worker');
  assert.equal($('engine-note').hidden, true, 'no fallback notice while the model runs');
  assert.equal($('status-main').textContent, 'Calibrating');
  ok(`${DISTILL_ARG ? '?engine=distill' : 'default (no ?engine=)'} on the live camera path: the PhotonDistill engine slot runs data/distill.json, no fallback notice`);
} else if (WORKER_FAIL) {
  assert.ok(globalThis.__fakeWorker, 'the app started the Web Worker');
  assert.equal(globalThis.__fakeWorker.terminated, true, 'the failed worker is shut down');
  assert.equal($('engine-note').hidden, false, 'a worker error after loading must be visible on the measurement screen, not only in the debug text');
  assert.match($('engine-note').textContent, /neural-network part of the model stopped working.*logistic chain alone.*three result words are unchanged/, $('engine-note').textContent);
  assert.doesNotMatch($('engine-note').textContent, /iPhone|Android|Pixel|Samsung/, 'the notice is device-agnostic');
  assert.match($('dbg-measure').textContent, /engine logistic chain v2 \(baseline \/ fallback\).*model worker failed after loading \(simulated worker crash\)/, 'the debug text records the worker failure');
  assert.equal($('status-main').textContent, 'Calibrating', 'the measurement keeps running on the logistic chain');
  ok('CNN worker errors after loading: the logistic chain takes over the running measurement and the screen shows a visible, device-agnostic notice');
} else if (!HYBRID_FAIL) {
  assert.match($('dbg-measure').textContent, /engine oxydrop-hybrid-v1: v1 physics chain \+ OxyNet residual CNN, 3 seeds, 167,588 parameters per seed/, 'the default engine in the measurement is the hybrid (CNN)');
  assert.match($('dbg-measure').textContent, /conformal alpha 0\.1 /, 'the conformal level is shown in the debug summary');
  assert.equal($('engine-note').hidden, true, 'no fallback notice while the hybrid model runs');
  ok('?engine=hybrid: calibration starts after exposure setup and is not restarted while the finger stays on; the hybrid CNN runs');
} else {
  assert.match($('dbg-measure').textContent, /engine logistic chain v2 \(baseline \/ fallback\).*hybrid model could not be loaded/, 'the debug text records the load failure');
  assert.equal($('engine-note').hidden, false, 'the load failure must be visible on the measurement screen, not only in the debug text');
  assert.match($('engine-note').textContent, /neural-network part of the model could not be loaded.*logistic chain alone/, $('engine-note').textContent);
  assert.doesNotMatch($('engine-note').textContent, /iPhone|Android|Pixel|Samsung/, 'the notice is device-agnostic');
  assert.equal($('status-main').textContent, 'Calibrating', 'the measurement itself still runs on the fallback chain');
  ok('hybrid assets fail to load (HTTP 503): the logistic chain runs and the measurement screen shows a visible, device-agnostic notice');
}

// lifting the finger for > 2 s inside the calibration minute voids it; putting it back starts the minute again from zero
fingerOff = true; await sleep(2800);
assert.equal($('phase').textContent, 'Calibrating', 'not restarted while the finger is still off');
fingerOff = false; await sleep(1500);
assert.equal($('phase').textContent, 'Calibrating (restarted)');
assert.match($('status-sub').textContent, /came off the lens or flash during calibration, so the minute was started again/);
ok('finger lifted during calibration: the minute restarts once the finger is back');
if (WORKER_FAIL) { assert.equal($('engine-note').hidden, false, 'the notice stays visible after the calibration minute restarts'); assert.match($('dbg-measure').textContent, /engine logistic chain v2/, 'the restarted measurement uses the logistic chain, not the dead worker'); ok('the restarted measurement stays on the logistic chain with the notice shown'); }

// ---------------- 3. a stale / rejected deviceId falls back to facingMode ----------------
await $('wm-stop').click(); fresh();
devices = [{ kind: 'videoinput', deviceId: 'stale-lens-0', label: 'camera2 0, facing back' }];
rejectIds = new Set(['stale-lens-0']);
await $('btn-live').click();
await $('btn-cam').click();
assert.equal($('setup-live').hidden, false, 'a rejected exact deviceId must not block the setup UI');
const exactTries = log.gum.filter((v) => v && v.deviceId && v.deviceId.exact === 'stale-lens-0').length, facingTries = log.gum.filter((v) => v && v.facingMode && !v.deviceId).length;
assert.ok(exactTries >= 1, 'the stale exact id was tried first');
assert.ok(facingTries >= 1, 'and retried by facingMode');
assert.match(lastCamState(), /Camera on/, lastCamState());
assert.match($('probe-scores').textContent, /\[exact deviceId rejected \(NotFoundError: Requested device not found\)/, 'the probe table says the lens fell back');
assert.equal(openTracks(), 1);
// the same fallback on the live camera: choosing a lens whose id has gone stale in the lens menu
$('lens').value = 'stale-lens-0'; await $('lens').handlers.change[0]();
assert.match(lastCamState(), /Camera on/, lastCamState());
await sleep(700);
assert.match($('dbg-setup').textContent, /device fallback: exact deviceId rejected \(NotFoundError: Requested device not found\)/, 'the debug view says why');
assert.equal(openTracks(), 1, 'the old stream was released when the lens changed');
ok('NotFoundError on an exact deviceId: retried by facingMode, surfaced in the debug view');

await $('wm-stop').click(); await sleep(100);
assert.equal(openTracks(), 0, 'stopping releases every stream');
ok('stop releases the camera');
console.log(`FLOW PASS (${nOk} checks)`);
process.exit(0);
