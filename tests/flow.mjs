// Integration test of the live start flow: tap "Turn on camera" -> lens probe -> start camera -> setup screen visible.
// It loads the real app/js/app.js in Node against a minimal fake DOM and a fake navigator.mediaDevices (synthetic 72 bpm fingertip frames),
// so it exercises the session / camera-generation guards that the unit tests for lensprobe.js and the stall logic cannot see.
// Regression this guards: startCamera() bumps the camera generation, and the caller then compared it with the value read before the probe, so the
// setup UI never appeared (REVIEW.md section 3). Cases: (1) cancel during the countdown, (2) full flow, (3) stale deviceId falls back to facingMode.
// Usage: node tests/flow.mjs        (about 25 s of real time: the app's own 3 s get-ready countdown and 3.5 s lens probe are not mocked)
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
    this.handlers = {}; this.classList = { toggle() {}, add() {}, remove() {} }; this.className = ''; this.selectedOptions = []; this._html = '';
  }
  addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
  removeEventListener() {}
  appendChild(c) { this.children.push(c); return c; }
  remove() {}
  getContext() { return ctx; }
  set innerHTML(v) { this._html = v; if (v === '') this.children = []; }
  get innerHTML() { return this._html; }
  async click() { const hs = this.handlers.click || []; await Promise.all(hs.map((h) => h({ target: this }))); }
}
const noop = () => {};
let pulseAt = () => [150, 110, 80];
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
pulseAt = (t) => { const x = 2 * Math.PI * 1.2 * t, w = Math.sin(x) + 0.5 * Math.sin(2 * x + 1); return [150 * (1 + 0.006 * w), 110 * (1 + 0.010 * w), 80 * (1 + 0.008 * w)]; };

class FakeVideo {
  constructor() { this.videoWidth = 640; this.videoHeight = 480; this.srcObject = null; this.n = 0; }
  async play() {}
  requestVideoFrameCallback(cb) { return setTimeout(() => { const t = performance.now(); cb(t, { captureTime: t, presentedFrames: ++this.n }); }, 33); }
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
globalThis.fetch = async (url) => ({ json: async () => JSON.parse(fs.readFileSync(new URL('../app/' + url, import.meta.url), 'utf8')) });

await import(process.env.FLOW_APP || '../app/js/app.js');

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
await sleep(1200);
assert.match($('dbg-setup').textContent, /samples [1-9]\d*/, 'frames reach the engine after the transition (the new camera generation is live)');
ok('probe -> start -> setup UI visible, one stream open, frames flowing');

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
