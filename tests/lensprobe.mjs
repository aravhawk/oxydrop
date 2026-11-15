// Lens probe, mode ladder and adaptive exposure, with a mocked navigator.mediaDevices and synthetic frames. Usage: node tests/lensprobe.mjs
// No phone is needed: each fake lens has its own pulse amplitude / noise / saturation, so the selection logic can be checked in Node.
import assert from 'node:assert/strict';
import { scoreSignal, probeLens, probeAll, chooseMode, runProbeLadder, liveQuality, ringState, probeSummary, SCORE_MIN, FS, PROBE_SEC, SETTLE_SEC } from '../app/js/lensprobe.js';
import { listBackCameras, pickDefault, CameraSource } from '../app/js/camera.js';
let n = 0; const t = (name, fn) => fn().then(() => { n++; console.log('ok  ', name); });

let seed = 11; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296), gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
/** One synthetic RGB sample: level L, relative pulse amplitude amp (R/G/B scaled 1/1.6/1.3), additive noise sigma, clamped to 0..255. */
const sample = (i, p) => {
  const x = 2 * Math.PI * 1.2 * i / FS, w = Math.sin(x) + 0.5 * Math.sin(2 * x + 1), k = [1, 1.6, 1.3];
  return [150, 110, 80].map((v, c) => Math.min(255, Math.max(0, p.level * v / 150 * (1 + p.amp * k[c] * w) + gauss() * p.noise)));
};
const series = (p, secs = 3.5) => Array.from({ length: Math.round(secs * FS) }, (_, i) => sample(i, p));

// profile per fake lens: [torch-on profile, torch-off profile]
const PROF = {
  wide:   { torch: { level: 150, amp: 0.004, noise: 0.15 }, ambient: { level: 60, amp: 0.002, noise: 0.4 } },     // default main lens: decent
  ultra:  { torch: { level: 170, amp: 0.015, noise: 0.20 }, ambient: { level: 70, amp: 0.004, noise: 0.1 } },     // strongest pulse (flash next to it)
  tele:   { torch: { level: 120, amp: 0.0, noise: 1.5 },    ambient: { level: 50, amp: 0.0, noise: 1.5 } },       // not covered: noise only
  macro:  { torch: { level: 600, amp: 0.02, noise: 0.5 },   ambient: { level: 60, amp: 0.0, noise: 1.0 } },       // saturated (clipped at 255)
  front:  { torch: { level: 140, amp: 0.006, noise: 0.2 },  ambient: { level: 140, amp: 0.006, noise: 0.2 } },    // white-screen front camera
};
const DEVICES = [
  { deviceId: 'w', kind: 'videoinput', label: 'Back Triple Camera', p: PROF.wide },
  { deviceId: 'u', kind: 'videoinput', label: 'Back Ultra Wide Camera', p: PROF.ultra },
  { deviceId: 't', kind: 'videoinput', label: 'Back Telephoto Camera', p: PROF.tele },
  { deviceId: 'm', kind: 'videoinput', label: 'Back Macro Camera', p: PROF.macro },
  { deviceId: 'f', kind: 'videoinput', label: 'Front Camera', p: PROF.front },
];

/** Install a mocked navigator.mediaDevices. cfg: devices, torchMode: 'ok' | 'unsupported' | 'throws'. Tracks open streams. */
function mockNavigator(cfg) {
  const state = { open: 0, maxOpen: 0, opened: [], closed: [], constraints: [], torchApplied: [] };
  const mediaDevices = {
    enumerateDevices: async () => cfg.devices.map(({ deviceId, kind, label }) => ({ deviceId, kind, label })),
    getUserMedia: async ({ video }) => {
      state.constraints.push(video);
      const dev = video.deviceId ? cfg.devices.find((d) => d.deviceId === video.deviceId.exact) : (video.facingMode.ideal === 'user' ? cfg.devices.find((d) => /front/i.test(d.label)) : cfg.devices[0]);
      if (!dev) throw new Error('NotFoundError');
      if (dev.openFails) throw Object.assign(new Error('busy'), { name: 'NotReadableError' });
      state.open++; state.maxOpen = Math.max(state.maxOpen, state.open); state.opened.push(dev.deviceId);
      const isFront = /front/i.test(dev.label);
      const track = {
        label: dev.label, torch: false, stopped: false,
        getCapabilities: () => (cfg.torchMode === 'unsupported' || isFront ? {} : { torch: true }),
        getSettings: () => ({ deviceId: dev.deviceId, width: 640, height: 480 }),
        applyConstraints: async (c) => { const tc = c.advanced && c.advanced[0] && c.advanced[0].torch; if (tc !== undefined) { if (cfg.torchMode === 'throws') throw new Error('OverconstrainedError'); track.torch = !!tc; state.torchApplied.push([dev.deviceId, !!tc]); } },
        stop: () => { if (!track.stopped) { track.stopped = true; state.open--; state.closed.push(dev.deviceId); } },
        addEventListener() {}, removeEventListener() {},
      };
      return { getVideoTracks: () => [track], getTracks: () => [track], dev, track };
    },
  };
  Object.defineProperty(globalThis, 'navigator', { value: { mediaDevices }, configurable: true, writable: true });
  return state;
}

/** Probe camera backed by the mocked getUserMedia: emits synthetic 30 Hz samples (fast, not wall-clock) while the stream is open. */
function fakeCamFactory() {
  return (onSample) => {
    const cam = { info: { torchSupported: false, torch: false, lens: '' }, satFrac: 0, _timer: null, _stream: null };
    cam.start = async ({ deviceId, facing, torch }) => {
      const v = { width: { ideal: 640 } }; if (deviceId) v.deviceId = { exact: deviceId }; else v.facingMode = { ideal: facing };
      const s = await navigator.mediaDevices.getUserMedia({ video: v }); cam._stream = s; const tr = s.track;
      cam.info.lens = tr.label; cam.info.torchSupported = !!tr.getCapabilities().torch;
      if (torch === 'on' && cam.info.torchSupported) { try { await tr.applyConstraints({ advanced: [{ torch: true }] }); cam.info.torch = true; } catch (e) { cam.info.torch = false; } }
      const prof = s.dev.p[cam.info.torch || /front/i.test(s.dev.label) ? 'torch' : 'ambient'];
      let i = 0; cam._timer = setInterval(() => { for (let k = 0; k < 15; k++) { const px = sample(i++, prof); cam.satFrac = Math.min(px[0], px[1], px[2]) >= 250 ? 1 : 0; onSample(...px); } }, 1);
    };
    cam.stop = async () => { clearInterval(cam._timer); if (cam._stream) cam._stream.track.stop(); };
    return cam;
  };
}
const FAST = { gapMs: 0, timeoutMs: 20000 };

// ---------- scoreSignal on raw series
await t('clean pulse scores high, noise-only scores low, flat/dark rejected', async () => {
  const good = scoreSignal(series(PROF.ultra.torch)), noise = scoreSignal(series(PROF.tele.torch)), dark = scoreSignal(series({ level: 1, amp: 0.01, noise: 0.05 }));
  console.log(`      good ${good.score.toFixed(1)} (acdc ${(good.acdc * 100).toFixed(2)}% sqi ${good.sqi.toFixed(2)} acf ${good.acf.toFixed(2)})  noise ${noise.score.toFixed(1)}  dark ${dark.score}`);
  assert.ok(good.score >= SCORE_MIN + 20, `good ${good.score}`); assert.ok(noise.score < SCORE_MIN, `noise ${noise.score}`);
  assert.equal(dark.reason, 'dark'); assert.equal(dark.score, 0);
  assert.ok(Math.abs(good.hr - 72) < 6, `hr ${good.hr}`);
});
await t('saturated lens is penalised', async () => {
  const sat = scoreSignal(series(PROF.macro.torch)), clean = scoreSignal(series({ ...PROF.macro.torch, level: 170 }));
  console.log(`      saturated ${sat.score.toFixed(1)} (sat ${(sat.sat * 100).toFixed(0)}%)  same pulse unsaturated ${clean.score.toFixed(1)}`);
  assert.ok(sat.sat > 0.3); assert.equal(sat.reason, 'saturated'); assert.ok(sat.score < clean.score * 0.5); assert.ok(sat.score < SCORE_MIN);
});
await t('weak but clean (diffuse) beats strong but noisy', async () => {
  const weak = scoreSignal(series({ level: 150, amp: 0.0025, noise: 0.03 })), strong = scoreSignal(series({ level: 150, amp: 0.01, noise: 4 }));
  console.log(`      weak-clean ${weak.score.toFixed(1)} (acdc ${(weak.acdc * 100).toFixed(2)}%)  strong-noisy ${strong.score.toFixed(1)} (acdc ${(strong.acdc * 100).toFixed(2)}%)`);
  assert.ok(strong.acdc > weak.acdc * 2, 'the noisy one has the larger raw amplitude'); assert.ok(weak.score > strong.score + 10); assert.ok(weak.score >= SCORE_MIN, 'weak-clean must still be usable');
});

// ---------- lens probe through the mocked camera API
await t('enumerates back lenses only; default heuristic is the plain wide lens', async () => {
  mockNavigator({ devices: DEVICES, torchMode: 'ok' });
  const back = await listBackCameras();
  assert.deepEqual(back.map((b) => b.id), ['w', 'u', 't', 'm']); assert.notEqual(pickDefault(back).id, 'w', 'physical lens preferred over virtual triple');
});
await t('probe picks the strongest-pulse lens, not the default wide lens; each stream is released before the next opens', async () => {
  const st = mockNavigator({ devices: DEVICES, torchMode: 'ok' }), back = await listBackCameras();
  const t0 = Date.now();
  const res = await probeAll(fakeCamFactory(), back.map((b) => ({ ...b, mode: 'torch' })), FAST);
  console.log('      ' + res.map((r) => `${r.label}: ${r.score.toFixed(1)} (${r.reason})`).join(' | '), `[${Date.now() - t0} ms]`);
  assert.equal(res[0].deviceId, 'u'); assert.ok(res[0].score > res.find((r) => r.deviceId === 'w').score, 'strongest measured pulse beats virtual default');
  assert.equal(res.find((r) => r.deviceId === 't').usable, false, 'noise-only lens rejected');
  assert.equal(res.find((r) => r.deviceId === 'm').reason, 'saturated');
  assert.ok(res.find((r) => r.deviceId === 'm').score < res.find((r) => r.deviceId === 'w').score, 'saturated lens ranks below the clean one');
  assert.ok(res.every((r) => r.samples === Math.round((PROBE_SEC + SETTLE_SEC) * FS)), 'each probe collects the complete settled acquisition budget');
  assert.equal(st.maxOpen, 1, 'never two streams open at once'); assert.equal(st.open, 0, 'everything released'); assert.deepEqual(st.opened, ['w', 'u', 't', 'm']);
  assert.ok(st.torchApplied.every(([, on]) => on), 'torch requested for torch probes');
  const c = chooseMode(res); assert.equal(c.mode, 'torch'); assert.equal(c.deviceId, 'u'); assert.ok(c.usable); assert.equal(c.scores.length, 4);
  assert.match(probeSummary(res, c), /Back Ultra Wide Camera/);
});
await t('a lens that fails to open does not stop the probe and releases nothing it did not open', async () => {
  const dev = DEVICES.map((d) => (d.deviceId === 'u' ? { ...d, openFails: true } : d)), st = mockNavigator({ devices: dev, torchMode: 'ok' });
  const res = await probeAll(fakeCamFactory(), ['w', 'u'].map((id) => ({ id, label: dev.find((d) => d.deviceId === id).label, mode: 'torch' })), FAST);
  assert.equal(res.find((r) => r.deviceId === 'u').reason, 'open_failed'); assert.equal(res[0].deviceId, 'w'); assert.equal(st.open, 0);
});

// ---------- mode ladder
const backTargets = (devs) => devs.filter((d) => !/front/i.test(d.label)).map((d) => ({ id: d.deviceId, label: d.label }));
await t('torch unsupported -> ambient-light mode on the best ambient lens', async () => {
  const st = mockNavigator({ devices: DEVICES, torchMode: 'unsupported' });
  const { choice, results } = await runProbeLadder(fakeCamFactory(), { back: backTargets(DEVICES), front: true }, FAST);
  console.log('      ' + choice.mode + ': ' + choice.reason);
  assert.equal(choice.mode, 'ambient'); assert.match(choice.reason, /torch could not be used \(torch_unsupported\)/);
  assert.ok(results.filter((r) => r.mode === 'ambient').length <= 2); assert.equal(st.open, 0);
});
await t('torch throws -> reports torch_apply_failed and falls through', async () => {
  mockNavigator({ devices: DEVICES, torchMode: 'throws' });
  const { choice, results } = await runProbeLadder(fakeCamFactory(), { back: backTargets(DEVICES), front: true }, FAST);
  assert.ok(results.filter((r) => r.mode === 'torch').every((r) => r.error === 'torch_apply_failed')); assert.match(choice.reason, /torch_apply_failed/); assert.notEqual(choice.mode, 'torch');
});
await t('torch fails and ambient is too dark -> front camera with white screen', async () => {
  const dev = DEVICES.map((d) => (/front/i.test(d.label) ? d : { ...d, p: { torch: d.p.torch, ambient: { level: 1, amp: 0.001, noise: 0.3 } } }));
  mockNavigator({ devices: dev, torchMode: 'throws' });
  const { choice } = await runProbeLadder(fakeCamFactory(), { back: backTargets(dev), front: true }, FAST);
  console.log('      ' + choice.mode + ': ' + choice.reason);
  assert.equal(choice.mode, 'front'); assert.equal(choice.deviceId, null); assert.ok(choice.usable);
});
await t('torch works but every back lens scores below threshold -> front camera is picked automatically (first-class mode)', async () => {
  const dev = DEVICES.map((d) => (/front/i.test(d.label) ? d : { ...d, p: { torch: { level: 130, amp: 0.0, noise: 1.5 }, ambient: d.p.ambient } }));
  mockNavigator({ devices: dev, torchMode: 'ok' });
  const calls = []; const { choice, results } = await runProbeLadder(fakeCamFactory(), { back: backTargets(dev), front: true }, { ...FAST, beforeFront: async () => calls.push('white-on'), afterFront: async () => calls.push('white-off') });
  console.log('      ' + choice.mode + ': ' + choice.reason);
  assert.equal(choice.mode, 'front'); assert.deepEqual(calls, ['white-on', 'white-off']); assert.ok(results.some((r) => r.mode === 'front' && r.usable));
  assert.equal(results.filter((r) => r.mode === 'ambient').length, 0, 'front is tried before ambient when the torch itself works');
});
await t('torch path succeeds -> front camera is never opened', async () => {
  const st = mockNavigator({ devices: DEVICES, torchMode: 'ok' });
  const { choice } = await runProbeLadder(fakeCamFactory(), { back: backTargets(DEVICES), front: true }, FAST);
  assert.equal(choice.mode, 'torch'); assert.ok(!st.opened.includes('f'));
});
await t('nothing usable anywhere -> best effort with usable=false and the per-lens scores for the debug view', async () => {
  const dev = DEVICES.map((d) => ({ ...d, p: { torch: { level: 130, amp: 0, noise: 1.5 }, ambient: { level: 1, amp: 0, noise: 0.3 } } }));
  mockNavigator({ devices: dev, torchMode: 'ok' });
  const { choice, results } = await runProbeLadder(fakeCamFactory(), { back: backTargets(dev), front: true }, FAST);
  assert.equal(choice.usable, false); assert.ok(choice.scores.length >= 5); assert.match(choice.reason, /No mode reached/); assert.ok(results.some((r) => r.mode === 'front'));
});
await t('chooseMode on hand-made results: ladder order and reasons', async () => {
  const r = (mode, score, extra = {}) => ({ mode, score, ok: true, torchOn: mode === 'torch', deviceId: mode + score, label: mode, ...extra });
  assert.equal(chooseMode([r('torch', 60), r('torch', 20)]).deviceId, 'torch60');
  assert.equal(chooseMode([r('torch', 10), r('ambient', 50), r('front', 70)]).mode, 'front');
  assert.equal(chooseMode([r('torch', 0, { ok: false, torchOn: false, error: 'torch_unsupported' }), r('ambient', 50), r('front', 70)]).mode, 'ambient');
  assert.equal(chooseMode([r('torch', 0, { ok: false, torchOn: false, error: 'torch_unsupported' }), r('ambient', 10), r('front', 70)]).mode, 'front');
});

// ---------- live quality meter / ring
await t('live quality: reposition prompt instead of a reading when SNR is low', async () => {
  const good = liveQuality(series(PROF.ultra.torch, 8)), bad = liveQuality(series(PROF.tele.torch, 8)), dark = liveQuality(series({ level: 1, amp: 0, noise: 0.1 }, 8));
  assert.equal(good.level, 'good'); assert.equal(good.prompt, ''); assert.equal(bad.level, 'reposition'); assert.match(bad.prompt, /Re-position/); assert.equal(dark.level, 'reposition'); assert.match(dark.prompt, /lens AND the flash/);
  assert.equal(liveQuality(series(PROF.ultra.torch, 1)).reason, 'too_short');
  assert.equal(ringState([200, 60, 40]).covered, true); assert.equal(ringState([90, 100, 110]).covered, false); assert.equal(ringState([250, 20, 10], 0.5).covered, false);
});

// ---------- adaptive exposure
await t('adaptive exposure steps a weak diffuse image up into the 90-200 luma band and records it in info.adapt', async () => {
  const cam = new CameraSource(() => {}), model = { ec: 0 }, applied = [];
  cam.caps = { exposureCompensation: { min: -3, max: 3, step: 0.5 } };
  cam.track = { getSettings: () => ({ exposureCompensation: model.ec }), applyConstraints: async (c) => { const v = c.advanced[0].exposureCompensation; model.ec = v; applied.push(v); cam.stat = { ...cam.stat, luma: 40 + 35 * v, satFrac: 0 }; } };
  cam.stat = { mean: [0, 0, 0], satFrac: 0, spatialCV: 0, luma: 40 };
  const a = await cam.tryAdaptExposure({ sleep: async () => {}, settleMs: 0 });
  assert.equal(a.result, 'in_band'); assert.ok(cam.stat.luma >= 90 && cam.stat.luma <= 200); assert.ok(applied.length >= 1 && applied.length <= 6); assert.ok(applied.every((v) => v >= -3 && v <= 3));
  assert.equal(cam.info.adapt, a); assert.deepEqual(a.supported, ['exposureCompensation']);
});
await t('adaptive exposure: steps down when saturated, bounded, and a no-op without capabilities', async () => {
  const cam = new CameraSource(() => {}), model = { ec: 2 };
  cam.caps = { exposureCompensation: { min: -3, max: 3, step: 1 } };
  cam.track = { getSettings: () => ({ exposureCompensation: model.ec }), applyConstraints: async (c) => { model.ec = c.advanced[0].exposureCompensation; cam.stat = { ...cam.stat, luma: 150 + 40 * model.ec, satFrac: model.ec > 0 ? 0.4 : 0.02 }; } };
  cam.stat = { mean: [0, 0, 0], satFrac: 0.4, spatialCV: 0, luma: 230 };
  const a = await cam.tryAdaptExposure({ sleep: async () => {}, settleMs: 0 });
  assert.ok(cam.stat.satFrac < 0.10 && a.result === 'in_band', JSON.stringify(a)); assert.ok(model.ec <= 0.5);
  const none = new CameraSource(() => {}); none.caps = {}; none.stat = { luma: 30, satFrac: 0 };
  assert.equal((await none.tryAdaptExposure({ sleep: async () => {} })).result, 'unsupported_left_on_auto');
});

// ---------- review fixes: settle-frame saturation and cancellation
await t('saturation during the settle frames does not zero a lens score', async () => {
  mockNavigator({ devices: DEVICES, torchMode: 'ok' });
  const settleN = Math.round(SETTLE_SEC * FS);
  const wrap = (satOf) => (onSample) => { let k = 0; const cam = fakeCamFactory()((r, g, b) => { k++; cam.satFrac = satOf(k); onSample(r, g, b); }); return cam; };
  const target = { id: 'u', label: 'Back Ultra Wide Camera', mode: 'torch' };
  const early = await probeLens(wrap((k) => (k <= settleN ? 0.9 : 0)), target, FAST);
  const clean = await probeLens(wrap(() => 0), target, FAST);
  const late = await probeLens(wrap((k) => (k <= settleN ? 0 : 0.9)), target, FAST);
  console.log(`      settle-sat ${early.score.toFixed(1)}  no-sat ${clean.score.toFixed(1)}  body-sat ${late.score.toFixed(1)}`);
  assert.ok(clean.score >= SCORE_MIN); assert.ok(Math.abs(early.score - clean.score) < 3, 'settle saturation ignored');
  assert.equal(late.reason, 'saturated'); assert.ok(late.score < SCORE_MIN, 'saturation after settle still penalised');
});
await t('cancelling mid-probe stops the ladder, releases the stream and never shows the white screen', async () => {
  const st = mockNavigator({ devices: DEVICES, torchMode: 'ok' }), calls = [];
  let abort = false, opened = 0; const base = fakeCamFactory(), fac = (cb) => { if (++opened === 2) abort = true; return base(cb); };   // cancel while lens 2 is in flight
  const t0 = Date.now();
  const out = await runProbeLadder(fac, { back: backTargets(DEVICES), front: true }, { ...FAST, shouldAbort: () => abort, beforeFront: async () => { calls.push('before'); }, afterFront: async () => { calls.push('after'); } });
  assert.ok(out.cancelled); assert.ok(Date.now() - t0 < 2500, 'returned promptly'); assert.equal(st.open, 0, 'no stream left open');
  assert.ok(out.results.length < 4 + 1); assert.ok(!calls.includes('before'), 'white overlay never shown after cancel');
});
await t('an abort set before the overlay hook runs still pairs afterFront', async () => {
  const dev = DEVICES.map((d) => (/front/i.test(d.label) ? d : { ...d, p: { torch: { level: 130, amp: 0.0, noise: 0 }, ambient: d.p.ambient } })), calls = []; let abort = false;
  mockNavigator({ devices: dev, torchMode: 'ok' });
  const out = await runProbeLadder(fakeCamFactory(), { back: backTargets(dev), front: true }, { ...FAST, shouldAbort: () => abort, beforeFront: async () => { calls.push('before'); abort = true; }, afterFront: async () => { calls.push('after'); } });
  assert.ok(out.cancelled); assert.deepEqual(calls, ['before', 'after']);
});
console.log(`LENS PASS (${n} checks)`);
