// Model-agnostic lens selection. No phone layout is assumed: every physical back lens is probed for >=2.5 s after real frames and warm-up with the torch on and the one with the
// strongest pulsatile signal wins. If no back lens works, the ladder falls to ambient light, then the front camera with a white screen.
// Pure ES module (no DOM). The camera is injected as a factory so the same code runs in the browser and in the Node test.
//
// Probe camera interface (see probeCamFactory in camera.js):  const cam = camFactory(onSample)   // onSample(r, g, b) at 30 Hz
//   await cam.start({ deviceId, facing, torch })   cam.info.{torchSupported, torch}   cam.satFrac (optional)   await cam.stop()
import { SOS, SAT_MAX } from './engine.js';
import { estimateHr, GATE } from './hr.js';

export const FS = 30, PROBE_SEC = 2.5, SETTLE_SEC = 1.0, SCORE_MIN = 35, SAT_LEVEL = 250, DARK_LEVEL = 3;
const NOISE_ACF = 0.40, CLEAN_ACF = 0.75, NOISE_SQI = 0.30, CLEAN_SQI = 0.40;
const PROBE_GATE = { ...GATE, MIN_SEC: 1.25, F_LO: 0.7, F_HI: 3.5 };              // a 3 s probe is far shorter than the 9.5 s decision gate; used for ranking only
const mean = (a) => { let s = 0; for (const v of a) s += v; return a.length ? s / a.length : 0; };
const stdp = (a) => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return a.length ? Math.sqrt(s / a.length) : 0; };
const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** The engine's 0.7-4 Hz band-pass (same SOS as Chain), run on one detrended log channel. */
function bandpass(x) {
  const z = SOS.map(() => [0, 0]), out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let v = x[i];
    for (let s = 0; s < SOS.length; s++) { const [b0, b1, b2, , a1, a2] = SOS[s], zz = z[s]; const y = b0 * v + zz[0]; zz[0] = b1 * v - a1 * y + zz[1]; zz[1] = b2 * v - a2 * y; v = y; }
    out[i] = v;
  }
  return out;
}

// A short acquisition probe cannot establish an HR. Rank narrow-band energy
// against residual energy using a sinusoid fit with linear trend removed.
function shortBandFit(x, fs) {
  const n = x.length, center = (n - 1) / 2;
  const detrend = (v) => { const m = mean(v); let xy = 0, xx = 0; for (let i = 0; i < n; i++) { xy += (i - center) * (v[i] - m); xx += (i - center) ** 2; } return v.map((a, i) => a - m - xy / xx * (i - center)); };
  const y = detrend(x), power = y.reduce((a, b) => a + b * b, 0); let best = { quality: 0, hz: NaN };
  for (let hz = 0.7; hz <= 3.5001; hz += 0.05) {
    const sn = detrend(x.map((_, i) => Math.sin(2 * Math.PI * hz * i / fs))), cs = detrend(x.map((_, i) => Math.cos(2 * Math.PI * hz * i / fs)));
    let ss = 0, cc = 0, sc = 0, ys = 0, yc = 0;
    for (let i = 0; i < n; i++) { ss += sn[i] ** 2; cc += cs[i] ** 2; sc += sn[i] * cs[i]; ys += y[i] * sn[i]; yc += y[i] * cs[i]; }
    const det = ss * cc - sc * sc; if (det <= 1e-12 || power <= 1e-15) continue;
    const a = (ys * cc - yc * sc) / det, b = (yc * ss - ys * sc) / det;
    const quality = clamp01((a * ys + b * yc) / power);
    if (quality > best.quality) best = { quality, hz };
  }
  return best;
}

/** Saturation penalty: 1 up to the engine's SAT_MAX, falling linearly to 0 at 40 % saturated samples. */
export const satFactor = (sat) => (sat <= SAT_MAX ? 1 : clamp01(1 - (sat - SAT_MAX) / 0.30));

/**
 * Score a short RGB record (array of [r,g,b] at fs Hz, 0..255).
 * acdc = std of the band-passed log signal (about AC/DC); sqi/acf come from hr.js estimateHr; snr = sqi/(1-sqi) (peak power vs rest of band).
 * score 0..100 = 100 * saturationFactor * quality * (0.4 + 0.6 * amplitude); quality weights spectral concentration and periodicity,
 * so a weak but clean (diffuse) pulse beats a strong but noisy one.
 */
export function scoreSignal(rgb, fs = FS, opts = {}) {
  const settle = Math.round((opts.settleSec ?? SETTLE_SEC) * fs);
  const res = { n: rgb.length, acdc: 0, snr: 0, sqi: 0, acf: 0, sat: 0, dc: 0, luma: 0, channel: 1, hr: NaN, factor: 1, score: 0, reason: '' };
  if (rgb.length < settle + PROBE_GATE.MIN_SEC * fs) { res.reason = 'too_short'; return res; }
  const body = rgb.slice(settle);
  res.dc = Math.max(mean(body.map((p) => p[0])), mean(body.map((p) => p[1])), mean(body.map((p) => p[2])));
  res.luma = mean(body.map((p) => 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]));
  res.channelSat = [0, 1, 2].map((c) => body.filter((p) => p[c] >= SAT_LEVEL).length / body.length);
  res.sat = Math.max(opts.satFrac ?? 0, Math.min(...res.channelSat));
  res.factor = satFactor(res.sat);
  if (res.dc <= DARK_LEVEL) { res.reason = 'dark'; return res; }
  let best = null;
  for (let c = 0; c < 3; c++) {
    const sat = Math.max(opts.satFrac ?? 0, res.channelSat[c]);
    if (sat > SAT_MAX || mean(body.map((p) => p[c])) < 5) continue;
    const lg = rgb.map((p) => Math.log(Math.max(p[c], 1e-3) + 1)), m = mean(lg), y = bandpass(lg.map((v) => v - m)).slice(settle);
    const acdc = stdp(y); if (acdc < 1e-5) continue;
    const e = estimateHr(y, fs, PROBE_GATE);
    // Short probes rank cameras, never authorize an HR reading. Correct ACF's
    // overlap bias so a 1.5 s clean record isn't punished for containing few beats.
    if (body.length <= 3 * fs && Number.isFinite(e.hrAcf)) { const lag = fs * 60 / e.hrAcf; e.acf = Math.min(1, e.acf * body.length / Math.max(1, body.length - lag)); }
    // A 2.5 s window is too short for hr.js's absolute gate: band-passed noise already reaches sqi ~0.3 and acf ~0.4, a clean pulse sqi ~0.4 and acf ~0.75.
    // So the quality is measured above that noise floor, plus the spectral and autocorrelation heart rates must agree (hr.js AGREE).
    const agree = Number.isFinite(e.hr) && Number.isFinite(e.hrAcf) && e.hr > 0 && Math.abs(e.hrAcf - e.hr) / e.hr < GATE.AGREE ? 1 : 0;
    let q = 0.45 * clamp01((e.acf - NOISE_ACF) / (CLEAN_ACF - NOISE_ACF)) + 0.35 * clamp01((e.sqi - NOISE_SQI) / (CLEAN_SQI - NOISE_SQI)) + 0.2 * agree;
    if (body.length <= 3 * fs) {
      const fit = shortBandFit(body.map((p) => Math.log(p[c] + 1)), fs);
      q = clamp01((fit.quality - 0.45) / 0.5);
      e.sqi = fit.quality; e.hr = fit.hz * 60;
    }
    if (!best || q > best.q) best = { c, acdc, e, q, sat };
  }
  if (!best) { res.reason = Math.max(res.sat, ...res.channelSat) > SAT_MAX ? 'saturated' : 'flat'; return res; }
  res.sat = best.sat; res.factor = satFactor(best.sat);
  const amp = clamp01((Math.log10(best.acdc) - Math.log10(2e-4)) / (Math.log10(1e-2) - Math.log10(2e-4)));
  Object.assign(res, { acdc: best.acdc, sqi: best.e.sqi, acf: best.e.acf, snr: best.e.sqi / (1 - best.e.sqi + 1e-9), channel: best.c, hr: best.e.hr });
  res.score = 100 * res.factor * best.q * (0.4 + 0.6 * amp);
  res.reason = res.factor < 1 ? 'saturated' : (res.score < SCORE_MIN ? 'no_pulse' : 'ok');
  return res;
}

/**
 * Probe one lens: open it, collect (SETTLE_SEC + PROBE_SEC) of 30 Hz samples, score them, and ALWAYS release the stream before returning.
 * target = { id, label, mode: 'torch'|'ambient'|'front' }. Never throws: failures come back as { ok:false, error }.
 */
export async function probeLens(camFactory, target, opts = {}) {
  const sec = opts.sec ?? PROBE_SEC, settle = opts.settleSec ?? SETTLE_SEC;
  const need = Math.ceil((sec + settle) * FS), settleN = Math.ceil(settle * FS);
  const out = { deviceId: target.id ?? null, label: target.label || '', mode: target.mode || 'torch', ok: false, error: '', fallback: '', torchSupported: false, torchOn: false, samples: 0, frames: 0, fps: 0, startupMs: 0, score: 0, usable: false, reason: '' };
  const buf = []; let satSum = 0, satN = 0, cam = null, timer = null, poll = null, aborted = false, noFrames = false, done;
  const openedAt = performance.now(); let firstAt = null;
  const abortNow = () => !!(opts.shouldAbort && opts.shouldAbort());
  const finished = new Promise(r => { done = r; });
  const sampleDeadline = () => { clearTimeout(timer); timer = setTimeout(done, opts.timeoutMs ?? (sec + settle) * 1000 + 4000); };
  try {
    // The acquisition deadline starts with the first delivered sample, never getUserMedia/play().
    // Warm-up samples are discarded by scoreSignal; the body still contains >=2.5 s at 30 Hz.
    timer = setTimeout(() => { noFrames = true; done(); }, opts.firstFrameTimeoutMs ?? 8000);
    cam = camFactory((r, g, b) => {
      if (firstAt === null) { firstAt = performance.now(); out.startupMs = firstAt - openedAt; sampleDeadline(); }
      if (buf.length >= need) return;
      buf.push([r, g, b]);
      if (buf.length > settleN && cam && Number.isFinite(cam.satFrac)) { satSum += cam.satFrac; satN++; }
      if (buf.length >= need) done();
    });
    if (opts.shouldAbort) poll = setInterval(() => { if (abortNow()) { aborted = true; done(); } }, 100);
    const wantTorch = out.mode === 'torch';
    await cam.start({ deviceId: target.id || null, facing: out.mode === 'front' ? 'user' : 'environment', torch: wantTorch ? 'on' : 'off', firstFrameTimeoutMs: opts.firstFrameTimeoutMs ?? 8000, shouldAbort: opts.shouldAbort });
    out.torchSupported = !!cam.info?.torchSupported; out.torchOn = !!cam.info?.torch; out.torchState = cam.info?.torchState || (out.torchOn ? 'on-unverified' : 'off');
    if (!out.label && cam.info?.lens) out.label = cam.info.lens;
    if (cam.info?.fallback) { out.fallback = cam.info.fallback; if (cam.info.deviceId) out.deviceId = cam.info.deviceId; }
    if (abortNow()) { aborted = true; done(); }
    await finished;
    out.samples = buf.length; out.frames = cam.stats?.frames ?? buf.length; out.fps = cam.stats?.fps || 0;
    if (!out.fps && firstAt !== null && buf.length > 1) out.fps = (buf.length - 1) * 1000 / Math.max(1, performance.now() - firstAt);
    if (aborted) { out.error = 'cancelled'; out.reason = 'cancelled'; return out; }
    if (noFrames || !buf.length) { out.error = 'no_frames'; out.reason = 'no_frames'; return out; }
    if (buf.length < need) { out.reason = 'too_short'; out.error = 'incomplete_capture'; return out; }
    const measured = scoreSignal(buf, FS, { settleSec: settle, satFrac: satN ? satSum / satN : 0 });
    Object.assign(out, measured, { samples: buf.length, ok: true, reason: measured.reason });
    if (wantTorch && !out.torchOn) { out.error = out.torchSupported ? 'torch_apply_failed' : 'torch_unsupported'; out.reason = out.error; out.score = 0; }
    out.usable = out.score >= (opts.scoreMin ?? SCORE_MIN);
  } catch (e) {
    out.samples = buf.length; out.frames = cam?.stats?.frames ?? 0; out.fps = cam?.stats?.fps || 0;
    out.error = `${e?.name || 'error'}: ${e?.message || e}`;
    out.reason = e?.code === 'no_frames' || e?.name === 'FrameStartupTimeout' || (!out.samples && noFrames) ? 'no_frames' : 'open_failed';
  } finally {
    clearTimeout(timer); if (poll) clearInterval(poll);
    if (firstAt === null) out.startupMs = performance.now() - openedAt;
    out.torchSupported = !!cam?.info?.torchSupported; out.torchOn = !!cam?.info?.torch;
    out.torchState = cam?.info?.torchState || (out.torchOn ? 'on-unverified' : 'off');
    if (cam) { try { await cam.stop(); } catch (e) { /* ignore */ } }
    if (cam) await new Promise(r => setTimeout(r, opts.gapMs ?? 200));
  }
  return out;
}

/** Probe several targets one after another (never two streams open at once). Returns results sorted by score, best first. */
export async function probeAll(camFactory, targets, opts = {}) {
  const res = [];
  for (const t of targets) { if (opts.shouldAbort && opts.shouldAbort()) break; res.push(await probeLens(camFactory, t, opts)); if (opts.onProgress) opts.onProgress(res[res.length - 1], res.length, targets.length); }
  return res.sort((a, b) => b.score - a.score);
}

export const isVirtualCamera = label => /dual|triple|virtual|fusion|logical|multi.?cam/i.test(label || '');

const bestOf = (rs, min) => rs.filter((r) => r.ok && r.score >= min).sort((a, b) => b.score - a.score)[0] || null;

/**
 * Decide the measurement mode from probe results (each has .mode 'torch'|'ambient'|'front', .score, .error).
 * 1. best back lens under torch if it reaches minScore;
 * 2. otherwise, if the torch itself failed (unsupported / apply error), ambient-light fingertip on the best ambient lens, then the front camera;
 * 3. if the torch worked but no back lens produced a pulse (odd layout, lens too big to seal), the front camera with a white screen first, then ambient.
 * Returns { mode, deviceId, label, score, usable, reason, scores }. usable=false means nothing reached minScore (best effort returned).
 */
export function chooseMode(results, ctx = {}) {
  const min = ctx.minScore ?? SCORE_MIN, by = (m) => results.filter((r) => r.mode === m);
  const torch = by('torch'), amb = by('ambient'), front = by('front');
  const scores = results.map((r) => ({ mode: r.mode, deviceId: r.deviceId, label: r.label, score: Math.round(r.score * 10) / 10, acdc: r.acdc, sqi: r.sqi, acf: r.acf, snr: r.snr, sat: r.sat, reason: r.reason || r.error, fallback: r.fallback || '' }));
  const pack = (r, mode, reason, usable = true) => ({ mode, deviceId: r ? r.deviceId : null, label: r ? r.label : '', score: r ? r.score : 0, usable, reason, scores });
  const physical = torch.filter((r) => !/dual|triple|virtual|fusion|logical|multi.?cam/i.test(r.label || ''));
  const t = bestOf(physical, min) || bestOf(torch, min);
  if (t) return pack(t, 'torch', `Back lens "${t.label || t.deviceId}" had the strongest pulse under the torch (score ${t.score.toFixed(0)}).`);
  const torchWorked = torch.some((r) => r.ok && r.torchOn);
  const why = !torch.length ? 'The torch path was not probed.'
    : torchWorked ? `No back lens reached a pulse score of ${min} under the torch (best ${Math.max(...torch.map((r) => r.score)).toFixed(0)}).`
    : `The torch could not be used (${[...new Set(torch.map((r) => r.error || r.reason))].join(', ') || 'unknown'}).`;
  const a = bestOf(amb, min), f = bestOf(front, min);
  const ambR = a && pack(a, 'ambient', `${why} Using ambient light on "${a.label || a.deviceId}" (score ${a.score.toFixed(0)}).`);
  const frR = f && pack(f, 'front', `${why} Using the front camera with a full-white screen as the light (score ${f.score.toFixed(0)}).`);
  const order = torchWorked ? [frR, ambR] : [ambR, frR];
  for (const o of order) if (o) return o;
  const all = [...results].sort((x, y) => y.score - x.score)[0] || null;
  return pack(all, all ? all.mode : 'front', `${why} No mode reached a pulse score of ${min}; the best effort is shown with a reposition prompt.`, false);
}

/**
 * Whole ladder. targets = { back: [{id,label}], front: true|false }. Stops at the first stage that yields a usable mode.
 * opts.beforeFront / afterFront: hooks to switch the full-white screen on/off around the front-camera probe.
 */
export async function runProbeLadder(camFactory, targets, opts = {}) {
  const back = (targets.back || []).filter(c => !isVirtualCamera(c.label)), results = [];
  results.push(...await probeAll(camFactory, back.map((c) => ({ ...c, mode: 'torch' })), opts));
  const cancelled = () => !!(opts.shouldAbort && opts.shouldAbort());
  let choice = chooseMode(results, opts);
  if (choice.usable || cancelled()) return { choice, results, cancelled: cancelled() };
  // An empty acquisition says nothing about the fingertip. Keep a default rear
  // stream alive instead of spending more time on empty front/ambient probes.
  if (!results.length || results.some(r => r.reason === 'no_frames' || r.reason === 'too_short')) {
    const camera = back[0] || null;
    choice = { mode: 'torch', deviceId: camera?.id || null, label: camera?.label || 'Default back camera', score: 0, usable: false, continuous: true, reason: 'Probe did not receive enough frames. Continuing on the default back camera with the light on; waiting for live frames.', scores: choice.scores };
    return { choice, results, cancelled: false };
  }

  const torchWorked = results.some((r) => r.ok && r.torchOn);
  const ambTargets = (torchWorked ? [...results].sort((a, b) => b.score - a.score).map((r) => ({ id: r.deviceId, label: r.label })) : back).slice(0, opts.ambientMax ?? 2);
  const doAmbient = async () => { if (ambTargets.length && !cancelled()) results.push(...await probeAll(camFactory, ambTargets.map((c) => ({ ...c, mode: 'ambient' })), opts)); };
  const doFront = async () => {
    if (!targets.front || cancelled()) return;
    if (opts.beforeFront) await opts.beforeFront();
    if (cancelled()) { if (opts.afterFront) await opts.afterFront(); return; }
    try { results.push(...await probeAll(camFactory, [{ id: null, label: 'Front camera', mode: 'front' }], opts)); } finally { if (opts.afterFront) await opts.afterFront(); }
  };
  if (torchWorked) { await doFront(); choice = chooseMode(results, opts); if (!choice.usable) { await doAmbient(); choice = chooseMode(results, opts); } }
  else { await doAmbient(); choice = chooseMode(results, opts); if (!choice.usable) { await doFront(); choice = chooseMode(results, opts); } }
  return { choice, results, cancelled: cancelled() };
}

/** Live signal-quality verdict for the on-screen meter, from the last ~6-10 s of 30 Hz RGB. level: 'good' | 'weak' | 'reposition'. */
export function liveQuality(rgb, fs = FS) {
  const s = scoreSignal(rgb, fs, { settleSec: 0 });
  const prompts = {
    too_short: 'Hold still, measuring the signal...',
    dark: 'No light is coming through. Lay your finger flat over the lens AND the flash, with light pressure.',
    flat: 'No signal. Lay your finger flat over the lens AND the flash, with light pressure.',
    saturated: 'Too bright or pressing too hard. Ease the pressure slightly and keep the lens covered.',
    no_pulse: 'No clear pulse. Re-position: cover the lens and the flash with the pad of your finger, lighter pressure, hold still.',
    ok: '',
  };
  const level = s.reason === 'too_short' ? 'weak' : s.score >= 55 ? 'good' : s.score >= SCORE_MIN ? 'weak' : 'reposition';
  return { ...s, level, prompt: level === 'good' ? '' : (prompts[s.reason] || prompts.no_pulse) };
}

/** Preview ring state from one frame's mean RGB: brightness 0..1 and redness (red share of total) 0..1, plus whether it looks like a covered lens. */
export function ringState(mean3, satFrac = 0) {
  const [r, g, b] = mean3, tot = r + g + b + 1e-9;
  const brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255, redness = r / tot;
  const covered = redness > 0.5 && Math.max(r, g, b) > DARK_LEVEL && satFrac <= SAT_MAX;
  return { brightness, redness, covered };
}

/** Human readable per-lens probe table for the debug view. */
export function probeSummary(results, choice) {
  const rows = results.map((r) => `${r.mode.padEnd(7)} ${(r.label || r.deviceId || '?').slice(0, 28).padEnd(28)} samples ${r.samples || 0} frames ${r.frames || 0} fps ${(r.fps || 0).toFixed(1)} torch ${r.torchState || (r.torchOn ? 'on-unverified' : 'off')} startup ${((r.startupMs || 0) / 1000).toFixed(1)}s score ${r.score.toFixed(1).padStart(5)}  acdc ${(r.acdc * 100).toFixed(2)}%  sqi ${(r.sqi || 0).toFixed(2)}  acf ${(r.acf || 0).toFixed(2)}  sat ${((r.sat || 0) * 100).toFixed(0)}%  ${r.reason || r.error || ''}${r.fallback ? '  [' + r.fallback + ']' : ''}`);
  return [...rows, choice ? `-> ${choice.mode} ${choice.label || ''}${choice.usable ? '' : ' (nothing reached threshold)'}: ${choice.reason}` : ''].join('\n');
}
