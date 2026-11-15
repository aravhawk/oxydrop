import { Chain, FS, CAL, WIN } from './engine.js';
import { CameraSource, Motion, listBackCameras, pickDefault, listCameras, probeCamFactory } from './camera.js';
import { runProbeLadder, liveQuality, ringState, probeSummary } from './lensprobe.js';
import { MIN_TRACKABLE_BPM } from './hr.js';
import { decide, hrToShow, assessPlacement, stalledDecision, gapContext, STATE, STATE_TEXT, MOTION_MAX, STALL_MS } from './decision.js';

const $ = (id) => document.getElementById(id);
const MIN_HR = MIN_TRACKABLE_BPM, MAX_HR = 110; // plausible *resting* heart rate at calibration; MIN_HR follows the HR estimator's band edge (F_LO = 0.85 Hz = 51 bpm)
const LOCK_NEED = 3;            // consecutive valid-lock windows before calibration can start
const ANYWAY_AFTER_S = 20;      // seconds without a lock before "start anyway" is offered
const CAL_VALID_MIN = 0.6;      // fraction of calibration windows that must have a valid pulse lock
const CAL_FPS_MIN = 20, CAL_GAPS_MAX = 20, CAL_GAP_S_MAX = 1.5;   // a single frame stall longer than 1.5 s anywhere in the calibration minute rejects it (as in v1)
const LOCK_AFTER_MS = 3000;     // fingertip placement must read "good" this long before exposure/white balance/focus are locked
const UNLOCK_AFTER_MS = 2000;   // in setup, lift the lock again if the placement has been bad this long (the next lock is taken on the new scene)
const CAL_WARM_S = 20;          // windows starting at s <= 20 still contain the band-pass filter's start-up transient (always invalid, even on a clean signal), so they are not counted
const PROBE_MAX_LENSES = 4;     // back lenses probed (about 3.7 s each, so the torch probe stays under ~15 s)
const GET_READY_S = 3;          // countdown so the finger is on the plateau before the first probe
const WEAK_PROMPT_MS = 4000;    // live signal quality must stay in 'reposition' this long before the re-position prompt replaces a reading
const WEAK_MSG = 'Weak signal: slide your finger so it covers the lens and the flash, press lightly.';
const RAW_MAX = 15 * 60 * FS;   // raw trace ring buffer: 15 minutes at 30 Hz

let mode = null;                // null | 'setup' | 'measure'
let kind = null;                // 'live' | 'replay'
let chain = null, weights = null, cam = null, motion = null, wake = null, camGen = 0;
let replay = null, replayTimer = null, uiTimer = null, anywayTimer = null, tickN = 0;
let strip = [], tally = {}, calMotion = [], calFps = [], calWin = [], calGaps0 = 0, calFailed = false, calChecked = false, calOk = true;
let lockHist = [], lockRun = 0, camStartedAt = 0, place = { level: 'warn', title: '', msg: '' }, fingerHist = [], gapLog = [], lastGaps = 0, lastD = null, lastDecision = null, lastWaveCh = 1;
let probe = null, qual = null, weakSince = 0, weakNow = false, weakShown = false, chosenLabel = '';
let goodSince = 0, badSince = 0, lockBusy = false, stalledShown = false, calStartN = 0;
const raw = new Float32Array(RAW_MAX * 3); let rawN = 0, decLog = [];

function show(id) { document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === id)); window.scrollTo(0, 0); }
document.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => { cleanup(); show(b.dataset.go); }));
$('ack').addEventListener('change', (e) => { $('btn-live').disabled = !e.target.checked; });
const hasCam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
$('compat').textContent = hasCam ? '' : 'Camera access is not available in this browser or context (needs HTTPS). The recorded demo still works.';
if (!hasCam) { $('btn-live').disabled = true; $('ack').disabled = true; }
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

async function loadWeights() { if (!weights) weights = await (await fetch('data/weights.json')).json(); return weights; }

function cleanup() {
  camGen++; clearInterval(replayTimer); clearInterval(uiTimer); clearTimeout(anywayTimer); uiTimer = null;
  if (cam) { cam.stop(); cam = null; } if (motion) { motion.stop(); motion = null; }
  if (wake) { try { wake.release(); } catch (e) { /* ignore */ } wake = null; }
  hideWhite(); weakSince = 0; weakNow = false; weakShown = false;
  replay = null; mode = null; kind = null; chain = null;
}
const isFront = () => lightMode() === 'front';
function showWhite() { $('white').hidden = false; }
function hideWhite() { $('white').hidden = true; }
$('wm-stop').addEventListener('click', () => { cleanup(); show('home'); });
$('wm-begin').addEventListener('click', () => { if (mode === 'setup') startMeasure('live'); else if (mode === 'measure') startMeasure('live'); });
async function keepAwake() { try { if (navigator.wakeLock) wake = await navigator.wakeLock.request('screen'); } catch (e) { /* ignore */ } }
function startUi() { clearInterval(uiTimer); tickN = 0; uiTimer = setInterval(uiTick, 250); }

// ---------------- raw trace recorder (bounded ring buffer) ----------------
function resetRecorder() { rawN = 0; decLog = []; lastD = null; lastDecision = null; }
function record(r, g, b) { const i = (rawN % RAW_MAX) * 3; raw[i] = r; raw[i + 1] = g; raw[i + 2] = b; rawN++; }
function logDecision(d, dec) { decLog.push({ idx: rawN - 1, d, dec }); if (decLog.length > RAW_MAX / (2 * FS) + 5) decLog.shift(); }
function buildCsv() {
  const start = Math.max(0, rawN - RAW_MAX), byIdx = new Map(decLog.map((x) => [x.idx, x]));
  const rows = ['t_s,R,G,B,sqi,acf,hr_est,hr_beat,valid,sat,label,p,state,reason,hr_shown'];
  for (let n = start; n < rawN; n++) {
    const i = (n % RAW_MAX) * 3; let tail = ',,,,,,,,,,';
    const x = byIdx.get(n);
    if (x) { const d = x.d, c = x.dec; tail = [f4(d.sqi), f4(d.acf), f2(d.hrEst), f2(d.hrBeat), d.valid ? 1 : 0, f4(d.sat), d.label || (d.calibrating ? 'calibrating' : ''), f4(d.p), c ? c.state : '', c && c.reason ? c.reason : '', c && c.hr != null ? c.hr.toFixed(1) : ''].join(','); }
    rows.push(`${(n / FS).toFixed(3)},${raw[i].toFixed(3)},${raw[i + 1].toFixed(3)},${raw[i + 2].toFixed(3)},${tail}`);
  }
  return rows.join('\n') + '\n';
}
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : ''), f4 = (v) => (Number.isFinite(v) ? v.toFixed(4) : '');
function download() {
  const url = URL.createObjectURL(new Blob([buildCsv()], { type: 'text/csv' }));
  const a = document.createElement('a'); a.href = url; a.download = `oxydrop-trace-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 5000);
}
async function copyDebug(btn) {
  const txt = debugText(); const old = btn.textContent;
  try { await navigator.clipboard.writeText(txt); btn.textContent = 'Copied'; }
  catch (e) {
    const ta = document.createElement('textarea'); ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select();
    let okc = false; try { okc = document.execCommand('copy'); } catch (e2) { /* ignore */ } ta.remove(); btn.textContent = okc ? 'Copied' : 'Copy failed: select the text above';
  }
  setTimeout(() => { btn.textContent = old; }, 2000);
}
document.querySelectorAll('[data-dl]').forEach((b) => b.addEventListener('click', download));
document.querySelectorAll('[data-copy]').forEach((b) => b.addEventListener('click', () => copyDebug(b)));

function debugText() {
  const L = []; const st = cam ? cam.stats : null;
  L.push(`mode ${mode || '-'} / ${kind || '-'}   samples ${rawN}   t ${(rawN / FS).toFixed(1)} s`);
  if (cam && st) {
    L.push(`fps ${st.fps.toFixed(1)}  jitter ${st.jitterMs.toFixed(1)} ms  gaps ${st.gaps} (max ${(st.maxGap * 1000).toFixed(0)} ms)  skipped ${st.skipped}  frames ${st.frames}`);
    L.push(`last frame ${(st.staleMs / 1000).toFixed(1)} s ago  interrupted ${st.interrupted || 'no'}  longest gap in last 30 s ${(cam.rs.gapStats(cam.rs.n - 30 * FS).max * 1000).toFixed(0)} ms`);
    L.push(`lens "${cam.info.lens}"  ${cam.info.width}x${cam.info.height}`);
    L.push(`torch ${cam.info.torchSupported ? (cam.info.torch ? 'on' : 'off') : 'unsupported'}  locks ${JSON.stringify(cam.info.locks)}`);
    L.push(`mean RGB ${st.mean.map((v) => v.toFixed(1)).join(' / ')}  frame-saturated ${(st.satFrac * 100).toFixed(1)}%  spatialCV ${st.spatialCV.toFixed(3)}  luma ${st.luma.toFixed(1)}`);
    L.push(`placement ${place.level}: ${place.title}`);
    if (cam.info.adapt) L.push(`adaptive exposure: ${cam.info.adapt.result}  luma ${f2(cam.info.adapt.lumaBefore)} -> ${f2(cam.info.adapt.lumaAfter)}  sat ${f4(cam.info.adapt.satAfter)}  applied ${JSON.stringify(cam.info.adapt.applied)}  supported ${(cam.info.adapt.supported || []).join(',') || 'none'}`);
  }
  L.push(`light mode ${lightMode()}  chosen lens "${chosenLabel || '-'}"${qual ? `  live quality ${qual.level} (score ${qual.score.toFixed(0)}, acdc ${(qual.acdc * 100).toFixed(2)}%, sqi ${f2(qual.sqi)})` : ''}${weakNow ? '  REPOSITION PROMPT SHOWN' : ''}`);
  if (probe) { L.push('lens probe scores (per lens, 0-100, usable >= 35):'); L.push(probeSummary(probe.results, probe.choice)); } else L.push('lens probe: not run');
  if (lastD) {
    const d = lastD; L.push(`window s=${d.s}  SQI ${f4(d.sqi)}  ACF ${f4(d.acf)}  hrEst ${f2(d.hrEst)}  hrBeat ${f2(d.hrBeat)}  ch ${d.ch}  sat ${f4(d.sat)}  valid ${d.valid}  label ${d.label || 'calibrating'}  p ${f4(d.p)}`);
    if (lastDecision) L.push(`decision ${lastDecision.state}${lastDecision.reason ? ' (' + lastDecision.reason + ')' : ''}  hr shown ${lastDecision.hr != null ? lastDecision.hr.toFixed(1) : '-'}`);
  } else L.push('no window yet');
  L.push(`lock history (last 5): ${lockHist.map((v) => (v ? '1' : '0')).join('') || '-'}  motion ${motion && motion.ok && motion.level(10) != null ? motion.level(10).toFixed(2) + ' m/s2' : 'no sensor'}`);
  L.push(navigator.userAgent);
  return L.join('\n');
}

// ---------------- live camera flow ----------------
const lightMode = () => (document.querySelector('input[name=light]:checked') || { value: 'flash' }).value;

$('btn-live').addEventListener('click', () => { cleanup(); show('setup'); $('setup-live').hidden = true; $('btn-cam').hidden = false; $('cam-state').textContent = 'Camera off'; $('checks').innerHTML = ''; $('probe').hidden = true; probe = null; qual = null; chosenLabel = ''; });

$('btn-cam').addEventListener('click', async () => {
  $('btn-cam').hidden = true; $('cam-state').textContent = 'Starting camera…';
  try {
    cleanup(); motion = new Motion(); const mp = motion.start();   // start the motion permission prompt first: iOS needs it inside the tap
    await loadWeights(); await keepAwake();
    kind = 'live'; mode = 'setup'; startUi();
    const gen = camGen, choice = await runProbe(gen);
    if (gen !== camGen) return;
    const lm = choice.mode === 'torch' ? 'flash' : choice.mode;
    const radio = document.querySelector(`input[name=light][value=${lm}]`); if (radio) radio.checked = true;
    chosenLabel = choice.label || '';
    await startCamera(choice.mode === 'front' ? null : choice.deviceId);
    if (gen !== camGen) return;
    await populateLens(); if (choice.mode !== 'front' && choice.deviceId) $('lens').value = choice.deviceId;
    await mp;
    $('setup-live').hidden = false; renderChecks(); renderPlace();
  } catch (e) {
    cleanup(); $('cam-state').textContent = 'Could not start the camera: ' + (e && e.message ? e.message : e) + '. Allow camera access and reload, or use the recorded demo.';
    $('btn-cam').hidden = false; $('setup-live').hidden = true; $('probe').hidden = true;
  }
});

// Lens probe: every back lens for ~3 s under the torch, best pulse wins; if none works, ambient light or the front camera + white screen (lensprobe.js).
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function runProbe(gen) {
  $('probe').hidden = false; $('probe-scores').textContent = ''; probe = null; qual = null;
  const gum = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });   // one grant first: device labels are empty until the camera is allowed
  gum.getTracks().forEach((t) => t.stop());
  const all = await listCameras(), pd = pickDefault(all.back), back = [...all.back].sort((a, b) => (b === pd) - (a === pd)).slice(0, PROBE_MAX_LENSES);
  const say = (t) => { $('probe-text').textContent = t; $('cam-state').textContent = t; if (!$('white').hidden) { $('wm-title').textContent = t; $('wm-msg').textContent = ''; } };
  for (let k = GET_READY_S; k > 0; k--) { say(`Lay your finger flat across the whole camera plateau now (lens and flash), light pressure. Testing starts in ${k}…`); await sleep(1000); if (gen !== camGen) return null; }
  const video = document.createElement('video'), base = probeCamFactory(video), partial = [];
  const factory = (cb) => { if (gen !== camGen) throw new Error('cancelled'); return base(cb); };
  say(`Testing lens 1 of ${back.length}, keep your finger flat across the camera plateau`);
  const out = await runProbeLadder(factory, { back, front: all.front }, {
    ambientMax: 1, shouldAbort: () => gen !== camGen,
    onProgress: (r, k) => {
      partial.push(r); $('probe-scores').textContent = probeSummary(partial, null);
      say(k < back.length ? `Testing lens ${k + 1} of ${back.length}, keep your finger flat across the camera plateau` : 'Back lenses gave no clear pulse. Trying the fallback mode (ambient light, then front camera with white screen)…');
    },
    beforeFront: async () => { if (gen !== camGen) return; showWhite(); say('Testing the front camera. Cover it with your fingertip.'); await sleep(1200); },
    afterFront: async () => { if (gen === camGen) hideWhite(); },   // a stale probe must not hide a newer session's overlay
  });
  if (gen !== camGen) return null;
  probe = out; $('probe-scores').textContent = probeSummary(out.results, out.choice);
  $('probe-text').textContent = out.choice.reason + (out.choice.usable ? '' : ' Re-position your finger and try again, or pick a mode below.');
  return out.choice;
}

async function startCamera(deviceId) {
  const gen = ++camGen;
  if (cam) { cam.stop(); cam = null; }
  chain = new Chain(weights); resetRecorder(); lockHist = []; lockRun = 0; fingerHist = []; gapLog = []; lastGaps = 0; goodSince = 0; badSince = 0; lockBusy = false; stalledShown = false; place = { level: 'warn', title: 'Looking for your fingertip…', msg: '' };
  $('btn-begin').disabled = true; $('btn-begin-anyway').hidden = true; $('q-fill').style.width = '0%'; $('q-text').textContent = 'Signal quality: waiting for a pulse lock';
  const c = new CameraSource((r, g, b) => { if (gen === camGen) sink(r, g, b); });
  cam = c; weakSince = 0; weakNow = false; weakShown = false; qual = null;
  const front = isFront(); if (front) showWhite(); else hideWhite();
  await c.start(document.createElement('video'), { deviceId: front ? null : deviceId, torch: lightMode() === 'flash' ? 'auto' : 'off', facing: front ? 'user' : 'environment' });
  camStartedAt = performance.now();
  clearTimeout(anywayTimer); anywayTimer = setTimeout(() => { if (mode === 'setup' && $('btn-begin').disabled) $('btn-begin-anyway').hidden = false; }, ANYWAY_AFTER_S * 1000);
  $('cam-state').textContent = front ? 'Front camera on, white screen is the light' : c.info.torch ? 'Camera on, flash on' : (lightMode() === 'ambient' ? 'Camera on, flash off (ambient light)' : 'Camera on (flash control not available in this browser)');
}
async function populateLens() {
  const cams = await listBackCameras(), sel = $('lens'); sel.innerHTML = '';
  for (const c of cams) { const o = document.createElement('option'); o.value = c.id; o.textContent = c.label; sel.appendChild(o); }
  if (cam && cam.info.deviceId && cams.some((c) => c.id === cam.info.deviceId)) sel.value = cam.info.deviceId;
  $('lens-help').textContent = cams.length > 1 ? 'The app picked the lens with the strongest pulse (see the probe scores in the debug view). You can override it here.' : 'Only one back camera was found.';
  return cams;
}
$('lens').addEventListener('change', async () => {
  if (mode !== 'setup' || !cam) return;
  if (isFront()) document.querySelector('input[name=light][value=flash]').checked = true;
  chosenLabel = $('lens').selectedOptions[0] ? $('lens').selectedOptions[0].textContent : '';
  try { await startCamera($('lens').value); renderChecks(); } catch (e) { $('cam-state').textContent = 'Could not switch lens: ' + (e && e.message ? e.message : e); }
});
document.querySelectorAll('input[name=light]').forEach((r) => r.addEventListener('change', async () => {
  if (mode !== 'setup' || !cam) return;
  const front = isFront(), wasFront = cam.info.facing === 'user';
  if (front || wasFront) {   // the front camera is a different device: restart on it (or back on the chosen back lens)
    try { chosenLabel = front ? 'Front camera' : ($('lens').selectedOptions[0] ? $('lens').selectedOptions[0].textContent : ''); await startCamera(front ? null : $('lens').value || null); renderChecks(); }
    catch (e) { $('cam-state').textContent = 'Could not switch mode: ' + (e && e.message ? e.message : e); }
    return;
  }
  const on = lightMode() !== 'ambient'; await cam.setTorch(on); await cam.unlockExposure(); goodSince = 0; badSince = 0;   // the scene brightness just changed by a large factor: re-lock on the new scene
  chain = new Chain(weights); lockHist = []; lockRun = 0; $('btn-begin').disabled = true; camStartedAt = performance.now();
  clearTimeout(anywayTimer); $('btn-begin-anyway').hidden = true; anywayTimer = setTimeout(() => { if (mode === 'setup' && $('btn-begin').disabled) $('btn-begin-anyway').hidden = false; }, ANYWAY_AFTER_S * 1000);
  $('cam-state').textContent = on ? (cam.info.torch ? 'Camera on, flash on' : 'Camera on (flash control not available in this browser)') : 'Camera on, flash off (ambient light)';
  renderChecks();
}));

// every resampled 30 Hz camera/replay sample goes through here
function sink(r, g, b) {
  record(r, g, b);
  if (!chain) return;
  const d = chain.push(r, g, b);
  if (d) { if (mode === 'setup') onSetupWindow(d); else if (mode === 'measure') onWindow(d); }
}

function liveCtx() {
  if (kind !== 'live' || !cam) return { fps: 30, recentGaps: 0, finger: true, moving: false, calOk };
  const st = cam.stats, m = motion && motion.level(10);
  const finger = fingerHist.length < 4 ? place.level === 'good' : fingerHist.filter(Boolean).length / fingerHist.length >= 0.8;
  return { fps: st.fps > 0 ? st.fps : 30, recentGaps: gapLog.reduce((a, x) => a + x[1], 0), ...gapContext(cam.rs, FS), stalledMs: camStalledMs(st), finger, moving: m != null && m > MOTION_MAX, calOk };
}

// wall-clock stall: no frame for STALL_MS, or the track ended / was muted / the page was hidden (reported as an arbitrarily long stall)
function camStalledMs(st) { return st.interrupted ? 1e9 : st.staleMs; }
function stalledNow() { return kind === 'live' && cam ? camStalledMs(cam.stats) > STALL_MS : false; }
async function lockWithAdapt(c) { try { await c.tryAdaptExposure(); } catch (e) { /* adaptive exposure is best effort */ } await c.lockExposure(); }
function resetSetupChain() { chain = new Chain(weights); lockHist = []; lockRun = 0; $('btn-begin').disabled = true; }

function onSetupWindow(d) {
  lastD = d; lastDecision = null; lastWaveCh = d.ch;
  logDecision(d, null);
  lockHist.push(!!d.valid); if (lockHist.length > 5) lockHist.shift(); lockRun = d.valid ? lockRun + 1 : 0;
  const q = lockHist.filter(Boolean).length / 5, good = lockRun >= LOCK_NEED || q >= 0.8;
  $('btn-begin').disabled = lockRun < LOCK_NEED || weakNow; if (lockRun >= LOCK_NEED) $('btn-begin-anyway').hidden = true;
}
const WEAK_PLACE = { level: 'bad', title: 'Weak signal: re-position', msg: WEAK_MSG };
function renderPlace(p = place) { const el = $('place'); el.className = 'place ' + p.level; $('place-title').textContent = p.title; $('place-msg').textContent = p.msg; }
function renderChecks() {
  if (!cam) return; const st = cam.stats, li = [], lk = cam.info.locks || {}, ambient = lightMode() === 'ambient';
  if (ambient) li.push([true, 'Flash off (ambient light). Face a bright lamp or window.']);
  else li.push([cam.info.torch, cam.info.torch ? 'Flash on' : 'No flash control here (common on iPhone Safari). Use bright light; accuracy is unvalidated.']);
  const locked = ['exposure', 'wb', 'focus'].filter((k) => lk[k]), missed = ['exposure', 'wb', 'focus'].filter((k) => !lk[k]);
  const cp = cam.caps || {}, canLock = !!(cp.exposureMode || cp.whiteBalanceMode || cp.focusMode);
  if (!canLock) li.push([false, 'This browser cannot lock exposure, white balance or focus (common on iPhone Safari; can reduce accuracy)']);
  else if (!cam.locked) li.push([true, goodSince ? 'Hold still: locking exposure, white balance and focus…' : 'Exposure locks once your fingertip is steady on the lens']);
  else li.push([locked.includes('exposure') && locked.includes('wb'), locked.length ? `Locked: ${locked.join(', ')}` + (missed.length ? `. Not lockable here: ${missed.join(', ')}` : '') : 'Exposure and white balance could not be locked on this device (can reduce accuracy)']);
  const ad = cam.info.adapt; if (ad) li.push([ad.result === 'in_band' || ad.result === 'already_in_band', `Exposure for weak or diffuse light: ${ad.result.replace(/_/g, ' ')} (brightness ${Math.round(ad.lumaBefore)} to ${Math.round(ad.lumaAfter)})`]);
  li.push([true, `Mode: ${isFront() ? 'front camera + white screen' : ambient ? 'ambient light' : 'torch'}; lens: ${chosenLabel || cam.info.lens || '-'}`]);
  li.push([st.fps >= CAL_FPS_MIN, st.fps ? `Camera ${st.fps.toFixed(0)} fps, jitter ${st.jitterMs.toFixed(1)} ms` + (st.fps < CAL_FPS_MIN ? ' (too low)' : '') : 'Measuring camera speed…']);
  const g = gapLog.reduce((a, x) => a + x[1], 0);
  li.push([g === 0, g === 0 ? `No dropped frames (${st.gaps} total)` : `${g} dropped-frame gaps in the last 10 s`]);
  li.push([motion && motion.ok, motion && motion.ok ? 'Motion sensor on' : 'No motion sensor: hold very still yourself']);
  $('checks').innerHTML = li.map(([ok, t]) => `<li class="${ok ? 'ok' : 'warn'}">${ok ? '✓' : '!'} ${t}</li>`).join('');
}

$('btn-begin').addEventListener('click', () => startMeasure('live'));
$('btn-begin-anyway').addEventListener('click', () => startMeasure('live'));

// ---------------- replay flow ----------------
$('btn-replay').addEventListener('click', async () => {
  cleanup(); await loadWeights(); show('measure'); kind = 'replay'; resetRecorder();
  const pick = new URLSearchParams(location.search).get('rec') || (Math.random() < 0.5 ? '100001L' : '100004R');
  const [meta, buf] = await Promise.all([fetch(`data/replay/${pick}.json`).then((r) => r.json()), fetch(`data/replay/${pick}.bin`).then((r) => r.arrayBuffer())]);
  replay = { meta, rgb: new Float32Array(buf), i: 0, speed: 30 };
  startMeasure('replay', meta.weights);
  const note = $('replay-note'); note.hidden = false;
  note.innerHTML = `Recorded demo: study participant ${meta.person} (UW Hoffman 2022 data, MIT licence). The model used here was fitted <b>without</b> this person, so these are genuinely held-out decisions. Replaying at ${replay.speed}× speed.${meta.person === 4 ? ' <b>This is the hardest person in the study: a resting heart rate of 43 bpm is below the 51 bpm this version can follow, so it correctly answers &ldquo;unsure&rdquo; throughout.</b>' : ''}`;
  replayTimer = setInterval(() => {
    const per = Math.round(replay.speed * FS * 0.05);
    for (let k = 0; k < per && replay.i < replay.meta.n; k++, replay.i++) { const j = replay.i * 3; sink(replay.rgb[j], replay.rgb[j + 1], replay.rgb[j + 2]); }
    if (replay.i >= replay.meta.n) { clearInterval(replayTimer); $('phase').textContent = 'Replay finished'; }
  }, 50);
});

// ---------------- shared measurement logic ----------------
function startMeasure(k, w) {
  kind = k; mode = 'measure'; chain = new Chain(w || weights); chain.n = 0;
  stalledShown = false; calStartN = cam ? cam.rs.n : 0; if (k === 'live' && cam && !cam.locked && !lockBusy) { lockBusy = true; cam.lockExposure().then(() => { lockBusy = false; }, () => { lockBusy = false; }); }
  calMotion = []; calFps = []; calWin = []; calFailed = false; calChecked = false; calOk = true; strip = []; $('strip').innerHTML = ''; lastD = null; lastDecision = null;
  if (k === 'live') { calGaps0 = cam ? cam.stats.gaps : 0; decLog = []; }
  tally = { normal: 0, low: 0, uns: 0, pause: 0, right: 0, wrong: 0 };
  clearTimeout(anywayTimer); show('measure'); $('btn-restart').hidden = k !== 'live'; $('ref-tile').hidden = k !== 'replay'; if (k !== 'replay') $('replay-note').hidden = true;
  setStatus('calibrating', '◔', 'Calibrating', 'Rest, keep still. Learning your normal heart-rate and light baseline.');
  $('phase').textContent = 'Calibrating'; $('cal-progress').hidden = false; $('tally').textContent = '';
  $('hr').textContent = '–'; $('dhr').textContent = '–'; $('sig').textContent = '–'; $('sig-u').textContent = '';
  if (k !== 'live') { $('mot').textContent = '–'; $('mot-u').textContent = 'n/a in replay'; }
  startUi();
}
$('btn-restart').addEventListener('click', () => startMeasure('live'));
$('btn-stop').addEventListener('click', () => { cleanup(); show('home'); });

function uiTick() {
  tickN++;
  if (kind === 'live' && cam) {
    const st = cam.stats, now = performance.now();
    if (st.gaps > lastGaps) { gapLog.push([now, st.gaps - lastGaps]); lastGaps = st.gaps; }
    while (gapLog.length && now - gapLog[0][0] > 10000) gapLog.shift();
    place = assessPlacement(st, { torch: cam.info.torch });
    if (isFront() && place.level !== 'good' && place.title !== 'Looking for your fingertip…') place = { level: place.level, title: 'Cover the front camera completely', msg: 'Light pressure. The white screen must shine through your fingertip, and nothing else should cover the screen.' };
    const stalled = camStalledMs(st) > STALL_MS;
    if (stalled) place = { level: 'bad', title: 'Camera stalled', msg: st.interrupted || 'No camera frames are arriving. Keep the screen on and the page in front.' };
    // exposure / white balance / focus lock: taken on the covered lens once the placement has been good for LOCK_AFTER_MS
    if (place.level === 'good') { badSince = 0; if (!goodSince) goodSince = now; } else { goodSince = 0; if (!badSince) badSince = now; }
    if (!cam.locked && !lockBusy && goodSince && now - goodSince >= LOCK_AFTER_MS) { const c0 = cam; lockBusy = true; lockWithAdapt(c0).then(() => { lockBusy = false; if (cam === c0 && mode === 'setup') resetSetupChain(); }, () => { lockBusy = false; }); }
    else if (mode === 'setup' && cam.locked && !lockBusy && badSince && now - badSince >= UNLOCK_AFTER_MS) { const c0 = cam; lockBusy = true; c0.unlockExposure().then(() => { lockBusy = false; }, () => { lockBusy = false; }); }
    if (!stalled) { fingerHist.push(place.level === 'good'); if (fingerHist.length > 40) fingerHist.shift(); }   // a stall is not evidence about the finger
    updateQuality(st, now, stalled);
    if (mode === 'setup') { renderPlace(weakNow && !stalled ? WEAK_PLACE : place); if (tickN % 4 === 0) renderChecks(); if (stalled) { $('btn-begin').disabled = true; lockRun = 0; } }
    if (mode === 'measure') { watchStall(stalled, st); if (!stalled) watchWeak(); }
    updateRing(st); updateWhite();
    if (mode === 'measure') { const m = motion && motion.level(10); if (m != null) { $('mot').textContent = m < MOTION_MAX ? 'Still' : 'Moving'; $('mot-u').textContent = m.toFixed(2) + ' m/s²'; } else { $('mot').textContent = '?'; $('mot-u').textContent = 'no sensor'; } }
  }
  if (mode === 'setup' && chain) drawWave($('wave-setup'), chain.ac[lastWaveCh].slice(-300));
  if (mode === 'measure' && chain) {
    const s = chain.n / FS; $('clock').textContent = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; $('cal-fill').style.width = Math.min(100, (s / (CAL + WIN)) * 100) + '%';
    drawWave($('wave'), chain.ac[lastWaveCh].slice(-300));
  }
  const rawCv = mode === 'setup' ? $('raw-setup') : $('raw-measure'); drawRaw(rawCv);
  if (tickN % 2 === 0) { $(mode === 'setup' ? 'dbg-setup' : 'dbg-measure').textContent = debugText(); }
}

// Between windows no decision is produced, so a stalled camera would leave the last answer on screen. Force UNSURE and drop the heart-rate tile.
function watchStall(stalled, st) {
  if (stalled) {
    if (!stalledShown) { stalledShown = true; pushStrip('d-uns', 'unsure: camera_stalled'); }
    const dec = stalledDecision(); lastDecision = dec;
    if (calFailed) return;
    setStatus('uns', '?', dec.text, (st && st.interrupted ? st.interrupted + ' ' : '') + dec.detail); $('hr').textContent = '–'; $('dhr').textContent = '–'; $('sig').textContent = 'No signal'; $('sig-u').textContent = '';
  } else if (stalledShown) {
    stalledShown = false;
    if (!calFailed && chain && chain.n < (CAL + WIN) * FS) setStatus('calibrating', '◔', 'Calibrating', 'Rest, keep still. Learning your normal heart-rate and light baseline.');
  }
}

// Live signal-quality meter: AC/DC + spectral SNR + periodicity over the last ~8 s (lensprobe.liveQuality). A sustained 'reposition' verdict that the
// engine's own pulse lock does not contradict replaces the reading with a re-position prompt. The Chain math and decision thresholds are untouched.
function recentRgb(sec) {
  const n = Math.min(rawN, Math.round(sec * FS), RAW_MAX), rgb = new Array(n);
  for (let k = 0; k < n; k++) { const i = (((rawN - n + k) % RAW_MAX) * 3); rgb[k] = [raw[i], raw[i + 1], raw[i + 2]]; }
  return rgb;
}
function updateQuality(st, now, stalled) {
  if (stalled || rawN < 6 * FS) return;
  if (tickN % 4 === 0) {
    qual = liveQuality(recentRgb(8));
    const contradicted = !!(lastD && lastD.valid);
    if (qual.level === 'reposition' && !contradicted) { if (!weakSince) weakSince = now; } else weakSince = 0;
    weakNow = !!weakSince && now - weakSince >= WEAK_PROMPT_MS;
    const pct = Math.max(0, Math.min(100, qual.score));
    $('q-fill').style.width = pct + '%'; $('q-fill').style.background = qual.level === 'good' ? 'var(--green)' : qual.level === 'weak' ? 'var(--amber)' : 'var(--red)';
    $('q-text').textContent = weakNow ? WEAK_MSG : qual.level === 'good' ? `Good signal quality (score ${pct.toFixed(0)} of 100). Keep still.` : `Signal quality ${pct.toFixed(0)} of 100. ` + (qual.prompt || 'Keep still, holding the finger flat.');
  }
}
function watchWeak() {
  if (weakNow && !calFailed) {
    if (!weakShown) { weakShown = true; pushStrip('d-uns', 'unsure: weak_signal'); }
    setStatus('uns', '?', STATE_TEXT.UNSURE, WEAK_MSG); $('hr').textContent = '–'; $('dhr').textContent = '–'; $('sig').textContent = 'Weak'; $('sig-u').textContent = 're-position';
  } else if (weakShown) {
    weakShown = false;
    if (!calFailed && chain && chain.n < (CAL + WIN) * FS) setStatus('calibrating', '◔', 'Calibrating', 'Rest, keep still. Learning your normal heart-rate and light baseline.');
  }
}
// live preview ring: brightness and redness of the picked lens's image; green border when it looks covered by a fingertip
function updateRing(st) {
  const rs = ringState(st.mean, st.satFrac), [r, g, b] = st.mean, ring = $('ring');
  ring.style.background = `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`; ring.classList.toggle('on', rs.covered);
  $('ring-name').textContent = 'Cover this camera: ' + (isFront() ? 'Front camera' : (chosenLabel || cam.info.lens || '-'));
  $('ring-msg').textContent = `brightness ${Math.round(rs.brightness * 100)}%, redness ${Math.round(rs.redness * 100)}%` + (rs.covered ? ': covered' : ': not covered yet. Lay the finger flat over this lens and the flash.');
}
// the front-camera mode shows a full-white screen; this bar mirrors the placement / status text on top of it
function updateWhite() {
  if ($('white').hidden) return;
  if (mode === 'setup') { const p = weakNow ? WEAK_PLACE : place; $('wm-title').textContent = p.title; $('wm-msg').textContent = p.msg; }
  else if (mode === 'measure') { $('wm-title').textContent = $('status-main').textContent; $('wm-msg').textContent = $('status-sub').textContent + ($('hr').textContent !== '–' ? `  Heart rate ${$('hr').textContent} bpm.` : ''); }
  const canBegin = mode === 'setup' && (!$('btn-begin').disabled || !$('btn-begin-anyway').hidden), canRestart = mode === 'measure' && !$('btn-restart').hidden && kind === 'live' && calFailed;
  $('wm-begin').hidden = !(canBegin || canRestart); $('wm-begin').textContent = canRestart ? 'Restart calibration' : 'Start 60-second calibration';
}

function setStatus(cls, icon, main, sub) { const el = $('status'); el.className = 'status-card ' + cls; $('status-icon').textContent = icon; $('status-main').textContent = main; $('status-sub').textContent = sub; }
function pushStrip(cls, title) { strip.push([cls, title]); if (strip.length > 90) strip.shift(); $('strip').innerHTML = strip.map(([c, t]) => `<i class="${c}" title="${t}"></i>`).join(''); }

function refAt(s) { // window-mean reference SpO2 (study oximeters), replay only
  const g = replay.meta.spo2.slice(s, s + WIN).filter((v) => v != null); return g.length >= 5 ? g.reduce((a, b) => a + b, 0) / g.length : null;
}
function showHr(d, ctx) { const h = hrToShow(d, ctx); $('hr').textContent = h == null ? '–' : String(Math.round(h)); }

function onWindow(d) {
  lastD = d; lastWaveCh = d.ch; const ctx = liveCtx();
  if (d.calibrating) {
    calWin.push([d.s, !!d.valid]); if (kind === 'live') { calMotion.push(ctx.moving); calFps.push(ctx.fps); }
    logDecision(d, null); showHr(d, ctx); setSig(d); return;
  }
  if (!calChecked) {
    calChecked = true; const why = checkCalibration();
    if (why && kind === 'live') { calFailed = true; calOk = false; setStatus('fail', '✕', 'Unsure, re-measure', why + ' Rest for a couple of minutes, then restart calibration.'); $('phase').textContent = 'Restart needed'; $('cal-progress').hidden = true; $('btn-restart').hidden = false; }
    else { $('phase').textContent = why ? 'Measuring (no baseline)' : 'Measuring'; $('cal-progress').hidden = true; }
  }
  if (calFailed) { logDecision(d, null); showHr(d, ctx); return; }
  let dec = decide(d, ctx);
  if (weakNow && kind === 'live') dec = { state: STATE.UNSURE, reason: 'weak_signal', text: STATE_TEXT.UNSURE, detail: WEAK_MSG, hr: null };
  lastDecision = dec; logDecision(d, dec);
  showHr(d, ctx); if (weakNow && kind === 'live') $('hr').textContent = '–'; setSig(d); $('dhr').textContent = d.valid && dec.hr != null && Number.isFinite(d.dHR) ? (d.dHR >= 0 ? '+' : '') + d.dHR.toFixed(0) : '–';
  let ref = null; if (kind === 'replay') { ref = refAt(d.s); $('ref').textContent = ref == null ? '–' : ref.toFixed(0); }
  const t = `${d.s + WIN}s` + (Number.isFinite(d.p) ? ` p=${d.p.toFixed(2)}` : '');
  if (dec.state === STATE.NORMAL) { tally.normal++; setStatus('normal', '✓', dec.text, 'No drop from your own baseline detected in this window.'); pushStrip('d-normal', 'normal ' + t); }
  else if (dec.state === STATE.LOW) { tally.low++; setStatus('hyp', '↓', dec.text, 'Your signs look like a drop from your own baseline. This is not a diagnosis. If you feel unwell, get medical help.'); pushStrip('d-hyp', 'possible low oxygen ' + t); }
  else if (dec.reason === 'moving') { tally.pause++; setStatus('uns', '✋', dec.text, dec.detail); pushStrip('d-pause', 'unsure: moving'); }
  else { tally.uns++; setStatus('uns', '?', dec.text, dec.detail); pushStrip('d-uns', 'unsure: ' + dec.reason); }
  if (ref != null && dec.state !== STATE.UNSURE) { ((dec.state === STATE.LOW) === (ref < 90)) ? tally.right++ : tally.wrong++; }
  const n = tally.normal + tally.low + tally.uns + tally.pause;
  $('tally').textContent = `${n} readings: ${tally.normal} normal, ${tally.low} possible low oxygen, ${tally.uns + tally.pause} unsure.` + (kind === 'replay' ? ` Against the study reference, ${tally.right} of ${tally.right + tally.wrong} answered readings were correct.` : '');
}
function setSig(d) {
  const lvl = d.valid ? (d.sqi >= 0.45 ? 'Good' : 'Fair') : 'Poor';
  $('sig').textContent = lvl; $('sig-u').textContent = Number.isFinite(d.sqi) ? 'quality ' + d.sqi.toFixed(2) : '';
}

function checkCalibration() {
  const s = chain.calibrationSummary();
  if (kind === 'live' && calFps.length && Math.min(...calFps) < CAL_FPS_MIN) return 'The camera frame rate was too low during calibration.';
  if (kind === 'live' && cam && cam.rs.gapStats(calStartN).max > CAL_GAP_S_MAX) return 'The camera stalled during calibration.';
  if (kind === 'live' && cam && cam.stats.gaps - calGaps0 > CAL_GAPS_MAX) return 'The camera dropped too many frames during calibration.';
  if (!Number.isFinite(s.hr)) return 'No pulse was found during calibration.';
  const cw = calWin.filter((x) => x[0] > CAL_WARM_S), vf = cw.length ? cw.filter((x) => x[1]).length / cw.length : 0;
  if (vf < CAL_VALID_MIN) return `The pulse was only clearly detected in ${Math.round(vf * 100)}% of the calibration minute (need ${Math.round(CAL_VALID_MIN * 100)}%).`;
  if (s.hr < MIN_HR || s.hr > MAX_HR) return `Your resting heart rate during calibration was ${Math.round(s.hr)} bpm, outside the ${MIN_HR}-${MAX_HR} range this prototype assumes.`;
  if (s.hrSpread > 30) return 'Your heart rate was not steady during calibration.';
  if (calMotion.length && calMotion.filter(Boolean).length / calMotion.length > 0.25) return 'You moved too much during calibration.';
  return null;
}

function drawWave(cv, data) {
  const c = cv.getContext('2d'), w = cv.width, h = cv.height; c.clearRect(0, 0, w, h); if (!data || data.length < 10) return;
  let mn = Infinity, mx = -Infinity; for (const v of data) { if (v < mn) mn = v; if (v > mx) mx = v; } const sc = (mx - mn) || 1;
  c.strokeStyle = '#3ad0c0'; c.lineWidth = 2; c.beginPath();
  data.forEach((v, i) => { const x = (i / (data.length - 1)) * w, y = h - 8 - ((v - mn) / sc) * (h - 16); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke();
}
function drawRaw(cv) { // last 10 s of raw R, G, B means, each scaled to its own range
  const c = cv.getContext('2d'), w = cv.width, h = cv.height; c.clearRect(0, 0, w, h);
  const n = Math.min(rawN, 10 * FS, RAW_MAX); if (n < 10) return;
  ['#ff6b7a', '#3ddc97', '#6aa8ff'].forEach((col, ch) => {
    let mn = Infinity, mx = -Infinity; const v = new Array(n);
    for (let k = 0; k < n; k++) { const x = raw[(((rawN - n + k) % RAW_MAX) * 3) + ch]; v[k] = x; if (x < mn) mn = x; if (x > mx) mx = x; }
    const sc = (mx - mn) || 1; c.strokeStyle = col; c.lineWidth = 1.5; c.beginPath();
    v.forEach((x, k) => { const px = (k / (n - 1)) * w, py = h - 6 - ((x - mn) / sc) * (h - 12); k ? c.lineTo(px, py) : c.moveTo(px, py); }); c.stroke();
  });
}
