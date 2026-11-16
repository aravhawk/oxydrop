import { Chain, FS, CAL, WIN } from './engine.js';
import { CameraSource, Motion, listBackCameras, pickDefault, listCameras, probeCamFactory } from './camera.js';
import { runProbeLadder, liveQuality, ringState, probeSummary, isVirtualCamera } from './lensprobe.js';
import { MIN_TRACKABLE_BPM, GATE } from './hr.js';
import { HybridEngine, loadHybridAssets, ENGINE_ID } from './hybrid_engine.js';
import { HybridWorker, HybridWorkerEngine, loadHybridLite } from './hybrid_async.js';
import { PhotonEngine, loadPhotonAssets, DEFAULT_PHOTON_MODEL } from './photon_engine.js';
import { decide, hrToShow, assessPlacement, stalledDecision, gapContext, calibrationInterrupt, calibrationFingerReason, STATE, STATE_TEXT, MOTION_MAX, STALL_MS } from './decision.js';

const $ = (id) => document.getElementById(id);
const MIN_HR = MIN_TRACKABLE_BPM, MAX_HR = 110; // plausible *resting* heart rate at calibration; MIN_HR follows the HR estimator's band edge (F_LO = 0.85 Hz = 51 bpm)
const LOCK_NEED = 3;            // consecutive valid-lock windows before calibration can start
const ANYWAY_AFTER_S = 20;      // seconds without a lock before "start anyway" is offered
const CAL_VALID_MIN = 0.6;      // fraction of calibration windows that must have a valid pulse lock
const CAL_FPS_MIN = 20, CAL_GAPS_MAX = 20, CAL_GAP_S_MAX = 1.5;   // a single frame stall longer than 1.5 s anywhere in the calibration minute rejects it (as in v1)
const LOCK_AFTER_MS = 3000;     // fingertip placement must read "good" this long before exposure/white balance/focus are locked
const UNLOCK_AFTER_MS = 2000;   // in setup, lift the lock again if the placement has been bad this long (the next lock is taken on the new scene)
const CAL_WARM_S = 20;          // windows starting at s <= 20 still contain the band-pass filter's start-up transient (always invalid, even on a clean signal), so they are not counted
const PROBE_MAX_LENSES = Infinity;     // enumerate and probe every exposed back lens; no model-specific lens count
const GET_READY_S = 3;          // countdown so the finger is on the plateau before the first probe
const WEAK_PROMPT_MS = 4000;    // live signal quality must stay in 'reposition' this long before the re-position prompt replaces a reading
const WEAK_MSG = 'Weak signal: slide your finger so it covers the lens and the flash, press lightly.';
const RAW_MAX = 15 * 60 * FS;   // raw trace ring buffer: 15 minutes at 30 Hz
// Engine: the default is the engine slot (app/js/photon_engine.js) running PhotonDistill (app/js/distill_engine.js + distill_features.js, model file app/data/distill.json:
// a trained student MLP whose own output and abstain head are shown with no extra masking; ?engine=distill names it explicitly, ?model=<name> picks another file of a
// registered format, debug). ?engine=hybrid runs the previous hybrid (v1 physics chain + residual CNN + conformal rule, app/js/hybrid_engine.js), ?engine=logistic the
// v2 logistic chain (the baseline of the evaluation). ?alpha=0.05 sets the hybrid's conformal level (debug; default is the pre-declared 0.10).
const QS = new URLSearchParams(location.search);
const ENGINE_ARG = QS.get('engine');
const ENGINE_REQ = ENGINE_ARG === 'logistic' ? 'logistic' : ENGINE_ARG === 'hybrid' ? 'hybrid' : 'photon';   // 'photon' = the engine slot (default)
const PHOTON_MODEL = QS.get('model') || DEFAULT_PHOTON_MODEL;
const CAMERA_HR_GATE = { ...GATE, harmonicGuard: true };
const ALPHA_REQ = QS.get('alpha') ? Number(QS.get('alpha')) : null;
let engineUsed = ENGINE_REQ, engineNote = '', hybridLite = null, hybridFull = null, hybridWorker = null, hybridAlpha = null, photonAssets = null;

let mode = null;                // null | 'setup' | 'measure'
let kind = null;                // 'live' | 'replay'
let chain = null, weights = null, cam = null, motion = null, wake = null, camGen = 0, sessionGen = 0;   // sessionGen: one live setup attempt (bumped only by cleanup()); camGen: one CameraSource instance (bumped by every startCamera and by cleanup())
let replay = null, replayTimer = null, uiTimer = null, anywayTimer = null, tickN = 0;
let strip = [], tally = {}, calMotion = [], calFps = [], calWin = [], calGaps0 = 0, calFailed = false, calChecked = false, calOk = true;
let lockHist = [], lockRun = 0, camStartedAt = 0, place = { level: 'warn', title: '', msg: '' }, fingerHist = [], gapLog = [], lastGaps = 0, lastD = null, lastDecision = null, lastWaveCh = 1;
let probe = null, qual = null, weakSince = 0, weakNow = false, weakShown = false, chosenLabel = '';
let goodSince = 0, badSince = 0, lockBusy = false, stalledShown = false, calStartN = 0, beginning = false;
let calFingerGood = 0, calFingerN = 0, calBadSince = 0, calDirty = false, calSig = '';   // acquisition watch over the calibration minute
const raw = new Float32Array(RAW_MAX * 3); let rawN = 0, decLog = [];

function show(id) { document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === id)); window.scrollTo(0, 0); }
document.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => { cleanup(); show(b.dataset.go); }));
$('ack').addEventListener('change', (e) => { $('btn-live').disabled = !e.target.checked; });
const hasCam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
$('compat').textContent = hasCam ? '' : 'Camera access is not available in this browser or context (needs HTTPS). The recorded demo still works.';
if (!hasCam) { $('btn-live').disabled = true; $('ack').disabled = true; }
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

let debugVisible = QS.get('debug') === '1';
const debugToggle = document.createElement('button'); debugToggle.id = 'capture-debug-toggle'; debugToggle.textContent = 'Capture debug'; debugToggle.className = 'capture-debug-toggle';
const debugPanel = document.createElement('section'); debugPanel.id = 'capture-debug'; debugPanel.className = 'capture-debug'; debugPanel.hidden = !debugVisible;
const debugPre = document.createElement('pre'); debugPre.id = 'capture-debug-text'; debugPre.textContent = 'Waiting for camera. Cover the light AND the lens; press lightly.';
debugPanel.appendChild(debugPre); document.body.appendChild(debugPanel); document.body.appendChild(debugToggle);
debugToggle.addEventListener('click', () => { debugVisible = !debugVisible; debugPanel.hidden = !debugVisible; });

const fetchJson = async (rel) => { const r = await fetch(rel); if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`); return r.json(); };
async function loadWeights() { if (!weights) weights = await fetchJson('data/weights.json'); return weights; }
// Hybrid assets. Live CNN runs in a module Web Worker; heavy in-page inference is reserved for replay. Any live worker startup failure falls back to the logistic chain and says so in the debug text (never silently).
async function loadHybrid(needFull) {
  if (ENGINE_REQ !== 'hybrid' || engineUsed !== 'hybrid') return;
  try {
    if (!hybridLite) {
      hybridLite = await loadHybridLite(fetchJson);
      const a = ALPHA_REQ ?? hybridLite.conformal.default_alpha, t = hybridLite.conformal.alphas[a.toFixed(2)];
      if (!t || t.past_crossing) { engineNote = `alpha ${ALPHA_REQ} is not available; using ${hybridLite.conformal.default_alpha}. `; hybridAlpha = hybridLite.conformal.default_alpha; } else hybridAlpha = a;
    }
    if (needFull) { if (!hybridFull) hybridFull = await loadHybridAssets(fetchJson); return; }
    if (!hybridWorker && !hybridFull && typeof Worker !== 'undefined') {
      try { const w = new HybridWorker(new URL('./hybrid_worker.js', import.meta.url), hybridAlpha); await w.ready; hybridWorker = w; hybridWorker.onError = (m) => workerFailed(m); }
      catch (e) { throw new Error(`model worker unavailable (${e && e.message ? e.message : e})`); }
    }
    if (!hybridWorker) throw new Error('model worker unavailable; capture requires a free page thread');
  } catch (e) { engineUsed = 'logistic'; engineNote = `hybrid model could not be loaded (${e && e.message ? e.message : e}); using the logistic chain. `; }
}
// Engine slot assets (default engine; the model file data/<PHOTON_MODEL>.json and the v2 chain weights; runs in the page thread). A load failure falls back to the logistic
// chain and says so on screen (never silently, and never to the hybrid: that would swap in a different model with different numbers).
async function loadPhoton() {
  if (ENGINE_REQ !== 'photon' || engineUsed !== 'photon') return;
  try { if (!photonAssets) photonAssets = await loadPhotonAssets(fetchJson, PHOTON_MODEL); }
  catch (e) { engineUsed = 'logistic'; noticeText = PHOTON_FALLBACK_NOTICE; engineNote = `model file data/${PHOTON_MODEL}.json could not be loaded (${e && e.message ? e.message : e}); using the logistic chain. `; }
}
// The hybrid model failing must never be silent: the page switched to the logistic chain alone, so the measurement screen says so (text only, no device logic).
// Two cases: the assets could not be loaded, or the Web Worker errored or stopped answering after it had loaded. Not shown when the chain was asked for on purpose (?engine=logistic).
const FALLBACK_NOTICE = 'Notice: the neural-network part of the model could not be loaded, so these answers come from the simpler logistic chain alone. Reload the page to try again. The three result words are unchanged.';
const WORKER_NOTICE = 'Notice: the neural-network part of the model stopped working, so from now on these answers come from the simpler logistic chain alone. Reload the page to try again. The three result words are unchanged.';
// The same rule for the photon engine slot (?engine=photon): its model file could not be loaded, or the model threw during a measurement.
const PHOTON_FALLBACK_NOTICE = 'Notice: the selected on-device model could not be loaded, so these answers come from the simpler logistic chain alone. Reload the page to try again. The three result words are unchanged.';
const PHOTON_FAIL_NOTICE = 'Notice: the selected on-device model stopped working, so from now on these answers come from the simpler logistic chain alone. Reload the page to try again. The three result words are unchanged.';
const WORKER_STALL_MS = 2000;  // a window the worker has not answered for this long counts as a dead worker (normal latency is 50 to 300 ms)
let noticeText = FALLBACK_NOTICE;
function showEngineNotice() {
  const el = $('engine-note'), fell = ENGINE_REQ !== 'logistic' && engineUsed === 'logistic';
  el.hidden = !fell; el.textContent = fell ? noticeText : '';
}
// The photon model threw during a measurement: the engine already serves the logistic chain's windows (its v2 chain holds the calibration); leave the
// photon slot for the rest of this page load and say so on screen.
function photonFailed(msg) {
  if (engineUsed !== 'photon') return;
  engineUsed = 'logistic'; noticeText = PHOTON_FAIL_NOTICE; engineNote += `model file data/${PHOTON_MODEL}.json failed during the measurement (${msg}); using the logistic chain. `;
  if (chain && chain.degrade) chain.degrade(msg);
  showEngineNotice();
}
// The CNN worker errored, or stopped answering, after it had loaded: leave the hybrid for good (this page load), keep the running measurement on the logistic chain
// (the quality chain inside the worker engine already holds the calibration), and say so on screen.
function workerFailed(msg) {
  if (engineUsed !== 'hybrid') return;
  engineUsed = 'logistic'; noticeText = WORKER_NOTICE; engineNote += `model worker failed after loading (${msg}); using the logistic chain. `;
  if (hybridWorker) { hybridWorker.onError = null; hybridWorker.terminate(); hybridWorker = null; }
  if (chain && chain.degrade) chain.degrade();
  showEngineNotice();
}
// a fresh engine for one measurement (calibration restarts get a new one)
function newMeasureEngine(w) {
  if (engineUsed === 'photon' && photonAssets) {
    try { return new PhotonEngine(photonAssets, kind === 'live' ? { hrGate: CAMERA_HR_GATE } : {}); }
    catch (e) { engineUsed = 'logistic'; noticeText = PHOTON_FALLBACK_NOTICE; engineNote = `model file data/${PHOTON_MODEL}.json could not be started (${e && e.message ? e.message : e}); using the logistic chain. `; }
  }
  if (engineUsed === 'hybrid' && hybridLite) {
    if (kind === 'live' && hybridWorker) { const e = new HybridWorkerEngine(hybridLite, hybridWorker, hybridAlpha, (d) => { if (chain === e) dispatchWindow(d); }, { hrGate: CAMERA_HR_GATE }); return e; }
    if (hybridFull) return new HybridEngine(hybridFull, { alpha: hybridAlpha });
  }
  return new Chain(w || weights, kind === 'live' ? { hrGate: CAMERA_HR_GATE } : {});
}
function dispatchWindow(d) {
  // Live HR requires the spectral/ACF/beat consensus; a disagreeing CNN cannot override it.
  if (kind === 'live' && d.valid && Number.isFinite(d.hrEst) && Number.isFinite(d.hr) && Math.abs(d.hr - d.hrEst) > 5) { d = { ...d, hr: d.hrEst, cameraHrOverride: true }; }
  if (mode === 'setup') onSetupWindow(d); else if (mode === 'measure') onWindow(d); }

function cleanup() {
  sessionGen++; camGen++; clearInterval(replayTimer); clearInterval(uiTimer); clearTimeout(anywayTimer); uiTimer = null;
  if (cam) { cam.stop(); cam = null; } if (motion) { motion.stop(); motion = null; }
  if (wake) { try { wake.release(); } catch (e) { /* ignore */ } wake = null; }
  hideWhite(); weakSince = 0; weakNow = false; weakShown = false;
  if (chain && chain.close) chain.close();
  replay = null; mode = null; kind = null; chain = null;
}
const isFront = () => lightMode() === 'front';
function showWhite() { $('white').hidden = false; }
function hideWhite() { $('white').hidden = true; }
$('wm-stop').addEventListener('click', () => { cleanup(); show('home'); });
$('wm-begin').addEventListener('click', () => { if (mode === 'setup') beginLive(); else if (mode === 'measure') startMeasure('live'); });
async function keepAwake() { try { if (navigator.wakeLock) wake = await navigator.wakeLock.request('screen'); } catch (e) { /* ignore */ } }
function startUi() { clearInterval(uiTimer); tickN = 0; uiTimer = setInterval(uiTick, 250); }

// ---------------- raw trace recorder (bounded ring buffer) ----------------
function resetRecorder() { rawN = 0; decLog = []; lastD = null; lastDecision = null; }
function record(r, g, b) { const i = (rawN % RAW_MAX) * 3; raw[i] = r; raw[i + 1] = g; raw[i + 2] = b; rawN++; }
function logDecision(d, dec) { decLog.push({ idx: (Number.isFinite(d.n) ? d.n : rawN) - 1, d, dec });   // hybrid windows arrive after their last sample: d.n is where they ended
  if (decLog.length > RAW_MAX / (2 * FS) + 5) decLog.shift(); }
function buildCsv() {
  const start = Math.max(0, rawN - RAW_MAX), byIdx = new Map(decLog.map((x) => [x.idx, x]));
  const rows = ['t_s,R,G,B,sqi,acf,hr_est,hr_beat,valid,sat,label,p,state,reason,hr_shown,z,cnn_hr,engine'];
  for (let n = start; n < rawN; n++) {
    const i = (n % RAW_MAX) * 3; let tail = ',,,,,,,,,,,,,';
    const x = byIdx.get(n);
    if (x) { const d = x.d, c = x.dec; tail = [f4(d.sqi), f4(d.acf), f2(d.hrEst), f2(d.hrBeat), d.valid ? 1 : 0, f4(d.sat), d.label || (d.calibrating ? 'calibrating' : ''), f4(d.p), c ? c.state : '', c && c.reason ? c.reason : '', c && c.hr != null ? c.hr.toFixed(1) : '', f4(d.z), f2(d.cnnHr), d.engine || 'logistic'].join(','); }
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

function engineLine() {
  const inf = chain && chain.info, lat = chain && chain.latency ? chain.latency() : null, f1 = (v) => (Number.isFinite(v) ? v.toFixed(0) : '-'), f2l = (v) => (Number.isFinite(v) ? v.toFixed(1) : '-');
  const latTxt = lat && lat.n ? `last ${f1(lat.last)} ms, median ${f1(lat.median)} ms, p95 ${f1(lat.p95)} ms` + (lat.async ? ` (model ${f1(lat.coreMedian)} ms in the worker, quality chain ${f1(lat.v2Median)} ms on the page)` : '') : 'no window yet';
  if (engineUsed === 'photon' && inf && chain instanceof PhotonEngine) return `engine ${inf.id} (default engine slot, ?engine=distill, model file data/${PHOTON_MODEL}.json): ${inf.model}, ${(inf.perSeed || [inf.params]).map((v) => v.toLocaleString('en-US')).join(' / ')} parameters per seed (${inf.paramsTotal.toLocaleString('en-US')} in total), runs in the page thread; output shown as the model gives it (its own abstention = Unsure); latency per 2 s window: ${lat && lat.n ? `last ${f2l(lat.last)} ms, median ${f2l(lat.median)} ms, p95 ${f2l(lat.p95)} ms` : 'no window yet'}${engineNote ? '  ' + engineNote : ''}`;
  if (engineUsed === 'photon') return `engine slot (default, ?engine=distill, model file data/${PHOTON_MODEL}.json; only the quality chain runs during setup, the model starts with the calibration minute)${engineNote ? '  ' + engineNote : ''}`;
  if (engineUsed !== 'hybrid' || !inf) return `engine ${engineUsed === 'hybrid' ? 'hybrid (only the quality chain runs during setup; the CNN starts with the calibration minute)' : 'logistic chain v2 (baseline / fallback)'}${ENGINE_REQ === 'logistic' ? ' (?engine=logistic)' : ''}${engineNote ? '  ' + engineNote : ''}`;
  const th = inf.thresholds;
  return `engine ${ENGINE_ID}: v1 physics chain + ${inf.model}, ${inf.params.toLocaleString('en-US')} parameters per seed (${inf.paramsTotal.toLocaleString('en-US')} total), runs ${chain.worker ? 'in a Web Worker' : 'in the page thread'}; conformal alpha ${inf.alpha} (Normal below z ${th.normal_below.toFixed(3)}, Low above ${th.low_above.toFixed(3)}); latency per 2 s window: ${latTxt}${ENGINE_ARG === 'hybrid' ? ' (?engine=hybrid)' : ''}${engineNote ? '  ' + engineNote : ''}`;
}
function debugText() {
  const L = []; const st = cam ? cam.stats : null;
  L.push('capture build oxydrop-v3.6-photon-distill2');
  L.push(`mode ${mode || '-'} / ${kind || '-'}   samples ${rawN}   t ${(rawN / FS).toFixed(1)} s`);
  L.push(engineLine());
  if (cam && st) {
    L.push(`capture ${st.captureState || '-'} callbacks native ${st.nativeCallbacks || 0} fallback ${st.fallbackCallbacks || 0} frame errors ${st.frameErrors || 0} ${st.lastFrameError || ''}; video readyState ${st.videoReadyState || 0} ${st.videoWidth || 0}x${st.videoHeight || 0}`);
    L.push(`fps ${st.fps.toFixed(1)}  jitter ${st.jitterMs.toFixed(1)} ms  gaps ${st.gaps} (max ${(st.maxGap * 1000).toFixed(0)} ms)  skipped ${st.skipped}  frames ${st.frames}`);
    L.push(`last frame ${(st.staleMs / 1000).toFixed(1)} s ago  interrupted ${st.interrupted || 'no'}  longest gap in last 30 s ${(cam.rs.gapStats(cam.rs.n - 30 * FS).max * 1000).toFixed(0)} ms`);
    L.push(`lens "${cam.info.lens}"  ${cam.info.width}x${cam.info.height}`);
    if (cam.info.fallback) L.push(`device fallback: ${cam.info.fallback}`);
    L.push(`torch ${cam.info.torchState || (cam.info.torchSupported ? (cam.info.torch ? 'on-unverified' : 'off') : 'unsupported')}  locks ${JSON.stringify(cam.info.locks)}`);
    L.push(`mean RGB ${st.mean.map((v) => v.toFixed(1)).join(' / ')}  frame-saturated ${(st.satFrac * 100).toFixed(1)}%  spatialCV ${st.spatialCV.toFixed(3)}  luma ${st.luma.toFixed(1)}`);
    L.push(`placement ${place.level}: ${place.title}; near-black maxDC<=3, red ratios R/G=${f2(st.mean[0] / Math.max(st.mean[1], 0.01))} R/B=${f2(st.mean[0] / Math.max(st.mean[2], 0.01))} (need >=1.4); channel saturation ${(st.channelSat || []).map(v => (100*v).toFixed(1)).join('/')}%; ROI ${JSON.stringify(st.roi || 'full')}`);
    if (st.fullMean) L.push(`full-frame RGB DC ${st.fullMean.map(f2).join('/')} (ROI RGB DC above)`);
    if (cam.info.adapt) L.push(`adaptive exposure: ${cam.info.adapt.result}  luma ${f2(cam.info.adapt.lumaBefore)} -> ${f2(cam.info.adapt.lumaAfter)}  sat ${f4(cam.info.adapt.satAfter)}  applied ${JSON.stringify(cam.info.adapt.applied)}  supported ${(cam.info.adapt.supported || []).join(',') || 'none'}`);
  }
  L.push(`light mode ${lightMode()}  chosen lens "${chosenLabel || '-'}"${qual ? `  live quality ${qual.level} (score ${qual.score.toFixed(0)}, acdc ${(qual.acdc * 100).toFixed(2)}%, SNR ${f2(qual.snr)}, sqi ${f2(qual.sqi)}, gate ${qual.reason})` : ''}${weakNow ? '  REPOSITION PROMPT SHOWN' : ''}`);
  if (probe) { L.push('lens probe scores (per lens, 0-100, usable >= 35):'); L.push(probeSummary(probe.results, probe.choice)); } else L.push('lens probe: not run');
  if (lastD) {
    const d = lastD; L.push(`live HR override ${!!d.cameraHrOverride}; window s=${d.s}  SQI ${f4(d.sqi)}  ACF ${f4(d.acf)}  hrEst ${f2(d.hrEst)}  hrBeat ${f2(d.hrBeat)}  ch ${d.ch}  sat ${f4(d.sat)}  valid ${d.valid}  label ${d.label || 'calibrating'}  p ${f4(d.p)}`);
    if (d.model !== undefined && d.pulseLock !== undefined) {   // photon slot window: the model's raw output, then the model-specific extras
      const m = d.model || {};
      L.push(`model ${d.engine}: raw P(SpO2<90) ${f4(d.modelP)}  abstain ${d.modelAbstain} (abstain head ${f4(m.conf)})  label ${d.label}  model latency ${f2(m.ms)} ms  v2 pulse lock ${d.pulseLock}  | EXPERIMENTAL SpO2 estimate of the model (not validated, do not use): ${f2(d.modelSpo2)}`);
    } else if (Number.isFinite(d.z)) L.push(`hybrid: chain logit c ${f2(d.c)}  CNN correction ${f2(d.corr)}  z ${f2(d.z)}  CNN HR ${f2(d.cnnHr)}  spectral HR ${f2(d.specHr)}  | EXPERIMENTAL SpO2 from the CNN head (not validated, do not use): ${f2(d.spo2Exp)}`);
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
  let gen = -1;
  try {
    cleanup(); gen = sessionGen; motion = new Motion(); const mp = motion.start();   // start the motion permission prompt first: iOS needs it inside the tap
    await loadWeights(); $('cam-state').textContent = 'Loading the on-device model…'; await loadHybrid(false); await loadPhoton(); await keepAwake();
    kind = 'live'; mode = 'setup'; startUi();
    const choice = await runProbe(gen);   // session token: startCamera() below bumps camGen on purpose, which must not cancel this flow
    if (gen !== sessionGen) return;
    const lm = choice.mode === 'torch' ? 'flash' : choice.mode;
    const radio = document.querySelector(`input[name=light][value=${lm}]`); if (radio) radio.checked = true;
    chosenLabel = choice.label || '';
    await startCamera(choice.mode === 'front' ? null : choice.deviceId);
    if (gen !== sessionGen) return;
    await populateLens(); if (choice.mode !== 'front' && choice.deviceId && !(cam && cam.info.fallback)) $('lens').value = choice.deviceId;
    await mp;
    if (gen !== sessionGen) return;
    $('setup-live').hidden = false; renderChecks(); renderPlace();
  } catch (e) {
    if (gen !== -1 && gen !== sessionGen) return;   // this attempt was cancelled; do not tear down the session that replaced it
    cleanup(); $('cam-state').textContent = 'Could not start the camera: ' + (e && e.message ? e.message : e) + '. Allow camera access and reload, or use the recorded demo.';
    $('btn-cam').hidden = false; $('setup-live').hidden = true; $('probe').hidden = true;
  }
});

// Lens probe: every back lens for ~3 s under the torch, best pulse wins; if none works, ambient light or the front camera + white screen (lensprobe.js).
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function runProbe(gen) {
  $('probe').hidden = false; $('probe-scores').textContent = ''; probe = null; qual = null;
  const gum = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' } } });   // one grant first: device labels are empty until the camera is allowed
  gum.getTracks().forEach((t) => t.stop());
  const all = await listCameras(), physical = all.back.filter(c => !isVirtualCamera(c.label)), pd = pickDefault(physical), back = [...physical].sort((a, b) => (b === pd) - (a === pd)).slice(0, PROBE_MAX_LENSES);
  const say = (t) => { $('probe-text').textContent = t; $('cam-state').textContent = t; if (!$('white').hidden) { $('wm-title').textContent = t; $('wm-msg').textContent = ''; } };
  for (let k = GET_READY_S; k > 0; k--) { say(`Lay your finger flat across the whole camera plateau now (lens and flash), light pressure. Testing starts in ${k}…`); await sleep(1000); if (gen !== sessionGen) return null; }
  const video = document.createElement('video'), base = probeCamFactory(video), partial = [];
  const factory = (cb) => { if (gen !== sessionGen) throw new Error('cancelled'); return base(cb); };
  say(`Testing lens 1 of ${back.length}, keep your finger flat across the camera plateau`);
  const out = await runProbeLadder(factory, { back, front: all.front }, {
    ambientMax: 1, shouldAbort: () => gen !== sessionGen,
    onProgress: (r, k) => {
      partial.push(r); probe = { results: [...partial], choice: null }; $('probe-scores').textContent = probeSummary(partial, null);
      say(k < back.length ? `Testing lens ${k + 1} of ${back.length}, keep your finger flat across the camera plateau` : 'Rear-camera samples collected. Selecting a lens or continuing live capture…');
    },
    beforeFront: async () => { if (gen !== sessionGen) return; showWhite(); say('Testing the front camera. Cover it with your fingertip.'); await sleep(1200); },
    afterFront: async () => { if (gen === sessionGen) hideWhite(); },   // a stale probe must not hide a newer session's overlay
  });
  if (gen !== sessionGen) return null;
  probe = out; $('probe-scores').textContent = probeSummary(out.results, out.choice);
  $('probe-text').textContent = out.choice.reason + (out.choice.usable || out.choice.continuous ? '' : ' Re-position your finger and try again, or pick a mode below.');
  return out.choice;
}

async function startCamera(deviceId) {
  const gen = ++camGen;
  if (cam) { cam.stop(); cam = null; }
  chain = new Chain(weights, { hrGate: CAMERA_HR_GATE }); resetRecorder(); lockHist = []; lockRun = 0; fingerHist = []; gapLog = []; lastGaps = 0; goodSince = 0; badSince = 0; lockBusy = false; stalledShown = false; place = { level: 'warn', title: 'Looking for your fingertip…', msg: '' };
  $('btn-begin').disabled = true; $('btn-begin-anyway').hidden = true; $('q-fill').style.width = '0%'; $('q-text').textContent = 'Signal quality: waiting for a pulse lock';
  const c = new CameraSource((r, g, b) => { if (gen === camGen) sink(r, g, b); }, (st) => { if (gen === camGen && mode === 'setup' && st.roiChanged && st.frames > 1) resetSetupChain(); });
  cam = c; weakSince = 0; weakNow = false; weakShown = false; qual = null;
  const front = isFront(); if (front) showWhite(); else hideWhite();
  try { await c.start(document.createElement('video'), { deviceId: front ? null : deviceId, torch: lightMode() === 'flash' ? 'auto' : 'off', facing: front ? 'user' : 'environment', allowFrameTimeout: true }); }
  catch (e) { c.stop(); if (cam === c) cam = null; throw e; }
  if (gen !== camGen) { c.stop(); return; }   // cleanup() or a newer startCamera() ran while this camera was opening: release the stream we just got
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
  chain = new Chain(weights, { hrGate: CAMERA_HR_GATE }); lockHist = []; lockRun = 0; $('btn-begin').disabled = true; camStartedAt = performance.now();
  clearTimeout(anywayTimer); $('btn-begin-anyway').hidden = true; anywayTimer = setTimeout(() => { if (mode === 'setup' && $('btn-begin').disabled) $('btn-begin-anyway').hidden = false; }, ANYWAY_AFTER_S * 1000);
  $('cam-state').textContent = on ? (cam.info.torch ? 'Camera on, flash on' : 'Camera on (flash control not available in this browser)') : 'Camera on, flash off (ambient light)';
  renderChecks();
}));

// every resampled 30 Hz camera/replay sample goes through here
function sink(r, g, b) {
  record(r, g, b);
  if (!chain) return;
  const d = chain.push(r, g, b);
  if (d) dispatchWindow(d);
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
function resetSetupChain() { chain = new Chain(weights, { hrGate: CAMERA_HR_GATE }); lockHist = []; lockRun = 0; $('btn-begin').disabled = true; }

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

// Calibration must not start until exposure / white balance / focus are settled: a lock taken after the baseline minute has begun changes the image
// level and the baseline would then describe a different camera state than the readings that follow.
async function beginLive() {
  if (beginning || mode !== 'setup' || !cam) return;
  beginning = true;
  try {
    const c0 = cam, g = sessionGen;
    while (lockBusy && g === sessionGen) await sleep(100);
    if (g === sessionGen && cam === c0 && !c0.locked) { lockBusy = true; try { await lockWithAdapt(c0); } catch (e) { /* best effort */ } finally { lockBusy = false; } }
    if (g !== sessionGen || cam !== c0 || mode !== 'setup') return;
    startMeasure('live');
  } finally { beginning = false; }
}
$('btn-begin').addEventListener('click', beginLive);
$('btn-begin-anyway').addEventListener('click', beginLive);

// ---------------- replay flow ----------------
$('btn-replay').addEventListener('click', async () => {
  cleanup(); await loadWeights(); await loadHybrid(true); await loadPhoton(); show('measure'); kind = 'replay'; resetRecorder();
  const pick = new URLSearchParams(location.search).get('rec') || (Math.random() < 0.5 ? '100001L' : '100004R');
  const [meta, buf] = await Promise.all([fetch(`data/replay/${pick}.json`).then((r) => r.json()), fetch(`data/replay/${pick}.bin`).then((r) => r.arrayBuffer())]);
  replay = { meta, rgb: new Float32Array(buf), i: 0, speed: 30 };
  startMeasure('replay', meta.weights);
  const note = $('replay-note'); note.hidden = false;
  const heldOut = !(chain && chain.core);   // the logistic replay weights are fitted without this person; the hybrid weights are not
  note.innerHTML = `Recorded demo: study participant ${meta.person} (UW Hoffman 2022 data, MIT licence). ` + (heldOut
    ? 'The model used here was fitted <b>without</b> this person, so these are genuinely held-out decisions. '
    : '<b>Model trained on this person: demo only, not a test.</b> The on-device model saw this participant during training, so these answers show how the screen behaves, not how accurate the model is. The held-out results are in the results table (README). ') + `Replaying at ${replay.speed}× speed.${meta.person === 4 ? ' <b>This is the hardest person in the study: a resting heart rate of 43 bpm is below the 51 bpm this version can follow, so it correctly answers &ldquo;unsure&rdquo; throughout.</b>' : ''}`;
  replayTimer = setInterval(() => {
    const per = Math.round(replay.speed * FS * 0.05);
    for (let k = 0; k < per && replay.i < replay.meta.n; k++, replay.i++) { const j = replay.i * 3; sink(replay.rgb[j], replay.rgb[j + 1], replay.rgb[j + 2]); }
    if (replay.i >= replay.meta.n) { clearInterval(replayTimer); $('phase').textContent = 'Replay finished'; }
  }, 50);
});

// ---------------- shared measurement logic ----------------
function startMeasure(k, w) {
  kind = k; mode = 'measure'; if (chain && chain.close) chain.close(); chain = newMeasureEngine(w);
  stalledShown = false; calStartN = cam ? cam.rs.n : 0;
  calFingerGood = 0; calFingerN = 0; calBadSince = 0; calDirty = false; calSig = acquisitionSig();
  calMotion = []; calFps = []; calWin = []; calFailed = false; calChecked = false; calOk = true; strip = []; $('strip').innerHTML = ''; lastD = null; lastDecision = null;
  if (k === 'live') { calGaps0 = cam ? cam.stats.gaps : 0; decLog = []; }
  tally = { normal: 0, low: 0, uns: 0, pause: 0, right: 0, wrong: 0 };
  clearTimeout(anywayTimer); show('measure'); $('btn-restart').hidden = k !== 'live'; $('ref-tile').hidden = k !== 'replay'; if (k !== 'replay') $('replay-note').hidden = true;
  showEngineNotice();
  setStatus('calibrating', '◔', 'Calibrating', 'Rest, keep still. Learning your normal heart-rate and light baseline.');
  $('phase').textContent = 'Calibrating'; $('cal-progress').hidden = false; $('tally').textContent = '';
  $('hr').textContent = '–'; $('dhr').textContent = '–'; $('sig').textContent = '–'; $('sig-u').textContent = '';
  if (k !== 'live') { $('mot').textContent = '–'; $('mot-u').textContent = 'n/a in replay'; }
  startUi();
}
$('btn-restart').addEventListener('click', () => startMeasure('live'));
$('btn-stop').addEventListener('click', () => { cleanup(); show('home'); });

// Anything that changes how the camera sees the finger voids the calibration minute (the baseline would describe a different camera state).
const acquisitionSig = () => (kind === 'live' && cam ? `${cam.info.deviceId}|${cam.locked}|${JSON.stringify(cam.info.locks)}|${cam.info.torch}|${isFront()}` : '');
const RESTART_TEXT = {
  finger_lost: 'Your fingertip came off the lens or flash during calibration, so the minute was started again. Keep it covering both and still.',
  settings_changed: 'The camera settings changed during calibration, so the minute was started again. Keep still.',
};
function restartCalibration(why) { startMeasure('live'); setStatus('calibrating', '◔', 'Calibrating', RESTART_TEXT[why]); $('phase').textContent = 'Calibrating (restarted)'; }
// Called every UI tick while the calibration minute runs (live only). A frame stall is left to the calibration stall rule, not counted as finger loss.
function watchCalibration(now, stalled) {
  if (stalled) { calBadSince = 0; return false; }
  const good = place.level === 'good';
  calFingerN++; if (good) { calFingerGood++; calBadSince = 0; } else if (!calBadSince) calBadSince = now;
  const why = calibrationInterrupt({ placeGood: good, stalled, badMs: calBadSince ? now - calBadSince : 0, settingsChanged: acquisitionSig() !== calSig });
  if (why === 'settings_changed') { restartCalibration(why); return true; }
  if (why === 'finger_lost') calDirty = true;
  if (calDirty && good) { restartCalibration('finger_lost'); return true; }   // the finger is back: begin the minute again from zero
  return false;
}

function uiTick() {
  tickN++;
  if (chain && chain.degraded && engineUsed === 'hybrid') workerFailed('model latency exceeded capture budget');
  if (chain && chain.degraded && engineUsed === 'photon') photonFailed(chain.failMsg || 'model error');
  if (chain && chain.worker && chain.oldestPendingMs && chain.oldestPendingMs() > WORKER_STALL_MS) workerFailed('no answer from the model worker for ' + Math.round(WORKER_STALL_MS / 1000) + ' s');
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
    if (mode === 'measure' && !calChecked && !calFailed && chain && watchCalibration(now, stalled)) return;
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
  if (tickN % 2 === 0) { const txt = debugText(); $(mode === 'setup' ? 'dbg-setup' : 'dbg-measure').textContent = txt; if (debugVisible) $('capture-debug-text').textContent = txt; }
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
  // the engine slot's model judges the signal itself (its abstain head): no weak-signal override of its answers during the measurement
  if (weakNow && !calFailed && !(chain && chain.ownQuality)) {
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
  if (weakNow && kind === 'live' && !d.ownQuality) dec = { state: STATE.UNSURE, reason: 'weak_signal', text: STATE_TEXT.UNSURE, detail: WEAK_MSG, hr: null };
  lastDecision = dec; logDecision(d, dec);
  showHr(d, ctx); if (weakNow && kind === 'live') $('hr').textContent = '–'; setSig(d); $('dhr').textContent = d.valid && dec.hr != null && Number.isFinite(d.dHR) ? (d.dHR >= 0 ? '+' : '') + d.dHR.toFixed(0) : '–';
  let ref = null; if (kind === 'replay') { ref = refAt(d.s); $('ref').textContent = ref == null ? '–' : ref.toFixed(0); }
  const t = `${d.s + WIN}s` + (Number.isFinite(d.p) ? ` p=${d.p.toFixed(2)}` : '');
  if (dec.state === STATE.NORMAL) { tally.normal++; setStatus('normal', '✓', dec.text, 'No drop from your own first minute detected in this window. It cannot know whether you were already low when you started.'); pushStrip('d-normal', 'normal ' + t); }
  else if (dec.state === STATE.LOW) { tally.low++; setStatus('hyp', '↓', dec.text, 'Your signs look like a drop from your own baseline. This is not a diagnosis. If you feel unwell, get medical help.'); pushStrip('d-hyp', 'possible low oxygen ' + t); }
  else if (dec.reason === 'moving') { tally.pause++; setStatus('uns', '✋', dec.text, dec.detail); pushStrip('d-pause', 'unsure: moving'); }
  else { tally.uns++; setStatus('uns', '?', dec.text, dec.detail); pushStrip('d-uns', 'unsure: ' + dec.reason); }
  if (ref != null && dec.state !== STATE.UNSURE) { ((dec.state === STATE.LOW) === (ref < 90)) ? tally.right++ : tally.wrong++; }
  const n = tally.normal + tally.low + tally.uns + tally.pause;
  $('tally').textContent = `${n} readings: ${tally.normal} normal, ${tally.low} possible low oxygen, ${tally.uns + tally.pause} unsure.` + (kind === 'replay' ? ` Against the study reference, ${tally.right} of ${tally.right + tally.wrong} answered readings were correct` + (chain && chain.core ? ' (the model saw this person: demo only, not a test).' : '.') : '');
}
function setSig(d) {
  const lock = d.pulseLock ?? d.valid;     // photon slot windows are marked valid (the model decides); the signal tile still reports the v2 pulse lock
  const lvl = lock ? (d.sqi >= 0.45 ? 'Good' : 'Fair') : 'Poor';
  $('sig').textContent = lvl; $('sig-u').textContent = Number.isFinite(d.sqi) ? 'quality ' + d.sqi.toFixed(2) : '';
}

function checkCalibration() {
  const s = chain.calibrationSummary();
  if (kind === 'live' && calFps.length && Math.min(...calFps) < CAL_FPS_MIN) return 'The camera frame rate was too low during calibration.';
  if (kind === 'live' && cam && cam.rs.gapStats(calStartN).max > CAL_GAP_S_MAX) return 'The camera stalled during calibration.';
  if (kind === 'live' && cam && cam.stats.gaps - calGaps0 > CAL_GAPS_MAX) return 'The camera dropped too many frames during calibration.';
  if (kind === 'live') { const fr = calibrationFingerReason(calFingerGood, calFingerN); if (fr) return fr; }
  if (kind === 'live' && calDirty) return 'Your fingertip came off the lens or flash during calibration and was not put back.';
  if (chain && chain.ownQuality) return calMotion.length && calMotion.filter(Boolean).length / calMotion.length > 0.25 ? 'You moved too much during calibration.' : null;   // engine slot: no pulse-signal gate on top of the model (its abstain head decides)
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
