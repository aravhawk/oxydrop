import { Chain, FS, CAL, WIN } from './engine.js';
import { CameraSource, Motion } from './camera.js';

const $ = (id) => document.getElementById(id);
const SQI_MIN = 0.12;          // rule-based quality gate; Hoffman windows: 1st percentile 0.143, min 0.099
const MOTION_MAX = 0.35;       // m/s^2 std of |accel| over 3 s: above this we pause decisions
const MIN_HR = 40, MAX_HR = 110; // plausible *resting* heart rate at calibration
let mode = null, chain = null, weights = null, cam = null, motion = null, wake = null;
let replay = null, replayTimer = null, timers = [], strip = [], tally = {}, calSqi = [], calMotion = [], calFailed = false, calChecked = false, lastWave = [];

function show(id) { document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === id)); window.scrollTo(0, 0); }
document.querySelectorAll('[data-go]').forEach((b) => b.addEventListener('click', () => { cleanup(); show(b.dataset.go); }));
$('ack').addEventListener('change', (e) => { $('btn-live').disabled = !e.target.checked; });
const hasCam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
$('compat').textContent = hasCam ? '' : 'Camera access is not available in this browser or context (needs HTTPS). The recorded demo still works.';
if (!hasCam) { $('btn-live').disabled = true; $('ack').disabled = true; }
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

async function loadWeights() { if (!weights) weights = await (await fetch('data/weights.json')).json(); return weights; }

function cleanup() {
  timers.forEach(clearInterval); timers = []; clearInterval(replayTimer); if (cam) { cam.stop(); cam = null; } if (motion) { motion.stop(); motion = null; }
  if (wake) { try { wake.release(); } catch (e) { /* ignore */ } wake = null; } replay = null; mode = null;
}
async function keepAwake() { try { if (navigator.wakeLock) wake = await navigator.wakeLock.request('screen'); } catch (e) { /* ignore */ } }

// ---------------- live camera flow ----------------
$('btn-live').addEventListener('click', () => { cleanup(); show('setup'); $('setup-live').hidden = true; $('btn-cam').hidden = false; $('cam-state').textContent = 'Camera off'; });

$('btn-cam').addEventListener('click', async () => {
  $('btn-cam').hidden = true; $('cam-state').textContent = 'Starting camera…';
  try {
    await loadWeights(); await keepAwake();
    chain = new Chain(weights);
    const video = document.createElement('video');
    cam = new CameraSource((r, g, b) => { const d = chain.push(r, g, b); if (d) onSetupWindow(d); lastWave = chain.ac[1].slice(-300); }, () => renderChecks());
    await cam.start(video); mode = 'setup';
    motion = new Motion(); await motion.start();
    $('setup-live').hidden = false; $('cam-state').textContent = cam.info.torch ? 'Camera on, flash on' : 'Camera on (flash not available in this browser)';
    renderChecks(); timers.push(setInterval(() => drawWave($('wave-setup'), lastWave), 100));
  } catch (e) {
    $('cam-state').textContent = 'Could not start the camera: ' + (e && e.message ? e.message : e) + '. Allow camera access and reload, or use the recorded demo.';
    $('btn-cam').hidden = false;
  }
});

let setupSqi = 0, setupHr = NaN;
function onSetupWindow(d) { setupSqi = d.sqi; setupHr = d.hr; const q = Math.max(0, Math.min(1, (d.sqi - 0.08) / 0.2)); $('q-fill').style.width = (q * 100) + '%'; $('q-fill').style.background = d.sqi >= SQI_MIN + 0.04 ? 'var(--green)' : 'var(--amber)'; $('q-text').textContent = d.sqi >= SQI_MIN + 0.04 ? 'Good pulse signal. Keep still.' : 'Weak signal. Cover the camera and flash fully, press lightly.'; $('btn-begin').disabled = !(d.sqi >= SQI_MIN + 0.04); renderChecks(); }
function renderChecks() {
  if (!cam) return; const li = [];
  li.push([cam.info.torch, cam.info.torch ? 'Flash on' : 'No flash control here (common on iPhone Safari). Use bright light; accuracy is unvalidated.']);
  li.push([cam.info.exposureLocked, cam.info.exposureLocked ? 'Exposure locked' : 'Exposure not locked on this device (can reduce accuracy)']);
  li.push([cam.info.fps >= 20, `Camera ${cam.info.fps.toFixed(0)} fps` + (cam.info.fps < 20 ? ' (low)' : '')]);
  li.push([motion && motion.ok, motion && motion.ok ? 'Motion sensor on' : 'No motion sensor: hold very still yourself']);
  $('checks').innerHTML = li.map(([ok, t]) => `<li class="${ok ? 'ok' : 'warn'}">${ok ? '✓' : '!'} ${t}</li>`).join('');
}

$('btn-begin').addEventListener('click', () => { startMeasure('live'); });

// ---------------- replay flow ----------------
$('btn-replay').addEventListener('click', async () => {
  cleanup(); await loadWeights(); show('measure');
  const pick = new URLSearchParams(location.search).get('rec') || (Math.random() < 0.5 ? '100001L' : '100004R');
  const [meta, buf] = await Promise.all([fetch(`data/replay/${pick}.json`).then((r) => r.json()), fetch(`data/replay/${pick}.bin`).then((r) => r.arrayBuffer())]);
  replay = { meta, rgb: new Float32Array(buf), i: 0, speed: 30 };
  startMeasure('replay', meta.weights);
  const note = $('replay-note'); note.hidden = false;
  note.innerHTML = `Recorded demo: study participant ${meta.person} (UW Hoffman 2022 data, MIT licence). The model used here was fitted <b>without</b> this person, so these are genuinely held-out decisions. Replaying at ${replay.speed}× speed.${meta.person === 4 ? ' <b>This is the hardest person in the study (78% accurate).</b>' : ''}`;
  replayTimer = setInterval(() => {
    const per = Math.round(replay.speed * FS * 0.05);
    for (let k = 0; k < per && replay.i < replay.meta.n; k++, replay.i++) { const j = replay.i * 3; feed(replay.rgb[j], replay.rgb[j + 1], replay.rgb[j + 2]); }
    if (replay.i >= replay.meta.n) { clearInterval(replayTimer); $('phase').textContent = 'Replay finished'; }
  }, 50);
});

// ---------------- shared measurement logic ----------------
function startMeasure(kind, w) {
  if (kind === 'live') { timers.forEach(clearInterval); timers = []; }
  mode = kind; chain = new Chain(w || weights); calSqi = []; calMotion = []; calFailed = false; calChecked = false; strip = []; $('strip').innerHTML = '';
  tally = { normal: 0, hyp: 0, uns: 0, pause: 0, right: 0, wrong: 0 };
  show('measure'); $('btn-restart').hidden = kind !== 'live'; $('ref-tile').hidden = kind !== 'replay'; if (kind !== 'replay') $('replay-note').hidden = true;
  setStatus('calibrating', '◔', 'Calibrating', 'Rest, keep still. Learning your normal heart-rate and light baseline.');
  $('phase').textContent = 'Calibrating'; $('cal-progress').hidden = false; $('tally').textContent = '';
  if (kind === 'live') {
    cam.onSample = (r, g, b) => feed(r, g, b);
    timers.push(setInterval(() => { const m = motion && motion.level(); if (m != null) { $('mot').textContent = m < MOTION_MAX ? 'Still' : 'Moving'; $('mot-u').textContent = m.toFixed(2) + ' m/s²'; } else { $('mot').textContent = '?'; $('mot-u').textContent = 'no sensor'; } }, 500));
  } else { $('mot').textContent = '–'; $('mot-u').textContent = 'n/a in replay'; }
  timers.push(setInterval(() => drawWave($('wave'), chain.ac[1].slice(-300)), 100));
  timers.push(setInterval(tick, 250));
}
$('btn-restart').addEventListener('click', () => startMeasure('live'));
$('btn-stop').addEventListener('click', () => { cleanup(); show('home'); });

function feed(r, g, b) { const d = chain.push(r, g, b); if (d) onWindow(d); }
function tick() { if (!chain) return; const s = chain.n / FS; $('clock').textContent = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; $('cal-fill').style.width = Math.min(100, (s / (CAL + WIN)) * 100) + '%'; }

function setStatus(cls, icon, main, sub) { const el = $('status'); el.className = 'status-card ' + cls; $('status-icon').textContent = icon; $('status-main').textContent = main; $('status-sub').textContent = sub; }
function pushStrip(cls, title) { strip.push([cls, title]); if (strip.length > 90) strip.shift(); $('strip').innerHTML = strip.map(([c, t]) => `<i class="${c}" title="${t}"></i>`).join(''); }

function refAt(s) { // window-mean reference SpO2 (study oximeters), replay only
  const g = replay.meta.spo2.slice(s, s + WIN).filter((v) => v != null); return g.length >= 5 ? g.reduce((a, b) => a + b, 0) / g.length : null;
}

function onWindow(d) {
  const m = motion && motion.level(10);
  if (d.calibrating) {
    calSqi.push(d.sqi); if (mode === 'live') calMotion.push(m != null && m > MOTION_MAX);
    $('hr').textContent = Number.isFinite(d.hr) ? Math.round(d.hr) : '–'; setSig(d.sqi); return;
  }
  if (!calChecked) { calChecked = true; const why = checkCalibration(); if (why) { calFailed = true; setStatus('fail', '✕', 'Calibration failed', why + ' Rest for a couple of minutes, then restart calibration.'); $('phase').textContent = 'Restart needed'; $('btn-restart').hidden = mode !== 'live'; return; } $('phase').textContent = 'Measuring'; $('cal-progress').hidden = true; }
  if (calFailed) return;
  $('hr').textContent = Number.isFinite(d.hr) ? Math.round(d.hr) : '–'; $('dhr').textContent = Number.isFinite(d.dHR) ? (d.dHR >= 0 ? '+' : '') + d.dHR.toFixed(0) : '–'; setSig(d.sqi);
  let ref = null; if (mode === 'replay') { ref = refAt(d.s); $('ref').textContent = ref == null ? '–' : ref.toFixed(0); }
  const moving = mode === 'live' && m != null && m > MOTION_MAX;
  if (moving) { tally.pause++; setStatus('pause', '✋', 'Hold still', 'Movement detected, so this reading is skipped. Rest your hand and the phone on a table.'); pushStrip('d-pause', 'paused: moving'); return; }
  if (d.sqi < SQI_MIN || !Number.isFinite(d.hr)) { tally.uns++; setStatus('uns', '?', 'Unsure, re-measure', 'Weak pulse signal. Cover the camera and flash fully and press lightly.'); pushStrip('d-uns', 'unsure: poor signal'); return; }
  const t = `${d.s + WIN}s p=${d.p.toFixed(2)}`;
  if (d.label === 'unsure') { tally.uns++; setStatus('uns', '?', 'Unsure, re-measure', 'The model is not confident for this window. Keep still; the next reading is 2 seconds away.'); pushStrip('d-uns', 'unsure ' + t); }
  else if (d.label === 'HYPOXIC') { tally.hyp++; setStatus('hyp', '↓', 'Possible low-oxygen episode', 'Your signs look like a drop from your baseline (SpO₂ under 90%). Check with a medical pulse oximeter. If you feel unwell, get medical help.'); pushStrip('d-hyp', 'low-oxygen ' + t); }
  else { tally.normal++; setStatus('normal', '✓', 'Normal vs your baseline', 'No drop from your own baseline detected in this window.'); pushStrip('d-normal', 'normal ' + t); }
  if (ref != null && d.label !== 'unsure') { ((d.label === 'HYPOXIC') === (ref < 90)) ? tally.right++ : tally.wrong++; }
  const n = tally.normal + tally.hyp + tally.uns + tally.pause;
  $('tally').textContent = `${n} readings: ${tally.normal} normal, ${tally.hyp} low-oxygen, ${tally.uns} unsure, ${tally.pause} paused.` + (mode === 'replay' ? ` Against the study reference, ${tally.right} of ${tally.right + tally.wrong} answered readings were correct.` : '');
}
function setSig(q) { $('sig').textContent = q >= SQI_MIN + 0.08 ? 'Good' : q >= SQI_MIN ? 'Fair' : 'Poor'; $('sig-u').textContent = 'quality ' + q.toFixed(2); }

function checkCalibration() {
  const s = chain.calibrationSummary(); const med = [...calSqi].sort((a, b) => a - b)[Math.floor(calSqi.length / 2)];
  if (mode === 'live' && cam && cam.info.maxGap > 1.5) return 'The camera stalled during calibration.';
  if (!Number.isFinite(s.hr)) return 'No pulse was found during calibration.';
  if (med < SQI_MIN) return 'The pulse signal was too weak during calibration.';
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
