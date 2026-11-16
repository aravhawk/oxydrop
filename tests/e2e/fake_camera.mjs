// End-to-end browser test of the live-camera path with synthetic fingertip videos (no real camera needed).
// Case "good": a 72 bpm pulse. Expect: lens select works, ambient/flash toggle works, placement reads "Fingertip detected", calibration begins and completes,
//              the default engine is the hybrid CNN (in a Web Worker), heart rate within +-6 of 72 (the CNN heart-rate head reads 4-5 bpm high on this synthetic pulse;
//              the spectral estimate is exact), state "Normal vs your baseline", debug CSV non-empty, zero console errors.
// Case "bad":  grey noise, no finger. Expect: placement refuses, "start anyway" appears after 20 s, result is "Unsure, re-measure", and no heart-rate number.
// ENGINE=<distill|logistic|hybrid> selects ?engine= (unset = no query = the app's default engine, PhotonDistill) and the engine-specific assertions; see tests/e2e/README.md.
// Usage: see tests/e2e/README.md. Env: ENGINE, CHROME (chromium binary), FAKE_GOOD / FAKE_BAD (y4m files), PLAYWRIGHT_CORE (path to playwright-core/index.mjs if not installed locally), CASES=good,bad,replay,stall,calstall
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ? pathToFileURL(process.env.PLAYWRIGHT_CORE).href : 'playwright-core');
const APP = fileURLToPath(new URL('../../app/', import.meta.url));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.webmanifest': 'application/manifest+json', '.bin': 'application/octet-stream' };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, 'http://x').pathname), f = path.normalize(path.join(APP, u === '/' ? 'index.html' : u));
  if (!f.startsWith(APP) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(res);
}).listen(0, '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const URL_BASE = process.env.LIVE_URL || `http://127.0.0.1:${server.address().port}/index.html`;   // LIVE_URL=https://aravhawk.github.io/oxydrop/index.html runs the same cases against the deployed site
// ENGINE unset: the app's default engine with no query string (PhotonDistill); 'distill' | 'logistic' | 'hybrid' add ?engine=
const ENGINE = process.env.ENGINE || '';
const KIND = ENGINE === 'logistic' ? 'logistic' : ENGINE === 'hybrid' ? 'hybrid' : 'distill';
const withQuery = (extra = '') => { const q = [ENGINE ? 'engine=' + ENGINE : '', extra].filter(Boolean).join('&'); return URL_BASE + (q ? '?' + q : ''); };
const URL0 = withQuery();
console.log(`engine under test: ${KIND}${ENGINE ? ' (?engine=' + ENGINE + ')' : ' (default, no ?engine=)'}  url ${URL0}`);

let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  (' + extra + ')' : ''}`); if (!ok) failures++; };
const UNSURE = 'Unsure, re-measure', NORMAL = 'Normal vs your baseline', LOWTXT = 'Possible low oxygen: re-check with a pulse oximeter';
const STATE_TXT = { NORMAL, LOW: LOWTXT, UNSURE };
// reasons decision.js may give a PhotonDistill window besides the model's own answer: measurement validity only (camera, placement, motion, frame rate)
const VALIDITY_REASONS = new Set(['camera_stalled', 'calibration_failed', 'no_finger', 'moving', 'low_fps', 'frame_gaps']);
// One consistent read of the status card and the debug view: the shown state must be exactly what the model line says (abstain -> Unsure, P > 0.5 -> Low, else Normal)
// unless the app reports a measurement-validity reason (the status-sub text then names it).
async function readModelAndState(p) {
  return p.evaluate(() => {
    const dbg = document.getElementById('dbg-measure').textContent, m = /model \S+: raw P\(SpO2<90\) (\S+)  abstain (true|false) \(abstain head (\S+)\)/.exec(dbg);
    return { status: document.getElementById('status-main').textContent, sub: document.getElementById('status-sub').textContent, line: m ? m[0] : null, p: m ? Number(m[1]) : NaN, abstain: m ? m[2] === 'true' : null, conf: m ? Number(m[3]) : NaN };
  });
}
const expectedText = (r) => (r.abstain ? UNSURE : r.p > 0.5 ? LOWTXT : NORMAL);
// a measurement-validity reason (decision.js REASON_TEXT) may replace the model's answer with Unsure; nothing else may
const VALIDITY_TEXT = /calibration minute did not pass|No fingertip detected|Movement detected|running too slowly|dropped frames|camera stopped delivering/i;
const stateMatches = (r) => r.status === expectedText(r) || (r.status === UNSURE && VALIDITY_TEXT.test(r.sub));
// Audit of the debug CSV for the distill engine: every post-calibration decision row is the model's answer or carries a validity reason, never a pulse/saturation/SNR reason.
function auditCsv(csv) {
  const col = csv[0].split(','), I = (n) => col.indexOf(n), rows = csv.slice(1).map((l) => l.split(',')).filter((r) => r[I('state')]);
  const bad = [], n = { rows: rows.length, normal: 0, low: 0, unsure_model: 0, unsure_validity: 0 };
  for (const r of rows) {
    const label = r[I('label')], state = r[I('state')], reason = r[I('reason')], p = Number(r[I('p')]);
    if (reason && !VALIDITY_REASONS.has(reason) && reason !== 'model_unsure') bad.push(`${r[0]}s: reason ${reason}`);
    else if (reason === 'model_unsure') { if (label !== 'unsure' || state !== 'UNSURE') bad.push(`${r[0]}s: model_unsure with label ${label} state ${state}`); n.unsure_model++; }
    else if (reason) { n.unsure_validity++; if (state !== 'UNSURE') bad.push(`${r[0]}s: validity reason ${reason} with state ${state}`); }
    else if (label === 'unsure' || (label === 'HYPOXIC') !== (state === 'LOW') || (label === 'normal') !== (state === 'NORMAL') || (state === 'UNSURE')) bad.push(`${r[0]}s: label ${label} -> state ${state} without a reason`);
    else { if (Number.isFinite(p) && Math.abs(p - 0.5) > 1e-3 && (p > 0.5) !== (state === 'LOW')) bad.push(`${r[0]}s: p ${p} -> ${state}`); state === 'LOW' ? n.low++ : n.normal++; }
  }
  return { bad, n };
}

async function open(y4m, url = URL0) {
  const b = await chromium.launch({ executablePath: process.env.CHROME, args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + y4m] });
  const ctx = await b.newContext({ permissions: ['camera', 'clipboard-read', 'clipboard-write'], viewport: { width: 414, height: 860 }, acceptDownloads: true });
  const p = await ctx.newPage(); const errs = [];
  // test hook: window.__freeze = true makes the page stop receiving camera frames (a stalled / interrupted camera), without touching app code
  await p.addInitScript(() => { const o = HTMLVideoElement.prototype.requestVideoFrameCallback; if (!o) return; HTMLVideoElement.prototype.requestVideoFrameCallback = function (cb) { return o.call(this, (now, meta) => { if (window.__freeze) { this.requestVideoFrameCallback(cb); return; } cb(now, meta); }); }; });
  p.on('pageerror', (e) => errs.push(String(e))); p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(url);
  return { b, p, errs };
}

let modelStateSeen = '';
async function good() {
  console.log('--- case good: 72 bpm fingertip');
  const { b, p, errs } = await open(process.env.FAKE_GOOD || 'fake_good.y4m');
  await p.check('#ack'); await p.click('#btn-live'); await p.click('#btn-cam');
  await p.waitForSelector('#setup-live:not([hidden])', { timeout: 20000 });
  const nLens = await p.$$eval('#lens option', (o) => o.length); check('lens list populated', nLens >= 1, nLens + ' lens');
  await p.selectOption('#lens', { index: 0 }); await p.waitForTimeout(2500);
  check('camera restarted on lens change', /Camera on/.test(await p.textContent('#cam-state')), await p.textContent('#cam-state'));
  await p.check('input[name=light][value=ambient]'); await p.waitForTimeout(1000);
  check('ambient mode disables the flash', /flash off/i.test(await p.textContent('#cam-state')), await p.textContent('#cam-state'));
  await p.check('input[name=light][value=flash]'); await p.waitForTimeout(500);
  await p.waitForFunction(() => document.getElementById('place-title').textContent === 'Fingertip detected', null, { timeout: 30000 });
  check('placement reaches "Fingertip detected"', true);
  check('begin disabled until a lock', await p.isDisabled('#btn-begin'));
  await p.waitForFunction(() => !document.getElementById('btn-begin').disabled, null, { timeout: 60000 });
  check('btn-begin enables after consecutive locked windows', true, await p.textContent('#q-text'));
  check('checks list shown', (await p.$$eval('#checks li', (l) => l.length)) >= 4, (await p.$$eval('#checks li', (l) => l.map((x) => x.textContent))).join(' ; '));
  await p.click('#btn-begin');
  const t0 = Date.now();
  await p.waitForFunction(() => /Measuring|Restart/.test(document.getElementById('phase').textContent), null, { timeout: 120000 });
  console.log('       calibration took', ((Date.now() - t0) / 1000).toFixed(0), 's; phase =', await p.textContent('#phase'));
  check('calibration completes', /Measuring/.test(await p.textContent('#phase')), await p.textContent('#status-sub'));
  await p.waitForFunction((s) => document.getElementById('status-main').textContent !== 'Calibrating' && document.getElementById('hr').textContent !== '–', null, { timeout: 30000 });
  await p.waitForTimeout(8000);
  const hr = Number(await p.textContent('#hr')), st = await p.textContent('#status-main');
  check('heart rate within 72 +-6 (v2 spectral / pulse-lock HR display; the hybrid adds its CNN HR head)', Math.abs(hr - 72) <= 6, 'hr=' + hr);
  const dbg = await p.textContent('#dbg-measure');
  const engLine = dbg.split('\n').find((l) => l.startsWith('engine'));
  if (KIND === 'distill') {
    check('default engine is PhotonDistill (nn_runtime students, page thread)', /engine oxydrop-photon-distill2-v1 \(default engine slot, \?engine=distill, model file data\/distill\.json\): .*2,435 \/ 2,435 \/ 6,915 parameters per seed \(11,785 in total\), runs in the page thread; output shown as the model gives it \(its own abstention = Unsure\)/.test(dbg), engLine);
    check('debug view shows the model line: engine id, raw P(SpO2<90), abstain flag and abstain-head confidence', /model oxydrop-photon-distill2-v1: raw P\(SpO2<90\) \d\.\d{4}  abstain (true|false) \(abstain head \d\.\d{4}\)/.test(dbg), dbg.split('\n').find((l) => l.startsWith('model')));
    check('experimental SpO2 only in the debug view, labelled', /EXPERIMENTAL SpO2 estimate of the model \(not validated, do not use\)/.test(dbg) && !/SpO/.test(await p.textContent('#status')));
  } else if (KIND === 'logistic') {
    check('?engine=logistic runs the v2 logistic chain', /engine logistic chain v2 \(baseline \/ fallback\) \(\?engine=logistic\)/.test(dbg), engLine);
  } else {
    check('?engine=hybrid is the hybrid CNN, in a Web Worker', /engine oxydrop-hybrid-v1: .*167,588 parameters per seed.*runs in a Web Worker; conformal alpha 0\.1 /.test(dbg), engLine);
    check('experimental SpO2 only in the debug view, labelled', /EXPERIMENTAL SpO2 from the CNN head \(not validated, do not use\)/.test(dbg) && !/SpO/.test(await p.textContent('#status')));
  }
  if (KIND !== 'logistic') check('latency per window is reported', /latency per 2 s window: last [\d.]+ ms, median [\d.]+ ms, p95 [\d.]+ ms/.test(dbg), engLine);
  if (KIND === 'distill') {
    // no app-side override: the shown state is the model's own output (abstain -> Unsure, P > 0.5 -> Low, else Normal). Re-read once if a new window arrived between the reads.
    let r = await readModelAndState(p);
    for (let i = 0; i < 4 && !(r.line && stateMatches(r)); i++) { await p.waitForTimeout(700); r = await readModelAndState(p); }   // the debug text and the card refresh on different ticks
    check('state shown equals the model line (abstain -> Unsure, P > 0.5 -> Low, else Normal)', !!r.line && stateMatches(r), `${r.status} | ${r.line} | ${r.sub.slice(0, 80)}`);
    modelStateSeen = r.status;
  } else check('state is NORMAL text', st === NORMAL, st);
  console.log('       dHR', await p.textContent('#dhr'), '| sig', await p.textContent('#sig'), '| tally', await p.textContent('#tally'));
  await p.click('#debug-measure > summary');
  const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#debug-measure [data-dl]')]);
  const csv = fs.readFileSync(await dl.path(), 'utf8').trim().split('\n');
  check('debug CSV non-empty', csv.length > 500 && csv[0].startsWith('t_s,R,G,B,'), csv.length + ' lines');
  if (KIND === 'distill') {
    const a = auditCsv(csv);
    console.log('       CSV audit', JSON.stringify(a.n), a.bad.length ? 'VIOLATIONS ' + a.bad.slice(0, 5).join(' ; ') : '');
    check('debug CSV carries decisions (rows with a state)', a.n.rows >= 3, JSON.stringify(a.n));
    check('every decision row is the model answer or a measurement-validity reason (no pulse / saturation / SNR override)', a.bad.length === 0, a.bad.slice(0, 5).join(' ; '));
  } else check('debug CSV carries decisions', csv.some((l) => l.includes(',NORMAL,')));
  await p.click('#debug-measure [data-copy]'); await p.waitForTimeout(300);
  const clip = await p.evaluate(() => navigator.clipboard.readText()).catch(() => '');
  check('copy debug summary works', /fps/.test(clip) && /hrBeat/.test(clip), clip.split('\n')[0]);
  check('debug text populated', (await p.textContent('#dbg-measure')).includes('SQI'));
  await p.screenshot({ path: process.env.SHOT_GOOD || 'live_good.png' });
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close();
}

async function bad() {
  console.log('--- case bad: noise only, no finger');
  const { b, p, errs } = await open(process.env.FAKE_BAD || 'fake_bad.y4m');
  await p.check('#ack'); await p.click('#btn-live'); await p.click('#btn-cam');
  await p.waitForSelector('#setup-live:not([hidden])', { timeout: 20000 });
  await p.waitForFunction(() => document.getElementById('place-title').textContent === 'Cover the lens AND the flash', null, { timeout: 15000 });
  check('placement refuses a grey frame', true, await p.textContent('#place-msg'));
  check('begin disabled without a lock', await p.isDisabled('#btn-begin'));
  check('start-anyway hidden at first', await p.isHidden('#btn-begin-anyway'));
  await p.waitForSelector('#btn-begin-anyway:not([hidden])', { timeout: 40000 });
  check('start-anyway offered after 20 s without a lock', true);
  check('begin still disabled', await p.isDisabled('#btn-begin'));
  await p.click('#btn-begin-anyway');
  await p.waitForFunction((u) => document.getElementById('status-main').textContent === u, UNSURE, { timeout: 120000 });
  await p.waitForTimeout(2500);
  const hr = await p.textContent('#hr'), st = await p.textContent('#status-main');
  check('state is UNSURE', st === UNSURE, st + ' | ' + await p.textContent('#status-sub'));
  check('no heart-rate number shown', hr === '–', 'hr=' + hr);
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close();
}

async function replayCase(rec, engine = KIND) {
  console.log('--- case replay ' + rec + ' (' + engine + '): recorded Hoffman demo, same decision function, no camera');
  const { b, p, errs } = await open(process.env.FAKE_GOOD || 'fake_good.y4m', engine === 'logistic' ? `${URL_BASE}?rec=${rec}&engine=logistic` : withQuery('rec=' + rec));
  await p.click('#btn-replay');
  await p.waitForFunction(() => document.getElementById('phase').textContent === 'Replay finished', null, { timeout: 120000 });
  const tally = await p.textContent('#tally'), st = await p.textContent('#status-main'), hr = await p.textContent('#hr');
  console.log('       ', tally, '| last state:', st, '| hr:', hr);
  const note = await p.textContent('#replay-note'), dbgm = await p.textContent('#dbg-measure');
  const trainedNote = /Model trained on this person: demo only, not a test/.test(note) && /demo only, not a test/.test(tally);
  if (engine === 'distill') {
    // the final PhotonDistill refit saw all 6 Hoffman people, so the replay is a demo, not a test; the screen must show the model's own states
    check('replay says the model was trained on this person (demo only, not a test)', trainedNote, note.slice(0, 120));
    check('replay runs PhotonDistill and the debug view shows its model line', /engine oxydrop-photon-distill2-v1 /.test(dbgm) && /model oxydrop-photon-distill2-v1: raw P\(SpO2<90\) \d\.\d{4}  abstain (true|false) \(abstain head \d\.\d{4}\)/.test(dbgm), (dbgm.split('\n').find((l) => l.startsWith('model')) || '').slice(0, 120));
    await p.click('#debug-measure > summary');
    const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#debug-measure [data-dl]')]);
    const csv = fs.readFileSync(await dl.path(), 'utf8').trim().split('\n'), a = auditCsv(csv);
    console.log('       CSV audit', JSON.stringify(a.n), a.bad.length ? 'VIOLATIONS ' + a.bad.slice(0, 5).join(' ; ') : '');
    check('replay: every decision is the model answer (abstain -> "Unsure, re-measure") or a measurement-validity reason', a.n.rows > 100 && a.bad.length === 0, JSON.stringify(a.n) + ' ' + a.bad.slice(0, 3).join(' ; '));
    const last = await readModelAndState(p);
    check('replay: the last shown state equals the last model line', !!last.line && (stateMatches(last) || /moving|stopped|dropped|slowly|fingertip/i.test(last.sub)), `${last.status} | ${last.line}`);
    check('person ' + rec + ' gets some answers or abstentions in the tally', /\d+ readings: \d+ normal, \d+ possible low oxygen, \d+ unsure/.test(tally), tally);
  } else if (engine === 'hybrid') { check('replay says the model was trained on this person (demo only, not a test)', trainedNote, note.slice(0, 120)); check('replay runs the hybrid engine', /engine oxydrop-hybrid-v1/.test(dbgm)); }
  else { check('?engine=logistic replay runs the logistic chain and says it is held out', /fitted without/.test(note) && /logistic chain v2/.test(dbgm), note.slice(0, 120)); }
  if (engine !== 'distill') {   // the pulse-lock "unsure throughout" behaviour belongs to the hybrid / logistic chains; the distill model decides for itself (reported above)
    if (rec === '100004R') { check('person 4 (43 bpm) is UNSURE throughout', st === UNSURE && /: 0 normal, 0 possible low oxygen,/.test(tally), tally); check('no HR number without a lock', hr === '–', hr); }
    else check('person 1 gets confident answers', /[1-9]\d* (normal|possible)/.test(tally) || / [1-9]\d* normal/.test(tally), tally);
  }
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close();
}

// camera.js samples frames from a requestAnimationFrame fallback loop when the native video-frame callback is silent for 250 ms, so blocking the callback alone no longer
// stops the camera: also pause the <video> (a paused element shows the same frame, and the app's frame-dedup drops it), and resume it on unfreeze.
const freeze = (p, on) => p.evaluate((f) => { window.__freeze = f; document.querySelectorAll('video').forEach((v) => { if (f) v.pause(); else v.play().catch(() => {}); }); }, on);
async function startLive(p) {
  await p.check('#ack'); await p.click('#btn-live'); await p.click('#btn-cam');
  await p.waitForSelector('#setup-live:not([hidden])', { timeout: 20000 });
  await p.waitForFunction(() => !document.getElementById('btn-begin').disabled, null, { timeout: 90000 });
  await p.click('#btn-begin');
}
const txt = (p, id) => p.textContent('#' + id);

// A stalled camera must flip the answer to UNSURE within ~2 s (no frames, so no new window would ever replace the old one), hide the HR, and recover afterwards.
async function stall() {
  console.log('--- case stall: camera stops delivering frames for 4 s after calibration');
  const { b, p, errs } = await open(process.env.FAKE_GOOD || 'fake_good.y4m');
  await startLive(p);
  await p.waitForFunction(() => document.getElementById('status-main').textContent === 'Normal vs your baseline', null, { timeout: 150000 });
  check('NORMAL before the stall', true, 'hr=' + await txt(p, 'hr'));
  await freeze(p, true); await p.waitForTimeout(2600);
  check('stalled camera: state flips to UNSURE within 2.6 s', await txt(p, 'status-main') === UNSURE, await txt(p, 'status-main') + ' | ' + await txt(p, 'status-sub'));
  check('stalled camera: reason names the camera', /camera stopped delivering/i.test(await txt(p, 'status-sub')));
  check('stalled camera: heart-rate tile is an en dash', await txt(p, 'hr') === '–', await txt(p, 'hr'));
  await p.waitForTimeout(1400); await freeze(p, false);
  await p.waitForTimeout(3000);
  check('after the stall: still UNSURE, for the right reason (dropped frames, not "no fingertip")', await txt(p, 'status-main') === UNSURE && /dropped frames/i.test(await txt(p, 'status-sub')), await txt(p, 'status-main') + ' | ' + await txt(p, 'status-sub'));
  await p.waitForFunction(() => document.getElementById('status-main').textContent === 'Normal vs your baseline', null, { timeout: 90000 });
  check('recovers to NORMAL after the gap leaves the window', true, 'hr=' + await txt(p, 'hr'));
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close();
}
// A single stall longer than 1.5 s anywhere in the calibration minute rejects the calibration.
async function calstall() {
  console.log('--- case calstall: camera stops for 3 s during calibration');
  const { b, p, errs } = await open(process.env.FAKE_GOOD || 'fake_good.y4m');
  await startLive(p);
  await p.waitForTimeout(25000); await freeze(p, true); await p.waitForTimeout(3000); await freeze(p, false);
  await p.waitForFunction(() => /Measuring|Restart/.test(document.getElementById('phase').textContent), null, { timeout: 120000 });
  check('calibration is rejected', /Restart/.test(await txt(p, 'phase')), await txt(p, 'phase') + ' | ' + await txt(p, 'status-sub'));
  check('reason says the camera stalled', /stalled during calibration/i.test(await txt(p, 'status-sub')), await txt(p, 'status-sub'));
  check('state is UNSURE', await txt(p, 'status-main') === UNSURE);
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close();
}

const cases = (process.env.CASES || 'good,bad,replay,replaylog,stall,calstall').split(',');
try { if (cases.includes('good')) await good(); if (cases.includes('bad')) await bad(); if (cases.includes('replay')) { await replayCase('100001L'); await replayCase('100004R'); } if (cases.includes('replaylog')) { await replayCase('100001L', 'logistic'); await replayCase('100004R', 'logistic'); } if (cases.includes('stall')) await stall(); if (cases.includes('calstall')) await calstall(); }
catch (e) { console.log('FAIL  exception:', e.message); failures++; }
server.close();
console.log(failures ? `E2E FAIL (${failures})` : 'E2E PASS');
process.exit(failures ? 1 : 0);
