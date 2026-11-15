// End-to-end browser test of the live-camera path with synthetic fingertip videos (no real camera needed).
// Case "good": a 72 bpm pulse. Expect: lens select works, ambient/flash toggle works, placement reads "Fingertip detected", calibration begins and completes,
//              heart rate within +-3 of 72, state "Normal vs your baseline", debug CSV non-empty, zero console errors.
// Case "bad":  grey noise, no finger. Expect: placement refuses, "start anyway" appears after 20 s, result is "Unsure, re-measure", and no heart-rate number.
// Usage: see tests/e2e/README.md. Env: CHROME (chromium binary), FAKE_GOOD / FAKE_BAD (y4m files), PLAYWRIGHT_CORE (path to playwright-core/index.mjs if not installed locally), CASES=good,bad,replay
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
const URL0 = `http://127.0.0.1:${server.address().port}/index.html`;

let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  (' + extra + ')' : ''}`); if (!ok) failures++; };
const UNSURE = 'Unsure, re-measure', NORMAL = 'Normal vs your baseline';

async function open(y4m, url = URL0) {
  const b = await chromium.launch({ executablePath: process.env.CHROME, args: ['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + y4m] });
  const ctx = await b.newContext({ permissions: ['camera', 'clipboard-read', 'clipboard-write'], viewport: { width: 414, height: 860 }, acceptDownloads: true });
  const p = await ctx.newPage(); const errs = [];
  p.on('pageerror', (e) => errs.push(String(e))); p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(url);
  return { b, p, errs };
}

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
  check('heart rate within 72 +-3', Math.abs(hr - 72) <= 3, 'hr=' + hr);
  check('state is NORMAL text', st === NORMAL, st);
  console.log('       dHR', await p.textContent('#dhr'), '| sig', await p.textContent('#sig'), '| tally', await p.textContent('#tally'));
  await p.click('#debug-measure > summary');
  const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#debug-measure [data-dl]')]);
  const csv = fs.readFileSync(await dl.path(), 'utf8').trim().split('\n');
  check('debug CSV non-empty', csv.length > 500 && csv[0].startsWith('t_s,R,G,B,'), csv.length + ' lines');
  check('debug CSV carries decisions', csv.some((l) => l.includes(',NORMAL,')));
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

async function replayCase(rec) {
  console.log('--- case replay ' + rec + ': recorded Hoffman demo, same decision function, no camera');
  const { b, p, errs } = await open(process.env.FAKE_GOOD || 'fake_good.y4m', URL0 + '?rec=' + rec);
  await p.click('#btn-replay');
  await p.waitForFunction(() => document.getElementById('phase').textContent === 'Replay finished', null, { timeout: 120000 });
  const tally = await p.textContent('#tally'), st = await p.textContent('#status-main'), hr = await p.textContent('#hr');
  console.log('       ', tally, '| last state:', st, '| hr:', hr);
  if (rec === '100004R') { check('person 4 (43 bpm) is UNSURE throughout', st === UNSURE && /: 0 normal, 0 possible low oxygen,/.test(tally), tally); check('no HR number without a lock', hr === '–', hr); }
  else check('person 1 gets confident answers', /[1-9]\d* (normal|possible)/.test(tally) || / [1-9]\d* normal/.test(tally), tally);
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close();
}

const cases = (process.env.CASES || 'good,bad,replay').split(',');
try { if (cases.includes('good')) await good(); if (cases.includes('bad')) await bad(); if (cases.includes('replay')) { await replayCase('100001L'); await replayCase('100004R'); } }
catch (e) { console.log('FAIL  exception:', e.message); failures++; }
server.close();
console.log(failures ? `E2E FAIL (${failures})` : 'E2E PASS');
process.exit(failures ? 1 : 0);
