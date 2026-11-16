// Contract between the UI layer (app/index.html, app/style.css, app/js/ui.js) and the capture/decision layer (app/js/app.js).
// The UI was rebuilt from scratch on top of the existing pipeline; app.js addresses the screen only through element ids and
// a small set of state class names. This test fails the build if the markup drops an id app.js writes to, if the stylesheet
// loses a state class app.js sets, if a required safety / limitation / attribution sentence disappears, or if the
// accessibility invariants of the new UI regress. Usage: node tests/ui_contract.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  stateFromStatusClass, kickerFor, railFor, fractionFromWidth, dialSubFor, confLineFor, arcOffset, stepText, FIRST_ANSWER_S,
} from '../app/js/ui.js';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const read = (p) => fs.readFileSync(ROOT + p, 'utf8');
const html = read('app/index.html'), css = read('app/style.css'), app = read('app/js/app.js'), ui = read('app/js/ui.js');
const sw = read('app/sw.js');

let n = 0, fail = 0;
const ok = (m) => { n++; console.log('ok  ', m); };
const check = (cond, m) => { if (cond) { n++; } else { console.log('FAIL', m); fail++; } };

// ----------------------------------------------------------------- 1. every id app.js writes to exists in the markup
const MADE_BY_APP = new Set(['capture-debug', 'capture-debug-toggle', 'capture-debug-text']);
const ids = [...new Set([...app.matchAll(/\$\('([a-z0-9-]+)'\)/g)].map((m) => m[1]))].filter((i) => !MADE_BY_APP.has(i));
assert.ok(ids.length > 30, `expected app.js to address many ids, found ${ids.length}`);
for (const id of ids) check(html.includes(`id="${id}"`), `app.js writes to #${id} but index.html has no such id`);
ok(`${ids.length} element ids addressed by app.js are all present in index.html`);

// ids the UI adapter needs, plus the two app.js reaches through a computed lookup the sweep above cannot see
// ($(mode === 'setup' ? 'dbg-setup' : 'dbg-measure'))
for (const id of ['dial-arc', 'dial-pct', 'dial-sub', 'status-kicker', 'conf-level', 'rail-cal', 'rail-res', 'dbg-setup', 'dbg-measure']) {
  check(html.includes(`id="${id}"`), `#${id} is required but missing from index.html`);
}
ok('the ids ui.js mirrors onto, and the two debug panes app.js picks at runtime, are present');

// ----------------------------------------------------------------- 2. state classes app.js sets must be styled
// setStatus() writes 'status-card <cls>'; renderPlace() writes 'place <level>'; pushStrip() writes the strip classes.
for (const c of ['calibrating', 'normal', 'hyp', 'uns', 'pause', 'fail']) {
  check(new RegExp(`\\.status-card[\\s\\S]{0,400}?\\.${c}\\b`).test(css) || css.includes(`.status-card.${c}`), `.status-card.${c} has no style`);
}
for (const c of ['good', 'bad']) check(css.includes(`.place.${c}`), `.place.${c} has no style`);
for (const c of ['d-normal', 'd-hyp', 'd-uns', 'd-pause']) check(css.includes(`.${c}`), `.${c} (decision strip) has no style`);
for (const c of ['.checks li.ok', '.checks li.warn', '.ring.on', '.meter-fill', '#cal-fill', '.screen.active', '.white[hidden]']) {
  check(css.includes(c), `${c} has no style`);
}
// app.js assigns these custom properties inline on the quality meter, so they must exist as tokens
for (const v of ['--green', '--amber', '--red', '--gray']) check(new RegExp(`\\${v}\\s*:`).test(css), `token ${v} is read by app.js but not defined`);
ok('every state class and token app.js sets is defined in style.css');

// ----------------------------------------------------------------- 3. safety, limitation and attribution text survives
const text = html.replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;|&#\d+;/g, ' ').replace(/\s+/g, ' ');
const MUST_SAY = [
  'This project is a prototype and is not medical advice',
  'It does not measure SpO',                                   // "does not measure SpO2, it is not a medical device"
  'it is not a medical device, and it must never replace a pulse oximeter or a clinician',
  'If you feel short of breath, confused, or unwell, get medical help now',
  'Works only',                                                 // seated at rest
  'seated at rest',
  'Detects a',                                                  // drop from your own first minute
  'drop from your own first minute',
  'Validated on',
  'only 6 people',
  'Your camera video never leaves your phone',
  'I understand this is a research prototype, not medical advice',
  'Not medical advice. Not a medical device',
  'Oxygen status at start unknown',
  // verified numbers (recomputed from results/ml by scripts/check_claims.py)
  'time-matched balanced accuracy is 0.809',
  'the nested estimate for the whole selection procedure is 0.779',
  'mean error 2.94 against 3.07 points',
  'it would refuse 6.7% (MTHS) and 8.8% (BUT PPG) of windows',
  '167,588 parameters',
  // attribution
  'UW Hoffman 2022',
  'https://github.com/aravhawk/oxydrop',
];
for (const s of MUST_SAY) check(text.includes(s) || html.includes(s), `required text missing from index.html: "${s}"`);
ok(`${MUST_SAY.length} required safety / limitation / number / attribution strings are still in the page`);
check(!html.includes('—'), 'index.html must not contain an em dash (scripts/check_claims.py enforces this too)');
for (const [name, src] of [['style.css', css], ['js/ui.js', ui]]) check(!src.includes('—'), `${name} must not contain an em dash`);
ok('no em dashes in the UI sources');

// nothing in the UI may present the prototype as a finished medical product
for (const banned of ['diagnose your', 'measures your SpO', 'medical-grade', 'clinically validated']) {
  check(!text.toLowerCase().includes(banned.toLowerCase()), `index.html must not claim "${banned}"`);
}
ok('no overclaiming language in the page');

// ----------------------------------------------------------------- 4. the three modes and the debug view still work
for (const v of ['flash', 'ambient', 'front']) {
  check(new RegExp(`name="light"\\s+value="${v}"`).test(html), `the ${v} light mode radio is missing`);
}
check(/role="radiogroup"/.test(html), 'the light-source group must be a radiogroup');
ok('the flash / ambient / front-camera mode picker is intact');
for (const id of ['debug-setup', 'debug-measure']) {
  const m = new RegExp(`<details[^>]*id="${id}"[^>]*>`).exec(html);
  check(!!m, `${id} must still be a <details>`);
  check(!!m && !/\sopen[\s>]/.test(m[0]), `${id} must be collapsed by default`);
}
check(html.includes('data-dl') && html.includes('data-copy'), 'the raw-trace download and debug-copy buttons must stay');
check(css.includes('.capture-debug') && css.includes('.capture-debug-toggle'), 'the ?debug=1 capture overlay must still be styled');
// the floating toggle must not be dimmed with opacity: that fades its label below 4.5:1 as well
check(!/\.capture-debug-toggle\{[^}]*opacity:/.test(css), 'the capture-debug toggle must not use opacity to look quiet');
ok('both debug views are present, collapsed by default, and the ?debug=1 overlay is styled');

// ----------------------------------------------------------------- 5. accessibility invariants of the new UI
check(/content="width=device-width, initial-scale=1, viewport-fit=cover"/.test(html), 'viewport must allow zoom and cover the safe area');
check(!/maximum-scale|user-scalable\s*=\s*no/.test(html), 'the page must never block pinch zoom');
check(/<html lang="en">/.test(html), 'the document needs a lang attribute');
// every control that is not a <button> with text needs an accessible name
const inputs = [...html.matchAll(/<input\b[^>]*>/g)].map((m) => m[0]);
for (const el of inputs) {
  const id = /id="([^"]+)"/.exec(el);
  const named = /aria-label=/.test(el) || (id && new RegExp(`for="${id[1]}"`).test(html)) || /<label[^>]*>\s*(<[^>]+>\s*)*$/.test(html.slice(0, html.indexOf(el)));
  check(named, `input without an accessible name: ${el.slice(0, 70)}`);
}
check(/<select id="lens"[^>]*aria-label="Camera lens"/.test(html), 'the lens select needs a label');
check(/<select id="lens"[^>]*aria-describedby="lens-help"/.test(html), 'the lens override hint must be associated with the select');
// decorative svg must be hidden from the tree; the one informative drawing must be labelled
check(/class="guide-svg"[^>]*role="img"[^>]*aria-labelledby="guide-title guide-desc"/.test(html), 'the placement illustration must be a labelled role="img"');
check(html.includes('<title id="guide-title">') && html.includes('<desc id="guide-desc">'), 'the illustration needs a title and a description');
const decorativeSvgParents = (html.match(/aria-hidden="true"/g) || []).length;
check(decorativeSvgParents >= 8, 'decorative icons must be hidden from assistive technology');
// colour is never the only channel: each state carries an icon shape and a word
check(css.includes('.place.good .place-ico::after') && css.includes('.place.bad .place-ico::after'), 'placement states need an icon, not only a colour');
check(css.includes('.strip i.d-uns') && css.includes('.strip i.d-pause'), 'the decision strip must differ in shape as well as colour');
// app.js rewrites the placement and verdict text on a timer; a live region on those panels would repeat
// the same sentence every tick, so the announcements go through de-duped sr-only mirrors in ui.js.
check(/<span id="status-announce" class="sr-only" role="status" aria-live="polite">/.test(html), 'the verdict needs a de-duped announcer');
check(/<span id="place-announce" class="sr-only" role="status" aria-live="polite">/.test(html), 'the placement prompt needs a de-duped announcer');
check(!/id="place"[^>]*aria-live/.test(html) && !/id="place"[^>]*role="status"/.test(html), '#place must not itself be a live region');
check(!/class="status-live"[^>]*aria-live/.test(html), '.status-live must not itself be a live region');
check(/announcer\('place-announce'/.test(ui) && /announcer\('status-announce'/.test(ui), 'ui.js must drive both announcers');
// the step rail's state must be real text, not a CSS tick mark on a font-size:0 digit
check(/class="sr-only" id="rail-cal-sr"/.test(html) && /class="sr-only" id="rail-res-sr"/.test(html), 'the step rail needs spoken state text');
check(/<span class="rail-n" aria-hidden="true">/.test(html), 'the rail digits are decorative and must be hidden');
// the visible step word is hidden on narrow measure screens, so the spoken text must name the step
for (const n of ['Place', 'Calibrate', 'Result']) check(new RegExp(`Step \\d of 3, ${n}, `).test(html), `the spoken step text must name ${n}`);
check(/#measure \.rail-t\{display:none\}/.test(css), 'the measure rail drops its labels on narrow screens');
check(css.includes('prefers-reduced-motion'), 'reduced motion must be respected');
check(css.includes('prefers-color-scheme: light'), 'a light theme must be defined');
check(/\.wave\{[^}]*background:#0d1526/.test(css), 'the waveform canvas must keep a fixed dark ground: app.js draws it in fixed colours');
check(!/color:#1a1200/.test(css), 'state glyph colours must follow the theme (--amber-glyph), not be hard-coded');
check(css.includes('forced-colors') || css.includes('prefers-contrast'), 'high-contrast / forced-colors handling must be present');
check(/--tap:\s*5[0-9]px/.test(css), 'the minimum tap target must stay at least 50px');
check(html.includes('class="skip"'), 'a skip link must be present');
ok('accessibility invariants hold (zoom, names, labels, live region, non-colour cues, reduced motion, light theme, tap size)');

// ----------------------------------------------------------------- 6. one-handed layout plumbing
check(css.includes('env(safe-area-inset-bottom)'), 'the bottom dock must respect the home-indicator inset');
check(css.includes('position:sticky') && css.includes('.dock-bottom'), 'the primary actions must stay reachable at the bottom');
check(/min-height:100dvh/.test(css), 'use dynamic viewport height so iOS Safari chrome does not clip the page');
ok('one-handed layout plumbing is in place');

// ----------------------------------------------------------------- 7. ui.js is loaded, precached and side-effect free in Node
check(html.includes('js/ui.js') && html.indexOf('js/ui.js') < html.indexOf('js/app.js'), 'ui.js must be loaded before app.js');
check(sw.includes("'js/ui.js'"), 'ui.js must be precached by the service worker');
check(/const V = 'oxydrop-v3\.6\.1-photon-distill2'/.test(sw), 'the service worker cache version must be bumped for this deploy');
check(!/\b(getUserMedia|CameraSource|Chain|HybridEngine|decide\()/.test(ui), 'ui.js must not touch capture, model or decision logic');
ok('ui.js is wired in, precached, and contains no pipeline logic');

// ----------------------------------------------------------------- 8. ui.js pure helpers
assert.equal(FIRST_ANSWER_S, 70, 'first answer is the 60 s calibration plus the first 10 s window');
assert.equal(stateFromStatusClass('status-card hyp'), 'hyp');
assert.equal(stateFromStatusClass('status-card fail'), 'fail');
assert.equal(stateFromStatusClass('status-card uns'), 'uns');
assert.equal(stateFromStatusClass('status-card pause'), 'pause');
assert.equal(stateFromStatusClass('status-card calibrating'), 'calibrating');
assert.equal(stateFromStatusClass(''), 'calibrating', 'an unknown class never shows a verdict card');
assert.equal(stateFromStatusClass('status-card fail normal'), 'fail', 'a failed baseline wins over any verdict class');
assert.equal(kickerFor('calibrating'), 'Measuring, no answer yet');
assert.equal(kickerFor('fail'), 'No usable baseline');
assert.equal(kickerFor('normal'), 'Oxygen vs your baseline');
assert.deepEqual(railFor('setup', 'cal'), { place: 'on', cal: '', res: '' });
assert.deepEqual(railFor('measure', 'cal'), { place: 'done', cal: 'on', res: '' });
assert.deepEqual(railFor('measure', 'live'), { place: 'done', cal: 'done', res: 'on' });
assert.equal(fractionFromWidth('42.5%'), 0.425);
assert.equal(fractionFromWidth('120%'), 1, 'clamped');
assert.equal(fractionFromWidth('-5%'), 0, 'clamped');
assert.equal(fractionFromWidth(''), 0);
assert.equal(fractionFromWidth('80px'), 0, 'only percentages are meaningful here');
assert.match(dialSubFor(0), /about 70 seconds/);
assert.match(dialSubFor(0.5), /about 35 s/);
assert.match(dialSubFor(1), /any moment/);
assert.equal(confLineFor(0.1), 'Answers are held to a 90% confidence level. Below it, the app says it cannot tell.');
assert.equal(confLineFor(0.05), 'Answers are held to a 95% confidence level. Below it, the app says it cannot tell.');
assert.equal(stepText(2, 'Calibrate', 'on'), 'Step 2 of 3, Calibrate, current.');
assert.equal(stepText(3, 'Result', 'done'), 'Step 3 of 3, Result, done.');
assert.equal(stepText(3, 'Result', ''), 'Step 3 of 3, Result, not started.');
assert.match(confLineFor(undefined), /fixed confidence level/, 'a missing level never invents a number');
assert.match(confLineFor(0), /fixed confidence level/);
assert.ok(Math.abs(arcOffset(0) - 2 * Math.PI * 52) < 1e-6);
assert.ok(Math.abs(arcOffset(1)) < 1e-6);
ok('ui.js pure helpers behave (state mapping, rail, dial, conformal level wording)');

// the default conformal level the app ships must be the one the confidence line will quote
const conformal = JSON.parse(read('app/data/conformal.json'));
assert.equal(conformal.default_alpha, 0.1);
assert.equal(confLineFor(conformal.default_alpha), 'Answers are held to a 90% confidence level. Below it, the app says it cannot tell.');
ok('the confidence line is derived from app/data/conformal.json, not hard-coded');

if (fail) { console.log(`UI CONTRACT FAIL (${fail} problems)`); process.exit(1); }
console.log(`UI CONTRACT PASS (${n} checks)`);
