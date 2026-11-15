// Runtime behaviour of the presentation adapter app/js/ui.js.
// tests/ui_contract.mjs checks the markup and ui.js's pure helpers; tests/flow.mjs drives the real app.js but its fake
// document returns [] from querySelectorAll and auto-creates any id, so neither of them ever runs ui.js's DOM wiring.
// This test does: it installs a minimal fake DOM (ids, .screen lookups, attributes, classList, style, and a MutationObserver
// stand-in that fires the observer callback synchronously when a watched attribute changes) BEFORE importing ui.js, then
// writes exactly what app.js writes - the status card's class (setStatus), #cal-progress.hidden, #cal-fill.style.width,
// and the .screen.active toggle from show() - and asserts the chrome ui.js owns follows along.
// No capture, probe, model or inference code is touched or imported here.
// Usage: node tests/ui_flow.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';

let nOk = 0; const ok = (name) => { nOk++; console.log('ok  ', name); };

// ---------------- fake DOM (just enough for ui.js) ----------------
const els = new Map();
class El {
  constructor(id = '') {
    this.id = id; this.textContent = ''; this.attrs = new Map(); this.obs = [];
    this.classes = new Set();
    // setting .style.<prop> mutates the style attribute in a real browser, so it must notify like one
    this.style = new Proxy({}, { set: (t, p, v) => { t[p] = v; this.notify('style'); return true; } });
    const self = this;
    this.classList = {
      add(c) { self.classes.add(c); self.notify('class'); },
      remove(c) { self.classes.delete(c); self.notify('class'); },
      contains(c) { return self.classes.has(c); },
      toggle(c, on) { const want = on === undefined ? !self.classes.has(c) : !!on; want ? self.classes.add(c) : self.classes.delete(c); self.notify('class'); },
    };
  }
  get className() { return [...this.classes].join(' '); }
  set className(v) { this.classes = new Set(String(v).split(/\s+/).filter(Boolean)); this.notify('class'); }
  get hidden() { return this.attrs.has('hidden'); }
  set hidden(v) { v ? this.attrs.set('hidden', '') : this.attrs.delete('hidden'); this.notify('hidden'); }
  setAttribute(name, value) {
    if (name === 'class') { this.className = value; return; }
    if (name === 'hidden') { this.hidden = true; return; }
    this.attrs.set(name, String(value)); this.notify(name);
  }
  getAttribute(name) { return name === 'class' ? this.className : (this.attrs.has(name) ? this.attrs.get(name) : null); }
  removeAttribute(name) { if (name === 'hidden') { this.hidden = false; return; } this.attrs.delete(name); this.notify(name); }
  notify(attributeName) { for (const o of this.obs) if (!o.filter || o.filter.includes(attributeName)) o.cb([{ type: 'attributes', attributeName, target: this }], o.observer); }
}
const $ = (id) => { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); };
const SCREENS = ['home', 'setup', 'measure'];
for (const id of SCREENS) $(id).className = 'screen';
$('home').classList.add('active');

globalThis.document = {
  readyState: 'complete',
  getElementById: (id) => $(id),
  querySelector: (sel) => {
    if (sel === '.screen.active') return SCREENS.map($).find((s) => s.classList.contains('screen') && s.classList.contains('active')) || null;
    return null;
  },
  querySelectorAll: (sel) => (sel === '.screen' ? SCREENS.map($) : []),
  addEventListener() {}, removeEventListener() {},
};
globalThis.MutationObserver = class {
  constructor(cb) { this.cb = cb; }
  observe(el, opts = {}) { el.obs.push({ cb: this.cb, filter: opts.attributeFilter, observer: this }); }
  disconnect() {}
  takeRecords() { return []; }
};
globalThis.fetch = async (url) => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(new URL('../app/' + url, import.meta.url), 'utf8')) });

// the state app.js leaves the page in before a measurement starts (as index.html ships it)
$('status').className = 'status-card calibrating';
$('measure').setAttribute('data-phase', 'cal');
$('measure').setAttribute('data-state', 'calibrating');
$('cal-progress').hidden = false;
$('rail-cal').className = 'rail-step on';
$('rail-res').className = 'rail-step';

// ---------------- the adapter, imported only after the fakes exist ----------------
const ui = await import('../app/js/ui.js');
const { arcOffset, dialSubFor, FIRST_ANSWER_S } = ui;

assert.equal($('measure').getAttribute('data-state'), 'calibrating', 'init() must publish a state at once');
assert.equal($('dial-arc').getAttribute('stroke-dasharray'), (2 * Math.PI * 52).toFixed(1), 'init() must size the dial arc');
assert.equal($('dial-arc').getAttribute('stroke-dashoffset'), (2 * Math.PI * 52).toFixed(1), 'an empty dial starts fully offset');
assert.equal($('status-kicker').textContent, 'Measuring, no answer yet', 'the kicker must never state a result while calibrating');
// on the home screen no measurement step is current yet
assert.equal($('rail-cal').className, 'rail-step', 'off the measure screen, step 2 is not current');
assert.equal($('rail-res').getAttribute('aria-current'), null);
ok('ui.js wires itself up on import and publishes the initial state, dial and rail');

// ---------------- app.js show('measure'): the step rail follows the active screen ----------------
SCREENS.map($).forEach((s) => s.classList.toggle('active', s.id === 'measure'));
assert.equal($('measure').getAttribute('data-phase'), 'cal', 'the calibration bar is still visible, so the phase is cal');
assert.equal($('rail-cal').className, 'rail-step on', 'step 2 becomes current on the measure screen');
assert.equal($('rail-cal').getAttribute('aria-current'), 'step');
assert.equal($('rail-res').className, 'rail-step');
ok('the active .screen toggle moves the step rail onto step 2 (Calibrate)');

// ---------------- (a) the status card's class drives the next-steps card ----------------
$('status').className = 'status-card hyp';
assert.equal($('measure').getAttribute('data-state'), 'hyp', 'setStatus("hyp", ...) must select the hyp next-steps card');
assert.equal($('status-kicker').textContent, 'Oxygen vs your baseline');
// every class app.js hands to setStatus has to land on a state the measure screen can select
for (const [cls, state] of [['calibrating', 'calibrating'], ['normal', 'normal'], ['hyp', 'hyp'], ['uns', 'uns'], ['pause', 'pause']]) {
  $('status').className = 'status-card ' + cls;
  assert.equal($('measure').getAttribute('data-state'), state, `status-card ${cls} must publish data-state="${state}"`);
}
$('status').className = 'status-card hyp';
ok('status-card class -> measure data-state and kicker, for every class setStatus writes');

// ---------------- (d) #cal-fill width drives the dial ----------------
$('cal-fill').style.width = '45%';
assert.equal($('dial-arc').getAttribute('stroke-dashoffset'), arcOffset(0.45).toFixed(1), 'the dial arc must follow the progress fill');
assert.equal($('dial-pct').textContent, '45%');
assert.equal($('dial-sub').textContent, dialSubFor(0.45));
assert.match($('dial-sub').textContent, /First answer in about \d+ s\./, $('dial-sub').textContent);
$('cal-fill').style.width = '100%';
assert.ok(Math.abs(Number($('dial-arc').getAttribute('stroke-dashoffset'))) < 0.05, 'a full minute closes the arc');
assert.equal($('dial-pct').textContent, '100%');
assert.equal($('dial-sub').textContent, 'First answer any moment now.');
assert.equal(FIRST_ANSWER_S, 70, 'the dial counts down the calibration minute plus one window');
ok('#cal-fill width -> dial arc offset, percentage and the time-to-first-answer line');

// ---------------- (b) hiding the calibration bar ends the calibration phase ----------------
$('cal-progress').hidden = true;
assert.equal($('measure').getAttribute('data-phase'), 'live', 'hiding #cal-progress means the measurement is live');
assert.equal($('rail-cal').className, 'rail-step done', 'step 2 is finished');
assert.equal($('rail-cal').getAttribute('aria-current'), null, 'a finished step is no longer the current one');
assert.equal($('rail-res').className, 'rail-step on', 'step 3 becomes current');
assert.equal($('rail-res').getAttribute('aria-current'), 'step');
ok('#cal-progress hidden -> data-phase="live" and the step rail advances to step 3 (Result)');

// ---------------- (c) a failed baseline wins over the progress bar ----------------
$('status').className = 'status-card fail';
assert.equal($('measure').getAttribute('data-state'), 'fail');
assert.equal($('measure').getAttribute('data-phase'), 'fail', 'a failed calibration is a phase of its own, bar hidden');
assert.equal($('status-kicker').textContent, 'No usable baseline');
$('cal-progress').hidden = false;   // app.js re-shows the bar when the minute is restarted; until the class changes it is still a failure
assert.equal($('measure').getAttribute('data-phase'), 'fail', 'the status class, not the progress bar, decides the fail phase');
$('status').className = 'status-card calibrating';
assert.equal($('measure').getAttribute('data-phase'), 'cal', 'restarting the minute returns to the calibration phase');
assert.equal($('rail-cal').className, 'rail-step on', 'and the rail goes back to step 2');
ok('status-card fail -> data-phase="fail" whatever #cal-progress says; a restart returns to cal');

// ---------------- the confidence line comes from the shipped conformal table ----------------
await new Promise((r) => setTimeout(r, 0));
const alpha = JSON.parse(fs.readFileSync(new URL('../app/data/conformal.json', import.meta.url), 'utf8')).default_alpha;
assert.equal($('conf-level').textContent, ui.confLineFor(alpha), 'the confidence sentence must be fetched, not left as the markup fallback');
assert.match($('conf-level').textContent, /\d+% confidence level/, $('conf-level').textContent);
ok('the confidence line is filled in at runtime from app/data/conformal.json');

console.log(`UI FLOW PASS (${nOk} checks)`);
process.exit(0);
