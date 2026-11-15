// Presentation adapter for the new UI. It owns nothing the pipeline owns: it only *watches* the DOM that
// app.js already writes (the status card's class, the calibration progress fill, which .screen is active) and
// mirrors it onto the chrome that pure CSS cannot reach: the step rail, the calibration dial, the phase and
// state attributes that select the "what to do next" card, and the plain-language confidence line.
//
// No capture, probe, signal-processing, model, worker or inference logic lives here, and nothing here calls
// into any of it. Everything is derived from values app.js has already put on screen, so the two can be
// changed independently. The pure helpers are exported for tests/ui_contract.mjs.

import { CAL, WIN } from './engine.js';

/** Seconds from the start of the measurement to the first answer: the calibration minute plus one window. */
export const FIRST_ANSWER_S = CAL + WIN;

/** Status-card class (as app.js sets it) -> the data-state the measure screen uses to pick a next-steps card. */
export function stateFromStatusClass(className = '') {
  const c = String(className).split(/\s+/);
  for (const k of ['fail', 'normal', 'hyp', 'uns', 'pause', 'calibrating']) if (c.includes(k)) return k;
  return 'calibrating';
}

/** Kicker above the verdict. Never states a result: it only says what the card below is about. */
export function kickerFor(state) {
  if (state === 'calibrating') return 'Measuring, no answer yet';
  if (state === 'fail') return 'No usable baseline';
  return 'Oxygen vs your baseline';
}

/** Spoken description of one step, so the rail does not depend on a CSS-only tick mark and the visible
 *  word can be dropped on a narrow screen without losing anything. */
export function stepText(n, name, state) {
  const word = state === 'on' ? 'current' : state === 'done' ? 'done' : 'not started';
  return `Step ${n} of 3, ${name}, ${word}.`;
}

/** Which of the three steps is current, and which are done. screen: 'home' | 'setup' | 'measure'. */
export function railFor(screen, phase) {
  if (screen !== 'measure') return { place: 'on', cal: '', res: '' };
  if (phase === 'cal') return { place: 'done', cal: 'on', res: '' };
  return { place: 'done', cal: 'done', res: 'on' };
}

/** "45%" -> 0.45. Anything unparseable is 0. */
export function fractionFromWidth(width) {
  const m = /^\s*(-?[\d.]+)\s*%\s*$/.exec(String(width || ''));
  if (!m) return 0;
  const v = Number(m[1]);
  return Number.isFinite(v) ? Math.max(0, Math.min(1, v / 100)) : 0;
}

/** Text under the calibration dial: how long until the first answer can appear. */
export function dialSubFor(frac, firstAnswerS = FIRST_ANSWER_S) {
  const left = Math.max(0, Math.ceil(firstAnswerS * (1 - frac)));
  if (frac <= 0) return `Rest for about ${firstAnswerS} seconds.`;
  if (left <= 0) return 'First answer any moment now.';
  return `First answer in about ${left} s.`;
}

/** Plain-language confidence line from the shipped conformal table's own default level. */
export function confLineFor(alpha) {
  const a = Number(alpha);
  if (!Number.isFinite(a) || !(a > 0) || !(a < 1)) return 'Answers are held to a fixed confidence level set in advance.';
  const pct = Math.round((1 - a) * 100);
  return `Answers are held to a ${pct}% confidence level. Below it, the app says it cannot tell.`;
}

/** Dash offset for the dial arc of radius r at fraction frac. */
export function arcOffset(frac, r = 52) {
  const c = 2 * Math.PI * r;
  return c * (1 - Math.max(0, Math.min(1, frac)));
}

// ------------------------------------------------------------------ DOM wiring
function init() {
  const $ = (id) => document.getElementById(id);
  const measure = $('measure'), status = $('status'), calProgress = $('cal-progress'), calFill = $('cal-fill');
  const dialArc = $('dial-arc'), dialPct = $('dial-pct'), dialSub = $('dial-sub'), kicker = $('status-kicker');
  const conf = $('conf-level'), railCal = $('rail-cal'), railRes = $('rail-res');
  const railCalSr = $('rail-cal-sr'), railResSr = $('rail-res-sr');
  if (!measure || !status) return;

  const C = 2 * Math.PI * 52;
  if (dialArc) { dialArc.setAttribute('stroke-dasharray', C.toFixed(1)); dialArc.setAttribute('stroke-dashoffset', C.toFixed(1)); }

  const activeScreen = () => {
    const s = document.querySelector('.screen.active');
    return s ? s.id : 'home';
  };

  function syncState() {
    const state = stateFromStatusClass(status.className);
    measure.setAttribute('data-state', state);
    if (kicker) kicker.textContent = kickerFor(state);
    const phase = state === 'fail' ? 'fail' : (calProgress && calProgress.hidden ? 'live' : 'cal');
    measure.setAttribute('data-phase', phase);
    const rail = railFor(activeScreen(), phase);
    if (railCal) railCal.className = 'rail-step ' + rail.cal;
    if (railRes) railRes.className = 'rail-step ' + rail.res;
    if (railCal) rail.cal === 'on' ? railCal.setAttribute('aria-current', 'step') : railCal.removeAttribute('aria-current');
    if (railRes) rail.res === 'on' ? railRes.setAttribute('aria-current', 'step') : railRes.removeAttribute('aria-current');
    if (railCalSr) railCalSr.textContent = stepText(2, 'Calibrate', rail.cal);
    if (railResSr) railResSr.textContent = stepText(3, 'Result', rail.res);
  }

  function syncDial() {
    if (!calFill) return;
    const frac = fractionFromWidth(calFill.style.width);
    if (dialArc) dialArc.setAttribute('stroke-dashoffset', arcOffset(frac).toFixed(1));
    if (dialPct) dialPct.textContent = Math.round(frac * 100) + '%';
    if (dialSub) dialSub.textContent = dialSubFor(frac);
  }

  const obs = new MutationObserver(() => { syncState(); syncDial(); });
  obs.observe(status, { attributes: true, attributeFilter: ['class'] });
  if (calProgress) obs.observe(calProgress, { attributes: true, attributeFilter: ['hidden'] });
  if (calFill) obs.observe(calFill, { attributes: true, attributeFilter: ['style'] });
  // which screen is active is toggled on the sections themselves
  const screens = new MutationObserver(syncState);
  document.querySelectorAll('.screen').forEach((s) => screens.observe(s, { attributes: true, attributeFilter: ['class'] }));

  syncState(); syncDial();

  // app.js rewrites the placement and verdict strings on a timer, and reassigning textContent makes a
  // polite live region speak again even when the sentence has not changed. These mirrors speak once per
  // actual change, so the panels themselves are no longer live regions.
  function announcer(targetId, sourceIds) {
    const out = $(targetId), src = sourceIds.map($).filter(Boolean);
    if (!out || !src.length) return;
    let last = null;
    const say = () => {
      const t = src.map((e) => (e.textContent || '').trim()).filter(Boolean).join('. ');
      if (t !== last) { last = t; out.textContent = t; }
    };
    const mo = new MutationObserver(say);
    for (const e of src) mo.observe(e, { subtree: true, childList: true, characterData: true });
    say();
  }
  announcer('place-announce', ['place-title', 'place-msg']);
  announcer('status-announce', ['status-main', 'status-sub']);

  // the front-camera overlay covers the whole screen, so the page behind it is taken out of the
  // accessibility tree and out of the tab order for as long as it is up (presentation only: app.js
  // owns when the overlay is shown and hidden, and nothing here changes that).
  const white = $('white'), page = document.getElementById('app');
  if (white && page) {
    let returnTo = null;
    const syncInert = () => {
      const up = !white.hidden;
      try { page.inert = up; } catch (e) { /* older browsers: the overlay still covers the page visually */ }
      try {
        if (up && !white.contains(document.activeElement)) { returnTo = document.activeElement; white.focus(); }
        else if (!up && returnTo) { const t = returnTo; returnTo = null; if (t.isConnected) t.focus(); }
      } catch (e) { /* focus moves are best effort */ }
    };
    new MutationObserver(syncInert).observe(white, { attributes: true, attributeFilter: ['hidden'] });
    syncInert();
  }

  // the confidence level is read from the shipped conformal table, never hard-coded here
  if (conf) {
    fetch('data/conformal.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => { if (j) conf.textContent = confLineFor(j.default_alpha); })
      .catch(() => { /* the fallback sentence is already in the markup */ });
  }
}

if (typeof document !== 'undefined' && typeof MutationObserver !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
}
