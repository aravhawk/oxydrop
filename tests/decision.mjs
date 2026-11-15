// Unit tests for the three-state decision logic and the fingertip-placement check. Usage: node tests/decision.mjs
import assert from 'node:assert/strict';
import { decide, hrToShow, assessPlacement, calibrationInterrupt, calibrationFingerReason, CAL_FINGER_LOSS_MS, CAL_FINGER_MIN, STATE_TEXT, REASON_TEXT } from '../app/js/decision.js';

const good = (o = {}) => ({ calibrating: false, valid: true, sat: 0.0, hr: 72.4, label: 'normal', p: 0.1, ...o });
const ok = (o = {}) => ({ fps: 30, recentGaps: 0, finger: true, moving: false, calOk: true, ...o });
let n = 0; const t = (name, fn) => { fn(); n++; console.log('ok  ', name); };

// exact user-facing strings
t('state strings', () => {
  assert.equal(STATE_TEXT.NORMAL, 'Normal vs your baseline');
  assert.equal(STATE_TEXT.LOW, 'Possible low oxygen: re-check with a pulse oximeter');
  assert.equal(STATE_TEXT.UNSURE, 'Unsure, re-measure');
  for (const s of Object.values(STATE_TEXT)) assert.ok(!/\u2014/.test(s));
});

t('NORMAL', () => { const r = decide(good(), ok()); assert.equal(r.state, 'NORMAL'); assert.equal(r.reason, null); assert.equal(r.text, STATE_TEXT.NORMAL); assert.equal(r.hr, 72.4); });
t('LOW', () => { const r = decide(good({ label: 'HYPOXIC', p: 0.9 }), ok()); assert.equal(r.state, 'LOW'); assert.equal(r.reason, null); assert.equal(r.text, STATE_TEXT.LOW); assert.equal(r.hr, 72.4); });

// every UNSURE reason
const cases = [
  ['engine label nolock', good({ label: 'nolock', valid: false, hr: NaN }), ok(), 'invalid'],
  ['engine label nolock with valid flag (inconsistent)', good({ label: 'nolock' }), ok(), 'nolock'],
  ['engine label low_ac (amplitude below the trained range)', good({ label: 'low_ac' }), ok(), 'low_ac'],
  ['engine label unsure', good({ label: 'unsure', p: 0.52 }), ok(), 'model_unsure'],
  ['unknown label', good({ label: 'whatever' }), ok(), 'invalid'],
  ['d.valid false', good({ valid: false, hr: NaN }), ok(), 'invalid'],
  ['saturation > 10%', good({ sat: 0.11 }), ok(), 'saturation'],
  ['saturation NaN', good({ sat: NaN }), ok(), 'saturation'],
  ['fps < 20', good(), ok({ fps: 19.5 }), 'low_fps'],
  ['recent frame gaps', good(), ok({ recentGaps: 3 }), 'frame_gaps'],
  ['finger not detected', good(), ok({ finger: false }), 'no_finger'],
  ['moving per accelerometer', good(), ok({ moving: true }), 'moving'],
  ['calibration did not pass', good(), ok({ calOk: false }), 'calibration_failed'],
];
for (const [name, d, ctx, reason] of cases) {
  t('UNSURE: ' + name, () => {
    const r = decide(d, ctx);
    assert.equal(r.state, 'UNSURE'); assert.equal(r.reason, reason); assert.equal(r.text, 'Unsure, re-measure'); assert.equal(r.detail, REASON_TEXT[reason]);
  });
}

t('a LOW label never survives a failed gate', () => {
  for (const ctx of [ok({ fps: 10 }), ok({ recentGaps: 5 }), ok({ finger: false }), ok({ moving: true }), ok({ calOk: false })]) assert.equal(decide(good({ label: 'HYPOXIC' }), ctx).state, 'UNSURE');
  assert.equal(decide(good({ label: 'HYPOXIC', sat: 0.2 }), ok()).state, 'UNSURE');
  assert.equal(decide(good({ label: 'HYPOXIC', valid: false }), ok()).state, 'UNSURE');
});
t('boundaries: fps exactly 20, sat exactly 0.10, 2 gaps are allowed', () => { assert.equal(decide(good({ sat: 0.10 }), ok({ fps: 20, recentGaps: 2 })).state, 'NORMAL'); });
t('ctx defaults pass', () => { assert.equal(decide(good()).state, 'NORMAL'); });

// heart-rate display rule
t('HR shown only when d.valid', () => { assert.equal(hrToShow(good(), ok()), 72.4); assert.equal(hrToShow(good({ valid: false }), ok()), null); assert.equal(hrToShow(good({ hr: NaN }), ok()), null); });
t('HR hidden on every signal fault', () => {
  for (const ctx of [ok({ finger: false }), ok({ moving: true }), ok({ fps: 12 }), ok({ recentGaps: 4 })]) assert.equal(hrToShow(good(), ctx), null);
  assert.equal(hrToShow(good({ sat: 0.3 }), ok()), null);
});
t('HR still shown when the model is unsure or calibration failed (signal itself is clean)', () => { assert.equal(decide(good({ label: 'unsure' }), ok()).hr, 72.4); assert.equal(decide(good(), ok({ calOk: false })).hr, 72.4); });
t('UNSURE from a bad signal carries no HR', () => { assert.equal(decide(good({ valid: false, hr: 150 }), ok()).hr, null); assert.equal(decide(good({ hr: 150 }), ok({ moving: true })).hr, null); });

// placement
const st = (mean, o = {}) => ({ mean, satFrac: 0, spatialCV: 0.05, frames: 30, ...o });
t('placement: waiting for frames', () => { assert.equal(assessPlacement(null).level, 'warn'); assert.equal(assessPlacement(st([200, 80, 60], { frames: 2 })).level, 'warn'); });
t('placement: good finger with flash', () => { const p = assessPlacement(st([197, 76, 60]), { torch: true }); assert.equal(p.level, 'good'); assert.equal(p.title, 'Fingertip detected'); });
t('placement: good finger ambient', () => { assert.equal(assessPlacement(st([140, 50, 35]), { torch: false }).level, 'good'); });
t('placement: dark with flash', () => { const p = assessPlacement(st([20, 10, 8]), { torch: true }); assert.equal(p.level, 'bad'); assert.equal(p.title, 'Cover the lens AND the flash'); });
t('placement: dark ambient', () => { assert.equal(assessPlacement(st([20, 10, 8]), { torch: false }).title, 'Too dark'); });
t('placement: open scene (grey, not finger)', () => { const p = assessPlacement(st([128, 128, 128]), { torch: true }); assert.equal(p.level, 'bad'); assert.equal(p.title, 'Cover the lens AND the flash'); });
t('placement: saturated', () => { const p = assessPlacement(st([255, 120, 90], { satFrac: 0.6 }), { torch: true }); assert.equal(p.level, 'bad'); assert.equal(p.title, 'Press more lightly (too bright/saturated)'); });
t('placement: uneven image', () => { assert.equal(assessPlacement(st([197, 76, 60], { spatialCV: 0.5 })).level, 'warn'); });

// calibration-minute acquisition gates (REVIEW.md section 3, wrong-answer path 3: rejected live samples must not poison the baseline)
t('calibration interrupt: short finger dip is tolerated, a sustained one voids the minute', () => {
  assert.equal(calibrationInterrupt({ placeGood: true, badMs: 0 }), null);
  assert.equal(calibrationInterrupt({ placeGood: false, badMs: CAL_FINGER_LOSS_MS - 1 }), null);
  assert.equal(calibrationInterrupt({ placeGood: false, badMs: CAL_FINGER_LOSS_MS }), 'finger_lost');
});
t('calibration interrupt: a settings change voids the minute; a stall is left to the stall rule', () => {
  assert.equal(calibrationInterrupt({ placeGood: true, settingsChanged: true }), 'settings_changed');
  assert.equal(calibrationInterrupt({ placeGood: false, badMs: 99999, stalled: true, settingsChanged: true }), null);
});
t('calibration finger coverage: needs 90% good placement, ignores too few ticks', () => {
  assert.equal(calibrationFingerReason(0, 5), null);
  assert.equal(calibrationFingerReason(CAL_FINGER_MIN * 100, 100), null);
  assert.match(calibrationFingerReason(89, 100), /only 89%/);
  assert.match(calibrationFingerReason(0, 240), /only 0%/);
});

console.log(`DECISION PASS (${n} checks)`);
