// Three-state output logic and fingertip-placement check. Pure functions, no DOM: shared by the app and by tests/decision.mjs.
// Rule: a confident answer (NORMAL or LOW) is shown only when every quality gate passes. Anything else is UNSURE.

export const STATE = { NORMAL: 'NORMAL', LOW: 'LOW', UNSURE: 'UNSURE' };
export const STATE_TEXT = {
  NORMAL: 'Normal vs your baseline',
  LOW: 'Possible low oxygen: re-check with a pulse oximeter',
  UNSURE: 'Unsure, re-measure',
};
export const SAT_MAX = 0.10;      // fraction of saturated samples in the window (same value as the engine's own gate)
export const FPS_MIN = 20;        // below this the 30 Hz resampling is not trustworthy
export const GAPS_MAX = 3;        // frame-gap events tolerated in the last 10 s
export const GAP_MAX_S = 1.0;     // longest single frame gap (seconds) tolerated inside the 10 s decision window
export const GAP_TOTAL_S = 3.0;   // total missing time (seconds) tolerated in the last 30 s (the heart-rate estimator's span)
export const STALL_MS = 1500;     // no camera frame for this long (wall clock): the camera has stalled or was interrupted
export const CAL_FINGER_LOSS_MS = 2000;  // fingertip not covering lens and flash for this long inside the calibration minute: the minute is void
export const CAL_FINGER_MIN = 0.9;       // fraction of calibration-minute ticks with good fingertip placement needed for the baseline to be accepted
export const MOTION_MAX = 0.35;   // m/s^2 std of |accel| over 10 s; above this the phone is moving

// Plain-language reason shown under "Unsure, re-measure". Order of this table is the order the checks are applied.
export const REASON_TEXT = {
  calibration_failed: 'The calibration minute did not pass its checks. Rest for a couple of minutes, then restart calibration.',
  no_finger: 'No fingertip detected on the lens. Cover the lens and the flash and press lightly.',
  moving: 'Movement detected. Hold still: rest your hand and the phone on a table.',
  low_fps: 'The camera is running too slowly for a reliable reading.',
  frame_gaps: 'The camera dropped frames. Keep the screen on and close other apps.',
  camera_stalled: 'The camera stopped delivering frames (screen lock, app switch or interruption). Keep the screen on, then re-measure.',
  saturation: 'The image is too bright (saturated). Press more lightly.',
  invalid: 'No clean pulse in this window. Cover the lens and flash fully, press lightly, keep still.',
  low_ac: 'The pulse signal is weaker than anything this model was trained on (common when the flash is far from the lens). Cover the lens and the flash fully, press lightly, keep still, or try another lens.',
  nolock: 'No steady pulse lock for this window. Cover the lens and flash fully, press lightly, keep still.',
  model_unsure: 'The model is not confident for this window. Keep still; the next reading is 2 seconds away.',
};
// reasons that mean the raw signal itself is bad, so no heart-rate number is shown
const SIGNAL_REASONS = new Set(['no_finger', 'moving', 'low_fps', 'frame_gaps', 'camera_stalled', 'saturation']);

/** First failing quality gate for a window, or null.
 *  ctx: {fps, recentGaps (gap events in 10 s), gapMax (longest gap in the 10 s window, s), gapTotal (missing time in the last 30 s, s),
 *        stalledMs (wall-clock ms since the last camera frame), finger, moving, calOk}. */
export function qualityReason(d, ctx = {}) {
  const c = { fps: 30, recentGaps: 0, gapMax: 0, gapTotal: 0, stalledMs: 0, finger: true, moving: false, calOk: true, ...ctx };
  if (c.stalledMs > STALL_MS) return 'camera_stalled';
  if (c.calOk === false) return 'calibration_failed';
  if (c.finger === false) return 'no_finger';
  if (c.moving === true) return 'moving';
  if (c.fps < FPS_MIN) return 'low_fps';
  if (c.recentGaps >= GAPS_MAX || c.gapMax > GAP_MAX_S || c.gapTotal > GAP_TOTAL_S) return 'frame_gaps';
  if (!(d.sat <= SAT_MAX)) return 'saturation';      // also true when sat is NaN/undefined
  if (!d.valid) return 'invalid';
  return null;
}

/** Heart rate to display, or null (shown as an en dash). Needs d.valid, a finite estimate and a clean signal (no placement/motion/frame/saturation fault). */
export function hrToShow(d, ctx = {}) {
  const r = qualityReason(d, { ...ctx, calOk: true });
  if (r && SIGNAL_REASONS.has(r)) return null;
  return d.valid && Number.isFinite(d.hr) ? d.hr : null;
}

/** Decide the user-facing state for one finished (non-calibration) engine window. */
export function decide(d, ctx = {}) {
  let reason = qualityReason(d, ctx), state = STATE.UNSURE;
  if (!reason) {
    if (d.label === 'nolock') reason = 'nolock';
    else if (d.label === 'low_ac') reason = 'low_ac';
    else if (d.label === 'unsure') reason = 'model_unsure';
    else if (d.label === 'HYPOXIC') state = STATE.LOW;
    else if (d.label === 'normal') state = STATE.NORMAL;
    else reason = 'invalid';                           // unknown label: never guess
  }
  return { state, reason, text: STATE_TEXT[state], detail: reason ? REASON_TEXT[reason] : null, hr: hrToShow(d, ctx) };
}

/** Gap context for the decision from a Resampler: longest gap overlapping the last 10 s, missing time in the last 30 s. */
export function gapContext(rs, fs = 30) {
  const g10 = rs.gapStats(rs.n - 10 * fs), g30 = rs.gapStats(rs.n - 30 * fs);
  return { gapMax: g10.max, gapTotal: g30.total };
}

/** Forced answer when frames stop arriving (used by the live UI between windows): UNSURE, no heart rate. */
export function stalledDecision() { return { state: STATE.UNSURE, reason: 'camera_stalled', text: STATE_TEXT.UNSURE, detail: REASON_TEXT.camera_stalled, hr: null }; }

/** Fingertip placement from live camera stats. stats: {mean:[R,G,B], satFrac, spatialCV, frames}. Returns {level: 'good'|'warn'|'bad', title, msg}. */
export function assessPlacement(stats, opts = {}) {
  const torch = !!opts.torch;
  if (!stats || !stats.mean || !(stats.frames >= 5)) return { level: 'warn', title: 'Looking for your fingertip…', msg: '' };
  const [R, G, B] = stats.mean;
  if (R < 50) return torch
    ? { level: 'bad', title: 'Cover the lens AND the flash', msg: 'It is too dark: the flash light is not passing through your fingertip.' }
    : { level: 'bad', title: 'Too dark', msg: 'Face a lamp or window, or switch the flash on, then cover the lens.' };
  if (R < 1.4 * G || R < 1.4 * B) return { level: 'bad', title: 'Cover the lens AND the flash', msg: 'The camera still sees light that has not passed through your fingertip.' };
  if (stats.satFrac > SAT_MAX) return { level: 'bad', title: 'Press more lightly (too bright/saturated)', msg: 'Too much light is reaching the sensor. Ease the pressure and cover the lens fully.' };
  if (stats.spatialCV > 0.35) return { level: 'warn', title: 'Cover the whole lens', msg: 'The image is uneven. Centre your fingertip on the lens.' };
  return { level: 'good', title: 'Fingertip detected', msg: 'Keep still.' };
}

/** Calibration-minute interruption from the live acquisition state: 'settings_changed' | 'finger_lost' | null.
 *  The engine builds the baseline from every window with a valid pulse and cannot see placement, so the app voids and restarts the minute instead.
 *  A frame stall is not handled here: the calibration stall rule rejects the minute at its end (checkCalibration).
 *  s: {placeGood, stalled, badMs (how long the placement has been not-good), settingsChanged}. */
export function calibrationInterrupt(s = {}) {
  if (s.stalled) return null;
  if (s.settingsChanged) return 'settings_changed';
  if (!s.placeGood && (s.badMs || 0) >= CAL_FINGER_LOSS_MS) return 'finger_lost';
  return null;
}

/** Finger-coverage requirement for accepting a calibration minute: reason text, or null. good/total count UI ticks (not stalled) with good placement. */
export function calibrationFingerReason(good, total) {
  if (!(total >= 20)) return null;
  const f = good / total;
  return f >= CAL_FINGER_MIN ? null : `Your fingertip covered the lens and flash for only ${Math.round(f * 100)}% of the calibration minute (need ${Math.round(CAL_FINGER_MIN * 100)}%).`;
}
