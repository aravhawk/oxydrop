"""v2 per-window features (field-hardened). Same physics/cardio statistics as v1 (features.py) but:
  * heart rate = robust spectral + autocorrelation estimate (hr.py) instead of peak-interval HR (which doubles on dicrotic notches);
  * beat detection uses a minimum beat spacing derived from that HR (no double counting);
  * the best of the three colour channels is used if the default channel does not give a locked pulse;
  * every window gets a validity flag (pulse locked, spectral/autocorrelation agree, beat-HR agrees, channel not clipped).
Invalid windows contribute NO features (they are NaN) and the app abstains on them.
"""
import numpy as np
from scipy.signal import find_peaks
from .features import FS, rec_feats, resp_rate  # noqa: F401  (same causal filter / log transform as v1)
from . import hr as H

HR_SOURCE = 'beat'     # model input hr = median beat interval (robust-spectral hr as model input scored 1.8 pts worse: see results/eval_v1_vs_v2.csv notes); robust HR is used for gating, beat spacing and display
ADAPT_DIST = True
BEAT_AGREE = 0.12      # beat-interval HR must match the robust HR within 12 %
SAT_MAX = 0.10         # max fraction of clipped (>=250/255) frames in the chosen channel over the last 10 s


def win_feats(lg, ac, te):
    f = {}
    a10 = ac[te - 10 * FS:te]; a30 = ac[max(0, te - 30 * FS):te]; l10 = lg[te - 10 * FS:te]
    sd = a10.std(0) + 1e-9; dc = l10.mean(0)
    f['ror_RG'] = np.log(sd[0] / sd[1]); f['ror_RB'] = np.log(sd[0] / sd[2]); f['ror_GB'] = np.log(sd[1] / sd[2])
    C = np.cov(a10.T)
    f['sl_RG'] = C[0, 1] / (C[1, 1] + 1e-12); f['sl_RB'] = C[0, 2] / (C[2, 2] + 1e-12)
    C3 = np.cov(a30.T)
    f['sl30_RG'] = C3[0, 1] / (C3[1, 1] + 1e-12); f['ror30_RG'] = 0.5 * np.log(C3[0, 0] / (C3[1, 1] + 1e-12))
    f['dc_R'] = dc[0]; f['dc_G'] = dc[1]; f['dc_B'] = dc[2]
    for i, c in enumerate('RGB'):
        f[f'pi_{c}'] = np.log(sd[i])
    primary = 1 if sd[1] > sd[0] * 0.3 else 0
    est, ch = H.best_channel_estimate(a30, primary)
    x = -a30[:, ch]
    f_hr = est['hr'] / 60 if np.isfinite(est['hr']) else None
    dist = max(int(FS * 0.33), int(0.55 * FS / f_hr)) if (f_hr and ADAPT_DIST) else int(FS * 0.33)
    pk, _ = find_peaks(x, distance=dist, prominence=np.std(x) * 0.3)
    ibi = np.array([])
    if len(pk) >= 6:
        ibi = np.diff(pk) / FS
        ibi = ibi[(ibi > 0.3) & (ibi < 1.6)]
    hr_beat = 60 / np.median(ibi) if len(ibi) >= 5 else np.nan
    sat = float((l10[:, ch] >= np.log(251.0)).mean())   # raw intensity >= 250 of 255
    valid = bool(est['valid'] and np.isfinite(hr_beat) and abs(hr_beat - est['hr']) / est['hr'] < BEAT_AGREE and sat <= SAT_MAX)
    f['_valid'] = valid; f['_sqi'] = est['sqi']; f['_acf'] = est['acf']; f['_ch'] = ch; f['_hr_est'] = est['hr']; f['_hr_beat'] = hr_beat; f['_sat'] = sat
    if valid:
        f['hr'] = est['hr'] if HR_SOURCE == 'spec' else hr_beat
        f['rmssd'] = np.log(np.sqrt(np.mean(np.diff(ibi) ** 2)) + 1e-3)
        f['sdnn'] = np.log(ibi.std() + 1e-3)
        amp = x[pk]; tt = pk / FS
        ti = np.arange(tt[0], tt[-1], 0.25)
        if len(ti) > 16:
            f['rr_amp'] = resp_rate(np.interp(ti, tt, amp), 4)
            f['rr_ibi'] = resp_rate(np.interp(ti, tt[1:], np.diff(pk) / FS), 4)
        f['beat_cv'] = np.std(amp) / (np.mean(np.abs(amp)) + 1e-9)
    return f
