"""Robust heart-rate estimator (v2). Replaces peak-interval HR, which double-counts dicrotic notches (reads ~2x the true rate).

Per trailing window of a causal band-passed PPG channel:
  1. Hann-windowed, zero-padded (4096) magnitude spectrum.
  2. Harmonic-sum score S(f) = P(f) + 0.7 P(2f) over F_LO-3.0 Hz: prefers the FUNDAMENTAL over its 2nd harmonic.
  3. Autocorrelation: smallest-lag peak that is >= 0.85 of the strongest peak (again the fundamental).
  4. Valid only if the two estimates agree within AGREE (15 %), the spectral concentration (SQI) and the autocorrelation peak are strong.
Anything else is reported as 'not locked' so the app abstains instead of printing a wrong number.
"""
import numpy as np

FS = 30
NFFT = 4096
F_LO, F_HI = 0.85, 3.0   # lowest trackable resting HR = 0.85 Hz = 51 bpm (cross-fit chose it over 0.75 and 0.65, see docs/HR_VALIDATION.md)
SQI_MIN = 0.30      # fraction of band power at the fundamental and 2nd harmonic
ACF_MIN = 0.30      # normalised autocorrelation at the chosen period
AGREE = 0.15        # max relative disagreement of spectral vs autocorrelation HR
# SQI_MIN / ACF_MIN / AGREE / F_LO were chosen by a subject-level cross-fit on BUT PPG (scripts/hr_crossfit.py), not on the scored subjects.
MIN_SEC = 9.5       # shortest window we will estimate on


def estimate_hr(x, fs=FS, f_lo=None):
    """x: 1-D band-passed window. Returns dict(hr, hr_acf, sqi, acf, valid). hr/hr_acf in bpm (nan if not computable).
    `valid` applies the module gate (SQI_MIN, ACF_MIN, AGREE). The raw hr/hr_acf/sqi/acf are returned so a different gate can be re-applied (see gate_ok)."""
    f_lo = F_LO if f_lo is None else f_lo
    x = np.asarray(x, float)
    out = dict(hr=np.nan, hr_acf=np.nan, sqi=0.0, acf=0.0, valid=False)
    n = len(x)
    if n < MIN_SEC * fs or np.std(x) < 1e-9:
        return out
    xc = x - x.mean()
    P = np.abs(np.fft.rfft(xc * np.hanning(n), n=NFFT))
    df = fs / NFFT
    k0, k1 = int(np.ceil(f_lo / df)), int(np.floor(F_HI / df))
    ks = np.arange(k0, k1 + 1)
    score = P[ks] + 0.7 * P[2 * ks]
    kb = ks[np.argmax(score)]
    kf = float(kb)
    # parabolic refinement only if kb is a true local maximum of the spectrum, and never more than half a bin
    if 0 < kb < len(P) - 1 and P[kb] >= P[kb - 1] and P[kb] >= P[kb + 1]:
        a, b, c = P[kb - 1], P[kb], P[kb + 1]
        d = a - 2 * b + c
        if d < 0:
            kf = kb + float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    f0 = kf * df
    P2 = P ** 2
    tot = P2[int(np.ceil(0.65 / df)):int(np.floor(4.0 / df)) + 1].sum() + 1e-18
    w = int(round(0.10 / df))
    near = P2[max(kb - w, 0):kb + w + 1].sum() + P2[max(2 * kb - w, 0):2 * kb + w + 1].sum()
    sqi = float(near / tot)
    # autocorrelation (biased, normalised), FFT based
    m = 1 << int(np.ceil(np.log2(2 * n)))
    r = np.fft.irfft(np.abs(np.fft.rfft(xc, n=m)) ** 2)[:n]
    r = r / (r[0] + 1e-18)
    lo, hi = int(np.ceil(fs / 3.0)), min(int(np.floor(fs / f_lo)), n - 2)
    seg = r[lo - 1:hi + 2]
    pk = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    out.update(hr=60 * f0, sqi=sqi)
    if not pk:
        return out
    rmax = max(seg[i] for i in pk)
    # fundamental = smallest-lag peak >= 0.85 of the strongest; if every peak is negative that test is never met, so use the strongest
    lag_i = next((i for i in pk if seg[i] >= 0.85 * rmax), max(pk, key=lambda i: seg[i]))
    a, b, c = seg[lag_i - 1], seg[lag_i], seg[lag_i + 1]
    d = a - 2 * b + c
    lag = (lo - 1) + lag_i + (0.5 * (a - c) / d if d != 0 else 0.0)
    out.update(hr_acf=60 * fs / lag, acf=float(seg[lag_i]))
    out['valid'] = gate_ok(out)
    return out


def gate_ok(e, sqi_min=None, acf_min=None, agree=None):
    """Validity gate on a raw estimate dict. f0 must be positive and finite before the agreement test."""
    sqi_min = SQI_MIN if sqi_min is None else sqi_min
    acf_min = ACF_MIN if acf_min is None else acf_min
    agree = AGREE if agree is None else agree
    h, ha = e['hr'], e['hr_acf']
    if not (np.isfinite(h) and np.isfinite(ha) and h > 0):
        return False
    return bool(abs(ha - h) / h < agree and e['sqi'] >= sqi_min and e['acf'] >= acf_min)


def best_channel_estimate(acs, primary, fs=FS):
    """acs: (n,3) band-passed R,G,B window. Try the primary channel first, then the other two by decreasing amplitude.
    Returns (estimate dict, channel index). If none is valid, the primary's (invalid) estimate is returned."""
    sd = acs.std(0)
    order = [primary] + [c for c in np.argsort(-sd) if c != primary]
    first = None
    for c in order:
        e = estimate_hr(acs[:, c], fs)
        if first is None:
            first = (e, c)
        if e['valid']:
            return e, int(c)
    return first
