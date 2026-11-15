"""Robust heart-rate estimator (v2). Replaces peak-interval HR, which double-counts dicrotic notches (reads ~2x the true rate).

Per trailing window of a causal band-passed PPG channel:
  1. Hann-windowed, zero-padded (4096) magnitude spectrum.
  2. Harmonic-sum score S(f) = P(f) + 0.7 P(2f) over 0.65-3.0 Hz: prefers the FUNDAMENTAL over its 2nd harmonic.
  3. Autocorrelation: smallest-lag peak that is >= 0.85 of the strongest peak (again the fundamental).
  4. Valid only if the two estimates agree within 10 %, the spectral concentration (SQI) and the autocorrelation peak are strong.
Anything else is reported as 'not locked' so the app abstains instead of printing a wrong number.
"""
import numpy as np

FS = 30
NFFT = 4096
F_LO, F_HI = 0.65, 3.0
SQI_MIN = 0.30      # fraction of band power at the fundamental and 2nd harmonic (see docs/HR_VALIDATION.md for the sweep)
ACF_MIN = 0.30      # normalised autocorrelation at the chosen period
AGREE = 0.10        # max relative disagreement of spectral vs autocorrelation HR
MIN_SEC = 9.5       # shortest window we will estimate on


def estimate_hr(x, fs=FS):
    """x: 1-D band-passed window. Returns dict(hr, hr_acf, sqi, acf, valid, ok_len). hr/hr_acf in bpm (nan if not computable)."""
    x = np.asarray(x, float)
    out = dict(hr=np.nan, hr_acf=np.nan, sqi=0.0, acf=0.0, valid=False)
    n = len(x)
    if n < MIN_SEC * fs or np.std(x) < 1e-9:
        return out
    xc = x - x.mean()
    P = np.abs(np.fft.rfft(xc * np.hanning(n), n=NFFT))
    df = fs / NFFT
    k0, k1 = int(np.ceil(F_LO / df)), int(np.floor(F_HI / df))
    ks = np.arange(k0, k1 + 1)
    score = P[ks] + 0.7 * P[2 * ks]
    kb = ks[np.argmax(score)]
    if 0 < kb < len(P) - 1:   # parabolic refinement on the spectrum
        a, b, c = P[kb - 1], P[kb], P[kb + 1]
        d = a - 2 * b + c
        kf = kb + (0.5 * (a - c) / d if d != 0 else 0.0)
    else:
        kf = float(kb)
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
    lo, hi = int(np.ceil(fs / 3.0)), min(int(np.floor(fs / 0.65)), n - 2)
    seg = r[lo - 1:hi + 2]
    pk = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not pk:
        out.update(hr=60 * f0, sqi=sqi)
        return out
    rmax = max(seg[i] for i in pk)
    lag_i = next(i for i in pk if seg[i] >= 0.85 * rmax)
    a, b, c = seg[lag_i - 1], seg[lag_i], seg[lag_i + 1]
    d = a - 2 * b + c
    lag = (lo - 1) + lag_i + (0.5 * (a - c) / d if d != 0 else 0.0)
    f_acf = fs / lag
    acf = float(seg[lag_i])
    agree = abs(f_acf - f0) / f0 < AGREE
    out.update(hr=60 * f0, hr_acf=60 * f_acf, sqi=sqi, acf=acf, valid=bool(agree and sqi >= SQI_MIN and acf >= ACF_MIN))
    return out


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
