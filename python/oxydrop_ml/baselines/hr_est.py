"""Heart-rate estimators for P2 (no learned parameters, so the leave-one-subject-out evaluation only has to select the v2 SQI gate on training subjects).

All three read the SAME window buffer as the CNN: the shard's [CTX + WIN = 450, 3] log-RGB buffer (5 s warm-up, reflected at the start of a recording, + the
10 s window), causally band-passed (features.causal_ac, 0.7-4 Hz, warm start).
  fft_plain   largest FFT peak 0.7-3.5 Hz of the primary channel, 10 s window (scripts/validate_hr.py estimator B; no quality check, always answers)
  v1_peak     peak-interval counter of engine_v1 / python/oxydrop/features.py: find_peaks on -g (distance 0.33 s, prominence 0.3 std), median of intervals in
              0.3-1.6 s, needs >= 6 peaks and >= 5 intervals (else no answer). It sees the whole 15 s buffer, shorter than the 30 s trailing context of the shipped v1.
  v2          python/oxydrop/hr.py estimate_hr on the 10 s window, channels tried in best_channel_estimate order (primary, then by amplitude); raw per-channel
              (hr, hr_acf, sqi, acf) are kept so any SQI/ACF gate can be re-applied without recomputation (`select`).
Primary channel rule (as in the app): green if std_G > 0.3 std_R of the 10 s window, else red.
"""
import os

import numpy as np
from scipy.signal import find_peaks

from oxydrop import hr as HR

from .. import features as F
from . import common as C

FS = 30
WIN = 300
GATE_SQI = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]      # grid of the nested gate selection (shipped value 0.30 is on the grid)
GATE_ACF = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
SHIPPED_GATE = (HR.SQI_MIN, HR.ACF_MIN)


def naive_hr(x, fs=FS):
    x = np.asarray(x, float)
    P = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n=4096))
    fr = np.fft.rfftfreq(4096, 1 / fs)
    b = (fr >= 0.7) & (fr <= 3.5)
    return 60 * fr[b][np.argmax(P[b])]


def v1_peak_hr(a_buf, primary):
    """Peak-interval HR (features.py win_feats) on the buffer; NaN when fewer than 6 peaks / 5 valid intervals."""
    x = -a_buf[:, primary]
    pk, _ = find_peaks(x, distance=int(FS * 0.33), prominence=np.std(x) * 0.3)
    if len(pk) < 6:
        return np.nan
    ibi = np.diff(pk) / FS
    ibi = ibi[(ibi > 0.3) & (ibi < 1.6)]
    return 60 / np.median(ibi) if len(ibi) >= 5 else np.nan


def raw_estimates(ws, chunk=512):
    """Raw estimates for every window of a shard. Returns dict of arrays: fft, v1, hr[N,3], hr_acf[N,3], sqi[N,3], acf[N,3] (channel order of best_channel_estimate)."""
    os.makedirs(C.CACHE_B, exist_ok=True)
    p = C.CACHE_B / f'hr_raw_{ws.ds}_{len(ws)}.npz'
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    N = len(ws)
    out = dict(fft=np.full(N, np.nan), v1=np.full(N, np.nan), **{k: np.full((N, 3), np.nan) for k in ('hr', 'hr_acf', 'sqi', 'acf')})
    for a in range(0, N, chunk):
        idx = np.arange(a, min(N, a + chunk))
        ac = F.causal_ac(ws.lrgb(idx).astype(np.float64))
        for j, i in enumerate(idx):
            w = ac[j, -WIN:]
            sd = w.std(0)
            prim = 1 if sd[1] > 0.3 * sd[0] else 0
            out['fft'][i] = naive_hr(w[:, prim])
            out['v1'][i] = v1_peak_hr(ac[j], prim)
            order = [prim] + [int(c) for c in np.argsort(-sd) if c != prim]
            for k, c in enumerate(order):
                e = HR.estimate_hr(w[:, c], FS)
                out['hr'][i, k] = e['hr']; out['hr_acf'][i, k] = e['hr_acf']; out['sqi'][i, k] = e['sqi']; out['acf'][i, k] = e['acf']
    np.savez_compressed(p, **out)
    return out


def gate_ok_vec(hr, hra, sqi, acf, sqi_min, acf_min, agree=HR.AGREE):
    """Vectorised hr.gate_ok over arrays."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.isfinite(hr) & np.isfinite(hra) & (hr > 0) & (np.abs(hra - hr) / hr < agree) & (sqi >= sqi_min) & (acf >= acf_min)


def select(raw, sqi_min, acf_min):
    """Re-implementation of hr.best_channel_estimate for any gate: first channel (in order) that passes, else the primary's raw estimate.
    Returns (hr [N], valid [N] bool)."""
    ok = gate_ok_vec(raw['hr'], raw['hr_acf'], raw['sqi'], raw['acf'], sqi_min, acf_min)        # [N, 3]
    first = np.where(ok.any(1), ok.argmax(1), 0)
    hr = raw['hr'][np.arange(len(first)), first]
    return hr, ok.any(1)
