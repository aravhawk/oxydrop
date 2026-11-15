"""OPTICS v2 per-window feature extraction (causal). Imported by optics_experiment.py."""
import sys
sys.path.insert(0, '/home/aravhawk/.t3/scratch/veritas/submission/oxydrop/python')
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
from oxydrop.hr import best_channel_estimate
from oxydrop.features import rec_feats

FS = 30
SOS = butter(3, [0.7, 4.0], btype='band', fs=FS, output='sos')
WIN, STEP = 10, 2
RAW = ['hr', 'ror_RG', 'ror_RB', 'ror_GB', 'pi_R', 'pi_G', 'pi_B', 'dc_R', 'dc_G', 'dc_B', 'amp_cv']
SAT_MAX, CV_MAX, NB_MIN = 0.05, 0.60, 4      # a-priori gate thresholds, label-free


def window_optics(acl, lin, accl):
    """acl: (300,3) band-passed log-RGB (for HR), lin: (300,3) raw 0-255 intensity, accl: (300,3) band-passed LINEAR intensity.
    Returns dict of raw (uncalibrated) optical features and gate info."""
    f = {k: np.nan for k in RAW}
    sd = acl.std(0)
    primary = 1 if sd[1] > sd[0] * 0.3 else 0
    e, ch = best_channel_estimate(acl, primary)
    f['valid_hr'] = bool(e['valid']); f['sqi'] = e['sqi']; f['acf'] = e['acf']
    f['sat'] = float((lin >= 250).any(1).mean())
    f['hr'] = e['hr'] if np.isfinite(e['hr']) else np.nan
    f['nb'] = 0
    dcw = np.clip(lin.mean(0), 0.5, None)
    for i, c in enumerate('RGB'):
        f[f'dc_{c}'] = -np.log(dcw[i] / 255.0)           # DC absorbance (calibration subtracts the person's own baseline)
    if not np.isfinite(e['hr']):
        return f
    T = FS * 60.0 / e['hr']
    pk, _ = find_peaks(-acl[:, ch], distance=max(int(0.65 * T), 3), prominence=0.3 * acl[:, ch].std())   # systolic absorbance peaks
    AC, DC = [], []
    for a, b in zip(pk[:-1], pk[1:]):
        if not (0.6 * T <= b - a <= 1.5 * T):
            continue
        sl = slice(a, b + 1)
        AC.append(accl[sl].max(0) - accl[sl].min(0)); DC.append(lin[sl].mean(0))
    f['nb'] = len(AC)
    if len(AC) < NB_MIN:
        return f
    AC = np.array(AC); DC = np.clip(np.array(DC), 0.5, None); R = AC / DC           # (nb,3) relative pulsatile amplitude per beat
    R = np.clip(R, 1e-9, None)
    f['ror_RG'] = np.log(np.median(R[:, 0] / R[:, 1])); f['ror_RB'] = np.log(np.median(R[:, 0] / R[:, 2]))
    f['ror_GB'] = np.log(np.median(R[:, 1] / R[:, 2]))
    for i, c in enumerate('RGB'):
        f[f'pi_{c}'] = np.log(np.median(R[:, i]))
    f['amp_cv'] = float(R[:, ch].std() / (R[:, ch].mean() + 1e-12))
    return f


def rec_optics(rgb, n_sec=None):
    """All trailing 10 s windows (2 s hop) of one recording. Returns (list of window start s, list of feature dicts)."""
    rgb = np.clip(np.asarray(rgb, float), 0, 255)
    lg, acl_full = rec_feats(rgb)
    acc_full = np.stack([sosfilt(SOS, rgb[:, c]) for c in range(3)], 1)
    n = len(rgb) // FS if n_sec is None else n_sec
    ts, rows = [], []
    for s in range(0, n - WIN + 1, STEP):
        a, b = s * FS, (s + WIN) * FS
        rows.append(window_optics(acl_full[a:b], rgb[a:b], acc_full[a:b])); ts.append(s)
    return ts, rows
