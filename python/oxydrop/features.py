"""Causal per-window stage features from per-frame mean RGB at 30 Hz.

S1 cardio stage : heart rate, HRV, respiratory rate, perfusion index (beat detection on a causal band-passed green channel)
S2 physics stage: Beer-Lambert pseudo-absorbance, ratio-of-ratios, red/green slopes, DC absorbance
Everything is trailing-window (causal): a window ending at sample `te` only sees samples before `te`.
"""
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks

FS = 30
SOS_CARDIAC = butter(3, [0.7, 4.0], btype='band', fs=FS, output='sos')   # causal single-pass filter (sosfilt, never filtfilt)


def resp_rate(x, fs):
    """Dominant frequency (breaths/min) of a slowly sampled series, 0.1-0.6 Hz band."""
    x = x - x.mean()
    if len(x) < 8 or x.std() == 0:
        return np.nan
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=1024))
    fr = np.fft.rfftfreq(1024, 1 / fs)
    b = (fr >= 0.1) & (fr <= 0.6)
    return fr[b][np.argmax(sp[b])] * 60


def rec_feats(rgb):
    """rgb: (N,3) per-frame mean R,G,B on a 0-255 scale. Returns log-intensity and its causal band-passed version."""
    lg = np.log(np.clip(rgb, 1e-3, None) + 1.0)
    ac = np.stack([sosfilt(SOS_CARDIAC, lg[:, c]) for c in range(3)], 1)
    return lg, ac


def win_feats(lg, ac, te):
    """Features of the 10 s window ending at sample index te (exclusive), with 30 s trailing context for beats."""
    f = {}
    a10 = ac[te - 10 * FS:te]
    a30 = ac[max(0, te - 30 * FS):te]
    l10 = lg[te - 10 * FS:te]
    sd = a10.std(0) + 1e-9
    dc = l10.mean(0)
    # S2 physics
    f['ror_RG'] = np.log(sd[0] / sd[1]); f['ror_RB'] = np.log(sd[0] / sd[2]); f['ror_GB'] = np.log(sd[1] / sd[2])
    C = np.cov(a10.T)
    f['sl_RG'] = C[0, 1] / (C[1, 1] + 1e-12); f['sl_RB'] = C[0, 2] / (C[2, 2] + 1e-12)
    C3 = np.cov(a30.T)
    f['sl30_RG'] = C3[0, 1] / (C3[1, 1] + 1e-12); f['ror30_RG'] = 0.5 * np.log(C3[0, 0] / (C3[1, 1] + 1e-12))
    f['dc_R'] = dc[0]; f['dc_G'] = dc[1]; f['dc_B'] = dc[2]
    # S1 cardio
    for i, c in enumerate('RGB'):
        f[f'pi_{c}'] = np.log(sd[i])
    g = a30[:, 1] if sd[1] > sd[0] * 0.3 else a30[:, 0]
    x = -g
    pk, _ = find_peaks(x, distance=int(FS * 0.33), prominence=np.std(x) * 0.3)
    ibi = np.array([])
    if len(pk) >= 6:
        ibi = np.diff(pk) / FS
        ibi = ibi[(ibi > 0.3) & (ibi < 1.6)]
    if len(ibi) >= 5:
        f['hr'] = 60 / np.median(ibi)
        f['rmssd'] = np.log(np.sqrt(np.mean(np.diff(ibi) ** 2)) + 1e-3)
        f['sdnn'] = np.log(ibi.std() + 1e-3)
        amp = x[pk]; tt = pk / FS
        ti = np.arange(tt[0], tt[-1], 0.25)
        if len(ti) > 16:
            f['rr_amp'] = resp_rate(np.interp(ti, tt, amp), 4)
            f['rr_ibi'] = resp_rate(np.interp(ti, tt[1:], np.diff(pk) / FS), 4)
        f['beat_cv'] = np.std(amp) / (np.mean(np.abs(amp)) + 1e-9)
    return f
