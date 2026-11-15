"""Label-free, per-person score post-processing for the hybrid (FEAT-008): everything here uses only the person's own earlier windows.

  * `baseline_median`   per-user calibration: median of a feature over the person's own first-minute windows of the same recording.
  * `causal_mean`       multi-window aggregation: mean of the last k windows of the same recording (2 s hop, so k windows cover 2 (k - 1) + 10 s).

Test-time BatchNorm adaptation (tried in FEAT-006, it hurts) lives in `stack_run.py`. A window is addressed by (rec, t_start); `order` helpers keep the recording order. The functions are pure numpy so tests can run them without a model.
"""
from __future__ import annotations

import numpy as np


def _groups(rec):
    rec = np.asarray(rec)
    return {r: np.nonzero(rec == r)[0] for r in np.unique(rec)}


def baseline_median(feat, rec, t_start, in_base):
    """Per recording: median of `feat` ([n] or [n, d]) over the rows flagged `in_base` (windows that end inside the first minute).
    Returns an array shaped like feat with each row holding its recording's baseline median (0 when the recording has no baseline row)."""
    feat = np.asarray(feat, float)
    out = np.zeros_like(feat)
    in_base = np.asarray(in_base, bool)
    for r, idx in _groups(rec).items():
        b = idx[in_base[idx]]
        if len(b):
            out[idx] = np.median(feat[b], axis=0)
    return out


def causal_mean(feat, rec, t_start, k):
    """Trailing mean over the last k rows (ordered by t_start) of the same recording; the first rows use the rows that exist. k = 1 returns feat."""
    feat = np.asarray(feat, float)
    if k <= 1:
        return feat.copy()
    out = np.empty_like(feat)
    t_start = np.asarray(t_start, float)
    for r, idx in _groups(rec).items():
        o = idx[np.argsort(t_start[idx], kind='stable')]
        v = feat[o]
        c = np.cumsum(np.concatenate([np.zeros((1,) + v.shape[1:]), v], axis=0), axis=0)
        n = np.arange(1, len(o) + 1)
        lo = np.maximum(n - k, 0)
        s = c[n] - c[lo]
        out[o] = s / (n - lo).reshape((-1,) + (1,) * (v.ndim - 1))
    return out


def calibrate_and_smooth(feat, rec, t_start, in_base, cal, k, base_feat=None):
    """The two levers of the hybrid pipeline in the declared order: subtract the own-baseline median (cal), then average the last k windows.
    `feat` rows that are baseline windows are kept in the arrays (they are needed for the median); `base_feat` can hold the features to take the median over."""
    f = np.asarray(feat, float)
    if cal:
        f = f - baseline_median(f if base_feat is None else base_feat, rec, t_start, in_base)
    return causal_mean(f, rec, t_start, k)
