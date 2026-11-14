"""TINY chain: per-person calibration deltas -> two logistic regressions (dHR alone; 19-input fusion) -> averaged probability.

Weights: 62 floats (python/tiny_chain_weights.json, copied to app/data/weights.json). No neural network.
Hyperparameters were fixed to the nested leave-one-person-out modal choice: C=0.01 (dHR LR), C=0.1 (fusion LR), no smoothing.
"""
import json
import numpy as np
from .features import rec_feats, win_feats, FS

FUSION = ['hr', 'rmssd', 'sdnn', 'rr_amp', 'rr_ibi', 'pi_R', 'pi_G', 'pi_B', 'beat_cv', 'ror_RG', 'ror_RB', 'ror_GB',
          'sl_RG', 'sl_RB', 'sl30_RG', 'ror30_RG', 'dc_R', 'dc_G', 'dc_B']
CAL, WIN, STEP, BAND = 60, 10, 2, 0.10   # calibration seconds, window, hop, default abstain half-width


def deltas(rgb):
    """Window start times and calibration-relative features. Causal forward fill, then calibration-median fill."""
    lg, ac = rec_feats(rgb)
    n = len(rgb) // FS
    rows = []
    for s in range(0, n - WIN + 1, STEP):
        f = win_feats(lg, ac, (s + WIN) * FS)
        rows.append([s] + [f.get(c, np.nan) for c in FUSION])
    A = np.array(rows, float)
    t, X = A[:, 0], A[:, 1:]
    for j in range(X.shape[1]):
        for i in range(1, len(X)):
            if np.isnan(X[i, j]):
                X[i, j] = X[i - 1, j]
    base = np.nanmedian(X[t + WIN <= CAL], 0)
    return t, np.where(np.isnan(X), base, X) - base


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _lr(m, X):
    return _sigmoid(((X - np.array(m['mu'])) / np.array(m['sd'])) @ np.array(m['coef']) + m['b'])


def infer(rgb, weights, band=BAND):
    """rgb (N,3) at 30 fps -> list of (window_start_s, p_hypoxic, 'HYPOXIC'|'normal'|'unsure') for windows with t >= 60 s."""
    t, X = deltas(rgb)
    p = 0.5 * (_lr(weights['hr'], X[:, :1]) + _lr(weights['fusion'], X))
    return [(int(ti), float(pi), 'unsure' if abs(pi - .5) < band else ('HYPOXIC' if pi > .5 else 'normal'))
            for ti, pi in zip(t, p) if ti >= CAL]


def fit(recs, exclude_sid=None):
    """Fit both logistic regressions on windows (t >= 60 s) of `recs` (list of dicts with rec, sid, rgb, y_by_window).
    exclude_sid: hold one person out (both hands) for honest replay weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs, ys = [], []
    for r in recs:
        if exclude_sid is not None and r['sid'] == exclude_sid:
            continue
        t, X = deltas(r['rgb'])
        keep = np.isin(t, list(r['y'].keys())) & (t >= CAL)
        Xs.append(X[keep]); ys.append(np.array([r['y'][ti] < 90 for ti in t[keep]], int))
    X = np.vstack(Xs); y = np.concatenate(ys); W = {}
    for name, cols, C in (('hr', [0], 0.01), ('fusion', list(range(len(FUSION))), 0.1)):
        sc = StandardScaler().fit(X[:, cols])
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(sc.transform(X[:, cols]), y)
        W[name] = dict(cols=[FUSION[c] for c in cols], mu=sc.mean_.tolist(), sd=sc.scale_.tolist(),
                       coef=m.coef_[0].tolist(), b=float(m.intercept_[0]))
    return W


def load_weights(path):
    return json.load(open(path))
