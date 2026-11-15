"""TINY chain v2: same two logistic regressions on calibration-relative features, but with v2 (robust-HR, gated) features.
Decisions: 'HYPOXIC' / 'normal' / 'unsure' (low confidence) / 'nolock' (pulse not locked or signal invalid: abstain)."""
import json
import numpy as np
from .features_v2 import win_feats, rec_feats, FS
from .chain import FUSION, CAL, WIN, STEP, BAND, _lr


def deltas(rgb):
    """Returns window starts t, deltas X (n,19), and valid flags. Invalid windows have NaN features (forward-filled, never used for the baseline)."""
    lg, ac = rec_feats(rgb)
    n = len(rgb) // FS
    rows, valid, aux = [], [], []
    for s in range(0, n - WIN + 1, STEP):
        f = win_feats(lg, ac, (s + WIN) * FS)
        v = f['_valid']
        rows.append([s] + [f.get(c, np.nan) if v else np.nan for c in FUSION]); valid.append(v)
        aux.append((f['_hr_est'], f['_sqi']))
    A = np.array(rows, float); t, X = A[:, 0], A[:, 1:]
    valid = np.array(valid)
    for j in range(X.shape[1]):
        for i in range(1, len(X)):
            if np.isnan(X[i, j]):
                X[i, j] = X[i - 1, j]
    base = np.nanmedian(X[t + WIN <= CAL], 0)
    if np.isnan(base).any():
        # no baseline for some feature (e.g. the pulse never locked during the 60 s calibration, such as a resting HR below the 51 bpm band edge):
        # the whole recording is 'nolock' rather than being scored against a missing baseline
        base = np.where(np.isnan(base), 0.0, base); valid = np.zeros_like(valid)
    return t, np.where(np.isnan(X), base, X) - base, valid, np.array(aux)


def infer(rgb, weights, band=BAND):
    t, X, valid, aux = deltas(rgb)
    p = 0.5 * (_lr(weights['hr'], X[:, :1]) + _lr(weights['fusion'], X))
    out = []
    for ti, pi, v in zip(t, p, valid):
        if ti < CAL:
            continue
        lab = 'nolock' if not v else ('unsure' if abs(pi - .5) < band else ('HYPOXIC' if pi > .5 else 'normal'))
        out.append((int(ti), float(pi), lab))
    return out


def fit(recs, exclude_sid=None):
    """recs: hoffman.load() records. Trains on VALID windows with t >= 60 s."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs, ys = [], []
    for r in recs:
        if exclude_sid is not None and r['sid'] == exclude_sid:
            continue
        t, X, valid, _ = r.setdefault('_d2', deltas(r['rgb']))
        keep = np.isin(t, list(r['y'].keys())) & (t >= CAL) & valid
        Xs.append(X[keep]); ys.append(np.array([r['y'][ti] < 90 for ti in t[keep]], int))
    X = np.vstack(Xs); y = np.concatenate(ys); W = {}
    for name, cols, C in (('hr', [0], 0.01), ('fusion', list(range(len(FUSION))), 0.1)):
        sc = StandardScaler().fit(X[:, cols])
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(sc.transform(X[:, cols]), y)
        W[name] = dict(cols=[FUSION[c] for c in cols], mu=sc.mean_.tolist(), sd=sc.scale_.tolist(), coef=m.coef_[0].tolist(), b=float(m.intercept_[0]))
    return W
