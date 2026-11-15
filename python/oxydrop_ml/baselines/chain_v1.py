"""v1 logistic chain (python/oxydrop/chain.py) refit under the common LOSO harness.

Model: two logistic regressions (dHR alone, and the 19 calibration-relative FUSION features), class-balanced, standardised, probability = mean of the two.
`chain_v1` uses the C values of the shipped model (C = 0.01 for dHR, 0.1 for fusion) and the shipped decision rule (p > 0.5, abstain band 0.10 as its own variant).
`chain_v1_nestedC` selects (C_hr, C_fus) on a 5 x 5 grid by the inner subject-grouped CV of the training people (selection metric: time-matched balanced accuracy).
Features are computed from the ORIGINAL Hoffman CSVs with python/oxydrop/chain.py (when the repository is present) so the refit reproduces
results/eval_v1_vs_v2.csv; otherwise they are rebuilt from the float32 shard (differences ~1e-7 relative).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import common as C

SHIPPED = dict(C_hr=0.01, C_fus=0.1)
GRID_C = [dict(C_hr=a, C_fus=b) for a in (0.001, 0.01, 0.1, 1.0, 10.0) for b in (0.001, 0.01, 0.1, 1.0, 10.0)]


def hoffman_records(ws):
    """[(rec_name_in_shard, rgb (N,3) linear 0-255)] for the 12 Hoffman recordings: raw CSVs if available, else exp(series) - 1 from the shard."""
    try:
        from oxydrop import hoffman
        from ..data.hoffman import ROOT
        return [('hof_' + r['rec'], r['rgb']) for r in hoffman.load(ROOT)], 'raw Hoffman CSVs'
    except Exception:                                    # pragma: no cover - dataset repository not present
        out = []
        for _, r in ws.recs.iterrows():
            seg = np.asarray(ws.series[int(r.offset):int(r.offset) + int(r.length)], np.float64)
            out.append((r['rec'], np.expm1(seg)))
        return out, 'float32 shard (exp(series) - 1)'


def chain_features(ws, rows_meta, version):
    """Calibration-relative FUSION deltas for the rows of rows_meta (DataFrame with rec, t_start). version 1 or 2.
    Returns X [n, 19], valid [n] (all True for v1), source string."""
    from oxydrop import chain, chain_v2
    recs, src = hoffman_records(ws)
    key = {(r, int(t)): i for i, (r, t) in enumerate(zip(rows_meta.rec.values, rows_meta.t_start.values))}
    X = np.full((len(rows_meta), 19), np.nan); valid = np.zeros(len(rows_meta), bool)
    for rec, rgb in recs:
        if version == 1:
            t, D = chain.deltas(rgb); v = np.ones(len(t), bool)
        else:
            t, D, v, _ = chain_v2.deltas(rgb)
        for ti, di, vi in zip(t, D, v):
            i = key.get((rec, int(ti)))
            if i is not None:
                X[i] = di; valid[i] = vi
    assert np.isfinite(X).all(), 'every evaluated window must have chain features'
    return X, valid, src


class ChainPred:
    def __init__(self, models, X, valid_arr=None):
        self.m = models; self.X = X; self._v = valid_arr

    def score(self, rows):
        x = self.X[rows]
        p = 0.0
        for cols, sc, lr in self.m:
            p = p + lr.predict_proba(sc.transform(x[:, cols]))[:, 1]
        return p / len(self.m)

    def valid(self, rows):
        return np.ones(len(rows), bool) if self._v is None else self._v[rows]


def fit_chain(X, y, valid, only_valid):
    """Spec.fit for the chains: fit both logistic regressions on the given training rows (v2 trains on valid windows only)."""
    def fit(hp, rows):
        r = rows[valid[rows]] if only_valid else rows
        models = []
        for cols, c in (([0], hp['C_hr']), (list(range(19)), hp['C_fus'])):
            sc = StandardScaler().fit(X[r][:, cols])
            lr = LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(sc.transform(X[r][:, cols]), y[r])
            models.append((cols, sc, lr))
        return ChainPred(models, X, valid if only_valid else None)
    return fit


def specs(ctx, version=1):
    X = ctx.F[f'fusion19_v{version}']; valid = ctx.valid[version]
    fit = fit_chain(X, ctx.y, valid, only_valid=(version == 2))
    nm = f'chain_v{version}'
    return [C.Spec(nm, fit, [SHIPPED], thr_mode='fixed', fixed_thr=0.5, shipped_band=0.10,
                   note=f'v{version} logistic chain refit LOSO, shipped C (0.01 dHR / 0.1 fusion), p>0.5' + ('; no-lock windows forced to answer with forward-filled features' if version == 2 else '')),
            C.Spec(nm + '_nestedC', fit, GRID_C, thr_mode='fixed', fixed_thr=0.5, shipped_band=0.10,
                   note=f'v{version} logistic chain, (C_hr, C_fus) selected by inner CV on training people, p>0.5')]
