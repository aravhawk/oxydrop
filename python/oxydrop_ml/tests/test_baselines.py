import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import DATA_DIR  # noqa: E402
from oxydrop_ml.baselines import common as C, hr_est as H, ror  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HAVE_HOF = os.path.exists(os.path.join(DATA_DIR, 'hoffman', 'meta.parquet'))


class _Ctx:
    pass


def _synthetic(n_sub=6, n_per=120, seed=0):
    rng = np.random.default_rng(seed)
    ctx = _Ctx()
    ctx.subject = np.repeat([f's{i}' for i in range(n_sub)], n_per)
    ctx.t = np.tile(np.arange(n_per) * 2.0, n_sub)
    x = rng.normal(size=(n_sub * n_per, 3))
    ctx.y = (x[:, 0] + 0.5 * rng.normal(size=len(x)) > 0).astype(int)
    ctx.x = x
    return ctx


def _logit_spec(ctx):
    from sklearn.linear_model import LogisticRegression

    class P:
        def __init__(self, m): self.m = m
        def score(self, rows): return self.m.predict_proba(ctx.x[rows])[:, 1]

    def fit(hp, rows):
        return P(LogisticRegression(C=hp['C'], max_iter=500).fit(ctx.x[rows], ctx.y[rows]))
    return C.Spec('toy', fit, [dict(C=c) for c in (0.01, 1.0, 100.0)], thr_mode='tune', select_on='bacc')


def test_nested_selection_never_sees_the_test_person():
    """Scrambling the labels (and features) of the held-out person must not change the selected hp, threshold, band or the scores of that fold."""
    ctx = _synthetic()
    a = C.run_cls(ctx, _logit_spec(ctx))
    ctx2 = _synthetic()
    te = np.nonzero(ctx2.subject == 's2')[0]
    rng = np.random.default_rng(1)
    ctx2.y[te] = rng.integers(0, 2, len(te)); ctx2.t[te] = rng.permutation(ctx2.t[te])
    b = C.run_cls(ctx2, _logit_spec(ctx2))
    fa = [f for f in a if f.subject == 's2'][0]; fb = [f for f in b if f.subject == 's2'][0]
    assert fa.hp == fb.hp and fa.thr == fb.thr and fa.band == fb.band and fa.spec_thr == fb.spec_thr
    assert np.array_equal(fa.s_te, fb.s_te) and np.array_equal(fa.s_tr, fb.s_tr)
    for f in a:                                                       # the leave-one-subject-out contract
        assert not (set(ctx.subject[f.tr]) & set(ctx.subject[f.te]))
        assert set(ctx.subject[f.te]) == {f.subject}


def test_choose_band_smallest_c_reaching_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = np.clip(0.5 + (y - 0.5) * rng.uniform(0, 0.9, 4000) + rng.normal(0, 0.15, 4000), 0, 1)
    c = C.choose_band(s, y, 0.5, target=0.90)
    ans = np.abs(s - 0.5) >= c
    assert ans.mean() < 1.0 and ((s[ans] > 0.5) == y[ans]).mean() >= 0.90
    prev = C.BAND_GRID[np.nonzero(C.BAND_GRID == c)[0][0] - 1] if c > 0 else None
    if prev is not None:
        a2 = np.abs(s - 0.5) >= prev
        assert ((s[a2] > 0.5) == y[a2]).mean() < 0.90 or C.M.classification_metrics(y[a2], s[a2], pred=(s[a2] > 0.5).astype(int))['bacc'] < 0.90


def test_spec_threshold_gives_requested_specificity_on_oof():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000); s = rng.normal(y, 1.0)
    thr = C.choose_spec_thr(s, y, 0.90)
    assert abs(((s[y == 0] <= thr).mean()) - 0.90) < 0.01


def test_gap_is_positive_when_the_model_overfits():
    r = C.summarize_metric([0.7, 0.8], [0.9, 0.95], 'acc')
    assert r['gap'] > 0 and abs(r['gap'] - 0.175) < 1e-12
    r = C.summarize_metric([2.0, 3.0], [1.0, 1.5], 'mae')            # lower is better: held-out error larger than train error is a positive gap
    assert r['gap'] > 0


def test_textbook_ratio_of_ratios():
    assert abs(ror.textbook_spo2(np.log(0.8)) - 90.0) < 1e-9
    assert abs(ror.textbook_spo2(0.0) - 85.0) < 1e-9
    assert ror.textbook_spo2(np.log(5)) == 70.0 and ror.textbook_spo2(np.log(0.1)) == 100.0


def _pulse(bpm, n=450, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    f = bpm / 60.0
    base = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t + 1.0)
    return np.stack([a * base + noise * rng.normal(size=n) for a in (0.6, 1.0, 0.4)], 1)


def test_hr_estimators_on_a_clean_pulse():
    ac = _pulse(72)
    w = ac[-300:]
    assert abs(H.naive_hr(w[:, 1]) - 72) < 3
    assert abs(H.v1_peak_hr(ac, 1) - 72) < 4
    from oxydrop import hr as HR
    e, ch = HR.best_channel_estimate(w, 1)
    assert e['valid'] and abs(e['hr'] - 72) < 3


def test_vectorised_gate_and_channel_selection_match_hr_module():
    from oxydrop import hr as HR
    rng = np.random.default_rng(0)
    raws, ref = [], []
    for k in range(12):
        ac = _pulse(rng.uniform(55, 150), noise=rng.uniform(0.02, 1.5), seed=k)
        w = ac[-300:]
        sd = w.std(0); prim = 1 if sd[1] > 0.3 * sd[0] else 0
        order = [prim] + [int(c) for c in np.argsort(-sd) if c != prim]
        rec = {m: np.full(3, np.nan) for m in ('hr', 'hr_acf', 'sqi', 'acf')}
        for j, c in enumerate(order):
            e = HR.estimate_hr(w[:, c], 30)
            for m in rec:
                rec[m][j] = e[m]
        raws.append(rec)
        e, _ = HR.best_channel_estimate(w, prim)
        ref.append((e['hr'], e['valid']))
    raw = {m: np.stack([r[m] for r in raws]) for m in raws[0]}
    hr, valid = H.select(raw, HR.SQI_MIN, HR.ACF_MIN)
    for (h, v), a, b in zip(ref, hr, valid):
        assert bool(v) == bool(b) and (np.isnan(h) and np.isnan(a) or abs(h - a) < 1e-9)


def test_gate_selection_uses_training_rows_only():
    from oxydrop_ml.baselines import p2
    n = 400
    ctx = _Ctx()
    rng = np.random.default_rng(0)
    ctx.gates = [(0.1, 0.1), (0.5, 0.5)]
    ref = rng.uniform(60, 100, n)
    good = rng.random(n) < 0.5
    est = np.where(good, ref + 1, ref + 30)
    ctx.ref = ref
    ctx.gate_est = {(0.1, 0.1): (est, np.ones(n, bool)), (0.5, 0.5): (est, good)}
    tr = np.arange(0, 300)
    g1, _ = p2.select_gate(ctx, tr)
    ctx.ref = ref.copy(); ctx.ref[300:] = rng.uniform(40, 200, 100)          # scramble the test rows' reference
    g2, _ = p2.select_gate(ctx, tr)
    assert g1 == g2 == (0.5, 0.5)


@pytest.mark.skipif(not HAVE_HOF, reason='Hoffman shard not built in the cache')
def test_v1_refit_reproduces_published_row():
    """chain_v1 refit LOSO (shipped C) matches results/eval_v1_vs_v2.csv row 1: acc .8754, bacc .8733, sens .8875, spec .8592, AUROC .9475, TM acc .7962."""
    from oxydrop_ml.baselines import p1
    ctx = p1.build_ctx()
    spec = [s for s in p1.all_specs(ctx, only=['chain_v1']) if s.name == 'chain_v1'][0]
    folds = C.run_cls(ctx, spec)
    order = np.concatenate([f.te for f in folds])
    s = np.concatenate([f.s_te for f in folds]); pred = (s > 0.5).astype(int)
    m = C.cls_metrics(ctx.y[order], ctx.t[order], s, pred, np.ones(len(s), bool))
    pub = pd.read_csv(REPO / 'results' / 'eval_v1_vs_v2.csv').iloc[0]
    for k, pk in (('acc', 'acc'), ('bacc', 'bacc'), ('sens', 'sens'), ('spec', 'spec'), ('auroc', 'auc')):
        assert abs(m[k] - pub[pk]) < 6e-4, (k, m[k], pub[pk])
    assert abs(m['tm_bacc'] - pub['tm_acc']) < 2e-3
    assert abs(float(ctx.y.mean()) - 0.5748) < 1e-4                         # majority class share
