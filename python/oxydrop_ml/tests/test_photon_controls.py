import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml.baselines import common as C  # noqa: E402
from oxydrop_ml.photon import controls as K  # noqa: E402


class _Ctx:
    pass


def _synthetic(n_sub=6, n_per=120, seed=0):
    """6 people x 120 windows, 19 chain-like columns where dc_R / dc_B and dHR carry the label."""
    rng = np.random.default_rng(seed)
    ctx = _Ctx()
    ctx.subject = np.repeat([f's{i}' for i in range(n_sub)], n_per)
    ctx.t = np.tile(np.arange(n_per) * 2.0 + 60, n_sub)
    n = n_sub * n_per
    ctx.y = (rng.normal(size=n) > 0).astype(int)
    X = rng.normal(size=(n, 19))
    for name, w in (('dc_R', 1.0), ('dc_B', -0.8), ('hr', 0.6)):
        X[:, K.FUSION.index(name)] += w * (ctx.y - 0.5)
    ctx.X = X
    return ctx


def test_feature_sets_and_column_order():
    assert K.FUSION == ['hr', 'rmssd', 'sdnn', 'rr_amp', 'rr_ibi', 'pi_R', 'pi_G', 'pi_B', 'beat_cv', 'ror_RG', 'ror_RB', 'ror_GB',
                        'sl_RG', 'sl_RB', 'sl30_RG', 'ror30_RG', 'dc_R', 'dc_G', 'dc_B']
    assert K.columns(K.MODELS['ctl_dc_lr']) == [16, 17, 18]
    assert K.columns(K.MODELS['ctl_dc_hr_lr']) == [16, 17, 18, 0]
    assert set(K.MODELS['ctl_dc_ror_lr']) == {'dc_R', 'dc_G', 'dc_B', 'ror_RG', 'ror_RB', 'ror_GB', 'pi_R', 'pi_G', 'pi_B'}


def test_scrambling_the_held_out_person_changes_no_C_threshold_or_band():
    """Labels, times and features of the held-out person are scrambled: chosen C, threshold, abstain half-width and the fold's training scores stay identical."""
    ctx = _synthetic()
    for name, feats in K.MODELS.items():
        a = C.run_cls(ctx, K.make_spec(ctx.X, ctx.y, name, feats))
        ctx2 = _synthetic()
        te = np.nonzero(ctx2.subject == 's2')[0]
        rng = np.random.default_rng(1)
        ctx2.y[te] = rng.integers(0, 2, len(te)); ctx2.t[te] = rng.permutation(ctx2.t[te])
        ctx2.X[te] = rng.normal(size=(len(te), 19))
        b = C.run_cls(ctx2, K.make_spec(ctx2.X, ctx2.y, name, feats))
        fa = [f for f in a if f.subject == 's2'][0]; fb = [f for f in b if f.subject == 's2'][0]
        assert fa.hp == fb.hp and fa.thr == fb.thr == K.THR and fa.band == fb.band
        assert np.array_equal(fa.s_tr, fb.s_tr)
        for f in a:                                                   # leave-one-person-out contract
            assert not (set(ctx.subject[f.tr]) & set(ctx.subject[f.te]))


def test_standardiser_is_fit_on_the_training_rows_only():
    ctx = _synthetic()
    spec = K.make_spec(ctx.X, ctx.y, 'ctl_dc_lr', K.MODELS['ctl_dc_lr'])
    rows = np.nonzero(ctx.subject != 's0')[0]
    m = spec.fit(dict(C=1.0), rows)
    assert np.allclose(m.scaler.mean_, ctx.X[rows][:, K.columns(K.DC)].mean(0))
    assert m.lr.class_weight == 'balanced'
    X2 = ctx.X.copy(); X2[ctx.subject == 's0'] += 100.0               # degrading the held-out person must not change the fitted model
    assert np.array_equal(m.on(X2).lr.coef_, m.lr.coef_) and np.allclose(m.on(X2).scaler.mean_, m.scaler.mean_)
    assert not np.allclose(m.on(X2).score(np.nonzero(ctx.subject == 's0')[0]), m.score(np.nonzero(ctx.subject == 's0')[0]))


def test_signal_is_learned_on_the_synthetic_cue():
    ctx = _synthetic()
    spec = K.make_spec(ctx.X, ctx.y, 'ctl_dc_hr_lr', K.MODELS['ctl_dc_hr_lr'])
    folds = C.run_cls(ctx, spec)
    s = np.concatenate([f.s_te for f in folds]); y = ctx.y[np.concatenate([f.te for f in folds])]
    assert ((s > 0.5) == y).mean() > 0.65
    assert all(f.hp['C'] in {g['C'] for g in K.GRID_C} for f in folds)
