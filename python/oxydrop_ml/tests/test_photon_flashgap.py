"""FlashGap (python/oxydrop_ml/photon/flashgap.py): augmentation disjoint from the stress evaluation, feature sanity, abstention rule, leakage guard."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml.photon import flashgap as F  # noqa: E402
from oxydrop_ml.photon import stress as ST  # noqa: E402


def _ppg(n_sec=150, seed=0, dead_blue=False):
    """Synthetic per-frame ROI-mean RGB at 30 fps: DC (45, 60, 40) with a 1.2 Hz pulse and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_sec * F.FS) / F.FS
    pulse = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t + 0.5)
    dc = np.array([45.0, 60.0, 40.0])
    amp = np.array([0.004, 0.006, 0.005])
    x = dc[None] * np.exp(-amp[None] * pulse[:, None]) + 0.02 * rng.standard_normal((len(t), 3))
    if dead_blue:
        x[:, 2] = 40.0
    return x


# ------------------------------------------------------------------------------------------------ disjointness
def test_aug_keys_live_in_their_own_namespace():
    keys = [F.aug_key(s, r, k) for s in range(3) for r in range(12) for k in range(F.K_COPIES)]
    pairs = [(lv, s) for lv in ST.LEVELS + list(ST.X_LEVELS) for s in range(3)]
    assert F.assert_disjoint(keys, pairs)
    assert all(k[0] == F.AUG_TAG for k in keys)
    assert F.AUG_TAG >= F.EVAL_SEED_MAX
    with pytest.raises(AssertionError):
        F.assert_disjoint([(0, 0, 0, 0)], pairs)                      # a key outside the FlashGap namespace
    with pytest.raises(AssertionError):
        F.assert_disjoint(keys, [(3, F.AUG_TAG)])                     # an evaluation seed that could start an augmentation stream


def test_aug_draws_never_equal_a_stress_level():
    params = [F.sample_params(F.aug_rng(s, r, k)) for s in range(3) for r in range(12) for k in range(F.K_COPIES)]
    assert F.assert_disjoint([], [], params)
    lo, hi = F.AUG_PRIOR['d_rho_mm'][1:]
    assert all(lo <= p['d_rho_mm'] <= hi and 0 <= p['leak'] <= F.AUG_PRIOR['leak'][2] for p in params)
    bad = dict(params[0], d_rho_mm=float(ST.D_RHO_MM[3]), leak=float(ST.AMBIENT[3]))
    with pytest.raises(AssertionError):
        F.assert_disjoint([], [], [bad])


def test_aug_stream_differs_from_every_eval_stream():
    rgb = _ppg(20)
    aug = np.stack([F.aug_rng(s, r, k).standard_normal(16) for s in range(3) for r in range(12) for k in range(F.K_COPIES)])
    for lv in ST.LEVELS[1:] + [100 + i for i in range(len(ST.X_LEVELS))]:
        for seed in range(5):
            ev = ST._rng(rgb, lv, seed).standard_normal(16)
            assert not np.any(np.all(np.isclose(aug, ev[None]), axis=1))


def test_augmentation_never_calls_the_stress_test(monkeypatch):
    rgb = _ppg(40)
    ref = [ST.degrade(rgb, lv, s) for lv in ST.LEVELS[1:] for s in range(2)] + [ST.degrade_x(rgb, x, 0) for x in ST.X_LEVELS]

    def boom(*a, **k):
        raise RuntimeError('FlashGap augmentation must not call the stress test')
    monkeypatch.setattr(ST, 'degrade', boom)
    monkeypatch.setattr(ST, 'degrade_x', boom)
    for k in range(F.K_COPIES):
        p, rng = F.draw(0, 0, k)
        out = F.flash_augment(rgb, p, rng)
        assert out.shape == rgb.shape and np.isfinite(out).all() and out.min() >= 0 and out.max() <= 255
        assert all(not np.array_equal(out, r) for r in ref)


def test_augmentation_is_deterministic_and_physical():
    rgb = _ppg(40)
    p1, r1 = F.draw(1, 3, 2)
    p2, r2 = F.draw(1, 3, 2)
    assert p1 == p2
    assert np.array_equal(F.flash_augment(rgb, p1, r1), F.flash_augment(rgb, p2, r2))
    p = dict(F.sample_params(F.aug_rng(0, 0, 0)), d_rho_mm=8.0, path_scale=0.25, rho0_mm=3.0)
    a, m = F.optics_gain(p)
    assert a[2] < a[1] < a[0] < 1.0                                    # blue dies first, red last


def test_training_conditions_are_clean_or_augmented_only():
    Xd = {'clean': np.zeros((10, len(F.COLS))), ('L', 1, 0): np.zeros((10, len(F.COLS)))}
    with pytest.raises(AssertionError):
        F.fit_fold(Xd, np.zeros(10, int), np.arange(10.0), np.array(['a'] * 10), np.arange(10), ['clean', ('L', 1, 0)], F.MODELS['flashgap'])


# ------------------------------------------------------------------------------------------------ features
def test_quality_features_flag_a_dead_channel():
    t, X = F.rec_features(_ppg(150, dead_blue=True))
    assert X.shape == (len(t), len(F.COLS))
    ev = t >= 60
    q = {c: X[ev, F.COLS.index(f'snr_{c}')] for c in 'RGB'}
    assert np.median(q['R']) > 0 and np.median(q['G']) > 0
    assert np.all(q['B'] == -6.0)
    assert np.median(X[ev, F.COLS.index('acf')]) > 0.6
    assert np.isfinite(X[:, F.I_FUS]).all()


def test_respiratory_group_is_causal():
    rgb = _ppg(150)
    t, X = F.rec_features(rgb)
    rgb2 = rgb.copy(); rgb2[100 * F.FS:] *= 1.0 + 0.02 * np.sin(np.arange(len(rgb2) - 100 * F.FS)[:, None] / 7.0)
    t2, X2 = F.rec_features(rgb2)
    early = t + F.WIN <= 100
    assert np.allclose(X[early][:, F.I_RESP], X2[early][:, F.I_RESP], equal_nan=True)


# ------------------------------------------------------------------------------------------------ abstention rule
def test_select_tau_reaches_the_target_when_it_can():
    rng = np.random.default_rng(0)
    n = 4000
    g = rng.random(n)
    y = rng.integers(0, 2, n)
    correct = rng.random(n) < 0.5 + 0.5 * g                          # accuracy rises with g
    pred = np.where(correct, y, 1 - y)
    tau, cert, cov, acc = F.select_tau(g, pred, y)
    a = g >= tau
    assert cert and 0.05 < cov < 1.0
    assert (pred[a] == y[a]).mean() >= 0.9
    tau2, cert2, _, _ = F.select_tau(rng.random(n), np.where(rng.random(n) < 0.6, y, 1 - y), y)
    assert not cert2


# ------------------------------------------------------------------------------------------------ leakage guard on a synthetic LOSO
def _synthetic(seed=0, n_sub=6, n_per=110):
    rng = np.random.default_rng(seed)
    sid = np.repeat([f'p{i}' for i in range(n_sub)], n_per)
    t = np.tile(60.0 + 2 * np.arange(n_per), n_sub)
    y = ((t > 120) & (t < 230)).astype(int)
    X = rng.normal(size=(len(y), len(F.COLS)))
    X[:, 0] += 1.2 * y                                                 # dHR carries the label
    X[:, 9] += 0.8 * y
    Xa = X + rng.normal(scale=1.0, size=X.shape)                       # a noisier 'augmented' copy
    return {'clean': X, ('aug', 0, 0): Xa}, y, t, sid


@pytest.mark.parametrize('name', ['flashgap', 'flashgap_gate'])
def test_scrambling_test_labels_changes_nothing(name):
    Xd, y, t, sid = _synthetic()
    spec = dict(F.MODELS[name])
    conds = ['clean', ('aug', 0, 0)]
    a = F.loso_fit(Xd, y, t, sid, conds, spec, log=lambda *x: None)
    y2 = y.copy()
    held = sid == 'p2'
    y2[held] = np.random.default_rng(1).permutation(y2[held])
    b = F.loso_fit(Xd, y2, t, sid, conds, spec, log=lambda *x: None)
    fa = next(f for f in a if f['sid'] == 'p2'); fb = next(f for f in b if f['sid'] == 'p2')
    assert fa['info']['cfg'] == fb['info']['cfg'] and fa['tau'] == fb['tau']
    pa = fa['model'].predict(Xd['clean'][fa['te']]); pb = fb['model'].predict(Xd['clean'][fb['te']])
    assert np.array_equal(pa, pb)


def test_null_labels_keep_prevalence_per_person():
    Xd, y, t, sid = _synthetic()
    yn = F.null_labels(y, sid, 'p0')
    for s in np.unique(sid):
        assert yn[sid == s].sum() == y[sid == s].sum()
    assert not np.array_equal(yn, y)
    assert np.array_equal(yn, F.null_labels(y, sid, 'p0'))
    yn1 = F.null_labels(y, sid, 'p0', null_seed=1)
    assert not np.array_equal(yn1, yn) and all(yn1[sid == s].sum() == y[sid == s].sum() for s in np.unique(sid))
    assert np.array_equal(F.null_labels(y, sid, 'p0', null_seed=0), yn)


def test_gate_routes_by_snr_only():
    Q = np.zeros((4, len(F.QUAL)))
    Q[:, 0] = [-2.0, 0.0, 1.0, -1.0]                                   # snr_R
    Q[:, 1] = [-2.0, -1.0, 0.5, 2.0]                                   # snr_G
    pa, pb = np.full(4, 0.9), np.full(4, 0.1)
    out = F.Gate(0.25).predict(pa, pb, Q)
    assert np.array_equal(out, [0.1, 0.1, 0.9, 0.9])                   # max(snr_R, snr_G) >= 0.25 -> chain expert A
    assert np.array_equal(F.Gate(-np.inf).predict(pa, pb, Q), pa) and np.array_equal(F.Gate(np.inf).predict(pa, pb, Q), pb)


# ------------------------------------------------------------------------------------------------ flashgap_gate (post hoc, requested 17:15 CT)
def test_gate_family_is_labelled_post_hoc():
    assert F.POST_HOC == 'post hoc, requested 17:15 CT after outer results were seen'
    for nm in ('flashgap_gate', 'flashgap_gate_s1', 'flashgap_gate_s2', 'flashgap_gate_null'):
        sp = F.MODELS[nm]
        assert sp['post_hoc'] and sp['status'] == F.POST_HOC and sp['note'].startswith(F.POST_HOC)
        assert sp['gate_mode'] == 'occam' and sp['sel'] == 'clean_acc'
    assert F.MODELS['flashgap_v1']['status'] == F.STATUS_PREREG
    assert all(not sp.get('post_hoc') for nm, sp in F.MODELS.items() if not nm.startswith('flashgap_gate'))
    assert F.MODELS['flashgap_gate_null']['null'] and {F.MODELS[n]['aug_seed'] for n in ('flashgap_gate', 'flashgap_gate_s1', 'flashgap_gate_s2')} == {0, 1, 2}


def test_gate_choices_are_hard_snr_gates_and_tau_meets_the_clean_inner_target():
    Xd, y, t, sid = _synthetic()
    folds = F.loso_fit(Xd, y, t, sid, ['clean', ('aug', 0, 0)], dict(F.MODELS['flashgap_gate']), log=lambda *x: None)
    assert len(folds) == len(np.unique(sid))
    for fo in folds:
        i = fo['info']
        assert i['gate'].startswith('hard gate theta=') and isinstance(fo['model'].st, F.Gate)
        assert all(g['kind'] == 'gate' for g in i['gate_trace'])
        if i['tau_certified']:
            assert i['inner_answered_acc'] >= F.SEL_TARGET
        assert 0.0 <= i['inner_aug_coverage'] <= 1.0
        assert not np.any(np.isin(fo['te'], np.nonzero(sid != fo['sid'])[0]))
