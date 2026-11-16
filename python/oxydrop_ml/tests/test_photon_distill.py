"""PhotonDistill (python/oxydrop_ml/photon/distill_student.py): student learns, teachers and nesting see training rows only, nulls, export round trip."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml.photon import distill_student as D  # noqa: E402
from oxydrop_ml.photon import flashgap as F  # noqa: E402
from oxydrop_ml.photon import stress as ST  # noqa: E402


def _toy(n_people=4, n=80, seed=0):
    """Synthetic table: 39 features, hypoxia driven by the hr column (index 0) and dc_R; 2 conditions (clean + one 'aug' copy)."""
    rng = np.random.default_rng(seed)
    N = n_people * n
    sid = np.repeat([f'p{i}' for i in range(n_people)], n)
    t = np.tile(np.arange(60, 60 + 2 * n, 2), n_people).astype(float)
    y = (np.tile(np.arange(n), n_people) % 20 >= 10).astype(int)
    X = rng.standard_normal((N, len(F.COLS)))
    X[:, 0] += 2.0 * y
    X[:, F.COLS.index('dc_R')] -= 1.5 * y
    Xa = X + 0.5 * rng.standard_normal(X.shape)
    Xa[:, 0] = rng.standard_normal(N)                     # heart-rate cue destroyed in the degraded copy, colour kept
    conds = ['clean', ('aug', 0, 0)]
    return {'clean': X, ('aug', 0, 0): Xa}, conds, y, t, sid, 90 - 8 * y + rng.standard_normal(N)


def test_student_learns_and_heads_have_the_right_shape():
    Xd, conds, y, t, sid, s2 = _toy()
    rows = np.arange(len(y))
    Xs, ys, w, q = D.soft_targets(Xd, conds, rows, y)
    assert Xs.shape == (2 * len(y), len(F.COLS)) and np.all((q > 0) & (q < 1))
    m = D.train_student(Xs, ys, w, q, np.tile(s2, 2), 1.0, 30, 0)[30]
    p, sp, a = m.predict(Xd['clean'])
    assert p.shape == sp.shape == a.shape == (len(y),)
    assert np.mean((p > 0.5) == y) > 0.85
    assert np.mean(np.abs(sp - s2)) < 4.0
    assert np.all((a >= 0) & (a <= 1))


def test_fit_fold_uses_only_training_rows():
    Xd, conds, y, t, sid, s2 = _toy()
    tr = np.nonzero(sid != 'p0')[0]
    spec = dict(seed=0, kd=True, null=None)
    m1, tau1, info1 = D.fit_fold(Xd, conds, tr, y[tr], s2[tr], t[tr], sid[tr], spec)
    te = np.nonzero(sid == 'p0')[0]
    Xd2 = {c: X.copy() for c, X in Xd.items()}
    for X in Xd2.values():
        X[te] = 1e3 * np.random.default_rng(1).standard_normal((len(te), X.shape[1]))   # held-out rows scrambled
    m2, tau2, info2 = D.fit_fold(Xd2, conds, tr, y[tr], s2[tr], t[tr], sid[tr], spec)
    assert tau1 == tau2 and info1['grid'] == info2['grid']
    assert np.allclose(m1.predict(Xd['clean'][te])[0], m2.predict(Xd['clean'][te])[0])
    assert set(info1) >= {'lam', 'epochs', 'tau', 'inner_crit'}


def test_null_permutation_is_deterministic_and_draws_differ():
    a = D.null_perm(100, '1', 0); b = D.null_perm(100, '1', 0); c = D.null_perm(100, '1', 1)
    assert (a == b).all() and not (a == c).all() and sorted(a) == list(range(100))


def test_augmentation_disjoint_from_stress():
    keys = [F.aug_key(s, r, k) for s in range(3) for r in range(12) for k in range(F.K_COPIES)]
    pairs = [(lv, 0) for lv in ST.LEVELS + list(ST.X_LEVELS)]
    assert F.assert_disjoint(keys, pairs)
    assert D.train_conds(1) == ['clean'] + [('aug', 1, k) for k in range(F.K_COPIES)]
    assert 'degrade_x' not in Path(D.__file__).read_text()


def test_export_json_math_matches_student():
    from oxydrop_ml.photon import distill_export as E
    Xd, conds, y, t, sid, s2 = _toy()
    rows = np.arange(len(y))
    Xs, ys, w, q = D.soft_targets(Xd, conds, rows, y)
    m = D.train_student(Xs, ys, w, q, np.tile(s2, 2), 1.0, 5, 0)[5]
    st = E.student_json(m)
    p = {k: np.asarray(v) for k, v in st['prep'].items()}
    X = Xd['clean'][:7]
    z = (np.clip(np.where(np.isfinite(X), X, p['med']), p['lo'], p['hi']) - p['mu']) / p['sd']
    h = np.maximum(z @ np.asarray(st['w1']).T + st['b1'], 0); h = np.maximum(h @ np.asarray(st['w2']).T + st['b2'], 0)
    o = h @ np.asarray(st['wo']).T + st['bo']
    assert np.allclose(1 / (1 + np.exp(-o[:, 0])), m.predict(X)[0], atol=1e-6)


def test_pack_mode_reads_features_and_windows(tmp_path, monkeypatch):
    """Remote (Daytona) runs read a compact pack instead of the raw recordings; the pack round-trips the feature matrices and the window table."""
    n = 6
    X = np.arange(n * len(F.COLS), dtype=np.float32).reshape(n, len(F.COLS))
    p = tmp_path / 'pack.npz'
    np.savez_compressed(p, rec=np.array(['a'] * n, dtype='U'), t=np.arange(60, 60 + 2 * n, 2), sid=np.array([1, 1, 1, 2, 2, 2]), y_spo2=np.full(n, 95.0),
                        y=np.zeros(n, int), ri=np.zeros(n, int), stress_hash='x', disjoint_checked=True, n_aug_params=0,
                        X_clean=X, X_aug_s0_k0=X + 1)
    monkeypatch.setenv(D.PACK_ENV, str(p))
    monkeypatch.setattr(D, '_PACK', {})
    monkeypatch.setattr(F, 'meta', F.meta)                  # restored after the test
    Xd, params = D.features(['clean', ('aug', 0, 0)])
    assert params is None and np.allclose(Xd[('aug', 0, 0)], X + 1) and Xd['clean'].dtype == np.float64
    assert len(F.meta()) == n and list(F.meta().sid) == [1, 1, 1, 2, 2, 2]
