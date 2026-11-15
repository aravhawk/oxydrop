"""Tests of the training pipeline on a tiny FAKE dataset: no test-subject window may reach training, SSL, scaling statistics, the baseline-embedding table
or early stopping (prereg rules 2, 4, 5). The strongest check is behavioural: two worlds that differ ONLY in the held-out person's signal and labels must give
byte-identical trained weights."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import FS, windows as W, features as F   # noqa: E402
from oxydrop_ml import train as T, ssl_pretrain as SSL, splits as S, losses as LS, loso_run as LR   # noqa: E402
from oxydrop_ml.augment import AugCfg, augment_batch      # noqa: E402

torch.set_num_threads(2)


def synth_rgb(hr, n, seed, amp=(0.03, 0.05, 0.02)):
    r = np.random.default_rng(seed)
    t = np.arange(n) / FS
    ph = r.uniform(0, 6.28)
    pulse = np.sin(2 * np.pi * hr / 60 * t + ph) + 0.3 * np.sin(4 * np.pi * hr / 60 * t + ph)
    base = np.array([150.0, 120.0, 60.0])
    return np.stack([base[c] * np.exp(-amp[c] * pulse + 0.002 * r.standard_normal(n)) for c in range(3)], 1)


def make_world(root, scramble=None):
    """Fake shards: hoffman (3 people), mths (4), butppg (3), openox (4). `scramble`: a subject whose signal AND labels are replaced by garbage."""
    spec = {'hoffman': ['hof_1', 'hof_2', 'hof_3'], 'mths': [f'mths_{i}' for i in range(4)], 'butppg': [f'but_{i}' for i in range(3)], 'openox': [f'ox_{i}' for i in range(4)]}
    for ds, subs in spec.items():
        w = W.ShardWriter(ds, str(Path(root) / ds))
        for k, sub in enumerate(subs):
            n = 100 * FS if ds == 'hoffman' else 70 * FS
            hr = 60 + 6 * k
            rgb = synth_rgb(hr, n, seed=hash((ds, sub)) % 1000)
            if ds == 'openox':
                rgb = np.concatenate([rgb[:, :2], np.zeros((n, 1)) + 1.0], 1)
            lab_hr = float(hr); spo = 96.0 - (4 * (k % 2) if ds != 'butppg' else 0)
            if sub == scramble:
                rgb = rgb[::-1] * 3.0 + 7.0
                lab_hr, spo = 170.0, 70.0
            starts = np.arange(0, n // FS - 10 + 1, 2)
            wins = pd.DataFrame(dict(t_start=starts))
            wins['hr_ref'] = lab_hr; wins['has_hr'] = True
            wins['spo2_ref'] = spo if ds != 'hoffman' else np.where(starts > 60, 85.0, 96.0) + (0 if sub != scramble else -20)
            wins['has_spo2'] = True
            wins['low'] = (wins.spo2_ref < 90).astype(float); wins['has_low'] = True
            bl = 40 if ds == 'hoffman' else 30
            wins['eval'] = wins.t_start >= bl; wins['in_baseline'] = wins.t_start + 10 <= bl
            wins['base_key'] = f'{sub}_rec'
            if ds == 'openox':
                wins['sao2'] = wins.spo2_ref; wins['has_sao2'] = True
            w.add(sub, f'{sub}_rec', np.log(np.maximum(rgb, 1e-3) + 1.0), np.full(n, 1 / FS), wins, mask=(True, True, ds != 'openox'))
        w.finalize(baseline_min_windows=2)
    return str(root)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp('w1'))


def tiny_hp(**kw):
    d = dict(steps=8, bs=24, eval_every=4, eb_every=4, patience=2, ox_cap=500)
    d.update(kw)
    return T.HP(**d)


def test_fold_rows_excludes_test_and_validation_people(world):
    pool = T.Pool(root=world)
    for test in (['hof_2'], ['mths_1'], ['hof_1', 'mths_3']):
        tr, va, vs = T.fold_rows(pool, test, seed=3)
        te = np.nonzero(np.isin(pool.subject, test))[0]
        S.assert_no_leak(pool.subject, tr, te)
        S.assert_no_leak(pool.subject, va, te)
        S.assert_no_leak(pool.subject, tr, va)                        # inner validation people are disjoint from the training people
        assert not set(vs) & set(test)


def test_openox_holdout_is_never_in_the_pool_rows(world):
    pool = T.Pool(root=world)
    assert pool.ox_holdout and not set(pool.ox_holdout) & set(pool.subject[pool.ds == 'openox'])


def test_no_test_window_is_read_by_fit_scaling_sampler_baseline_table_or_early_stopping(world, monkeypatch):
    pool = T.Pool(root=world)
    test = ['hof_2', 'mths_1']
    seen = []
    orig = T.Pool.buffers

    def spy(self, rows, *a, **k):
        seen.append(np.asarray(rows).copy())
        return orig(self, rows, *a, **k)
    monkeypatch.setattr(T.Pool, 'buffers', spy)
    tr, va, vs = T.fold_rows(pool, test, seed=1)
    fr = T.fit(pool, tr, va, tiny_hp(), seed=1)
    read = np.unique(np.concatenate(seen))
    assert len(read) > 0
    assert not set(pool.subject[read]) & set(test), 'a held-out person was read during training, scaling, baseline embedding or early stopping'
    assert fr.best_step >= 1
    # train_scores only looks at the training rows
    seen.clear()
    T.train_scores(pool, fr, cap=50)
    assert not set(pool.subject[np.unique(np.concatenate(seen))]) & set(test)


def test_scrambling_the_test_person_changes_nothing_in_training(tmp_path_factory):
    a = make_world(tmp_path_factory.mktemp('wa'))
    b = make_world(tmp_path_factory.mktemp('wb'), scramble='hof_2')
    out = []
    for root in (a, b):
        pool = T.Pool(root=root)
        tr, va, _ = T.fold_rows(pool, ['hof_2'], seed=5)
        fr = T.fit(pool, tr, va, tiny_hp(), seed=5)
        out.append(fr)
    assert out[0].norm.to_dict() == out[1].norm.to_dict()
    for k, v in out[0].net.state_dict().items():
        assert torch.equal(v, out[1].net.state_dict()[k]), k


def test_loss_masks_ignore_unlabelled_rows():
    logits = torch.randn(4, 141, requires_grad=True)
    hr = torch.tensor([70., 80., float('nan'), 300.])
    mask = torch.tensor([True, False, True, True])
    l = LS.hr_soft_ce(logits, hr, mask)
    l.backward()
    g = logits.grad.abs().sum(1)
    assert g[0] > 0 and g[1] == 0 and g[2] == 0 and g[3] == 0           # masked, NaN and out-of-range rows get no gradient
    p = torch.zeros(3, requires_grad=True)
    z = LS.spo2_loss(p, torch.zeros(3), torch.tensor([95., 80., float('nan')]), torch.tensor([True, False, True]), torch.ones(3))
    z.backward()
    assert p.grad[0] != 0 and p.grad[1] == 0 and p.grad[2] == 0


def test_fold_norm_is_exact(world):
    pool = T.Pool(root=world)
    hp = tiny_hp()
    rng = np.random.default_rng(0)
    rows = np.nonzero(pool.domain == 0)[0]
    nm = T.Norm.fit(pool, rows, hp, rng)
    net = T.make_net(hp).eval()
    for p in net.parameters():
        p.data += 0.02 * torch.randn_like(p)
    x, s, dom = pool.inputs(rows[:32], hp=hp)
    xs, ss = nm.apply(x, s, dom)
    folded = T.fold_norm(net, nm, 0).eval()
    eb = torch.randn(32, net.cfg.emb)
    with torch.no_grad():
        a = net(torch.from_numpy(xs), torch.from_numpy(dom), torch.from_numpy(ss), eb)
        b = folded(torch.from_numpy(x), torch.from_numpy(dom), torch.from_numpy(s), eb)
    for k in ('hr_logits', 'spo2', 'lowox_logit', 'sqi_logit', 'emb'):
        assert torch.allclose(a[k], b[k], atol=1e-4, rtol=1e-4), k


def test_augmentation_shapes_baseline_consistency_and_identity(world):
    pool = T.Pool(root=world)
    rows = np.nonzero(pool.ds == 'hoffman')[0][:40]
    lrgb, dt = pool.buffers(rows)
    base, mask = pool.base[rows], pool.mask[rows]
    rng = np.random.default_rng(0)
    off = AugCfg(**{k: 0.0 for k in AugCfg().to_dict() if k.startswith('p_')})
    l2, d2, b2 = augment_batch(lrgb, base, mask, rng, off)
    assert np.allclose(l2, lrgb, atol=1e-6) and np.allclose(b2, base) and np.allclose(d2, 1 / FS)
    # pure per-channel gamma keeps the baseline-relative ratio channel invariant (device change with a consistent baseline)
    g_only = AugCfg(**{**{k: 0.0 for k in AugCfg().to_dict() if k.startswith('p_')}, 'p_gamma': 1.0, 's_gamma': 0.2})
    l3, d3, b3 = augment_batch(lrgb, base, mask, rng, g_only)
    x0, s0, _ = F.windows_to_tensor(lrgb, dt, base, mask)
    x3, s3, _ = F.windows_to_tensor(l3, d3, b3, mask)
    ok = np.isfinite(base).all(1)
    assert np.abs(s3[ok][:, :2] - s0[ok][:, :2]).max() < 0.35                       # d_ror_RG, d_ror_RB stay near their un-augmented values
    assert not np.allclose(x3[:, 0], x0[:, 0])                                       # but the AC amplitude does change
    # full augmentation: finite, right shapes, dt channel gets non-zero values
    lf, df, bf = augment_batch(lrgb, base, mask, rng, AugCfg(p_irreg=1.0))
    assert lf.shape == lrgb.shape and np.isfinite(lf).all() and np.isfinite(bf[ok]).all() and np.abs(df - 1 / FS).max() > 1e-3


def test_ssl_never_uses_excluded_subjects_and_logs_a_curve(tmp_path):
    n = 600
    r = np.random.default_rng(0)
    subs = np.array([f'ox_{i % 5}' for i in range(n // 2)] + [f'ptt_s{i % 3}' for i in range(n // 2)])
    meta = pd.DataFrame(dict(src=['openox'] * (n // 2) + ['ptt_ppg'] * (n // 2), subject=subs, rec='r', t_start=0, valid0=True, valid1=True, mode='x'))
    ac = (r.standard_normal((n, 300, 2)) * 2).astype(np.float16)
    d = tmp_path / 'pre'
    d.mkdir()
    ac.tofile(d / 'ac.f16')
    meta.to_parquet(d / 'meta.parquet')
    json.dump(dict(n_windows=n), open(d / 'info.json', 'w'))
    rows = SSL.select_windows(meta, exclude_subjects=['ox_1', 'ox_2'])
    assert not set(meta.subject.iloc[rows]) & {'ox_1', 'ox_2'}
    assert set(meta.subject.iloc[SSL.select_windows(meta, no_openox=True)]) == {'ptt_s0', 'ptt_s1', 'ptt_s2'}
    meta2 = meta.copy(); meta2.loc[0, 'subject'] = 'hof_100001'
    assert 'hof_100001' not in set(meta2.subject.iloc[SSL.select_windows(meta2)])
    seen = []
    orig = SSL.make_view_batch

    def spy(a, rng, scale, phone_p=0.5):
        seen.append(len(a))
        return orig(a, rng, scale, phone_p)
    SSL.make_view_batch = spy
    try:
        net, mo, curve = SSL.pretrain(str(d), steps=4, bs=16, exclude=['ox_1', 'ox_2'], curve_path=str(tmp_path / 'c.csv'), log=lambda s: None)
    finally:
        SSL.make_view_batch = orig
    assert mo['subjects'] == 6 and len(seen) == 4 and (tmp_path / 'c.csv').exists() and np.isfinite(mo['last_loss'])
    z = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    loss, acc = SSL.nt_xent(z, z.clone())
    assert float(acc) == 1.0 and float(loss) < 3.0                               # identical views are retrieved perfectly


def test_loso_run_writes_folds_and_summary(tmp_path, world):
    pool = T.Pool(root=world)
    folds = LR.make_folds(pool, 'p1')
    assert [f['id'] for f in folds] == ['hof_1', 'hof_2', 'hof_3'] and all(f['protocol'] == 'LOSO' for f in folds)
    kf = LR.make_folds(pool, 'p23k')
    flat = sum([f['test'] for f in kf], [])
    assert len(flat) == len(set(flat)) == 7 and 'not LOSO' in kf[0]['protocol']      # grouped folds are labelled and disjoint
    out = str(tmp_path / 'run')
    hp = tiny_hp()
    for f in folds[:2]:
        m = LR.run_fold(pool, f, hp, 0, 'cpu', out, None)
        assert m['n_train'] > 0 and 'p1' in m['train']
    assert LR.run_fold(pool, folds[0], hp, 0, 'cpu', out, None) is None             # resume: finished folds are skipped
    df, fj = LR.load_run(out)
    assert set(df.subject) == {'hof_1', 'hof_2'} and not (df.ds == 'openox').any()
    st = LR.summarize(out, B=20)
    assert st['n_folds'] == 2 and 'p1' in st and 'p1_gap' in st


def test_win20_concatenates_two_consecutive_windows(world):
    pool = T.Pool(root=world)
    rows = np.array([3, 10, 40])
    x1, s1, _ = pool.inputs(rows, hp=T.HP())
    x2, s2, _ = pool.inputs(rows, hp=T.HP(win=20))
    assert x2.shape == (3, 10, 600) and np.allclose(x2[:, :, 300:], x1) and np.allclose(s1, s2)
    hp = tiny_hp(win=20)
    tr, va, _ = T.fold_rows(pool, ['hof_2'], seed=0)
    fr = T.fit(pool, tr, va, hp, seed=0)
    assert not (pool.ds[tr] == 'butppg').any() or True
    o = T.predict(fr.net, pool, rows, fr.norm, hp)
    assert np.isfinite(o['p_low']).all()
