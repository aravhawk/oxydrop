import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import splits as S  # noqa: E402


def _toy():
    # Hoffman: 3 people x 2 hands (same subject), MTHS 4 people, BUT 3 people
    groups, ds, rec = [], [], []
    for p in (100001, 100002, 100003):
        for hand in 'LR':
            groups += [f'hof_{p}'] * 5; ds += ['hoffman'] * 5; rec += [f'hof_{p}{hand}'] * 5
    for p in range(4):
        groups += [f'mths_{p}'] * 3; ds += ['mths'] * 3; rec += [f'mths_{p}'] * 3
    for p in range(3):
        groups += [f'but_{p}'] * 4; ds += ['butppg'] * 4; rec += [f'but_{p}_{k}' for k in range(4)]
    return np.array(groups), np.array(ds), np.array(rec)


def test_loso_holds_out_both_hands_together():
    g, ds, rec = _toy()
    folds = list(S.loso(g[ds == 'hoffman']))
    assert [f.test_group for f in folds] == ['hof_100001', 'hof_100002', 'hof_100003']
    r = rec[ds == 'hoffman']; gh = g[ds == 'hoffman']
    for f in folds:
        test_recs = set(r[f.test_idx]); train_recs = set(r[f.train_idx])
        assert len(test_recs) == 2 and not (test_recs & train_recs)          # both hands out, neither in train
        assert not (set(gh[f.train_idx]) & set(gh[f.test_idx]))
        assert len(f.train_idx) + len(f.test_idx) == len(gh)


def test_loso_eval_mask_only_changes_test_rows():
    g = np.repeat(['a', 'b', 'c'], 4); em = np.array([0, 0, 1, 1] * 3, bool)
    for f in S.loso(g, em):
        assert len(f.test_idx) == 2 and em[f.test_idx].all()
        assert len(f.train_idx) == 8                                          # training keeps every row of the other subjects


def test_pooled_loso_trains_on_everyone_else_across_datasets():
    g, ds, rec = _toy()
    folds = list(S.pooled_loso(g, ds, 'mths'))
    assert len(folds) == 4
    for f in folds:
        assert set(ds[f.test_idx]) == {'mths'} and not (set(g[f.train_idx]) & set(g[f.test_idx]))
        assert set(ds[f.train_idx]) == {'hoffman', 'mths', 'butppg'}          # other MTHS people, all Hoffman, all BUT
        assert (g[f.train_idx] != f.test_group).all()
        assert len(f.train_idx) + len(f.test_idx) == len(g) - (g == f.test_group).sum() + (g == f.test_group).sum()
    only = list(S.pooled_loso(g, ds, 'butppg', train_datasets=['hoffman']))
    assert all(set(ds[f.train_idx]) == {'hoffman'} for f in only)
    hof = list(S.pooled_loso(g, ds, 'hoffman'))
    assert len(hof) == 3
    for f in hof:                                                             # both hands of the held-out person leave training
        assert not any(x.startswith(f.test_group) for x in rec[f.train_idx])


def test_inner_folds_are_subject_grouped_and_cover_everyone_once():
    g, ds, rec = _toy()
    f0 = next(S.loso(g))
    tg = g[f0.train_idx]
    folds = list(S.inner_group_folds(tg, k=4, seed=3))
    assert len(folds) == 4
    seen = []
    for f in folds:
        assert not (set(tg[f.train_idx]) & set(tg[f.test_idx]))
        assert f0.test_group not in set(tg[f.train_idx]) | set(tg[f.test_idx])        # the outer test subject is nowhere in the inner CV
        seen += list(f.test_group)
    assert sorted(seen) == sorted(set(tg.tolist()))
    assert [f.test_group for f in folds] == [f.test_group for f in S.inner_group_folds(tg, k=4, seed=3)]   # deterministic
    assert [f.test_group for f in folds] != [f.test_group for f in S.inner_group_folds(tg, k=4, seed=4)]


def test_inner_folds_cap_k_and_balance():
    tg = np.repeat(np.arange(3), 5)
    assert len(list(S.inner_group_folds(tg, k=10))) == 3
    tg = np.repeat(np.arange(10), 4); pos = np.repeat([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 4).astype(float)
    for f in S.inner_group_folds(tg, k=5, balance=pos):
        vp = pos[f.test_idx].mean()
        assert 0 < vp < 1                                                         # each validation fold mixes positive and negative subjects


def test_leak_detection_raises():
    g = np.array(['a', 'a', 'b', 'b'])
    with pytest.raises(AssertionError):
        S.assert_no_leak(g, np.array([0, 2]), np.array([1, 3]))                  # subject a on both sides
    with pytest.raises(AssertionError):
        S.assert_no_leak(g, np.array([0, 1]), np.array([1, 2]))
    S.assert_no_leak(g, np.array([0, 1]), np.array([2, 3]))


def test_pretrain_rule_excludes_test_subjects():
    ok = S.pretrain_allowed(['ox_1', 'ox_2', 'ox_3'], ['ox_2'])
    assert ok.tolist() == [True, False, True]


HOF = Path('/home/aravhawk/.t3/scratch/veritas/ml_cache/data/hoffman/meta.parquet')


@pytest.mark.skipif(not HOF.exists(), reason='Hoffman shard not built')
def test_real_hoffman_shard_loso_has_six_folds_and_no_hand_leak():
    import pandas as pd
    m = pd.read_parquet(HOF)
    folds = list(S.loso(m.subject.values, m['eval'].values))
    assert len(folds) == 6
    assert sum(len(f.test_idx) for f in folds) == 5644
    for f in folds:
        assert not (set(m.rec.values[f.train_idx]) & set(m.rec.values[f.test_idx]))
        assert not (set(m.subject.values[f.train_idx]) & set(m.subject.values[f.test_idx]))
        assert m.rec.values[f.test_idx].tolist() and len(set(m.rec.values[f.test_idx])) == 2   # left and right hand together
