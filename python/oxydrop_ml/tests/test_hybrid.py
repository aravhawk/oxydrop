"""Tests of the FEAT-008 hybrid: chain prior without leakage, residual / distilled losses, per-user calibration and aggregation, inner-fold-only selection,
and the job runner on a tiny FAKE world (the same fake data as test_train.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oxydrop_ml import distill as DS, hybrid as H, hybrid_run as HR, tta, train as T   # noqa: E402
from test_train import make_world, tiny_hp                                                # noqa: E402

torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ tta
def test_causal_mean_uses_only_past_windows_of_the_same_recording():
    f = np.array([1., 2., 3., 4., 10., 20., 30.])
    rec = np.array(['a'] * 4 + ['b'] * 3)
    t = np.array([0., 2., 4., 6., 0., 2., 4.])
    out = tta.causal_mean(f, rec, t, 3)
    assert np.allclose(out, [1, 1.5, 2, 3, 10, 15, 20])
    assert np.allclose(tta.causal_mean(f, rec, t, 1), f)
    f2 = f.copy(); f2[3] = 1000.0                                   # a later window changes nothing earlier
    assert np.allclose(tta.causal_mean(f2, rec, t, 3)[:3], out[:3])
    perm = np.random.default_rng(0).permutation(len(f))             # row order does not matter, only t_start
    assert np.allclose(tta.causal_mean(f[perm], rec[perm], t[perm], 3), out[perm])


def test_baseline_median_is_per_recording_and_calibration_removes_it():
    f = np.array([5., 7., 9., 100., 1., 3.])
    rec = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
    ib = np.array([True, True, False, False, True, False])
    bm = tta.baseline_median(f, rec, np.arange(6.), ib)
    assert np.allclose(bm, [6, 6, 6, 6, 1, 1])
    out = tta.calibrate_and_smooth(f, rec, np.arange(6.), ib, True, 1)
    assert np.allclose(out, f - bm)


# ------------------------------------------------------------------------------------------------ chain prior
def fake_chain_data(n_per=60, people=('a', 'b', 'c', 'd'), seed=0):
    r = np.random.default_rng(seed)
    subj = np.repeat(people, n_per)
    y = (r.random(len(subj)) < 0.5).astype(int)
    X = r.standard_normal((len(subj), 19)) + y[:, None] * 0.6
    return X, y, subj, np.ones(len(subj), bool)


def test_prior_of_a_person_never_depends_on_that_persons_labels():
    X, y, subj, ev = fake_chain_data()
    p0 = DS.chain_prior(X, y, subj, ev, ['a', 'b', 'c'], ['d'])
    y2 = y.copy(); y2[subj == 'd'] = 1 - y2[subj == 'd']                 # held-out person's labels flipped
    assert np.array_equal(p0, DS.chain_prior(X, y2, subj, ev, ['a', 'b', 'c'], ['d']))
    y3 = y.copy(); y3[subj == 'b'] = 1 - y3[subj == 'b']                 # a TRAINING person's labels flipped: only its OWN prior must stay the same
    p3 = DS.chain_prior(X, y3, subj, ev, ['a', 'b', 'c'], ['d'])
    assert np.array_equal(p0[subj == 'b'], p3[subj == 'b'])
    assert not np.array_equal(p0[subj == 'a'], p3[subj == 'a'])
    insample = DS.Chain(X[np.isin(subj, ['a', 'b', 'c'])], y[np.isin(subj, ['a', 'b', 'c'])]).logit(X[subj == 'a'])
    assert not np.allclose(insample, p0[subj == 'a'])                    # the prior of a training person is out of person, not the in-sample fit


# ------------------------------------------------------------------------------------------------ losses
@pytest.fixture(scope='module')
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp('hw'))


def fake_x19(pool, seed=0):
    ws = pool.ws['hoffman']
    n = int(max(pool.local[pool.ds == 'hoffman'])) + 1
    return np.random.default_rng(seed).standard_normal((n, 19))


def test_install_restores_the_original_training_functions(world):
    pool = T.Pool(root=world)
    orig_l, orig_p = T.batch_loss, T.predict
    with DS.install('R', np.zeros(pool.N, np.float32)):
        assert T.batch_loss is not orig_l and T.predict is not orig_p
    assert T.batch_loss is orig_l and T.predict is orig_p
    with pytest.raises(AssertionError):
        with DS.install('R', None):
            pass


def test_residual_loss_adds_the_prior_and_distillation_adds_a_term(world):
    pool = T.Pool(root=world)
    hp = tiny_hp()
    rows = np.nonzero((pool.ds == 'hoffman') & pool.eval)[0][:24]
    x, s, dom = pool.inputs(rows, hp=hp)
    net = T.make_net(hp)
    xt, st, dt = torch.from_numpy(x), torch.from_numpy(s), torch.from_numpy(dom)
    eb = torch.zeros(len(rows), net.cfg.emb)
    big = np.where(pool.low > 0.5, 30.0, -30.0).astype(np.float32)           # a perfect prior
    with DS.install('R', big):
        lR, LR_ = T.batch_loss(net, pool, rows, xt, st, dt, eb, hp, 1.0)
    with DS.install('R', np.zeros(pool.N, np.float32)):
        l0, L0 = T.batch_loss(net, pool, rows, xt, st, dt, eb, hp, 1.0)
    assert float(LR_['low'].detach()) < float(L0['low'].detach()) and float(LR_['low'].detach()) < 1e-3     # a perfect prior leaves nothing for the net to learn
    with DS.install('D', big):
        lD, LD = T.batch_loss(net, pool, rows, xt, st, dt, eb, hp, 1.0)
    assert 'distill' in LD and float(lD.detach()) > 0
    with DS.install('P'):
        lP, LP = T.batch_loss(net, pool, rows, xt, st, dt, eb, hp, 1.0)
    assert 'distill' not in LP


def test_residual_fit_is_byte_identical_when_the_held_out_person_is_scrambled(tmp_path_factory):
    a = make_world(tmp_path_factory.mktemp('ra'))
    b = make_world(tmp_path_factory.mktemp('rb'), scramble='hof_2')
    ws = []
    for root in (a, b):
        pool = T.Pool(root=root)
        X19 = fake_x19(pool)
        hof = pool.subjects_of('hoffman')
        excl = ['hof_2']
        prior = DS.prior_for_pool(pool, X19, [p for p in hof if p not in excl], excl)
        tr, va, _ = T.fold_rows(pool, excl, 5)
        with DS.install('R', prior):
            fr = T.fit(pool, tr, va, tiny_hp(), seed=5)
        ws.append(fr.net.state_dict())
    for k, v in ws[0].items():
        assert torch.equal(v, ws[1][k]), k


# ------------------------------------------------------------------------------------------------ the job runner
def test_inner_and_outer_jobs_write_only_the_allowed_rows(tmp_path, world):
    pool = T.Pool(root=world)
    X19 = fake_x19(pool)
    hp = tiny_hp()
    for fam in ('P', 'R', 'D'):
        assert HR.run_job(pool, X19, ('inner', 'hof_1', 'hof_2'), fam, hp, 0, 'cpu', str(tmp_path), log=lambda s: None)
        d = pd.read_csv(tmp_path / f'{fam}_s0' / 'inner_hof_1_hof_2.csv')
        assert set(d[d.target == 'inner'].subject) == {'hof_2'} and set(d[d.target == 'member'].subject) == {'hof_1'}
        assert 'hof_3' not in set(d.subject) and d.logit.notna().all()
        assert not HR.run_job(pool, X19, ('inner', 'hof_1', 'hof_2'), fam, hp, 0, 'cpu', str(tmp_path), log=lambda s: None)    # finished jobs are skipped
    assert HR.run_job(pool, X19, ('outer', 'hof_3', None), 'R', hp, 0, 'cpu', str(tmp_path), log=lambda s: None)
    d = pd.read_csv(tmp_path / 'R_s0' / 'outer_hof_3.csv')
    assert set(d[d.target == 'outer'].subject) == {'hof_3'} and set(d[d.target == 'train'].subject) == {'hof_1', 'hof_2'}


# ------------------------------------------------------------------------------------------------ selection
def synth_tables(seed=0, flip_outer=False):
    """Inner OOF table of 5 people and an outer table with the columns the pipeline reads (c, p, d, corr, r)."""
    r = np.random.default_rng(seed)

    def person(name, n=160):
        t = np.arange(n) * 2.0 + 60
        y = (r.random(n) < 0.5).astype(float)                           # both classes in every 100 s bin
        sig = (y - 0.5) * 2
        d = dict(subject=name, rec=name + 'L', t_start=t, low=y, eval=True, in_base=False, hr_ref=60.0)
        d['c'] = 0.8 * sig + r.standard_normal(n)
        d['p'] = 0.5 * sig + 1.2 * r.standard_normal(n)
        d['d'] = 0.5 * sig + 1.2 * r.standard_normal(n)
        d['corr'] = 0.2 * sig + r.standard_normal(n)
        d['r'] = d['c'] + d['corr']
        return pd.DataFrame(d)
    inner = pd.concat([person(f'q{i}').assign(inner=f'q{i}') for i in range(5)], ignore_index=True)
    outer = person('o').assign(inner='')
    if flip_outer:
        outer['low'] = 1 - outer['low']
    return inner, outer


def test_selection_depends_only_on_the_inner_table_and_stays_in_the_cnn_set():
    inner, outer = synth_tables()
    ch = H.select(inner, H.CNN_ORDER)
    assert ch['source'] in H.CNN_ORDER and ch['k'] in H.KS
    inner2, outer2 = synth_tables(flip_outer=True)                     # the outer person's labels are flipped
    ch2 = H.select(inner2, H.CNN_ORDER)
    assert (ch['source'], ch['cal'], ch['k']) == (ch2['source'], ch2['cal'], ch2['k'])
    s1, _ = H.apply_pipeline(inner, outer, ch['source'], ch['cal'], ch['k'])
    s2, _ = H.apply_pipeline(inner2, outer2, ch['source'], ch['cal'], ch['k'])
    assert np.array_equal(s1, s2)                                      # scores of the outer person never see its labels
    chc = H.select(inner, H.CHAIN_ORDER)
    assert chc['source'] == 'chain_platt'


def test_tie_rule_takes_the_earliest_candidate_within_the_margin():
    inner, _ = synth_tables()
    inner['p'] = inner['c'].values                                      # P is an exact copy of the chain, so the stacks cannot beat it
    inner['d'] = inner['c'].values
    ch = H.select(inner, ['P', 'D', 'c+p'])
    assert ch['source'] == 'P'


def test_meta_scores_are_invariant_to_row_order_of_the_inner_table():
    inner, outer = synth_tables()
    sc, _ = H.apply_pipeline(inner, outer, 'c+p', False, 1)
    perm = np.random.default_rng(1).permutation(len(inner))
    sc2, _ = H.apply_pipeline(inner.iloc[perm].reset_index(drop=True), outer, 'c+p', False, 1)
    assert np.allclose(sc, sc2, atol=1e-6)


# ------------------------------------------------------------------------------------------------ P2 rules and export
def test_hr_rules():
    from oxydrop_ml import hybrid_hr as HH
    cnn = np.array([140., 71., 90., 60.]); spec = np.array([70., 70., 60., 80.]); valid = np.array([True, False, True, False])
    assert np.allclose(HH.apply_rule('cnn', cnn, spec), cnn)
    assert np.allclose(HH.apply_rule('spec', cnn, spec), spec)
    assert np.allclose(HH.apply_rule('gate', cnn, spec, valid), [70., 71., 60., 60.])
    assert np.allclose(HH.apply_rule('guard', cnn, spec), [70., 71., 90., 60.])          # 140 = 2 x 70 is a harmonic; 90 is not within 8 of 120 or 30
    assert np.allclose(HH.apply_rule('agree', cnn, spec), [140., 70.5, 90., 60.])
    assert np.allclose(HH.apply_rule('gate', cnn, np.array([np.nan] * 4), np.ones(4, bool)), cnn)     # no spectral answer: the CNN answers


def test_headline_hr_rule_always_contains_the_cnn():
    from oxydrop_ml import hybrid_hr as HH
    r = np.random.default_rng(0)
    n = 400
    ref = r.uniform(55, 120, n)
    runs = pd.DataFrame(dict(family='P', target='inner', outer='o', inner='q', local=np.arange(n), hr=ref + 30 * r.standard_normal(n), hr_ref=ref, subject='q'))
    spec = ref + 0.5 * r.standard_normal(n)                              # the spectral estimator is nearly perfect, so `spec` alone would win
    t = HH.hoffman_rule_table(runs, spec, np.ones(n, bool))
    assert t.loc[t.rule == 'spec', 'within5_full'].iloc[0] == t.within5_full.max()
    assert t.rule[t.selected].iloc[0] in HH.CNN_RULES and t.selected.sum() == 1


def test_deploy_pipeline_is_chosen_by_the_same_rule_on_the_stacked_outer_tables():
    from oxydrop_ml import export_hybrid as EH
    inner, outer = synth_tables()
    rows = []
    for i in range(4):
        _, o = synth_tables(seed=10 + i)
        rows.append(o.assign(subject=f'o{i}', rec=f'o{i}L', target='outer'))
    w = pd.concat(rows, ignore_index=True)
    ch, cols, f, oof = EH.deploy_pipeline(w, order=['P', 'c+p'])
    assert ch['source'] in ('P', 'c+p') and cols == H.CANDS[ch['source']]
    assert len(f.params['coef']) == len(cols) and set(oof.inner) == set(w.subject)
