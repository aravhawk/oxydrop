"""Tests of the conformal 3-state output (FEAT-007): the method on synthetic scores, the nested protocol (no leakage), and the committed results recomputed from the committed score files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import conformal as C, export_conformal as EC   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / 'results' / 'ml'


# ------------------------------------------------------------------------------------------------ the method
def test_finite_sample_quantile_rank():
    v = np.arange(1.0, 10.0)                                   # n = 9
    assert C.q_high(v, 0.1) == 9.0                             # ceil(10 * 0.9) = 9 -> the maximum
    assert C.q_high(v, 0.5) == 5.0                             # ceil(10 * 0.5) = 5
    assert np.isinf(C.q_high(np.arange(4.0), 0.1))             # ceil(5 * 0.9) = 5 > 4: no finite quantile
    assert np.isinf(C.q_high([], 0.1))


def synth(n, rng, shift=1.5):
    y = (rng.random(n) < 0.5).astype(int)
    z = rng.normal(0, 1.5, n) + shift * (2 * y - 1) * 1.5
    return z, y


@pytest.mark.parametrize('method', ['mondrian', 'marginal'])
def test_empirical_coverage_meets_target_on_exchangeable_data(method):
    rng = np.random.default_rng(0)
    zc, yc = synth(2000, rng)
    zt, yt = synth(60000, rng)
    for a in (0.05, 0.1, 0.2):
        t0, t1 = C.thresholds(zc, yc, a, method)
        s, c0, c1 = C.states(zt, t0, t1)
        cov = np.where(yt == 1, c1, c0)
        if method == 'mondrian':
            assert cov[yt == 1].mean() >= 1 - a - 0.02 and cov[yt == 0].mean() >= 1 - a - 0.02
        else:
            assert cov.mean() >= 1 - a - 0.02


def test_state_map_and_the_two_thresholds():
    # contains 0 iff z <= t0, contains 1 iff z >= t1
    z = np.array([-5., -1., 0., 1., 5.])
    s, c0, c1 = C.states(z, t0=0.5, t1=-0.5)                  # t1 < t0: middle band holds both classes -> unsure
    assert s.tolist() == [0, 0, 2, 1, 1]
    s2, _, _ = C.states(z, t0=-0.5, t1=0.5)                   # t0 < t1: middle band holds empty sets -> unsure
    assert s2.tolist() == [0, 0, 2, 1, 1]
    s3, _, _ = C.states(z, t0=np.inf, t1=-np.inf)             # everything contains both: always unsure
    assert (s3 == 2).all()


def test_marginal_thresholds_are_symmetric_in_logit():
    rng = np.random.default_rng(1)
    z, y = synth(1500, rng)
    t0, t1 = C.thresholds(z, y, 0.1, 'marginal')
    assert t1 == pytest.approx(-t0)


def test_band_matches_probability_band():
    t0, t1 = C.band_thresholds(0.1)
    p = expit(np.array([-3., -0.2, 0.0, 0.3, 2.0]))
    s, _, _ = C.states(np.array([-3., -0.2, 0.0, 0.3, 2.0]), t0, t1)
    assert s.tolist() == [0, 2, 2, 2, 1] or s.tolist() == [0, 2, 2, 1, 1]
    assert [int(abs(x - 0.5) >= 0.1) for x in p] == [int(v != 2) for v in s]


def test_counts_are_consistent():
    rng = np.random.default_rng(2)
    z, y = synth(900, rng)
    t0, t1 = C.thresholds(z, y, 0.1, 'mondrian')
    c = C.count_vector(z, y, t0, t1)
    n0, n1, N0, N1, L0, L1, U0, U1, cov0, cov1 = c
    assert n0 + n1 == 900 and N0 + L0 + U0 == n0 and N1 + L1 + U1 == n1
    m = C.metrics_from_counts(c)
    assert m['coverage'] == pytest.approx((N0 + N1 + L0 + L1) / 900)
    assert m['share_normal'] + m['share_low'] + m['share_unsure'] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ nested protocol
def make_world(rng, people=('a', 'b', 'c')):
    S, Cal = [], []
    for o in people:
        z, y = synth(300, rng)
        S.append(pd.DataFrame(dict(outer=o, subject=o, y=y, z=z)))
        for q in people:
            if q != o:
                zq, yq = synth(300, rng)
                Cal.append(pd.DataFrame(dict(outer=o, inner=q, y=yq, z=zq)))
    return pd.concat(S, ignore_index=True), pd.concat(Cal, ignore_index=True)


def test_thresholds_of_a_person_depend_only_on_its_own_calibration_rows():
    rng = np.random.default_rng(3)
    S, Cal = make_world(rng)
    _, t_ref = C.person_counts(S, Cal, 'mondrian', 0.1)
    S2 = S.copy(); S2.loc[S2.outer == 'a', 'y'] = 1 - S2.loc[S2.outer == 'a', 'y']; S2.loc[S2.outer == 'a', 'z'] += 10      # the evaluated person's own windows change
    _, t_a = C.person_counts(S2, Cal, 'mondrian', 0.1)
    assert np.array_equal(t_ref.t0.values, t_a.t0.values) and np.array_equal(t_ref.t1.values, t_a.t1.values)
    Cal3 = Cal.copy(); Cal3.loc[Cal3.outer == 'b', 'z'] += 3.0                                                                   # another person's calibration set changes
    _, t_b = C.person_counts(S, Cal3, 'mondrian', 0.1)
    assert t_b.set_index('outer').loc['a'].t0 == t_ref.set_index('outer').loc['a'].t0
    assert t_b.set_index('outer').loc['b'].t0 != t_ref.set_index('outer').loc['b'].t0


def test_calibration_set_never_contains_the_evaluated_person():
    d = pd.read_csv(RES / 'scores' / 'conformal_inputs_hoffman.csv')
    inner = d[d.kind == 'inner']
    assert (inner.outer != inner.inner).all()
    assert set(d.system.unique()) == set(C.SYSTEMS)
    for s in C.SYSTEMS:
        i = inner[inner.system == s]
        assert i.groupby('outer').inner.nunique().eq(5).all()


# ------------------------------------------------------------------------------------------------ committed results recomputed from the committed scores
def test_final_outer_scores_equal_the_committed_hybrid_scores():
    d = pd.read_csv(RES / 'scores' / 'conformal_inputs_hoffman.csv')
    o = d[(d.system == 'final') & (d.kind == 'outer')].set_index(['outer', 'local']).z
    f = pd.read_csv(RES / 'scores' / 'hybrid_final_hoffman.csv').set_index(['outer', 'local']).logit
    assert len(o) == len(f) == 5644
    assert np.allclose(o.sort_index().values, f.reindex(o.sort_index().index).values, atol=1e-6)


def test_references_reproduce_the_v1_and_v2_numbers():
    ref = pd.read_csv(RES / 'conformal_reference_bands.csv').set_index('system')
    assert ref.loc['v1_shipped_band', 'acc'] == pytest.approx(0.939, abs=5e-4) and ref.loc['v1_shipped_band', 'coverage'] == pytest.approx(0.844, abs=5e-4)
    assert ref.loc['v2_shipped_band', 'acc'] == pytest.approx(0.947, abs=5e-4) and ref.loc['v2_shipped_band', 'coverage'] == pytest.approx(0.680, abs=5e-4)


def test_curve_point_values_recompute_from_the_inputs():
    cu = pd.read_csv(RES / 'conformal_curve.csv')
    for system in C.SYSTEMS:
        S, Cal = C.load_scores(str(RES), system)
        for method in C.METHODS:
            for a in (0.05, 0.10, 0.20):
                cnt, _ = C.person_counts(S, Cal, method, a)
                m = C.metrics_from_counts(cnt.sum(0))
                r = cu[(cu.system == system) & (cu.method == method) & np.isclose(cu.param, a)].iloc[0]
                assert r.coverage == pytest.approx(float(m['coverage'])) and r.acc == pytest.approx(float(m['acc'])) and r.alert_share_all == pytest.approx(float(m['alert_share_all']))
        cnt, _ = C.person_counts(S, Cal, 'band', 0.10, kind='band')
        r = cu[(cu.system == system) & (cu.method == 'band') & np.isclose(cu.param, 0.10)].iloc[0]
        assert r.coverage == pytest.approx(float(C.metrics_from_counts(cnt.sum(0))['coverage']))


def test_chain_control_band_equals_the_shipped_v1_band():
    cu = pd.read_csv(RES / 'conformal_curve.csv')
    r = cu[(cu.system == 'chain_v1') & (cu.method == 'band') & np.isclose(cu.param, 0.10)].iloc[0]
    assert r.coverage == pytest.approx(0.8439, abs=1e-3) and r.acc == pytest.approx(0.9391, abs=1e-3)


def test_matched_coverage_rows_are_interpolations_of_the_curve():
    cu = pd.read_csv(RES / 'conformal_curve.csv')
    ma = pd.read_csv(RES / 'conformal_matched.csv')
    for _, r in ma[ma.variant.str.startswith('conformal_')].iterrows():
        method = r.variant.split('_')[1]
        c = cu[(cu.system == r.system) & (cu.method == method) & (cu.regime == 'main')]
        assert c.coverage.min() <= r.target_coverage <= c.coverage.max()
        assert r.acc == pytest.approx(C.acc_at_coverage(c.coverage.values, c.acc.values, r.target_coverage))


def test_shipped_json_is_rebuilt_from_the_committed_scores(tmp_path):
    out = tmp_path / 'conformal.json'
    EC.build(str(RES), str(out), str(ROOT / 'app' / 'data' / 'hybrid_weights.json'))
    assert json.loads(out.read_text()) == json.loads((ROOT / 'app' / 'data' / 'conformal.json').read_text())


def test_shipped_json_applies_like_the_python_states():
    doc = json.loads((ROOT / 'app' / 'data' / 'conformal.json').read_text())
    t = pd.read_csv(RES / 'scores' / 'conformal_deployed_oof_hoffman.csv')
    hw = json.loads((ROOT / 'app' / 'data' / 'hybrid_weights.json').read_text())['pipeline']
    assert doc['pipeline']['source'] == hw['source'] == str(t.source.iloc[0]) and doc['pipeline']['meta_intercept'] == hw['meta']['intercept']
    a = doc['alphas'][f'{doc["default_alpha"]:.2f}']
    s, _, _ = C.states(t.z.values, a['t0'], a['t1'])
    lo, hi = a['normal_below'], a['low_above']
    js = np.where(t.z.values < lo, 0, np.where(t.z.values > hi, 1, 2))
    assert (s == js).all()
    for ex in doc['examples']:
        for k, v in ex['state'].items():
            aa = doc['alphas'][k]
            expect = 'NORMAL' if ex['z'] < aa['normal_below'] else ('LOW' if ex['z'] > aa['low_above'] else 'UNSURE')
            assert v == expect
    assert not doc['alphas'][f'{doc["default_alpha"]:.2f}']['past_crossing']


def test_conformal_md_numbers_come_from_the_csv():
    md = (RES / 'CONFORMAL.md').read_text()
    cu = pd.read_csv(RES / 'conformal_curve.csv')
    r = cu[(cu.system == 'final') & (cu.method == 'mondrian') & np.isclose(cu.param, 0.10)].iloc[0]
    assert f'{100 * r.coverage:.1f}%' in md and f'{100 * r.acc:.1f}%' in md
    assert 'Generated by' in md
