"""Tests of the OpenOximetry SaO2 test (FEAT-007): exact norm handling of folded checkpoints, labels at a lag, metrics, the aggregation on a synthetic world, and the committed aggregates."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import metrics as M, openox_spo2 as O, train as T   # noqa: E402
from oxydrop_ml.model import OxyCfg, OxyNet                           # noqa: E402

torch.set_num_threads(2)
RES = Path(__file__).resolve().parents[3] / 'results' / 'ml'


def random_norm(rng):
    xs = rng.uniform(0.3, 3.0, (2, 10)).astype(np.float32)
    for d in (0, 1):
        xs[d, 1] = xs[d, 2] = xs[d, 0]                                    # Norm.fit gives the three waveform slots one scale
    return T.Norm(xs, rng.uniform(0.3, 3.0, (2, 9)).astype(np.float32))


def test_folded_checkpoint_equals_the_unfolded_net_on_the_contact_domain():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = OxyNet(OxyCfg()); net.eval()
    for p in net.parameters():
        p.data += 0.05 * torch.randn_like(p)
    norm = random_norm(rng)
    folded = T.fold_norm(net, norm, 0)
    x = rng.standard_normal((6, 10, 300)).astype(np.float32); s = rng.standard_normal((6, 9)).astype(np.float32)
    dom = np.ones(6, np.int64)
    ref = O.aux_predict(net, O.input_transform(norm.to_dict(), None), x, s, dom)                 # unfolded net, x / scale[domain 1]
    got = O.aux_predict(folded, O.input_transform(norm.to_dict(), 0), x, s, dom)                 # folded for domain 0, rescaled to domain 1
    assert np.allclose(ref, got, atol=2e-4), np.abs(ref - got).max()
    wrong = O.aux_predict(folded, O.input_transform(norm.to_dict(), None), x, s, dom)            # without the rescale the result differs
    assert not np.allclose(ref, wrong, atol=1e-3)


def test_skin_groups_and_tertiles():
    assert [O.monk_group(v) for v in (1, 3, 3.4, 3.6, 6, 6.5, 7, 10, np.nan)] == ['1-3', '1-3', '1-3', '4-6', '4-6', '7+', '7+', '7+', 'unknown']
    cuts = O.ita_cuts(np.arange(30.0))
    assert O.ita_group(0, cuts) == 'darkest tertile' and O.ita_group(29, cuts) == 'lightest tertile' and O.ita_group(15, cuts) == 'mid tertile' and O.ita_group(np.nan, cuts) == 'unknown'


def test_error_metrics_known_values():
    y = np.array([95., 90., 85., 80.]); p = np.array([96., 88., 85., 83.])
    m = O.err_metrics(y, p)
    assert m['mae'] == pytest.approx((1 + 2 + 0 + 3) / 4) and m['bias'] == pytest.approx((1 - 2 + 0 + 3) / 4)
    assert m['arms'] == pytest.approx(np.sqrt((1 + 4 + 0 + 9) / 4)) and m['arms_70_100'] == m['arms']
    assert m['auroc_lt90'] == pytest.approx(M.auroc((y < 90).astype(int), -p))
    assert O.err_metrics([60., 95.], [60., 95.])['arms_70_100'] == 0.0 and O.err_metrics([np.nan], [1.])['n'] == 0


def test_baselines_are_the_declared_formulas():
    ror = np.array([-0.2, 0.0, 0.3])
    par = dict(train_mean=88.0, a=90.0, b=-10.0)
    b = O.baseline_predictions(ror, par)
    assert np.allclose(b['ror_textbook'], np.clip(110 - 25 * np.exp(ror), 70, 100)) and np.allclose(b['train_mean'], 88.0) and np.allclose(b['ror_calibrated'], 90 - 10 * ror)
    tr = pd.DataFrame(dict(subject=['a'] * 50 + ['b'] * 50, abs_ror_RG=np.linspace(-0.3, 0.4, 100), y=90 - 12 * np.linspace(-0.3, 0.4, 100)))
    p = O.fit_baselines(tr)
    assert p['b'] == pytest.approx(-12.0) and p['a'] == pytest.approx(90.0) and p['n_train_people'] == 2


def test_labels_at_a_lag_use_the_reference_after_the_window():
    sao = np.arange(200.0)
    refs = SimpleNamespace(by={'r': (None, None, sao)})
    meta = pd.DataFrame(dict(rec=['r'] * 3, t_start=[10, 20, 30]))
    y0 = O.labels_at(meta, refs, 0)
    y19 = O.labels_at(meta, refs, 19)
    assert np.allclose(y0, [14.5, 24.5, 34.5]) and np.allclose(y19 - y0, 19)
    assert np.isnan(O.labels_at(pd.DataFrame(dict(rec=['q'], t_start=[10])), refs, 0)[0])


# ------------------------------------------------------------------------------------------------ the aggregation on a synthetic world
def synth_world(tmp, rng, n_people=14):
    subj = [f'p{i}' for i in range(n_people)]
    monk = np.array([2, 2, 2, 5, 5, 5, 5, 8, 8, 8, 2, 5, 5, 8], float)[:n_people]
    ita = np.linspace(10, 50, n_people)
    people = pd.DataFrame(dict(subject=subj, monk=monk, ita=ita, held_out=True))
    rows = []
    for i, s in enumerate(subj):
        n = 200
        y = rng.uniform(75, 100, n)
        rows.append(pd.DataFrame(dict(subject=s, rec=f'r{i}', t_start=np.arange(n) * 2.0, local=np.arange(n), spo2_ref=y + 1, sao2=y, sao2_draw=np.where(np.arange(n) % 50 == 0, y, np.nan),
                                      has_sao2=True, has_sao2_draw=(np.arange(n) % 50 == 0), monk=monk[i], ita=ita[i], skin_group='x', abs_ror_RG=-(y - 90) / 40,
                                      y_lagstar=y, y_lag0=y, pred_R=y + rng.normal(0, 1, n), pred_P=y + rng.normal(0, 3, n), pred_stageB=y + rng.normal(0, 2, n),
                                      pred_train_mean=np.full(n, 90.0), pred_ror_textbook=y + rng.normal(0, 4, n), pred_ror_calibrated=y + rng.normal(0, 5, n))))
    df = pd.concat(rows, ignore_index=True)
    for L in (-10, 0, 10):
        df[f'y_lag{L}'] = df.sao2
    df.to_parquet(tmp / 'preds.parquet')
    prof = {str(float(x)): 0.0 for x in range(-10, 61)}
    json.dump(dict(ppg=dict(lag=19.0, n_enc=10, confident=True, ci=[15.0, 27.0], profile=prof), abg=dict(lag=18.0, n_enc=10, confident=True, ci=[14.0, 22.0], profile=prof), lag_star=19.0, holdout=subj),
              open(tmp / 'lag.json', 'w'))
    json.dump(dict(train_mean=88.0, a=90.0, b=-10.0, n_train=1000, n_train_people=60), open(tmp / 'baseline_params.json', 'w'))
    people.to_parquet(tmp / 'people.parquet')
    df[['subject', 'rec', 't_start', 'abs_ror_RG', 'monk', 'ita']].assign(y=df.sao2).to_parquet(tmp / 'all_people.parquet')
    dr = pd.DataFrame(dict(sample=np.arange(n_people * 4), t0=0.0, t1=1.0, sao2=rng.uniform(70, 100, n_people * 4), spo2_dev=rng.uniform(70, 100, n_people * 4), rec=[f'r{i // 4}' for i in range(n_people * 4)],
                           subject=[subj[i // 4] for i in range(n_people * 4)], monk=np.repeat(monk, 4), ita=np.repeat(ita, 4), aligned=True))
    dr.to_parquet(tmp / 'draws_all.parquet')


def test_analyze_on_a_synthetic_world(tmp_path, monkeypatch):
    monkeypatch.setattr(O, 'MIN_GROUP', 3)   # the synthetic world has 14 people; the production minimum is 5 (see test_small_groups_are_suppressed)
    rng = np.random.default_rng(0)
    synth_world(tmp_path, rng)
    out = tmp_path / 'agg'
    O.analyze(str(tmp_path), str(out), B=40)
    ov = pd.read_csv(out / 'openox_spo2_overall.csv')
    r = ov[(ov.label == 'lag_star') & (ov.model == 'R')].iloc[0]
    assert r.mae == pytest.approx(0.8, abs=0.1) and r.n_people == 14 and r.mae_lo <= r.mae <= r.mae_hi
    pr = pd.read_csv(out / 'openox_spo2_paired.csv')
    a = pr[(pr.label == 'lag_star') & (pr.model == 'R') & (pr.baseline == 'ror_textbook')].iloc[0]
    assert a.diff_model_minus_baseline == pytest.approx(a.model_mae - a.baseline_mae) and a.verdict == 'model better'
    gp = pd.read_csv(out / 'openox_spo2_gaps.csv')
    assert gp[(gp.scheme == 'ita') & (gp.model == 'R')].estimable.all()
    assert gp[(gp.scheme == 'monk')].gap.notna().all()                                           # 3+ people in Monk 7+ in this world
    assert pd.read_csv(out / 'openox_spo2_counts.csv').iloc[0].lag_star == 19.0
    cb = pd.read_csv(out / 'openox_spo2_contact_bias.csv')
    assert set(cb.scheme) == {'monk', 'ita'}


def test_a_group_with_fewer_than_three_people_has_no_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(O, 'MIN_GROUP', 3)
    rng = np.random.default_rng(1)
    synth_world(tmp_path, rng)
    p = pd.read_parquet(tmp_path / 'people.parquet'); p.loc[p.monk == 8, 'monk'] = np.where(np.arange((p.monk == 8).sum()) < 2, 8, 5); p.to_parquet(tmp_path / 'people.parquet')
    O.analyze(str(tmp_path), str(tmp_path / 'agg'), B=20)
    gp = pd.read_csv(tmp_path / 'agg' / 'openox_spo2_gaps.csv')
    assert (~gp[gp.scheme == 'monk'].estimable).all() and gp[gp.scheme == 'monk'].gap.isna().all()


# ------------------------------------------------------------------------------------------------ committed aggregates
def test_committed_aggregates_are_internally_consistent():
    ov = pd.read_csv(RES / 'openox_spo2_overall.csv'); pr = pd.read_csv(RES / 'openox_spo2_paired.csv'); ct = pd.read_csv(RES / 'openox_spo2_counts.csv').iloc[0]
    lg = pd.read_csv(RES / 'openox_spo2_lag.csv'); cb = pd.read_csv(RES / 'openox_spo2_contact_bias.csv'); cg = pd.read_csv(RES / 'openox_spo2_contact_bias_gaps.csv')
    for _, r in pr.iterrows():
        a = ov[(ov.label == r.label) & (ov.model == r.model)].iloc[0].mae
        b = ov[(ov.label == r.label) & (ov.model == r.baseline)].iloc[0].mae
        assert r.model_mae == pytest.approx(a) and r.baseline_mae == pytest.approx(b) and r.diff_model_minus_baseline == pytest.approx(a - b)
        assert r.verdict == ('model better' if r.hi < 0 else 'model worse' if r.lo > 0 else 'tie')
    assert int(ct.heldout_people) == 34 and int(ct.draws_in_aligned_recordings) == 3818 and int(ct.draw_people_aligned) == 93 and int(ct.draw_encounters_aligned) == 164
    est = lg[(lg.kind == 'estimate') & (lg.method == 'ppg')].iloc[0]
    assert est.ci_lo <= est.lag <= est.ci_hi and ct.lag_star == est.lag
    for scheme in ('monk', 'ita'):
        g = cg[cg.scheme == scheme].iloc[0]
        d = cb[(cb.scheme == scheme)].set_index('group').bias
        assert g.gap == pytest.approx(d[g.dark] - d[g.light]) and g.lo <= g.gap <= g.hi
    # the primary lag-star label uses the lag estimated on training people only
    assert ov[ov.label == 'lag_star'].n_people.iloc[0] <= int(ct.heldout_people_with_sao2)


def test_held_out_people_are_not_in_the_ssl_corpus():
    man = json.loads((RES / 'ssl_manifest.json').read_text())
    ct = pd.read_csv(RES / 'openox_spo2_counts.csv').iloc[0]
    assert man['ssl_full']['n_excluded_subjects'] == man['ssl_noox']['n_excluded_subjects'] == int(ct.heldout_people) == 34


def test_openox_md_numbers_come_from_the_csv():
    md = (RES / 'OPENOX_SPO2.md').read_text()
    ov = pd.read_csv(RES / 'openox_spo2_overall.csv')
    r = ov[(ov.label == 'lag_star') & (ov.model == 'R')].iloc[0]
    assert f'{r.mae:.2f} [{r.mae_lo:.2f}, {r.mae_hi:.2f}]' in md and 'contact red/infrared finger-clip sensor test' in md


def test_small_groups_are_suppressed():
    """Minimum group size: a per-group cell with fewer than MIN_GROUP (5) people keeps only its people count; the metrics are blank and `suppressed` is True."""
    assert O.MIN_GROUP == 5
    rows = O.suppress_small([dict(label='l', scheme='monk', group='7+', model='R', n_people=2, n_windows=300, mae=3.1, mae_lo=1.0, mae_hi=5.0),
                             dict(label='l', scheme='monk', group='1-3', model='R', n_people=8, n_windows=900, mae=2.9, mae_lo=2.0, mae_hi=4.0)])
    a, b = rows
    assert a['suppressed'] and a['n_people'] == 2 and all(np.isnan(a[k]) for k in ('n_windows', 'mae', 'mae_lo', 'mae_hi')) and a['group'] == '7+'
    assert not b['suppressed'] and b['mae'] == 2.9 and b['n_windows'] == 900


def test_committed_group_cells_respect_the_minimum_group_size():
    for name in ('skin', 'contact_bias', 'ror_all_people'):
        d = pd.read_csv(RES / f'openox_spo2_{name}.csv')
        small = d.n_people < O.MIN_GROUP
        assert (d.suppressed == small).all(), name
        assert d[small].drop(columns=[c for c in ('label', 'scheme', 'group', 'model', 'n_people', 'suppressed') if c in d.columns]).isna().all().all(), f'{name}: a group under {O.MIN_GROUP} people has values'
    g = pd.read_csv(RES / 'openox_spo2_gaps.csv')
    assert (g.estimable == ((g.n_dark >= O.MIN_GROUP) & (g.n_light >= O.MIN_GROUP))).all()
    hv = pd.read_csv(RES / 'hr_vote_slow_pulse.csv')
    assert ((hv.dataset == 'hoffman') | (hv.n_people >= O.MIN_GROUP)).all() and 'mths' not in set(hv.dataset), 'a restricted-licence dataset must not appear as a one-person row'
