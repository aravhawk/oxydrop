import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import metrics as M  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def test_auroc_basic():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(M.auroc([1, 1], [0.1, 0.2]))
    # weights: double the weight of one positive changes the value as the formula says
    y = np.array([1, 1, 0]); s = np.array([0.9, 0.1, 0.5])
    assert abs(M.auroc(y, s) - 0.5) < 1e-12
    assert abs(M.auroc(y, s, np.array([3., 1., 1.])) - 0.75) < 1e-12


def test_classification_metrics():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0]); p = np.array([.9, .8, .2, .1, .1, .6, .2, .3])
    c = M.classification_metrics(y, p)
    assert c['acc'] == 6 / 8 and abs(c['sens'] - 2 / 3) < 1e-12 and abs(c['spec'] - 4 / 5) < 1e-12
    assert abs(c['bacc'] - (2 / 3 + 4 / 5) / 2) < 1e-12
    assert 0 <= c['ece'] <= 1


def test_ece_perfectly_calibrated_and_not():
    rng = np.random.default_rng(0)
    p = rng.random(20000); y = (rng.random(20000) < p).astype(int)
    assert M.ece(y, p) < 0.02
    assert M.ece(y, np.clip(p * 0.3, 0, 1)) > 0.2


def test_hr_metrics_doubling_and_coverage():
    ref = np.array([60., 60, 60, 60, 80, 80])
    pred = np.array([61., 120, 30, np.nan, 90, 82])
    h = M.hr_metrics(pred, ref)
    assert h['n'] == 6 and h['n_answered'] == 5 and abs(h['coverage'] - 5 / 6) < 1e-12
    assert abs(h['within5'] - 2 / 5) < 1e-12              # 61 and 82
    assert abs(h['within5_full'] - 2 / 6) < 1e-12
    assert abs(h['doubling'] - 1 / 5) < 1e-12 and abs(h['halving'] - 1 / 5) < 1e-12
    assert abs(h['mae'] - np.mean([1, 60, 30, 10, 2])) < 1e-12


def test_spo2_metrics():
    ref = np.array([98., 96, 85, 80]); pred = np.array([97., 97, 88, 82])
    s = M.spo2_metrics(pred, ref)
    assert abs(s['mae'] - 1.75) < 1e-12 and abs(s['bias'] - 1.25) < 1e-12 and s['auroc_lt'] == 1.0 and s['n_pos'] == 2


def test_time_matched_clock_scores_chance_and_perfect_scores_one():
    rng = np.random.default_rng(1)
    t = np.tile(np.arange(0, 700, 2.0), 6)
    # hypoxia becomes more frequent with elapsed time in every person (the shortcut the metric removes)
    y = (rng.random(len(t)) < np.clip(t / 700, 0.15, 0.85)).astype(int)
    clock = (t > 350).astype(int)
    tm = M.time_matched(t, y, clock, clock.astype(float))
    assert abs(tm['tm_acc'] - 0.5) < 0.06, tm
    perfect = M.time_matched(t, y, y, y.astype(float))
    assert perfect['tm_acc'] == 1.0 and perfect['tm_auroc'] == 1.0
    plain = (clock == y).mean()
    assert plain > tm['tm_acc'] + 0.03          # the clock looks fine on plain accuracy, not on TM


def test_time_matched_matches_monte_carlo_of_evalkit():
    sys.path.insert(0, str(REPO / 'python'))
    from oxydrop import evalkit
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 600, 3000); y = (rng.random(3000) < 0.4 + 0.3 * (t > 300)).astype(int)
    s = np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1); pred = (s > 0.5).astype(int)
    mc = evalkit.tm(t, y, pred, s, nrep=400, seed=0)
    assert abs(M.time_matched(t, y, pred, s)['tm_acc'] - mc) < 0.01


V1 = REPO / 'results' / 'loPO_v1_windows.csv'


@pytest.mark.skipif(not V1.exists(), reason='results/loPO_v1_windows.csv not present')
def test_reproduces_v1_forced_row_of_eval_v1_vs_v2():
    """Regression: v1 chain, LOPO on Hoffman, every window forced to answer: acc 87.5 %, TM accuracy 0.796 (results/eval_v1_vs_v2.csv row 1)."""
    d = pd.read_csv(V1)
    assert len(d) == 5644
    c = M.classification_metrics(d.y.values, d.p.values)
    tm = M.time_matched(d.t.values, d.y.values, (d.p.values > 0.5).astype(int), d.p.values)
    row = pd.read_csv(REPO / 'results' / 'eval_v1_vs_v2.csv').iloc[0]
    assert round(c['acc'], 4) == row['acc'] == 0.8754
    assert round(c['bacc'], 4) == row['bacc'] and round(c['sens'], 4) == row['sens'] and round(c['spec'], 4) == row['spec']
    assert round(c['auroc'], 4) == row['auc']
    assert round(tm['tm_acc'], 3) == round(row['tm_acc'], 3) == 0.796
    assert round(tm['tm_auroc'], 3) == 0.887
    ci = M.person_bootstrap(d.sid.values, M.classification_stat(d.y.values, d.p.values), B=1000, seed=1)
    lo, hi = eval(row['acc_ci'])
    assert abs(ci['lo'] - lo) < 0.01 and abs(ci['hi'] - hi) < 0.01


def test_selective_metrics_and_curve():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0]); pred = np.array([1, 1, 0, 0, 0, 0, 1, 0]); ans = np.array([1, 1, 1, 0, 1, 1, 0, 1], bool)
    s = M.selective_metrics(y, pred, ans)
    assert abs(s['coverage'] - 6 / 8) < 1e-12 and abs(s['acc_answered'] - 5 / 6) < 1e-12
    assert abs(s['alert_share_all'] - 2 / 4) < 1e-12        # 2 of ALL 4 hypoxic windows alerted
    conf = np.array([.9, .9, .6, .1, .8, .8, .2, .7])
    curve = M.accuracy_vs_coverage(y, pred, conf, grid=[0.5, 1.0])
    assert abs(curve[1]['acc'] - (pred == y).mean()) < 1e-12 and curve[0]['coverage'] == 0.5


def test_person_bootstrap_resamples_people_not_windows():
    rng = np.random.default_rng(3)
    subj = np.repeat(np.arange(8), 50); acc_p = rng.uniform(0.5, 0.95, 8)
    correct = (rng.random(400) < acc_p[subj]).astype(float)
    stat = lambda idx: correct[idx].mean()
    b = M.person_bootstrap(subj, stat, B=500, seed=0)
    assert b['lo'] < b['point'] < b['hi']
    # window-level bootstrap would give a much narrower interval than people-level
    w = np.array([correct[rng.integers(0, 400, 400)].mean() for _ in range(500)])
    assert (b['hi'] - b['lo']) > 1.5 * (np.percentile(w, 97.5) - np.percentile(w, 2.5))


def test_paired_bootstrap_detects_real_difference_and_a_tie():
    rng = np.random.default_rng(4)
    subj = np.repeat(np.arange(20), 40); y = rng.integers(0, 2, 800)
    good = np.where(rng.random(800) < 0.9, y, 1 - y).astype(float); ok = np.where(rng.random(800) < 0.7, y, 1 - y).astype(float)
    r = M.paired_person_bootstrap(subj, M.classification_stat(y, good), M.classification_stat(y, ok), B=400)
    assert r['win'] and r['lo'] > 0 and abs(r['diff'] - (r['a'] - r['b'])) < 1e-12
    same = M.paired_person_bootstrap(subj, M.classification_stat(y, good), M.classification_stat(y, good), B=200)
    assert same['diff'] == 0 and not same['win'] and not same['excludes_zero']
    lower = M.paired_person_bootstrap(subj, lambda i: np.abs(good[i] - y[i]).mean(), lambda i: np.abs(ok[i] - y[i]).mean(), B=200, higher_is_better=False)
    assert lower['win']                                         # lower error is better


def test_fold_summary_and_gap():
    s = M.summarize_folds([0.8, 0.9, 1.0, np.nan])
    assert s['n'] == 3 and abs(s['mean'] - 0.9) < 1e-12 and abs(s['sd'] - 0.1) < 1e-12 and s['lo'] <= 0.9 <= s['hi']
    g = M.train_vs_test_gap([0.99, 0.98, 0.97], [0.8, 0.9, 0.7])
    assert g['gap'] > 0 and abs(g['gap'] - np.mean([0.19, 0.08, 0.27])) < 1e-12
    e = M.train_vs_test_gap([1.0, 1.2], [3.0, 2.8], higher_is_better=False)       # errors: test larger than train = over-fit = positive
    assert e['gap'] > 0
