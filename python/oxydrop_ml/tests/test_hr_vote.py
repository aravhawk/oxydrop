"""Tests of the HR voting rule (FEAT-007): the per-window rule, the causal median, and the committed tables recomputed from the committed input CSVs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import hr_vote as V, metrics as M   # noqa: E402

RES = Path(__file__).resolve().parents[3] / 'results' / 'ml'


def test_agreement_tolerance_is_max_of_5_bpm_and_6_percent():
    assert V.tol(60) == 5.0 and V.tol(100) == 6.0
    assert V.agree(70, 74.9) and not V.agree(70, 76.5)
    assert V.agree(150, 158) and not V.agree(150, 160)


def test_vote_cases():
    assert V.vote_window(70, 71, 90) == pytest.approx(70.5)                      # two classical estimates agree, the CNN is out
    assert V.vote_window(70, 90, 71) == pytest.approx(70.5)                      # spectral and CNN agree
    assert V.vote_window(70, 71, 72) == pytest.approx(71.0)                      # all agree: mean of the group
    assert V.vote_window(140, 141, 70, harmonic=True, octave=False) == pytest.approx(140.5)       # two classical doubled, no octave check
    assert V.vote_window(140, 141, 70, harmonic=True, octave=True) == pytest.approx(70.0)         # the octave check returns the CNN
    assert V.vote_window(140, 90, 70, harmonic=False, octave=False) == pytest.approx(70.0)        # nothing agrees, no correction: the CNN
    assert V.vote_window(140, 35, 70, harmonic=True, octave=True) == pytest.approx(70.0)          # 35 is out of range and dropped; 140 is moved to 70 and agrees with the CNN
    assert np.isnan(V.vote_window(np.nan, np.nan, np.nan))
    assert V.vote_window(np.nan, np.nan, 66.0) == 66.0                                              # only the CNN
    assert V.vote_window(120, np.nan, 125) == pytest.approx(122.5)


def test_harmonic_correction_moves_to_the_octave_of_the_cnn():
    # no pair agrees directly (45, 130, 66); 130 -> 65 agrees with 66, 45 -> 90 is farther than 45
    assert V.vote_window(130, 45, 66, harmonic=True, octave=True) == pytest.approx(65.5, abs=1.0)
    assert V.vote_window(130, 45, 66, harmonic=False, octave=False) == pytest.approx(66.0)


def test_causal_median_uses_only_past_windows():
    x = np.array([60., 61., 200., 62., 63., 64., 300.])
    rec = np.array(['a'] * 7)
    t = np.arange(7) * 2.0
    out = V.causal_median(x, rec, t, 5)
    assert out[0] == 60 and out[1] == 60.5 and out[2] == 61 and out[3] == 61.5 and out[4] == 62
    x2 = x.copy(); x2[6] = 1.0
    assert np.array_equal(V.causal_median(x2, rec, t, 5)[:6], out[:6])
    r2 = np.array(['a', 'b'] * 3 + ['a'])                                      # recordings do not mix
    o2 = V.causal_median(x, r2, t, 5)
    assert o2[1] == x[1] and o2[0] == 60.0


def test_variants_apply_in_the_declared_order():
    df = pd.DataFrame(dict(rec=['r'] * 6, t_start=np.arange(6) * 2.0, hr_spec=[140, 141, 142, 70, 71, 72], hr_acf=[141, 140, 141, 71, 70, 71], hr_cnn=[70, 71, 70, 70, 71, 70], hr_p2=[70.] * 6))
    full = V.apply_variant(df, 'vote+harmonic+octave+median5')
    octv = V.apply_variant(df, 'vote+harmonic+octave')
    assert np.allclose(octv, [70, 71, 70, 70.5, 70.5, 71], atol=1.0)
    assert np.allclose(full, V.causal_median(octv, df.rec.values, df.t_start.values))


# ------------------------------------------------------------------------------------------------ committed results
def test_butppg_votes_recompute_from_the_committed_inputs():
    d = pd.read_csv(RES / 'scores' / 'hr_vote_inputs_butppg.csv')
    v = pd.read_csv(RES / 'scores' / 'hr_vote_butppg.csv')
    assert len(d) == len(v) == 3840
    est = V.apply_variant(d, 'vote+harmonic+octave+median5')
    assert np.allclose(est, v.hr_vote.values, equal_nan=True)
    assert d.groupby('rec').size().max() == 1                                   # one window per BUT recording: the median of 5 is a no-op
    assert np.allclose(V.apply_variant(d, 'vote+harmonic+octave'), est, equal_nan=True)


def test_summary_rows_recompute_from_the_inputs():
    t = pd.read_csv(RES / 'hr_vote_summary.csv')
    for ds in ('butppg', 'hoffman'):
        d = pd.read_csv(RES / 'scores' / f'hr_vote_inputs_{ds}.csv')
        for v in ('cnn', 'vote+harmonic+octave+median5'):
            est = V.apply_variant(d, v)
            for scope in ('all', 'clean'):
                m = d.gate_ok.values.astype(bool) if scope == 'clean' else np.ones(len(d), bool)
                mm = M.hr_metrics(est[m], d.hr_ref.values[m])
                r = t[(t.dataset == ds) & (t.variant == v) & (t.scope == scope)].iloc[0]
                assert r.mae == pytest.approx(mm['mae']) and r.within5 == pytest.approx(mm['within5']) and r.doubling == pytest.approx(mm['doubling'])
                assert r.n_windows == int(np.isfinite(d.hr_ref.values[m]).sum())


def test_gate_flag_matches_the_committed_p2_estimates():
    d = pd.read_csv(RES / 'scores' / 'hr_vote_inputs_butppg.csv')
    f = pd.read_csv(RES / 'scores' / 'hybrid_final_butppg.csv')
    assert np.array_equal(d.gate_ok.values.astype(bool), f.spec_gate_ok.values.astype(bool))
    assert np.allclose(d.spec_forced_p2.values, f.hr_spec.values) and np.allclose(d.hr_p2.values, f.hr_final.values) and np.allclose(d.hr_cnn.values, f.hr_cnn.values)


def test_target_flag_follows_the_declared_numbers():
    t = pd.read_csv(RES / 'hr_vote_summary.csv')
    c = t[(t.scope == 'clean') & (t.variant == 'vote+harmonic+octave+median5')]
    for _, r in c.iterrows():
        assert bool(r.meets_target) == bool(r.mae < V.TARGET_MAE and r.doubling < V.TARGET_DOUBLING)


def test_md_numbers_come_from_the_csv():
    md = (RES / 'HR_VOTE.md').read_text()
    t = pd.read_csv(RES / 'hr_vote_summary.csv')
    r = t[(t.dataset == 'butppg') & (t.scope == 'clean') & (t.variant == 'vote+harmonic+octave+median5')].iloc[0]
    assert f'{r.mae:.2f}' in md and 'Generated by' in md
