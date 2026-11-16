import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import metrics as M  # noqa: E402
from oxydrop_ml.baselines import chain_v1, common as C  # noqa: E402
from oxydrop_ml.photon import evalkit as E, stress as S, winner as W  # noqa: E402

HAVE_WINDOWS = E.WINDOWS.exists()


def _rec(n=3000, seed=0):
    """A synthetic finger recording: slow DC drift plus a 1.2 Hz pulse of 1 % in all channels, Nexus-6P-like levels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    base = np.array([55.0, 50.0, 42.0])
    pulse = 0.01 * np.sin(2 * np.pi * 1.2 * t)[:, None] * np.array([1.0, 1.5, 1.1])
    return base * (1 + pulse + 0.002 * rng.standard_normal((n, 3))) * (1 + 0.05 * np.sin(2 * np.pi * t / 90))[:, None]


def test_level_zero_is_identity():
    x = _rec()
    y = S.degrade(x, 0, seed=3)
    assert y.dtype == np.float64 and y.shape == x.shape and np.array_equal(y, x) and y is not x


def test_degrade_is_deterministic_per_seed_and_level():
    x = _rec()
    for lev in (1, 3, 5):
        assert np.array_equal(S.degrade(x, lev, 7), S.degrade(x, lev, 7))
        assert not np.array_equal(S.degrade(x, lev, 7), S.degrade(x, lev, 8))
    assert not np.array_equal(S.degrade(x, 2, 0), S.degrade(x, 3, 0))
    assert np.array_equal(S.degrade_x(x, 'X1', 4), S.degrade_x(x, 'X1', 4))
    assert not np.array_equal(S.degrade_x(x, 'X1', 4), S.degrade_x(x, 'X1', 5))
    assert S.degrade_x(x, 'X1', 0).shape == x.shape


def test_output_is_quantised_and_clipped():
    y = S.degrade(_rec() * 10, 1, 0)
    assert y.min() >= 0 and y.max() <= 255
    assert np.allclose(y / S.QUANT_STEP, np.rint(y / S.QUANT_STEP))


def test_blue_attenuates_more_than_red_as_level_grows():
    prev = np.ones(3)
    for lev in S.LEVELS[1:]:
        a = np.array(S.LEVEL_INFO[lev]['a_c'])
        assert a[2] < a[1] < a[0] < prev[0] and (a < prev).all()               # every channel dims with the level, blue the most, red the least
        assert a[2] / a[0] < prev[2] / prev[0] or lev == 1                      # the blue/red ratio keeps falling
        prev = a
    assert S.LEVEL_INFO[1]['light_frac_R'] < 1 and S.LEVEL_INFO[5]['light_frac_R'] < S.LEVEL_INFO[2]['light_frac_R']
    # the optics are applied to the recording: the blue/red ratio of the DC of the degraded signal falls with the level (leak excluded by using noise-free output)
    x = _rec()
    r = [S.degrade(x, l, 0).mean(0) for l in (1, 3)]
    assert r[1][2] / r[1][0] < r[0][2] / r[0][0] or r[1][2] / r[1][0] < 0.6


def test_renderer_sign_fix_gives_physical_decay():
    """Farrell's two dipole terms add. With the (wrong) subtraction of optics.py the red light at +2 mm would be ~1e-18; here it must be a few percent to tens of percent."""
    a = S.LEVEL_INFO[1]['a_c'][0]
    assert 1e-3 < a < 0.5


def test_time_matched_bootstrap_matches_metrics_closed_form():
    rng = np.random.default_rng(0)
    n = 900
    ref = pd.DataFrame(dict(rec=np.repeat([f'r{i}' for i in range(6)], n // 6), t=np.tile(np.arange(60, 60 + 2 * (n // 6), 2), 6), sid=np.repeat([f's{i // 2}' for i in range(6)], n // 6)))
    ref['y'] = ((ref.t // 100) % 2 == 0).astype(int) ^ (rng.random(n) < 0.35)
    pred = (rng.random(n) < 0.5 + 0.2 * (ref.y - 0.5)).astype(int)
    bt = E.Boot(ref, B=40, seed=1)
    point, boot, per = bt.tm(pred)
    assert point == pytest.approx(M.time_matched(ref.t.values, ref.y.values, pred)['tm_bacc'])
    for b in (0, 7, 33):
        idx = np.concatenate([bt.rows[i] for i in np.repeat(np.arange(bt.P), bt.W[b].astype(int))])
        assert boot[b] == pytest.approx(M.time_matched(ref.t.values[idx], ref.y.values[idx], pred[idx])['tm_bacc'])


@pytest.mark.skipif(not HAVE_WINDOWS, reason='results/scores/windows.csv not present')
def test_clock_only_score_is_chance_under_the_evaluator_and_null_flag(tmp_path):
    ref = E.eval_windows()
    assert len(ref) == 5644
    clock = ((ref.t // E.BIN_S) % 2).astype(int)                           # constant inside every 100 s bin: a pure clock
    for name, pred in (('clock_null', clock), ('leaky_null', ref.y.values), ('chain_v1', clock)):
        pd.DataFrame(dict(rec=ref.rec, t=ref.t, sid=ref.sid, y_spo2=ref.y_spo2, y=ref.y, score=pred.astype(float), pred=pred, answer=True)).to_csv(tmp_path / f'{name}__L0.csv', index=False)
    T = E.run(tmp_path, tmp_path / 'out', B=30, log=lambda *_: None)
    r = T.set_index('model')
    assert r.loc['clock_null', 'tm_bacc'] == pytest.approx(0.5, abs=1e-9)
    assert r.loc['clock_null', 'null_check'] == 'ok'
    assert r.loc['leaky_null', 'null_check'].startswith('FLAG')
    assert 'FLAG' in (tmp_path / 'out' / 'TABLE.md').read_text()
    assert (tmp_path / 'out' / 'STRESS.csv').exists()


def test_sign_test_exact():
    assert E.sign_test(6, 6) == pytest.approx(1 / 64)
    assert E.sign_test(5, 6) == pytest.approx(7 / 64)
    assert E.sign_test(0, 5) == pytest.approx(1.0)
    assert np.isnan(E.sign_test(0, 0))


def test_score_file_names_include_stress_family():
    assert E.parse_name('x/flashgap__L3.csv') == ('flashgap', 3)
    assert E.parse_name('chain_v1_nestedC__X1.csv') == ('chain_v1_nestedC', 'X1')
    assert E.parse_name('notes.csv') == (None, None)


class _Ctx:
    pass


def test_chain_threshold_and_C_ignore_the_test_person_even_when_its_features_are_degraded():
    """Leakage guard like tests/test_baselines.py, for the chain path used by baselines_stress: scrambling the held-out person's labels and features changes none of
    the selected C, the threshold, the abstain band or the (clean) training-side scores of that fold."""
    def make(seed=0):
        rng = np.random.default_rng(seed)
        c = _Ctx(); n_sub, n_per = 6, 120
        c.subject = np.repeat([f's{i}' for i in range(n_sub)], n_per)
        c.t = np.tile(np.arange(n_per) * 2.0, n_sub)
        X = rng.normal(size=(n_sub * n_per, 19)); c.y = (X[:, 0] + 0.5 * rng.normal(size=len(X)) > 0).astype(int)
        c.F = dict(fusion19_v1=X); c.valid = {1: np.ones(len(X), bool)}
        return c
    a_ctx = make(); b_ctx = make()
    te = np.nonzero(b_ctx.subject == 's2')[0]
    rng = np.random.default_rng(5)
    b_ctx.y[te] = rng.integers(0, 2, len(te))
    b_ctx.F['fusion19_v1'][te] = 3.0 * rng.normal(size=(len(te), 19)) + 1.0          # a different, 'degraded' held-out person
    spec = lambda c: chain_v1.specs(c, 1)[1]
    fa = [f for f in C.run_cls(a_ctx, spec(a_ctx)) if f.subject == 's2'][0]
    fb = [f for f in C.run_cls(b_ctx, spec(b_ctx)) if f.subject == 's2'][0]
    assert fa.hp == fb.hp and fa.thr == fb.thr and fa.band == fb.band and fa.spec_thr == fb.spec_thr
    assert np.array_equal(fa.s_tr, fb.s_tr)


def test_winner_report_is_generated_from_table_csv(tmp_path, monkeypatch):
    """WINNER.md is written from TABLE.csv: the shipped model's numbers appear verbatim and the no-win statement is present."""
    import shutil
    if not (W.RES / 'TABLE.csv').exists() or not (W.RES / 'NULLS.csv').exists():
        pytest.skip('results/photon/TABLE.csv or NULLS.csv not present')
    for f in ('TABLE.csv', 'NULLS.csv'):
        shutil.copy(W.RES / f, tmp_path / f)
    monkeypatch.setattr(W, 'RES', tmp_path)
    W.main(['--no-nulls'])
    text = (tmp_path / 'WINNER.md').read_text(encoding='utf-8')
    t = pd.read_csv(tmp_path / 'TABLE.csv', dtype={'level': str})
    r = t[(t.model == W.SHIPPED) & (t.level == '3')].iloc[0]
    assert f'{r.tm_bacc:.3f} [{r.tm_lo:.3f}, {r.tm_hi:.3f}]' in text
    assert 'No candidate beats logistic chain v1' in text
    assert 'No skin-tone claim' in text
