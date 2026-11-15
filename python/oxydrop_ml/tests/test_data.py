import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from oxydrop_ml import CACHE, CTX, FS, WIN  # noqa: E402
from oxydrop_ml import features as F  # noqa: E402
from oxydrop_ml import windows as W  # noqa: E402

L = CTX + WIN


def synth(hr=72, a=(0.010, 0.020, 0.008), n=L, noise=0.0005, seed=0, dc=(5.0, 4.5, 4.0)):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS; f = hr / 60
    p = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t + 0.5)
    return np.stack([dc[c] + a[c] * p for c in range(3)], 1) + noise * rng.standard_normal((n, 3))


def test_channel_layout_constants():
    assert F.C_IN == 10 and F.S_IN == 9 and len(F.ABS) == 11 and len(F.BASE) == 9
    assert F.CHANNELS[:3] == ['ac_R', 'ac_G', 'ac_B'] and F.CHANNELS[3] == 'dt' and F.CHANNELS[6] == 'beat_ok'
    x, s = F.window_to_tensor(np.exp(synth()) - 1, None, None)
    assert x.shape == (10, WIN) and s.shape == (9,) and x.dtype == np.float32


def test_beats_hr_and_ratio_of_ratios_recovered():
    for hr in (55, 72, 100, 150):
        lr = synth(hr)
        _, _, a = F.windows_to_tensor(lr[None], np.full((1, L), 1 / FS))
        a = dict(zip(F.ABS, a[0]))
        assert abs(a['hr_beat'] - hr) < 4, (hr, a['hr_beat'])
        assert abs(a['ror_RG'] - np.log(0.010 / 0.020)) < 0.1 and abs(a['ror_RB'] - np.log(0.010 / 0.008)) < 0.1
        assert a['beat_frac'] > 0.9


def test_causality_future_samples_do_not_change_past_outputs():
    """Perturbing the last 40 samples must leave every output sample before them (ac, dc, ror, beat_ok) unchanged."""
    lr = synth(80, seed=1)
    base = np.array([5.0, 4.5, 4.0, -4.9, -4.2, -5.1, -0.7, 0.2, 80.0])
    x0, _ = F.window_to_tensor(np.exp(lr) - 1, None, base)
    lr2 = lr.copy(); rng = np.random.default_rng(5); lr2[-40:] += 0.05 * rng.standard_normal((40, 3)) + 0.02
    x1, _ = F.window_to_tensor(np.exp(lr2) - 1, None, base)
    cut = WIN - 40
    assert np.allclose(x0[:, :cut], x1[:, :cut], atol=1e-6)
    assert not np.allclose(x0[:, cut:], x1[:, cut:])
    # 1 sample of look-ahead is the only allowed delay of a beat: a peak is confirmed one sample after it occurs
    lr3 = lr.copy(); lr3[-1] += 1.0
    x3, _ = F.window_to_tensor(np.exp(lr3) - 1, None, base)
    assert np.allclose(x0[:, :-1], x3[:, :-1], atol=1e-6)


def test_batched_equals_single_and_chunk_independent():
    lrs = np.stack([synth(60 + 10 * i, seed=i) for i in range(5)])
    dt = np.full((5, L), 1 / FS)
    x, s, a = F.windows_to_tensor(lrs, dt)
    for i in range(5):
        xi, si, ai = F.windows_to_tensor(lrs[i:i + 1], dt[i:i + 1])
        assert np.allclose(x[i], xi[0], atol=1e-6) and np.allclose(s[i], si[0], atol=1e-6) and np.allclose(a[i], ai[0], atol=1e-5, equal_nan=True)


def test_relative_channels_zero_at_baseline_and_shift_with_it():
    lr = synth(72)
    _, _, a = F.windows_to_tensor(lr[None], np.full((1, L), 1 / FS))
    base = F.baseline_from_abs(np.repeat(a, 5, 0))
    assert np.isfinite(base).all() and abs(base[8] - 72) < 4
    x, s = F.window_to_tensor(np.exp(lr) - 1, None, base)
    assert np.abs(s[:8]).max() < 0.05 and abs(x[7:10]).max() < 0.1          # the baseline window sits at the baseline
    lr_dim = lr.copy(); lr_dim[:, 0] -= 0.2                                   # red DC drops by 0.2 nepers
    x2, s2 = F.window_to_tensor(np.exp(lr_dim) - 1, None, base)
    assert abs(s2[2] + 0.2) < 0.01 and abs(x2[7, -1] + 0.2) < 0.02
    lr_ratio = lr.copy(); lr_ratio[:, 0] = lr_ratio[:, 0].mean() + (lr_ratio[:, 0] - lr_ratio[:, 0].mean()) * 1.5    # red pulse 1.5x larger
    _, s3 = F.window_to_tensor(np.exp(lr_ratio) - 1, None, base)
    assert abs(s3[0] - np.log(1.5)) < 0.12


def test_missing_baseline_gives_zero_relative_channels():
    x, s = F.window_to_tensor(np.exp(synth()) - 1, None, None)
    assert np.all(x[4] == 0) and np.all(x[5] == 0) and np.all(x[7:10] == 0) and np.all(s[:8] == 0)


def test_no_pulse_means_no_beats_and_zero_ror_channels():
    rng = np.random.default_rng(0)
    lr = 5.0 + 1e-6 * rng.standard_normal((L, 3)) * 0 + np.linspace(0, 0.001, L)[:, None]
    x, s = F.window_to_tensor(np.exp(lr) - 1, None, np.array([5, 5, 5, -5, -5, -5, 0, 0, 70.]))
    assert x[6].mean() < 0.2 or np.abs(x[4]).max() <= F.ROR_CLIP * 2


def test_contact_two_wavelength_mask_zeroes_blue():
    lr = synth(); lr[:, 2] = 0.0
    x, s, a = F.windows_to_tensor(lr[None], np.full((1, L), 1 / FS), None, np.array([[True, True, False]]))
    assert np.all(x[0, 2] == 0) and np.all(x[0, 5] == 0) and np.all(x[0, 9] == 0)
    assert np.isnan(a[0, 7]) and np.isfinite(a[0, 6])


def test_dt_channel_flags_stalls():
    lr = synth(); dt = np.full(L, 1 / FS); dt[L - 100:L - 90] = 0.2
    x, _ = F.window_to_tensor(np.exp(lr) - 1, dt, None)
    assert np.allclose(x[3, :150], 0, atol=1e-6) and x[3, 200:210].max() > 4.0


def test_resample_uniform_and_dt():
    t = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(5.3, 10, 1 / 30)])                # a 0.3 s stall at t = 5
    X = np.stack([np.sin(t), np.cos(t), t], 1)
    out, dt = W.resample_uniform(t, X)
    assert abs(len(out) - 10 * 30) <= 2 and np.allclose(out[:, 2], np.arange(len(out)) / 30 + t[0], atol=1e-6)
    assert dt.max() > 0.3 and np.median(dt) < 0.04 and (dt > 0.1).sum() >= 8                     # the stall shows up as a run of large dt


def test_simulate_irregular_keeps_length_and_flags_timing():
    rng = np.random.default_rng(0)
    clean = synth(72, n=600, noise=0.0)
    out, dt = W.simulate_irregular(clean, rng, p_stall=0.02)
    assert out.shape == clean.shape and dt.shape == (600,)
    assert dt.max() > 1 / 30 + 0.05 and np.abs(out - clean).max() < 0.05
    out0, dt0 = W.simulate_irregular(clean, rng, jitter=0.0, p_stall=0.0, p_drop=0.0)
    assert np.allclose(dt0, 1 / 30, atol=1e-3) and np.abs(out0[:-3] - clean[:-3]).max() < 1e-3


def _mini_shard(tmp_path, ds='hoffman'):
    w = W.ShardWriter(ds, str(tmp_path / ds))
    for k, subj in enumerate(['s1', 's2']):
        n = 90 * FS
        lr = np.concatenate([synth(70 + 5 * k, n=n, seed=k)])
        wins = pd.DataFrame(dict(t_start=np.arange(0, 80, 2)))
        wins['spo2_ref'] = 95.0; wins['has_spo2'] = True; wins['eval'] = wins.t_start >= 60; wins['in_baseline'] = wins.t_start + 10 <= 60
        wins['base_key'] = f'rec{k}'
        w.add(subj, f'rec{k}', lr, np.full(n, 1 / FS), wins)
    return w.finalize()


def test_shard_roundtrip_reflection_pad_and_baseline(tmp_path):
    ws = _mini_shard(tmp_path)
    m = ws.meta
    assert len(m) == 2 * 40 and m.subject.nunique() == 2 and m['eval'].sum() == 2 * 10
    idx = np.array([0, 3, 40])
    buf = ws.lrgb(idx)
    assert buf.shape == (3, L, 3)
    # window 0 starts at t=0: the 5 s before it are the odd reflection around the first sample
    first = ws.series[0]
    assert np.allclose(buf[0, CTX], first, atol=1e-6) and np.allclose(buf[0, CTX - 1], 2 * first - ws.series[1], atol=1e-6)
    # window 3 (t_start = 6 s) has genuine context (6 s >= 5 s of warm-up)
    assert np.allclose(buf[1], ws.series[(6 * FS + WIN) - L + np.arange(L)], atol=1e-6)
    # the second recording's first window must NOT read the first recording's samples
    assert np.allclose(buf[2, CTX], ws.series[ws.meta.rec_start.values[40]], atol=1e-6)
    assert m.has_base.all() and np.isfinite(m.base_hr).all() and abs(m.base_hr.iloc[0] - 70) < 4 and abs(m.base_hr.iloc[40] - 75) < 4
    x, s = ws.tensors(np.arange(5))
    assert x.shape == (5, 10, WIN) and s.shape == (5, 9)
    ws.build_tensor_cache()
    xc, sc = ws.cached()
    assert np.allclose(xc[:5], x, atol=2e-2) and np.allclose(sc[:5], s, atol=1e-5)
    info = json.load(open(Path(ws.dir) / 'info.json'))
    assert info['n_windows'] == 80


def test_label_helpers():
    assert W.window_starts(30 * FS).tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    x = np.arange(20, dtype=float); x[3] = np.nan
    assert abs(W.mean_label(x, 0) - np.nanmean(x[:10])) < 1e-12
    x[:8] = np.nan
    assert np.isnan(W.mean_label(x, 0))                                      # fewer than 5 valid seconds


def test_openox_lag_estimator_recovers_a_known_lag_and_ignores_test_subjects(tmp_path):
    from oxydrop_ml.data import openox as O
    rng = np.random.default_rng(0)
    true_lag = 17
    recs, ref_rows, meta = [], [], []
    for k in range(12):
        n = 1500
        spo2 = 97 - 20 * np.clip(np.sin(np.arange(n) / 150.0 + k), 0, 1) + rng.normal(0, 0.3, n)            # tissue-true SpO2 per second
        ref = np.roll(spo2, true_lag)                                                                    # the display shows it `true_lag` s later: ref[t] = spo2[t - lag]
        ref_rows.append(pd.DataFrame(dict(rec=f'r{k}', t=np.arange(n), ref_hr=70.0, ref_spo2=ref, sao2_est=ref)))
        ts = np.arange(0, n - 10, 2)
        tissue = np.array([spo2[t:t + 10].mean() for t in ts])
        meta.append(pd.DataFrame(dict(subject=f's{k}', rec=f'r{k}', aligned=True, t_start=ts, abs_ror_RG=-0.03 * (tissue - 97) + rng.normal(0, 0.05, len(ts)))))
    pd.concat(ref_rows).to_parquet(tmp_path / 'ref.parquet')
    ws = SimpleNamespace(meta=pd.concat(meta, ignore_index=True), dir=str(tmp_path))
    r = O.estimate_lag([f's{k}' for k in range(8)], ws)
    assert r['confident'] and abs(r['lag'] - true_lag) <= 2 and r['n_enc'] == 8
    assert np.isnan(O.estimate_lag(['s0', 's1'], ws)['lag'])
    # the estimator only reads the subjects it is given: scrambling a TEST subject's data does not change the answer
    ws2 = SimpleNamespace(meta=ws.meta.copy(), dir=str(tmp_path))
    ws2.meta.loc[ws2.meta.subject == 's11', 'abs_ror_RG'] = rng.normal(size=int((ws2.meta.subject == 's11').sum()))
    assert O.estimate_lag([f's{k}' for k in range(8)], ws2)['lag'] == r['lag']
    # win mean helper with a lag
    x = np.arange(100, dtype=float)
    assert abs(O._win_mean(x, np.array([10]), 5)[0] - np.mean(x[15:25])) < 1e-12
    assert np.isnan(O._win_mean(x, np.array([95]), 0)[0])


# ---- built shards (skipped when the cache is not present, e.g. in CI)
def _shard(ds):
    p = Path(CACHE) / 'data' / ds / 'meta.parquet'
    return pd.read_parquet(p) if p.exists() else None


@pytest.mark.skipif(_shard('hoffman') is None, reason='hoffman shard not built')
def test_built_hoffman_matches_documented_counts():
    m = _shard('hoffman')
    assert int(m['eval'].sum()) == 5644 and m.subject.nunique() == 6 and m.rec.nunique() == 12 and m[m['eval']].has_base.all()


@pytest.mark.skipif(_shard('mths') is None or _shard('butppg') is None, reason='mths/butppg shards not built')
def test_built_mths_butppg_counts():
    assert _shard('mths').subject.nunique() == 62
    b = _shard('butppg')
    assert b[b.prereg_cohort].subject.nunique() == 38 and b.has_hr.all()


@pytest.mark.skipif(_shard('openox') is None, reason='openox shard not built')
def test_built_openox_has_hr_sao2_and_skin_tone():
    m = _shard('openox')
    assert m.has_hr.sum() > 100000 and m.has_sao2.sum() > 100000 and m.has_sao2_draw.sum() > 10000
    assert m.groupby('subject').monk.first().notna().sum() > 100 and (m.domain == 1).all()
    assert m[m.has_sao2].aligned.all()                                      # labels only on clock-aligned encounters
