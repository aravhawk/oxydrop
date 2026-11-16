"""OxyTrack (python/oxydrop_ml/photon/oxytrack.py): causality, gated label-free adaptation, SNR behaviour, null control and the leakage guard.
Synthetic data only (no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from oxydrop_ml.photon import oxytrack as O


def _fake_em(seed=0):
    rng = np.random.default_rng(seed)
    cal = {}
    for proj in O.PROJ:
        for kappa in ('none',) + O.KAPPA:
            cal[(proj, kappa)] = dict(beta=np.array([-0.8, 0.9]), R=np.array([[40.0, -8.0], [-8.0, 20.0]]), soff2=np.array([2.0, 4.0]))
    return dict(sd=np.ones(19), W={'ridge': rng.normal(0, 0.3, 19), 'lr': rng.normal(0, 0.3, 19)}, q_emp=0.1, p0=0.1, mu_b=97.0, sig_b=1.0,
                cal=cal, n_typ=400.0, snr_med=np.array([5.0, 4.0]))


def _cfgs():
    return [dict(proj='ridge', q_mult=1.0, kappa=1.0, rho_b=4.0, nu0=10.0, eta=0.05, adapt=True),
            dict(proj='lr', q_mult=16.0, kappa=None, rho_b=1.0, nu0=40.0, eta=0.1, adapt=True),
            dict(proj='ridge', q_mult=4.0, kappa=0.3, rho_b=0.0, nu0=1.0, eta=0.0, adapt=False)]


def _desat_obs(n=200, seed=0):
    """Observations of a person who desaturates by 15 points after window 80 (dHR up, fus projection down), with noise."""
    rng = np.random.default_rng(seed)
    d = np.where(np.arange(n) < 80, 0.0, -15.0)
    u = np.stack([-0.8 * d + rng.normal(0, 3, n), 0.9 * d + rng.normal(0, 2, n)], 1)
    t = np.arange(n) * O.STEP
    return t, u


def _outputs(out, k):
    return [out[key][:, :k + 1] for key in ('m', 'P', 'lam', 'gate', 'b', 'score', 'spo2_hat', 'spo2_lo', 'spo2_hi')]


@pytest.mark.parametrize('k', [5, 29, 30, 31, 90, 150])
def test_filter_is_causal(k):
    em = _fake_em(); cfgs = _cfgs()
    t, u = _desat_obs()
    snr = np.random.default_rng(1).uniform(0.5, 20, (len(t), 3))
    a = O.attenuation(snr, cfgs); base = t + O.WIN <= O.CAL
    ref = O.track(u, a, base, O.params(em, cfgs))
    u2 = u.copy(); a2 = a.copy()
    rng = np.random.default_rng(2)
    u2[k + 1:] += rng.normal(0, 50, u2[k + 1:].shape); a2[:, k + 1:] = rng.uniform(0, 1, a2[:, k + 1:].shape)
    per = O.track(u2, a2, base, O.params(em, cfgs))
    for x, y in zip(_outputs(ref, k), _outputs(per, k)):
        np.testing.assert_array_equal(x, y)
    assert not np.allclose(ref['m'][:, k + 1:], per['m'][:, k + 1:])     # the perturbation does reach later windows


def _synthetic_rgb(sec=150, seed=0):
    rng = np.random.default_rng(seed)
    n = sec * 30
    tt = np.arange(n) / 30.0
    hr = 1.2 + 0.1 * np.sin(2 * np.pi * tt / 40)
    ph = 2 * np.pi * np.cumsum(hr) / 30.0
    pulse = np.sin(ph) + 0.4 * np.sin(2 * ph + 0.5)
    dc = np.array([150.0, 60.0, 40.0])
    amp = np.array([0.010, 0.008, 0.006])
    return dc[None, :] * (1 + amp[None, :] * pulse[:, None]) + rng.normal(0, 0.05, (n, 3))


def test_end_to_end_causal_from_raw_rgb():
    """Perturbing every frame after the end of window k (raw RGB) leaves the features and all outputs of windows <= k unchanged."""
    em = _fake_em(); cfgs = _cfgs()
    rgb = _synthetic_rgb()
    f = O.rec_features(rgb)
    ref = O.run_filter(em, cfgs, f)
    for t_k in (60, 84):
        k = int(np.nonzero(f['t'] == t_k)[0][0])
        rgb2 = rgb.copy()
        end = (t_k + O.WIN) * O.FS
        rgb2[end:] = rgb2[end:] * 0.3 + np.random.default_rng(3).normal(0, 5, rgb2[end:].shape)
        f2 = O.rec_features(rgb2)
        np.testing.assert_array_equal(f['D'][:k + 1], f2['D'][:k + 1])
        np.testing.assert_array_equal(f['snr'][:k + 1], f2['snr'][:k + 1])
        per = O.run_filter(em, cfgs, f2)
        for x, y in zip(_outputs(ref, k), _outputs(per, k)):
            np.testing.assert_array_equal(x, y)
        assert not np.allclose(f['D'][k + 1:], f2['D'][k + 1:])


def test_ema_is_causal():
    p = np.random.default_rng(0).uniform(0, 1, 100)
    for span in O.EMA_SPANS:
        q = p.copy(); q[41:] = 1 - q[41:]
        np.testing.assert_array_equal(O.ema(p, span)[:41], O.ema(q, span)[:41])
    np.testing.assert_array_equal(O.ema(p, 1), p)


def test_adaptation_only_in_baseline_or_clearly_normal():
    """Offsets and noise scales change only on windows inside the first minute or where the filter already called the window clearly normal."""
    em = _fake_em(); cfgs = _cfgs()[:2]
    t, u = _desat_obs()
    out = O.track(u, O.attenuation(np.full((len(t), 3), 10.0), cfgs), t + O.WIN <= O.CAL, O.params(em, cfgs))
    for g in range(len(cfgs)):
        closed = ~out['gate'][g]
        assert closed[100:].mean() > 0.9                                  # during the desaturation the gate is shut
        idx = np.nonzero(closed)[0]
        idx = idx[idx > 0]
        np.testing.assert_array_equal(out['b'][g, idx], out['b'][g, idx - 1])
        np.testing.assert_array_equal(out['lam'][g, idx], out['lam'][g, idx - 1])
        assert out['score'][g, 120:].mean() > 0.9                         # the desaturation is not absorbed as drift


def test_low_snr_leans_on_prior_and_widens():
    em = _fake_em(); cfgs = [dict(proj='ridge', q_mult=1.0, kappa=1.0, rho_b=0.0, nu0=1.0, eta=0.0, adapt=False)]
    t, u = _desat_obs()
    hi = O.track(u, O.attenuation(np.full((len(t), 3), 50.0), cfgs), t + O.WIN <= O.CAL, O.params(em, cfgs))
    lo = O.track(u, O.attenuation(np.full((len(t), 3), 0.01), cfgs), t + O.WIN <= O.CAL, O.params(em, cfgs))
    assert (lo['P'][0, 100:] > 5 * hi['P'][0, 100:]).all()               # posterior widens when the pulse is buried in noise
    assert (lo['spo2_hi'] - lo['spo2_lo'])[0, -1] > (hi['spo2_hi'] - hi['spo2_lo'])[0, -1]
    assert abs(lo['score'][0, -1] - 0.5) < abs(hi['score'][0, -1] - 0.5)  # less confident
    nos = [dict(cfgs[0], kappa=None)]
    a1 = O.track(u, O.attenuation(np.full((len(t), 3), 50.0), nos), t + O.WIN <= O.CAL, O.params(em, nos))
    a2 = O.track(u, O.attenuation(np.full((len(t), 3), 0.01), nos), t + O.WIN <= O.CAL, O.params(em, nos))
    np.testing.assert_array_equal(a1['score'], a2['score'])                # nosnr control ignores SNR


def test_blank_null_is_a_function_of_window_index_only():
    em = _fake_em(); cfgs = _cfgs()
    f1 = dict(t=np.arange(300) * 2, D=np.random.default_rng(0).normal(size=(300, 19)), snr=np.random.default_rng(1).uniform(0, 9, (300, 3)))
    f2 = dict(t=np.arange(250) * 2, D=np.random.default_rng(2).normal(size=(250, 19)), snr=np.random.default_rng(3).uniform(0, 9, (250, 3)))
    s1 = O.run_filter(em, cfgs, f1, blank=True)['score']; s2 = O.run_filter(em, cfgs, f2, blank=True)['score']
    np.testing.assert_array_equal(s1[:, :250], s2)


def _synthetic_dataset(seed=0):
    rng = np.random.default_rng(seed)
    feats, rows = {}, []
    for i, sid in enumerate(range(100001, 100007)):
        onset = 90 + 10 * i
        for hand in 'LR':
            n = 130
            t = np.arange(n) * 2
            spo2 = np.where(t < onset, 97.5, 97.5 - np.minimum((t - onset) / 4.0, 18.0)) + rng.normal(0, 0.3, n)
            d = spo2 - 97.5
            D = rng.normal(0, 1, (n, 19)) * 0.8
            D[:, 0] += -0.8 * d; D[:, 9] += 0.05 * d; D[:, 16] += 0.04 * d
            D[t + 10 <= 60] -= np.median(D[t + 10 <= 60], 0)
            rec = f'{sid}{hand}'
            feats[rec] = dict(t=t, D=D, snr=rng.uniform(1, 10, (n, 3)))
            rows += [dict(sid=sid, t=int(a), y=float(b), rec=rec) for a, b in zip(t, spo2)]
    return feats, pd.DataFrame(rows)


def test_heldout_labels_never_used():
    """Scrambling the held-out person's labels changes nothing of that person's fold (choices, thresholds, scores)."""
    feats, wdf = _synthetic_dataset()
    models = ('oxytrack_noadapt', 'oxytrack_ema')
    ref = O.nested_folds(feats, wdf, models=models, log=lambda *a: None)
    p = 100003
    w2 = wdf.copy()
    m = w2.sid == p
    w2.loc[m, 'y'] = np.random.default_rng(5).permutation(w2.loc[m, 'y'].values)[::-1] - 5.0
    scr = O.nested_folds(feats, w2, models=models, log=lambda *a: None)
    for model in models:
        fa = [f for f in ref[model] if f['sid'] == p][0]; fb = [f for f in scr[model] if f['sid'] == p][0]
        assert fa['cfg'] == fb['cfg'] and fa['thr'] == fb['thr'] and fa['c'] == fb['c']
        assert fa['band_scale'] == fb['band_scale'] and fa['sd_max'] == fb['sd_max']
        da = O.score_level({model: [fa]}, model, feats, wdf); db = O.score_level({model: [fb]}, model, feats, w2)
        np.testing.assert_array_equal(da.score.values, db.score.values)
        np.testing.assert_array_equal(da.pred.values, db.pred.values)
        np.testing.assert_array_equal(da.answer.values, db.answer.values)


def test_information_gate_abstains_when_posterior_is_wide():
    feats, wdf = _synthetic_dataset()
    em = _fake_em()
    cfg = dict(proj='ridge', q_mult=1.0, kappa=1.0, rho_b=0.0, nu0=1.0, eta=0.0, adapt=False)
    fo = dict(sid=100002, cfg=cfg, thr=0.5, c=0.0, em=em, chain=None, band_scale=1.0, sd_max=np.inf)
    a = O.score_level({'oxytrack': [fo]}, 'oxytrack', feats, wdf)
    b = O.score_level({'oxytrack': [dict(fo, sd_max=0.0)]}, 'oxytrack', feats, wdf)
    assert a.answer.all() and not b.answer.any()
    np.testing.assert_array_equal(a.score.values, b.score.values)        # the gate never changes score or pred
    np.testing.assert_array_equal(a.pred.values, b.pred.values)
    c = O.score_level({'oxytrack': [dict(fo, band_scale=2.0)]}, 'oxytrack', feats, wdf)
    np.testing.assert_allclose((c.spo2_hi - c.spo2_lo).values, 2.0 * (a.spo2_hi - a.spo2_lo).values)
