"""PhotonTwin renderer and PhotonTwin-Inverse (photon/twin_inverse.py) unit tests: finiteness, Farrell 1992 Eq. 10, ratio-of-ratios monotonicity,
the label-free inversion and the leakage guards. No dataset needed."""
import numpy as np
import pandas as pd
import pytest
import torch

from oxydrop_ml import optics as OP
from oxydrop_ml import spectra as SP


def _k(n, v):
    return torch.full((n,), float(v), dtype=torch.float64)


def _farrell_eq10(mua, musp, rho):
    """Independent numpy evaluation of Farrell, Patterson and Wilson 1992 Eq. 10 (the two dipole terms ADDED), linear space."""
    mut = mua + musp
    a_p = musp / mut
    z0 = 1.0 / mut
    zb = 2.0 * OP.A_BND / (3.0 * mut)
    mueff = np.sqrt(3.0 * mua * mut)
    r1 = np.sqrt(rho ** 2 + z0 ** 2)
    r2 = np.sqrt(rho ** 2 + (z0 + 2 * zb) ** 2)
    return a_p / (4 * np.pi) * (z0 * (mueff + 1 / r1) * np.exp(-mueff * r1) / r1 ** 2
                                + (z0 + 2 * zb) * (mueff + 1 / r2) * np.exp(-mueff * r2) / r2 ** 2)


@pytest.mark.parametrize('rho', [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0])
def test_renderer_finite_float64(rho):
    tw = OP.PhotonTwin()
    sat = torch.linspace(0.5, 1.0, 11, dtype=torch.float64)
    n = len(sat)
    with torch.no_grad():
        out = tw.render(sat, _k(n, 0.02), _k(n, 0.03), _k(n, rho), _k(n, 0.008), torch.zeros(n, dtype=torch.long))
    for k, v in out.items():
        assert v.dtype == torch.float64
        assert torch.isfinite(v).all(), (k, rho)
    assert (out['ln_dc'][:, 0] > out['ln_dc'][:, 1]).all() and (out['ln_dc'][:, 1] > out['ln_dc'][:, 2]).all(), 'pre-camera B < G < R'


def test_renderer_gradients_finite():
    tw = OP.PhotonTwin()
    sat = torch.full((4,), 0.9, dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([1.0, 3.0, 8.0, 15.0], dtype=torch.float64, requires_grad=True)
    out = tw.render(sat, _k(4, 0.02), _k(4, 0.03), rho, _k(4, 0.008), torch.zeros(4, dtype=torch.long))
    (out['ln_dc'].sum() + out['ror'].sum()).backward()
    assert torch.isfinite(sat.grad).all() and torch.isfinite(rho.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in tw.parameters() if p.grad is not None)


@pytest.mark.parametrize('rho', [0.5, 1.0, 3.0, 6.0, 10.0])
def test_ln_reflectance_matches_farrell_eq10(rho):
    """ln R of the renderer equals Farrell Eq. 10 (sum of the real-source and image-source terms) where linear space does not underflow."""
    tw = OP.PhotonTwin()
    sat = np.array([0.7, 0.97])
    mua = tw.mu_a(torch.as_tensor(sat), _k(2, 0.02), _k(2, 0.03))
    with torch.no_grad():
        ln_r = tw.ln_reflectance(mua, _k(2, rho)).numpy()
    ref = _farrell_eq10(mua.detach().numpy(), tw.mu_sp().detach().numpy()[None], rho)
    ok = (ref > 1e-250) & (np.log(np.maximum(ref, 1e-300)) > tw.cfg.clamp_lnR + 1.0)      # linear space representable, above the renderer's ln R floor
    assert ok.any()
    np.testing.assert_allclose(ln_r[ok], np.log(ref[ok]), rtol=0, atol=1e-9)


def test_reflectance_decays_with_rho_and_blue_first():
    tw = OP.PhotonTwin()
    rho = torch.tensor([1.0, 2.0, 4.0, 8.0, 15.0], dtype=torch.float64)
    with torch.no_grad():
        dc = tw.ln_channels(_k(5, 0.97), _k(5, 0.02), _k(5, 0.03), rho, torch.zeros(5, dtype=torch.long)).numpy()
    assert (np.diff(dc, axis=0) < 0).all(), 'DC falls with separation in every channel'
    att = dc[-1] - dc[0]
    assert att[2] < att[1] < att[0], 'blue dies first, red last'


@pytest.mark.parametrize('rho', [1.0, 2.0, 3.0, 4.0, 5.5, 8.0])
def test_ratio_of_ratios_monotone_in_saturation(rho):
    """ln ratio-of-ratios R/G and R/B fall monotonically as saturation rises (deoxy-Hb raises red pulsatile absorption) at phone-like separations,
    including the stress-test range rho0 + 0.25 d_rho = 3-5.5 mm."""
    tw = OP.PhotonTwin()
    grid = np.linspace(0.6, 1.0, 41)
    for pair in (0, 1):
        c = tw.calibration_curve(grid, rho=rho, pair=pair)
        assert np.isfinite(c).all()
        assert (np.diff(c) < 0).all(), (rho, pair)


def test_ratio_of_ratios_monotone_at_large_rho():
    """At 15 mm the band-integrated curve is still monotone in saturation (its direction may flip: the band shift moves the curve)."""
    tw = OP.PhotonTwin()
    grid = np.linspace(0.6, 1.0, 41)
    for pair in (0, 1):
        d = np.diff(tw.calibration_curve(grid, rho=15.0, pair=pair))
        assert (d > 0).all() or (d < 0).all()


def test_lam_subsampling_close_to_full_grid():
    full = OP.PhotonTwin()
    sub = OP.PhotonTwin(OP.OpticsCfg(lam_max=720.0, n_quad=31))
    assert len(sub.lam) == 31 and float(sub.lam.max()) <= 720.0
    grid = np.linspace(0.7, 1.0, 7)
    for rho in (2.0, 3.0, 5.0):
        a = full.calibration_curve(grid, rho=rho); b = sub.calibration_curve(grid, rho=rho)
        assert np.max(np.abs(a - b)) < 0.05


# ------------------------------------------------------------------------------------------------ twin_inverse pieces
def test_gn_inversion_recovers_saturation_change():
    from oxydrop_ml.photon import twin_inverse as TI
    dec = TI.Decoder(learn=False)
    n = 7
    z_true = torch.zeros(n, 4, dtype=torch.float64)
    z_true[:, 0] = torch.linspace(-0.15, 0.0, n)
    with torch.no_grad():
        y = dec(z_true, _k(n, 0.975), _k(n, 3.0)).numpy()
    var = np.full((n, 6), 1e-6)
    z, sd = TI.gn_invert(y, var, np.full(n, 0.975), np.full(n, 3.0), n_iter=15, dec=dec)
    np.testing.assert_allclose(z[:, 0], z_true[:, 0].numpy(), atol=0.01)
    assert np.isfinite(sd).all()
    # weak light: inflate the noise -> wider posterior on d_sat
    _, sd_noisy = TI.gn_invert(y, var * 1e4, np.full(n, 0.975), np.full(n, 3.0), n_iter=15, dec=dec)
    assert (sd_noisy[:, 0] > sd[:, 0]).all()


def test_rho_inversion_rises_with_blue_loss():
    from oxydrop_ml.photon import twin_inverse as TI
    curve = TI._chroma_curve()
    i3 = int(np.argmin(np.abs(TI._RHO_GRID - 3.0))); i5 = int(np.argmin(np.abs(TI._RHO_GRID - 5.0)))
    fit = TI.RhoFit(np.stack([curve[i3] + 0.1, curve[i3] - 0.1]))          # training recordings around rho = 3
    r3, r5 = fit.rho(curve[i3]), fit.rho(curve[i5])
    assert abs(float(r3[0]) - 3.0) < 0.3
    assert float(r5[0]) > float(r3[0])


def _fake_feats(n_people=3, n_t=60, seed=0):
    from oxydrop_ml.photon import twin_inverse as TI
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_people):
        for hand in 'LR':
            t = np.arange(0, 2 * n_t, 2)
            y = 97 - 12 * (t > 60) * (np.sin(t / 30 + p) > 0)
            d = pd.DataFrame(dict(rec=f'{p}{hand}', sid=str(p), t=t, y_spo2=y.astype(float)))
            for c in TI.FUSION:
                d['D_' + c] = rng.normal(size=len(t)) * 0.05
            d['D_dc_R'] = -(97 - y) * 0.01 + rng.normal(size=len(t)) * 0.01
            for c in 'RGB':
                d['snr_' + c] = 2.0; d['snrb_' + c] = 2.0
                d['bdc_' + c] = {'R': 4.0, 'G': 3.9, 'B': 3.7}[c] + rng.normal() * 0.05; d['blac_' + c] = -5.0
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_fit_refuses_held_out_person():
    from oxydrop_ml.photon import twin_inverse as TI
    df = TI._train_rows(TI._labels(_fake_feats()))
    with pytest.raises(AssertionError):
        TI._fit(df, dict(res_l2=0.1, ps=1.0), 0, True, held_out=['1'], steps=2, snaps=(2,))


def test_fit_predict_runs_and_ignores_test_labels():
    """A fit sees only its training rows: scrambling the labels of the person it does not train on changes nothing it predicts."""
    from oxydrop_ml.photon import twin_inverse as TI
    feats = TI._labels(_fake_feats())
    tr = TI._train_rows(feats[feats.sid != '2'])
    te = feats[(feats.sid == '2') & (feats.t >= 60)]
    prep, net, snaps = TI._fit(tr, dict(res_l2=0.1, ps=1.0), 0, True, held_out=['2'], steps=20, snaps=(20,))
    p1 = TI._predict(prep, net, snaps[-1][1], te)
    te2 = te.copy(); te2['y_spo2'] = te2.y_spo2.values[::-1]; te2['y_lab'] = te2.y_spo2
    p2 = TI._predict(prep, net, snaps[-1][1], te2)
    for k in ('spo2', 'sd_lap'):
        assert np.isfinite(p1[k]).all()
        np.testing.assert_array_equal(p1[k], p2[k])


def test_perm_null_permutes_within_each_recording():
    from oxydrop_ml.photon import twin_inverse as TI
    f = _fake_feats(n_people=2, n_t=80)
    d = TI._labels(f, null='perm')
    for rec, g in d.groupby('rec'):
        np.testing.assert_array_equal(np.sort(g.y_lab.values), np.sort(g.y_spo2.values))
    assert (d.y_lab.values != d.y_spo2.values).any()


def test_red_snr_db_recovers_known_snr():
    from oxydrop_ml.photon import twin_inverse as TI
    rng = np.random.default_rng(0)
    t = np.arange(30 * 60) / 30.0
    for sd, tol in ((0.002, 2.0), (0.005, 2.0), (0.02, 2.5)):
        lnr = np.log(100.0) + 0.01 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(size=len(t)) * sd
        rgb = np.stack([np.exp(lnr)] * 3, 1)
        est = TI.red_snr_db(rgb, np.arange(0, 50, 2))
        true = 10 * np.log10((0.01 ** 2 / 2) / (sd ** 2 * (2.8 / 15.0)))
        assert abs(np.median(est) - true) < tol, (sd, np.median(est), true)


def test_gate_frames_pick_paths_by_snr():
    from oxydrop_ml.photon import twin_inverse as TI
    n = 6
    base = pd.DataFrame(dict(rec='1L', t=np.arange(n) * 2 + 60, sid='1', y_spo2=95.0, y=0))
    chain = base.assign(score=0.9, pred=1, answer=True)
    twin = base.assign(score=0.1, pred=0, answer=False, conf=0.3)
    snr = base[['rec', 't']].assign(snr_db=[-20.0, 0.0, 9.9, 10.0, 20.0, 40.0])
    hard, soft = TI.gate_frames(chain, twin, snr)
    np.testing.assert_array_equal(hard.pred.values, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(hard.answer.values, [False, False, False, True, True, True])
    assert soft.score.is_monotonic_increasing and abs(soft.score.values[3] - 0.5) < 1e-9


def test_null_labels_shift_by_200s():
    from oxydrop_ml.photon import twin_inverse as TI
    f = _fake_feats(n_people=1, n_t=150)
    d = TI._labels(f, null=True)
    g = d[d.rec == '0L'].sort_values('t')
    np.testing.assert_array_equal(g.y_lab.values, np.roll(g.y_spo2.values, TI.NULL_SHIFT_WIN))
    assert TI.NULL_SHIFT_WIN * TI.STEP_S >= 200


def test_support_gate_abstains_below_training_snr_and_is_idempotent():
    from oxydrop_ml.photon import twin_inverse as TI
    F0 = _fake_feats()
    F0['snr_R'] = 2.0 + np.random.default_rng(1).normal(size=len(F0)) * 0.1
    F0['snr_G'] = F0['snr_R']
    Fl = F0.copy()
    te = (Fl.sid == '2').values
    Fl.loc[te & (Fl.t >= 100), ['snr_R', 'snr_G']] = -3.0                  # a degraded stretch of the held-out person
    d = Fl[te & (Fl.t >= 60)][['rec', 't', 'sid']].copy()
    d['spo2_hat'] = 80.0; d['spo2_lo'] = 70.0; d['spo2_hi'] = 90.0; d['answer'] = True; d['conf'] = 0.5
    g1 = TI.apply_support_gate(d, Fl, F0)
    assert not g1.answer[g1.t >= 100].any() and g1.answer[g1.t < 100].mean() > 0.9
    g2 = TI.apply_support_gate(g1, Fl, F0)
    np.testing.assert_array_equal(g1.answer.values, g2.answer.values)
    np.testing.assert_allclose(g1.conf.values, g2.conf.values)
