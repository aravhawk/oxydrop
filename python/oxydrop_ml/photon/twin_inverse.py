"""Candidate A, PhotonTwin-Inverse: amortised inversion of the differentiable band-integrated diffusion optics model.

    cd python && python -m oxydrop_ml.photon.twin_inverse            # all models, all levels (writes results/photon/scores/twin_*)
    cd python && python -m oxydrop_ml.photon.twin_inverse --quick    # smoke run (1 outer fold, few steps), writes nothing

WHAT IS INFERRED
Every 10 s window is explained as a CHANGE of the physical state of the finger relative to the same recording's label-free first minute:
    z = (d_sat, d_ln_fb, d_ln_alpha, d_ln_rho)        saturation change, blood-volume change, pulsatile-swing change, path/geometry change
and the recording itself has a geometry latent rho_rec (source-to-detector separation, mm) inferred, without labels, from the colour of its first
minute (blue dies first as rho grows). rho is never assumed: training and test recordings both get their own inferred rho_rec.
Only CHANGES are inverted, so the unknown white balance, exposure and lens gain of the camera cancel (the red-team's "camera colour mismatch" risk).

THE MODEL (`TwinNet`)
  observables y [6]  = window minus first-minute: d ln DC_R,G,B and d ln |AC|_R,G,B   (oxydrop.features.win_feats dc_c and pi_c, as in the chain)
  encoder            = linear map of the six optical changes y (winsorised to the training range) -> q(z) = N(mu, s^2) (amortised posterior)
  decoder            = PhotonTwin (python/oxydrop_ml/optics.py, Farrell diffusion, band-integrated camera channels) rendered at the window state
                       (sat_b + d_sat, fb e^d_ln_fb, fm, rho_rec e^d_ln_rho, alpha e^d_ln_alpha) minus the render at the first-minute state; DC changes are
                       multiplied by a learned per-channel tissue-light fraction eta_c (light that never crossed the pulsing bed dilutes DC changes)
  noise              = heteroscedastic: var_k = e^{2 a_k} + e^{2 b_k} (e^{-snr_w} + e^{-snr_b}) per observable (weak light -> wide likelihood)
  loss               = Gaussian NLL of y under the render + KL(q || physics prior) + Huber(sat vs reference SpO2) + BCE(P(SpO2 < 90)) + L2 on the residual
  residual (optional)= linear head on the NON-optical physiology deltas (dHR, RMSSD, SDNN, two respiratory rates, beat CV) in SpO2 points, L2-penalised;
                       it adds to the physics saturation and never enters the decoder. Its share of the prediction is reported.
  learned physics    = the PhotonTwin scalar corrections of the published spectra (k_musp, k_mel, k_base, k_blood) and eta_c, shared by everyone in the fold.
POSTERIOR WIDTH AND ABSTENTION
  The width of the saturation posterior is a Laplace (Gauss-Newton) approximation of the PHYSICS posterior at the encoder mean: J = d render / d z by central
  differences, precision = J' diag(1/var) J + prior precision, sd_sat = sqrt(cov[0, 0]). As the light gets weaker the SNR falls and var grows, so the
  posterior relaxes toward the prior. Predictive: SpO2 ~ N(spo2_hat, (kappa 100 sd_sat)^2 + tau^2), score = P(SpO2 < 90).
  kappa and tau are calibrated on the inner out-of-fold predictions of the training people. conf = 1 - min(1, width / |spo2_hat - 90|) (the posterior width
  normalised by the distance to the decision threshold); answer = conf >= c*, c* chosen on the inner out-of-fold windows (answered accuracy and balanced
  accuracy >= 0.90, else the best answered balanced accuracy at coverage >= 0.20), the same rule as baselines.common.choose_band, AND the window inside
  the input support (red and green pulse SNR >= the 1 % quantile of the fold's clean training windows; `apply_support_gate`).
  MEASURED (results/photon/twin_inverse_log.md): the calibration dropped the Laplace width (kappa = 0) in 4 of 6 folds and the width hardly grows with
  the stress level, so abstention under weak light comes from the input-support gate, not from the width.
VERSIONS (disclosed in the log): v1 also fed the raw SNR to the encoder (files twin_inverse_v1__*; it extrapolated into confident errors under
  stress); v2 (this code) keeps SNR out of the encoder and winsorises its inputs; the input-support gate was added after seeing v2's flat coverage.
PROTOCOL
  Outer: leave one PERSON out (both hands). Inner: person-grouped folds over the 5 training people (each training person held out once). Every choice
  (residual L2, prior width, epochs via a mid-training snapshot, kappa, tau, c*) uses inner out-of-fold predictions of training people only; every scaler,
  normaliser, prior centre (sat_b of the test recording = mean first-minute label of the training recordings), colour offset of the rho_rec inversion and
  every learned parameter is fitted on the training people of the fold. `_fit` asserts that the held-out person is not among its rows.
  Trained on CLEAN training people only (no augmentation); the held-out person is degraded by stress.degrade (seed STRESS_SEED, as the baselines) and
  scored with the SAME fitted models at every level.
MODELS WRITTEN (results/photon/scores/<model>__L<level>.csv, out-of-fold, CONTRACT.md columns + spo2_hat, spo2_lo, spo2_hi, conf)
  twin_inverse            3-seed ensemble (mixture of the 3 posteriors) of TwinNet with the residual head            levels 0-5 and X1
  twin_inverse_seed<k>    the single seeds at level 0 (the winner rule uses the mean over seeds)
  twin_inverse_noresid    ablation without the residual head (physics path only), 3-seed ensemble                  levels 0-5 and X1
  twin_inverse_null       null control: the whole twin_inverse pipeline with every TRAINING recording's label trace circularly shifted by >= 200 s
                          (seed 0); must score ~0.50 TM                                                                level 0
  twin_chain              logistic chain v1 (shipped C 0.01 / 0.1, p > 0.5) with 2 extra fusion inputs: the label-free Gauss-Newton inversion of the
                          first-minute changes into d_sat and d_ln_rho at the nominal finger (fixed published physics, nothing fitted)  levels 0-5, X1
  twin_chain_nestedC      the same with the fusion C chosen by inner CV                                                levels 0-5, X1
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import CACHE, FS
from .. import metrics as M
from .. import optics as OP
from .. import splits as S
from . import stress as ST

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
CDIR = Path(CACHE) / 'photon_twin'
WINDOWS = REPO / 'results' / 'scores' / 'windows.csv'

STRESS_SEED = 0                       # the degradation seed used by baselines_stress.py, so every model sees the same degraded recordings
LEVEL_SET = list(ST.LEVELS) + ['X1']
WIN_S, STEP_S, CAL_S = 10, 2, 60
N_WORKERS = 3                         # 3 processes x 1 thread (shared box: 3 threads per job)
SEEDS = (0, 1, 2)

FUSION = ['hr', 'rmssd', 'sdnn', 'rr_amp', 'rr_ibi', 'pi_R', 'pi_G', 'pi_B', 'beat_cv', 'ror_RG', 'ror_RB', 'ror_GB',
          'sl_RG', 'sl_RB', 'sl30_RG', 'ror30_RG', 'dc_R', 'dc_G', 'dc_B']          # = oxydrop.chain.FUSION (asserted in load_level)
I_OBS = [16, 17, 18, 5, 6, 7]         # d ln DC_R,G,B and d ln|AC|_R,G,B (pi_c = ln std of the band-passed ln RGB)
I_RES = [0, 1, 2, 3, 4, 8]            # non-optical physiology: dHR, RMSSD, SDNN, rr_amp, rr_ibi, beat_cv
OBS_NAMES = ['d_dc_R', 'd_dc_G', 'd_dc_B', 'd_lac_R', 'd_lac_G', 'd_lac_B']
LAT_NAMES = ['d_sat', 'd_ln_fb', 'd_ln_alpha', 'd_ln_rho']

NOM = dict(sat=0.975, fb=0.02, fm=0.03, alpha=0.008)     # nominal first-minute finger (optics.PRIOR_MU; stress.NOMINAL)
RHO_PRIOR = (np.log(ST.RHO0), 0.5)                      # ln rho_rec prior (mean ln 3 mm as stress.RHO0 / optics.PRIOR_MU, sd 0.5)
P_SD = np.array([0.15, 0.30, 0.50, 0.15])               # prior sd of z (physics prior of a change within one recording)
GN_SIG = np.array([0.03, 0.03, 0.03, 0.20, 0.20, 0.20])  # fixed a-priori noise sd of y for the label-free inversion used by twin_chain
OPT_CFG = dict(learn_bands=False, lam_max=720.0, n_quad=25)   # phone band only (IR-cut ~700 nm), 12.5 nm quadrature

# training schedule (fixed, stated); epochs are selected between the two snapshots by inner CV
BATCH, STEPS, SNAPS, LR = 384, 300, (150, 225, 300), 6e-3
W_HUB, W_BCE, BETA = 1.0, 1.0, 1.0
LR_RES = 0.05                          # the linear residual head converges within the short schedule; L2 does the restraining
ENC_HIDDEN = 0                         # 0 = linear amortised encoder (5 training people: an MLP memorises people)
# v2 (17:25 CT, after the v1 run, disclosed in the log): the encoder sees ONLY the six optical changes, winsorised to the training range; the window
# SNR enters the model only through the likelihood variance (and so the Laplace width). v1 also fed the raw SNR values to the linear encoder, which
# extrapolated on degraded recordings (SNR far below anything in training) into confident, wrong saturations. Set ENC_SNR = True, WINSOR = 0 for v1.
ENC_SNR = False
WINSOR = 0.005                         # clip encoder and residual inputs to the [0.5, 99.5] % range of the fold's training rows
WD = 1e-3
GRID_RES = [dict(res_l2=0.03, ps=1.0), dict(res_l2=0.3, ps=1.0)]
GRID_NORES = [dict(res_l2=0.0, ps=1.0), dict(res_l2=0.0, ps=2.0)]
KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TAUS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
CONF_GRID = np.round(np.linspace(0.0, 0.95, 39), 4)
NULL_SHIFT_WIN = 100                  # 100 windows x 2 s = 200 s circular shift of each training recording's label trace


def _f64(x):
    return torch.as_tensor(np.array(x, np.float64), dtype=torch.float64)


# ================================================================================================ features (raw RGB -> per-window table)
def window_snr(ac, starts):
    """Per-window, per-channel log SNR of the band-passed log signal: power within +-0.2 Hz of the pulse fundamental and its 2nd harmonic over the
    rest of the 0.7-4 Hz band (Hann-windowed FFT of the same 10 s the features use). ac [N, 3]; starts [n] s. Returns [n, 3]."""
    L = WIN_S * FS
    idx = (np.asarray(starts) * FS)[:, None] + np.arange(L)[None]
    seg = ac[idx]                                                           # [n, L, 3]
    seg = seg - seg.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(seg * np.hanning(L)[None, :, None], axis=1)) ** 2  # [n, F, 3]
    f = np.fft.rfftfreq(L, 1.0 / FS)
    band = (f >= 0.7) & (f <= 4.0)
    cand = (f >= 0.7) & (f <= 3.5)
    Pn = P / (P[:, band].sum(1, keepdims=True) + 1e-30)
    f0 = f[cand][np.argmax(Pn[:, cand].sum(2), axis=1)]                     # [n] fundamental (channel-pooled)
    sig = ((np.abs(f[None] - f0[:, None]) <= 0.2) | (np.abs(f[None] - 2 * f0[:, None]) <= 0.2)) & band[None]
    ps = (P * sig[:, :, None]).sum(1)
    pn = (P * (band[None] & ~sig)[:, :, None]).sum(1)
    return np.log((ps + 1e-30) / (pn + 1e-30))


def rec_features(rgb):
    """All windows (10 s, 2 s hop) of one recording: chain feature values A [n, 19] (oxydrop.features.win_feats, causal forward fill), their
    first-minute baseline (median over windows inside the first 60 s), deltas D = A - base (chain.deltas convention), window log-SNR [n, 3] and its
    first-minute median. No labels."""
    from oxydrop.features import rec_feats, win_feats
    rgb = np.clip(np.asarray(rgb, np.float64), 0.0, None)
    lg, ac = rec_feats(rgb)
    n = len(rgb) // FS
    starts = np.arange(0, n - WIN_S + 1, STEP_S)
    A = np.array([[win_feats(lg, ac, (s + WIN_S) * FS).get(c, np.nan) for c in FUSION] for s in starts], float)
    for j in range(A.shape[1]):
        col = A[:, j]
        for i in range(1, len(col)):
            if np.isnan(col[i]):
                col[i] = col[i - 1]
    inb = starts + WIN_S <= CAL_S
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = np.nanmedian(A[inb], 0)
    D = np.where(np.isnan(A), base, A) - base
    snr = window_snr(ac, starts)
    return dict(t=starts, A=A, D=np.nan_to_num(D), base=base, snr=snr, snr_b=np.median(snr[inb], 0))


def _recordings():
    from ..data import hoffman as DH          # puts python/ on sys.path for the legacy `oxydrop` package
    from oxydrop import hoffman
    return hoffman.load(DH.ROOT)


def _degrade(rgb, level):
    if level == 0:
        return np.asarray(rgb, np.float64)
    if isinstance(level, str):
        return ST.degrade_x(rgb, level, STRESS_SEED)
    return ST.degrade(rgb, level, STRESS_SEED)


def _feat_job(args):
    rec, sid, rgb, level = args
    torch.set_num_threads(1)
    f = rec_features(_degrade(rgb, level))
    return rec, sid, level, f


def build_levels(levels=LEVEL_SET, log=print):
    """Feature tables of every recording at every level, cached in CDIR (npz per level). Returns dict level -> DataFrame."""
    CDIR.mkdir(parents=True, exist_ok=True)
    recs = None
    out = {}
    todo = []
    for lv in levels:
        p = CDIR / f'feats_L{lv}.pkl'
        if p.exists():
            out[lv] = pd.read_pickle(p)
        else:
            todo.append(lv)
    if todo:
        recs = _recordings()
        jobs = [(r['rec'], r['sid'], r['rgb'], lv) for lv in todo for r in recs]
        t0 = time.time()
        with ProcessPoolExecutor(N_WORKERS) as ex:
            res = list(ex.map(_feat_job, jobs))
        ymap = {r['rec']: r['y'] for r in recs}
        for lv in todo:
            parts = []
            for rec, sid, l2, f in res:
                if l2 != lv:
                    continue
                y = ymap[rec]
                n = len(f['t'])
                d = pd.DataFrame(dict(rec=rec, sid=str(sid), t=f['t'].astype(int), y_spo2=[y.get(int(t), np.nan) for t in f['t']]))
                for j, c in enumerate(FUSION):
                    d['D_' + c] = f['D'][:, j]
                for j, c in enumerate('RGB'):
                    d['snr_' + c] = f['snr'][:, j]; d['snrb_' + c] = f['snr_b'][j]
                    d['bdc_' + c] = f['base'][16 + j]; d['blac_' + c] = f['base'][5 + j]
                d['A_dc_R'] = f['A'][:, 16]; d['A_dc_G'] = f['A'][:, 17]; d['A_dc_B'] = f['A'][:, 18]
                assert len(d) == n
                parts.append(d)
            df = pd.concat(parts, ignore_index=True)
            df.to_pickle(CDIR / f'feats_L{lv}.pkl')
            out[lv] = df
        log(f'features for levels {todo} built in {time.time() - t0:.0f}s')
    return out


# ================================================================================================ physics decoder (shared)
class Decoder(nn.Module):
    """y_hat [B, 6] = render(window state) - render(first-minute state), PhotonTwin in float64."""

    def __init__(self, learn=True):
        super().__init__()
        self.twin = OP.PhotonTwin(OP.OpticsCfg(learn_scales=learn, **OPT_CFG))
        self.eta_raw = nn.Parameter(torch.full((3,), 2.0, dtype=torch.float64), requires_grad=learn)   # sigmoid(2) = 0.88

    def eta(self):
        return torch.sigmoid(self.eta_raw) if self.eta_raw.requires_grad else torch.ones(3, dtype=torch.float64)

    def _render(self, sat, fb, rho, alpha):
        B = sat.shape[0]
        fm = torch.full((B,), NOM['fm'], dtype=torch.float64)
        o = self.twin.render(sat, fb, fm, rho, alpha, torch.zeros(B, dtype=torch.long))
        return o['ln_dc'], o['ln_amp']

    def base(self, sat_b, rho_b):
        B = sat_b.shape[0]
        return self._render(sat_b, torch.full((B,), NOM['fb'], dtype=torch.float64), rho_b, torch.full((B,), NOM['alpha'], dtype=torch.float64))

    def forward(self, z, sat_b, rho_b, base=None):
        dc_b, amp_b = self.base(sat_b, rho_b) if base is None else base
        sat = torch.clamp(sat_b + z[:, 0], 0.30, 1.0)
        dc, amp = self._render(sat, NOM['fb'] * torch.exp(z[:, 1]), rho_b * torch.exp(z[:, 3]), NOM['alpha'] * torch.exp(z[:, 2]))
        return torch.cat([self.eta()[None] * (dc - dc_b), amp - amp_b], 1)


def jac(dec, z, sat_b, rho_b, h=1e-4):
    """Central-difference Jacobian d y_hat / d z [B, 6, 4] (float64)."""
    base = dec.base(sat_b, rho_b)
    cols = []
    for k in range(z.shape[1]):
        e = torch.zeros_like(z); e[:, k] = h
        cols.append((dec(z + e, sat_b, rho_b, base) - dec(z - e, sat_b, rho_b, base)) / (2 * h))
    return torch.stack(cols, 2)


def laplace_cov(J, var, prior_sd):
    """Gauss-Newton posterior covariance [B, 4, 4] = (J' diag(1/var) J + diag(1/prior_sd^2))^-1."""
    JtW = J.transpose(1, 2) / var[:, None, :]
    H = JtW @ J + torch.diag_embed(torch.as_tensor(1.0 / prior_sd ** 2, dtype=torch.float64).expand(J.shape[0], -1))
    return torch.linalg.inv(H)


@torch.no_grad()
def gn_invert(y, var, sat_b, rho_b, n_iter=10, dec=None):
    """Label-free MAP inversion of observed first-minute changes y [B, 6] (noise variance var [B, 6]) into z [B, 4] with the FIXED published physics
    (no learned scalars, eta = 1) and the prior P_SD. Damped Gauss-Newton. Returns (z [B, 4], posterior sd [B, 4])."""
    dec = dec or Decoder(learn=False)
    y, var = _f64(y), _f64(var)
    sat_b, rho_b = _f64(sat_b), _f64(rho_b)
    z = torch.zeros(y.shape[0], 4, dtype=torch.float64)
    pinv = torch.as_tensor(1.0 / P_SD ** 2, dtype=torch.float64)
    base = dec.base(sat_b, rho_b)
    for _ in range(n_iter):
        f0 = dec(z, sat_b, rho_b, base)
        J = jac(dec, z, sat_b, rho_b)
        JtW = J.transpose(1, 2) / var[:, None, :]
        H = JtW @ J + torch.diag_embed(pinv.expand(len(z), -1)) + 1e-6 * torch.eye(4, dtype=torch.float64)
        g = (JtW @ (y - f0)[:, :, None])[:, :, 0] - pinv * z
        dz = torch.linalg.solve(H, g[:, :, None])[:, :, 0]
        z = z + torch.clamp(dz, -0.3, 0.3)
    J = jac(dec, z, sat_b, rho_b)
    cov = laplace_cov(J, var, P_SD)
    return z.numpy(), torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2)).numpy()


# ================================================================================================ geometry: rho of a recording from its first-minute colour
_RHO_GRID = np.exp(np.linspace(np.log(0.8), np.log(20.0), 161))


@torch.no_grad()
def _chroma_curve():
    """Twin chromaticity (ln I_R - ln I_G, ln I_R - ln I_B) of the nominal finger over _RHO_GRID, fixed published physics. [n_grid, 2]."""
    dec = Decoder(learn=False)
    n = len(_RHO_GRID)
    dc, _ = dec.base(torch.full((n,), NOM['sat'], dtype=torch.float64), _f64(_RHO_GRID))
    dc = dc.numpy()
    return np.stack([dc[:, 0] - dc[:, 1], dc[:, 0] - dc[:, 2]], 1)


def rec_chroma(df):
    """[n_rec] first-minute measured chromaticity (ln-DC R-G, R-B) per recording of a feature table (one row per recording)."""
    g = df.groupby('rec', sort=False).first()
    return g.index.values, np.stack([g.bdc_R - g.bdc_G, g.bdc_R - g.bdc_B], 1)


class RhoFit:
    """Label-free geometry inversion. offset = mean (over the fold's TRAINING recordings) of measured minus twin chromaticity at the prior rho;
    rho_rec = argmax over the grid of the Gaussian chromaticity likelihood (sd SIG) times the ln-rho prior."""
    SIG = 0.15

    def __init__(self, chroma_train):
        self.curve = _chroma_curve()
        i0 = int(np.argmin(np.abs(np.log(_RHO_GRID) - RHO_PRIOR[0])))
        self.offset = np.mean(chroma_train, 0) - self.curve[i0]

    def rho(self, chroma):
        c = np.atleast_2d(chroma) - self.offset
        nll = ((c[:, None, :] - self.curve[None]) ** 2).sum(2) / (2 * self.SIG ** 2) + (np.log(_RHO_GRID)[None] - RHO_PRIOR[0]) ** 2 / (2 * RHO_PRIOR[1] ** 2)
        return _RHO_GRID[np.argmin(nll, 1)]


# ================================================================================================ the amortised model
class TwinNet(nn.Module):
    def __init__(self, n_in, n_res, use_res, ps):
        super().__init__()
        self.dec = Decoder(learn=True)
        h = ENC_HIDDEN
        self.enc = (nn.Sequential(nn.Linear(n_in, 8)) if h == 0 else nn.Sequential(nn.Linear(n_in, h), nn.Tanh(), nn.Linear(h, 8))).double()
        with torch.no_grad():
            self.enc[-1].weight.mul_(0.1); self.enc[-1].bias.zero_()
            self.enc[-1].bias[4:] = torch.as_tensor(np.log(P_SD * ps * 0.5), dtype=torch.float64)
        self.a = nn.Parameter(torch.full((6,), np.log(0.03), dtype=torch.float64))
        self.b = nn.Parameter(torch.full((6,), np.log(0.03), dtype=torch.float64))
        self.use_res = use_res
        self.w_res = nn.Parameter(torch.zeros(n_res, dtype=torch.float64), requires_grad=use_res)
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.02), dtype=torch.float64))
        self.prior_sd = P_SD * ps

    def var(self, snr_w, snr_b):
        e = torch.exp(-snr_w) + torch.exp(-snr_b)                               # [B, 3]
        e6 = torch.cat([e, e], 1)
        return torch.exp(2 * self.a)[None] + torch.exp(2 * self.b)[None] * e6

    def q(self, x):
        o = self.enc(x)
        return o[:, :4], torch.clamp(o[:, 4:], -8.0, 2.0)

    def resid(self, xr):
        return (xr @ self.w_res) / 100.0 if self.use_res else torch.zeros(xr.shape[0], dtype=torch.float64)


def _huber(x, d):
    a = x.abs()
    return torch.where(a <= d, 0.5 * x * x, d * (a - 0.5 * d))


class Prep:
    """Everything fitted on the training rows of one fit: input scalers, winsorisation bounds, the rho inversion offset, sat_b of unseen recordings."""

    def __init__(self, df_tr):
        self.cols_in = [f'D_{FUSION[i]}' for i in I_OBS] + (['snr_R', 'snr_G', 'snr_B', 'snrb_R', 'snrb_G', 'snrb_B'] if ENC_SNR else [])
        self.cols_res = [f'D_{FUSION[i]}' for i in I_RES]
        X = df_tr[self.cols_in].values; R = df_tr[self.cols_res].values
        self.mu_in, self.sd_in = X.mean(0), X.std(0) + 1e-6
        self.mu_res, self.sd_res = R.mean(0), R.std(0) + 1e-6
        q = (WINSOR, 1 - WINSOR) if WINSOR > 0 else (0.0, 1.0)
        self.lo_in, self.hi_in = (np.quantile(X, q[0], 0), np.quantile(X, q[1], 0)) if WINSOR > 0 else (np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf))
        self.lo_res, self.hi_res = (np.quantile(R, q[0], 0), np.quantile(R, q[1], 0)) if WINSOR > 0 else (np.full(R.shape[1], -np.inf), np.full(R.shape[1], np.inf))
        recs, ch = rec_chroma(df_tr)
        self.rhofit = RhoFit(ch)
        satb = df_tr.groupby('rec').sat_b_lab.first()
        self.sat_b_test = float(np.nanmean(satb.values)) if satb.notna().any() else NOM['sat']

    def tensors(self, df, train):
        x = (np.clip(df[self.cols_in].values, self.lo_in, self.hi_in) - self.mu_in) / self.sd_in
        xr = (np.clip(df[self.cols_res].values, self.lo_res, self.hi_res) - self.mu_res) / self.sd_res
        recs, ch = rec_chroma(df)
        rho_map = dict(zip(recs, self.rhofit.rho(ch)))
        rho = df.rec.map(rho_map).values.astype(float)
        if train:
            sat_b = df.sat_b_lab.values.astype(float)
        else:
            sat_b = np.full(len(df), self.sat_b_test)
        y = df[[f'D_{FUSION[i]}' for i in I_OBS]].values
        snr_w = df[['snr_R', 'snr_G', 'snr_B']].values; snr_b = df[['snrb_R', 'snrb_G', 'snrb_B']].values
        return dict(x=_f64(x), xr=_f64(xr), y=_f64(y), snr_w=_f64(snr_w), snr_b=_f64(snr_b), sat_b=_f64(sat_b), rho=_f64(rho))


def _fit(df_tr, hp, seed, use_res, held_out=(), steps=STEPS, snaps=SNAPS):
    """Train one TwinNet on the rows df_tr (labels y_lab = training-side SpO2, possibly shifted for the null control). Returns [(step, state)]."""
    assert not set(map(str, held_out)) & set(df_tr.sid.astype(str)), 'held-out person among the training rows'
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    prep = Prep(df_tr)
    T = prep.tensors(df_tr, train=True)
    ysat = _f64(df_tr.y_lab.values / 100.0)
    yb = _f64((df_tr.y_lab.values < 90).astype(float))
    n1 = float(yb.sum()); n0 = float(len(yb) - n1)
    wpos, wneg = len(yb) / (2 * max(n1, 1)), len(yb) / (2 * max(n0, 1))
    net = TwinNet(T['x'].shape[1], T['xr'].shape[1], use_res, hp['ps'])
    enc_p = list(net.enc.parameters())
    other = [p for n_, p in net.named_parameters() if p.requires_grad and not n_.startswith('enc.') and n_ != 'w_res']
    groups = [dict(params=enc_p, weight_decay=WD), dict(params=other)] + ([dict(params=[net.w_res], lr=LR_RES)] if use_res else [])
    opt = torch.optim.Adam(groups, lr=LR)
    out = []
    N = len(df_tr)
    psd = _f64(net.prior_sd)
    for step in range(1, steps + 1):
        bi = torch.as_tensor(rng.integers(0, N, min(BATCH, N)))
        x, xr, y = T['x'][bi], T['xr'][bi], T['y'][bi]
        mu, ls = net.q(x)
        s = torch.exp(ls)
        z = mu + s * torch.randn_like(mu)
        yhat = net.dec(z, T['sat_b'][bi], T['rho'][bi])
        var = net.var(T['snr_w'][bi], T['snr_b'][bi])
        nll = 0.5 * (((y - yhat) ** 2) / var + torch.log(var)).sum(1).mean()
        kl = (torch.log(psd[None] / s) + (s ** 2 + mu ** 2) / (2 * psd[None] ** 2) - 0.5).sum(1).mean()
        r = net.resid(xr)
        sat_b = T['sat_b'][bi]
        hub = _huber((sat_b + z[:, 0] + r - ysat[bi]) * 100.0, 2.0).mean()
        m_sat = sat_b + mu[:, 0] + r
        sd = torch.sqrt(s[:, 0] ** 2 + torch.exp(2 * net.log_tau))
        p = torch.special.ndtr((0.90 - m_sat) / sd).clamp(1e-6, 1 - 1e-6)
        yy = yb[bi]
        bce = -(wpos * yy * torch.log(p) + wneg * (1 - yy) * torch.log(1 - p)).mean()
        l2 = hp['res_l2'] * (net.w_res ** 2).sum() if use_res else 0.0
        loss = nll + BETA * kl + W_HUB * hub + W_BCE * bce + l2
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step in snaps:
            out.append((step, {k: v.detach().clone() for k, v in net.state_dict().items()}))
    return prep, net, out


@torch.no_grad()
def _predict(prep, net, state, df):
    """Per-window prediction: spo2_hat (points), Laplace physics sd of sat (fraction), encoder sd, residual part (points), d_ln_rho, rho_rec."""
    net.load_state_dict(state)
    T = prep.tensors(df, train=False)
    mu, ls = net.q(T['x'])
    r = net.resid(T['xr'])
    spo2 = 100.0 * (T['sat_b'] + mu[:, 0] + r)
    J = jac(net.dec, mu, T['sat_b'], T['rho'])
    cov = laplace_cov(J, net.var(T['snr_w'], T['snr_b']), net.prior_sd)
    return dict(spo2=spo2.numpy(), sd_lap=torch.sqrt(cov[:, 0, 0]).numpy(), sd_enc=torch.exp(ls[:, 0]).numpy(), res=100.0 * r.numpy(),
                phys=100.0 * mu[:, 0].numpy(), d_ln_rho=mu[:, 3].numpy(), rho_rec=T['rho'].numpy())


def _params_report(net, state):
    net.load_state_dict(state)
    rep = net.dec.twin.param_report()
    rep.pop('band_centre', None); rep.pop('band_fwhm', None)
    rep['eta'] = [float(v) for v in net.dec.eta().detach()]
    rep['w_res'] = [float(v) for v in net.w_res.detach()]
    rep['tau'] = float(torch.exp(net.log_tau.detach()))
    return rep


# ================================================================================================ nested LOSO driver
def _labels(df, null=False, outer_so=None):
    """Training-side labels y_lab (SpO2) and sat_b_lab (mean first-minute label of the recording, fraction). null=True: each recording's label trace
    (ordered by t) is circularly shifted by NULL_SHIFT_WIN windows (200 s) before anything else; scoring always uses the true y_spo2."""
    df = df.copy()
    df['y_lab'] = df.y_spo2
    if null == 'gperm':                                     # global permutation control: labels shuffled across ALL windows of the outer fold's
        rng = np.random.default_rng(zlib.crc32(str(outer_so).encode()))       # training people (the held-out person's labels are never touched)
        idx = df.index[df.y_spo2.notna() & (df.sid.astype(str) != str(outer_so))]
        df.loc[idx, 'y_lab'] = rng.permutation(df.loc[idx, 'y_spo2'].values)
    elif null == 'perm':                                    # label-permutation control: each recording's labels shuffled among its own windows
        rng = np.random.default_rng(20261115)
        for rec, g in df.groupby('rec', sort=True):
            idx = g.index[g.y_spo2.notna()]
            df.loc[idx, 'y_lab'] = rng.permutation(df.loc[idx, 'y_spo2'].values)
    elif null:
        for rec, g in df.groupby('rec'):
            g = g.sort_values('t')
            v = g.y_spo2.values
            df.loc[g.index, 'y_lab'] = np.roll(v, NULL_SHIFT_WIN)
    b = df[df.t + WIN_S <= CAL_S].groupby('rec').y_lab.mean() / 100.0
    df['sat_b_lab'] = df.rec.map(b)
    return df


def _train_rows(df):
    return df[(df.t >= CAL_S) & df.y_lab.notna()]


def _task(args):
    """One fit (worker process). kind 'inner': fit on train people, predict the inner validation person at level 0 for every snapshot.
    kind 'outer': fit on the 5 training people, predict the held-out person at every level."""
    kind, key, hp, seed, use_res, null, tr_sids, te_sids, levels = args
    torch.set_num_threads(1)
    F = {lv: _labels(pd.read_pickle(CDIR / f'feats_L{lv}.pkl'), null=null, outer_so=key[0]) for lv in levels}
    d0 = F[0]
    tr = _train_rows(d0[d0.sid.isin(tr_sids)])
    prep, net, snaps = _fit(tr, hp, seed, use_res, held_out=te_sids)
    out = {}
    for step, st in snaps:
        for lv in levels:
            d = F[lv]
            te = d[d.sid.isin(te_sids) & (d.t >= CAL_S)]
            p = _predict(prep, net, st, te)
            p.update(rec=te.rec.values, t=te.t.values, sid=te.sid.values)
            out[(step, lv)] = p
    rep = _params_report(net, snaps[-1][1]) if kind == 'outer' else None
    return kind, key, out, rep


def calibrate(spo2, sd_lap, y):
    """(kappa, tau) maximising the class-balanced likelihood of y under score = Phi((90 - spo2) / sqrt((kappa 100 sd)^2 + tau^2)). Inner OOF only."""
    from scipy.special import ndtr
    yb = (y < 90).astype(float)
    w = np.where(yb == 1, len(yb) / (2 * max(yb.sum(), 1)), len(yb) / (2 * max(len(yb) - yb.sum(), 1)))
    best = None
    for k in KAPPAS:
        for t in TAUS:
            p = np.clip(ndtr((90 - spo2) / np.sqrt((k * 100 * sd_lap) ** 2 + t ** 2)), 1e-6, 1 - 1e-6)
            ll = -(w * (yb * np.log(p) + (1 - yb) * np.log(1 - p))).mean()
            if best is None or ll < best[0] - 1e-12:
                best = (ll, k, t)
    return best[1], best[2]


def score_of(spo2, sd_tot):
    from scipy.special import ndtr
    return ndtr((90 - spo2) / sd_tot)


def conf_of(spo2, sd_tot):
    return 1.0 - np.minimum(1.0, sd_tot / np.maximum(np.abs(spo2 - 90.0), 1e-6))


def choose_conf(conf, y, pred, target=0.90, min_cov=0.20):
    """Smallest c on CONF_GRID whose answered inner-OOF windows (conf >= c) reach accuracy and balanced accuracy >= target (>= 20 windows, both
    classes); else the c with the best answered balanced accuracy among those keeping >= min_cov of the windows (baselines.common.choose_band rule)."""
    yb = (y < 90).astype(int)
    best = (-1.0, 0.0)
    for c in CONF_GRID:
        a = conf >= c
        if a.sum() < 20 or len(np.unique(yb[a])) < 2:
            continue
        cm = M.classification_metrics(yb[a], pred[a].astype(float), pred=pred[a])
        if cm['acc'] >= target and cm['bacc'] >= target:
            return float(c)
        if a.mean() >= min_cov and cm['bacc'] > best[0]:
            best = (cm['bacc'], float(c))
    return best[1]


def _tm(t, y_spo2, pred):
    return M.time_matched(t, (np.asarray(y_spo2) < 90).astype(int), pred)['tm_bacc']


def run_twin(F, use_res, null=False, seeds=SEEDS, levels=LEVEL_SET, quick=False, log=print, tag=''):
    """Nested LOSO for one TwinNet variant. Returns (per-level ensemble frames, per-seed level-0 frames, fold log rows, parameter reports)."""
    grid = GRID_RES if use_res else GRID_NORES
    d0 = _labels(F[0], null=null)
    sids = sorted(d0.sid.unique())
    if quick:
        sids_outer = sids[:1]
    else:
        sids_outer = sids
    jobs = []
    for so in sids_outer:
        tr_s = [s for s in sids if s != so]
        for gi, hp in enumerate(grid):
            for si in tr_s:
                inner_tr = [s for s in tr_s if s != si]
                jobs.append(('inner', (so, gi, si), hp, 0, use_res, null, inner_tr, [si], [0]))
    t0 = time.time()
    inner = {}
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for kind, key, out, _ in ex.map(_task, jobs):
            inner[key] = out
    log(f'  {tag}: {len(jobs)} inner fits in {time.time() - t0:.0f}s')
    # selection per outer fold (inner OOF of the training people only)
    choice = {}
    rows_log = []
    for so in sids_outer:
        tr_s = [s for s in sids if s != so]
        best = None
        for gi, hp in enumerate(grid):
            for step in SNAPS:
                P = [inner[(so, gi, si)][(step, 0)] for si in tr_s]
                spo2 = np.concatenate([p['spo2'] for p in P]); t = np.concatenate([p['t'] for p in P])
                key_rows = pd.DataFrame(dict(rec=np.concatenate([p['rec'] for p in P]), t=t))
                d0f = _labels(F[0], null=null, outer_so=so) if null == 'gperm' else d0
                yv = key_rows.merge(d0f[['rec', 't', 'y_lab']], on=['rec', 't'], how='left').y_lab.values      # training-side labels (null: shifted / permuted)
                ok = np.isfinite(yv)
                v = _tm(t[ok], yv[ok], (spo2[ok] < 90).astype(int))
                v = -1.0 if not np.isfinite(v) else v
                if best is None or v > best[0] + 1e-12:
                    sdl = np.concatenate([p['sd_lap'] for p in P])
                    best = (v, gi, step, spo2[ok], sdl[ok], yv[ok])
        v, gi, step, spo2, sdl, yv = best
        kappa, tau = calibrate(spo2, sdl, yv)
        sd_tot = np.sqrt((kappa * 100 * sdl) ** 2 + tau ** 2)
        c_star = choose_conf(conf_of(spo2, sd_tot), yv, (spo2 < 90).astype(int))
        choice[so] = dict(gi=gi, hp=grid[gi], step=step, kappa=kappa, tau=tau, c_star=c_star, inner_tm=v)
        rows_log.append(dict(model=tag, held_out=so, **grid[gi], steps=step, kappa=kappa, tau=tau, c_star=c_star, inner_tm=round(v, 4)))
    # outer fits with the chosen hp, every seed, every level
    jobs = []
    for so in sids_outer:
        tr_s = [s for s in sids if s != so]
        ch = choice[so]
        for sd in seeds:
            jobs.append(('outer', (so, sd), ch['hp'], sd, use_res, null, tr_s, [so], levels if not null else [0]))
    t0 = time.time()
    outer, reps = {}, {}
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for kind, key, out, rep in ex.map(_task, jobs):
            outer[key] = out; reps[key] = rep
    log(f'  {tag}: {len(jobs)} outer fits in {time.time() - t0:.0f}s')
    lv_list = levels if not null else [0]
    ens, per_seed = {}, {sd: [] for sd in seeds}
    for lv in lv_list:
        parts = []
        for so in sids_outer:
            ch = choice[so]
            P = [outer[(so, sd)][(ch['step'], lv)] for sd in seeds]
            mu_s = np.stack([p['spo2'] for p in P]); sdl_s = np.stack([p['sd_lap'] for p in P])
            spo2 = mu_s.mean(0)
            sd_tot = np.sqrt((ch['kappa'] * 100) ** 2 * (sdl_s ** 2).mean(0) + ch['tau'] ** 2 + (mu_s.var(0) if len(seeds) > 1 else 0.0))
            parts.append(_frame(P[0], spo2, sd_tot, ch['c_star'], extra=dict(res=np.mean([p['res'] for p in P], 0), phys=np.mean([p['phys'] for p in P], 0),
                                                                             sd_lap=np.sqrt((sdl_s ** 2).mean(0)), d_ln_rho=np.mean([p['d_ln_rho'] for p in P], 0),
                                                                             rho_rec=P[0]['rho_rec'])))
            if lv == 0:
                for sd, p in zip(seeds, P):
                    st = np.sqrt((ch['kappa'] * 100 * p['sd_lap']) ** 2 + ch['tau'] ** 2)
                    per_seed[sd].append(_frame(p, p['spo2'], st, ch['c_star']))
        ens[lv] = _attach_truth(pd.concat(parts, ignore_index=True), F[0])
    per_seed = {sd: _attach_truth(pd.concat(v, ignore_index=True), F[0]) for sd, v in per_seed.items() if v}
    return ens, per_seed, rows_log, reps


def _frame(p, spo2, sd_tot, c_star, extra=None):
    sc = score_of(spo2, sd_tot)
    cf = conf_of(spo2, sd_tot)
    d = pd.DataFrame(dict(rec=p['rec'], t=p['t'].astype(int), sid=p['sid'], score=sc, pred=(spo2 < 90).astype(int), answer=cf >= c_star,
                          spo2_hat=spo2, spo2_lo=spo2 - 1.96 * sd_tot, spo2_hi=spo2 + 1.96 * sd_tot, conf=cf))
    for k, v in (extra or {}).items():
        d[k] = v
    return d


def _attach_truth(d, F0):
    truth = F0[['rec', 't', 'y_spo2']]
    d = d.merge(truth, on=['rec', 't'], how='left')
    d['y'] = (d.y_spo2 < 90).astype(int)
    return d


# ================================================================================================ twin_chain: chain v1 + label-free inversion features
def chain_inputs(F_lv, sat_b=NOM['sat']):
    """[n, 21] = 19 chain deltas + (d_sat, d_ln_rho) of the label-free Gauss-Newton inversion of the 6 first-minute changes at the nominal finger
    (fixed published physics, fixed noise GN_SIG inflated by the window SNR, rho_b = stress.RHO0). Nothing here is fitted."""
    y = F_lv[[f'D_{FUSION[i]}' for i in I_OBS]].values
    e = np.exp(-F_lv[['snr_R', 'snr_G', 'snr_B']].values) + np.exp(-F_lv[['snrb_R', 'snrb_G', 'snrb_B']].values)
    var = GN_SIG[None] ** 2 * (1.0 + np.concatenate([e, e], 1))
    z, sdz = gn_invert(y, var, np.full(len(y), sat_b), np.full(len(y), ST.RHO0))
    X = np.concatenate([F_lv[[f'D_{c}' for c in FUSION]].values, z[:, [0, 3]]], 1)
    return X, z, sdz


def _lr_fit(X, y, C_hr, C_fus):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ms = []
    for cols, c in (([0], C_hr), (list(range(X.shape[1])), C_fus)):
        sc = StandardScaler().fit(X[:, cols])
        lr = LogisticRegression(C=c, class_weight='balanced', max_iter=3000).fit(sc.transform(X[:, cols]), y)
        ms.append((cols, sc, lr))
    return ms


def _lr_score(ms, X):
    return np.mean([lr.predict_proba(sc.transform(X[:, cols]))[:, 1] for cols, sc, lr in ms], 0)


def run_twin_chain(F, nested_c, levels=LEVEL_SET, log=print):
    from ..baselines.common import choose_band
    Xl, zl = {}, {}
    for lv in levels:
        X, z, sdz = chain_inputs(F[lv])
        Xl[lv], zl[lv] = X, (z, sdz)
    d0 = _labels(F[0])
    ev = (d0.t >= CAL_S).values & d0.y_spo2.notna().values
    sids = sorted(d0.sid.unique())
    grid = [0.001, 0.01, 0.1, 1.0, 10.0] if nested_c else [0.1]
    out = {lv: [] for lv in levels}
    flog = []
    for so in sids:
        tr = np.nonzero(ev & (d0.sid != so).values)[0]
        te = np.nonzero(ev & (d0.sid == so).values)[0]
        S.assert_no_leak(d0.sid.values, tr, te)
        y_tr = (d0.y_lab.values[tr] < 90).astype(int)
        g_tr = d0.sid.values[tr]
        inner = list(S.inner_group_folds(g_tr, k=5, seed=0, balance=y_tr))
        best = None
        for c in grid:
            oof = np.full(len(tr), np.nan)
            for fi in inner:
                ms = _lr_fit(Xl[0][tr[fi.train_idx]], y_tr[fi.train_idx], 0.01, c)
                oof[fi.test_idx] = _lr_score(ms, Xl[0][tr[fi.test_idx]])
            v = M.time_matched(d0.t.values[tr], y_tr, (oof > 0.5).astype(int))['tm_bacc']
            if best is None or v > best[0] + 1e-12:
                best = (v, c, oof)
        v, c, oof = best
        band = choose_band(oof, y_tr, 0.5)
        ms = _lr_fit(Xl[0][tr], y_tr, 0.01, c)
        flog.append(dict(model='twin_chain_nestedC' if nested_c else 'twin_chain', held_out=so, C_hr=0.01, C_fus=c, band=band, inner_tm=round(float(v), 4),
                         coef_d_sat=float(ms[1][2].coef_[0][-2]), coef_d_ln_rho=float(ms[1][2].coef_[0][-1])))
        for lv in levels:
            s = _lr_score(ms, Xl[lv][te])
            z, sdz = zl[lv]
            out[lv].append(pd.DataFrame(dict(rec=F[lv].rec.values[te], t=F[lv].t.values[te].astype(int), sid=F[lv].sid.values[te], score=s,
                                             pred=(s > 0.5).astype(int), answer=np.abs(s - 0.5) >= band, conf=np.abs(s - 0.5),
                                             d_sat=z[te, 0], d_ln_rho=z[te, 3], sd_d_sat=sdz[te, 0])))
    return {lv: _attach_truth(pd.concat(v, ignore_index=True), F[0]) for lv, v in out.items()}, flog


# ================================================================================================ output
COLS = ['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer', 'spo2_hat', 'spo2_lo', 'spo2_hi', 'conf']
SUPPORT_Q = 0.01                      # input-support gate: a window is answerable only if its red AND green pulse SNR >= the 1 % quantile of the fold's training windows


def apply_support_gate(d, F_lv, F0):
    """Input-support abstention (added 17:30 CT after the v2 run showed a flat coverage under stress; disclosed): answer only when the window's red and
    green pulse-band log-SNR are inside the range seen on the CLEAN training people of the fold (>= their SUPPORT_Q quantile). Label-free; the quantiles
    use training people only. `pred`, `score` and the TM metric are unchanged; out-of-support windows get answer = False and conf - 1 (ranked last)."""
    d = d.copy()
    snr = d[['rec', 't']].merge(F_lv[['rec', 't', 'snr_R', 'snr_G']], on=['rec', 't'], how='left')
    ok = np.ones(len(d), bool)
    for sid in d.sid.unique():
        tr = F0[(F0.sid != str(sid)) & (F0.t >= CAL_S)]
        m = (d.sid.astype(str) == str(sid)).values
        for c in ('snr_R', 'snr_G'):
            ok[m] &= snr[c].values[m] >= np.quantile(tr[c].values, SUPPORT_Q)
    a_post = (d['answer_post'] if 'answer_post' in d else d['answer']).astype(str).str.lower().isin(['true', '1', '1.0']).values
    conf_post = conf_of(d.spo2_hat.values, (d.spo2_hi.values - d.spo2_lo.values) / (2 * 1.96))      # idempotent: recomputed from the posterior
    d['answer_post'] = a_post
    d['in_support'] = ok
    d['answer'] = a_post & ok
    d['conf'] = np.where(ok, conf_post, conf_post - 1.0)
    return d


def write_scores(d, model, level):
    SCORES.mkdir(parents=True, exist_ok=True)
    d = d[d.y_spo2.notna()]                                  # only the labelled windows of results/scores/windows.csv (5,644 at t >= 60 s)
    cols = [c for c in COLS + ['answer_post', 'in_support'] if c in d.columns]
    p = SCORES / f'{model}__{"L" + str(level) if not isinstance(level, str) else level}.csv'
    d.sort_values(['rec', 't'])[cols].to_csv(p, index=False, float_format='%.6g')
    return p


def quick_metrics(d):
    d = d[d.y_spo2.notna()]
    tm = M.time_matched(d.t.values, d.y.values, d.pred.values)
    per = {s: M.time_matched(g.t.values, g.y.values, g.pred.values)['tm_bacc'] for s, g in d.groupby('sid')}
    a = d.answer.values.astype(bool)
    return dict(tm=tm['tm_bacc'], per=per, cov=float(a.mean()), acc_ans=float((d.pred.values[a] == d.y.values[a]).mean()) if a.any() else np.nan,
                mae=float(np.nanmean(np.abs(d.spo2_hat - d.y_spo2))) if 'spo2_hat' in d else np.nan)


def write_log(summary, path=RES / 'twin_inverse_log.md', extra_md=''):
    """results/photon/twin_inverse_log.md: quick numbers, per-person TM, nested choices per fold, seeds, learned physics, inferred rho vs level."""
    f3 = lambda v: '' if v is None or not np.isfinite(v) else f'{v:.3f}'
    lv_name = lambda lv: f'L{lv}' if not isinstance(lv, str) else lv
    L = ['# PhotonTwin-Inverse (candidate A) run log', '',
         'Generated by `python -m oxydrop_ml.photon.twin_inverse` (the module docstring states the method). Hoffman, leave-one-person-out (both hands '
         'together), 5,644 evaluated windows (t >= 60 s); every selection is nested on the 5 training people; the held-out person is degraded by '
         f'`stress.degrade(rgb, level, seed={STRESS_SEED})` / `stress.degrade_x(rgb, "X1", {STRESS_SEED})` (the seed of baselines_stress.py) and scored by the '
         'models fitted once on clean training people. TM = time-matched balanced accuracy (metrics.time_matched) of the forced decision `pred`. '
         'Stress levels are simulations.', '']
    models = [m for m in ('twin_inverse', 'twin_inverse_noresid', 'twin_chain', 'twin_chain_nestedC', 'twin_gate', 'twin_gate_soft', 'twin_inverse_null',
                          'twin_inverse_noresid_null', 'twin_gate_null', 'twin_gate_soft_null', 'chain_v1_refit_null', 'twin_inverse_noresid_perm_null',
                          'twin_gate_perm_null', 'twin_gate_soft_perm_null', 'chain_v1_refit_perm_null', 'twin_inverse_noresid_gperm_null',
                          'twin_gate_gperm_null', 'twin_gate_soft_gperm_null', 'chain_v1_refit_gperm_null') if (m, 0) in summary]
    L += ['## Quick numbers (computed here; results/photon/TABLE.md has the evalkit CIs and paired differences)', '',
          '| model | metric | ' + ' | '.join(lv_name(lv) for lv in LEVEL_SET) + ' |', '|---|---|' + '---|' * len(LEVEL_SET)]
    for m in models:
        for key, nm, fmt in (('tm', 'TM bal. acc.', f3), ('cov', 'coverage of answer', lambda v: f'{v:.2f}'), ('acc_ans', 'accuracy answered', f3),
                             ('mae', 'SpO2 MAE', lambda v: '' if not np.isfinite(v) else f'{v:.2f}')):
            if key == 'mae' and not m.startswith('twin_inverse'):
                continue
            L.append(f'| {m} | {nm} | ' + ' | '.join(fmt(summary[(m, lv)][key]) if (m, lv) in summary else '' for lv in LEVEL_SET) + ' |')
    L.append('')
    for base in ('twin_inverse', 'twin_inverse_noresid'):
        ks = sorted(k[0] for k in summary if isinstance(k[0], str) and k[0].startswith(base + '_seed') and k[1] == 0)
        if ks:
            v = [summary[(k, 0)]['tm'] for k in ks]
            L += [f'{base} single seeds at L0: ' + ', '.join(f'{k}: {f3(x)}' for k, x in zip(ks, v)) + f'; **mean of the 3 seeds {f3(float(np.mean(v)))}** '
                  f'(the `{base}` file is the 3-seed posterior mixture).', '']
    people = sorted(summary[(models[0], 0)]['per'])
    L += ['## Per-person TM (held-out person, both hands; NaN = no 100 s bin with both classes)', '', '| model | level | ' + ' | '.join(people) + ' |',
          '|---|---|' + '---|' * len(people)]
    for m in models:
        for lv in LEVEL_SET:
            if (m, lv) in summary:
                per = summary[(m, lv)]['per']
                L.append(f'| {m} | {lv_name(lv)} | ' + ' | '.join(f3(per.get(s, np.nan)) for s in people) + ' |')
    L += ['', '## Nested choices per outer fold (inner person-grouped CV on the 5 training people only)', '']
    for m in models:
        fl = summary.get((m, 'folds'))
        if not fl:
            continue
        cols = list(fl[0].keys())
        L += [f'### {m}', '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
        for r in fl:
            L.append('| ' + ' | '.join(f'{r[c]:.4g}' if isinstance(r[c], float) else str(r[c]) for c in cols) + ' |')
        L.append('')
    for m in ('twin_inverse', 'twin_inverse_noresid'):
        pr = summary.get((m, 'params'))
        if pr:
            L += [f'## Learned physics of {m} (outer fits; k_* multiply the published spectra of spectra.py; eta = tissue-light fraction of the DC changes R, G, B)', '',
                  '| fit | k_musp | k_mel | k_base | k_blood | eta R, G, B | tau | residual weights, SpO2 points per SD (dHR, RMSSD, SDNN, rr_amp, rr_ibi, beat_cv) |',
                  '|---|---|---|---|---|---|---|---|']
            for k in sorted(pr):
                r = pr[k]
                w = ', '.join(f'{v:+.2f}' for v in r['w_res']) if any(abs(v) > 0 for v in r['w_res']) else 'none'
                L.append(f'| {k} | {r["k_musp"]:.3f} | {r["k_mel"]:.3f} | {r["k_base"]:.3f} | {r["k_blood"]:.3f} | ' + ', '.join(f'{v:.2f}' for v in r['eta']) +
                         f' | {r["tau"]:.4f} | {w} |')
            L.append('')
        g = summary.get((m, 'geom'))
        if g:
            L += [f'## Geometry awareness and posterior width vs level ({m})', '',
                  'rho_rec: label-free inversion of each recording\'s first-minute colour (ln-DC R-G, R-B) through the fixed published-physics renderer '
                  '(colour offset fitted on the training recordings of the fold, prior ln rho ~ N(ln 3 mm, 0.5)). The stress module renders its levels with the '
                  'same twin at rho0 + 0.25 d_rho, so a rising rho_rec is a consistency check of the inversion, not independent evidence. d_ln_rho: encoder '
                  'change of ln rho inside a recording (the whole recording is degraded, so ~0 is expected). sd_sat: Laplace posterior sd of saturation. '
                  'Residual share = SD(residual part) / (SD(residual part) + SD(physics part)) of spo2_hat.', '',
                  '| level | stress rho (mm) | rho_rec median [min, max], 12 recordings | mean d_ln_rho | median sd_sat (points) | median predictive sd (points) | coverage | residual share |',
                  '|---|---|---|---|---|---|---|---|']
            for r in g:
                L.append(f'| {lv_name(r["level"])} | {r["rho_stress"]:.2f} | {r["rho_med"]:.2f} [{r["rho_min"]:.2f}, {r["rho_max"]:.2f}] | {r["dlr"]:+.3f} | '
                         f'{r["sd_sat"]:.2f} | {r["sd_pred"]:.2f} | {r["cov"]:.2f} | {r["res_share"]:.2f} |')
            L.append('')
    if not extra_md:                                        # keep the evalkit rows and disclosures of the last run when the log is regenerated
        extra_md = '\n'.join(open(CDIR / n).read() for n in ('evalkit_md.txt', 'disclosures_md.txt', 'twin_gate_prereg_md.txt', 'twin_gate_md.txt') if (CDIR / n).exists())
    if extra_md:
        L += [extra_md, '']
    Path(path).write_text('\n'.join(L) + '\n')
    return path


def geom_rows(ens):
    """Per level: stress-module rho, distribution of the inferred rho_rec over recordings, mean d_ln_rho, widths, coverage, residual share."""
    out = []
    for lv, d in ens.items():
        per_rec = d.groupby('rec').rho_rec.first()
        rs = ST.RHO0 + ST.PATH_SCALE * (ST.D_RHO_MM[lv] if lv in ST.D_RHO_MM else ST.D_RHO_MM[ST.X_INFO[lv]['base_level']])
        sd_pred = (d.spo2_hi - d.spo2_lo) / (2 * 1.96)
        sr, sp = float(np.std(d.res)), float(np.std(d.phys))
        out.append(dict(level=lv, rho_stress=rs, rho_med=float(per_rec.median()), rho_min=float(per_rec.min()), rho_max=float(per_rec.max()),
                        dlr=float(d.d_ln_rho.mean()), sd_sat=float(100 * np.median(d.sd_lap)), sd_pred=float(np.median(sd_pred)), cov=float(d.answer.mean()),
                        res_share=sr / (sr + sp) if (sr + sp) > 0 else 0.0))
    return out


def run(level: int = 0, seed: int = 0) -> pd.DataFrame:
    """CONTRACT.md entry point: out-of-fold twin_inverse predictions of ONE seed at one stress level (nested LOSO, models trained on clean people)."""
    torch.set_num_threads(3)
    F = build_levels([0] if level == 0 else [0, level])
    ens, per_seed, _, _ = run_twin(F, use_res=True, seeds=(seed,), levels=[0] if level == 0 else [0, level], tag=f'twin_inverse seed {seed}')
    d = ens[level]
    return d[[c for c in COLS if c in d.columns]]


# ================================================================================================ twin_gate (POST HOC, requested 17:37 CT after outer results were seen)
GATE_T_DB, GATE_SOFT_DB = 10.0, 3.0   # fixed a priori by the planner (no tuning): hard threshold and soft-blend scale in dB


def red_snr_db(rgb, starts):
    """Label-free red pulsatile SNR [dB] of each 10 s window of a recording (stress.snr_db-style, but from the window alone): power of ln R in the
    0.7-3.5 Hz pulse band over the white-noise power in that band, the noise density taken from the 5-14 Hz band of the same window. [n] float."""
    L = WIN_S * FS
    x = np.log(np.maximum(np.asarray(rgb, np.float64)[:, 0], 0.5))
    idx = (np.asarray(starts) * FS)[:, None] + np.arange(L)[None]
    seg = x[idx]
    tt = np.arange(L) - (L - 1) / 2
    slope = (seg * tt[None]).sum(1) / (tt ** 2).sum()
    seg = seg - seg.mean(1, keepdims=True) - slope[:, None] * tt[None]          # remove mean and linear drift
    P = np.abs(np.fft.rfft(seg * np.hanning(L)[None], axis=1)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / FS)
    band = (f >= 0.7) & (f <= 3.5); hf = (f >= 5.0) & (f <= 14.0)
    noise = P[:, hf].mean(1) * band.sum()
    sig = np.maximum(P[:, band].sum(1) - noise, 1e-30)
    return 10.0 * np.log10(sig / np.maximum(noise, 1e-30))


def _snr_job(args):
    rec, rgb, level, starts = args
    return rec, level, red_snr_db(_degrade(rgb, level), starts)


def build_gate_snr(levels=LEVEL_SET, log=print):
    """[rec, t, snr_db] per level (held-out recordings degraded exactly as for scoring), cached in CDIR."""
    p = CDIR / 'gate_snr.pkl'
    if p.exists():
        return pd.read_pickle(p)
    recs = _recordings()
    F0 = pd.read_pickle(CDIR / 'feats_L0.pkl')
    out = {}
    for lv in levels:
        parts = []
        for r in recs:
            st = F0[F0.rec == r['rec']].t.values
            parts.append(pd.DataFrame(dict(rec=r['rec'], t=st.astype(int), snr_db=red_snr_db(_degrade(r['rgb'], lv), st))))
        out[lv] = pd.concat(parts, ignore_index=True)
        log(f'gate SNR {lv}: median {out[lv].snr_db.median():.1f} dB, share >= {GATE_T_DB:g} dB {float((out[lv].snr_db >= GATE_T_DB).mean()):.2f}')
    pd.to_pickle(out, p)
    return out


def _read_scores(name, lv):
    d = pd.read_csv(SCORES / f'{name}__{"L" + str(lv) if not isinstance(lv, str) else lv}.csv')
    d['rec'] = d.rec.astype(str).str.replace('hof_', '', regex=False); d['sid'] = d.sid.astype(str)
    d['answer'] = d.answer.astype(str).str.lower().isin(['true', '1', '1.0'])
    if 'conf' not in d:
        d['conf'] = (d.score - 0.5).abs()
    return d


def gate_frames(chain, twin, snr):
    """(hard, soft) frames from aligned chain_v1 and twin frames and the window SNR [dB]."""
    m = chain[['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer']].merge(
        twin[['rec', 't', 'score', 'pred', 'answer', 'conf'] + [c for c in ('spo2_hat', 'spo2_lo', 'spo2_hi') if c in twin]], on=['rec', 't'], suffixes=('_c', '_t'))
    m = m.merge(snr, on=['rec', 't'], how='left')
    assert m.snr_db.notna().all() and len(m) == len(chain)
    hi = (m.snr_db >= GATE_T_DB).values
    conf_c = 2 * (m.score_c - 0.5).abs()
    hard = pd.DataFrame(dict(rec=m.rec, t=m.t, sid=m.sid, y_spo2=m.y_spo2, y=m.y, score=np.where(hi, m.score_c, m.score_t), pred=np.where(hi, m.pred_c, m.pred_t),
                             answer=np.where(hi, m.answer_c, m.answer_t), conf=np.where(hi, conf_c, m.conf), snr_db=m.snr_db, path=np.where(hi, 'chain', 'twin')))
    w = 1.0 / (1.0 + np.exp(-(m.snr_db.values - GATE_T_DB) / GATE_SOFT_DB))
    s = w * m.score_c.values + (1 - w) * m.score_t.values
    soft = pd.DataFrame(dict(rec=m.rec, t=m.t, sid=m.sid, y_spo2=m.y_spo2, y=m.y, score=s, pred=(s > 0.5).astype(int), answer=np.where(w >= 0.5, m.answer_c, m.answer_t),
                             conf=w * conf_c + (1 - w) * m.conf.values, snr_db=m.snr_db, w_chain=w))
    return hard, soft


def chain_null(F0, mode=True):
    """chain_v1 (shipped C 0.01 / 0.1, p > 0.5, band 0.10) refit LOSO on null training labels (mode True: 200 s shift, 'perm': within-recording
    permutation), scored on the true labels (null control)."""
    d0 = _labels(F0, null=mode)
    ev = (d0.t >= CAL_S).values & d0.y_spo2.notna().values
    X = d0[[f'D_{c}' for c in FUSION]].values
    parts = []
    for so in sorted(d0.sid.unique()):
        if mode == 'gperm':
            d0 = _labels(F0, null=mode, outer_so=so)
        tr = np.nonzero(ev & (d0.sid != so).values)[0]; te = np.nonzero(ev & (d0.sid == so).values)[0]
        S.assert_no_leak(d0.sid.values, tr, te)
        ms = _lr_fit(X[tr], (d0.y_lab.values[tr] < 90).astype(int), 0.01, 0.1)
        s = _lr_score(ms, X[te])
        parts.append(pd.DataFrame(dict(rec=d0.rec.values[te], t=d0.t.values[te].astype(int), sid=d0.sid.values[te], score=s, pred=(s > 0.5).astype(int),
                                       answer=np.abs(s - 0.5) >= 0.10)))
    return _attach_truth(pd.concat(parts, ignore_index=True), F0)


def chain_null_distribution(F0, n=20):
    """Spread of the null: chain_v1 refit LOSO on n independent global label permutations (per outer fold, among the training people only).
    Returns the list of TM values (one per permutation draw)."""
    out = []
    d_true = _labels(F0)
    ev = (d_true.t >= CAL_S).values & d_true.y_spo2.notna().values
    X = d_true[[f'D_{c}' for c in FUSION]].values
    for k in range(n):
        parts = []
        for so in sorted(d_true.sid.unique()):
            tr = np.nonzero(ev & (d_true.sid != so).values)[0]; te = np.nonzero(ev & (d_true.sid == so).values)[0]
            rng = np.random.default_rng([k, zlib.crc32(str(so).encode())])
            y_tr = (rng.permutation(d_true.y_spo2.values[tr]) < 90).astype(int)
            s = _lr_score(_lr_fit(X[tr], y_tr, 0.01, 0.1), X[te])
            parts.append(pd.DataFrame(dict(t=d_true.t.values[te], y=(d_true.y_spo2.values[te] < 90).astype(int), pred=(s > 0.5).astype(int))))
        d = pd.concat(parts)
        out.append(float(M.time_matched(d.t.values, d.y.values, d.pred.values)['tm_bacc']))
    return out


def run_gate(log=print):
    F0 = pd.read_pickle(CDIR / 'feats_L0.pkl')
    snr = build_gate_snr(log=log)
    out = {}
    for lv in LEVEL_SET:
        hard, soft = gate_frames(_read_scores('chain_v1', lv), _read_scores('twin_inverse_noresid', lv), snr[lv])
        write_scores(hard, 'twin_gate', lv); write_scores(soft, 'twin_gate_soft', lv)
        out[('twin_gate', lv)] = quick_metrics(hard); out[('twin_gate_soft', lv)] = quick_metrics(soft)
        out[('twin_gate', lv)]['share_chain'] = float((hard.path == 'chain').mean())
        log(f'twin_gate {lv}: TM {out[("twin_gate", lv)]["tm"]:.3f} (chain path on {out[("twin_gate", lv)]["share_chain"]:.2f} of windows); '
            f'soft TM {out[("twin_gate_soft", lv)]["tm"]:.3f}')
    for suf, mode in (('', True), ('_perm', 'perm'), ('_gperm', 'gperm')):
        src = f'twin_inverse_noresid{suf}_null'
        if not (SCORES / f'{src}__L0.csv').exists():
            continue
        cn = chain_null(F0, mode)
        hard, soft = gate_frames(cn, _read_scores(src, 0), snr[0])
        write_scores(hard, f'twin_gate{suf}_null', 0); write_scores(soft, f'twin_gate_soft{suf}_null', 0)
        out[(f'twin_gate{suf}_null', 0)] = quick_metrics(hard); out[(f'twin_gate_soft{suf}_null', 0)] = quick_metrics(soft)
        out[(f'chain_v1_refit{suf}_null', 0)] = quick_metrics(cn)
        log(f'twin_gate{suf}_null L0: TM {out[(f"twin_gate{suf}_null", 0)]["tm"]:.3f}; soft {out[(f"twin_gate_soft{suf}_null", 0)]["tm"]:.3f}; '
            f'chain_v1 refit on the same null labels {out[(f"chain_v1_refit{suf}_null", 0)]["tm"]:.3f}')
    return out


def regate(extra_md=''):
    """Apply the input-support gate (idempotent) to the twin_inverse / twin_inverse_noresid files already written, refresh the summary and the log."""
    F = build_levels(LEVEL_SET)
    summary = pd.read_pickle(CDIR / 'summary.pkl') if (CDIR / 'summary.pkl').exists() else {}
    for name in ('twin_inverse', 'twin_inverse_noresid'):
        p = CDIR / f'{name}_ens.pkl'
        if not p.exists():
            continue
        ens = pd.read_pickle(p)
        g = {}
        for lv, d in ens.items():
            g[lv] = apply_support_gate(d, F[lv], F[0])
            write_scores(g[lv], name, lv)
            summary[(name, lv)] = quick_metrics(g[lv])
        summary[(name, 'geom')] = geom_rows(g)
        for sd in SEEDS:
            q = SCORES / f'{name}_seed{sd}__L0.csv'
            if q.exists():
                d = pd.read_csv(q)
                d['rec'] = d.rec.astype(str); d['sid'] = d.sid.astype(str)
                dg = apply_support_gate(d, F[0], F[0])
                write_scores(dg, f'{name}_seed{sd}', 0)
                summary[(f'{name}_seed{sd}', 0)] = quick_metrics(dg)
    pd.to_pickle(summary, CDIR / 'summary.pkl')
    write_log(summary, extra_md=extra_md)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default='chain,twin,noresid,null')
    ap.add_argument('--regate', action='store_true', help='re-apply the input-support gate to the written twin_inverse files and rewrite the log')
    ap.add_argument('--gate', action='store_true', help='POST HOC twin_gate / twin_gate_soft from chain_v1 and twin_inverse_noresid files (+ null control)')
    a = ap.parse_args(argv)
    if a.regate:
        return regate()
    if a.gate:
        torch.set_num_threads(3)
        summary = pd.read_pickle(CDIR / 'summary.pkl') if (CDIR / 'summary.pkl').exists() else {}
        summary.update(run_gate())
        pd.to_pickle(summary, CDIR / 'summary.pkl')
        write_log(summary)
        return summary
    torch.set_num_threads(3)
    os.environ.setdefault('OMP_NUM_THREADS', '3')
    t0 = time.time()
    lines = []
    log = lambda s: (print(s, flush=True), lines.append(s))
    F = build_levels(LEVEL_SET, log=log)
    only = set(a.only.split(','))
    summary = pd.read_pickle(CDIR / 'summary.pkl') if (CDIR / 'summary.pkl').exists() and not a.quick else {}
    if 'chain' in only and not a.quick:
        for nested in (False, True):
            name = 'twin_chain_nestedC' if nested else 'twin_chain'
            outs, flog = run_twin_chain(F, nested, log=log)
            for lv, d in outs.items():
                write_scores(d, name, lv)
                summary[(name, lv)] = quick_metrics(d)
            summary[(name, 'folds')] = flog
            pd.to_pickle(outs, CDIR / f'{name}_outs.pkl')
            log(f'{name}: ' + ', '.join(f'L{lv} TM {summary[(name, lv)]["tm"]:.3f}' for lv in LEVEL_SET) + f'  ({time.time() - t0:.0f}s)')
    for key, use_res, null, name in (('twin', True, False, 'twin_inverse'), ('noresid', False, False, 'twin_inverse_noresid'), ('null', True, True, 'twin_inverse_null'),
                                     ('null_noresid', False, True, 'twin_inverse_noresid_null'), ('perm_noresid', False, 'perm', 'twin_inverse_noresid_perm_null'),
                                     ('gperm_noresid', False, 'gperm', 'twin_inverse_noresid_gperm_null')):
        if key not in only:
            continue
        ens, per_seed, flog, reps = run_twin(F, use_res, null=null, seeds=(0,) if (null or a.quick) else SEEDS, levels=[0] if a.quick else LEVEL_SET,
                                             quick=a.quick, log=log, tag=name)
        if a.quick:
            for lv, d in ens.items():
                qm = quick_metrics(d)
                log(f'QUICK {name} L{lv}: TM {qm["tm"]:.3f} cov {qm["cov"]:.2f} acc_ans {qm["acc_ans"]:.3f} MAE {qm["mae"]:.2f}')
            continue
        for lv, d in ens.items():
            dg = apply_support_gate(d, F[lv], F[0]) if not null else d
            write_scores(dg, name, lv)
            summary[(name, lv)] = quick_metrics(dg)
        if not null:
            for sd, d in per_seed.items():
                dg = apply_support_gate(d, F[0], F[0])
                write_scores(dg, f'{name}_seed{sd}', 0)
                summary[(f'{name}_seed{sd}', 0)] = quick_metrics(dg)
        summary[(name, 'folds')] = flog
        summary[(name, 'params')] = {f'{k[0]}_seed{k[1]}': v for k, v in reps.items()}
        if not null:
            summary[(name, 'geom')] = geom_rows({lv: apply_support_gate(d, F[lv], F[0]) for lv, d in ens.items()})
        pd.to_pickle(ens, CDIR / f'{name}_ens.pkl')
        log(f'{name}: ' + ', '.join(f'L{lv} TM {summary[(name, lv)]["tm"]:.3f}' for lv in ens) + f'  ({time.time() - t0:.0f}s)')
    if not a.quick:
        pd.to_pickle(summary, CDIR / 'summary.pkl')
        with open(CDIR / 'run_log.txt', 'a') as fh:
            fh.write('\n'.join(lines) + '\n')
        write_log(summary)
    log(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
