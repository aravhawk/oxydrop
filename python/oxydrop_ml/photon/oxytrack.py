"""OxyTrack (photon candidate C): a causal, personal state-space tracker of SpO2 for the phone-camera oximeter.

Everything is STRICTLY causal within a recording: the output at window k (10 s window, 2 s hop) uses windows <= k of the same recording only, and
elapsed time is never an input (the filter only sees the per-window observations and SNR; its state is the same function of those for every person).

Model (one recording; d_k = s_k - s_base is the desaturation in SpO2 points relative to the person's label-free baseline minute)
  dynamics   d_k = d_{k-1} + w_k,  w_k ~ N(0, q)                zero-mean random walk, no drift and no learned trajectory shape (cannot act as a
                                                                protocol clock); q = q_mult * q_emp, q_emp = mean squared 2 s increment of the
                                                                training people's reference trajectories, q_mult nested
             b_jk = b_j,k-1 + v_jk,  v ~ N(0, qb_j)             per-person offset of emission channel j (label-free test-time adaptation)
  emission   u_jk = beta_j a_j(SNR_jk) d_k + b_jk + e_jk,  e_k ~ N(0, Lam_k^1/2 R Lam_k^1/2)
             channel hr : dHR, chain v1 FUSION feature 0 (bpm relative to the baseline minute)
             channel fus: ridge projection (no intercept, alpha chosen by leave-one-person-out inside the training people) of the 19 chain v1 FUSION
                          deltas onto d; this includes the baseline-relative breathing rates (rr_amp, rr_ibi), RMSSD and SDNN of chain v1
             a(SNR) = SNR / (SNR + kappa)                       expected attenuation of a pulse-amplitude feature when additive noise of power
                                                                SNR^-1 x signal is mixed in (kappa nested); SNR = per-window spectral pulse SNR
                                                                (power within 0.2 Hz of the fundamental and its harmonics over the in-band noise
                                                                floor, 0.7-4 Hz, causal band-pass), hr channel max(R, G), fus channel min(R, G)
             beta_j and R from the training people only (fus residuals out-of-person); R is inflated by the integrated autocorrelation of the
             residuals over the 5-window overlap span
  adaptation (label-free, causal, gated as required by the red-team review): offsets b and the per-channel noise scales Lam = diag(lam_hr, lam_fus)
             adapt only on windows inside the first 60 s or on windows the filter already calls clearly normal (predictive P(s<90) < 0.10);
             otherwise b is held fixed (Schmidt "consider" update) so the desaturation cannot be absorbed as drift.
             lam_j = (nu0 + sum_i g^(k-i) eps_ji) / (nu0 + sum_i g^(k-i)), eps = expected squared white residual / R_jj (online EM), g = 1 - eta;
             qb_j = rho_b * (training per-recording offset variance) / (typical recording length); nu0, eta, rho_b nested
  robustness Huber-type inflation of R for |innovation| > 3 predicted SD (beat-detection outliers)
  output     s_k | windows <= k ~ N(mu_b + m_k, sigma_b^2 + P_k)   (mu_b, sigma_b: baseline SpO2 of the training people)
             score = P(s_k < 90), pred = score > thr (thr nested by baselines.common.choose_thr on the inner OOF scores), answer = |score - thr| >= c
             (c nested by the shared rule baselines.common.choose_band: smallest c with answered inner-OOF acc and bacc >= 0.90) AND posterior SD
             <= the 0.99 quantile of the inner-OOF posterior SDs of the clean training people (information gate: in low light the posterior widens
             and the filter abstains); spo2_hat = posterior mean, spo2_lo/hi = 80 % band with its SD scaled so the inner-OOF band covers 80 %.

Nested protocol: outer LOSO by person (both hands together); every choice (ridge alpha, fus projection, q_mult, kappa, rho_b, nu0, eta, decision
threshold, c, band scale, SD gate; the EMA span of the control) is made on the inner person-grouped CV of the 5 training people (splits.inner_group_folds, k=5, i.e. leave-one-training-person-out) with time-matched
balanced accuracy (metrics.time_matched) of pred as the selection metric. Models are fit on CLEAN training people and not refit per stress level.

Variants (separate score files):
  oxytrack           the model above
  oxytrack_ema       control: chain v1 probability (shipped C) with a causal exponential moving average, span nested (is any gain just smoothing?)
  oxytrack_noadapt   control: no test-time adaptation (no offsets, lam = 1)
  oxytrack_nosnr     control: fixed observation noise and gain (a = 1)
  oxytrack_blank     null control (L0 only): the fitted oxytrack filter fed blank observations (u = 0, SNR = training median) aligned to t;
                     it must score about 0.5 TM bal. acc. (a dynamics prior that knows the protocol would score higher)

The filter is 3 states and 2 observations (a few hundred floats of parameters per fold), so it ports to browser JavaScript next to app/js/engine_v1.js.
Run: cd python && python -m oxydrop_ml.photon.oxytrack [--levels 0 1 2 ...] [--seed 0]
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .. import CACHE
from .. import metrics as M
from .. import splits as S
from ..baselines import common as BC

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
CDIR = Path(os.environ.get('OXYTRACK_CACHE', str(Path(CACHE) / 'photon_oxytrack')))
WINDOWS_CSV = REPO / 'results' / 'scores' / 'windows.csv'

MODELS = ('oxytrack', 'oxytrack_ema', 'oxytrack_noadapt', 'oxytrack_nosnr')
FS, WIN, STEP, CAL = 30, 10, 2, 60
THR = 90.0
Z80 = 1.2815515655446004
GATE_P = 0.10                      # adapt only where the predictive P(s < 90) is below this (or inside the baseline minute)
HUBER = 3.0
LAM_MIN, LAM_MAX = 0.25, 25.0
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
Q_MULT = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)
SD_GATE_Q = 0.99                   # information gate: abstain when the posterior SD exceeds this quantile of the inner-OOF SDs of the clean training people
KAPPA = (0.3, 1.0, 3.0)
RHO_B = (0.0, 1.0, 4.0)
TTA = ((10.0, 0.02), (40.0, 0.02), (10.0, 0.1), (40.0, 0.1))     # (nu0 = emission prior strength, eta = adaptation rate)
EMA_SPANS = (1, 2, 3, 5, 8, 13, 21, 34)
CHAIN_C = dict(C_hr=0.01, C_fus=0.1)                             # shipped chain v1
PROJ = ('ridge', 'lr')                                           # fus-channel projection: ridge onto d, or the chain-v1 logistic direction


def grid(model):
    """Nested grid of a variant (list of config dicts, simplest first)."""
    if model == 'oxytrack':
        return [dict(proj=p, q_mult=q, kappa=k, rho_b=r, nu0=n, eta=e, adapt=True) for p in PROJ for r in RHO_B for q in Q_MULT for k in KAPPA for n, e in TTA]
    if model == 'oxytrack_noadapt':
        return [dict(proj=p, q_mult=q, kappa=k, rho_b=0.0, nu0=1.0, eta=0.0, adapt=False) for p in PROJ for q in Q_MULT for k in KAPPA]
    if model == 'oxytrack_nosnr':
        return [dict(proj=p, q_mult=q, kappa=None, rho_b=r, nu0=n, eta=e, adapt=True) for p in PROJ for r in RHO_B for q in Q_MULT for n, e in TTA]
    if model == 'oxytrack_ema':
        return [dict(span=s) for s in EMA_SPANS]
    raise KeyError(model)


# ------------------------------------------------------------------------------------------------ causal features
NFFT = 1024
_F = np.fft.rfftfreq(NFFT, 1.0 / FS)
_BAND = (_F >= 0.7) & (_F <= 4.0)
_F0 = (_F >= 0.7) & (_F <= 3.5)
_HANN = np.hanning(WIN * FS)


def window_snr(rgb, t):
    """Per-window spectral pulse SNR (linear power ratio) of R, G, B for the 10 s windows starting at t (s). Causal: uses the causal 0.7-4 Hz
    band-passed log intensity of chain v1 (oxydrop.features.rec_feats) inside the window only. SNR = (P_sig - n * W_sig) / (n * W_band): P_sig = power
    within 0.2 Hz of the fundamental f0 (peak of the summed normalised R+G spectra in 0.7-3.5 Hz) and its 2nd/3rd harmonics, n = mean noise density
    of the remaining 0.7-4 Hz bins. Pure noise gives ~0. Returns (n, 3), clipped to [1e-3, 1e3]."""
    from oxydrop.features import rec_feats
    _, ac = rec_feats(np.asarray(rgb, float))
    t = np.asarray(t, int)
    idx = t[:, None] * FS + np.arange(WIN * FS)[None, :]
    seg = ac[idx]
    seg = seg - seg.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(seg * _HANN[None, :, None], n=NFFT, axis=1)) ** 2
    Pb = P * _BAND[None, :, None]
    tot = Pb.sum(1) + 1e-30
    comb = Pb[:, :, 0] / tot[:, None, 0] + Pb[:, :, 1] / tot[:, None, 1]
    comb = np.where(_F0[None, :], comb, -1.0)
    f0 = _F[np.argmax(comb, 1)]
    sig = np.zeros((len(t), len(_F)), bool)
    for h in (1, 2, 3):
        sig |= np.abs(_F[None, :] - h * f0[:, None]) <= 0.2
    sig &= _BAND[None, :]
    noi = _BAND[None, :] & ~sig
    ns = sig.sum(1).astype(float); nn = np.maximum(noi.sum(1), 1).astype(float); nb = float(_BAND.sum())
    Ps = (P * sig[:, :, None]).sum(1); Pn = (P * noi[:, :, None]).sum(1)
    dens = Pn / nn[:, None]
    snr = (Ps - dens * ns[:, None]) / (dens * nb + 1e-30)
    return np.clip(snr, 1e-3, 1e3)


def rec_features(rgb):
    """Causal per-window inputs of one recording: t (window starts, s), D (19 chain v1 FUSION deltas vs the baseline minute), snr (R, G, B)."""
    from oxydrop import chain
    rgb = np.asarray(rgb, float)
    t, D = chain.deltas(rgb)
    t = t.astype(int)
    return dict(t=t, D=np.asarray(D, float), snr=window_snr(rgb, t))


def snr_channels(snr):
    """(n, 3) R, G, B SNR -> (n, 2): hr channel (beat detection uses the better of G/R) and fus channel (optical ratios need both R and G)."""
    snr = np.asarray(snr, float)
    return np.stack([np.maximum(snr[:, 0], snr[:, 1]), np.minimum(snr[:, 0], snr[:, 1])], 1)


def att(snr2, kappa):
    return np.ones_like(snr2) if kappa is None else snr2 / (snr2 + kappa)


# ------------------------------------------------------------------------------------------------ data
_REC = {}


def recordings():
    """rec name ('100001L') -> raw Hoffman recording dict (rgb per frame, 30 fps)."""
    if not _REC:
        from ..data import hoffman as _h            # puts python/ on sys.path for the legacy `oxydrop` package
        from oxydrop import hoffman
        for r in hoffman.load(_h.ROOT):
            _REC[r['rec']] = r
    return _REC


def level_tag(level):
    """'L3' for an int stress level of stress.LEVELS, the name itself ('X1') for an out-of-family stress of stress.X_LEVELS."""
    return level if isinstance(level, str) else f'L{int(level)}'


def degrade_rec(rgb, level, seed):
    """Degrade one whole recording with the shared harness (stress.degrade, or stress.degrade_x for 'X1'); deterministic in (level, seed, rgb)."""
    from . import stress
    if isinstance(level, str):
        return stress.degrade_x(np.asarray(rgb, float), level, seed)
    return stress.degrade(np.asarray(rgb, float), int(level), seed)


def features(level=0, seed=0, recs=None):
    """rec -> causal features of the recording degraded at `level` (0 = clean). Cached under CDIR."""
    CDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    R = recordings()
    for rec in (sorted(R) if recs is None else recs):
        s = 0 if level == 0 else seed
        p = CDIR / f'feat_{level_tag(level)}_s{s}_{rec}.npz'
        if p.exists():
            z = np.load(p)
            out[rec] = dict(t=z['t'], D=z['D'], snr=z['snr'])
            continue
        rgb = R[rec]['rgb'] if level == 0 else degrade_rec(R[rec]['rgb'], level, seed)
        f = rec_features(rgb)
        np.savez(p, **f)
        out[rec] = f
    return out


def window_table():
    w = pd.read_csv(WINDOWS_CSV)
    return w


def rec_table(feats, wdf):
    """Attach the reference labels (training people only use them; the held-out person's labels are only used to score)."""
    lab = {(r, int(t)): float(y) for r, t, y in zip(wdf.rec.values, wdf.t.values, wdf.y.values)}
    out = {}
    for rec, f in feats.items():
        y = np.array([lab.get((rec, int(t)), np.nan) for t in f['t']])
        base = (f['t'] + WIN <= CAL) & np.isfinite(y)
        out[rec] = dict(rec=rec, sid=int(rec[:-1]), t=np.asarray(f['t'], int), D=np.asarray(f['D'], float), snr=np.asarray(f['snr'], float), y=y,
                        sbase=float(np.mean(y[base])) if base.any() else np.nan)
    return out


def eval_mask(r):
    return (r['t'] >= CAL) & np.isfinite(r['y'])


# ------------------------------------------------------------------------------------------------ emission model (training people only)
def _ridge(Z, d, alpha):
    return np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ d)


def _lagcorr(r, rid, lag):
    """Autocorrelation at `lag` of the per-recording centred residual series (pairs within a recording only)."""
    num = den = 0.0
    for g in np.unique(rid):
        x = r[rid == g]
        if len(x) > lag:
            num += float((x[lag:] * x[:-lag]).sum()); den += float((x * x).sum())
    return num / den if den > 0 else 0.0


def fit_emission(train):
    """Fit the emission model on a list of TRAINING recording dicts (labels used). Returns a dict of parameters for every kappa (and None)."""
    D, d, snr2, rid, sid, steps, base_d, ys = [], [], [], [], [], [], [], []
    sbase = []
    for i, r in enumerate(train):
        m = eval_mask(r)
        dd = r['y'][m] - r['sbase']
        D.append(r['D'][m]); d.append(dd); snr2.append(snr_channels(r['snr'][m])); rid.append(np.full(m.sum(), i)); sid.append(np.full(m.sum(), r['sid']))
        ys.append(r['y'][m])
        tt = r['t'][m]
        ok = np.diff(tt) == STEP
        steps.append(np.diff(dd)[ok])
        b = (r['t'] + WIN <= CAL) & np.isfinite(r['y'])
        base_d.append(r['y'][b] - r['sbase'])
        sbase.append(r['sbase'])
    D = np.vstack(D); d = np.concatenate(d); snr2 = np.vstack(snr2); rid = np.concatenate(rid); sid = np.concatenate(sid); ys = np.concatenate(ys)
    sd = D.std(0) + 1e-9
    Z = D / sd
    best = None
    for alpha in ALPHAS:
        oof = np.zeros(len(d))
        for p in np.unique(sid):
            te = sid == p
            oof[te] = Z[te] @ _ridge(Z[~te], d[~te], alpha)
        mse = float(np.mean((oof - d) ** 2))
        if best is None or mse < best[0] - 1e-12:
            best = (mse, alpha, oof)
    mse, alpha, oof = best
    W = dict(ridge=_ridge(Z, d, alpha))
    OOF = dict(ridge=oof)
    # chain-v1-direction projection: class-balanced logistic regression (shipped C_fus) on the same scaled deltas; u = w.z without intercept
    yb = (ys < THR).astype(int)
    oof_lr = np.zeros(len(d))
    for p in np.unique(sid):
        te = sid == p
        oof_lr[te] = Z[te] @ _lr_dir(Z[~te], yb[~te])
    W['lr'] = _lr_dir(Z, yb); OOF['lr'] = oof_lr
    cal = {}
    for proj in PROJ:
        U = np.stack([D[:, 0], OOF[proj]], 1)
        for kappa in (None,) + KAPPA:
            X = att(snr2, kappa) * d[:, None]
            beta = (U * X).sum(0) / np.maximum((X * X).sum(0), 1e-12)
            r = U - X * beta
            off = np.stack([r[rid == g].mean(0) for g in range(len(train))])
            rc = r - off[rid]
            sw2 = np.maximum((rc ** 2).mean(0), 1e-6)
            tau = np.array([1.0 + 2.0 * sum(max(_lagcorr(rc[:, j], rid, l), 0.0) for l in range(1, 5)) for j in range(2)])
            corr = float(np.clip((rc[:, 0] * rc[:, 1]).mean() / np.sqrt(sw2[0] * sw2[1]), -0.8, 0.8))
            v = tau * sw2
            R = np.array([[v[0], corr * np.sqrt(v[0] * v[1])], [corr * np.sqrt(v[0] * v[1]), v[1]]])
            cal[(proj, 'none' if kappa is None else kappa)] = dict(beta=beta, R=R, soff2=(off ** 2).mean(0), tau=tau, sw2=sw2, corr=corr)
    steps = np.concatenate(steps)
    base_d = np.concatenate(base_d)
    sbase = np.asarray(sbase)
    n_typ = float(np.median(np.bincount(rid)))
    return dict(sd=sd, W=W, alpha=alpha, ridge_oof_mse=mse, q_emp=float(np.mean(steps ** 2)), p0=float(max(np.var(base_d), 0.05)),
                mu_b=float(sbase.mean()), sig_b=float(max(sbase.std(ddof=1), 0.5)), cal=cal, n_typ=n_typ, snr_med=np.median(snr2, 0),
                train_sids=sorted({int(r['sid']) for r in train}))


def _lr_dir(Z, yb):
    from sklearn.linear_model import LogisticRegression
    sd = Z.std(0) + 1e-9
    lr = LogisticRegression(C=CHAIN_C['C_fus'], class_weight='balanced', max_iter=2000).fit(Z / sd, yb)
    return lr.coef_[0] / sd


def observations(em, D, proj='ridge'):
    """(n, 2) observations [dHR, projection of the scaled chain deltas] of a recording (projection 'ridge' or 'lr')."""
    D = np.asarray(D, float)
    return np.stack([D[:, 0], (D / em['sd']) @ em['W'][proj]], 1)


def params(em, cfgs):
    G = len(cfgs)
    beta = np.zeros((G, 2)); R = np.zeros((G, 2, 2)); q = np.zeros(G); qb = np.zeros((G, 2))
    nu0 = np.zeros(G); eta = np.zeros(G); adapt = np.zeros(G, bool)
    for g, c in enumerate(cfgs):
        cal = em['cal'][(c['proj'], 'none' if c['kappa'] is None else c['kappa'])]
        beta[g] = cal['beta']; R[g] = cal['R']; q[g] = em['q_emp'] * c['q_mult']
        qb[g] = c['rho_b'] * cal['soff2'] / em['n_typ']
        nu0[g] = c['nu0']; eta[g] = c['eta']; adapt[g] = c['adapt']
    return dict(beta=beta, R=R, q=q, qb=qb, nu0=nu0, eta=eta, adapt=adapt, p0=np.full(G, em['p0']), mu_b=em['mu_b'], sig_b=em['sig_b'])


def attenuation(snr, cfgs):
    s2 = snr_channels(snr)
    return np.stack([att(s2, c['kappa']) for c in cfgs], 0)


# ------------------------------------------------------------------------------------------------ the causal filter
def _inv2(S):
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    out = np.empty_like(S)
    out[:, 0, 0] = S[:, 1, 1]; out[:, 1, 1] = S[:, 0, 0]; out[:, 0, 1] = -S[:, 0, 1]; out[:, 1, 0] = -S[:, 1, 0]
    return out / det[:, None, None]


def track(u, a, base_mask, prm):
    """Run the filter causally over one recording for G configurations at once.
    u (n, 2) observations, a (G, n, 2) SNR attenuation, base_mask (n,) windows inside the label-free first minute, prm from params().
    Returns dict m, P (G, n): posterior mean / variance of d after window k; lam (G, n, 2); gate (G, n); score, spo2_hat, spo2_lo, spo2_hi (G, n)."""
    u = np.asarray(u, float); a = np.asarray(a, float)
    G, n = a.shape[0], a.shape[1]
    if u.ndim == 2:
        u = np.broadcast_to(u[None], (G,) + u.shape)
    beta, R0, q, qb = prm['beta'], prm['R'], prm['q'], prm['qb']
    nu0 = prm['nu0'][:, None]; gam = (1.0 - prm['eta'])[:, None]; adapt = prm['adapt'][:, None]
    mu_b, s2b = prm['mu_b'], prm['sig_b'] ** 2
    m = np.zeros((G, 3)); P = np.zeros((G, 3, 3)); P[:, 0, 0] = prm['p0']
    lam = np.ones((G, 2)); Sacc = np.zeros((G, 2)); Wacc = np.zeros((G, 1))
    I3 = np.eye(3)[None]
    dR0 = np.stack([R0[:, 0, 0], R0[:, 1, 1]], 1)
    H = np.zeros((G, 2, 3)); H[:, 0, 1] = 1.0; H[:, 1, 2] = 1.0
    om = np.empty((G, n)); oP = np.empty((G, n)); ol = np.empty((G, n, 2)); og = np.empty((G, n), bool); ob = np.empty((G, n, 2))
    for k in range(n):
        if k > 0:
            P[:, 0, 0] += q
        pp = ndtr((THR - mu_b - m[:, 0]) / np.sqrt(s2b + P[:, 0, 0]))
        gate = bool(base_mask[k]) | (pp < GATE_P)
        if k > 0:
            P[:, 1, 1] += np.where(gate, qb[:, 0], 0.0); P[:, 2, 2] += np.where(gate, qb[:, 1], 0.0)
        H[:, 0, 0] = beta[:, 0] * a[:, k, 0]; H[:, 1, 0] = beta[:, 1] * a[:, k, 1]
        sl = np.sqrt(lam)
        R = R0 * sl[:, :, None] * sl[:, None, :]
        v = u[:, k] - np.einsum('gij,gj->gi', H, m)
        PHt = P @ H.transpose(0, 2, 1)
        HPH = H @ PHt
        Sm = HPH + R
        z = np.abs(v) / np.sqrt(np.stack([Sm[:, 0, 0], Sm[:, 1, 1]], 1))
        infl = np.maximum(1.0, (z / HUBER) ** 2)
        if (infl > 1.0).any():
            si = np.sqrt(infl)
            R = R * si[:, :, None] * si[:, None, :]
            Sm = HPH + R
        K = PHt @ _inv2(Sm)
        K[~gate, 1:, :] = 0.0                    # Schmidt update: offsets frozen unless the gate is open
        m = m + np.einsum('gij,gj->gi', K, v)
        IKH = I3 - K @ H
        P = IKH @ P @ IKH.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)
        P = 0.5 * (P + P.transpose(0, 2, 1))
        if adapt.any():
            res = u[:, k] - np.einsum('gij,gj->gi', H, m)
            hph = np.einsum('gij,gjk,gik->gi', H, P, H)
            eps = (res ** 2 + hph) / dR0
            upd = gate[:, None] & adapt
            Sacc = np.where(upd, gam * Sacc + eps, Sacc); Wacc = np.where(upd, gam * Wacc + 1.0, Wacc)
            lam = np.where(adapt, np.clip((nu0 + Sacc) / (nu0 + Wacc), LAM_MIN, LAM_MAX), 1.0)
        om[:, k] = m[:, 0]; oP[:, k] = np.maximum(P[:, 0, 0], 1e-9); ol[:, k] = lam; og[:, k] = gate; ob[:, k] = m[:, 1:]
    sd = np.sqrt(s2b + oP)
    hat = mu_b + om
    return dict(m=om, P=oP, lam=ol, gate=og, b=ob, sd=sd, score=ndtr((THR - hat) / sd), spo2_hat=hat, spo2_lo=hat - Z80 * sd, spo2_hi=hat + Z80 * sd)


def run_filter(em, cfgs, r_feat, blank=False):
    """Filter one recording (features dict with t, D, snr) for the configs. blank=True feeds u = 0 and the training-median SNR (null control)."""
    t = np.asarray(r_feat['t'], int)
    if blank:
        u = np.zeros((len(t), 2))
        snr = np.tile(np.r_[em['snr_med'][0], em['snr_med'][1], em['snr_med'][1]], (len(t), 1))   # max(R,G), min(R,G) = training medians
    else:
        obs = {p: observations(em, r_feat['D'], p) for p in sorted({c['proj'] for c in cfgs})}
        u = np.stack([obs[c['proj']] for c in cfgs], 0); snr = r_feat['snr']
    return track(u, attenuation(snr, cfgs), t + WIN <= CAL, params(em, cfgs))


# ------------------------------------------------------------------------------------------------ control: causal EMA of the chain v1 probability
def chain_fit(train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([r['D'][eval_mask(r)] for r in train]); y = np.concatenate([(r['y'][eval_mask(r)] < THR).astype(int) for r in train])
    models = []
    for cols, c in (([0], CHAIN_C['C_hr']), (list(range(19)), CHAIN_C['C_fus'])):
        sc = StandardScaler().fit(X[:, cols])
        lr = LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(sc.transform(X[:, cols]), y)
        models.append((cols, sc, lr))
    return models


def chain_prob(models, D):
    D = np.asarray(D, float)
    return sum(lr.predict_proba(sc.transform(D[:, cols]))[:, 1] for cols, sc, lr in models) / len(models)


def ema(p, span):
    """Causal exponential moving average over the windows of one recording (alpha = 2 / (span + 1)); span 1 = no smoothing."""
    p = np.asarray(p, float)
    a = 2.0 / (span + 1.0)
    out = np.empty_like(p)
    acc = p[0]
    for i, x in enumerate(p):
        acc = x if i == 0 else a * x + (1 - a) * acc
        out[i] = acc
    return out


# ------------------------------------------------------------------------------------------------ nested LOSO
def _tm(t, y, pred):
    v = M.time_matched(t, y, pred)['tm_bacc']
    return -1.0 if not np.isfinite(v) else float(v)


def _scores_for(model, em, chain, cfgs, rf):
    """(G, n) scores of the configs of `model` on one recording's features."""
    if model == 'oxytrack_ema':
        p = chain_prob(chain, rf['D'])
        return np.stack([ema(p, c['span']) for c in cfgs], 0), None
    out = run_filter(em, cfgs, rf)
    return out['score'], out


def nested_folds(feats0, wdf, models=MODELS, log=print):
    """Nested LOSO on CLEAN features. Returns dict model -> list of fold dicts (one per held-out person) with the fitted model and the choices."""
    recs = rec_table(feats0, wdf)
    sids = sorted({r['sid'] for r in recs.values()})
    res = {m: [] for m in models}
    for p in sids:
        t0 = time.time()
        tr_recs = [r for r in recs.values() if r['sid'] != p]
        rows_sid = np.concatenate([np.full(eval_mask(r).sum(), r['sid']) for r in tr_recs])
        rows_y = np.concatenate([(r['y'][eval_mask(r)] < THR).astype(int) for r in tr_recs])
        inner = list(S.inner_group_folds(rows_sid, k=5, seed=0, balance=rows_y))
        oof = {m: {} for m in models}                   # model -> rec -> (G, n_eval) scores
        for fi in inner:
            va = set(int(s) for s in fi.test_group)
            itr = [r for r in tr_recs if r['sid'] not in va]
            iva = [r for r in tr_recs if r['sid'] in va]
            em = fit_emission(itr)
            ch = chain_fit(itr) if 'oxytrack_ema' in models else None
            for m in models:
                cfgs = grid(m)
                for r in iva:
                    sc, out = _scores_for(m, em, ch, cfgs, r)
                    em_r = eval_mask(r)
                    oof[m][r['rec']] = (sc[:, em_r], None if out is None else out['spo2_hat'][:, em_r], None if out is None else out['sd'][:, em_r])
        em_out = fit_emission(tr_recs)
        ch_out = chain_fit(tr_recs)
        t_all = np.concatenate([r['t'][eval_mask(r)] for r in tr_recs])
        y_all = np.concatenate([(r['y'][eval_mask(r)] < THR).astype(int) for r in tr_recs])
        s_all = np.concatenate([r['y'][eval_mask(r)] for r in tr_recs])
        for m in models:
            cfgs = grid(m)
            S_all = np.concatenate([oof[m][r['rec']][0] for r in tr_recs], 1)
            mode = 'fixed' if m == 'oxytrack_ema' else 'tune'          # chain v1 keeps its shipped p > 0.5; filter thresholds on P(s<90) are nested
            tv = [BC.choose_thr(S_all[g], y_all, t_all, 'tm_bacc', mode, 0.5) for g in range(len(cfgs))]
            vals = [v for _, v in tv]
            g = int(np.argmax(vals))
            thr = float(tv[g][0])
            c = BC.choose_band(S_all[g], y_all, thr)
            extra = dict(band_scale=1.0, sd_max=np.inf)
            if m != 'oxytrack_ema':
                hat = np.concatenate([oof[m][r['rec']][1][g] for r in tr_recs]); sdv = np.concatenate([oof[m][r['rec']][2][g] for r in tr_recs])
                extra = dict(band_scale=float(np.quantile(np.abs(s_all - hat) / sdv, 0.80) / Z80), sd_max=float(np.quantile(sdv, SD_GATE_Q)))
            res[m].append(dict(sid=p, cfg=cfgs[g], thr=thr, c=float(c), inner_tm=float(vals[g]), trace=list(zip(cfgs, vals)), em=em_out, chain=ch_out,
                               inner_cover=float((np.abs(S_all[g] - thr) >= c).mean()), **extra))
        log(f'  fold {p}: ' + '; '.join(f'{m} {_cfg_str(res[m][-1]["cfg"])} thr={res[m][-1]["thr"]:.2f} c={res[m][-1]["c"]:.2f} inner TM {res[m][-1]["inner_tm"]:.3f}'
                                         for m in models) + f'  ({time.time() - t0:.1f} s)')
    return res


def _cfg_str(c):
    if 'span' in c:
        return f'span={c["span"]}'
    return (f'{c["proj"]} q={c["q_mult"]:g}xq_emp kappa={c["kappa"] if c["kappa"] is not None else "fixed"} rho_b={c["rho_b"]:g}'
            + (f' nu0={c["nu0"]:g} eta={c["eta"]:g}' if c['adapt'] else ' no-TTA'))


def score_level(folds, model, feats, wdf, blank=False):
    """Apply the fitted fold models of `model` to the held-out recordings' features (any stress level). Returns the contract DataFrame
    (5,644 eval windows) and per-window extras for the log."""
    recs = rec_table(feats, wdf)
    rows = []
    for fo in folds[model]:
        for r in [r for r in recs.values() if r['sid'] == fo['sid']]:
            em = fo['em']
            if model == 'oxytrack_ema':
                s = ema(chain_prob(fo['chain'], r['D']), fo['cfg']['span'])
                out = None
            else:
                out = run_filter(em, [fo['cfg']], r, blank=blank)
                s = out['score'][0]
            m = eval_mask(r)
            thr = fo['thr']
            ans = np.abs(s[m] - thr) >= fo['c']
            if out is not None:
                ans = ans & (out['sd'][0][m] <= fo['sd_max'])      # information gate: the posterior is wider than on any clean training person
            df = pd.DataFrame(dict(rec=r['rec'], t=r['t'][m], sid=r['sid'], y_spo2=r['y'][m], y=(r['y'][m] < THR).astype(int), score=s[m],
                                   pred=(s[m] > thr).astype(int), answer=ans))
            if out is not None:
                hat, sdv = out['spo2_hat'][0][m], out['sd'][0][m] * fo['band_scale']
                df['spo2_hat'] = hat; df['spo2_lo'] = hat - Z80 * sdv; df['spo2_hi'] = hat + Z80 * sdv
                df['post_sd'] = out['sd'][0][m]
                df['lam_hr'] = out['lam'][0][m, 0]; df['lam_fus'] = out['lam'][0][m, 1]; df['gate'] = out['gate'][0][m]
            s2 = snr_channels(r['snr'][m])
            df['snr_hr'] = s2[:, 0]; df['snr_fus'] = s2[:, 1]
            df['conf'] = np.abs(df.score - thr)
            if out is not None:                                    # windows that fail the information gate rank below every other window
                df['conf'] = df['conf'] - (df['post_sd'] > fo['sd_max']).astype(float)
            rows.append(df)
    out = pd.concat(rows, ignore_index=True).sort_values(['sid', 'rec', 't']).reset_index(drop=True)
    return out


CONTRACT_COLS = ['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer', 'spo2_hat', 'spo2_lo', 'spo2_hi', 'conf']


def write_scores(df, model, level):
    SCORES.mkdir(parents=True, exist_ok=True)
    cols = [c for c in CONTRACT_COLS if c in df.columns]
    p = SCORES / f'{model}__{level_tag(level)}.csv'
    df[cols].to_csv(p, index=False, float_format='%.6g')
    return p


# ------------------------------------------------------------------------------------------------ entry points
_FOLDS = {}


def get_folds(log=print):
    if not _FOLDS:
        _FOLDS.update(nested_folds(features(0), window_table(), log=log))
    return _FOLDS


def run(level: int = 0, seed: int = 0, model: str = 'oxytrack') -> pd.DataFrame:
    """Contract entry point: out-of-fold scores of `model` on the held-out people at stress `level` (fit on clean training people)."""
    folds = get_folds()
    feats = features(level, seed)
    return score_level(folds, model, feats, window_table())


def tm_summary(df):
    tm = M.time_matched(df.t.values, df.y.values, df.pred.values, df.score.values)
    per = {}
    for s, g in df.groupby('sid'):
        per[int(s)] = M.time_matched(g.t.values, g.y.values, g.pred.values)['tm_bacc']
    cm = M.classification_metrics(df.y.values, df.score.values, pred=df.pred.values)
    a = df.answer.values.astype(bool)
    acc_ans = float((df.pred.values[a] == df.y.values[a]).mean()) if a.any() else np.nan
    out = dict(tm_bacc=tm['tm_bacc'], tm_auroc=tm['tm_auroc'], bacc=cm['bacc'], auroc=cm['auroc'], coverage=float(a.mean()), acc_answered=acc_ans, per=per)
    if 'spo2_hat' in df:
        out['mae'] = float(np.abs(df.spo2_hat - df.y_spo2).mean())
        out['band80'] = float(((df.y_spo2 >= df.spo2_lo) & (df.y_spo2 <= df.spo2_hi)).mean())
    return out


def paired_vs(df_a, df_b, B=2000):
    """Paired person bootstrap of the TM bal. acc. difference a - b on the same windows."""
    j = df_a[['rec', 't', 'sid', 'y', 'pred']].merge(df_b[['rec', 't', 'pred']], on=['rec', 't'], suffixes=('_a', '_b'))
    t, y, pa, pb = j.t.values, j.y.values, j.pred_a.values, j.pred_b.values
    f = lambda idx, p: M.time_matched(t[idx], y[idx], p[idx])['tm_bacc']
    r = M.paired_person_bootstrap(j.sid.values, lambda i: f(i, pa), lambda i: f(i, pb), B=B, seed=0)
    pa_per = {s: f(np.nonzero(j.sid.values == s)[0], pa) for s in np.unique(j.sid.values)}
    pb_per = {s: f(np.nonzero(j.sid.values == s)[0], pb) for s in np.unique(j.sid.values)}
    n_better = int(sum(pa_per[s] > pb_per[s] for s in pa_per))
    return dict(diff=r['diff'], lo=r['lo'], hi=r['hi'], n_better=n_better, n=len(pa_per))


def chain_df(folds, feats, wdf):
    """Unsmoothed chain v1 (shipped C) refit inside the same outer folds: the reference for the paired comparisons in the log."""
    recs = rec_table(feats, wdf)
    rows = []
    for fo in folds['oxytrack_ema']:
        for r in [r for r in recs.values() if r['sid'] == fo['sid']]:
            s = chain_prob(fo['chain'], r['D']); m = eval_mask(r)
            rows.append(pd.DataFrame(dict(rec=r['rec'], t=r['t'][m], sid=r['sid'], y_spo2=r['y'][m], y=(r['y'][m] < THR).astype(int), score=s[m],
                                          pred=(s[m] > 0.5).astype(int), answer=np.abs(s[m] - 0.5) >= 0.10)))
    return pd.concat(rows, ignore_index=True).sort_values(['sid', 'rec', 't']).reset_index(drop=True)


def _fmt(x, nd=3):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def verdict_lines(summ, paired, vs_ema, blank):
    """L0 verdict by the planner's rule, the smoothing check against the EMA control, and the low-light answer behaviour."""
    L = ['## Verdict at L0', '',
         'Rule (planner): a win needs the paired person-bootstrap CI vs chain v1 to exclude 0, >= 5/6 people better, the worst person not below '
         "chain v1's, and the null control passed. L0 is deterministic for this module (no seed enters), so the mean over seeds equals the single run.", '']
    ch = summ.get(('chain_v1_refit', 0))
    if ch is None:
        return L
    wc = float(np.nanmin(list(ch['per'].values())))
    null_ok = blank is not None and blank['tm_bacc'] <= 0.55
    for m in MODELS:
        s = summ.get((m, 0)); pr = paired.get((m, 0))
        if s is None or pr is None:
            continue
        wm = float(np.nanmin(list(s['per'].values())))
        win = pr['lo'] > 0 and pr['n_better'] >= 5 and wm >= wc and null_ok
        L.append(f'* {m}: TM {s["tm_bacc"]:.3f} vs chain v1 {ch["tm_bacc"]:.3f}, diff {pr["diff"]:+.3f} [{pr["lo"]:+.3f}, {pr["hi"]:+.3f}], '
                 f'{pr["n_better"]}/{pr["n"]} people better, worst person {wm:.3f} vs {wc:.3f}: {"WIN" if win else "tie (no win)"}')
    L += ['', 'Is it just smoothing? Paired TM difference of each filter variant vs the nested causal EMA of chain v1 (`oxytrack_ema`) at L0:', '']
    for m, pr in vs_ema.items():
        L.append(f'* {m} - oxytrack_ema: {pr["diff"]:+.3f} [{pr["lo"]:+.3f}, {pr["hi"]:+.3f}], {pr["n_better"]}/{pr["n"]} people better')
    e = paired.get(('oxytrack_ema', 0))
    if e is not None:
        msg = ('the CI includes 0: causal smoothing of chain v1 alone does not change TM detectably' if e['lo'] <= 0 <= e['hi']
               else 'the CI excludes 0: causal smoothing of chain v1 changes TM')
        L.append(f'* oxytrack_ema - chain v1: {e["diff"]:+.3f} [{e["lo"]:+.3f}, {e["hi"]:+.3f}]; {msg}.')
    L += ['', 'Low light (simulated): coverage and accuracy of the model\'s own `answer` (chain v1 refit uses its shipped 0.10 band):', '']
    for lv in [l for l in (2, 3, 4, 5, 'X1') if ('oxytrack', l) in summ]:
        parts = []
        for m in ('oxytrack', 'oxytrack_nosnr', 'chain_v1_refit'):
            s = summ.get((m, lv))
            if s is not None:
                parts.append(f'{m} {s["coverage"]:.2f} at {_fmt(s["acc_answered"])}')
        L.append(f'* {level_tag(lv)}: ' + '; '.join(parts))
    L.append('')
    return L


def write_log(folds, summ, paired, levels_done, blank, other_null, runtime, seed, stress_note, vs_ema=None):
    vs_ema = vs_ema or {}
    L = ['# OxyTrack log (photon candidate C)', '',
         'Generated by `python -m oxydrop_ml.photon.oxytrack`. All numbers are out-of-fold (outer LOSO by person, both hands together); every choice '
         'is nested on the 5 training people (inner leave-one-training-person-out, selection metric TM bal. acc. of `pred`). Models are fit on CLEAN '
         'training people and not refit per stress level. Stress levels are simulations (python/oxydrop_ml/photon/stress.py), applied to the whole '
         f'held-out recording only. Stress seed {seed}. {stress_note}', '',
         'Model: zero-mean random-walk desaturation state + 2 per-channel offsets (3-state Kalman filter), 2 observations per window (dHR and a '
         'projection of the 19 chain v1 FUSION deltas, nested between a ridge regression onto d and the chain v1 logistic direction; the deltas '
         'already include baseline-relative breathing rate (rr_amp, rr_ibi), RMSSD and SDNN, RSA amplitude was not added), observation gain '
         'attenuated by the measured per-window pulse SNR, label-free online-EM noise scales and offsets that adapt ONLY inside the first 60 s or on '
         'windows the filter already calls clearly normal (predictive P(s<90) < 0.10). L0 has no randomness, so the seed does not change L0 numbers.', '',
         'Shipped models and smoothing: chain v1 (python/oxydrop/chain.py) is per-window with no smoothing; the hybrid (python/oxydrop_ml/hybrid.py) '
         'already applies a causal mean over k windows (tta.calibrate_and_smooth). `oxytrack_ema` is the smoothing control for chain v1.', '',
         f'Runtime of the run that wrote this file: {runtime:.0f} s.', '',
         '## Development history (disclosed researcher degrees of freedom)', '',
         'Four versions were run on the outer LOSO, each nested; the design changes were made after seeing earlier OUTER results, so the final '
         'numbers carry that (small-n) selection and are not a pre-registered test:', '',
         '* v1: fus channel = ridge projection only, decision fixed at P(s<90) > 0.5, q_mult up to 4: L0 TM 0.673 (EMA control 0.798, chain v1 refit 0.796).',
         '* v2: fus projection nested in {ridge, chain-v1 logistic direction}; decision threshold on P(s<90) nested (baselines.common.choose_thr, as '
         'the tuned baselines); q_mult up to 16: L0 TM 0.774, nosnr 0.810, noadapt 0.787, EMA 0.798. Under stress the answer rule kept 88-99 % '
         'coverage at chance accuracy (L3 0.943 at 0.536) because a wide posterior centred on the baseline still gives P(s<90) far from the tuned '
         'threshold, and the 80 % band covered only 57 % of clean windows.',
         '* v3: q_mult grid extended to 64 (the inner selection sat on the edge, 16, in every fold); information gate added to `answer` '
         f'(abstain when the posterior SD exceeds the {SD_GATE_Q:.2f} quantile of the inner-OOF SDs of the clean training people); 80 % band SD scaled '
         'so the inner-OOF band covers 80 % of the training windows. The score and pred (and so TM) are unaffected by the gate and the band scale. '
         'L0 TM 0.815 (nosnr 0.834, noadapt 0.817, EMA 0.798); q_mult = 64, again the grid edge, in every fold.',
         '* v4 (this file): q_mult grid extended to 1024 (a near-flat dynamics prior, i.e. almost no smoothing, is now inside the grid); `conf` of the '
         'filter variants ranks windows that fail the information gate below all others (coverage curves only). Declared final before its outer '
         'results were seen.', '']
    L += ['## Results by stress level', '',
          '| level | model | TM bal. acc. | TM AUROC | bal. acc. | AUROC | coverage | acc. answered | SpO2 MAE | 80 % band cover | TM diff vs chain v1 [95 % CI] | people better |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for lv in levels_done:
        for m in list(MODELS) + ['chain_v1_refit']:
            s = summ.get((m, lv))
            if s is None:
                continue
            pr = paired.get((m, lv))
            ptxt = f'{pr["diff"]:+.3f} [{pr["lo"]:+.3f}, {pr["hi"]:+.3f}]' if pr else ''
            ntxt = f'{pr["n_better"]}/{pr["n"]}' if pr else ''
            L.append(f'| {level_tag(lv)} | {m} | {_fmt(s["tm_bacc"])} | {_fmt(s["tm_auroc"])} | {_fmt(s["bacc"])} | {_fmt(s["auroc"])} | {_fmt(s["coverage"])} | '
                     f'{_fmt(s["acc_answered"])} | {_fmt(s.get("mae", np.nan), 2)} | {_fmt(s.get("band80", np.nan))} | {ptxt} | {ntxt} |')
    L += ['', '`chain_v1_refit` is chain v1 (shipped C, p > 0.5) refit inside the same outer folds by this module; the paired differences use it on the '
          'same windows and the same stress signals (2000 person-bootstrap resamples). The official comparison is the evalkit TABLE.md vs chain_v1.', '']
    L += verdict_lines(summ, paired, vs_ema, blank)
    L += ['## Null controls (L0)', '',
          f'* `oxytrack_blank__L0.csv`: the fitted fold filters fed blank observations (u = 0, SNR = training median) aligned to t: TM bal. acc. '
          f'{_fmt(blank["tm_bacc"])} (requirement <= 0.55: {"PASS" if blank["tm_bacc"] <= 0.55 else "FAIL"}). The dynamics prior is a zero-mean random walk with no '
          'drift, so with blank input the score is the same function of the window index for every person.']
    if other_null is not None:
        L.append(f'* Other-person observations aligned to t (person i scored with the observations of person i+1, same hand): TM bal. acc. '
                 f'{_fmt(other_null["oxytrack"])} for oxytrack and {_fmt(other_null["chain"])} for chain v1 refit fed the same swapped inputs. This null measures '
                 'how much of the shared Hoffman protocol timing any input-tracking model inherits; it is not specific to the filter.')
    L.append('')
    L += ['## Per-person TM bal. acc.', '', '| level | model | ' + ' | '.join(str(s) for s in sorted(folds['oxytrack'][i]['sid'] for i in range(len(folds['oxytrack'])))) + ' | worst |',
          '|---|---|' + '---|' * (len(folds['oxytrack']) + 1)]
    for lv in levels_done:
        for m in list(MODELS) + ['chain_v1_refit']:
            s = summ.get((m, lv))
            if s is None:
                continue
            per = s['per']
            L.append(f'| {level_tag(lv)} | {m} | ' + ' | '.join(_fmt(per.get(k, np.nan)) for k in sorted(per)) + f' | {_fmt(np.nanmin(list(per.values())))} |')
    L += ['', '## Nested choices per outer fold', '']
    for m in MODELS:
        L += [f'### {m}', '', '| held-out person | chosen config | threshold on score | abstain c | inner TM | inner coverage at c | 80 % band scale | SD gate (SpO2 pts) | ridge alpha | beta (hr, fus) | R diag (hr, fus) | q_emp | mu_b, sigma_b |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for fo in folds[m]:
            em = fo['em']
            if m == 'oxytrack_ema':
                L.append(f'| {fo["sid"]} | {_cfg_str(fo["cfg"])} | {fo["thr"]:.3f} | {fo["c"]:.2f} | {fo["inner_tm"]:.3f} | {fo["inner_cover"]:.3f} | | | | | | | |')
                continue
            cal = em['cal'][(fo['cfg']['proj'], 'none' if fo['cfg']['kappa'] is None else fo['cfg']['kappa'])]
            L.append(f'| {fo["sid"]} | {_cfg_str(fo["cfg"])} | {fo["thr"]:.3f} | {fo["c"]:.2f} | {fo["inner_tm"]:.3f} | {fo["inner_cover"]:.3f} | {fo["band_scale"]:.2f} | {fo["sd_max"]:.2f} | {em["alpha"]:g} | '
                     f'{cal["beta"][0]:.3f}, {cal["beta"][1]:.3f} | {cal["R"][0, 0]:.3g}, {cal["R"][1, 1]:.3g} | {em["q_emp"]:.4f} | {em["mu_b"]:.2f}, {em["sig_b"]:.2f} |')
        L.append('')
    L += ['Grids: q_mult in ' + str(Q_MULT) + ', kappa in ' + str(KAPPA) + ', rho_b in ' + str(RHO_B) + ', (nu0, eta) in ' + str(TTA)
          + ', EMA span in ' + str(EMA_SPANS) + ' windows, ridge alpha in ' + str(ALPHAS) + ' (leave-one-person-out MSE inside the training people). '
          + 'Folds whose chosen q_mult is the grid maximum: '
          + ', '.join(f'{m} {sum(fo["cfg"]["q_mult"] == max(Q_MULT) for fo in folds[m])}/{len(folds[m])}' for m in MODELS if m != 'oxytrack_ema') + '. '
          'Abstention c: baselines.common.choose_band (smallest c on 0..0.45 whose answered inner-OOF windows reach acc and bal. acc. >= 0.90, else '
          'the best answered bal. acc. with coverage >= 0.20).', '',
          'Trajectory data for the held-out person 100002 (both hands, every level run): `results/photon/oxytrack_traj_100002.csv` '
          '(posterior mean, 80 % band, P(s<90), pred, answer, SNR, adaptation gate and noise scales per window).', '']
    (RES / 'oxytrack_log.md').write_text('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description='OxyTrack: write score files for every stress level (and the out-of-family X levels)')
    ap.add_argument('--levels', nargs='*', default=None, help='e.g. 0 1 2 X1 (default: stress.LEVELS + stress.X_LEVELS)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    stress_note = ''
    if a.levels is None:
        try:
            from . import stress
            levels = list(stress.LEVELS) + list(getattr(stress, 'X_LEVELS', []))
        except Exception as e:                       # stress module not available: clean level only
            levels = [0]; stress_note = f'(stress module unavailable: {type(e).__name__}; only L0 written)'
    else:
        levels = [int(x) if str(x).isdigit() else str(x) for x in a.levels]
    RES.mkdir(parents=True, exist_ok=True)
    wdf = window_table()
    folds = get_folds()
    summ, paired, done = {}, {}, []
    blank, other_null = None, None
    vs_ema = {}
    traj = []
    for lv in levels:
        feats = features(lv, a.seed)
        ref = chain_df(folds, feats, wdf)
        summ[('chain_v1_refit', lv)] = tm_summary(ref)
        dfs = {}
        for m in MODELS:
            df = score_level(folds, m, feats, wdf)
            write_scores(df, m, lv)
            dfs[m] = df
            summ[(m, lv)] = tm_summary(df)
            paired[(m, lv)] = paired_vs(df, ref)
            if m == 'oxytrack':
                tj = df[df.sid == 100002].copy(); tj.insert(0, 'level', level_tag(lv)); traj.append(tj)
        if lv == 0:
            vs_ema = {m: paired_vs(dfs[m], dfs['oxytrack_ema']) for m in MODELS if m != 'oxytrack_ema'}
            dfb = score_level(folds, 'oxytrack', feats, wdf, blank=True)
            write_scores(dfb, 'oxytrack_blank', 0)
            blank = tm_summary(dfb)
            other_null = other_person_null(folds, feats, wdf)
        done.append(lv)
        pd.concat(traj, ignore_index=True).to_csv(RES / 'oxytrack_traj_100002.csv', index=False, float_format='%.5g')
        write_log(folds, summ, paired, done, blank, other_null, time.time() - t0, a.seed, stress_note, vs_ema)
        print(f'{level_tag(lv)}: ' + ', '.join(f'{m} TM {summ[(m, lv)]["tm_bacc"]:.3f}' for m in list(MODELS) + ['chain_v1_refit'])
              + f'  ({time.time() - t0:.0f} s)', flush=True)
    return summ


def other_person_null(folds, feats, wdf):
    """Person i is scored with the observations of the next person (same hand), aligned to t and truncated to the shorter recording."""
    recs = rec_table(feats, wdf)
    sids = sorted({r['sid'] for r in recs.values()})
    nxt = {s: sids[(i + 1) % len(sids)] for i, s in enumerate(sids)}
    out = {}
    for name in ('oxytrack', 'chain'):
        T, Y, Pr = [], [], []
        src = folds['oxytrack'] if name == 'oxytrack' else folds['oxytrack_ema']
        for fo in src:
            for hand in 'LR':
                r = recs[f'{fo["sid"]}{hand}']; o = recs[f'{nxt[fo["sid"]]}{hand}']
                n = min(len(r['t']), len(o['t']))
                feat = dict(t=r['t'][:n], D=o['D'][:n], snr=o['snr'][:n])
                s = run_filter(fo['em'], [fo['cfg']], feat)['score'][0] if name == 'oxytrack' else chain_prob(fo['chain'], o['D'][:n])
                m = eval_mask(r)[:n]
                thr = fo['thr'] if name == 'oxytrack' else 0.5
                T.append(r['t'][:n][m]); Y.append((r['y'][:n][m] < THR).astype(int)); Pr.append((s[m] > thr).astype(int))
        out[name] = M.time_matched(np.concatenate(T), np.concatenate(Y), np.concatenate(Pr))['tm_bacc']
    return out


if __name__ == '__main__':
    main()
