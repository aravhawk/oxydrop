"""Candidate B, FlashGap: geometry-invariant training by physics-sampled distant-flash augmentation, plus calibrated selective prediction.

    cd python && python -m oxydrop_ml.photon.flashgap                    # every model below, all stress levels + X1, then results/photon/flashgap_log.md
    cd python && python -m oxydrop_ml.photon.flashgap --models flashgap --levels 0

IDEA
The shipped logistic chain v1 was fitted on one phone (Nexus 6P). A phone whose flash sits farther from the lens sees less light, a colour skew (blue dies
first), more gain noise, quantisation and a non-pulsatile leak. FlashGap keeps the chain for windows whose pulse is clean and adds an expert trained, on
TRAINING people only, for windows whose pulse is buried:
  1. augmentation: every training recording also enters training as K = 4 degraded copies drawn from a broad continuous physics prior (AUG_PRIOR) with an
     independent random stream (AUG_TAG); a copy can lose its green and/or blue channel entirely (p = 0.15 each), which teaches the model to fall back on
     red-only, heart-rate and DC cues;
  2. signal-quality (SNR) features per window: spectral SNR at the heart-rate peak per channel, absolute perfusion index and DC level per channel, a clipping
     fraction, a quantisation (flat-step) fraction per channel, the autocorrelation at the pulse period and a beat-detection fraction;
  3. an optional respiratory / autonomic feature group (planner steering, 16:49 CT): baseline-relative breathing rate from the inter-beat-interval series
     and from the red DC, RSA amplitude, 60 s RMSSD and the respiratory DC modulation (trailing 60 s, causal);
  4. expert A = the chain v1 exactly as shipped, refitted on clean training rows. Expert B = class-balanced logistic model in the chain's shape,
     p = mean(LR(dHR; C = 0.01), LR(features; C)) on clean + augmented rows (augmented rows weigh AUG_WEIGHT), optionally with feature x channel-SNR
     interaction terms (ratio-of-ratios down-weighted only when its channels are noisy) and optionally with the SNR features as main effects. LightGBM was
     dropped from the nested choice by the planner steering (LR with interactions instead);
  5. the gate (v3): either a hard gate on the beat-channel SNR (A when max(snr_R, snr_G) >= theta, B otherwise; theta = -inf is the chain itself) or a
     soft logistic stacker of logit A, logit B and their SNR interactions; chosen per outer fold as the candidate with the best augmented inner TM among
     those that keep the clean inner TM within GATE_TOL of expert A;
  6. a selective-prediction head: a logistic correctness model g = P(pred correct | |logit p|, SNR features, |logit p| x SNR), cross-fitted on the inner
     out-of-fold predictions (clean + augmented), and a threshold tau chosen on them so that the answered windows reach accuracy >= 0.90 and balanced
     accuracy >= 0.90 with a finite-sample (+1) correction (the largest such coverage, as in baselines/common.choose_band).
Everything that is chosen (expert B's feature groups, interactions, C, the gate, tau) is chosen on an inner person-grouped CV over the outer TRAINING
people (splits.inner_group_folds, k = 5 = leave-one-inner-person-out). Expert B's selection metric is the mean TM bal. acc. of its augmented inner OOF copies.
Primary output for scoring is `pred` at full coverage; `answer` is the model's own abstention.
HISTORY (disclosed): v1 (expert B alone, selection 0.5 clean + 0.5 augmented inner TM, SNR main effects always on) was scored first and lost at L0; v2
(soft stacker only) and v3 (the nested gate above, the primary `flashgap`) were designed after seeing v1's (and, for v3, v2's) held-out results. Both are kept
as score files (flashgap_v1, flashgap_v2) so the iteration is visible; their held-out numbers are post-hoc, not confirmatory.
flashgap_gate (post hoc, requested 17:15 CT after outer results were seen): the planner read the outer TABLE.csv and asked for a label-free SNR-gated mixture of
the clean chain expert A and the augmented expert B plus an abstention recalibration. Gate = hard threshold theta on the measured beat-channel spectral SNR
max(snr_R, snr_G) (theta = -inf is the chain, +inf is expert B), chosen per outer fold on the inner OOF table of clean training people and their FlashGap-prior
copies only: eligible thetas keep 0.5 clean + 0.5 augmented inner TM within GATE_TOL of the best, the pick is the highest clean inner TM (ties to the lower
theta). Abstention: tau = the highest coverage of the clean inner OOF windows whose answered accuracy (+1 corrected) is >= 0.90. Seeds flashgap_gate,
flashgap_gate_s1, flashgap_gate_s2 and the null control flashgap_gate_null. Its held-out numbers cannot be called a pre-registered win.

EVALUATION (CONTRACT.md): LOSO by person (both hands together). The model is fitted once per outer fold on clean + augmented TRAINING people and is not
refitted per level; the held-out person's two recordings are degraded with photon.stress.degrade(rgb, level, seed) (levels 1-5) or
photon.stress.degrade_x(rgb, 'X1', seed) (out-of-family stress) and scored.

DISJOINTNESS (asserted in `assert_disjoint`, tested in tests/test_photon_flashgap.py)
  * the augmentation never calls stress.degrade / degrade_x and never sees a (level, seed) pair: it draws a continuous parameter vector from AUG_PRIOR and
    applies it with its own code (`flash_augment`), seeded by SeedSequence([AUG_TAG, aug_seed, recording, copy]); evaluation streams are seeded by stress._rng
    from SeedSequence([seed, level or 100 + X index, crc, n]) with 0 <= seed < EVAL_SEED_MAX < AUG_TAG, so the first entropy word never coincides;
  * augmentation is applied to training people only; held-out recordings are only degraded by the stress module;
  * no augmentation draw equals a stress level's (d_rho, leak) pair.
OVERLAP OF PHYSICS (disclosed): the augmentation uses the same renderer as the stress test (PhotonTwin bands and absorption with the corrected Farrell sum,
stress._ln_farrell), the same kind of auto-exposure (gain restores the red DC), an additive non-pulsatile leak proportional to the recording's DC, shot +
read noise and quantisation. The prior is broader (random finger optics, separation, path scale, AE error, gain cap, electrons per DN, read noise,
quantisation step, leak tint, channel kills) but the L1-L5 stress levels sit inside it, so FlashGap's L1-L5 numbers are an in-simulator, optimistic
estimate. The X1 stress (motion-like DC steps, frame drops with sample-and-hold, per-channel tone curve, AR(1) noise) uses effects the augmentation never
samples (no DC steps, no frame drops, no tone curve, white noise only) and is the honest out-of-family check.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import time
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CACHE
from .. import metrics as M
from .. import splits as S

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
CACHE_DIR = Path(CACHE) / 'photon_flashgap'
FS, WIN, CAL = 30, 10, 60

# ------------------------------------------------------------------------------------------------ random streams and the augmentation prior
AUG_TAG = 0x46474150                # 'FGAP': first entropy word of every augmentation stream
EVAL_SEED_MAX = 1_000_000           # evaluation (stress) seeds are 0 <= seed < EVAL_SEED_MAX
assert AUG_TAG >= EVAL_SEED_MAX
K_COPIES = 4                        # degraded copies per training recording and augmentation seed
AUG_WEIGHT = 0.25                   # sample weight of an augmented row (clean rows 1): K copies weigh as much as the clean recording
AUG_PRIOR = dict(                   # (kind, lo, hi); 'u' uniform, 'lu' log-uniform, 'p' Bernoulli probability
    d_rho_mm=('u', 0.0, 12.0),      # extra flash-lens separation
    rho0_mm=('u', 2.5, 4.0),        # separation of the recording phone in the renderer's units
    path_scale=('u', 0.10, 0.30),   # renderer separation = rho0 + path_scale * d_rho (finger-slab correction, randomised)
    sat=('u', 0.85, 1.00), fb=('lu', 0.01, 0.04), fm=('lu', 0.01, 0.20), alpha=('lu', 0.004, 0.016),   # finger optics of the render
    leak=('u', 0.0, 0.40),          # non-pulsatile light share (ambient + shunt) of the red channel after gain
    leak_tint=('lu', 0.5, 2.0),     # per-channel colour of the leak relative to the recording's white balance
    read_noise=('lu', 5e-4, 1e-2),  # DN of the ROI mean at unit gain
    k_e_per_dn=('lu', 5e4, 1e6),    # electrons per DN of the ROI mean at unit gain (shot noise)
    gain_cap=('lu', 128.0, 2048.0), # AE gain ceiling
    ae_error=('lu', 0.8, 1.25),     # AE target error (red DC after gain / original)
    quant_step=('lu', 1.0 / 128, 0.25),   # resolution of the ROI mean [DN]
    kill_G=('p', 0.15), kill_B=('p', 0.15),   # channel dropout: the channel receives no tissue light (leak and noise only)
)

FUSION = ['hr', 'rmssd', 'sdnn', 'rr_amp', 'rr_ibi', 'pi_R', 'pi_G', 'pi_B', 'beat_cv', 'ror_RG', 'ror_RB', 'ror_GB',
          'sl_RG', 'sl_RB', 'sl30_RG', 'ror30_RG', 'dc_R', 'dc_G', 'dc_B']
QUAL = ['snr_R', 'snr_G', 'snr_B', 'lpi_R', 'lpi_G', 'lpi_B', 'ldc_R', 'ldc_G', 'ldc_B', 'clip', 'qz_R', 'qz_G', 'qz_B', 'acf', 'beat_frac']
RESP = ['rr_ibi60', 'rsa60', 'rmssd60', 'rr_dc60', 'resp_dc60']
COLS = FUSION + QUAL + RESP
I_FUS = list(range(len(FUSION)))
I_QUAL = list(range(len(FUSION), len(FUSION) + len(QUAL)))
I_RESP = list(range(len(FUSION) + len(QUAL), len(COLS)))
_q = {c: COLS.index(f'snr_{c}') for c in 'RGB'}
# the channel-SNR feature(s) each interacted feature depends on (min over the channels it uses; 'beat' = max(R, G): the beat channel is G or R)
INTER_Q = {'hr': 'beat', 'rmssd': 'beat', 'sdnn': 'beat', 'rr_amp': 'beat', 'rr_ibi': 'beat', 'beat_cv': 'beat', 'pi_R': 'R', 'pi_G': 'G', 'pi_B': 'B',
           'ror_RG': 'RG', 'ror_RB': 'RB', 'ror_GB': 'GB', 'sl_RG': 'RG', 'sl_RB': 'RB', 'sl30_RG': 'RG', 'ror30_RG': 'RG', 'dc_R': 'R', 'dc_G': 'G', 'dc_B': 'B',
           'rr_ibi60': 'beat', 'rsa60': 'beat', 'rmssd60': 'beat', 'rr_dc60': 'R', 'resp_dc60': 'R'}
FEAT_VERSION = 'fg3'

# nested grid (planner steering: LR with feature x SNR interactions instead of LightGBM; respiratory group nested-selectable)
GRID_C = (0.01, 0.1, 1.0)
C_HR = 0.01                          # dHR logistic regression of the shipped chain
SEL_TARGET = 0.90                    # answered accuracy and balanced accuracy target of the abstention head
SEL_MIN_ANSWERED = 200
SEL_FALLBACK_COV = 0.20
GATE_THETAS = tuple(np.round(np.linspace(-1.5, 1.5, 13), 2))   # hard-gate thresholds on the beat-channel spectral SNR (log power ratio)
GATE_TOL = 0.01                      # the gate may cost at most this much clean inner TM relative to the chain expert

POST_HOC = 'post hoc, requested 17:15 CT after outer results were seen'
STATUS_PREREG = 'pre-registered (v1, first declared)'
STATUS_ITER = 'within-feature iteration (designed after v1/v2 held-out results, before 17:15 CT)'

MODELS = {
    'flashgap':       dict(aug=True, aug_seed=0, snr=True, null=False, note='primary v3: nested gate (hard SNR gate or soft stacker) between the clean chain expert and the augmented expert (augmentation seed 0)'),
    'flashgap_s1':    dict(aug=True, aug_seed=1, snr=True, null=False, note='v3 replicate, augmentation seed 1'),
    'flashgap_s2':    dict(aug=True, aug_seed=2, snr=True, null=False, note='v3 replicate, augmentation seed 2'),
    'flashgap_noaug': dict(aug=False, aug_seed=None, snr=True, null=False, note='ablation: same features and gate, no augmentation'),
    'flashgap_nosnr': dict(aug=True, aug_seed=0, snr=False, null=False, note='ablation: no SNR features (classifier, gate and abstention head)'),
    'flashgap_null':  dict(aug=True, aug_seed=0, snr=True, null=True, note='null control: training labels permuted within each training person'),
    'flashgap_s1_null': dict(aug=True, aug_seed=1, snr=True, null=True, null_seed=1, status='extra null control, added 17:52 CT after outer results were seen',
                             note='null control of flashgap_s1 (augmentation seed 1, permutation stream 1; added 17:52 CT after flashgap_null scored 0.549)'),
    'flashgap_s2_null': dict(aug=True, aug_seed=2, snr=True, null=True, null_seed=2, status='extra null control, added 17:52 CT after outer results were seen',
                             note='null control of flashgap_s2 (augmentation seed 2, permutation stream 2; added 17:52 CT after flashgap_null scored 0.549)'),
    'flashgap_v1':    dict(aug=True, aug_seed=0, snr=True, null=False, v1=True,
                           note='v1 as first declared (augmented expert alone, criterion 0.5 clean + 0.5 aug); superseded after its L0 loss, kept for transparency'),
    'flashgap_v2':    dict(aug=True, aug_seed=0, snr=True, null=False, soft_only=True,
                           note='v2 (soft SNR-gated stacker only); superseded after its L0 result, kept for transparency'),
    # post hoc, requested 17:15 CT after outer results were seen: label-free SNR gate between the shipped chain and the augmented expert with an Occam
    # tie-break, abstention calibrated to answered accuracy >= 0.90 on the clean inner OOF windows
    'flashgap_gate':      dict(aug=True, aug_seed=0, snr=True, null=False, gate_mode='occam', sel='clean_acc', post_hoc=True,
                               note=f'{POST_HOC}: SNR-gated chain / augmented expert, augmentation seed 0'),
    'flashgap_gate_s1':   dict(aug=True, aug_seed=1, snr=True, null=False, gate_mode='occam', sel='clean_acc', post_hoc=True, note=f'{POST_HOC}: augmentation seed 1'),
    'flashgap_gate_s2':   dict(aug=True, aug_seed=2, snr=True, null=False, gate_mode='occam', sel='clean_acc', post_hoc=True, note=f'{POST_HOC}: augmentation seed 2'),
    'flashgap_gate_null': dict(aug=True, aug_seed=0, snr=True, null=True, gate_mode='occam', sel='clean_acc', post_hoc=True,
                               note=f'{POST_HOC}: null control of flashgap_gate, training labels permuted within each training person'),
    'flashgap_gate_s1_null': dict(aug=True, aug_seed=1, snr=True, null=True, null_seed=1, gate_mode='occam', sel='clean_acc', post_hoc=True,
                                  note=f'{POST_HOC}: null control of flashgap_gate_s1 (permutation stream 1; added 17:52 CT after flashgap_gate_null scored 0.551)'),
    'flashgap_gate_s2_null': dict(aug=True, aug_seed=2, snr=True, null=True, null_seed=2, gate_mode='occam', sel='clean_acc', post_hoc=True,
                                  note=f'{POST_HOC}: null control of flashgap_gate_s2 (permutation stream 2; added 17:52 CT after flashgap_gate_null scored 0.551)'),
}
for _nm, _sp in MODELS.items():
    _sp.setdefault('status', POST_HOC if _sp.get('post_hoc') else (STATUS_PREREG if _sp.get('v1') else STATUS_ITER))
del _nm, _sp
EVAL_LEVELS = [0, 1, 2, 3, 4, 5, 'X1']


def _stress():
    from . import stress
    return stress


def aug_rng(aug_seed, rec_idx, k):
    """The augmentation stream of copy k of recording rec_idx for augmentation seed aug_seed (never an evaluation stream, see assert_disjoint)."""
    return np.random.default_rng(np.random.SeedSequence([AUG_TAG, int(aug_seed), int(rec_idx), int(k)]))


def aug_key(aug_seed, rec_idx, k):
    return (AUG_TAG, int(aug_seed), int(rec_idx), int(k))


def sample_params(rng):
    """One draw of AUG_PRIOR (dict)."""
    p = {}
    for name, spec in AUG_PRIOR.items():
        if spec[0] == 'u':
            p[name] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == 'lu':
            p[name] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            p[name] = bool(rng.random() < spec[1])
    p['leak_tint'] = [float(np.exp(rng.uniform(np.log(AUG_PRIOR['leak_tint'][1]), np.log(AUG_PRIOR['leak_tint'][2])))) for _ in range(3)]
    p['leak_tint'][0] = 1.0                                  # the leak is defined as a share of the red light; its colour varies in G and B
    return p


@functools.lru_cache(maxsize=1)
def _twin():
    import torch
    from ..optics import PhotonTwin
    torch.set_num_threads(1)
    return PhotonTwin()


def optics_gain(p):
    """(a_c [3], m_c [3]) of a parameter draw: DC attenuation and pulsatile gain between rho0 and rho0 + path_scale * d_rho, rendered by PhotonTwin with the
    corrected Farrell sum (stress._ln_farrell) at the draw's finger optics."""
    import torch
    st = _stress()
    tw = _twin()
    rho = torch.tensor([p['rho0_mm'], p['rho0_mm'] + p['path_scale'] * p['d_rho_mm']], dtype=torch.float64)
    full = lambda v: torch.full((2,), float(v), dtype=torch.float64)
    with torch.no_grad():
        bands = tw.bands(torch.zeros(2, dtype=torch.long))

        def ln_i(fb):
            mua = tw.mu_a(full(p['sat']), full(fb), full(p['fm']))
            return torch.logsumexp(bands + st._ln_farrell(tw, mua, rho)[:, None, :], dim=-1)
        l0 = ln_i(p['fb']); l1 = ln_i(p['fb'] * (1.0 + p['alpha']))
    l0 = l0.numpy(); d = (l1 - torch.as_tensor(l0)).abs().numpy()
    return np.exp(l0[1] - l0[0]), d[1] / d[0]


def flash_augment(rgb, p, rng):
    """Apply one augmentation draw `p` (sample_params + optics_gain fields a_c, m_c) to a training recording rgb (N, 3), linear 0-255. Own code: no
    stress.degrade call, no (level, seed). Steps: DC / pulsatile split (5 s centred moving average), attenuation a_c and pulsatile gain m_c, channel kill,
    auto-exposure gain on red (with AE error and gain cap), tinted non-pulsatile leak, shot + read noise (white), quantisation at quant_step, clip to [0, 255]."""
    st = _stress()
    x = np.maximum(np.asarray(rgb, np.float64), 1e-3)
    a = np.asarray(p['a_c'], float); m = np.asarray(p['m_c'], float)
    dc = st._moving_average(x)
    pul = np.log(x / dc)
    g = float(min(p['ae_error'] / a[0], p['gain_cap']))
    sig = g * a[None] * dc * np.exp(m[None] * pul)
    kill = np.array([False, p['kill_G'], p['kill_B']])
    sig[:, kill] = 0.0
    f = p['leak']
    leak = (f / (1.0 - f)) * np.median(g * a[0] * dc[:, 0]) * np.median(dc, axis=0) / np.median(dc[:, 0]) * np.asarray(p['leak_tint'])
    y = sig + leak[None]
    sd = np.sqrt(np.maximum(g * y, 0.0) / p['k_e_per_dn'] + (g * p['read_noise']) ** 2)
    y = y + rng.standard_normal(y.shape) * sd
    q = p['quant_step']
    return np.clip(np.rint(y / q) * q, 0.0, 255.0)


def draw(aug_seed, rec_idx, k):
    """Parameter draw (with rendered a_c, m_c) and the rng that applies it, for copy k of recording rec_idx."""
    rng = aug_rng(aug_seed, rec_idx, k)
    p = sample_params(rng)
    a, m = optics_gain(p)
    p['a_c'] = [float(v) for v in a]; p['m_c'] = [float(v) for v in m]
    return p, rng


def assert_disjoint(aug_keys, eval_pairs, aug_params=()):
    """Raise AssertionError unless the augmentation streams and draws are disjoint from the evaluation (level, seed) pairs:
    every augmentation key starts with AUG_TAG, every evaluation seed is in [0, EVAL_SEED_MAX) (so no stress stream can start with AUG_TAG), the level
    is a stress level or an X level, and no augmentation draw reproduces a stress level's (d_rho, leak) pair."""
    st = _stress()
    xl = list(getattr(st, 'X_LEVELS', []))
    for k in aug_keys:
        assert len(k) == 4 and k[0] == AUG_TAG, f'augmentation key {k} is not in the FlashGap namespace'
    for lev, seed in eval_pairs:
        assert lev in st.LEVELS or lev in xl, f'unknown evaluation level {lev}'
        assert 0 <= int(seed) < EVAL_SEED_MAX, f'evaluation seed {seed} outside [0, {EVAL_SEED_MAX})'
        assert int(seed) != AUG_TAG
    lv = {(float(st.D_RHO_MM[l]), float(st.AMBIENT[l])) for l in st.LEVELS if l != 0}
    for p in aug_params:
        assert (float(p['d_rho_mm']), float(p['leak'])) not in lv, 'an augmentation draw equals a stress level'
    return True


# ------------------------------------------------------------------------------------------------ data and features
@functools.lru_cache(maxsize=1)
def recordings():
    """The 12 Hoffman recordings in loader order: list of dict(rec, sid, rgb, y)."""
    from oxydrop import hoffman
    from ..data.hoffman import ROOT
    return hoffman.load(ROOT)


@functools.lru_cache(maxsize=1)
def meta():
    """Evaluated windows (t >= 60 s with a label), in recording order: DataFrame rec, t, sid, y_spo2, y, ri (recording index)."""
    rows = []
    for i, r in enumerate(recordings()):
        n = len(r['rgb']) // FS
        for s in range(CAL, n - WIN + 1, 2):
            if s in r['y']:
                rows.append((r['rec'], s, int(r['sid']), float(r['y'][s]), int(r['y'][s] < 90), i))
    return pd.DataFrame(rows, columns=['rec', 't', 'sid', 'y_spo2', 'y', 'ri'])


def _ffill_base(t, X):
    """Causal forward fill, then baseline-median fill and subtraction (the chain's calibration rule)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        for i in range(1, len(col)):
            if np.isnan(col[i]):
                col[i] = col[i - 1]
    bm = t + WIN <= CAL
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base = np.nanmedian(X[bm], 0) if bm.any() else np.nanmedian(X, 0)
    base = np.where(np.isfinite(base), base, 0.0)
    return np.where(np.isnan(X), base, X) - base


def quality_feats(rgb, lg, ac, t):
    """Absolute signal-quality features [n_windows, len(QUAL)] for the 10 s windows starting at t (trailing 30 s for beats)."""
    from scipy.signal import find_peaks
    rgb = np.asarray(rgb, float)
    te = ((t + WIN) * FS).astype(int)
    idx = te[:, None] + np.arange(-WIN * FS, 0)[None]
    A = ac[idx]; A = A - A.mean(1, keepdims=True)                               # [W, 300, 3]
    R = rgb[idx]
    nf = 1024
    P = np.abs(np.fft.rfft(A * np.hanning(WIN * FS)[None, :, None], n=nf, axis=1)) ** 2
    fr = np.fft.rfftfreq(nf, 1.0 / FS)
    band = (fr >= 0.65) & (fr <= 4.0)
    tot = P[:, band].sum(1)                                                      # [W, 3]
    Pn = P / (tot[:, None, :] + 1e-30)
    Ssum = Pn.sum(2)
    ks = np.nonzero((fr >= 0.85) & (fr <= 3.0))[0]
    kb = ks[np.argmax(Ssum[:, ks] + 0.7 * Ssum[:, 2 * ks], 1)]
    w = int(round(0.10 / (FS / nf)))
    Cs = np.concatenate([np.zeros((len(P), 1, 3)), np.cumsum(P, 1)], 1)
    rs = lambda lo, hi: Cs[np.arange(len(P)), np.minimum(hi, P.shape[1]), :] - Cs[np.arange(len(P)), np.maximum(lo, 0), :]
    near = rs(kb - w, kb + w + 1) + rs(2 * kb - w, 2 * kb + w + 1)
    sd = A.std(1)
    snr = np.log((near + 1e-12) / (np.maximum(tot - near, 0) + 1e-12))
    snr = np.where(sd < 1e-6, -6.0, np.clip(snr, -6.0, 6.0))
    lpi = np.log(sd + 1e-6)
    ldc = np.log(R.mean(1) + 1.0)
    clip = ((R >= 254.5) | (R <= 0.5)).any(2).mean(1)
    qz = (np.diff(R, axis=1) == 0).mean(1)
    best = np.argmax(snr, 1)
    xb = A[np.arange(len(A)), :, best]                                            # [W, 300]
    r = np.fft.irfft(np.abs(np.fft.rfft(xb, n=1024, axis=1)) ** 2, n=1024, axis=1)[:, :WIN * FS]
    r = r / (r[:, :1] + 1e-30)
    acf = r[:, int(FS / 3.0):int(FS / 0.85) + 1].max(1)
    f0 = fr[kb]
    beat = np.zeros(len(t))
    for i, e in enumerate(te):
        a30 = ac[max(0, e - 30 * FS):e]
        s = a30.std(0)
        g = a30[:, 1] if s[1] > s[0] * 0.3 else a30[:, 0]
        x = -g
        pk, _ = find_peaks(x, distance=int(FS * 0.33), prominence=np.std(x) * 0.3 + 1e-12)
        ibi = np.diff(pk) / FS
        nv = int(((ibi > 0.3) & (ibi < 1.6)).sum())
        beat[i] = min(nv / max(f0[i] * len(a30) / FS - 1.0, 1.0), 2.0)
    return np.column_stack([snr, lpi, ldc, clip, qz, acf, beat])


def resp_feats(lg, ac, t):
    """Respiratory / autonomic group [n_windows, len(RESP)] over the trailing 60 s (causal): breathing rate of the IBI series (bpm), RSA amplitude
    (log sd of the 0.1-0.6 Hz band of the IBI series), log RMSSD, breathing rate of the red log-DC and log amplitude of its 0.1-0.6 Hz modulation."""
    from scipy.signal import find_peaks
    te = ((t + WIN) * FS).astype(int)
    out = np.full((len(t), len(RESP)), np.nan)
    for i, e in enumerate(te):
        lo = max(0, e - 60 * FS)
        a = ac[lo:e]
        s = a.std(0)
        g = a[:, 1] if s[1] > s[0] * 0.3 else a[:, 0]
        x = -g
        pk, _ = find_peaks(x, distance=int(FS * 0.33), prominence=np.std(x) * 0.3 + 1e-12)
        ibi = np.diff(pk) / FS; tt = pk[1:] / FS
        ok = (ibi > 0.3) & (ibi < 1.6)
        ibi, tt = ibi[ok], tt[ok]
        if len(ibi) >= 8:
            d = np.diff(ibi); d = d[np.abs(d) < 0.3]
            if len(d) >= 4:
                out[i, 2] = np.log(np.sqrt(np.mean(d ** 2)) + 1e-3)
            ti = np.arange(tt[0], tt[-1], 0.25)
            if len(ti) >= 32:
                sr = np.interp(ti, tt, ibi)
                sr = sr - np.polyval(np.polyfit(ti, sr, 1), ti)
                F = np.fft.rfft(sr * np.hanning(len(sr)), n=512)
                fq = np.fft.rfftfreq(512, 0.25)
                b = (fq >= 0.1) & (fq <= 0.6)
                out[i, 0] = fq[b][np.argmax(np.abs(F[b]))] * 60
                G = np.fft.rfft(sr); fg = np.fft.rfftfreq(len(sr), 0.25)
                G[(fg < 0.1) | (fg > 0.6)] = 0
                out[i, 1] = np.log(np.fft.irfft(G, len(sr)).std() + 1e-4)
        dc = lg[lo:e, 0]
        n3 = len(dc) // 10
        if n3 >= 48:
            d3 = dc[:n3 * 10].reshape(n3, 10).mean(1)
            tx = np.arange(n3) / 3.0
            d3 = d3 - np.polyval(np.polyfit(tx, d3, 1), tx)
            F = np.fft.rfft(d3 * np.hanning(n3), n=512); fq = np.fft.rfftfreq(512, 1 / 3.0)
            b = (fq >= 0.1) & (fq <= 0.6)
            out[i, 3] = fq[b][np.argmax(np.abs(F[b]))] * 60
            G = np.fft.rfft(d3); fg = np.fft.rfftfreq(n3, 1 / 3.0)
            G[(fg < 0.1) | (fg > 0.6)] = 0
            out[i, 4] = np.log(np.fft.irfft(G, n3).std() + 1e-6)
    return out


def rec_features(rgb):
    """(t, X [n_windows, len(COLS)]) for every 10 s window of a recording: chain v1 deltas, absolute quality features, baseline-relative respiratory group."""
    from oxydrop import chain
    from oxydrop.features import rec_feats
    t, D = chain.deltas(rgb)
    lg, ac = rec_feats(rgb)
    Q = quality_feats(rgb, lg, ac, t)
    Rf = _ffill_base(t, resp_feats(lg, ac, t))
    return t.astype(int), np.hstack([D, Q, Rf])


def cond_name(cond):
    if cond == 'clean':
        return 'clean'
    if cond[0] == 'aug':
        return f'aug_s{cond[1]}_k{cond[2]}'
    if cond[0] == 'L':
        return f'L{cond[1]}_seed{cond[2]}'
    return f'{cond[1]}_seed{cond[2]}'


@functools.lru_cache(maxsize=1)
def stress_hash():
    st = _stress()
    return '%08x' % zlib.crc32(Path(st.__file__).read_bytes())


def _cache_path(cond, ri):
    tag = f'{FEAT_VERSION}' + (f'_st{stress_hash()}' if cond != 'clean' and cond[0] in ('L', 'X') else '')
    return CACHE_DIR / tag / f'{cond_name(cond)}_r{ri:02d}.npz'


def degraded(cond, ri):
    """The rgb of recording ri under a condition: 'clean', ('aug', aug_seed, k), ('L', level, seed) or ('X', name, seed)."""
    rgb = recordings()[ri]['rgb']
    if cond == 'clean':
        return rgb, None
    st = _stress()
    if cond[0] == 'aug':
        p, rng = draw(cond[1], ri, cond[2])
        return flash_augment(rgb, p, rng), p
    if cond[0] == 'L':
        return st.degrade(rgb, int(cond[1]), int(cond[2])), None
    return st.degrade_x(rgb, cond[1], int(cond[2])), None


def _feat_job(args):
    cond, ri = args
    path = _cache_path(cond, ri)
    if path.exists():
        return cond, ri, None
    rgb, p = degraded(cond, ri)
    t, X = rec_features(rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, t=t, X=X, params=json.dumps(p) if p is not None else '')
    os.replace(tmp, path)
    return cond, ri, p


def compute_features(conds, jobs=3, log=print):
    """Compute (or load) the window features of every recording under every condition. Returns dict cond -> X [n_eval, len(COLS)] aligned with meta(),
    and dict (aug_seed, ri, k) -> parameter draw."""
    recs = recordings(); mt = meta()
    st = _stress()
    _ = st.LEVEL_INFO[1]                                     # warm the optics table before forking
    _twin()
    todo = [(c, ri) for c in conds for ri in range(len(recs)) if not _cache_path(c, ri).exists()]
    if todo:
        t0 = time.time()
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(jobs) as pool:
                for _ in pool.imap_unordered(_feat_job, todo, chunksize=1):
                    pass
        else:
            for a in todo:
                _feat_job(a)
        log(f'features: {len(todo)} recording-conditions in {time.time() - t0:.0f} s')
    out, params = {}, {}
    pos = {(ri, t): i for i, (ri, t) in enumerate(zip(mt.ri.values, mt.t.values))}
    for c in conds:
        X = np.full((len(mt), len(COLS)), np.nan)
        for ri in range(len(recs)):
            z = np.load(_cache_path(c, ri), allow_pickle=False)
            for ti, xi in zip(z['t'], z['X']):
                j = pos.get((ri, int(ti)))
                if j is not None:
                    X[j] = xi
            if c != 'clean' and c[0] == 'aug':
                params[(c[1], ri, c[2])] = json.loads(str(z['params']))
        assert np.isfinite(X[:, I_FUS]).all(), f'chain features missing under {cond_name(c)}'
        out[c] = X
    return out, params


# ------------------------------------------------------------------------------------------------ the classifier
@dataclass(frozen=True)
class Cfg:
    C: float
    inter: bool
    resp: bool
    snr: bool
    qmain: bool = True               # absolute SNR features as main effects of the classifier (v1: always)

    def name(self):
        return f'C={self.C:g},inter={int(self.inter)},resp={int(self.resp)},qmain={int(self.qmain and self.snr)}'


def grid(snr, v1=False):
    if v1:
        return [Cfg(c, i, r, snr, True) for i in ((False, True) if snr else (False,)) for r in (False, True) for c in GRID_C]
    return [Cfg(c, i, r, snr, qm) for qm in ((False, True) if snr else (False,)) for i in ((False, True) if snr else (False,))
            for r in (False, True) for c in GRID_C]


class Prep:
    """Median imputation, winsorisation at the 0.5 / 99.5 % training quantiles, standardisation. Fitted on training rows only."""

    def fit(self, X):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med)
        self.lo, self.hi = np.percentile(Xi, 0.5, 0), np.percentile(Xi, 99.5, 0)
        Xc = np.clip(Xi, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(0), Xc.std(0)
        self.sd = np.where(self.sd > 1e-9, self.sd, 1.0)
        return self

    def transform(self, X):
        Xi = np.where(np.isfinite(X), X, self.med)
        return (np.clip(Xi, self.lo, self.hi) - self.mu) / self.sd


def channel_q(Z):
    """Standardised channel SNRs [n, 3] and the derived quality of each interacted feature [n, len(FUSION) + len(RESP)]."""
    q = {c: Z[:, _q[c]] for c in 'RGB'}
    get = {'R': q['R'], 'G': q['G'], 'B': q['B'], 'beat': np.maximum(q['R'], q['G']), 'RG': np.minimum(q['R'], q['G']),
           'RB': np.minimum(q['R'], q['B']), 'GB': np.minimum(q['G'], q['B'])}
    return np.column_stack([get[INTER_Q[c]] for c in FUSION + RESP])


def design(Z, cfg):
    cols = list(I_FUS) + (list(I_RESP) if cfg.resp else []) + (list(I_QUAL) if cfg.snr and cfg.qmain else [])
    D = Z[:, cols]
    if cfg.inter:
        Q = channel_q(Z)
        idx = list(range(len(FUSION))) + ([len(FUSION) + j for j in range(len(RESP))] if cfg.resp else [])
        src = list(I_FUS) + (list(I_RESP) if cfg.resp else [])
        D = np.hstack([D, Z[:, src] * Q[:, idx]])
    return D


class FGModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, X, y, w):
        from sklearn.linear_model import LogisticRegression
        self.prep = Prep().fit(X)
        Z = self.prep.transform(X)
        self.lr_hr = LogisticRegression(C=C_HR, class_weight='balanced', max_iter=2000).fit(Z[:, :1], y, sample_weight=w)
        self.lr = LogisticRegression(C=self.cfg.C, class_weight='balanced', max_iter=3000).fit(design(Z, self.cfg), y, sample_weight=w)
        return self

    def predict(self, X):
        Z = self.prep.transform(X)
        return 0.5 * (self.lr_hr.predict_proba(Z[:, :1])[:, 1] + self.lr.predict_proba(design(Z, self.cfg))[:, 1])


class ChainExpert:
    """The clean expert: logistic chain v1 exactly as shipped (StandardScaler + class-balanced LR on dHR, C = 0.01, and on the 19 FUSION deltas, C = 0.1,
    probability = mean), fitted on CLEAN training rows only."""

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.m = []
        for cols, c in (([0], 0.01), (list(I_FUS), 0.1)):
            sc = StandardScaler().fit(X[:, cols])
            self.m.append((cols, sc, LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(sc.transform(X[:, cols]), y)))
        return self

    def predict(self, X):
        return np.mean([lr.predict_proba(sc.transform(X[:, cols]))[:, 1] for cols, sc, lr in self.m], 0)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Stacker:
    """SNR-gated combination of the clean expert A (chain) and the robust expert B (augmented FlashGap LR):
    logit p = b0 + a1 zA + a2 zB + qb (b1 + a3 zA + a4 zB), qb = standardised beat-channel SNR max(snr_R, snr_G); class-balanced LR (C = 1) on the inner
    out-of-fold table (clean rows weight 1, augmented rows AUG_WEIGHT). Without SNR (flashgap_nosnr): logit p = b0 + a1 zA + a2 zB."""

    def __init__(self, snr):
        self.snr = snr

    def _f(self, pa, pb, Q):
        za, zb = np.clip(_logit(pa), -8, 8), np.clip(_logit(pb), -8, 8)
        if not self.snr:
            return np.column_stack([za, zb])
        qb = (np.maximum(Q[:, 0], Q[:, 1]) - self.mu) / self.sd
        return np.column_stack([za, zb, qb, qb * za, qb * zb])

    def fit(self, pa, pb, Q, y, w):
        from sklearn.linear_model import LogisticRegression
        qb = np.maximum(Q[:, 0], Q[:, 1]); self.mu, self.sd = float(qb.mean()), float(qb.std() + 1e-9)
        self.lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(self._f(pa, pb, Q), y, sample_weight=w)
        return self

    def predict(self, pa, pb, Q):
        return self.lr.predict_proba(self._f(pa, pb, Q))[:, 1]

    def coef(self):
        return [round(float(v), 3) for v in np.r_[self.lr.intercept_, self.lr.coef_[0]]]


class Gate:
    """Hard SNR gate: the clean chain expert A where the beat-channel SNR max(snr_R, snr_G) >= theta, the augmented expert B elsewhere
    (theta = -inf: always A, i.e. the chain; +inf: always B)."""

    def __init__(self, theta):
        self.theta = float(theta)

    def predict(self, pa, pb, Q):
        if not np.isfinite(self.theta):
            return np.asarray(pa if self.theta < 0 else pb, float)
        return np.where(np.maximum(Q[:, 0], Q[:, 1]) >= self.theta, pa, pb)

    def coef(self):
        return self.theta


class System:
    """The fitted FlashGap predictor of one outer fold: expert B alone (flashgap_v1) or the SNR-gated stack of experts A and B."""

    def __init__(self, a, b, stacker):
        self.a, self.b, self.st = a, b, stacker

    def predict(self, X):
        pb = self.b.predict(X)
        if self.st is None:
            return pb
        return self.st.predict(self.a.predict(X), pb, X[:, I_QUAL])


# ------------------------------------------------------------------------------------------------ selective-prediction head
def _logit_abs(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.abs(np.log(p / (1 - p)))


class Head:
    """g = P(pred correct | |logit p|, SNR features): plain logistic regression (C = 0.1) on standardised inputs. With SNR: the 15 quality features as main
    effects plus |logit p| x (snr_R, snr_G, snr_B, acf, beat-channel SNR) interactions, so confidence can count for less when the pulse is buried."""

    def __init__(self, snr):
        self.snr = snr

    def _f(self, p, Q):
        z = _logit_abs(p)
        if not self.snr:
            return z[:, None]
        qi = np.column_stack([Q[:, 0], Q[:, 1], Q[:, 2], Q[:, QUAL.index('acf')], np.maximum(Q[:, 0], Q[:, 1])])
        return np.column_stack([z, Q, z[:, None] * qi])

    def fit(self, p, Q, y):
        from sklearn.linear_model import LogisticRegression
        F = self._f(p, Q)
        self.prep = Prep().fit(F)
        c = ((p > 0.5).astype(int) == y).astype(int)
        if len(np.unique(c)) < 2:
            self.const = float(c.mean()); return self
        self.const = None
        self.lr = LogisticRegression(C=0.1, max_iter=2000).fit(self.prep.transform(F), c)
        return self

    def predict(self, p, Q):
        if self.const is not None:
            return np.full(len(p), self.const)
        return self.lr.predict_proba(self.prep.transform(self._f(p, Q)))[:, 1]


def select_tau(g, pred, y, target=SEL_TARGET, min_answered=SEL_MIN_ANSWERED, use_bacc=True):
    """Selective risk control on inner out-of-fold windows. Windows are ranked by g; for coverages from the min_answered most confident windows up to 100 %
    in 1 % steps, the answered set passes when n_correct / (n + 1) >= target and the class-averaged n_correct_c / (n_c + 1) >= target (both classes present).
    tau = g at the LARGEST passing coverage (the shared BAND rule of baselines/common.choose_band picks the largest passing coverage too).
    If no coverage passes: tau at SEL_FALLBACK_COV coverage, certified False. Returns (tau, certified, inner coverage, inner answered accuracy)."""
    g = np.asarray(g, float); pred = np.asarray(pred).astype(int); y = np.asarray(y).astype(int)
    o = np.argsort(-g, kind='stable')
    cor = (pred[o] == y[o]).astype(float); yo = y[o]
    cc, c1, c0 = np.cumsum(cor), np.cumsum(cor * (yo == 1)), np.cumsum(cor * (yo == 0))
    n1, n0 = np.cumsum(yo == 1), np.cumsum(yo == 0)
    n = len(g)
    ks = sorted(set([min(n, min_answered)] + [int(round(f * n)) for f in np.arange(0.01, 1.0001, 0.01) if round(f * n) >= min_answered]))
    last = None
    for k in ks:
        i = k - 1
        if n1[i] > 0 and n0[i] > 0 and cc[i] / (k + 1) >= target and (not use_bacc or 0.5 * (c1[i] / (n1[i] + 1) + c0[i] / (n0[i] + 1)) >= target):
            last = k
    if last is None:
        k = max(1, int(round(SEL_FALLBACK_COV * n)))
        return float(g[o][k - 1]), False, k / n, float(cc[k - 1] / k)
    return float(g[o][last - 1]), True, last / n, float(cc[last - 1] / last)


# ------------------------------------------------------------------------------------------------ nested LOSO
def tm(t, y, pred):
    v = M.time_matched(t, y, pred)['tm_bacc']
    return float(v) if np.isfinite(v) else 0.5


def _stack(Xd, conds, rows):
    X = np.vstack([Xd[c][rows] for c in conds])
    w = np.concatenate([np.full(len(rows), 1.0 if c == 'clean' else AUG_WEIGHT) for c in conds])
    return X, w


def fit_fold(Xd, y_tr, t_tr, g_tr, tr, train_conds, spec, log=None):
    """Nested selection and the final fit for one outer fold. Xd: cond -> X [n_all, F]; tr: outer training row indices; y_tr / t_tr / g_tr: their labels
    (possibly permuted for the null control), times and people. Only these rows are used. Returns (System, head, tau, info dict).
    v1 (spec['v1']): expert B alone, selection criterion 0.5 TM(clean OOF) + 0.5 TM(aug OOF). v2 (default): expert B selected on TM(aug OOF) (TM(clean OOF)
    without augmentation), then the SNR-gated Stacker of the clean chain expert A and expert B, cross-fitted over inner people."""
    for c in train_conds:
        assert c == 'clean' or c[0] == 'aug', f'training condition {c} is not clean or FlashGap augmentation'
    v1 = spec.get('v1', False)
    inner = list(S.inner_group_folds(g_tr, k=5, seed=0, balance=y_tr))
    aug_conds = [c for c in train_conds if c != 'clean']
    tm_of = lambda oof: (tm(t_tr, y_tr, (oof['clean'] > 0.5).astype(int)),
                         float(np.mean([tm(t_tr, y_tr, (oof[c] > 0.5).astype(int)) for c in aug_conds])) if aug_conds else np.nan)
    # expert A (clean chain) inner OOF on every condition
    oofA = {c: np.full(len(tr), np.nan) for c in train_conds}
    if not v1:
        for fi in inner:
            itr, iva = tr[fi.train_idx], tr[fi.test_idx]
            m = ChainExpert().fit(Xd['clean'][itr], y_tr[fi.train_idx])
            for c in train_conds:
                oofA[c][fi.test_idx] = m.predict(Xd[c][iva])
    # expert B: nested configuration
    best, trace = None, []
    for cfg in grid(spec['snr'], v1):
        oof = {c: np.full(len(tr), np.nan) for c in train_conds}
        for fi in inner:
            itr, iva = tr[fi.train_idx], tr[fi.test_idx]
            X, w = _stack(Xd, train_conds, itr)
            m = FGModel(cfg).fit(X, np.tile(y_tr[fi.train_idx], len(train_conds)), w)
            for c in train_conds:
                oof[c][fi.test_idx] = m.predict(Xd[c][iva])
        v_clean, v_aug = tm_of(oof)
        if not aug_conds:
            crit = v_clean
        else:
            crit = 0.5 * v_clean + 0.5 * v_aug if v1 else v_aug
        trace.append(dict(cfg=cfg.name(), tm_clean=v_clean, tm_aug=v_aug, crit=crit))
        if best is None or crit > best[0] + 1e-12:
            best = (crit, cfg, oof, v_clean, v_aug)
    crit, cfg, oofB, v_clean, v_aug = best
    PA = np.concatenate([oofA[c] for c in train_conds]); PB = np.concatenate([oofB[c] for c in train_conds])
    Qm = np.vstack([Xd[c][tr][:, I_QUAL] for c in train_conds])
    Y = np.tile(y_tr, len(train_conds)); G = np.tile(g_tr, len(train_conds))
    W = np.concatenate([np.full(len(tr), 1.0 if c == 'clean' else AUG_WEIGHT) for c in train_conds])
    info = dict(cfg=cfg.name(), inner_crit=crit, inner_tm_clean=v_clean, inner_tm_aug=v_aug)
    if v1:
        P = PB; stacker = None
    else:
        Pst = np.full(len(PB), np.nan)
        for s in np.unique(G):                                       # cross-fitted soft stacker -> OOF of the stacked system
            st = Stacker(spec['snr']).fit(PA[G != s], PB[G != s], Qm[G != s], Y[G != s], W[G != s])
            Pst[G == s] = st.predict(PA[G == s], PB[G == s], Qm[G == s])
        n = len(tr)
        split = lambda P_: {c: P_[i * n:(i + 1) * n] for i, c in enumerate(train_conds)}
        a_clean, a_aug = tm_of(split(PA))
        info.update(inner_tm_chain_clean=a_clean, inner_tm_chain_aug=a_aug)
        # the gate (v3): soft stacker or a hard SNR gate (chain when the beat-channel SNR >= theta, expert B below), nested: the candidate with the best
        # augmented inner TM among those whose clean inner TM stays within GATE_TOL of the chain expert's clean inner TM (theta = -inf is the chain itself)
        cands = [('soft', None, Pst)]
        if spec.get('gate_mode') == 'occam':
            cands = []
        if not spec.get('soft_only'):
            qb = np.maximum(Qm[:, 0], Qm[:, 1]) if spec['snr'] else None
            thetas = [-np.inf, np.inf] + (list(GATE_THETAS) if spec['snr'] else [])
            for th in thetas:
                use_a = np.full(len(PA), th == -np.inf) if qb is None or not np.isfinite(th) else qb >= th
                cands.append(('gate', th, np.where(use_a, PA, PB)))
        gtrace = []
        for kind, th, Pc in cands:
            c_clean, c_aug = tm_of(split(Pc))
            ok = c_clean >= a_clean - GATE_TOL
            score = c_aug if aug_conds else c_clean
            gtrace.append(dict(kind=kind, theta=th, tm_clean=c_clean, tm_aug=c_aug, ok=ok))
        okc = [i for i, g_ in enumerate(gtrace) if g_['ok']] or list(range(len(gtrace)))
        pick = max(okc, key=lambda i: (gtrace[i]['tm_aug'] if aug_conds else gtrace[i]['tm_clean'], -i))
        if spec.get('gate_mode') == 'occam':
            # flashgap_gate (post hoc, requested 17:15 CT): hard gates only; eligible = mixture criterion 0.5 clean + 0.5 aug inner TM within GATE_TOL of the best;
            # among them the highest clean inner TM, ties -> the lower theta (more windows to the shipped chain)
            mix = [0.5 * g_['tm_clean'] + 0.5 * (g_['tm_aug'] if aug_conds else g_['tm_clean']) for g_ in gtrace]
            for g_, m_ in zip(gtrace, mix):
                g_['mix'] = m_; g_['ok'] = m_ >= max(mix) - GATE_TOL
            okc = [i for i, g_ in enumerate(gtrace) if g_['ok']]
            th_key = lambda i: -np.inf if cands[i][1] is None else cands[i][1]
            pick = max(okc, key=lambda i: (round(gtrace[i]['tm_clean'], 12), -th_key(i)))
        kind, th, P = cands[pick]
        stacker = Stacker(spec['snr']).fit(PA, PB, Qm, Y, W) if kind == 'soft' else Gate(th)
        s_clean, s_aug = gtrace[pick]['tm_clean'], gtrace[pick]['tm_aug']
        info.update(gate=('soft stacker ' + str(stacker.coef())) if kind == 'soft' else f'hard gate theta={th:g}', inner_tm_sys_clean=s_clean, inner_tm_sys_aug=s_aug,
                    stacker_coef=stacker.coef() if kind == 'soft' else th, gate_trace=gtrace)
    # abstention head: cross-fitted over inner people on the OOF table of the system
    g_oof = np.full(len(P), np.nan)
    for s in np.unique(G):
        h = Head(spec['snr']).fit(P[G != s], Qm[G != s], Y[G != s])
        g_oof[G == s] = h.predict(P[G == s], Qm[G == s])
    isc = np.concatenate([np.full(len(tr), c == 'clean') for c in train_conds])
    if spec.get('sel') == 'clean_acc':
        # flashgap_gate (post hoc, requested 17:15 CT): tau = the highest coverage of the CLEAN inner OOF windows whose answered accuracy (+1 corrected) >= 0.90;
        # the head itself is still fitted on clean + augmented rows, so g falls with the SNR on degraded windows
        tau, cert, cov_in, acc_in = select_tau(g_oof[isc], (P[isc] > 0.5).astype(int), Y[isc], use_bacc=False)
        if (~isc).any():                                         # disclosed, not used: the same tau on the augmented inner OOF windows
            aa = g_oof[~isc] >= tau
            info.update(inner_aug_coverage=float(aa.mean()),
                        inner_aug_answered_acc=float(((P[~isc][aa] > 0.5).astype(int) == Y[~isc][aa]).mean()) if aa.any() else float('nan'))
    else:
        tau, cert, cov_in, acc_in = select_tau(g_oof, (P > 0.5).astype(int), Y)
    hc = []
    for sub, msk in (('all', np.ones(len(P), bool)), ('clean', isc), ('aug', ~isc)):
        if not msk.any():
            continue
        for cv in (0.1, 0.2, 0.5, 1.0):
            o = np.argsort(-g_oof[msk])[:max(1, int(cv * msk.sum()))]
            yy, pp = Y[msk][o], (P[msk][o] > 0.5).astype(int)
            hc.append(dict(subset=sub, coverage=cv, acc=float((yy == pp).mean()), bacc=float(M.classification_metrics(yy, pp.astype(float), pred=pp)['bacc'])))
    info['head_curve'] = hc
    head = Head(spec['snr']).fit(P, Qm, Y)
    X, w = _stack(Xd, train_conds, tr)
    model_b = FGModel(cfg).fit(X, np.tile(y_tr, len(train_conds)), w)
    model_a = None if v1 else ChainExpert().fit(Xd['clean'][tr], y_tr)
    info.update(tau=tau, tau_certified=cert, inner_coverage=cov_in, inner_answered_acc=acc_in, trace=trace)
    return System(model_a, model_b, stacker), head, tau, info


def train_conds_of(spec):
    return ['clean'] + ([('aug', spec['aug_seed'], k) for k in range(K_COPIES)] if spec['aug'] else [])


def eval_cond(level, seed):
    if level == 0:
        return 'clean'
    if isinstance(level, str):
        return ('X', level, int(seed))
    return ('L', int(level), int(seed))


def fit_model(name, Xd=None, params=None, log=print):
    """Fit (or load) the six outer folds of a model. Returns list of dict(sid, model, head, tau, info)."""
    spec = MODELS[name]
    path = CACHE_DIR / 'models' / f'{FEAT_VERSION}_{name}.pkl'
    if path.exists():
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    mt = meta()
    conds = train_conds_of(spec)
    if Xd is None:
        Xd, params = compute_features(conds, log=log)
    if spec['aug']:
        keys = [aug_key(spec['aug_seed'], ri, k) for ri in range(len(recordings())) for k in range(K_COPIES)]
        assert_disjoint(keys, [], [params[(spec['aug_seed'], ri, k)] for ri in range(len(recordings())) for k in range(K_COPIES)])
    y, t, sid = mt.y.values.astype(int), mt.t.values.astype(float), mt.sid.values.astype(str)
    folds = loso_fit(Xd, y, t, sid, conds, spec, name=name, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(folds, fh)
    return folds


def null_labels(y_tr, g_tr, test_group, null_seed=0):
    """Null control: the training labels permuted within each training person (prevalence per person kept, the link to the signal destroyed).
    null_seed 0 is the first declared permutation stream; 1 and 2 are the extra permutations of the per-seed null controls."""
    key = [7, zlib.crc32(str(test_group).encode())] + ([int(null_seed)] if null_seed else [])
    rng = np.random.default_rng(key)
    y = np.asarray(y_tr).copy()
    for s in np.unique(g_tr):
        ix = np.nonzero(g_tr == s)[0]
        y[ix] = y[ix][rng.permutation(len(ix))]
    return y


def loso_fit(Xd, y, t, sid, conds, spec, name='', log=print):
    """Outer LOSO over people `sid` (both hands together). Only the training rows of a fold reach fit_fold."""
    folds = []
    for f in S.loso(sid):
        tr, te = f.train_idx, f.test_idx
        y_tr = null_labels(y[tr], sid[tr], f.test_group, spec.get('null_seed', 0)) if spec['null'] else y[tr].copy()
        t0 = time.time()
        model, head, tau, info = fit_fold(Xd, y_tr, t[tr], sid[tr], tr, conds, spec)
        info['seconds'] = round(time.time() - t0, 1)
        msg = (f'{name} fold {f.test_group}: {info["cfg"]} inner crit {info["inner_crit"]:.3f} (clean {info["inner_tm_clean"]:.3f}, aug {info["inner_tm_aug"]:.3f}); '
               f'tau {tau:.3f} ({"certified" if info["tau_certified"] else "NOT certified"}, inner coverage {info["inner_coverage"]:.2f}, acc {info["inner_answered_acc"]:.3f}); ')
        if 'stacker_coef' in info:
            msg += (f'system inner TM clean {info["inner_tm_sys_clean"]:.3f} aug {info["inner_tm_sys_aug"]:.3f} (chain {info["inner_tm_chain_clean"]:.3f} / '
                    f'{info["inner_tm_chain_aug"]:.3f}), stacker {info["stacker_coef"]}; ')
        log(msg + f'{info["seconds"]} s')
        folds.append(dict(sid=f.test_group, te=te, model=model, head=head, tau=tau, info=info))
    return folds


def score_level(folds, Xte, mt):
    """Score one test condition for every outer fold: DataFrame in the CONTRACT.md format."""
    n = len(mt)
    score, conf, ans = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    for fo in folds:
        te = fo['te']
        p = fo['model'].predict(Xte[te])
        g = fo['head'].predict(p, Xte[te][:, I_QUAL])
        score[te] = p; conf[te] = g; ans[te] = g >= fo['tau']
    assert np.isfinite(score).all()
    return pd.DataFrame(dict(rec=mt.rec.values, t=mt.t.values, sid=mt.sid.values, y_spo2=mt.y_spo2.values, y=mt.y.values, score=score,
                             pred=(score > 0.5).astype(int), answer=ans, conf=conf))


def score_model(name, levels=EVAL_LEVELS, seed=0, jobs=3, log=print):
    """Fit (cached) and score a model at every requested level. Returns dict level -> DataFrame."""
    spec = MODELS[name]
    if spec['null']:
        levels = [0]
    conds = train_conds_of(spec)
    eval_pairs = [(lv, seed) for lv in levels]
    test_conds = [eval_cond(lv, seed) for lv in levels]
    Xd, params = compute_features(list(dict.fromkeys(conds + test_conds)), jobs=jobs, log=log)
    if spec['aug']:
        keys = [aug_key(spec['aug_seed'], ri, k) for ri in range(len(recordings())) for k in range(K_COPIES)]
        assert_disjoint(keys, eval_pairs, [params[(spec['aug_seed'], ri, k)] for ri in range(len(recordings())) for k in range(K_COPIES)])
    folds = fit_model(name, Xd, params, log=log)
    mt = meta()
    return {lv: score_level(folds, Xd[c], mt) for lv, c in zip(levels, test_conds)}


def write_scores(name, res):
    SCORES.mkdir(parents=True, exist_ok=True)
    out = []
    for lv, df in res.items():
        p = SCORES / (f'{name}__L{lv}.csv' if not isinstance(lv, str) else f'{name}__{lv}.csv')
        df.to_csv(p, index=False, float_format='%.6g')
        out.append(p)
    return out


@functools.lru_cache(maxsize=None)
def _cached(name, seed):
    return score_model(name, seed=seed, log=lambda *a: None)


def run(level: int = 0, seed: int = 0, model: str = 'flashgap') -> pd.DataFrame:
    """CONTRACT.md entry point: out-of-fold FlashGap scores of the held-out people degraded to `level` with stress seed `seed`."""
    assert 0 <= int(seed) < EVAL_SEED_MAX
    return _cached(model, int(seed))[level]


# ------------------------------------------------------------------------------------------------ log
@functools.lru_cache(maxsize=2)
def _boot(B):
    """evalkit's closed-form person bootstrap on the meta() windows: same seed and people order as TABLE.csv, so the log's CIs equal evalkit's."""
    from .evalkit import Boot
    mt = meta()
    return Boot(pd.DataFrame(dict(sid=mt.sid.astype(str).values, t=mt.t.values, y=mt.y.values)), B=B)


def summarize(d, ref=None, B=2000):
    from .evalkit import ci
    bt = _boot(B)
    assert len(d) == len(meta()) and (d.t.values == meta().t.values).all()
    y, pred, sid = d.y.values, d.pred.values.astype(int), d.sid.values.astype(str)
    point, boot, per = bt.tm(pred)
    lo, hi = ci(boot)
    a = d.answer.values.astype(bool)
    r = dict(tm=point, tm_lo=lo, tm_hi=hi, auroc=M.auroc(y, d.score.values), coverage=a.mean(),
             acc_full=float((pred == y).mean()), acc_ans=float((pred[a] == y[a]).mean()) if a.any() else np.nan)
    # random abstention at the same per-person coverage (expected accuracy, exact)
    num = den = 0.0
    for s in np.unique(sid):
        m = sid == s
        k = a[m].sum(); num += k * (pred[m] == y[m]).mean(); den += k
    r['acc_random_matched'] = num / den if den else np.nan
    pp = dict(zip(bt.u, per)); r['pp'] = pp; r['worst'] = float(np.nanmin(per))
    if ref is not None:
        assert (ref.t.values == d.t.values).all()
        pb, bb, perb = bt.tm(ref.pred.values.astype(int))
        d_lo, d_hi = ci(boot - bb)
        r.update(diff=point - pb, d_lo=d_lo, d_hi=d_hi, ref_tm=pb)
        ok = np.isfinite(per) & np.isfinite(perb) & (per != perb)              # ties dropped, as in evalkit
        r['n_better'] = int((per[ok] > perb[ok]).sum()); r['n_cmp'] = int(ok.sum()); r['ref_worst'] = float(np.nanmin(perb))
        r['ref_pp'] = dict(zip(bt.u, perb))
    return r


def _load_ref(lv):
    p = SCORES / (f'chain_v1__L{lv}.csv' if not isinstance(lv, str) else f'chain_v1__{lv}.csv')
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d['rec'] = d.rec.astype(str).str.replace('hof_', '', regex=False); d['t'] = d.t.astype(int)
    mt = meta()
    d = mt[['rec', 't']].merge(d, on=['rec', 't'], how='left')
    return d if d.pred.notna().all() else None


def write_log(names, fold_info, B=2000, log=print):
    """results/photon/flashgap_log.md from the score files on disk and the fold information of this run."""
    st = _stress()
    f = lambda v, k=3: ('n/a' if v is None or not np.isfinite(v) else f'{v:.{k}f}')
    L = ['# FlashGap (candidate B) log', '',
         f'Generated by `python -m oxydrop_ml.photon.flashgap` at {time.strftime("%Y-%m-%d %H:%M %Z")}. stress.py crc32 {stress_hash()}. '
         'Primary metric: Hoffman time-matched balanced accuracy of `pred` at full coverage (metrics.time_matched), 95 % person bootstrap (2000). '
         'Stress levels L1-L5 and X1 are simulations, not measurements.', '',
         '## Model', '',
         '- Features: the 19 chain v1 FUSION deltas (dHR first), 15 absolute SNR / quality features (' + ', '.join(QUAL) + '), and the nested-selectable respiratory group ('
         + ', '.join(RESP) + ', trailing 60 s, baseline-relative).',
         f'- Expert A: the chain v1 as shipped (dHR LR C = 0.01, fusion LR C = 0.1, class-balanced), refitted on the clean training rows of the fold.',
         f'- Expert B: p = mean(LR(dHR; C = {C_HR}), LR(features; C)), class-balanced, median-imputed, winsorised at the training 0.5/99.5 % quantiles, standardised, trained on clean '
         f'rows (weight 1) + {K_COPIES} augmented copies per recording (weight {AUG_WEIGHT} each). Nested grid: C in ' + str(list(GRID_C)) + ' x SNR features as main effects {off, on} '
         'x feature x channel-SNR interactions {off, on} x respiratory group {off, on} (24 configurations; 6 without SNR). Selection: mean TM of the augmented inner OOF copies '
         '(clean inner TM without augmentation). LightGBM was dropped from the nested choice by the 16:49 CT planner steering.',
         f'- Gate (v3): hard gate on the beat-channel SNR max(snr_R, snr_G) with theta in {{-inf (chain), ' + ', '.join(f'{v:g}' for v in GATE_THETAS) + ', +inf (expert B)}} '
         f'or the soft stacker (LR on logit A, logit B, SNR and their products, cross-fitted); the candidate with the best augmented inner TM among those whose clean inner TM is within '
         f'{GATE_TOL} of expert A. Inner CV everywhere: splits.inner_group_folds k = 5 (leave-one-training-person-out), threshold 0.5.',
         f'- Abstention head: g = LR(|logit p|, SNR features, |logit p| x SNR) cross-fitted over inner people on the inner OOF table (clean + augmented); tau = largest coverage whose '
         f'answered inner windows keep accuracy and balanced accuracy >= {SEL_TARGET} after a +1 finite-sample correction (from {SEL_MIN_ANSWERED} windows, 1 % steps; fallback '
         f'{SEL_FALLBACK_COV:.0%} coverage, marked "not certified"). The guarantee assumes the held-out person is exchangeable with the inner people and that windows are independent '
         '(they overlap by 80 %), so it is a target, not a proof.',
         '- History (disclosed): v1 (expert B alone, criterion 0.5 clean + 0.5 augmented inner TM, SNR main effects always on) was the first declared model and lost at L0. '
         'v2 (soft stacker only) was designed after seeing v1 held-out results, v3 (the nested gate, primary `flashgap`) after seeing v1 and v2. flashgap_v1 / flashgap_v2 are kept '
         'as score files. v2 and v3 held-out numbers are therefore post-hoc. The abstention head gained its |logit p| x SNR terms after the inner OOF table showed that '
         'the first head ranked augmented windows above clean ones (an inner-CV observation); all models were rerun with the same head.',
         f'- **flashgap_gate ({POST_HOC}).** Same experts. Gate: hard SNR gate only (theta in {{-inf, grid, +inf}}); eligible thetas have the '
         f'mixture criterion 0.5 TM(clean inner OOF) + 0.5 TM(augmented inner OOF) within {GATE_TOL} of the best; among them the highest clean inner TM, ties to the lower theta (more '
         'windows to the shipped chain). Abstention: the same head, but tau = the highest coverage of the CLEAN inner OOF windows with answered accuracy (+1 corrected) >= 0.90 '
         '(accuracy only, no balanced-accuracy condition). The inner validation data are clean training people and their FlashGap-prior copies only; no stress.py level or seed '
         'enters any choice. Three augmentation seeds (flashgap_gate, _s1, _s2) and a null control (flashgap_gate_null).', '',
         '## Augmentation prior (training people only)', '', '| parameter | prior |', '|---|---|']
    for k, v in AUG_PRIOR.items():
        L.append(f'| {k} | ' + (f'Bernoulli({v[1]})' if v[0] == 'p' else f'{"uniform" if v[0] == "u" else "log-uniform"}({v[1]:g}, {v[2]:g})') + ' |')
    L += ['', 'Stress levels for comparison (stress.LEVEL_INFO): ' + '; '.join(f'L{l}: d_rho {st.D_RHO_MM[l]:g} mm, leak {st.AMBIENT[l]:g}' for l in st.LEVELS if l) + '. '
          f'stress.py: RHO0 {getattr(st, "RHO0", "?")}, PATH_SCALE {getattr(st, "PATH_SCALE", "?")}, GAIN_CAP {getattr(st, "GAIN_CAP", "?")}, K {getattr(st, "K_E_PER_DN", "?")}, '
          f'READ_NOISE {getattr(st, "READ_NOISE", "?")}, QUANT_STEP {getattr(st, "QUANT_STEP", "?")}.', '',
          '**Disjointness.** The augmentation draws continuous parameters with its own code (`flash_augment`), never calls `stress.degrade`/`degrade_x`, and is seeded by '
          f'SeedSequence([{AUG_TAG} (AUG_TAG), aug_seed, recording, copy]); evaluation seeds are restricted to [0, {EVAL_SEED_MAX}) so no stress stream (SeedSequence([seed, level, crc, n])) '
          'shares the first entropy word. `assert_disjoint` runs before every fit and every scoring call and is tested in tests/test_photon_flashgap.py. Augmentation touches training people only.', '',
          '**Overlap of physics (disclosed).** Same renderer (PhotonTwin bands/absorption with the corrected Farrell sum `stress._ln_farrell`), same auto-exposure idea (gain restores red), '
          'an additive non-pulsatile leak proportional to the recording DC, shot + read white noise and quantisation. The prior is broader (finger optics, rho0, path scale, AE error, gain cap, '
          'electrons/DN, read noise, quantisation step, leak tint, channel kills), but L1-L5 sit inside it: FlashGap L1-L5 results are in-simulator and optimistic. '
          'X1 (motion-like DC steps, sample-and-hold frame drops, per-channel tone curve, AR(1) noise) uses effects the augmentation never samples: it is the out-of-family check.', '']
    # results
    L += ['## Results (from the score files)', '',
          f'Status: "{STATUS_PREREG}" = flashgap_v1, the only FlashGap model declared before any FlashGap held-out number existed; "{STATUS_ITER}" = '
          'v2 / v3 (primary `flashgap`, its seeds, ablations and null control), which this feature designed after seeing v1 (and v2) held-out results, see History; '
          f'"{POST_HOC}" = the flashgap_gate family. Neither of the last two can be called a pre-registered win. flashgap_s1_null / flashgap_s2_null are extra null '
          'controls added at 17:52 CT (see the null-control lines below).', '',
          '| model | status | level | TM bal. acc. [95 % CI] | AUROC | diff vs chain_v1 [95 % CI] verdict | people better | worst person (chain) | coverage | acc answered | acc random @ same coverage |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    summ = {}
    for nm in names:
        levels = [0] if MODELS[nm]['null'] else EVAL_LEVELS
        for lv in levels:
            p = SCORES / (f'{nm}__L{lv}.csv' if not isinstance(lv, str) else f'{nm}__{lv}.csv')
            if not p.exists():
                continue
            d = pd.read_csv(p)
            ref = _load_ref(lv)
            r = summarize(d, ref, B=B)
            summ[(nm, lv)] = r
            L.append(f'| {nm} | {MODELS[nm]["status"]} | {lv} | {f(r["tm"])} [{f(r["tm_lo"])}, {f(r["tm_hi"])}] | {f(r["auroc"])} | '
                     + (f'{r["diff"]:+.3f} [{r["d_lo"]:+.3f}, {r["d_hi"]:+.3f}] {"win" if r["d_lo"] > 0 else ("loss" if r["d_hi"] < 0 else "tie")} | {r["n_better"]}/{r["n_cmp"]} | {f(r["worst"])} ({f(r["ref_worst"])}) | ' if 'diff' in r else 'n/a | n/a | ' + f(r['worst']) + ' | ')
                     + f'{f(r["coverage"], 2)} | {f(r["acc_ans"])} | {f(r["acc_random_matched"])} |')
    fams = [('flashgap', ('flashgap', 'flashgap_s1', 'flashgap_s2'), ('flashgap_null', 'flashgap_s1_null', 'flashgap_s2_null'), STATUS_ITER),
            ('flashgap_gate', ('flashgap_gate', 'flashgap_gate_s1', 'flashgap_gate_s2'), ('flashgap_gate_null', 'flashgap_gate_s1_null', 'flashgap_gate_s2_null'), POST_HOC)]
    ref0 = _load_ref(0)
    ref_cov = float(ref0.answer.astype(bool).mean()) if ref0 is not None and 'answer' in ref0 else np.nan
    for fam, members, null_nms, status in fams:
        seeds = [summ.get((nm, 0)) for nm in members]
        if not all(s is not None for s in seeds):
            continue
        L += ['', f'**{fam}, {status}: L0 TM bal. acc., mean of the 3 augmentation seeds = {np.mean([s["tm"] for s in seeds]):.3f}** '
              f'(seeds: {", ".join(f(s["tm"]) for s in seeds)}); chain_v1 L0 = {f(seeds[0].get("ref_tm", np.nan))}.']
        if 'diff' in seeds[0]:
            L.append(f'Mean paired difference vs chain_v1 at L0 over the 3 seeds: {np.mean([s["diff"] for s in seeds]):+.3f}; per seed '
                     + '; '.join(f'{s["diff"]:+.3f} [{s["d_lo"]:+.3f}, {s["d_hi"]:+.3f}], {s["n_better"]}/{s["n_cmp"]} people better, worst {f(s["worst"])} vs chain {f(s["ref_worst"])}' for s in seeds) + '.')
        L.append(f'Own abstention at L0: coverage {", ".join(f(s["coverage"], 2) for s in seeds)} (chain_v1 shipped band: {f(ref_cov, 2)}), answered accuracy '
                 f'{", ".join(f(s["acc_ans"]) for s in seeds)} (random abstention at the same per-person coverage: {", ".join(f(s["acc_random_matched"]) for s in seeds)}).')
        nl = [(nn, summ[(nn, 0)]) for nn in null_nms if (nn, 0) in summ]
        if nl:
            L.append('Null controls (training labels permuted within each training person, full pipeline, L0; evalkit flags TM > 0.55): '
                     + '; '.join(f'{nn} {f(r["tm"])} [{f(r["tm_lo"])}, {f(r["tm_hi"])}] AUROC {f(r["auroc"])} ' + ('ok' if r['tm'] <= 0.55 else 'FLAG > 0.55') for nn, r in nl)
                     + f'; mean TM {np.mean([r["tm"] for _, r in nl]):.3f}. The _s1/_s2 nulls (other permutation streams) were added at 17:52 CT after the first nulls scored '
                     '0.549 / 0.551, to show the spread of the null; none of them is dropped.')
        for lvs in (1, 2, 3, 4, 5, 'X1'):
            sl = [summ.get((nm, lvs)) for nm in members]
            if all(s is not None and 'diff' in s for s in sl):
                L.append(f'- {"L" + str(lvs) if not isinstance(lvs, str) else lvs}: mean TM over seeds {np.mean([s["tm"] for s in sl]):.3f} vs chain_v1 {f(sl[0]["ref_tm"])}; '
                         'per-seed paired difference ' + '; '.join(f'{s["diff"]:+.3f} [{s["d_lo"]:+.3f}, {s["d_hi"]:+.3f}] ({s["n_better"]}/{s["n_cmp"]} people better)' for s in sl)
                         + f'; coverage {", ".join(f(s["coverage"], 2) for s in sl)}.')
    # clean-vs-stress trade-off
    lv_names = [lv for lv in EVAL_LEVELS]
    L += ['', '## Clean-vs-stress trade-off (TM bal. acc. of `pred` at full coverage)', '', '| model | ' + ' | '.join(str(lv) for lv in lv_names) + ' |', '|---|' + '---|' * len(lv_names)]
    refrow = []
    for lv in lv_names:
        ref = _load_ref(lv)
        refrow.append(f(M.time_matched(ref.t.values, ref.y.values, ref.pred.values.astype(int))['tm_bacc']) if ref is not None else 'n/a')
    L.append('| chain_v1 | ' + ' | '.join(refrow) + ' |')
    for nm in names:
        if MODELS[nm]['null']:
            continue
        L.append(f'| {nm} | ' + ' | '.join(f(summ[(nm, lv)]['tm']) if (nm, lv) in summ else 'n/a' for lv in lv_names) + ' |')
    L += ['', 'X1 = stress.degrade_x(rgb, "X1", 0): level-3 optics + motion-like DC steps + AR(1) noise + sample-and-hold frame drops + per-channel tone curve. '
          'None of these four effects is in AUG_PRIOR (the augmentation has no DC steps, no frame drops, no tone curve and white noise only), so X1 is the out-of-family check; '
          'L1-L5 share the simulator family with the augmentation and are optimistic for FlashGap.']
    # ablations
    abl = [nm for nm in ('flashgap_noaug', 'flashgap_nosnr', 'flashgap_gate') if (nm, 0) in summ]
    if abl and ('flashgap', 0) in summ:
        L += ['', '## Ablations (TM bal. acc. minus flashgap at the same level; same augmentation seed 0)', '', '| model | ' + ' | '.join(str(lv) for lv in lv_names) + ' |',
              '|---|' + '---|' * len(lv_names)]
        for nm in abl:
            L.append(f'| {nm} | ' + ' | '.join(f'{summ[(nm, lv)]["tm"] - summ[("flashgap", lv)]["tm"]:+.3f}' if (nm, lv) in summ and ('flashgap', lv) in summ else 'n/a'
                                                for lv in lv_names) + ' |')
        L.append('flashgap_noaug: same features, experts, gate search and head, no augmented copies (expert B and the gate see clean inner windows only). flashgap_nosnr: no SNR '
                 'features anywhere (classifier, gate, head), so the gate can only pick the chain or expert B. flashgap_gate (' + POST_HOC + '): the hard-gate Occam rule and '
                 'the clean-window abstention target instead of the v3 rule.')
    # per-person at L0, L5 and X1
    for lvp in (0, 5, 'X1'):
        if ('flashgap', lvp) not in summ:
            continue
        r = summ[('flashgap', lvp)]
        pcols = [nm for nm in ('flashgap', 'flashgap_s1', 'flashgap_s2', 'flashgap_noaug', 'flashgap_nosnr', 'flashgap_gate', 'flashgap_gate_s1', 'flashgap_gate_s2')
                 if (nm, lvp) in summ]
        L += ['', f'## Per-person TM bal. acc. at {"L" + str(lvp) if not isinstance(lvp, str) else lvp} (vs chain_v1)', '',
              '| person | ' + ' | '.join(pcols) + ' | chain_v1 |', '|---|' + '---|' * (len(pcols) + 1)]
        for s in sorted(r['pp']):
            L.append(f'| {s} | ' + ' | '.join(f(summ[(nm, lvp)]['pp'].get(s, np.nan)) for nm in pcols) + f' | {f(r.get("ref_pp", {}).get(s, np.nan))} |')
    # nested choices
    L += ['', '## Nested choices per outer fold', '',
          'All choices below use the outer TRAINING people only (inner leave-one-training-person-out CV on clean rows and FlashGap-prior copies; no stress.py level, seed or X1 '
          'effect enters any choice). For the flashgap_gate family, tau is chosen on the clean inner OOF windows; the last column shows (disclosed, not used for selection) the '
          'coverage and answered accuracy the same tau gives on the augmented inner OOF windows.', '',
          '| model | held-out person | expert B configuration | B inner TM clean / aug | gate | chain inner TM clean / aug | system inner TM clean / aug | tau | certified | inner coverage | inner answered acc | aug inner coverage / acc at tau |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for nm in names:
        for fo in fold_info.get(nm, []):
            i = fo['info']
            gate = i.get('gate', 'expert B alone (v1)')
            L.append(f'| {nm} | {fo["sid"]} | {i["cfg"]} | {i["inner_tm_clean"]:.3f} / {f(i["inner_tm_aug"])} | {gate} | '
                     f'{f(i.get("inner_tm_chain_clean", np.nan))} / {f(i.get("inner_tm_chain_aug", np.nan))} | {f(i.get("inner_tm_sys_clean", np.nan))} / {f(i.get("inner_tm_sys_aug", np.nan))} | '
                     f'{i["tau"]:.3f} | {"yes" if i["tau_certified"] else "no"} | {i["inner_coverage"]:.2f} | {i["inner_answered_acc"]:.3f} | '
                     + (f'{i["inner_aug_coverage"]:.2f} / {f(i["inner_aug_answered_acc"])} |' if 'inner_aug_coverage' in i else 'n/a |'))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'flashgap_log.md').write_text('\n'.join(L) + '\n')
    rows = [dict(model=nm, sid=fo['sid'], **{k: v for k, v in fo['info'].items() if k != 'trace'}) for nm in names for fo in fold_info.get(nm, [])]
    pd.DataFrame(rows).to_csv(RES / 'flashgap_folds.csv', index=False, float_format='%.5g')
    tr = [dict(model=nm, sid=fo['sid'], **t) for nm in names for fo in fold_info.get(nm, []) for t in fo['info']['trace']]
    pd.DataFrame(tr).to_csv(RES / 'flashgap_inner_trace.csv', index=False, float_format='%.5g')
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--levels', default=None, help='comma list, e.g. 0,1,X1 (default all)')
    ap.add_argument('--seed', type=int, default=0, help='stress seed')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--no-log', action='store_true')
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args(argv)
    os.environ.setdefault('OMP_NUM_THREADS', '3')
    names = [n for n in a.models.split(',') if n]
    levels = EVAL_LEVELS if a.levels is None else [(int(v) if v.isdigit() else v) for v in a.levels.split(',')]
    info = {}
    for nm in names:
        t0 = time.time()
        res = score_model(nm, levels=levels, seed=a.seed, jobs=a.jobs)
        for p in write_scores(nm, res):
            print('wrote', p.relative_to(REPO))
        info[nm] = [dict(sid=fo['sid'], info=fo['info']) for fo in fit_model(nm)]
        print(f'{nm}: {time.time() - t0:.0f} s', flush=True)
    if not a.no_log:
        allnames = [n for n in MODELS if (CACHE_DIR / 'models' / f'{FEAT_VERSION}_{n}.pkl').exists()]
        info = {n: [dict(sid=fo['sid'], info=fo['info']) for fo in fit_model(n)] for n in allnames}
        write_log(allnames, info, B=a.B)
        print('wrote', (RES / 'flashgap_log.md').relative_to(REPO))


if __name__ == '__main__':
    # import under the package name so the pickled fold models refer to oxydrop_ml.photon.flashgap, not __main__
    import importlib
    importlib.import_module('oxydrop_ml.photon.flashgap').main()
