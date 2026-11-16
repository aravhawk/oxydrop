"""Distant-flash / low-light / oblique-illumination stress test for per-frame ROI-mean RGB recordings.

THIS IS A SIMULATION. It is not a measurement of any phone. It asks one question: how fast does a model fall apart when the flash is
farther from the lens than on the phone that made the recording (the iPhone 18 Pro / 13 Pro situation: little light survives the trip
through the finger, so the camera runs at high gain, the picture is noisy and the colours are skewed)? The Hoffman Nexus 6P recordings
are the clean level 0. Levels 1-5 degrade the WHOLE held-out recording (the label-free baseline minute included) and nothing else:
training people stay clean unless a candidate adds its own disclosed augmentation with an independent seed stream.

PIPELINE of `degrade(rgb, level, seed)` (all channels c = R, G, B; x = linear 0-255 per-frame ROI mean)
  1. split  x_c = DC_c * exp(p_c): DC_c is a centred 5 s moving average (slow part), p_c = ln(x_c / DC_c) the pulsatile part.
  2. optics (PhotonTwin, python/oxydrop_ml/optics.py): render the band-integrated camera channels at the recording geometry rho0 = RHO0 mm
     and at rho0 + PATH_SCALE * d_rho (the renderer's rho) for a nominal finger (sat 0.97, fb 0.02, fm 0.03, alpha 0.008).
       a_c = I_c(rho) / I_c(rho0)               DC attenuation (blue dies first, red last)
       m_c = |d ln I_c|(rho) / |d ln I_c|(rho0) gain of the pulsatile part (longer paths -> more modulation per unit blood swing; band shift)
     x'_c = a_c DC_c exp(m_c p_c).
  3. auto-exposure: gain g = min(1 / a_R, GAIN_CAP) restores the red DC (the gain ceiling is the phone's ISO + digital limit). Every channel is
     multiplied by g, so shot noise and read noise are amplified.
  4. oblique-illumination leak: an additive NON-pulsatile light fraction f (ambient and shunt light that never crossed the pulsing bed), with the
     white balance of the original recording, a share f of the red light after gain: leak_c = f / (1 - f) * median(g a_R DC_R) * DC_c / DC_R.
  5. noise: Gaussian with sd_c = sqrt(g (signal_c + leak_c) / K) (photon shot noise: K electrons per DN of ROI mean at unit gain, so the photon
     budget falls by a_c while the gain rises by 1 / a_R) combined with read noise g * READ_NOISE.
  6. quantisation of the per-frame ROI mean after gain to steps of QUANT_STEP = 1/32 DN and clipping to [0, 255]. The contract asks for 8-bit
     quantisation; applying a 1 DN step to a MEAN over ~10^4-10^5 dithered 8-bit pixels would be unphysical (that mean has a resolution of a small fraction
     of a DN) and would erase the 0.1-1 DN pulse of these recordings at level 1. The step 1/32 DN stands for a mean over ~10^3 pixels and is stated here as a
     deliberate deviation.
Deterministic in (level, seed, the recording): the random stream is seeded from (seed, level, a hash of the first samples of the input).
Level 0 returns the input unchanged (float64 copy).

DEVIATIONS FROM A PLAIN CALL OF PhotonTwin (stated so nobody is surprised):
  * `PhotonTwin.ln_reflectance` (optics.py) SUBTRACTS the image-source term of the Farrell et al. 1992 dipole. Farrell's Eq. 10 ADDS the two terms. With
    the subtraction the decay with rho is 10-20 orders of magnitude too fast, so this module re-evaluates the reflectance in log space with the
    correct sign (np.logaddexp) on the twin's own mu_a, mu_s', camera bands and flash spectrum. optics.py is not modified here.
  * Even with the correct sign a semi-infinite diffusion medium over-attenuates a real finger slab (the light that reaches a distant lens also travels
    through the finger thickness): +10 mm of separation would cost red about 10^-11. The separation passed to the renderer is therefore
    rho0 + PATH_SCALE * d_rho (PATH_SCALE = 0.25, fixed BEFORE any model was scored, from the input-side SNR of the recordings only) so that the five levels form a graded ladder instead of a cliff.
    The *shape* (blue dies first, modulation grows, band shift) is the renderer's; the absolute scale is a stated design choice, not a measurement.

PARAMETERS (module constants; LEVEL_INFO lists the per-level numbers): RHO0 = 3.0 mm, d_rho = 0, 2, 4, 6, 8, 10 mm, PATH_SCALE = 0.25, GAIN_CAP = 1024,
K = 2e5 e-/DN, READ_NOISE = 0.002 DN at unit gain, QUANT_STEP = 1/32 DN, leak fractions AMBIENT = 0, 0.02, 0.05, 0.10, 0.20, 0.35.
Constants were set once, from the input-side pulsatile-band SNR of the 12 Hoffman recordings (median red channel: 32, 24, 13, 0, -13 dB at levels 1-5),
before any model was scored on a degraded recording. Changing them after seeing model results would be a selection on the test people.
"""
from __future__ import annotations

import functools
import zlib

import numpy as np

LEVELS = [0, 1, 2, 3, 4, 5]
D_RHO_MM = {0: 0.0, 1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 10.0}
AMBIENT = {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.10, 4: 0.20, 5: 0.35}
RHO0 = 3.0                 # source-detector separation of the recording phone in the twin's units [mm] (optics.PRIOR_MU: ln 3)
PATH_SCALE = 0.25          # effective separation seen by the renderer = RHO0 + PATH_SCALE * d_rho (see docstring)
NOMINAL = dict(sat=0.97, fb=0.02, fm=0.03, alpha=0.008)
GAIN_CAP = 1024.0          # ISO + digital gain ceiling of the phone
K_E_PER_DN = 2.0e5         # electrons per DN of the ROI mean at unit gain
READ_NOISE = 0.002         # DN of the ROI mean at unit gain
QUANT_STEP = 1.0 / 32      # resolution of the quantised ROI mean [DN] (see docstring, step 6)
DC_WIN = 151               # samples (5 s at 30 fps, centred)


# ------------------------------------------------------------------------------------------------ optics
def _ln_farrell(twin, mua, rho):
    """ln R(rho) of Farrell et al. 1992 with the two dipole terms ADDED (see module docstring). mua [B, n_lam] 1/mm, rho [B] mm."""
    import torch
    from ..optics import A_BND
    musp = twin.mu_sp()[None]
    mut = mua + musp
    a_p = musp / mut
    z0 = 1.0 / mut
    zb = 2.0 * A_BND / (3.0 * mut)
    mueff = torch.sqrt(3.0 * mua * mut)
    r = rho[:, None]
    r1 = torch.sqrt(r * r + z0 * z0)
    r2 = torch.sqrt(r * r + (z0 + 2.0 * zb) ** 2)
    ln1 = torch.log(z0) + torch.log(mueff + 1.0 / r1) - mueff * r1 - 2.0 * torch.log(r1)
    ln2 = torch.log(z0 + 2.0 * zb) + torch.log(mueff + 1.0 / r2) - mueff * r2 - 2.0 * torch.log(r2)
    return torch.log(a_p) - np.log(4.0 * np.pi) + torch.logaddexp(ln1, ln2)


@functools.lru_cache(maxsize=None)
def _optics_table():
    """Per level: (ln a_c [3], ln m_c [3], ln DC_c at rho0 [3]) from PhotonTwin at the nominal finger. Cached."""
    import torch
    from ..optics import PhotonTwin
    twin = PhotonTwin()
    rho = torch.tensor([RHO0 + PATH_SCALE * D_RHO_MM[l] for l in LEVELS], dtype=torch.float64)
    n = len(LEVELS)
    full = lambda v: torch.full((n,), float(v), dtype=torch.float64)
    dom = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        bands = twin.bands(dom)

        def ln_i(fb):
            mua = twin.mu_a(full(NOMINAL['sat']), full(fb), full(NOMINAL['fm']))
            return torch.logsumexp(bands + _ln_farrell(twin, mua, rho)[:, None, :], dim=-1)
        l0 = ln_i(NOMINAL['fb'])
        l1 = ln_i(NOMINAL['fb'] * (1.0 + NOMINAL['alpha']))
    l0, d = l0.numpy(), (l1 - l0).abs().numpy()
    ln_a = l0 - l0[0]
    ln_m = np.log(d) - np.log(d[0])
    return {l: (ln_a[i].copy(), ln_m[i].copy()) for i, l in enumerate(LEVELS)}


def _params(level):
    ln_a, ln_m = _optics_table()[level]
    a, m = np.exp(ln_a), np.exp(ln_m)
    g = float(min(1.0 / a[0], GAIN_CAP))
    return a, m, g


def _build_info():
    out = {}
    for l in LEVELS:
        a, m, g = _params(l)
        f = AMBIENT[l]
        out[l] = dict(
            d_rho_mm=D_RHO_MM[l], light_frac_R=float(a[0]), ambient_frac=f, read_noise=float(g * READ_NOISE),
            a_c=[float(v) for v in a], m_c=[float(v) for v in m], gain=g, dc_frac_R=float(min(1.0, g * a[0])),
            desc=('clean recording (Nexus 6P geometry)' if l == 0 else
                  f'flash +{D_RHO_MM[l]:g} mm from the lens: red light x{a[0]:.2g}, G x{a[1]:.2g}, B x{a[2]:.2g}; gain x{g:.0f}; leak {100 * f:.0f} %'))
    return out


class _LazyInfo(dict):
    """LEVEL_INFO is filled on first access so that importing stress.py does not import torch."""
    _ready = False

    def _fill(self):
        if not self._ready:
            self._ready = True
            dict.update(self, _build_info())

    def __getitem__(self, k):
        self._fill(); return dict.__getitem__(self, k)

    def __iter__(self):
        self._fill(); return dict.__iter__(self)

    def __len__(self):
        self._fill(); return dict.__len__(self)

    def items(self):
        self._fill(); return dict.items(self)

    def keys(self):
        self._fill(); return dict.keys(self)

    def values(self):
        self._fill(); return dict.values(self)

    def get(self, k, d=None):
        self._fill(); return dict.get(self, k, d)

    def __contains__(self, k):
        self._fill(); return dict.__contains__(self, k)


LEVEL_INFO = _LazyInfo()


# ------------------------------------------------------------------------------------------------ degradation
def _moving_average(x, win=DC_WIN):
    """Centred moving average along axis 0 with a shrinking window at the edges. x [N, C]."""
    n = len(x)
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], 0)
    h = win // 2
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def _rng(rgb, level, seed):
    h = zlib.crc32(np.ascontiguousarray(rgb[:64], dtype=np.float64).tobytes())
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(level), int(h), len(rgb)]))


def _optics(rgb, level):
    """Steps 1-4: (signal [N, 3] after attenuation, modulation and gain, leak [3], gain g)."""
    x = np.maximum(np.asarray(rgb, np.float64), 1e-3)
    a, m, g = _params(level)
    dc = _moving_average(x)
    p = np.log(x / dc)
    sig = g * a[None] * dc * np.exp(m[None] * p)                         # steps 2-3
    f = AMBIENT[level]
    dc_r_med = np.median(dc[:, 0])
    leak = (f / (1.0 - f)) * np.median(g * a[0] * dc[:, 0]) * np.median(dc, axis=0) / dc_r_med if f > 0 else np.zeros(3)   # step 4
    return sig, leak, g


def _noise_sd(y, g):
    return np.sqrt(np.maximum(g * y, 0.0) / K_E_PER_DN + (g * READ_NOISE) ** 2)


def _quantise(y):
    return np.clip(np.rint(y / QUANT_STEP) * QUANT_STEP, 0.0, 255.0)       # step 6


def _degrade(rgb, level, seed, noise=True):
    sig, leak, g = _optics(rgb, level)
    y = sig + leak[None]
    if noise:                                                             # step 5
        y = y + _rng(rgb, level, seed).standard_normal(y.shape) * _noise_sd(y, g)
    return _quantise(y)


def degrade(rgb: np.ndarray, level: int, seed: int, fs: int = 30) -> np.ndarray:
    """Degrade a (N, 3) per-frame ROI-mean RGB recording (linear 0-255 scale of the raw recording) to stress `level` (0 = identity).
    Returns float64 (N, 3). Deterministic in (level, seed, rgb). `fs` must be 30 (the 5 s DC window is 151 samples at 30 fps)."""
    rgb = np.asarray(rgb, np.float64)
    assert rgb.ndim == 2 and rgb.shape[1] == 3, 'rgb must be (N, 3)'
    assert level in LEVELS, f'level must be one of {LEVELS}'
    if level == 0:
        return rgb.copy()
    assert fs == 30, 'the stress module is defined for 30 fps recordings'
    return _degrade(rgb, level, seed)


def snr_db(rgb, level, seed=0, fs=30):
    """Diagnostic: per-channel SNR [dB] of the pulsatile band (0.7-3.5 Hz) of ln(signal) after the optics step against the added noise, same band. Uses no labels."""
    from scipy.signal import butter, sosfiltfilt
    clean = _degrade(rgb, level, seed, noise=False) if level else np.asarray(rgb, float)
    noisy = degrade(rgb, level, seed)
    sos = butter(3, [0.7, 3.5], btype='band', fs=fs, output='sos')
    s = sosfiltfilt(sos, np.log(np.maximum(clean, 0.5)), axis=0)
    e = sosfiltfilt(sos, np.log(np.maximum(noisy, 0.5)) - np.log(np.maximum(clean, 0.5)), axis=0)
    return 20 * np.log10(s.std(0) / np.maximum(e.std(0), 1e-12))


# ------------------------------------------------------------------------------------------------ out-of-family stress
X_LEVELS = ['X1']
X_INFO = {'X1': dict(base_level=3, motion_rate_hz=1 / 20, motion_sd_common=0.08, motion_sd_channel=0.02, motion_tau_s=3.0, drop_start_p=0.01, drop_mean_len=3,
                     gamma=(0.75, 0.80, 0.85), ar1_phi=0.7, ar1_shared=0.5,
                     desc='out-of-family: level-3 optics (+6 mm), motion-like DC steps, sample-and-hold frame drops, per-channel gamma, AR(1) correlated noise')}


def degrade_x(rgb: np.ndarray, name: str = 'X1', seed: int = 0, fs: int = 30) -> np.ndarray:
    """OUT-OF-FAMILY stress `X1`: a degradation that no candidate augmentation is meant to sample (the L-family is white noise + attenuation + leak).
    A simulation, like `degrade`. Deterministic in (name, seed, rgb). Applied to the whole held-out recording. Steps:
      1. level-3 optics and leak of `degrade` (flash +6 mm: attenuation, colour skew, gain x171) WITHOUT its white noise;
      2. motion-like DC steps: Poisson events (mean one per 20 s); each multiplies the finger signal by exp(a_c exp(-(t - t0) / 3 s)) with a_c = common N(0, 0.08)
         + channel N(0, 0.02) (log scale), i.e. a step that relaxes over seconds, as a pressure or position change does;
      3. temporally correlated sensor noise: AR(1) (phi = 0.7, half of the variance shared by the three channels) with the marginal sd of the level-3 white noise,
         so its power piles up at low frequencies, inside the pulse band;
      4. random frame drops with sample-and-hold: a drop starts with p = 0.01 per frame and lasts a geometric number of frames (mean 3); the last kept sample is repeated;
      5. per-channel tone curve y -> 255 (y / 255)^gamma_c with gamma = (0.75, 0.80, 0.85) (a non-linear ISP the model never sees in training);
      6. the same quantisation and clipping as `degrade`.
    """
    from scipy.signal import lfilter
    assert name in X_LEVELS, f'name must be one of {X_LEVELS}'
    assert fs == 30, 'the stress module is defined for 30 fps recordings'
    rgb = np.asarray(rgb, np.float64)
    assert rgb.ndim == 2 and rgb.shape[1] == 3, 'rgb must be (N, 3)'
    P = X_INFO[name]
    n = len(rgb)
    rng = _rng(rgb, 100 + X_LEVELS.index(name), seed)
    sig, leak, g = _optics(rgb, P['base_level'])
    # 2. motion steps (log scale, relaxing)
    k = rng.poisson(P['motion_rate_hz'] * n / fs)
    t0 = np.sort(rng.integers(0, n, k)); tt = np.arange(n)
    mot = np.zeros((n, 3))
    for t in t0:
        amp = rng.normal(0, P['motion_sd_common']) + rng.normal(0, P['motion_sd_channel'], 3)
        mot[t:] += amp[None] * np.exp(-(tt[t:] - t)[:, None] / (P['motion_tau_s'] * fs))
    y = sig * np.exp(mot) + leak[None]
    # 3. AR(1) noise with the level-3 marginal sd
    phi = P['ar1_phi']
    e = np.sqrt(P['ar1_shared']) * rng.standard_normal((n, 1)) + np.sqrt(1 - P['ar1_shared']) * rng.standard_normal((n, 3))
    ar = lfilter([np.sqrt(1 - phi ** 2)], [1.0, -phi], e, axis=0)
    y = y + ar * _noise_sd(y, g)
    # 4. frame drops, sample-and-hold
    start = rng.random(n) < P['drop_start_p']
    drop = np.zeros(n, bool)
    for i in np.nonzero(start)[0]:
        drop[i:i + rng.geometric(1.0 / P['drop_mean_len'])] = True
    drop[0] = False
    idx = np.maximum.accumulate(np.where(drop, 0, np.arange(n)))
    y = y[idx]
    # 5. tone curve, 6. quantisation
    y = 255.0 * (np.clip(y, 0.0, 255.0) / 255.0) ** np.asarray(P['gamma'])[None]
    return _quantise(y)
