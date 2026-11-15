"""Device-agnostic training augmentation on raw log-RGB buffers (numpy, vectorised over the batch).

All operations act on `lrgb [B, L, 3]` (L = CTX + WIN = 450, natural-log RGB as in features.log_rgb) BEFORE the causal feature extractor runs, so
the physics channels (per-beat ratio-of-ratios, DC traces, dt) are recomputed from the perturbed trace exactly as the app would compute them.
The personal baseline vector `base [B, 9]` (features.BASE: dc_R,G,B, lac_R,G,B, ror_RG, ror_RB, hr) is transformed consistently for the
perturbations that model a DIFFERENT DEVICE (per-channel gamma and gain: the same device produced the first minute), and left untouched for
perturbations that happen WITHIN a recording (DC steps, exposure drift, motion bursts, pressure drift, saturation of a window...).

Operations (probabilities in AugCfg, all independent):
  gamma   per-channel gamma g_c ~ logN(0, s): lrgb' = g_c * lrgb. AC and DC scale by g_c. Consistent baseline: dc*g, lac + ln g, ror + ln(g_R/g_G).
  gain    per-channel multiplicative sensor gain (additive in log): lrgb' = lrgb + a_c, a_c ~ N(0, s). Consistent baseline: dc + a (device change)
          or baseline untouched (exposure drift inside the recording), 50/50.
  cmix    colour-matrix jitter: lrgb' = lrgb @ M^T, M = I + e, e ~ N(0, s) (white-balance / spectral response differences). Window-only (baseline dc
          mixed consistently; lac and ror baselines left, an approximation of the order s).
  dcstep  step in DC at a random time (light change), per-channel size ~ N(0, s).
  wander  1 or 2 sinusoidal baseline wanders (0.05 to 0.6 Hz) of amplitude ~ U(0.005, 0.05) per channel.
  irreg   frame-time irregularity: simulate_irregular (jitter, drops, stalls), resampled to the grid like real data; makes dt non-zero.
  phase   sub-sample resampling phase (linear interpolation by a fraction of a sample).
  satur   sensor saturation: clip one channel at a window quantile 0.6 to 0.97 (flattens the top of the waveform).
  motion  1 to 3 s motion burst: band-limited noise (0.5 to 6 Hz), correlated across channels, added in the last 10 s.
  press   finger-pressure drift: slow DC ramp plus slow amplitude modulation of the pulsatile part.
  noise   additive Gaussian sensor noise (sd 0.0005 to 0.003 in log units).
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import FS, WIN
from .windows import simulate_irregular


@dataclass
class AugCfg:
    p_gamma: float = 0.5
    s_gamma: float = 0.12
    p_gain: float = 0.5
    s_gain: float = 0.08
    p_cmix: float = 0.3
    s_cmix: float = 0.05
    p_dcstep: float = 0.2
    s_dcstep: float = 0.04
    p_wander: float = 0.4
    p_irreg: float = 0.5
    p_phase: float = 0.5
    p_satur: float = 0.15
    p_motion: float = 0.25
    p_press: float = 0.3
    p_noise: float = 0.4

    def to_dict(self):
        return asdict(self)


def _sel(rng, B, p):
    return rng.random(B) < p


def augment_batch(lrgb, base, mask, rng, cfg: AugCfg | None = None):
    """Returns (lrgb' [B, L, 3] float32, dt [B, L] float32 seconds, base' [B, 9] float32). `mask` [B, 3] bool: wavelengths that exist.
    The input arrays are not modified."""
    cfg = cfg or AugCfg()
    x = np.array(lrgb, np.float64)
    b = np.array(base, np.float64)
    B, L, _ = x.shape
    m = np.asarray(mask, bool)
    mf = m.astype(np.float64)
    dt = np.full((B, L), 1.0 / FS)
    t = np.arange(L) / FS
    # 1. per-channel gamma (consistent baseline)
    s = _sel(rng, B, cfg.p_gamma)
    if s.any():
        g = np.exp(cfg.s_gamma * rng.standard_normal((B, 3)))
        g = np.where(s[:, None] & m, g, 1.0)
        x = x * g[:, None, :]
        b[:, 0:3] *= g
        b[:, 3:6] += np.log(g)
        b[:, 6] += np.log(g[:, 0] / g[:, 1])
        b[:, 7] += np.log(g[:, 0] / g[:, 2])
    # 2. gain (device change: baseline follows; exposure drift: baseline fixed)
    s = _sel(rng, B, cfg.p_gain)
    if s.any():
        a = cfg.s_gain * rng.standard_normal((B, 3)) * s[:, None] * mf
        x = x + a[:, None, :]
        follow = rng.random(B) < 0.5
        b[:, 0:3] += a * follow[:, None]
    # 3. colour-matrix jitter
    s = _sel(rng, B, cfg.p_cmix)
    if s.any():
        E = cfg.s_cmix * rng.standard_normal((B, 3, 3)) * s[:, None, None]
        M = np.eye(3)[None] + E * (mf[:, :, None] * mf[:, None, :])        # only mix wavelengths that exist
        x = np.einsum('bij,blj->bli', M, x)
        b[:, 0:3] = np.einsum('bij,bj->bi', M, b[:, 0:3])
    # 4. DC step (window-only)
    s = _sel(rng, B, cfg.p_dcstep)
    if s.any():
        t0 = rng.integers(CTX_LO, L, B)
        sz = cfg.s_dcstep * rng.standard_normal((B, 3)) * mf
        step = (np.arange(L)[None, :] >= t0[:, None]).astype(np.float64)
        x = x + (step[:, :, None] * sz[:, None, :]) * s[:, None, None]
    # 5. baseline wander
    s = _sel(rng, B, cfg.p_wander)
    if s.any():
        for _ in range(2):
            f = rng.uniform(0.05, 0.6, (B, 1)); ph = rng.uniform(0, 2 * np.pi, (B, 1))
            amp = rng.uniform(0.005, 0.05, (B, 3)) * (rng.random((B, 1)) < 0.6)
            x = x + (np.sin(2 * np.pi * f * t[None, :] + ph)[:, :, None] * amp[:, None, :]) * (s[:, None, None] * mf[:, None, :])
    # 6. pressure drift: slow ramp + slow amplitude modulation of the pulsatile part
    s = _sel(rng, B, cfg.p_press)
    if s.any():
        sm = uniform_filter1d(x, size=FS, axis=1, mode='nearest')
        f = rng.uniform(0.03, 0.2, (B, 1)); ph = rng.uniform(0, 2 * np.pi, (B, 1))
        env = 1.0 + 0.3 * np.sin(2 * np.pi * f * t[None, :] + ph)
        ramp = rng.normal(0, 0.03, (B, 1)) * (t[None, :] / t[-1])
        k = s[:, None, None]
        x = np.where(k, sm + (x - sm) * env[:, :, None] + ramp[:, :, None] * mf[:, None, :], x)
    # 7. saturation clipping of one channel
    s = _sel(rng, B, cfg.p_satur)
    if s.any():
        ch = rng.integers(0, 3, B)
        q = rng.uniform(0.6, 0.97, B)
        for i in np.nonzero(s)[0]:
            if m[i, ch[i]]:
                lev = np.quantile(x[i, CTX_LO:, ch[i]], q[i])
                x[i, :, ch[i]] = np.minimum(x[i, :, ch[i]], lev)
    # 8. motion bursts
    s = _sel(rng, B, cfg.p_motion)
    if s.any():
        for i in np.nonzero(s)[0]:
            n = int(rng.uniform(1.0, 3.0) * FS)
            st = int(rng.integers(L - WIN, L - n))
            noise = rng.standard_normal(n)
            spec = np.fft.rfft(noise); fr = np.fft.rfftfreq(n, 1.0 / FS)
            spec[(fr < 0.5) | (fr > 6.0)] = 0
            nb = np.fft.irfft(spec, n); nb /= (nb.std() + 1e-9)
            win = np.hanning(n)
            x[i, st:st + n, :] += (rng.uniform(0.01, 0.05) * nb * win)[:, None] * (rng.uniform(0.5, 1.5, 3) * mf[i])[None, :]
    # 9. sensor noise
    s = _sel(rng, B, cfg.p_noise)
    if s.any():
        sd = rng.uniform(0.0005, 0.003, (B, 1, 1)) * s[:, None, None]
        x = x + sd * rng.standard_normal(x.shape) * mf[:, None, :]
    # 10. resampling phase
    s = _sel(rng, B, cfg.p_phase)
    if s.any():
        fr = (rng.random(B) * s)[:, None, None]
        xs = np.concatenate([x[:, :1], x[:, :-1]], 1)
        x = (1 - fr) * x + fr * xs
    # 11. frame-time irregularity (also fills dt)
    s = _sel(rng, B, cfg.p_irreg)
    for i in np.nonzero(s)[0]:
        jit = rng.uniform(0.05, 0.25)
        xi, dti = simulate_irregular(x[i], rng, jitter=jit, p_stall=rng.uniform(0.0, 0.004), p_drop=rng.uniform(0.0, 0.03))
        x[i] = xi; dt[i] = dti
    return x.astype(np.float32), dt.astype(np.float32), b.astype(np.float32)


CTX_LO = 150      # first sample of the scored window inside the 450-sample buffer (CTX); dc steps and clipping use the scored window


def augment_ssl_view(ac, rng):
    """Two-view augmentation for contrastive pretraining of contact PPG, on the band-passed AC windows `ac [B, 2, 300]` (red, IR; float32).
    Random: amplitude scale per channel (0.5 to 2), sign-preserving gaussian noise, circular-free time shift (+-0.7 s with edge hold), time
    warp (0.9 to 1.1x speed), channel dropout (one wavelength zeroed, p 0.1), time masking (one gap of 0.3 to 1.5 s, p 0.5), motion burst (p 0.3),
    baseline wander residual (p 0.3). Returns float32 [B, 2, 300]."""
    B, C, T = ac.shape
    x = ac.astype(np.float64).copy()
    x *= np.exp(rng.uniform(np.log(0.5), np.log(2.0), (B, C, 1)))
    sh = rng.integers(-21, 22, B)
    sp = rng.uniform(0.9, 1.1, B)
    ii = np.arange(T)[None, :] * sp[:, None] + sh[:, None]
    ii = np.clip(ii, 0, T - 1)
    i0 = np.floor(ii).astype(int); i1 = np.minimum(i0 + 1, T - 1); fr = ii - i0
    out = np.empty_like(x)
    for c in range(C):
        out[:, c] = np.take_along_axis(x[:, c], i0, 1) * (1 - fr) + np.take_along_axis(x[:, c], i1, 1) * fr
    x = out
    sd = x.std(axis=(1, 2), keepdims=True) + 1e-6
    x += rng.uniform(0.0, 0.3, (B, 1, 1)) * sd * rng.standard_normal(x.shape)
    t = np.arange(T) / FS
    for i in np.nonzero(rng.random(B) < 0.3)[0]:
        f = rng.uniform(0.1, 0.6); x[i] += rng.uniform(0.1, 0.8) * sd[i] * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))[None, :]
    for i in np.nonzero(rng.random(B) < 0.3)[0]:
        n = int(rng.uniform(1.0, 3.0) * FS); st = int(rng.integers(0, T - n))
        nb = rng.standard_normal(n); sp_ = np.fft.rfft(nb); fr_ = np.fft.rfftfreq(n, 1.0 / FS); sp_[(fr_ < 0.5) | (fr_ > 6)] = 0
        nb = np.fft.irfft(sp_, n); nb /= nb.std() + 1e-9
        x[i, :, st:st + n] += rng.uniform(0.5, 2.0) * sd[i] * (nb * np.hanning(n))[None, :]
    for i in np.nonzero(rng.random(B) < 0.5)[0]:
        n = int(rng.uniform(0.3, 1.5) * FS); st = int(rng.integers(0, T - n)); x[i, :, st:st + n] = 0
    for i in np.nonzero(rng.random(B) < 0.1)[0]:
        x[i, rng.integers(0, C)] = 0
    return x.astype(np.float32)
