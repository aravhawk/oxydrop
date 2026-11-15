"""Causal window tensor for the OxyDrop CNN, plus the physics channels. Reference implementation mirrored by app/js/cnn_features.js.

Everything here is causal: sample n of any output depends only on input samples <= n of the same window buffer. There is no
filtfilt and no whole-record statistic. The only record-level input is the label-free personal baseline (first 60 s), passed in.

INPUT BUFFER
  rgb30  [L, 3]   per-frame mean R,G,B resampled to the uniform 30 Hz grid, linear 0..255 scale (any positive scale works).
  dt     [L]      frame-time delta in seconds seen by the resampler at each grid sample (1/30 when frames arrive on time).
  L = CTX + WIN = 450: the last WIN = 300 samples (10 s) are the window, the first CTX = 150 samples (5 s) only warm up the
  filters, so a stateless call reproduces what a streaming implementation holds. The app keeps the last 15 s and calls this
  every 2 s. Windows that start inside the first 5 s of a recording are left-padded by odd reflection (windows.gather_windows).

STEP 1  lrgb = ln(max(rgb, 1e-3) + 1)                                       (same transform as oxydrop.features.rec_feats)
STEP 2  ac   = causal band-pass of lrgb, 0.7-4 Hz, Butterworth order 3 (SOS in SOS, same coefficients as oxydrop.features),
              scipy.signal.sosfilt with initial state sosfilt_zi(SOS) * lrgb[0] per channel ("warm start"), forward only.
STEP 3  dc   = first-order IIR  y[n] = y[n-1] + a_dc (lrgb[n] - y[n-1]),  y[-1] = lrgb[0],  a_dc = 1 - exp(-1/(DC_TAU * FS)).
STEP 4  physics channels (per detected beat, causal):
          x[n]  = -(ac_R + ac_G + ac_B)[n]                      beat signal (blood absorbs light, so a pulse is a minimum of intensity;
                                                                 channels with no pulse contribute nothing)
          v[n]  = v[n-1] + a_v (x[n]^2 - v[n-1]), v[-1] = x[0]^2, a_v = 1 - exp(-1/(V_TAU * FS))     running power of x
          A sample j is a beat PEAK when x[j-1] < x[j] >= x[j+1] (confirmed one sample later, at n = j+1), and
              j - previous_peak >= MIN_GAP (10 samples, 0.33 s, i.e. at most 180 bpm), and
              x[j] - trough >= PROM_K * sqrt(v[j]),    trough = min of x over the samples since the previous peak (start of buffer for the first)
          For a confirmed beat with trough sample i_t:  amp_c = |ac_c[j] - ac_c[i_t]|  (c = R,G,B; peak-to-trough in log units = AC/DC)
              ror_RG = ln((amp_R + AMP_EPS) / (amp_G + AMP_EPS)),  ror_RB = ln((amp_R + AMP_EPS) / (amp_B + AMP_EPS)), clipped to +-ROR_CLIP.
          This is ln of the classic ratio-of-ratios (AC_R/DC_R)/(AC_G/DC_G), since ln(1+AC/DC) ~ AC/DC and amp is already in log units.
          The value is held (zero-order hold) from the confirming sample until the next beat. beat_ok[n] = 1 if a beat has been
          seen and n - last_peak <= TIMEOUT (90 samples = 3 s), else 0 (the held value is not used).
STEP 5  baseline-relative: ror_rel = (ror - base_ror) * beat_ok, dc_rel = dc - base_dc (zeros when the baseline is missing).

OUTPUT x [C=10, 300] float32 (the last 300 samples of every trace), channel order CHANNELS:
   0 ac_R  1 ac_G  2 ac_B        AC_SCALE (=100) * ac  (percent modulation in log units)
   3 dt                          clip(dt * FS - 1, -1, 8)  (0 when frames arrive on time)
   4 ror_RG_rel  5 ror_RB_rel   per-beat log ratio-of-ratios minus the personal baseline value, held, 0 where beat_ok = 0
   6 beat_ok                     0/1
   7 dc_R_rel  8 dc_G_rel  9 dc_B_rel   slow log DC (tau DC_TAU) minus the personal baseline log DC (broadcast baseline, traced DC)
 and the side vector s [S=9], SIDE order:
   d_ror_RG, d_ror_RB (window mean over beat_ok samples minus baseline, 0 if none), d_dc_R/G/B (window mean lrgb minus baseline),
   d_lac_R/G/B (ln std of the window ac minus baseline), beat_frac (mean beat_ok).
 Channels of a missing wavelength (contact red/IR data has no B) are zeroed through ch_mask and excluded from x[n].

BASELINE (per recording, label-free): median over the windows fully inside the first 60 s of the ABS features
   [dc_R, dc_G, dc_B, lac_R, lac_G, lac_B, ror_RG, ror_RB, hr]  -> vector of BASE (9) values, NaN if fewer than MIN_BASE_WIN valid windows.
   hr is 60 * FS / median inter-peak interval of the window's detected beats (beat HR, only a baseline summary, not the HR output).
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import FS, WIN, CTX

SOS = butter(3, [0.7, 4.0], btype='band', fs=FS, output='sos')      # identical to oxydrop.features.SOS_CARDIAC
_ZI = sosfilt_zi(SOS)

AC_SCALE = 100.0
DC_TAU = 3.0                    # s
V_TAU = 2.0                     # s
MIN_GAP = int(round(0.33 * FS))  # 10 samples
PROM_K = 0.8
TIMEOUT = 3 * FS
AMP_EPS = 1e-4
ROR_CLIP = 3.0
MIN_BASE_WIN = 3

CHANNELS = ['ac_R', 'ac_G', 'ac_B', 'dt', 'ror_RG_rel', 'ror_RB_rel', 'beat_ok', 'dc_R_rel', 'dc_G_rel', 'dc_B_rel']
SIDE = ['d_ror_RG', 'd_ror_RB', 'd_dc_R', 'd_dc_G', 'd_dc_B', 'd_lac_R', 'd_lac_G', 'd_lac_B', 'beat_frac']
ABS = ['dc_R', 'dc_G', 'dc_B', 'lac_R', 'lac_G', 'lac_B', 'ror_RG', 'ror_RB', 'beat_frac', 'n_beats', 'hr_beat']
BASE = ['dc_R', 'dc_G', 'dc_B', 'lac_R', 'lac_G', 'lac_B', 'ror_RG', 'ror_RB', 'hr']
C_IN, S_IN = len(CHANNELS), len(SIDE)


def log_rgb(rgb):
    """STEP 1."""
    return np.log(np.clip(np.asarray(rgb, np.float64), 1e-3, None) + 1.0)


def causal_ac(lrgb):
    """STEP 2. lrgb [..., L, 3] -> band-passed, same shape (float64). Forward-only, warm start from the first sample."""
    lrgb = np.asarray(lrgb, np.float64)
    sh = lrgb.shape
    x = np.moveaxis(lrgb.reshape(-1, sh[-2], 3), -1, 1)                  # [M, 3, L]
    zi = _ZI[:, None, None, :] * x[None, :, :, 0, None]                  # [sections, M, 3, 2]
    y, _ = sosfilt(SOS, x, axis=-1, zi=zi)
    return np.moveaxis(y, 1, -1).reshape(sh)


def ema(x, tau, axis=-2):
    """STEP 3 style first-order IIR along `axis`, initial state = first sample (warm start)."""
    from scipy.signal import lfilter
    a = 1.0 - np.exp(-1.0 / (tau * FS))
    x = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    zi = (1.0 - a) * x[..., :1]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, axis=-1, zi=zi)
    return np.moveaxis(y, -1, axis)


def beat_trace(ac, mask=None):
    """STEP 4, batched. ac [N, L, 3] (log units) -> dict(ror [N, L, 2] held per-beat ln ratios, ok [N, L] bool, peak [N, L] bool).
    One Python loop over the L time steps with numpy ops across the N windows, i.e. the per-sample state machine run on N buffers at once."""
    ac = np.asarray(ac, np.float64)
    N, L, _ = ac.shape
    if mask is None:
        mask = np.ones((N, 3), bool)
    mask = np.asarray(mask, bool)
    x = -(ac * mask[:, None, :]).sum(-1)                                   # [N, L]
    a_v = 1.0 - np.exp(-1.0 / (V_TAU * FS))
    v = x[:, 0] ** 2
    ar = np.arange(N)
    last_peak = np.full(N, -10 ** 6)
    trough = np.full(N, np.inf)
    trough_i = np.zeros(N, int)
    seen = np.zeros(N, bool)
    ror = np.zeros((N, 2))
    out_ror = np.zeros((N, L, 2)); out_ok = np.zeros((N, L), bool); out_peak = np.zeros((N, L), bool)
    use_b = mask[:, 2]
    for t in range(L):
        xt = x[:, t]
        if t >= 2:
            j = t - 1
            cand = (x[:, j - 1] < x[:, j]) & (x[:, j] >= xt) & ((j - last_peak) >= MIN_GAP) & ((x[:, j] - trough) >= PROM_K * np.sqrt(v))
            if cand.any():
                ii = np.nonzero(cand)[0]
                ti = trough_i[ii]
                amp = np.abs(ac[ii, j, :] - ac[ii, ti, :])                  # [k, 3]
                r_g = np.log((amp[:, 0] + AMP_EPS) / (amp[:, 1] + AMP_EPS))
                r_b = np.log((amp[:, 0] + AMP_EPS) / (amp[:, 2] + AMP_EPS))
                r_b = np.where(use_b[ii], r_b, 0.0)
                ror[ii, 0] = np.clip(r_g, -ROR_CLIP, ROR_CLIP)
                ror[ii, 1] = np.clip(r_b, -ROR_CLIP, ROR_CLIP)
                seen[ii] = True
                last_peak[ii] = j
                out_peak[ii, j] = True
                trough[ii] = x[ii, j]
                trough_i[ii] = j
        out_ror[:, t] = ror
        out_ok[:, t] = seen & ((t - last_peak) <= TIMEOUT)
        lower = xt < trough
        trough = np.where(lower, xt, trough)
        trough_i = np.where(lower, t, trough_i)
        v = v + a_v * (xt * xt - v)
    return dict(ror=out_ror, ok=out_ok, peak=out_peak)


def _window_hr(peak_w):
    """Beat HR (bpm) of each window from its peak flags [N, WIN]: 60 FS / median inter-peak interval; NaN with < 3 peaks."""
    N = peak_w.shape[0]
    cnt = peak_w.sum(1)
    pmax = int(max(cnt.max(), 2)) if N else 2
    order = np.argsort(~peak_w, axis=1, kind='stable')[:, :pmax]            # peak positions first, in time order
    pos = np.where(np.arange(pmax)[None, :] < cnt[:, None], order, -1).astype(float)
    ibi = np.diff(pos, axis=1)
    ibi[(pos[:, 1:] < 0) | (pos[:, :-1] < 0)] = np.nan
    ibi[(ibi < 0.3 * FS) | (ibi > 1.6 * FS)] = np.nan
    ok = np.isfinite(ibi).sum(1) >= 2
    med = np.full(N, np.nan)
    if ok.any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med[ok] = np.nanmedian(ibi[ok], axis=1)
    return 60.0 * FS / med


def windows_to_tensor(lrgb, dt, base=None, mask=None):
    """Batched reference. lrgb [N, L, 3] log-RGB (L >= WIN; the last WIN samples are the window), dt [N, L] seconds,
    base [N, 9] baseline vectors (NaN rows allowed) or None, mask [N, 3] bool wavelength-present mask.
    Returns x [N, 10, WIN] float32, s [N, 9] float32, absf [N, 11] float32 (ABS features of the window, for baselines and lag estimation)."""
    lrgb = np.asarray(lrgb, np.float64)
    N, L, _ = lrgb.shape
    assert L >= WIN, 'need at least one full window'
    m = np.ones((N, 3), bool) if mask is None else np.asarray(mask, bool)
    mf = m.astype(np.float64)
    b = np.full((N, len(BASE)), np.nan) if base is None else np.asarray(base, np.float64).reshape(N, len(BASE))
    ac = causal_ac(lrgb) * mf[:, None, :]
    dc = ema(lrgb, DC_TAU)
    bt = beat_trace(ac, m)
    sl = slice(L - WIN, L)
    acw = ac[:, sl]; ok = bt['ok'][:, sl]; ror = bt['ror'][:, sl]
    ok_f = ok.astype(np.float64)
    bnan = ~np.isfinite(b)
    b0 = np.where(bnan, 0.0, b)
    x = np.zeros((N, len(CHANNELS), WIN), np.float64)
    x[:, 0:3] = np.moveaxis(acw, -1, 1) * AC_SCALE
    x[:, 3] = np.clip(np.asarray(dt, np.float64)[:, sl] * FS - 1.0, -1.0, 8.0)
    ror_rel = (ror - b0[:, None, 6:8]) * ok_f[:, :, None]
    ror_rel *= (~bnan[:, 6:8])[:, None, :]                                   # no baseline: relative channel is 0
    ror_rel[..., 0] *= (m[:, 0] & m[:, 1])[:, None]                          # ln(R/G) needs both wavelengths
    ror_rel[..., 1] *= (m[:, 0] & m[:, 2])[:, None]
    x[:, 4] = ror_rel[..., 0]; x[:, 5] = ror_rel[..., 1]; x[:, 6] = ok_f
    dcr = dc[:, sl] - b0[:, None, 0:3]
    dcr[np.broadcast_to(bnan[:, None, 0:3], dcr.shape)] = 0.0
    x[:, 7:10] = np.moveaxis(dcr * mf[:, None, :], -1, 1)
    # ABS features of the window
    lw = lrgb[:, sl]
    dcm = lw.mean(1)
    lac = np.log(acw.std(1) + 1e-6)
    cnt = ok.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rm = (ror * ok[:, :, None]).sum(1) / cnt[:, None]
    rm[cnt == 0] = np.nan
    bf = ok_f.mean(1)
    peak_w = bt['peak'][:, sl]
    absf = np.concatenate([dcm, lac, rm, bf[:, None], peak_w.sum(1)[:, None].astype(float), _window_hr(peak_w)[:, None]], 1)
    absf[:, 0:3][~m] = np.nan; absf[:, 3:6][~m] = np.nan; absf[:, 7][~m[:, 2]] = np.nan
    # side vector
    s = np.zeros((N, len(SIDE)))
    for k in range(2):
        d = rm[:, k] - b[:, 6 + k]
        s[:, k] = np.where(np.isfinite(d), d, 0.0)
    for k in range(3):
        d = dcm[:, k] - b[:, k]; s[:, 2 + k] = np.where(np.isfinite(d) & m[:, k], d, 0.0)
        d = lac[:, k] - b[:, 3 + k]; s[:, 5 + k] = np.where(np.isfinite(d) & m[:, k], d, 0.0)
    s[:, 8] = bf
    return x.astype(np.float32), s.astype(np.float32), absf.astype(np.float32)


def window_to_tensor(rgb30, dt, baseline=None, mask=None):
    """Single-window reference used by the model and by the JS parity fixtures.
    rgb30 [L, 3] linear RGB at 30 Hz with L >= 300 (last 300 samples are the window), dt [L] seconds, baseline [9] or None.
    Returns (x [10, 300], s [9]) float32."""
    rgb30 = np.asarray(rgb30, np.float64)
    dt = np.asarray(dt, np.float64) if dt is not None else np.full(len(rgb30), 1.0 / FS)
    x, s, _ = windows_to_tensor(log_rgb(rgb30)[None], dt[None], None if baseline is None else np.asarray(baseline)[None],
                                None if mask is None else np.asarray(mask)[None])
    return x[0], s[0]


def baseline_from_abs(absf, min_windows=MIN_BASE_WIN):
    """Label-free personal baseline from the ABS features [n, 11] of the windows fully inside the first 60 s of one recording.
    Returns a [9] vector (median per component, NaN-aware); all NaN if fewer than `min_windows` windows have beats."""
    absf = np.asarray(absf, np.float64)
    out = np.full(len(BASE), np.nan)
    if len(absf) == 0:
        return out
    good = absf[:, 8] > 0.3                                                  # windows that actually contain beats
    if good.sum() < min_windows:
        return out
    import warnings
    a = absf[good]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 10]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.nanmedian(a[:, cols], axis=0)
    return out
