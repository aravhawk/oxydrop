"""Window shards: build once per dataset into the cache (outside git), read back as memory maps.

A shard directory `<CACHE>/data/<ds>/` holds
  series.f32   float32 [n_total, 3]  uniform 30 Hz log-RGB of every recording back to back (log(rgb+1); OpenOx: red, IR, 0)
  dt.f16       float16 [n_total]     frame-time delta in seconds seen by the resampler at each grid sample (1/30 when on time)
  meta.parquet one row per WINDOW (schema below)
  recs.parquet one row per RECORDING (offset, length, label-free baseline vector)
  info.json    build parameters and counts
so a window is an index range of `series`, not a copy. `WindowSet.lrgb(idx)` materialises [B, CTX + WIN, 3] buffers (the 5 s of
filter warm-up before the window included, odd-reflection padded at the start of a recording) and `WindowSet.tensors(idx)` runs the
causal reference `features.windows_to_tensor` on them.

Window meta columns (all datasets; NaN / False where a label does not exist):
  ds, subject (group id for every split), rec, t_start, end_idx (global sample index one past the window), rec_start, domain (0 phone, 1 contact),
  eval (window is scored: after the baseline span), in_baseline (window fully inside the baseline span), base_key,
  hr_ref, has_hr, spo2_ref, has_spo2, low (spo2_ref < 90 as 0/1), has_low, sao2, has_sao2 (OpenOx ABG-anchored SaO2 target), sao2_draw, has_sao2_draw,
  q_expert, has_q (BUT expert quality), skin fields (monk, ita, fitz, skin_group) where known,
  abs_<name> for features.ABS, and base_<name> for features.BASE (personal baseline of the window's recording, NaN if unavailable).
"""
import json
import os

import numpy as np
import pandas as pd

from . import CTX, DATA_DIR, FS, HOP_S, WIN, WIN_S, DOMAIN
from . import features as F

L_BUF = CTX + WIN


# --------------------------------------------------------------------------------------------- resampling
def resample_uniform(t, X, fs=FS):
    """Irregular frames (times t [n] in s, values X [n, k]) -> uniform grid. Returns (X30 [m, k], dt30 [m]).
    Values are linearly interpolated. dt30[i] is the interval between the two source frames that bracket grid sample i
    (a causal resampler emits grid sample i when the later of the two frames arrives, so it knows this interval); a dropped frame or a stall
    shows up as a run of large dt."""
    t = np.asarray(t, float); X = np.asarray(X, float)
    keep = np.concatenate([[True], np.diff(t) > 0])                     # drop non-increasing timestamps
    t, X = t[keep], X[keep]
    tt = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.stack([np.interp(tt, t, X[:, c]) for c in range(X.shape[1])], 1)
    d = np.diff(t)                                                       # d[j] = t[j+1] - t[j], the interval that brackets grid samples in [t[j], t[j+1])
    j = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(d) - 1)
    return out, d[j]


def simulate_irregular(rgb30, rng, jitter=0.15, p_stall=0.002, stall_s=(0.15, 0.6), p_drop=0.01):
    """Augmentation (training only): re-sample a clean 30 Hz trace as if the camera delivered frames at irregular times.
    Frame intervals are (1/30)(1 + jitter N(0,1)), dropped frames double an interval, rare stalls add 0.15-0.6 s. The signal is the
    linear interpolation of the clean trace at those times; then it is mapped back to the uniform grid exactly like real data.
    Returns (rgb30', dt30') of the same length (truncated / edge-held)."""
    rgb30 = np.asarray(rgb30, float); n = len(rgb30)
    m = int(n * 1.1) + 10
    iv = (1.0 / FS) * np.clip(1.0 + jitter * rng.standard_normal(m), 0.4, 3.0)
    iv = iv * np.where(rng.random(m) < p_drop, 2.0, 1.0)
    iv = iv + np.where(rng.random(m) < p_stall, rng.uniform(*stall_s, m), 0.0)
    t = np.concatenate([[0.0], np.cumsum(iv)]); t = t[t <= (n - 1) / FS]
    src = np.stack([np.interp(t, np.arange(n) / FS, rgb30[:, c]) for c in range(rgb30.shape[1])], 1)
    out, dt = resample_uniform(t, src)
    if len(out) < n:
        pad = n - len(out)
        out = np.concatenate([out, np.repeat(out[-1:], pad, 0)]); dt = np.concatenate([dt, np.full(pad, 1.0 / FS)])
    return out[:n], dt[:n]


def window_starts(n_samples, win_s=WIN_S, hop_s=HOP_S):
    """Window start times in whole seconds for a recording of n_samples at 30 Hz: 0, 2, 4, ... while start + win_s <= n // 30
    (the convention of python/oxydrop/hoffman.py)."""
    n = n_samples // FS
    return np.arange(0, n - win_s + 1, hop_s)


def mean_label(series, t_start, win_s=WIN_S, min_valid=5, hz=1):
    """Window-mean of a 1 Hz label series over [t_start, t_start + win_s), needing >= min_valid finite values (hoffman.py convention)."""
    seg = np.asarray(series, float)[int(t_start * hz):int((t_start + win_s) * hz)]
    ok = np.isfinite(seg)
    return float(seg[ok].mean()) if ok.sum() >= min_valid else np.nan


# --------------------------------------------------------------------------------------------- writing
LABEL_COLS = {
    'hr_ref': np.nan, 'has_hr': False, 'spo2_ref': np.nan, 'has_spo2': False, 'low': np.nan, 'has_low': False,
    'sao2': np.nan, 'has_sao2': False, 'sao2_draw': np.nan, 'has_sao2_draw': False, 'q_expert': np.nan, 'has_q': False,
    'monk': np.nan, 'ita': np.nan, 'fitz': np.nan, 'skin_group': '', 'eval': True, 'in_baseline': False, 'base_key': '',
}


class ShardWriter:
    """Append recordings (series + per-window frame), then `finalize()` computes window features and baselines."""

    def __init__(self, ds, out_dir=None, chunk=2048):
        self.ds = ds
        self.dir = out_dir or os.path.join(DATA_DIR, ds)
        os.makedirs(self.dir, exist_ok=True)
        for f in ('series.f32', 'dt.f16'):
            p = os.path.join(self.dir, f)
            if os.path.exists(p):
                os.remove(p)
        self._s = open(os.path.join(self.dir, 'series.f32'), 'ab')
        self._d = open(os.path.join(self.dir, 'dt.f16'), 'ab')
        self.n = 0
        self.recs = []
        self.frames = []
        self.chunk = chunk
        self.mask = {}

    def add(self, subject, rec, lrgb, dt, wins, mask=(True, True, True), **rec_meta):
        """lrgb [n, 3] float log-RGB at 30 Hz, dt [n] seconds, wins: DataFrame with at least t_start (s) and any LABEL_COLS columns.
        Windows with t_start + WIN_S > n/30 s are dropped. Returns the number of windows kept."""
        lrgb = np.asarray(lrgb, np.float32); n = len(lrgb)
        w = wins.copy()
        w = w[(w.t_start * FS + WIN <= n)].reset_index(drop=True)
        for c, v in LABEL_COLS.items():
            if c not in w:
                w[c] = v
        w['ds'] = self.ds; w['subject'] = str(subject); w['rec'] = str(rec)
        w['end_idx'] = self.n + (w.t_start.values * FS).astype(np.int64) + WIN
        w['rec_start'] = self.n
        w['domain'] = DOMAIN[self.ds]
        w['m0'], w['m1'], w['m2'] = bool(mask[0]), bool(mask[1]), bool(mask[2])
        for k, v in rec_meta.items():
            w[k] = v
        self._s.write(lrgb.tobytes()); self._d.write(np.asarray(dt, np.float16).tobytes())
        self.recs.append(dict(ds=self.ds, subject=str(subject), rec=str(rec), offset=self.n, length=n, **rec_meta))
        self.n += n
        self.frames.append(w)
        return len(w)

    def finalize(self, info=None, baseline_min_windows=F.MIN_BASE_WIN):
        self._s.close(); self._d.close()
        meta = pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()
        meta['low'] = np.where(meta.has_low, meta.low, np.nan)
        meta.to_parquet(os.path.join(self.dir, 'meta.parquet'))
        pd.DataFrame(self.recs).to_parquet(os.path.join(self.dir, 'recs.parquet'))
        json.dump(dict(ds=self.ds, n_samples=int(self.n), n_windows=int(len(meta)), ctx=CTX, win=WIN, **(info or {})), open(os.path.join(self.dir, 'info.json'), 'w'), indent=1, default=str)
        ws = WindowSet(self.ds, self.dir, load_tensors=False)
        compute_abs_and_baselines(ws, baseline_min_windows)
        return WindowSet(self.ds, self.dir)


def compute_abs_and_baselines(ws, min_windows=F.MIN_BASE_WIN, chunk=2048):
    """Add abs_* and base_* columns to the shard's meta (in place on disk)."""
    meta = ws.meta
    N = len(meta)
    A = np.full((N, len(F.ABS)), np.nan, np.float32)
    mask = meta[['m0', 'm1', 'm2']].values.astype(bool)
    for a in range(0, N, chunk):
        idx = np.arange(a, min(N, a + chunk))
        _, _, ab = F.windows_to_tensor(ws.lrgb(idx), ws.dt(idx), None, mask[idx])
        A[idx] = ab
    for k, name in enumerate(F.ABS):
        meta['abs_' + name] = A[:, k]
    B = np.full((N, len(F.BASE)), np.nan)
    keys = meta.base_key.values
    for key in pd.unique(keys):
        rows = np.nonzero(keys == key)[0]
        sel = rows[meta.in_baseline.values[rows]]
        B[rows] = F.baseline_from_abs(A[sel], min_windows)
    for k, name in enumerate(F.BASE):
        meta['base_' + name] = B[:, k]
    # a baseline is usable when every channel PRESENT in this recording has a finite baseline (masked wavelengths are ignored)
    need = np.concatenate([mask, mask, (mask[:, 0] & mask[:, 1])[:, None], (mask[:, 0] & mask[:, 2])[:, None]], 1)   # dc(3), lac(3), ror RG, ror RB
    meta['has_base'] = (np.isfinite(B[:, :8]) | ~need).all(1)
    meta.to_parquet(os.path.join(ws.dir, 'meta.parquet'))
    ws.meta = meta
    return meta


# --------------------------------------------------------------------------------------------- reading
class WindowSet:
    def __init__(self, ds, d=None, load_tensors=True):
        self.ds = ds
        self.dir = d or os.path.join(DATA_DIR, ds)
        self.meta = pd.read_parquet(os.path.join(self.dir, 'meta.parquet'))
        self.recs = pd.read_parquet(os.path.join(self.dir, 'recs.parquet'))
        info = json.load(open(os.path.join(self.dir, 'info.json')))
        n = info['n_samples']
        self.info = info
        self.series = np.memmap(os.path.join(self.dir, 'series.f32'), dtype=np.float32, mode='r', shape=(n, 3))
        self.dtv = np.memmap(os.path.join(self.dir, 'dt.f16'), dtype=np.float16, mode='r', shape=(n,))

    def __len__(self):
        return len(self.meta)

    def _idx(self, idx):
        idx = np.asarray(idx)
        e = self.meta.end_idx.values[idx]; r = self.meta.rec_start.values[idx]
        return e[:, None] - L_BUF + np.arange(L_BUF)[None, :], r

    def lrgb(self, idx):
        """[B, CTX + WIN, 3] float32 log-RGB buffers; samples before the recording start are odd-reflected about the first sample."""
        ii, r = self._idx(idx)
        below = ii < r[:, None]
        src = np.where(below, 2 * r[:, None] - ii, ii)
        v = np.asarray(self.series[src], np.float32)
        first = np.asarray(self.series[r], np.float32)
        return np.where(below[..., None], 2 * first[:, None, :] - v, v)

    def dt(self, idx):
        ii, r = self._idx(idx)
        src = np.where(ii < r[:, None], r[:, None], ii)
        return np.asarray(self.dtv[src], np.float32)

    def base(self, idx):
        return self.meta[['base_' + b for b in F.BASE]].values[np.asarray(idx)].astype(np.float32)

    def mask(self, idx):
        return self.meta[['m0', 'm1', 'm2']].values[np.asarray(idx)].astype(bool)

    def tensors(self, idx, dt_override=None, lrgb_override=None):
        """x [B, 10, 300], s [B, 9] float32 from the shard (or from overridden buffers, e.g. augmented ones)."""
        idx = np.asarray(idx)
        x, s, _ = F.windows_to_tensor(self.lrgb(idx) if lrgb_override is None else lrgb_override,
                                      self.dt(idx) if dt_override is None else dt_override, self.base(idx), self.mask(idx))
        return x, s

    def build_tensor_cache(self, chunk=2048):
        """Write x.f16 [N, 10, 300] and s.f32 [N, 9] next to the shard (training reads these as memory maps)."""
        N = len(self)
        x = np.lib.format.open_memmap(os.path.join(self.dir, 'x.f16.npy'), mode='w+', dtype=np.float16, shape=(N, F.C_IN, WIN))
        s = np.lib.format.open_memmap(os.path.join(self.dir, 's.f32.npy'), mode='w+', dtype=np.float32, shape=(N, F.S_IN))
        for a in range(0, N, chunk):
            idx = np.arange(a, min(N, a + chunk))
            xx, ss = self.tensors(idx)
            x[idx] = xx; s[idx] = ss
        x.flush(); s.flush()

    def cached(self):
        return (np.load(os.path.join(self.dir, 'x.f16.npy'), mmap_mode='r'), np.load(os.path.join(self.dir, 's.f32.npy'), mmap_mode='r'))


def load(ds):
    return WindowSet(ds)


def counts(ws):
    """Window and subject counts for DATA_MANIFEST.md."""
    m = ws.meta
    return dict(windows=int(len(m)), subjects=int(m.subject.nunique()), recordings=int(m.rec.nunique()), eval_windows=int(m['eval'].sum()),
                hr=int(m.has_hr.sum()), spo2=int(m.has_spo2.sum()), low=int(m.has_low.sum()), sao2=int(m.has_sao2.sum()), sao2_draw=int(m.has_sao2_draw.sum()),
                quality=int(m.has_q.sum()), with_baseline=int(m.has_base.sum()) if 'has_base' in m else 0)
