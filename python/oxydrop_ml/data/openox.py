"""OpenOximetry Repository v1.1.1 (PhysioNet, CREDENTIALED ACCESS, data use agreement): healthy volunteers on a staircase hypoxia protocol at UCSF,
arterial blood gas (ABG) SaO2 as the reference, finger-clip CONTACT red/infrared PPG at 86 Hz, 2 Hz reference-device CSVs, skin-tone annotations.
It is NOT a phone-camera dataset. Row-level data and every per-window array built here stay in the cache directory: never commit, never upload to
Kaggle (Daytona only if compute is needed).

What this builds (shard `openox`, 307 PPG encounters, 139 people):
  * PPG -> the 30 Hz grid: red -> channel R, IR -> channel G (second wavelength; channel B absent, mask m2 = False, domain = 1). Raw counts are
    resampled with scipy.signal.resample_poly (anti-alias FIR, 86 -> 30 Hz = 15/43) BEFORE the log transform; dt is nominal.
  * time axis: PPG time = seconds since the record header start. The 2 Hz reference clock is joined through the header time of day (the header has no
    date; the midnight is resolved against the 2 Hz CSV). Per-encounter alignment (`aligned`, `Lstar`) is read from the earlier HR cross-correlation
    analysis (results/openoximetry.md: 164 of 307 aligned, 93 people). Per-window labels exist ONLY for aligned encounters; the other 143 keep their
    windows as unlabelled (pretraining / contrastive use).
  * HR label: median over the reference-device HR columns (30..220 bpm, 2 Hz), window mean over [t_end - 10 + Lstar, t_end + Lstar], Lstar = the
    encounter's HR lag (reference trails the PPG by 5 to 10 s for HR devices). Aligned encounters only.
  * SpO2 labels. `spo2_ref`: median over the contact-oximeter SpO2 columns (50..100.5). `sao2`: that display series mapped to ABG SaO2 with a per-encounter
    linear map fitted on the blood-gas draws of THAT encounter (SaO2 = a + b SpO2_device at the draw, b clipped to 0.6..1.6, offset-only when the draws
    do not span 2.5 points), i.e. interpolated between draws and bias-corrected to the draws (pulseoximeter.csv gives the device reading at each draw).
    `sao2_draw`: the ABG value itself for the windows whose interval contains the draw marker (`Sample` column of the 2 Hz CSV); this is the only label that is
    not a model of the reference. The label lag (the reference display trails tissue by about 10 to 30 s) is applied by `relabel(ws, lag)`;
    the stored labels use lag 0. `estimate_lag(train_subjects)` finds that lag from TRAINING subjects only.
  * skin tone: Monk (hand sites, median 1..10), Fitzpatrick, ITA (spectrophotometer palmar, ITA = atan2(L - 50, b)), per encounter.
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from .. import BASE_SEC, FS, HOP_S, WIN_S
from .. import features as F
from .. import windows as W
from . import VERITAS, cli

RAW = os.environ.get('OXY_OPENOX', f'{VERITAS}/multi/raw/openox')
ALIGN = os.environ.get('OXY_OPENOX_ALIGN', f'{VERITAS}/pivot/openox/results/align.csv')
MONK = {c: i + 1 for i, c in enumerate('ABCDEFGHIJ')}
HAND_SITES = ['monk_fingernail', 'monk_dorsal', 'monk_palmar']
LAG_GRID = np.arange(-10, 61, 1.0)


# --------------------------------------------------------------------------------------------- readers
def load_ref(path):
    """2 Hz reference CSV -> frame with t = seconds since the midnight of the first timestamp (0.5 s steps rebuilt when timestamps are whole seconds)."""
    d = pd.read_csv(path)
    ts = pd.to_datetime(d['Timestamp'])
    sec = (ts - ts.dt.normalize().iloc[0]).dt.total_seconds().to_numpy()
    whole = np.floor(sec)
    k = d.groupby(whole).cumcount().to_numpy(); n = d.groupby(whole)['Sample'].transform('size').to_numpy()
    frac = sec - whole
    d['t'] = whole + np.where(frac > 0, frac, k / n)
    return d


def ref_cols(d):
    import re
    out = {'spo2': [], 'hr': []}
    for c in d.columns:
        m = re.match(r'^dev(\d+)_(SpO2|HR|PR)$', c)
        if m:
            out['spo2' if m.group(2) == 'SpO2' else 'hr'].append(c)
    return out


def ref_median(d, cols, lo, hi):
    if not cols:
        return np.full(len(d), np.nan)
    a = np.column_stack([np.where((d[c].values >= lo) & (d[c].values <= hi), d[c].values, np.nan) for c in cols]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmedian(a, axis=1)


def per_second(vals, tt, n):
    """mean of vals in each integer second [0, n) of the axis tt (s)."""
    idx = np.floor(tt).astype(int); ok = np.isfinite(vals) & (idx >= 0) & (idx < n)
    s = np.bincount(idx[ok], weights=vals[ok], minlength=n); c = np.bincount(idx[ok], minlength=n)
    with np.errstate(all='ignore'):
        return np.where(c > 0, s / c, np.nan)


def read_ppg(base):
    """-> dict(red, ir, fs, n, nsig, tod) from a WFDB record `<base>` (no extension). None when a channel is missing."""
    import wfdb
    r = wfdb.rdrecord(base)
    names = [s.upper().strip() for s in r.sig_name]
    sig = r.p_signal if r.p_signal is not None else r.d_signal.astype(float)

    def pick(*c):
        for k in c:
            if k in names:
                return sig[:, names.index(k)].astype(np.float64)
        return None
    red, ir = pick('RED SIGNAL', 'RED'), pick('IR SIGNAL', 'IR')
    if red is None or ir is None:
        return None
    h = open(base + '.hea').readline().split()
    p = h[4].split(':') if len(h) > 4 else None
    tod = int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2]) if p else np.nan
    return dict(red=red, ir=ir, fs=float(r.fs), n=sig.shape[0], nsig=int(h[1]), tod=tod)


def to_30hz(x, fs):
    """Raw counts -> 30 Hz with an anti-alias polyphase FIR; non-positive / non-finite samples are repaired by linear interpolation first."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x) | (x <= 0)
    if bad.all():
        return None, 1.0
    if bad.any():
        i = np.arange(len(x)); x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    fr = Fraction(FS, 1) / Fraction(fs).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, padtype='line'), float(bad.mean())


def _skin(enc, sp):
    """Per-encounter skin fields from encounter.csv (Monk letters A..J) and spectrophotometer.csv (palmar ITA)."""
    def monk(row):
        v = [MONK[str(row[s]).strip().upper()] for s in HAND_SITES if isinstance(row[s], str) and str(row[s]).strip().upper() in MONK]
        return float(np.median(v)) if v else np.nan
    enc = enc.copy(); enc['monk'] = enc.apply(monk, axis=1)
    sp = sp[sp.group.astype(str).str.startswith('Palmar')].copy()
    sp['ita'] = np.degrees(np.arctan2(sp.lab_l - 50.0, sp.lab_b))
    ita = sp.groupby('encounter_id').ita.mean()
    enc['ita'] = enc.encounter_id.map(ita)

    def grp(x):
        if not np.isfinite(x):
            return 'unknown'
        r = int(np.floor(x + 0.5)); return 'light(1-3)' if r <= 3 else ('mid(4-6)' if r <= 6 else 'dark(7-10)')
    enc['skin_group'] = enc.monk.map(grp)
    return enc.set_index('encounter_id')


def _ppg_paths():
    out = {}
    for f in glob.glob(f'{RAW}/waveforms/*/*_ppg.hea'):
        out[os.path.basename(f)[:-8]] = f[:-4]
    return out


def _draws(d, tt0, bg_e, po_e):
    """Blood-gas draws of one encounter in PPG time. d: 2 Hz frame, tt0: abs time of the PPG start. -> DataFrame(sample, t0, t1, sao2, spo2_dev)."""
    m = d[d.Sample.notna()]
    if not len(m):
        return pd.DataFrame(columns=['sample', 't0', 't1', 'sao2', 'spo2_dev'])
    g = m.groupby('Sample').t.agg(['min', 'max']).reset_index().rename(columns={'Sample': 'sample', 'min': 't0', 'max': 't1'})
    g[['t0', 't1']] = g[['t0', 't1']] - tt0
    so = bg_e[bg_e.so2.between(40, 100.5)].groupby('sample').so2.mean()
    g['sao2'] = g['sample'].map(so)
    p = po_e[po_e.saturation.between(50, 100.5)].groupby('sample_number').saturation.mean()
    g['spo2_dev'] = g['sample'].map(p)
    return g


def _fit_map(dr):
    """Per-encounter map SaO2 = a + b * device SpO2 from the draws. -> (a, b, rmse, n)."""
    ok = dr.sao2.notna() & dr.spo2_dev.notna()
    s = dr.spo2_dev[ok].values; y = dr.sao2[ok].values
    if len(s) < 2:
        return np.nan, np.nan, np.nan, len(s)
    if len(s) >= 5 and s.std() >= 2.5:
        b, a = np.polyfit(s, y, 1); b = float(np.clip(b, 0.6, 1.6)); a = float(np.median(y - b * s))
    else:
        b, a = 1.0, float(np.median(y - s))
    rmse = float(np.sqrt(np.mean((y - (a + b * s)) ** 2)))
    return a, b, rmse, len(s)


def _win_mean(x, t_start, shift, win=WIN_S, min_valid=5):
    """Mean of the 1 Hz series x over seconds [t_start + shift, t_start + shift + win) with >= min_valid finite values (vectorised, integer shift)."""
    n = len(x)
    ok = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))]); cc = np.concatenate([[0], np.cumsum(ok)])
    a = np.clip(np.asarray(t_start, int) + int(round(shift)), 0, n); b = np.clip(a + win, 0, n)
    c = cc[b] - cc[a]
    with np.errstate(all='ignore'):
        m = (cs[b] - cs[a]) / c
    inside = (np.asarray(t_start, int) + int(round(shift)) >= 0) & (np.asarray(t_start, int) + int(round(shift)) + win <= n)
    return np.where((c >= min_valid) & inside, m, np.nan)


# --------------------------------------------------------------------------------------------- build
def build(out_dir=None, limit=None):
    from scipy.signal import resample_poly  # noqa: F401  (fail early)
    enc = pd.read_csv(f'{RAW}/encounter.csv'); pat = pd.read_csv(f'{RAW}/patient.csv')
    sp = pd.read_csv(f'{RAW}/spectrophotometer.csv'); skin = _skin(enc, sp)
    bg = pd.read_csv(f'{RAW}/bloodgas.csv'); bg = bg.loc[:, ~bg.columns.duplicated()]
    po = pd.read_csv(f'{RAW}/pulseoximeter.csv')
    po['sample_number'] = pd.to_numeric(po.sample_number, errors='coerce'); po['saturation'] = pd.to_numeric(po.saturation, errors='coerce')
    al = pd.read_csv(ALIGN).set_index('encounter_id')
    pid = enc.set_index('encounter_id').patient_id
    paths = _ppg_paths()
    encs = sorted(e for e in al.index if e in paths)
    if limit:
        encs = encs[:limit]
    w = W.ShardWriter('openox', out_dir)
    refrows, drawrows, encrows = [], [], []
    for k, e in enumerate(encs):
        cp = glob.glob(f'{RAW}/waveforms/*/{e}_2hz.csv')
        if not cp:
            print('  skip (no 2 Hz CSV):', e[:12]); continue
        pp = read_ppg(paths[e])
        if pp is None:
            print('  skip (missing red/IR channel):', e[:12]); continue
        d = load_ref(cp[0])
        t0r = float(d.t.iloc[0]); t1r = float(d.t.iloc[-1])
        kk = min((-1, 0, 1), key=lambda z: abs(pp['tod'] + 86400 * z - t0r)); pstart = pp['tod'] + 86400 * kk
        red, br = to_30hz(pp['red'], pp['fs']); ir, bi = to_30hz(pp['ir'], pp['fs'])
        if red is None or ir is None:
            print('  skip (no usable samples):', e[:12]); continue
        m = min(len(red), len(ir)); red, ir = red[:m], ir[:m]
        n_sec = m // FS
        lrgb = np.stack([np.log(np.clip(red, 1e-3, None) + 1.0), np.log(np.clip(ir, 1e-3, None) + 1.0), np.zeros(m)], 1)
        # reference series on the PPG axis (1 Hz)
        tr = d.t.values - pstart
        hr1 = per_second(ref_median(d, ref_cols(d)['hr'], 30, 220), tr, n_sec)
        sp1 = per_second(ref_median(d, ref_cols(d)['spo2'], 50, 100.5), tr, n_sec)
        dr = _draws(d, pstart, bg[bg.encounter_id == e], po[po.encounter_id == e])
        a, b, rmse, nfit = _fit_map(dr)
        sao1 = np.clip(a + b * sp1, 40, 100) if np.isfinite(a) else np.full(n_sec, np.nan)
        aligned = bool(al.loc[e, 'aligned']); lstar = float(al.loc[e, 'Lstar']) if aligned else np.nan
        sk = skin.loc[e] if e in skin.index else None
        starts = W.window_starts(m)
        rows = pd.DataFrame(dict(t_start=starts))
        if aligned:
            rows['hr_ref'] = _win_mean(hr1, starts, lstar); rows['has_hr'] = rows.hr_ref.notna()
            rows['spo2_ref'] = _win_mean(sp1, starts, 0); rows['has_spo2'] = rows.spo2_ref.notna()
            rows['sao2'] = _win_mean(sao1, starts, 0); rows['has_sao2'] = rows.sao2.notna() & np.isfinite(rmse)
            rows['low'] = np.where(rows.has_sao2, (rows.sao2 < 90).astype(float), np.nan); rows['has_low'] = rows.has_sao2
            # ABG draw windows: interval [t_start, t_start + 10] contains a draw marker
            sd = np.full(len(starts), np.nan); dd = np.full(len(starts), np.nan)
            for r in dr[dr.sao2.notna()].itertuples():
                hit = (starts <= r.t1) & (starts + WIN_S >= r.t0)
                sd[hit] = r.sao2; dd[hit] = (starts[hit] + WIN_S) - 0.5 * (r.t0 + r.t1)
            rows['sao2_draw'] = sd; rows['has_sao2_draw'] = np.isfinite(sd); rows['draw_dt'] = dd
        if sk is not None:
            rows['monk'] = sk.monk; rows['ita'] = sk.ita; rows['fitz'] = sk.fitzpatrick; rows['skin_group'] = sk.skin_group
        rows["eval"] = rows.t_start >= BASE_SEC
        rows["in_baseline"] = rows.t_start + WIN_S <= BASE_SEC; rows['base_key'] = f'ox_{e[:12]}'
        subj = f'ox_{pid[e][:12]}'
        w.add(subj, f'ox_{e[:12]}', lrgb, np.full(m, 1 / 30), rows, mask=(True, True, False),
              aligned=aligned, lstar=lstar, fit_a=a, fit_b=b, fit_rmse=rmse, n_draws=int(dr.sao2.notna().sum()), layout=pp['nsig'], bad_frac=max(br, bi))
        refrows.append(pd.DataFrame(dict(rec=f'ox_{e[:12]}', t=np.arange(n_sec), ref_hr=hr1.astype(np.float32), ref_spo2=sp1.astype(np.float32), sao2_est=sao1.astype(np.float32))))
        dr['rec'] = f'ox_{e[:12]}'; drawrows.append(dr)
        if k % 25 == 0:
            print(f'  openox {k}/{len(encs)}', flush=True)
    ws = w.finalize(info=dict(source='OpenOximetry v1.1.1 (PhysioNet credentialed)', align=ALIGN, baseline='first 60 s', note='contact red/IR PPG; red -> R, IR -> G'))
    pd.concat(refrows, ignore_index=True).to_parquet(os.path.join(ws.dir, 'ref.parquet'))
    pd.concat(drawrows, ignore_index=True).to_parquet(os.path.join(ws.dir, 'draws.parquet'))
    print('openox', W.counts(ws))
    return ws


# --------------------------------------------------------------------------------------------- labels at a lag
class RefSeries:
    """1 Hz reference series per recording (PPG time axis) from ref.parquet."""

    def __init__(self, ws):
        r = pd.read_parquet(os.path.join(ws.dir, 'ref.parquet'))
        self.by = {k: (g.ref_hr.values.astype(float), g.ref_spo2.values.astype(float), g.sao2_est.values.astype(float)) for k, g in r.groupby('rec', sort=False)}


def relabel(ws, lag_s, refs=None):
    """Window labels with the reference display lag applied: the label of a window ending at t is the reference mean over [t - 10 + lag, t + lag]
    (reference trails tissue by `lag` seconds). Returns a copy of ws.meta with spo2_ref, sao2, has_spo2, has_sao2, low, has_low replaced for aligned
    encounters (HR labels and ABG-draw labels are lag independent and untouched)."""
    refs = refs or RefSeries(ws)
    m = ws.meta.copy()
    for rec, idx in m.groupby('rec', sort=False).indices.items():
        if rec not in refs.by or not bool(m.aligned.values[idx[0]]):
            continue
        _, sp1, sao1 = refs.by[rec]
        ts = m.t_start.values[idx]
        m.iloc[idx, m.columns.get_loc('spo2_ref')] = _win_mean(sp1, ts, lag_s)
        m.iloc[idx, m.columns.get_loc('sao2')] = _win_mean(sao1, ts, lag_s)
    m['has_spo2'] = m.aligned & m.spo2_ref.notna(); m['has_sao2'] = m.aligned & m.sao2.notna() & np.isfinite(m.fit_rmse)
    m['has_low'] = m.has_sao2; m['low'] = np.where(m.has_sao2, (m.sao2 < 90).astype(float), np.nan)
    return m


def estimate_lag(train_subjects, ws=None, grid=LAG_GRID, min_windows=60, min_ref_sd=3.0, min_enc=5, method='ppg', n_boot=0, seed=0):
    """Reference-lag estimate learned ONLY from `train_subjects` (subject ids of the shard; never pass a test subject).

    method 'ppg' (default): per aligned encounter, correlate the window-mean per-beat ratio-of-ratios ln(AC_red/AC_IR) (abs_ror_RG, a physics feature of
      the PPG itself) with the reference SpO2 shifted by `lag` seconds, r_e(lag) = corr(ror_t, ref_spo2(t + lag)); the expected sign is negative
      (lower SpO2 means a higher ratio). The pooled score is the mean over encounters of -r_e(lag); lag* = argmax. Encounters need >= min_windows valid
      windows and a reference SpO2 SD >= min_ref_sd.
    method 'abg': the lag that minimises the mean absolute difference between the device display at (draw + lag) and the ABG SaO2 of the draw (bias-corrected
      display, so only the timing matters), pooled over the training encounters.
    Returns dict(lag, profile (lag -> score), n_enc, confident (peak score > 0.05 for 'ppg'), ci (bootstrap over encounters if n_boot > 0))."""
    ws = ws or W.load('openox')
    refs = RefSeries(ws)
    m = ws.meta
    train = set(train_subjects)
    sel = m[m.subject.isin(train) & m.aligned]
    per = []
    if method == 'ppg':
        for rec, g in sel.groupby('rec', sort=False):
            hr, sp1, _ = refs.by[rec]
            f = g.abs_ror_RG.values; ts = g.t_start.values
            ok0 = np.isfinite(f)
            if ok0.sum() < min_windows or np.nanstd(sp1) < min_ref_sd:
                continue
            rs = np.full(len(grid), np.nan)
            for i, L in enumerate(grid):
                y = _win_mean(sp1, ts, L)
                ok = ok0 & np.isfinite(y)
                if ok.sum() >= min_windows and f[ok].std() > 1e-9 and y[ok].std() > 1e-9:
                    rs[i] = -np.corrcoef(f[ok], y[ok])[0, 1]
            if np.isfinite(rs).sum() > len(grid) // 2:
                per.append(rs)
    else:
        dr = pd.read_parquet(os.path.join(ws.dir, 'draws.parquet'))
        recs = set(sel.rec)
        for rec, g in dr[dr.rec.isin(recs)].groupby('rec'):
            _, sp1, _ = refs.by[rec]
            g = g[g.sao2.notna()]
            if len(g) < 5 or np.nanstd(sp1) < min_ref_sd:
                continue
            rs = np.full(len(grid), np.nan)
            for i, L in enumerate(grid):
                t = np.round(0.5 * (g.t0.values + g.t1.values) + L).astype(int)
                ok = (t >= 0) & (t < len(sp1)); v = np.where(ok, sp1[np.clip(t, 0, len(sp1) - 1)], np.nan)
                good = np.isfinite(v)
                if good.sum() >= 4:
                    b = np.polyfit(v[good], g.sao2.values[good], 1) if np.std(v[good]) > 1 else (1.0, np.median(g.sao2.values[good] - v[good]))
                    rs[i] = -np.mean(np.abs(g.sao2.values[good] - (b[1] + b[0] * v[good])))        # higher is better
            if np.isfinite(rs).sum() > len(grid) // 2:
                per.append(rs)
    if len(per) < min_enc:
        return dict(lag=np.nan, profile=None, n_enc=len(per), confident=False, ci=None, method=method)
    P = np.vstack(per)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prof = np.nanmean(P, 0)
        best = float(grid[int(np.nanargmax(prof))])
        ci = None
        if n_boot:
            rng = np.random.default_rng(seed)
            bs = [grid[int(np.nanargmax(np.nanmean(P[rng.integers(0, len(P), len(P))], 0)))] for _ in range(n_boot)]
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        conf = bool(np.nanmax(prof) > 0.05) if method == 'ppg' else True
    return dict(lag=best, profile=dict(zip(grid.tolist(), prof.tolist())), n_enc=len(per), confident=conf, ci=ci, method=method)


def check():
    ws = W.load('openox'); m = ws.meta
    ok = m.subject.nunique() >= 130 and m.rec.nunique() >= 290 and m.has_sao2_draw.sum() > 1000 and m.has_hr.sum() > 10000
    e = ws.recs
    print(f'openox check: windows {len(m)} ({len(m) * HOP_S / 3600:.0f} h of 2 s hops), recordings {m.rec.nunique()}, people {m.subject.nunique()}, aligned recordings '
          f'{int(m.drop_duplicates("rec").aligned.sum())} ({m[m.aligned].subject.nunique()} people), HR labels {int(m.has_hr.sum())}, SpO2-ref labels {int(m.has_spo2.sum())}, '
          f'SaO2 (ABG-anchored) {int(m.has_sao2.sum())}, ABG-draw windows {int(m.has_sao2_draw.sum())}, skin tone known {int(m.monk.notna().groupby(m.subject).any().sum())} people, '
          f'with baseline {int(m.has_base.sum())}: {"OK" if ok else "FAIL"}')
    return bool(ok)


if __name__ == '__main__':
    cli(build, check, __doc__)
