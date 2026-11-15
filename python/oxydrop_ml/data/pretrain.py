"""Unlabeled pretraining corpus for self-supervised learning (no labels are read, only the PPG waveforms).

Sources (all disjoint from the Hoffman, BUT and MTHS people used for P1/P2/P3, so SSL never sees an outer test subject of those tasks):
  openox    OpenOximetry contact red / IR PPG (from the built `openox` shard; 137 people, credentialed: derived arrays stay in the cache). The OpenOx
            subjects ARE test subjects for the OpenOx SaO2 / HR tasks, so every window carries `subject`; use `splits.pretrain_allowed` to drop the
            held-out people when pretraining for an OpenOx evaluation.
  ptt_ppg   PhysioNet Pulse Transit Time PPG (CC BY, 22 people, sit / walk / run): MAX30101 red (pleth_1) and infrared (pleth_2), distal finger, 500 Hz.
  moshekam  PPG exercise dataset (14 people): red_chA and infrared_chA, 256 Hz.
  apnoea    PhysioNet apnoea neck pulse-oximeter photodiodes PD1 / PD2 (250 Hz), sensor voltages.
  bidmc     BIDMC ICU PPG (53 people, PLETH 125 Hz), amplitude-normalised (no usable DC, see below).
  healthring: NOT used (5.6 GB of undocumented-scale ring pickles; not needed to reach the target size).

Every source -> 30 Hz with an anti-alias polyphase FIR (scipy.signal.resample_poly) -> log(counts + 1) -> the same causal 0.7-4 Hz band-pass as the
model input (features.causal_ac, run over the whole recording, first 5 s skipped for the filter to settle). The window is 10 s = 300 samples,
stored as float16 ac * AC_SCALE for two slots: slot 0 = the red-like wavelength, slot 1 = the second wavelength (IR, or PD2). `valid` flags a slot that exists
and is not flat. BIDMC has no recoverable DC, so its ac is divided by the causal running std (10 s) and scaled to 1 % modulation (`mode` = 'norm').

Outputs under <CACHE>/data/pretrain/:  ac.f16 (float16 [N, 300, 2]), meta.parquet (src, subject, rec, t_start, valid0, valid1, mode), info.json.
`to_input(ac, valid)` converts rows to the CNN input tensor [N, 10, 300] (physics and DC channels are zero: SSL learns the waveform encoder).
Counts and sources are recorded in DATA_MANIFEST.md (`python -m oxydrop_ml.data.pretrain --manifest`).
"""
import glob
import json
import os
import re
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd

from .. import CACHE, DATA_DIR, FS, WIN
from .. import features as F
from .. import windows as W
from . import OXY_DATA, cli

HOPS = {'openox': 3, 'ptt_ppg': 1, 'moshekam': 1, 'apnoea': 1, 'bidmc': 1}     # window hop (s) per source
SETTLE = 5                                                                    # s skipped at the start of every record
MIN_STD = 1e-6                                                                # a slot is flat below this (log units)


def _to30(x, fs):
    from scipy.signal import resample_poly
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x) | (x <= 0)
    if bad.all():
        return None
    if bad.any():
        i = np.arange(len(x)); x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    fr = Fraction(FS, 1) / Fraction(fs).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, padtype='line')


def _windows_from(slots, hop, mode='log'):
    """slots: list of two 1-D arrays at 30 Hz (or None) -> (ac float16 [k, 300, 2], valid [k, 2], t_start [k])."""
    n = min(len(s) for s in slots if s is not None)
    cols = []
    for s in slots:
        cols.append(np.log(np.clip(s[:n], 1e-3, None) + 1.0) if s is not None else np.zeros(n))
    lr = np.stack(cols + [np.zeros(n)], 1)
    ac = F.causal_ac(lr)[:, :2]
    if mode == 'norm':
        sd = np.sqrt(F.ema(ac ** 2, 10.0, axis=0)) + 1e-9
        ac = ac / sd * 0.01
    present = np.array([s is not None for s in slots])
    starts = np.arange(SETTLE, n // FS - 10 + 1, hop)
    if not len(starts):
        return None
    idx = starts[:, None] * FS + np.arange(WIN)[None, :]
    w = ac[idx]                                                              # [k, 300, 2]
    sd = w.std(1)
    valid = (sd > MIN_STD) & present[None, :] & np.isfinite(sd)
    keep = valid.any(1) & np.isfinite(w).all((1, 2))
    return (w[keep] * F.AC_SCALE).astype(np.float16), valid[keep], starts[keep]


def _sources():
    """Generators of (src, subject, rec, slots at 30 Hz, mode)."""
    def ptt():
        for f in sorted(glob.glob(f'{OXY_DATA}/physionet_ptt_ppg/csv/s[0-9]*_*.csv')):
            b = os.path.basename(f)[:-4]; subj = 'ptt_' + b.split('_')[0]
            d = pd.read_csv(f, usecols=['pleth_1', 'pleth_2'], dtype=np.float32)
            a, b2 = _to30(d.pleth_1.values, 500), _to30(d.pleth_2.values, 500)
            if a is not None and b2 is not None:
                yield 'ptt_ppg', subj, 'ptt_' + b, [a, b2], 'log'

    def mosh():
        for f in sorted(glob.glob(f'{OXY_DATA}/moshekam_ppg_exercise/unz/Subject*.txt')):
            b = os.path.basename(f)[:-4]
            d = pd.read_csv(f, usecols=['red_chA', 'infrared_chA'], dtype=np.float32)
            a, b2 = _to30(d.red_chA.values, 256), _to30(d.infrared_chA.values, 256)
            if a is not None and b2 is not None:
                yield 'moshekam', 'mk_' + b, 'mk_' + b, [a, b2], 'log'

    def apn():
        for f in sorted(glob.glob(f'{OXY_DATA}/physionet_apnoea/Neck_Pulse_Oximeter_Data/Subject*/*_pulse.csv')):
            b = os.path.basename(f)[:-10]; subj = 'ap_' + b.split('_')[0]
            d = pd.read_csv(f, usecols=['PD1', 'PD2'], dtype=np.float32)
            a, b2 = _to30(d.PD1.values + 1.0, 250), _to30(d.PD2.values + 1.0, 250)
            if a is not None and b2 is not None:
                yield 'apnoea', subj, 'ap_' + b, [a, b2], 'log'

    def bidmc():
        import wfdb
        for f in sorted(glob.glob(f'{OXY_DATA}/bidmc/bidmc[0-9][0-9].hea')):
            base = f[:-4]
            r = wfdb.rdrecord(base)
            names = [s.strip().upper().rstrip(',') for s in r.sig_name]
            if 'PLETH' not in names:
                continue
            x = r.p_signal[:, names.index('PLETH')].astype(float)
            x = x - np.nanmin(x) + 1.0
            a = _to30(x, r.fs)
            if a is not None:
                yield 'bidmc', 'bm_' + os.path.basename(base), 'bm_' + os.path.basename(base), [a, None], 'norm'
    return dict(ptt_ppg=ptt, moshekam=mosh, apnoea=apn, bidmc=bidmc)


def build(out_dir=None, sources=None):
    out = out_dir or os.path.join(DATA_DIR, 'pretrain')
    os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, 'ac.f16')
    if os.path.exists(fpath):
        os.remove(fpath)
    f = open(fpath, 'ab'); rows = []; n = 0
    want = sources or ['openox', 'ptt_ppg', 'moshekam', 'apnoea', 'bidmc']

    def emit(src, subj, rec, res, mode):
        nonlocal n
        if res is None:
            return
        ac, valid, ts = res
        f.write(np.ascontiguousarray(ac).tobytes()); n += len(ac)
        rows.append(pd.DataFrame(dict(src=src, subject=subj, rec=rec, t_start=ts, valid0=valid[:, 0], valid1=valid[:, 1], mode=mode)))

    if 'openox' in want:
        ws = W.load('openox')
        for r in ws.recs.itertuples():
            lr = np.asarray(ws.series[r.offset:r.offset + r.length], np.float64)
            if r.length < (SETTLE + 10) * FS:
                continue
            ac = F.causal_ac(lr)[:, :2]
            starts = np.arange(SETTLE, r.length // FS - 10 + 1, HOPS['openox'])
            w = ac[starts[:, None] * FS + np.arange(WIN)[None, :]]
            sd = w.std(1); valid = (sd > MIN_STD) & np.isfinite(sd)
            keep = valid.any(1) & np.isfinite(w).all((1, 2))
            emit('openox', r.subject, r.rec, ((w[keep] * F.AC_SCALE).astype(np.float16), valid[keep], starts[keep]), 'log')
        print('  openox', n, flush=True)
    gens = _sources()
    for s in ('ptt_ppg', 'moshekam', 'apnoea', 'bidmc'):
        if s not in want:
            continue
        n0 = n
        for src, subj, rec, slots, mode in gens[s]():
            emit(src, subj, rec, _windows_from(slots, HOPS[src], mode), mode)
        print(f'  {s} {n - n0}', flush=True)
    f.close()
    meta = pd.concat(rows, ignore_index=True)
    meta.to_parquet(os.path.join(out, 'meta.parquet'))
    info = dict(n_windows=int(n), win=WIN, scale=F.AC_SCALE, hops=HOPS, sources={s: dict(windows=int((meta.src == s).sum()), subjects=int(meta[meta.src == s].subject.nunique())) for s in meta.src.unique()})
    json.dump(info, open(os.path.join(out, 'info.json'), 'w'), indent=1)
    print('pretrain', info)
    write_manifest()
    return info


def load(out_dir=None):
    out = out_dir or os.path.join(DATA_DIR, 'pretrain')
    info = json.load(open(os.path.join(out, 'info.json')))
    ac = np.memmap(os.path.join(out, 'ac.f16'), dtype=np.float16, mode='r', shape=(info['n_windows'], WIN, 2))
    return ac, pd.read_parquet(os.path.join(out, 'meta.parquet')), info


def to_input(ac, valid=None):
    """Pretraining rows -> CNN input [N, 10, 300] float32: slot 0 -> ac_R, slot 1 -> ac_G (a missing slot is zero), ac_B / dt / physics / DC channels zero."""
    ac = np.asarray(ac, np.float32)
    x = np.zeros((len(ac), F.C_IN, WIN), np.float32)
    x[:, 0] = ac[:, :, 0]; x[:, 1] = ac[:, :, 1]
    if valid is not None:
        v = np.asarray(valid, bool)
        x[:, 0] *= v[:, 0, None]; x[:, 1] *= v[:, 1, None]
    return x


def write_manifest(path=None):
    """Write <CACHE>/DATA_MANIFEST.md with the window counts of every shard and the pretraining corpus."""
    path = path or os.path.join(CACHE, 'DATA_MANIFEST.md')
    L = ['# OxyDrop ML data manifest', '', 'Generated by `python -m oxydrop_ml.data.pretrain --manifest`. Outside git. Contains only counts, no row-level data.', '',
         '| dataset | windows | eval windows | subjects | recordings | HR labels | SpO2-ref labels | SaO2 (ABG-anchored) | ABG-draw windows | expert quality | with baseline |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for ds in ('hoffman', 'mths', 'butppg', 'openox'):
        if os.path.exists(os.path.join(DATA_DIR, ds, 'meta.parquet')):
            c = W.counts(W.load(ds))
            L.append(f"| {ds} | {c['windows']} | {c['eval_windows']} | {c['subjects']} | {c['recordings']} | {c['hr']} | {c['spo2']} | {c['sao2']} | {c['sao2_draw']} | {c['quality']} | {c['with_baseline']} |")
    pi = os.path.join(DATA_DIR, 'pretrain', 'info.json')
    if os.path.exists(pi):
        info = json.load(open(pi))
        L += ['', f"## Pretraining corpus (unlabeled, float16 [N, 300, 2], {info['n_windows']} windows)", '', '| source | windows | subjects | hop (s) |', '|---|---|---|---|']
        for s, v in info['sources'].items():
            L.append(f"| {s} | {v['windows']} | {v['subjects']} | {info['hops'][s]} |")
        L += ['', 'All sources are disjoint from the Hoffman, BUT PPG and MTHS people. OpenOx windows keep their `subject` so held-out OpenOx people can be excluded.']
    open(path, 'w').write('\n'.join(L) + '\n')
    return path


def check():
    ac, meta, info = load()
    ok = info['n_windows'] >= 300000 and len(meta) == info['n_windows'] and np.isfinite(np.asarray(ac[:1000], np.float32)).all()
    forbidden = meta.subject.str.startswith(('hof_', 'mths_', 'but_')).any()
    print(f"pretrain check: {info['n_windows']} windows, sources {({k: v['windows'] for k, v in info['sources'].items()})}, "
          f"no Hoffman/MTHS/BUT subjects: {not forbidden}: {'OK' if ok and not forbidden else 'FAIL'}")
    return bool(ok and not forbidden)


if __name__ == '__main__':
    import sys
    if '--manifest' in sys.argv:
        print(write_manifest())
    else:
        cli(build, check, __doc__)
