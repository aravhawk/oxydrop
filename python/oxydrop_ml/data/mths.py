"""MTHS (Samavati et al. 2022; CC BY-NC-ND 4.0, NON-COMMERCIAL RESEARCH USE ONLY, never committed or redistributed, no adaptations).
62 people, iPhone 5s fingertip video (per-frame mean R,G,B at 30 fps, channel order as released), 1 Hz reference HR and SpO2 from a finger oximeter.
Recordings are 26 to ~90 s. The cohort is normoxic (about 1 % of reference seconds are below 90).

Windows: 10 s, 2 s hop. Labels: window mean of the 1 Hz reference with >= 5 valid seconds (the `scripts/mths_spo2.py` convention).
Baseline = windows fully inside the first 30 s (the 60 s minute does not exist in these recordings; 30 s is what scripts/mths_spo2.py uses);
`eval` = t_start >= 30 s. People whose recording is too short to have an eval window keep baseline-only windows (eval False).
No frame timestamps: dt is nominal. SpO2 labels exist for evaluation of P3 (absolute SpO2 MAE, normoxic) and HR labels for P2.
"""
import os

import numpy as np

from .. import WIN_S
from .. import features as F
from .. import windows as W
from . import OXY_DATA, cli

ROOT = os.environ.get('OXY_MTHS', f'{OXY_DATA}/mths/repo/MTHS/Data')
CAL = 30
EXPECT_SUBJECTS = 62


def build(root=ROOT, out_dir=None):
    import pandas as pd
    ids = sorted(int(f.split('_')[-1][:-4]) for f in os.listdir(root) if f.startswith('signal_'))
    w = W.ShardWriter('mths', out_dir)
    for i in ids:
        rgb = np.load(f'{root}/signal_{i}.npy').astype(float)
        lab = np.load(f'{root}/label_{i}.npy').astype(float)
        hr = np.where((lab[:, 0] >= 30) & (lab[:, 0] <= 220), lab[:, 0], np.nan)
        sp = np.where(lab[:, 1] > 0, lab[:, 1], np.nan)
        starts = W.window_starts(len(rgb))
        rows = []
        for s in starts:
            y = W.mean_label(sp, s); h = W.mean_label(hr, s)
            rows.append(dict(t_start=s, spo2_ref=y, has_spo2=bool(np.isfinite(y)), low=float(y < 90) if np.isfinite(y) else np.nan, has_low=bool(np.isfinite(y)),
                             hr_ref=h, has_hr=bool(np.isfinite(h)), **{'eval': s >= CAL, 'in_baseline': s + WIN_S <= CAL}, base_key=f'mths_{i}'))
        w.add(f'mths_{i}', f'mths_{i}', F.log_rgb(rgb), np.full(len(rgb), 1 / 30), pd.DataFrame(rows))
    ws = w.finalize(info=dict(source='MTHS (CC BY-NC-ND 4.0), non-commercial research only', baseline='first 30 s', eval='t_start >= 30 s'), baseline_min_windows=3)
    print('mths', W.counts(ws))
    return ws


def check():
    ws = W.load('mths'); m = ws.meta
    ok = m.subject.nunique() == EXPECT_SUBJECTS and m.has_hr.sum() > 1000 and m.has_spo2.sum() > 1000
    e = m[m['eval']]
    print(f'mths check: subjects {m.subject.nunique()} (expect {EXPECT_SUBJECTS}), windows {len(m)}, eval {len(e)} from {e.subject.nunique()} subjects, '
          f'SpO2 labels {int(m.has_spo2.sum())} (below 90: {int((m.spo2_ref < 90).sum())}), HR labels {int(m.has_hr.sum())}, '
          f'with baseline {int(m.has_base.sum())}: {"OK" if ok else "FAIL"}')
    return bool(ok)


if __name__ == '__main__':
    cli(build, check, __doc__)
