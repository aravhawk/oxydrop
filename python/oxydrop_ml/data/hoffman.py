"""UW Hoffman 2022 (MIT): 6 people x 2 hands, Nexus 6P, 30 fps, reference SpO2 from several oximeters at 1 Hz, ~1100 s each with a staircase hypoxia protocol.

Labels follow python/oxydrop/hoffman.py exactly: window-mean of the mean of the working reference oximeters ('SpO2 1', '2', '4', '5', zeros
dropped) over a 10 s window with >= 5 valid seconds; hypoxic = window mean < 90. 10 s windows, 2 s hop, windows with t_start >= 60 s are
evaluated (5,644 windows over the 12 recordings). The first minute of each recording is the label-free baseline (windows fully inside it).
Both hands of one person are the same `subject` (group), so they are always held out together.
HR reference: median over 'Pulse 1,2,4,5' (zeros dropped), window mean.  No per-frame timestamps exist in the release: dt is nominal.
"""
import os
import warnings

import numpy as np
import pandas as pd

from .. import DATA_DIR, WIN_S
from .. import features as F
from .. import windows as W
from . import OXY_DATA, VERITAS, cli

ROOT = os.environ.get('OXY_HOF', f'{OXY_DATA}/hoffman_uw/repo')
EXPECT_EVAL = 5644


def build(root=ROOT, out_dir=None):
    from oxydrop import hoffman
    recs = hoffman.load(root)
    w = W.ShardWriter('hoffman', out_dir)
    for r in recs:
        sid, rec = r['sid'], r['rec']
        g = pd.read_csv(f'{root}/data/gt/{sid}.csv', encoding='utf-8-sig')
        cols = [c for c in ['Pulse 1', 'Pulse 2', 'Pulse 4', 'Pulse 5'] if c in g]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hr = np.nanmedian(g[cols].replace(0, np.nan).values.astype(float), axis=1)
        lrgb = F.log_rgb(r['rgb'])
        n_sec = min(len(r['gt']), len(lrgb) // 30)
        starts = [s for s in W.window_starts(n_sec * 30) if s in r['y']]
        rows = []
        for s in starts:
            y = r['y'][s]
            h = W.mean_label(hr, s)
            rows.append(dict(t_start=s, spo2_ref=y, has_spo2=True, low=float(y < 90), has_low=True, hr_ref=h, has_hr=bool(np.isfinite(h)),
                             **{'eval': s >= 60, 'in_baseline': s + WIN_S <= 60}, base_key=f'hof_{rec}'))
        w.add(f'hof_{sid}', f'hof_{rec}', lrgb, np.full(len(lrgb), 1 / 30), pd.DataFrame(rows), hand=rec[-1])
    ws = w.finalize(info=dict(source='UW Hoffman 2022 (MIT), https://github.com/ubicomplab/oximetry-phone-cam-data', baseline='first 60 s of each recording', eval='t_start >= 60 s'))
    print('hoffman', W.counts(ws))
    return ws


def check():
    ws = W.load('hoffman'); m = ws.meta
    n_eval = int(m['eval'].sum())
    ok = (n_eval == EXPECT_EVAL) and m.subject.nunique() == 6 and m.rec.nunique() == 12
    # the labels must equal the committed LOPO window table (results/loPO_v1_windows.csv)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    p = os.path.join(here, 'results', 'loPO_v1_windows.csv')
    if os.path.exists(p):
        ref = pd.read_csv(p)
        e = m[m['eval']].copy(); e['rec'] = e.rec.str.replace('hof_', '', regex=False)
        j = e.merge(ref, left_on=['rec', 't_start'], right_on=['rec', 't'])
        ok &= len(j) == len(ref) and ((j.spo2_ref < 90).astype(int) == j.y).all()
    print(f'hoffman check: eval windows {n_eval} (expect {EXPECT_EVAL}), subjects {m.subject.nunique()}, recordings {m.rec.nunique()}, '
          f'baseline ok {int(m[m["eval"]].has_base.sum())}/{n_eval}: {"OK" if ok else "FAIL"}')
    return bool(ok)


if __name__ == '__main__':
    cli(build, check, __doc__)
