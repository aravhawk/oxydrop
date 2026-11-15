"""BUT PPG v2.0 (Brno Univ. of Technology smartphone PPG database; CC BY 4.0, PhysioNet): 10 s smartphone clips (Xiaomi Mi9 / Huawei P20 Pro,
30 fps, finger-on-camera and ear), 3,888 clips from 50 subjects, ECG-derived reference HR per clip, expert quality label (1 good, 0 bad), and for
38 subjects a spot SpO2 (one value per subject, measured near the recordings, NOT time aligned: it is a person-level label).

One clip = one recording = one window (t_start 0; the 5 s warm-up before it is the odd reflection of the clip start). Recordings are NOT
contiguous video, so the baseline of a person is the first 6 clips in id order (about 60 s of video, scripts/ vlib.py convention) and `eval`
is every other clip of that person. The prereg cohort for P2 is the 38 subjects with SpO2 (`has_spo2`). The release has 3,888 clips from 50 subjects;
the 48 clips of the other 12 subjects have a header with 300 signals instead of 3 (a different layout) and are skipped, so the shard has exactly the
38 prereg people and 3,840 clips (`prereg_cohort` is True for all of them).
Frame timestamps are NOT in the BUT PPG release (the .hea gives only fs = 30), so dt is nominal 1/30.
Skin tone: the dataset publishes no per-person tone (Fitzpatrick II-IV overall); ear_finger and motion codes are kept as columns.
"""
import os

import numpy as np
import pandas as pd

from .. import features as F
from .. import windows as W
from . import OXY_DATA, cli

ROOT = os.environ.get('OXY_BUT', f'{OXY_DATA}/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0')
N_BASE_CLIPS = 6
EXPECT_PREREG_SUBJECTS = 38


def _read_clip(root, rid):
    hdr = open(f'{root}/{rid}/{rid}_PPG.hea').read().split('\n')
    if int(hdr[0].split()[1]) != 3:
        return None
    sig = []
    for ln in hdr[1:4]:
        tok = ln.split()[2]
        sig.append((float(tok.split('(')[0]), float(tok.split('(')[1].split(')')[0])))
    raw = np.fromfile(f'{root}/{rid}/{rid}_PPG.dat', dtype='<i2')
    if len(raw) < 900:
        return None
    x = raw[:900].reshape(-1, 3).astype(float)
    return np.clip(np.stack([(x[:, c] - sig[c][1]) / sig[c][0] for c in range(3)], 1), 1e-3, None)


def build(root=ROOT, out_dir=None):
    q = pd.read_csv(f'{root}/quality-hr-ann.csv', encoding='utf-8-sig')
    info = pd.read_csv(f'{root}/subject-info.csv', encoding='utf-8-sig')
    info.columns = ['ID', 'Gender', 'Age', 'Height', 'Weight', 'EarFinger', 'Motion', 'BP', 'Glyc', 'SpO2']
    q = q.merge(info, on='ID').sort_values('ID').reset_index(drop=True)
    q['subj'] = q.ID // 1000
    spo2_subj = set(q.loc[q.SpO2.notna() & (q.SpO2 > 0), 'subj'])
    w = W.ShardWriter('butppg', out_dir)
    seen = {}
    for r in q.itertuples():
        rgb = _read_clip(root, int(r.ID))
        if rgb is None:
            continue
        k = seen.get(r.subj, 0); seen[r.subj] = k + 1
        has_sp = bool(pd.notna(r.SpO2) and r.SpO2 > 0)
        row = dict(t_start=0, hr_ref=float(r.HR), has_hr=bool(r.HR > 0), q_expert=float(r.Quality), has_q=True,
                   spo2_ref=float(r.SpO2) if has_sp else np.nan, has_spo2=has_sp, low=float(r.SpO2 < 90) if has_sp else np.nan, has_low=has_sp,
                   **{'eval': k >= N_BASE_CLIPS, 'in_baseline': k < N_BASE_CLIPS}, base_key=f'but_{r.subj}')
        w.add(f'but_{r.subj}', f'but_{int(r.ID)}', F.log_rgb(rgb), np.full(len(rgb), 1 / 30), pd.DataFrame([row]),
              clip=k, ear_finger=int(r.EarFinger), motion=str(r.Motion), prereg_cohort=bool(r.subj in spo2_subj))
    ws = w.finalize(info=dict(source='BUT PPG v2.0 (CC BY 4.0)', baseline=f'first {N_BASE_CLIPS} clips of each subject', eval='other clips', dt='nominal (no timestamps in release)'))
    print('butppg', W.counts(ws))
    return ws


def check():
    ws = W.load('butppg'); m = ws.meta
    n_pre = m[m.prereg_cohort].subject.nunique()
    ok = n_pre == EXPECT_PREREG_SUBJECTS and len(m) > 3500 and m.has_hr.all()
    print(f'butppg check: windows(clips) {len(m)}, subjects {m.subject.nunique()}, prereg cohort subjects {n_pre} (expect {EXPECT_PREREG_SUBJECTS}), '
          f'eval {int(m["eval"].sum())}, SpO2 labels {int(m.has_spo2.sum())}, expert quality good {int((m.q_expert == 1).sum())}, '
          f'with baseline {int(m.has_base.sum())}: {"OK" if ok else "FAIL"}')
    return bool(ok)


if __name__ == '__main__':
    cli(build, check, __doc__)
