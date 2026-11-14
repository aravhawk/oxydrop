"""One-off export (needs the original pivot/ tree). Copies the per-window out-of-fold score arrays and the minimal label table
into results/scores/ so that make_eval.py runs with numpy/scipy/matplotlib/pandas only."""
import sys, os, shutil, json, numpy as np, pandas as pd
VER = '/home/aravhawk/.t3/scratch/veritas'
sys.path.insert(0, VER + '/pivot'); import kit
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'scores'); os.makedirs(OUT, exist_ok=True)
kit.df[['sid', 't', 'y', 'rec']].to_csv(OUT + '/windows.csv', index=False)
np.save(OUT + '/tiny.npy', np.load(VER + '/pivot/results/opus_TINY_CHAIN.npy'))
np.save(OUT + '/dhr_only.npy', np.load(VER + '/pivot/results/opus_F_hr.npy'))  # nested-LOPO, post-leak-fix DeltaHR-only (0.751 acc)
np.save(OUT + '/time_only.npy', np.asarray(kit.TIME_ONLY))
shutil.copy(VER + '/pivot/results/opus_t5_exercise_fp.csv', OUT + '/exercise_fp.csv')
rows = []
for h in ('Left', 'Right'):
    d = VER + '/_feas_photon/repo/data/info/' + h
    for f in sorted(os.listdir(d)):
        s = json.loads(open(d + '/' + f).read().split('\n')[0])['subject_info']
        rows.append(dict(sid=int(f.split('-')[0]), hand=h, ethnicity=s['ethnicity'], age=s['age'], sex=s['sex'], skintone_hex=s['skintone']))
pd.DataFrame(rows).to_csv(OUT + '/demographics.csv', index=False)
print('exported', os.listdir(OUT))
