"""Fit production weights (v2 chain, all 6 Hoffman people) and held-out replay weights for the in-app recorded demo.
Usage: python fit_and_export.py <hoffman_repo_root> <out_dir>
Outputs: weights.json (v2 production, valid windows only), replay/<rec>.bin|.json (recording + 1 Hz reference SpO2 + weights fitted WITHOUT that person)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oxydrop import chain_v2, hoffman

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out + '/replay', exist_ok=True)
recs = hoffman.load(root)
W = chain_v2.fit(recs); json.dump(W, open(out + '/weights.json', 'w'), indent=1)
print('production weights written; fusion coef[0..2] =', np.round(W['fusion']['coef'][:3], 4))
for rid in ('100001L', '100004R'):
    r = next(x for x in recs if x['rec'] == rid)
    r['rgb'].astype('<f4').tofile(f"{out}/replay/{rid}.bin")
    Wh = chain_v2.fit(recs, exclude_sid=r['sid'])
    o = chain_v2.infer(r['rgb'], Wh)
    y = np.array([r['y'][t] < 90 for t, *_ in o]); lab = np.array([d[2] for d in o]); ans = ~np.isin(lab, ['unsure', 'nolock'])
    print(rid, 'held-out answered', ans.mean().round(3), 'acc on answered', np.mean(((lab == 'HYPOXIC') == y)[ans]).round(3))
    json.dump(dict(rec=rid, person=r['sid'] - 100000, fps=30, n=len(r['rgb']),
                   spo2=[None if np.isnan(v) else round(float(v), 1) for v in r['gt']], weights=Wh),
              open(f"{out}/replay/{rid}.json", 'w'))
