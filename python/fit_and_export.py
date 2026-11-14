"""Fit the production weights (all 6 Hoffman people) and the held-out replay weights used by the in-app recorded demo.
Usage: python fit_and_export.py <hoffman_repo_root> <out_dir>
Outputs: weights.json (production), replay/<rec>.bin|.json|_weights.json (recording + 1 Hz reference SpO2 + weights fitted WITHOUT that person)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oxydrop import chain, hoffman

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out + '/replay', exist_ok=True)
recs = hoffman.load(root)
W = chain.fit(recs); json.dump(W, open(out + '/weights.json', 'w'), indent=1)
old = json.load(open(os.path.dirname(os.path.abspath(__file__)) + '/tiny_chain_weights.json'))
print('production coef max diff vs shipped weights:', np.abs(np.array(W['fusion']['coef']) - np.array(old['fusion']['coef'])).max())
for rid in ('100001L', '100004R'):
    r = next(x for x in recs if x['rec'] == rid)
    r['rgb'].astype('<f4').tofile(f"{out}/replay/{rid}.bin")
    Wh = chain.fit(recs, exclude_sid=r['sid'])
    o = chain.infer(r['rgb'], Wh)
    y = np.array([r['y'][t] < 90 for t, *_ in o]); pred = np.array([d[2] for d in o])
    ans = pred != 'unsure'
    print(rid, 'held-out acc all', np.mean((np.array([d[1] for d in o]) > .5) == y).round(3), 'answered', ans.mean().round(3),
          np.mean(((pred == 'HYPOXIC') == y)[ans]).round(3))
    json.dump(dict(rec=rid, person=r['sid'] - 100000, fps=30, n=len(r['rgb']),
                   spo2=[None if np.isnan(v) else round(float(v), 1) for v in r['gt']], weights=Wh),
              open(f"{out}/replay/{rid}.json", 'w'))
