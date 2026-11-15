"""Leave-one-person-out evaluation of TINY v1 vs v2 on Hoffman with a simple fixed-hyperparameter protocol (C=0.01 / 0.1, as shipped).
Reports every window with t>=60 s; v2 windows whose pulse is not locked count as abstentions (and are also scored against the label as 'normal' in the strict row).
Usage: python scripts/eval_v2.py <hoffman_root> [out_dir]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from oxydrop import hoffman, chain, chain_v2, evalkit

root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'results'
recs = hoffman.load(root)
rows = []  # per-window table: rec, sid, t, y, p_v1, p_v2, valid
for r in recs:
    t1, X1 = chain.deltas(r['rgb']); t2, X2, valid, aux = chain_v2.deltas(r['rgb'])
    r['_d1'] = (t1, X1); r['_d2'] = (t2, X2, valid, aux)
PEOPLE = sorted({r['sid'] for r in recs}); res = {1: {}, 2: {}}
for p in PEOPLE:
    W1 = chain.fit([dict(r, y=r['y']) for r in recs], exclude_sid=p) if False else None
for p in PEOPLE:
    # v1 fit with cached deltas
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    def fit_cached(ver):
        Xs, ys = [], []
        for r in recs:
            if r['sid'] == p: continue
            if ver == 1: t, X = r['_d1']; v = np.ones(len(t), bool)
            else: t, X, v, _ = r['_d2']
            keep = np.isin(t, list(r['y'].keys())) & (t >= 60) & v
            Xs.append(X[keep]); ys.append(np.array([r['y'][ti] < 90 for ti in t[keep]], int))
        X = np.vstack(Xs); y = np.concatenate(ys); W = {}
        for name, cols, C in (('hr', [0], 0.01), ('fusion', list(range(19)), 0.1)):
            sc = StandardScaler().fit(X[:, cols]); m = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(sc.transform(X[:, cols]), y)
            W[name] = dict(mu=sc.mean_.tolist(), sd=sc.scale_.tolist(), coef=m.coef_[0].tolist(), b=float(m.intercept_[0]))
        return W
    for ver in (1, 2):
        W = fit_cached(ver)
        for r in recs:
            if r['sid'] != p: continue
            if ver == 1: t, X = r['_d1']; v = np.ones(len(t), bool)
            else: t, X, v, _ = r['_d2']
            pr = 0.5 * (chain._lr(W['hr'], X[:, :1]) + chain._lr(W['fusion'], X))
            for ti, pi, vi in zip(t, pr, v):
                if ti >= 60 and ti in r['y']:
                    res[ver].setdefault('rows', []).append((r['rec'], r['sid'], ti, int(r['y'][ti] < 90), pi, vi))
import pandas as pd
D1 = pd.DataFrame(res[1]['rows'], columns=['rec', 'sid', 't', 'y', 'p', 'valid']); D2 = pd.DataFrame(res[2]['rows'], columns=['rec', 'sid', 't', 'y', 'p', 'valid'])
os.makedirs(out, exist_ok=True); D1.to_csv(out + '/loPO_v1_windows.csv', index=False); D2.to_csv(out + '/loPO_v2_windows.csv', index=False)
def row(name, D, band=None, gate=False):
    y = D.y.values; s = D.p.values; pred = (s > .5).astype(int); a = np.ones(len(D), bool)
    if gate: a &= D.valid.values
    if band: a &= np.abs(s - .5) >= band
    r = evalkit.report(D.sid.values, D.t.values, y, s, pred, None if a.all() else a); r['model'] = name; return r
table = [row('v1 TINY (peak-interval HR), every window', D1), row('v2 TINY (robust HR + gate), every window forced to answer (no-lock windows reuse the last valid features, forward-filled)', D2),
         row('v2, no-lock windows abstain', D2, gate=True), row('v1, abstain band 0.10', D1, band=.10), row('v2, no-lock + abstain band 0.10', D2, band=.10, gate=True)]
print('v2 valid fraction on Hoffman:', round(float(D2.valid.mean()), 4))
T = pd.DataFrame(table)[['model', 'coverage', 'acc', 'bacc', 'sens', 'spec', 'auc', 'tm_acc', 'worst_person', 'acc_ci']]
T.to_csv(out + '/eval_v1_vs_v2.csv', index=False); print(T.to_string())
