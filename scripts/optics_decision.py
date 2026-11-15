"""Decision rule for shipping the optical ratio-of-ratios as a model input (fixed in advance, see docs/SPO2_OPTICS.md):

  Ship it ONLY IF a nested person-level LOPO model on Hoffman that uses the optical features beats the SHIPPED TINY v2 on accuracy AND on
  time-matched accuracy, with the person-bootstrap 95 % CI of each difference excluding zero (lower bound > 0).

Inputs: the nested-LOPO optics scores written by scripts/optics_experiment.py (scores.pkl in OXY_TMP) and results/loPO_v2_windows.csv
(shipped TINY v2 LOPO probabilities + lock flag). Compared on the SAME windows: evaluated (t >= 60 s), optics gate passed, v2 pulse locked.
Run: /home/aravhawk/.t3/scratch/veritas/oxy_push/.venv/bin/python -s scripts/optics_decision.py
Writes results/optics_experiment/decision_rule.csv
"""
import os, sys, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, '/home/aravhawk/.t3/scratch/veritas/pivot')
import kit
TMP = os.environ.get('OXY_TMP', '/home/aravhawk/.t3/scratch/optics_tmp')
S = pickle.load(open(TMP + '/scores.pkl', 'rb')); SC, gate = S['SCORES'], S['gate']
V2 = pd.read_csv(SUB + '/results/loPO_v2_windows.csv')
F = kit.df[['rec', 't']].merge(V2[['rec', 't', 'p', 'valid']], on=['rec', 't'], how='left')
assert (F.rec.values == kit.df.rec.values).all()
p_v2 = F.p.values; valid2 = F.valid.fillna(False).astype(bool).values
y = kit.y_all; sid = kit.sid; t = kit.t; PEOPLE = kit.PEOPLE
BASE = kit.ev & gate & valid2 & np.isfinite(p_v2)


def basic(yy, pp):
    return (pp == yy).mean()


def tm_exact(yy, pp, tt):
    b = (tt // 100).astype(int); num = den = 0.0
    for k in np.unique(b):
        mk = b == k; n1 = yy[mk].sum(); n0 = mk.sum() - n1
        if min(n1, n0) < max(5, 0.08 * mk.sum()): continue
        m = min(n1, n0); num += m * ((pp[mk & (yy == 1)] == 1).mean() + (pp[mk & (yy == 0)] == 0).mean()); den += 2 * m
    return num / den if den else np.nan


rows = []; rng = np.random.default_rng(11)
for name, sc in SC.items():
    m = BASE & np.isfinite(sc)
    if m.sum() < 500: continue
    pa = (sc[m] > .5).astype(int); pb = (p_v2[m] > .5).astype(int); yy = y[m]; tt = t[m]; ss = sid[m]
    idx = {q: np.where(ss == q)[0] for q in PEOPLE if (ss == q).any()}; keys = np.array(list(idx))
    da, dt = [], []
    for _ in range(500):
        ii = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        da.append(basic(yy[ii], pa[ii]) - basic(yy[ii], pb[ii])); dt.append(tm_exact(yy[ii], pa[ii], tt[ii]) - tm_exact(yy[ii], pb[ii], tt[ii]))
    lo_a, hi_a = np.nanpercentile(da, [2.5, 97.5]); lo_t, hi_t = np.nanpercentile(dt, [2.5, 97.5])
    r = dict(model=name, n_windows=int(m.sum()), acc=basic(yy, pa), acc_v2=basic(yy, pb), tm_acc=tm_exact(yy, pa, tt), tm_acc_v2=tm_exact(yy, pb, tt),
             d_acc=basic(yy, pa) - basic(yy, pb), d_acc_lo=lo_a, d_acc_hi=hi_a, d_tm=tm_exact(yy, pa, tt) - tm_exact(yy, pb, tt), d_tm_lo=lo_t, d_tm_hi=hi_t)
    r['ship'] = bool(lo_a > 0 and lo_t > 0)
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(SUB + '/results/optics_experiment/decision_rule.csv', index=False)
print(R.round(3).to_string(index=False))
print('\nSHIP optical ratio as a model input:', bool(R.ship.any()), '(models passing the rule: %s)' % (R[R.ship].model.tolist()))
