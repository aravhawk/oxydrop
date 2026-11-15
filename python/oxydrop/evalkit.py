"""Self-contained evaluation metrics (no private data cache): accuracy, balanced accuracy, sens/spec, AUROC, time-matched accuracy,
worst person, person bootstrap. Mirrors the research kit used for the audited v1 numbers."""
import numpy as np
from scipy.stats import rankdata


def auc(y, s):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def basic(y, p, s):
    tp = ((y == 1) & (p == 1)).sum(); tn = ((y == 0) & (p == 0)).sum()
    sens = tp / max(y.sum(), 1); spec = tn / max((1 - y).sum(), 1)
    return dict(acc=(p == y).mean(), bacc=(sens + spec) / 2, sens=sens, spec=spec, auc=auc(y, s))


def usable_bins(t, y):
    b = (t // 100).astype(int)
    return [k for k in np.unique(b) if min(y[b == k].sum(), (b == k).sum() - y[b == k].sum()) >= max(5, 0.08 * (b == k).sum())]


def tm(t, y, pred, score, nrep=300, seed=0):
    """Time-matched accuracy: every 100 s elapsed-time bin class-balanced, so a clock scores ~0.5."""
    rng = np.random.default_rng(seed); b = (t // 100).astype(int); use = usable_bins(t, y); out = []
    for _ in range(nrep):
        sel = []
        for k in use:
            i1 = np.where((b == k) & (y == 1))[0]; i0 = np.where((b == k) & (y == 0))[0]; m = min(len(i1), len(i0))
            sel += list(rng.choice(i1, m, replace=False)) + list(rng.choice(i0, m, replace=False))
        sel = np.array(sel); out.append(basic(y[sel], pred[sel], score[sel])['acc'])
    return float(np.mean(out))


def boot(sid, y, pred, score, nboot=500, seed=1):
    rng = np.random.default_rng(seed); ids = np.unique(sid); a = []
    for _ in range(nboot):
        idx = np.concatenate([np.where(sid == q)[0] for q in rng.choice(ids, len(ids))])
        a.append(basic(y[idx], pred[idx], score[idx])['acc'])
    return np.percentile(a, [2.5, 97.5]).round(3).tolist()


def report(sid, t, y, score, pred, answered=None):
    """All windows evaluated at t>=60. answered: bool mask of windows the model answers (rest = abstain)."""
    a = np.ones(len(y), bool) if answered is None else answered
    r = basic(y[a], pred[a], score[a]); r['coverage'] = a.mean()
    r['worst_person'] = min(float((pred[a & (sid == q)] == y[a & (sid == q)]).mean()) for q in np.unique(sid) if (a & (sid == q)).sum() > 20)
    if a.all():
        r['tm_acc'] = tm(t, y, pred, score); r['acc_ci'] = boot(sid, y, pred, score)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
