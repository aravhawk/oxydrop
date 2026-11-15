"""Metrics, person-level bootstrap and the train-vs-test gap, as pre-registered (python/oxydrop_ml/PREREGISTRATION.md).

Conventions: arrays are per window; NaN in a prediction means "no answer". Classification label y is 1 for hypoxic (reference SpO2 < 90).
Time-matched (TM) metrics use the construction of python/oxydrop/evalkit.py: windows are binned by elapsed time t (100 s bins); a bin counts only
if each class has >= max(5, 8 %) of its windows; inside a counted bin both classes are sampled equally often (min(n1, n0) each), so a model that only
knows the clock scores 0.5. Here the expectation over that sampling is computed in closed form (no Monte Carlo), which equals the long-run
value of evalkit.tm; it is therefore both exact and deterministic. Because the sample is class balanced, TM accuracy equals TM balanced accuracy.
"""
import numpy as np

# ------------------------------------------------------------------------------------------------ basics


def auroc(y, s, w=None):
    """AUROC (ties count 1/2), optionally with window weights w. NaN if a class is missing."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(s) & (w > 0)
    y, s, w = y[ok], s[ok], w[ok]
    wp = w * (y == 1); wn = w * (y == 0)
    if wp.sum() <= 0 or wn.sum() <= 0:
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    P = np.bincount(inv, weights=wp, minlength=len(u)); N = np.bincount(inv, weights=wn, minlength=len(u))
    below = np.cumsum(N) - N
    return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


def ece(y, p, n_bins=10):
    """Expected calibration error of a probability p for a binary label (equal-width bins)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(p) & np.isfinite(y); y, p = y[ok], p[ok]
    if not len(y):
        return np.nan
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def classification_metrics(y, score, thr=0.5, pred=None):
    """accuracy, balanced accuracy, sensitivity, specificity, AUROC, ECE, Brier for a binary label and a probability-like score."""
    y = np.asarray(y).astype(int); score = np.asarray(score, float)
    p = (score > thr).astype(int) if pred is None else np.asarray(pred).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    sens = tp / n1 if n1 else np.nan; spec = tn / n0 if n0 else np.nan
    return dict(n=len(y), n_pos=n1, acc=(tp + tn) / max(len(y), 1), bacc=np.nanmean([sens, spec]) if (n1 or n0) else np.nan, sens=sens, spec=spec,
                auroc=auroc(y, score), ece=ece(y, score) if np.nanmin(score) >= 0 and np.nanmax(score) <= 1 else np.nan,
                brier=float(np.mean((score - y) ** 2)) if np.nanmin(score) >= 0 and np.nanmax(score) <= 1 else np.nan)


# ------------------------------------------------------------------------------------------------ heart rate and SpO2


def hr_metrics(pred, ref, tol=5.0):
    """HR against the reference (bpm). pred NaN = no answer.
    mae (answered), within5 (share of ANSWERED windows within tol), within5_full (share of ALL labelled windows, unanswered counted wrong = coverage-1 point),
    doubling / halving rate (answered windows whose estimate is 1.8-2.2x / 0.45-0.55x the reference), coverage, bias."""
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    lab = np.isfinite(ref)
    ans = lab & np.isfinite(pred)
    e = np.abs(pred[ans] - ref[ans])
    r = pred[ans] / ref[ans]
    n_lab = int(lab.sum())
    return dict(n=n_lab, n_answered=int(ans.sum()), coverage=ans.sum() / max(n_lab, 1), mae=float(e.mean()) if len(e) else np.nan,
                bias=float((pred[ans] - ref[ans]).mean()) if len(e) else np.nan, within5=float((e <= tol).mean()) if len(e) else np.nan,
                within5_full=float((e <= tol).sum() / max(n_lab, 1)), doubling=float(((r >= 1.8) & (r <= 2.2)).mean()) if len(e) else np.nan,
                halving=float(((r >= 0.45) & (r <= 0.55)).mean()) if len(e) else np.nan)


def spo2_metrics(pred, ref, thr=90.0):
    """Absolute SpO2: MAE, bias (pred - ref), RMSE, and AUROC for ref < thr using -pred as the score."""
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    d = pred[ok] - ref[ok]
    y = (ref[ok] < thr).astype(int)
    return dict(n=int(ok.sum()), mae=float(np.abs(d).mean()) if ok.any() else np.nan, bias=float(d.mean()) if ok.any() else np.nan,
                rmse=float(np.sqrt((d ** 2).mean())) if ok.any() else np.nan, auroc_lt=auroc(y, -pred[ok]) if ok.any() else np.nan, n_pos=int(y.sum()))


# ------------------------------------------------------------------------------------------------ time-matched


def _bins(t, y, bin_s=100, min_frac=0.08, min_n=5):
    t = np.asarray(t, float); y = np.asarray(y).astype(int)
    b = (t // bin_s).astype(int)
    use = {}
    for k in np.unique(b):
        m = b == k
        n1 = int(y[m].sum()); n0 = int(m.sum()) - n1
        if min(n1, n0) >= max(min_n, min_frac * m.sum()):
            use[k] = (n1, n0)
    return b, use


def time_matched(t, y, pred, score=None, bin_s=100, min_frac=0.08, min_n=5):
    """Time-matched metrics, expectation form (see module docstring). Returns tm_acc (= tm_bacc), tm_sens, tm_spec, tm_auroc, n_bins, n_windows_used.
    Pass `score` for the AUROC. Same bin rule as python/oxydrop/evalkit.usable_bins."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    b, use = _bins(t, y, bin_s, min_frac, min_n)
    num_s = num_p = den = 0.0
    w = np.zeros(len(y))
    for k, (n1, n0) in use.items():
        m = b == k
        mm = min(n1, n0)
        tpr = ((pred == 1) & (y == 1) & m).sum() / n1; tnr = ((pred == 0) & (y == 0) & m).sum() / n0
        num_s += mm * tpr; num_p += mm * tnr; den += mm
        w[m & (y == 1)] = mm / n1; w[m & (y == 0)] = mm / n0
    if den == 0:
        return dict(tm_acc=np.nan, tm_bacc=np.nan, tm_sens=np.nan, tm_spec=np.nan, tm_auroc=np.nan, n_bins=0, n_used=0)
    out = dict(tm_acc=(num_s + num_p) / (2 * den), tm_bacc=(num_s + num_p) / (2 * den), tm_sens=num_s / den, tm_spec=num_p / den,
               n_bins=len(use), n_used=int((w > 0).sum()))
    out['tm_auroc'] = auroc(y, score, w) if score is not None else np.nan
    return out


# ------------------------------------------------------------------------------------------------ selective prediction


def selective_metrics(y, pred, answered):
    """Metrics when the model may abstain. answered: bool mask. Returns coverage, acc_answered, bacc_answered, sens/spec on answered,
    alert_share_all = share of ALL hypoxic windows that receive a low-oxygen alert (answered and predicted 1), not only the answered ones."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int); a = np.asarray(answered, bool)
    n1 = int((y == 1).sum())
    r = dict(coverage=float(a.mean()) if len(a) else np.nan, alert_share_all=float(((pred == 1) & (y == 1) & a).sum() / n1) if n1 else np.nan)
    if a.any():
        c = classification_metrics(y[a], pred[a].astype(float), pred=pred[a])
        r.update(acc_answered=c['acc'], bacc_answered=c['bacc'], sens_answered=c['sens'], spec_answered=c['spec'], n_answered=int(a.sum()))
    else:
        r.update(acc_answered=np.nan, bacc_answered=np.nan, sens_answered=np.nan, spec_answered=np.nan, n_answered=0)
    return r


def accuracy_vs_coverage(y, pred, confidence, grid=np.linspace(0.1, 1.0, 10)):
    """Accuracy on the `c` most confident windows for each coverage c in grid. Returns list of dict(coverage, acc, bacc, alert_share_all)."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int); conf = np.asarray(confidence, float)
    order = np.argsort(-conf, kind='stable')
    out = []
    for c in grid:
        k = max(1, int(round(c * len(y))))
        a = np.zeros(len(y), bool); a[order[:k]] = True
        s = selective_metrics(y, pred, a)
        out.append(dict(coverage=s['coverage'], acc=s['acc_answered'], bacc=s['bacc_answered'], alert_share_all=s['alert_share_all']))
    return out


def per_subject(subject, values):
    """Mean of `values` (e.g. a 0/1 correctness array or an absolute error) per subject. Returns dict subject -> mean."""
    subject = np.asarray(subject); values = np.asarray(values, float)
    return {s: float(np.nanmean(values[subject == s])) for s in np.unique(subject)}


# ------------------------------------------------------------------------------------------------ person bootstrap


def _person_index(subject):
    subject = np.asarray(subject)
    u, inv = np.unique(subject, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    cuts = np.cumsum(np.bincount(inv))[:-1]
    return u, np.split(order, cuts)


def person_bootstrap(subject, stat, B=1000, seed=0, alpha=0.05):
    """Percentile bootstrap that resamples PEOPLE with replacement (all windows of a drawn person stay together).
    stat(idx) -> float is evaluated on the concatenated window indices of each resample. Returns dict(point, lo, hi, se, boot)."""
    u, rows = _person_index(subject)
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(u), len(u))
        boot[b] = stat(np.concatenate([rows[i] for i in pick]))
    point = stat(np.arange(len(np.asarray(subject))))
    return dict(point=float(point), lo=float(np.nanpercentile(boot, 100 * alpha / 2)), hi=float(np.nanpercentile(boot, 100 * (1 - alpha / 2))),
                se=float(np.nanstd(boot, ddof=1)), boot=boot)


def paired_person_bootstrap(subject, stat_a, stat_b, B=1000, seed=0, alpha=0.05, higher_is_better=True):
    """Paired person-level bootstrap of the difference stat_a - stat_b of two models evaluated on the SAME windows (same resampled people for both).
    A pre-registered win needs the 95 % interval of the difference (in the better-is-positive direction) to exclude zero.
    Returns dict(diff, lo, hi, a, b, excludes_zero, win, boot)."""
    u, rows = _person_index(subject)
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(u), len(u))
        idx = np.concatenate([rows[j] for j in pick])
        d[i] = stat_a(idx) - stat_b(idx)
    allr = np.arange(len(np.asarray(subject)))
    a, b = stat_a(allr), stat_b(allr)
    sgn = 1.0 if higher_is_better else -1.0
    lo, hi = np.nanpercentile(sgn * d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(diff=float(sgn * (a - b)), lo=float(lo), hi=float(hi), a=float(a), b=float(b), excludes_zero=bool(lo > 0 or hi < 0), win=bool(lo > 0), boot=sgn * d)


def summarize_folds(values, B=2000, seed=0):
    """mean, SD (ddof 1), 95 % bootstrap CI of the mean over folds (subjects), min, max, n. NaN folds are dropped."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v):
        return dict(mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan, min=np.nan, max=np.nan, n=0)
    rng = np.random.default_rng(seed)
    bm = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def train_vs_test_gap(train_scores, test_scores, higher_is_better=True, B=2000, seed=0):
    """Per-fold train score vs held-out score of the same model on the same metric. gap = train - test when higher is better, test - train otherwise,
    so a positive gap always means over-fitting. Returns dict(train_mean, test_mean, gap, gap_sd, gap_lo, gap_hi, per_fold_gap)."""
    tr = np.asarray(train_scores, float); te = np.asarray(test_scores, float)
    assert len(tr) == len(te), 'one train and one test score per fold'
    g = (tr - te) if higher_is_better else (te - tr)
    s = summarize_folds(g, B, seed)
    return dict(train_mean=float(np.nanmean(tr)), test_mean=float(np.nanmean(te)), gap=s['mean'], gap_sd=s['sd'], gap_lo=s['lo'], gap_hi=s['hi'],
                per_fold_gap=g.tolist())


def classification_stat(y, score, thr=0.5, key='acc'):
    """Build a stat(idx) for the bootstrap helpers: classification metric `key` of the rows idx."""
    y = np.asarray(y); score = np.asarray(score, float)
    return lambda idx: classification_metrics(y[idx], score[idx], thr)[key]


def tm_stat(t, y, score, thr=0.5, key='tm_acc'):
    """stat(idx) for the time-matched metric `key` of the rows idx."""
    t = np.asarray(t); y = np.asarray(y); score = np.asarray(score, float)
    return lambda idx: time_matched(t[idx], y[idx], (score[idx] > thr).astype(int), score[idx])[key]
