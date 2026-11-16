"""Common evaluator of every photon-stress score file.

    cd python && python -m oxydrop_ml.photon.evalkit [--boot 2000] [--scores DIR] [--out DIR] [--only chain_v1,flashgap]

Reads results/photon/scores/<model>__L<level>.csv (columns rec, t, sid, y_spo2, y, score, pred, answer, optional spo2_hat, spo2_lo, spo2_hi, conf; see
results/photon/HARNESS.md) and writes results/photon/TABLE.csv, TABLE.md and STRESS.csv.

Every file is aligned to the 5,644 evaluated Hoffman windows (t >= 60 s) of results/scores/windows.csv; a file that misses any of them is reported and skipped.
Metrics per (model, level):
  tm_bacc (+95 % person-bootstrap CI, 2000 resamples)  Hoffman time-matched balanced accuracy at the forced decision `pred` (metrics.time_matched; clock-only = 0.5)
  bacc, auroc (+ CI)                                    plain pooled balanced accuracy and AUROC of `score`
  tm_<person>, worst_person_tm                          time-matched balanced accuracy of each held-out person (NaN when no 100 s bin holds both classes)
  diff_vs_chain_v1, diff_lo, diff_hi, verdict           paired person bootstrap of the tm_bacc difference to `chain_v1` at the SAME level (same resampled people);
                                                         win = the interval is above 0, loss = below 0, tie otherwise (a tie must be called a tie)
  coverage, acc_ans, bacc_ans                           the model's own selective decision (`answer`): share answered, accuracy and balanced accuracy on answered windows
  acc_cov100 .. acc_cov50, aurc                         accuracy of `pred` on the most confident 100/90/80/70/60/50 % of windows by `conf` (default |score - 0.5|), and the
                                                         area under the risk-coverage curve (mean error rate of the top-k set over k = 1..n)
  n_people_improved, n_people_compared, sign_p          held-out people whose own TM bal. acc. beats chain_v1 at the same level (ties / unusable people dropped) and the exact one-sided sign test
  null_check                                            for files named <model>_null__L0 or <model>_blank__L0: 'ok' when TM <= 0.55, otherwise a FLAG (also listed in TABLE.md)
  X1 files (<model>__X1.csv) are read like levels and reported after level 5 (stress.degrade_x).
  spo2_mae                                              mean |spo2_hat - y_spo2| when spo2_hat is present
The bootstrap resamples PEOPLE (both hands of a person stay together) and uses the same resampled people for every model, so differences are paired.
The time-matched statistic is recomputed from per-person bin counts in closed form (identical to metrics.time_matched on the resampled windows; tested).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import metrics as M
import math

from .stress import D_RHO_MM, LEVELS, X_INFO, X_LEVELS

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
WINDOWS = REPO / 'results' / 'scores' / 'windows.csv'
BASE = 'chain_v1'
COVS = (100, 90, 80, 70, 60, 50)
BIN_S, MIN_FRAC, MIN_N = 100, 0.08, 5
NULL_SUFFIX = ('_null', '_blank')      # null controls: score files <model>_null__L0 / <model>_blank__L0 must be ~0.50 TM
NULL_MAX = 0.55
ALL_LEVELS = LEVELS + X_LEVELS         # 0-5 and the out-of-family stress 'X1' (files <model>__X1.csv)


def level_key(l):
    return (0, l) if isinstance(l, (int, np.integer)) else (1, X_LEVELS.index(l))


def d_rho(l):
    return D_RHO_MM[l] if l in D_RHO_MM else float(D_RHO_MM[X_INFO[l]['base_level']])


def sign_test(n_better, n_total):
    """Exact one-sided sign test p = P(X >= n_better), X ~ Binomial(n_total, 0.5). NaN when n_total = 0."""
    if n_total <= 0:
        return np.nan
    return float(sum(math.comb(n_total, k) for k in range(n_better, n_total + 1)) / 2 ** n_total)


# ------------------------------------------------------------------------------------------------ the evaluated window set
def eval_windows():
    """DataFrame (rec, t, sid, y_spo2, y) of the evaluated windows (t >= 60 s), rec without the 'hof_' prefix."""
    w = pd.read_csv(WINDOWS)
    w = w[w.t >= 60].reset_index(drop=True)
    return pd.DataFrame(dict(rec=w.rec.astype(str), t=w.t.astype(int), sid=w.sid.astype(str), y_spo2=w.y.astype(float), y=(w.y < 90).astype(int)))


def _norm_rec(s):
    return s.astype(str).str.replace('hof_', '', regex=False)


def load_scores(path, ref):
    """Read one score file and align it to `ref` (eval_windows()). Returns a DataFrame with ref's columns + score, pred, answer, conf (+ spo2_hat...)."""
    d = pd.read_csv(path)
    d['rec'] = _norm_rec(d['rec']); d['t'] = d['t'].astype(int)
    d = d[d.t >= 60].drop_duplicates(['rec', 't'])
    m = ref[['rec', 't']].merge(d, on=['rec', 't'], how='left', validate='one_to_one')
    if m['score'].isna().any() or m['pred'].isna().any():
        raise ValueError(f'{path.name}: {int(m["score"].isna().sum())} of {len(ref)} evaluated windows have no score')
    out = ref.copy()
    out['score'] = m['score'].astype(float).values
    out['pred'] = m['pred'].astype(int).values
    ans = m['answer'] if 'answer' in m else pd.Series(True, index=m.index)
    out['answer'] = (ans.astype(str).str.lower().isin(['true', '1', '1.0'])).values
    out['conf'] = (m['conf'].astype(float) if 'conf' in m else (m['score'] - 0.5).abs()).values
    for c in ('spo2_hat', 'spo2_lo', 'spo2_hi'):
        if c in m:
            out[c] = m[c].astype(float).values
    return out


# ------------------------------------------------------------------------------------------------ closed-form time-matched bootstrap
class Boot:
    """Person bootstrap shared by all models: the same resampled people for every file, so differences are paired."""

    def __init__(self, ref, B=2000, seed=0):
        self.sid = ref.sid.values; self.t = ref.t.values.astype(float); self.y = ref.y.values
        self.u, self.rows = M._person_index(self.sid)
        P = len(self.u)
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, P, (B, P))
        self.W = np.stack([np.bincount(p, minlength=P) for p in picks]).astype(float)          # [B, P] times each person is drawn
        self.bin = (self.t // BIN_S).astype(int); self.nb = int(self.bin.max()) + 1
        self.B, self.P = B, P

    def counts(self, pred):
        """[P, nb, 4] per-person per-bin counts of n1, n0, tp, tn."""
        C = np.zeros((self.P, self.nb, 4))
        y = self.y; pred = np.asarray(pred).astype(int)
        for i, r in enumerate(self.rows):
            b = self.bin[r]
            np.add.at(C[i, :, 0], b, y[r] == 1); np.add.at(C[i, :, 1], b, y[r] == 0)
            np.add.at(C[i, :, 2], b, (y[r] == 1) & (pred[r] == 1)); np.add.at(C[i, :, 3], b, (y[r] == 0) & (pred[r] == 0))
        return C

    @staticmethod
    def _tm(tot):
        """tot [..., nb, 4] summed counts -> tm_bacc [...], metrics.time_matched rule."""
        n1, n0, tp, tn = (tot[..., k] for k in range(4))
        use = np.minimum(n1, n0) >= np.maximum(MIN_N, MIN_FRAC * (n1 + n0))
        mm = np.where(use, np.minimum(n1, n0), 0.0)
        den = mm.sum(-1)
        num = (mm * (tp / np.maximum(n1, 1) + tn / np.maximum(n0, 1))).sum(-1)
        return np.where(den > 0, num / np.maximum(2 * den, 1e-12), np.nan)

    def tm(self, pred):
        """(point, boot [B], per-person [P]) of the time-matched balanced accuracy."""
        C = self.counts(pred)
        point = float(self._tm(C.sum(0)))
        boot = self._tm(np.einsum('bp,pkc->bkc', self.W, C))
        per = np.array([float(self._tm(C[i])) for i in range(self.P)])
        return point, boot, per

    def bacc(self, pred):
        y = self.y; pred = np.asarray(pred).astype(int)
        C = np.zeros((self.P, 4))
        for i, r in enumerate(self.rows):
            C[i] = [(y[r] == 1).sum(), (y[r] == 0).sum(), ((y[r] == 1) & (pred[r] == 1)).sum(), ((y[r] == 0) & (pred[r] == 0)).sum()]
        f = lambda T: 0.5 * (T[..., 2] / np.maximum(T[..., 0], 1) + T[..., 3] / np.maximum(T[..., 1], 1))
        return float(f(C.sum(0))), f(self.W @ C)

    def auroc(self, score):
        score = np.asarray(score, float)
        point = M.auroc(self.y, score)
        out = np.empty(self.B)
        for b in range(self.B):
            w = self.W[b]
            idx = np.concatenate([self.rows[i] for i in np.nonzero(w)[0] for _ in range(int(w[i]))])
            out[b] = M.auroc(self.y[idx], score[idx])
        return point, out


def ci(v, alpha=0.05):
    v = np.asarray(v, float)
    return (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))) if np.isfinite(v).any() else (np.nan, np.nan)


# ------------------------------------------------------------------------------------------------ selective metrics
def acc_at_coverage(y, pred, conf, cov):
    order = np.argsort(-np.asarray(conf, float), kind='stable')
    k = max(1, int(round(cov * len(y))))
    top = order[:k]
    return float((np.asarray(pred)[top] == np.asarray(y)[top]).mean())


def aurc(y, pred, conf):
    """Area under the risk-coverage curve: mean over k of the error rate of the k most confident windows (lower is better)."""
    order = np.argsort(-np.asarray(conf, float), kind='stable')
    err = (np.asarray(pred)[order] != np.asarray(y)[order]).astype(float)
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


# ------------------------------------------------------------------------------------------------ one table row
def evaluate(d, boot):
    """Metrics of one aligned score frame. Returns (row dict, tm boot array)."""
    y = d.y.values; pred = d.pred.values; ans = d.answer.values
    tm, tm_b, per = boot.tm(pred)
    ba, ba_b = boot.bacc(pred)
    au, au_b = boot.auroc(d.score.values)
    r = dict(n=len(d), tm_bacc=tm, tm_lo=ci(tm_b)[0], tm_hi=ci(tm_b)[1], bacc=ba, bacc_lo=ci(ba_b)[0], bacc_hi=ci(ba_b)[1], auroc=au, auroc_lo=ci(au_b)[0], auroc_hi=ci(au_b)[1],
             worst_person_tm=float(np.nanmin(per)) if np.isfinite(per).any() else np.nan)
    for s, v in zip(boot.u, per):
        r[f'tm_{s}'] = v
    r['coverage'] = float(ans.mean())
    r['acc_ans'] = float((pred[ans] == y[ans]).mean()) if ans.any() else np.nan
    if ans.any():
        yy, pp = y[ans], pred[ans]
        s1, s0 = (pp[yy == 1] == 1).mean() if (yy == 1).any() else np.nan, (pp[yy == 0] == 0).mean() if (yy == 0).any() else np.nan
        r['bacc_ans'] = float(np.nanmean([s1, s0]))
    else:
        r['bacc_ans'] = np.nan
    for c in COVS:
        r[f'acc_cov{c}'] = acc_at_coverage(y, pred, d.conf.values, c / 100)
    r['aurc'] = aurc(y, pred, d.conf.values)
    r['spo2_mae'] = float(np.nanmean(np.abs(d.spo2_hat.values - d.y_spo2.values))) if 'spo2_hat' in d and np.isfinite(d.spo2_hat.values).any() else np.nan
    return r, tm_b


def verdict(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ''
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def parse_name(p):
    m = re.match(r'^(.+)__(?:L(\d+)|(X\d+))$', Path(p).stem)
    if not m:
        return None, None
    return (m.group(1), int(m.group(2))) if m.group(2) is not None else (m.group(1), m.group(3))


def run(scores_dir=SCORES, out_dir=RES, B=2000, only=None, log=print):
    ref = eval_windows()
    boot = Boot(ref, B=B)
    files = sorted(list(Path(scores_dir).glob('*__L*.csv')) + list(Path(scores_dir).glob('*__X*.csv')))
    rows, tmb, frames = [], {}, {}
    for p in files:
        model, level = parse_name(p)
        if model is None or level not in ALL_LEVELS or (only and model not in only):
            continue
        try:
            d = load_scores(p, ref)
        except Exception as e:                        # report, never silently drop
            log(f'SKIP {p.name}: {e}'); continue
        r, tm_b = evaluate(d, boot)
        rows.append(dict(model=model, level=level, d_rho_mm=d_rho(level), **r))
        tmb[(model, level)] = (r['tm_bacc'], tm_b); frames[(model, level)] = d
        log(f'{model} {level if isinstance(level, str) else "L" + str(level)}: TM bal. acc. {r["tm_bacc"]:.3f} [{r["tm_lo"]:.3f}, {r["tm_hi"]:.3f}]')
    if not rows:
        raise SystemExit(f'no score files in {scores_dir}')
    T = pd.DataFrame(rows)
    dd, dl, dh, vv = [], [], [], []
    for _, r in T.iterrows():
        k, kb = (r.model, r.level), (BASE, r.level)
        if r.model == BASE or kb not in tmb:
            dd.append(np.nan); dl.append(np.nan); dh.append(np.nan); vv.append('' if r.model != BASE else 'reference')
            continue
        pa, ba = tmb[k]; pb, bb = tmb[kb]
        lo, hi = ci(ba - bb)
        dd.append(pa - pb); dl.append(lo); dh.append(hi); vv.append(verdict(lo, hi))
    T['diff_vs_chain_v1'] = dd; T['diff_lo'] = dl; T['diff_hi'] = dh; T['verdict'] = vv
    # per-person comparison with chain_v1 at the same level (time-matched balanced accuracy of each held-out person) and the exact one-sided sign test
    pcols = [f'tm_{s}' for s in boot.u]
    nb, nc, sp = [], [], []
    for _, r in T.iterrows():
        b = T[(T.model == BASE) & (T.level == r.level)]
        if r.model == BASE or not len(b):
            nb.append(np.nan); nc.append(np.nan); sp.append(np.nan); continue
        a_, b_ = r[pcols].values.astype(float), b.iloc[0][pcols].values.astype(float)
        ok = np.isfinite(a_) & np.isfinite(b_) & (a_ != b_)
        nb.append(int((a_[ok] > b_[ok]).sum())); nc.append(int(ok.sum())); sp.append(sign_test(nb[-1], nc[-1]))
    T['n_people_improved'] = nb; T['n_people_compared'] = nc; T['sign_p'] = sp
    # null controls: *_null / *_blank at level 0 must score ~0.50 TM
    T['null_check'] = [('' if not (r.model.endswith(NULL_SUFFIX) and r.level == 0) else (f'FLAG: null control TM {r.tm_bacc:.3f} > {NULL_MAX}' if r.tm_bacc > NULL_MAX else 'ok'))
                       for _, r in T.iterrows()]
    T['_lk'] = T.level.map(lambda l: level_key(l)[0] * 100 + level_key(l)[1])
    T = T.sort_values(['_lk', 'tm_bacc'], ascending=[True, False]).drop(columns='_lk').reset_index(drop=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    T.to_csv(Path(out_dir) / 'TABLE.csv', index=False, float_format='%.5g')
    S = T[['model', 'level', 'd_rho_mm', 'tm_bacc', 'tm_lo', 'tm_hi', 'bacc', 'auroc', 'worst_person_tm', 'diff_vs_chain_v1', 'diff_lo', 'diff_hi', 'verdict',
           'n_people_improved', 'n_people_compared', 'sign_p', 'coverage', 'acc_ans', 'aurc', 'spo2_mae']].copy()
    S['_k'] = S.level.map(lambda l: level_key(l)[0] * 100 + level_key(l)[1])
    S.sort_values(['model', '_k']).drop(columns='_k').to_csv(Path(out_dir) / 'STRESS.csv', index=False, float_format='%.5g')
    (Path(out_dir) / 'TABLE.md').write_text(to_markdown(T, B))
    return T


def to_markdown(T, B=2000):
    f = lambda v, n=3: '' if not np.isfinite(v) else f'{v:.{n}f}'
    out = ['# Photon-stress table', '',
           f'Hoffman, leave-one-person-out (both hands together), {int(T.n.iloc[0])} evaluated windows (t >= 60 s). TM bal. acc. = Hoffman time-matched balanced accuracy at the forced decision '
           f'(clock-only = 0.5) with the {B}-resample person-bootstrap 95 % CI. Difference = paired person bootstrap against `{BASE}` at the same level: '
           '**win** only when the interval excludes 0 on the positive side, otherwise **tie** (or loss). "People better" = held-out people whose own TM balanced accuracy beats '
           f'`{BASE}` (ties and people without a usable 100 s bin dropped) with the exact one-sided sign test (6/6: p = 0.016, 5/6: p = 0.109). Levels 0-5 are the simulated '
           'distant-flash stress of `stress.py`; X1 is the out-of-family stress (motion steps, frame drops, tone curve, AR(1) noise). Both are simulations, not measurements '
           'of any phone. Full columns: `TABLE.csv`.', '']
    nul = T[T.null_check != '']
    if len(nul):
        out += ['## Null controls (level 0, must be ~0.50 TM; flagged above 0.55)', '', '| model | TM bal. acc. [95 % CI] | check |', '|---|---|---|']
        for _, r in nul.iterrows():
            out.append(f'| {r.model} | {f(r.tm_bacc)} [{f(r.tm_lo)}, {f(r.tm_hi)}] | {"**" + r.null_check + "**" if r.null_check != "ok" else "ok"} |')
        out.append('')
    for lev in sorted(T.level.unique(), key=level_key):
        g = T[T.level == lev]
        title = f'Level {lev} (flash +{D_RHO_MM[lev]:g} mm)' if lev in D_RHO_MM else f'Stress {lev}: {X_INFO[lev]["desc"]}'
        out += [f'## {title}', '',
                '| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |',
                '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in g.iterrows():
            d = '' if not np.isfinite(r.diff_vs_chain_v1) else f'{r.diff_vs_chain_v1:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]'
            pb = '' if not np.isfinite(r.n_people_compared) else f'{int(r.n_people_improved)}/{int(r.n_people_compared)} (p = {r.sign_p:.3f})'
            out.append(f'| {r.model} | {f(r.tm_bacc)} [{f(r.tm_lo)}, {f(r.tm_hi)}] | {f(r.bacc)} | {f(r.auroc)} | {d} | {r.verdict} | {pb} | {f(r.worst_person_tm)} | '
                       f'{f(r.coverage, 2)} | {f(r.acc_ans)} | {f(r.aurc)} | {f(r.spo2_mae, 2)} |')
        out.append('')
    out += ['## Accuracy at coverage (by `conf`)', '', '| model | level | ' + ' | '.join(f'{c} %' for c in COVS) + ' |', '|---|---|' + '---|' * len(COVS)]
    T = T.assign(_lk=T.level.map(lambda l: level_key(l)[0] * 100 + level_key(l)[1])).sort_values(['model', '_lk'])
    for _, r in T.iterrows():
        out.append(f'| {r.model} | {r.level} | ' + ' | '.join(f(r[f"acc_cov{c}"]) for c in COVS) + ' |')
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--scores', default=str(SCORES)); ap.add_argument('--out', default=str(RES))
    ap.add_argument('--only', default=None, help='comma-separated model names')
    a = ap.parse_args(argv)
    run(a.scores, a.out, B=a.boot, only=a.only.split(',') if a.only else None)


if __name__ == '__main__':
    main()
