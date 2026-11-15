"""Subject-level cross-fit selection of the HR gate (F_LO, SQI_MIN, ACF_MIN, AGREE) on BUT PPG.

Protocol (no evaluation subject ever helps choose the gate that is scored on it):
  fold A: choose on EVEN-numbered subjects, score on ODD-numbered subjects
  fold B: choose on ODD-numbered subjects,  score on EVEN-numbered subjects
Objective on the selection half: minimise the wrong-accepted rate (|HR error| > 10 bpm among accepted clips) subject to accepting
at least 70 % of the expert-good clips. The shipped config is the one the two folds support; if they disagree the more conservative one
(lower acceptance over all clips of its own selection half and the other half) is shipped.
Usage: python scripts/hr_crossfit.py   (needs the BUT cache written by `validate_hr.py but`)
Outputs: results/hr_validation/crossfit_grid.csv, crossfit_summary.csv, crossfit_choice.json
"""
import os, sys, json, itertools
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oxydrop import hr as HR  # noqa: E402
import validate_hr as V  # noqa: E402

OUT = V.OUT
CACHE = os.path.join(V.TMP, 'but_cache.pkl')
F_LOS = (0.65, 0.75, 0.85)
SQIS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
ACFS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
AGREES = (0.06, 0.10, 0.15)
MIN_GOOD_COVERAGE = 0.70


def raw_arrays(wins, f_lo):
    """For every clip: raw (hr, hr_acf, sqi, acf) of the 3 channels in best_channel_estimate order -> arrays (N,3)."""
    N = len(wins); H = np.full((N, 3), np.nan); HA = H.copy(); S = np.zeros((N, 3)); A = np.zeros((N, 3))
    for i, w in enumerate(wins):
        sd = w.std(0); primary = 1 if sd[1] > 0.3 * sd[0] else 0
        order = [primary] + [int(c) for c in np.argsort(-sd) if c != primary]
        for j, c in enumerate(order):
            e = HR.estimate_hr(w[:, c], V.FS, f_lo)
            H[i, j], HA[i, j], S[i, j], A[i, j] = e['hr'], e['hr_acf'], e['sqi'], e['acf']
    return H, HA, S, A


def apply_gate(R, sqi_min, acf_min, agree):
    """Vectorised best_channel_estimate: first channel in order that passes the gate; else the primary's (invalid) hr."""
    H, HA, S, A = R
    with np.errstate(invalid='ignore', divide='ignore'):
        ok = np.isfinite(H) & np.isfinite(HA) & (H > 0) & (np.abs(HA - H) / H < agree) & (S >= sqi_min) & (A >= acf_min)
    first = np.where(ok.any(1), ok.argmax(1), 0)
    h = H[np.arange(len(H)), first]
    return h, ok.any(1)


def stats(h, valid, D, mask):
    ref = D.ref.values[mask]; good = (D.q == 1).values[mask]
    h = h[mask]; v = valid[mask] & np.isfinite(h)
    err = np.abs(h - ref)
    e = err[v]; hh = h[v]; rr = ref[v]
    return dict(n=int(mask.sum()), n_acc=int(v.sum()), accept_all=v.mean(), accept_good=v[good].mean() if good.any() else np.nan,
                accept_bad=v[~good].mean() if (~good).any() else np.nan,
                MAE_acc=e.mean() if len(e) else np.nan, medAE_acc=np.median(e) if len(e) else np.nan,
                w5_acc=(e <= 5).mean() if len(e) else np.nan, w10_acc=(e <= 10).mean() if len(e) else np.nan,
                wrong10_acc=(e > 10).mean() if len(e) else np.nan,
                dbl_acc=(np.abs(hh - 2 * rr) < 0.24 * rr).mean() if len(e) else np.nan,
                half_acc=(np.abs(hh - 0.5 * rr) < 0.06 * rr).mean() if len(e) else np.nan,
                MAE_acc_good=err[v & good].mean() if (v & good).any() else np.nan,
                w5_acc_good=(err[v & good] <= 5).mean() if (v & good).any() else np.nan,
                MAE_all=np.nanmean(err), w5_all=np.nanmean(err <= 5))


def main():
    D, _, wins = pd.read_pickle(CACHE)
    D = D.reset_index(drop=True)
    even = (D.subj % 2 == 0).values
    print('clips', len(D), 'subjects', D.subj.nunique(), 'even', even.sum(), 'odd', (~even).sum(), flush=True)
    raws = {f: raw_arrays(wins, f) for f in F_LOS}
    print('raw estimates done', flush=True)
    rows = []
    for f, s_, a_, g_ in itertools.product(F_LOS, SQIS, ACFS, AGREES):
        h, v = apply_gate(raws[f], s_, a_, g_)
        for half, m in (('even', even), ('odd', ~even), ('all', np.ones(len(D), bool))):
            rows.append(dict(F_LO=f, SQI_MIN=s_, ACF_MIN=a_, AGREE=g_, half=half, **stats(h, v, D, m)))
    G = pd.DataFrame(rows)
    G.to_csv(f'{OUT}/crossfit_grid.csv', index=False)

    def choose(sel_half, f_lo=None):
        c = G[(G.half == sel_half) & (G.accept_good >= MIN_GOOD_COVERAGE)]
        if f_lo is not None:
            c = c[c.F_LO == f_lo]
        # minimise wrong-accepted rate; tie-break by higher good-clip coverage then lower MAE
        return c.sort_values(['wrong10_acc', 'accept_good', 'MAE_acc'], ascending=[True, False, True]).iloc[0]

    def lookup(cfg, half):
        q = G[(G.F_LO == cfg.F_LO) & (G.SQI_MIN == cfg.SQI_MIN) & (G.ACF_MIN == cfg.ACF_MIN) & (G.AGREE == cfg.AGREE) & (G.half == half)]
        return q.iloc[0]

    summ = []; choice = {}
    for f_lo in (None,) + F_LOS:
        tag = 'F_LO chosen by cross-fit' if f_lo is None else f'F_LO fixed {f_lo}'
        picks = {}
        held_h, held_v, held_m = np.full(len(D), np.nan), np.zeros(len(D), bool), np.zeros(len(D), bool)
        for sel_half, test_half in (('even', 'odd'), ('odd', 'even')):
            cfg = choose(sel_half, f_lo)
            picks[sel_half] = cfg
            tm = (~even) if test_half == 'odd' else even
            h, v = apply_gate(raws[cfg.F_LO], cfg.SQI_MIN, cfg.ACF_MIN, cfg.AGREE)
            held_h[tm] = h[tm]; held_v[tm] = v[tm]; held_m[tm] = True
            sel = lookup(cfg, sel_half); te = lookup(cfg, test_half)
            summ.append(dict(setting=tag, selected_on=sel_half + ' subjects', scored_on=test_half + ' subjects (held out)', F_LO=cfg.F_LO, SQI_MIN=cfg.SQI_MIN,
                             ACF_MIN=cfg.ACF_MIN, AGREE=cfg.AGREE, **{f'sel_{k}': sel[k] for k in ('accept_good', 'wrong10_acc', 'MAE_acc')},
                             **{k: te[k] for k in ('n', 'n_acc', 'accept_all', 'accept_good', 'accept_bad', 'MAE_acc', 'medAE_acc', 'w5_acc', 'w10_acc',
                                                   'wrong10_acc', 'dbl_acc', 'half_acc', 'MAE_acc_good', 'w5_acc_good')}))
        # pooled held-out (every clip scored with the gate chosen WITHOUT its own subject half)
        pooled = stats(held_h, held_v, D, held_m)
        summ.append(dict(setting=tag, selected_on='other half', scored_on='all subjects, pooled held-out',
                         F_LO='/'.join(str(picks[k].F_LO) for k in ('even', 'odd')),
                         SQI_MIN='/'.join(str(picks[k].SQI_MIN) for k in ('even', 'odd')), ACF_MIN='/'.join(str(picks[k].ACF_MIN) for k in ('even', 'odd')),
                         AGREE='/'.join(str(picks[k].AGREE) for k in ('even', 'odd')), **{k: pooled[k] for k in
                         ('n', 'n_acc', 'accept_all', 'accept_good', 'accept_bad', 'MAE_acc', 'medAE_acc', 'w5_acc', 'w10_acc', 'wrong10_acc', 'dbl_acc', 'half_acc',
                          'MAE_acc_good', 'w5_acc_good')}))
        # shipped config for this setting: identical picks -> that; else the more conservative (lower acceptance on all clips)
        a, b = picks['even'], picks['odd']
        same = all(a[k] == b[k] for k in ('F_LO', 'SQI_MIN', 'ACF_MIN', 'AGREE'))
        if same:
            ship, why = a, 'both folds chose the same gate'
        else:
            acc = lambda c: lookup(c, 'all').accept_all
            ship = a if acc(a) <= acc(b) else b
            why = 'folds disagreed; shipped the more conservative (lower acceptance over all clips)'
        choice[tag] = dict(F_LO=float(ship.F_LO), SQI_MIN=float(ship.SQI_MIN), ACF_MIN=float(ship.ACF_MIN), AGREE=float(ship.AGREE), why=why,
                           even=dict((k, float(a[k])) for k in ('F_LO', 'SQI_MIN', 'ACF_MIN', 'AGREE')), odd=dict((k, float(b[k])) for k in ('F_LO', 'SQI_MIN', 'ACF_MIN', 'AGREE')))
    S = pd.DataFrame(summ); S.to_csv(f'{OUT}/crossfit_summary.csv', index=False)
    json.dump(choice, open(f'{OUT}/crossfit_choice.json', 'w'), indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(S.round(3).to_string()); print(json.dumps(choice, indent=1))


if __name__ == '__main__':
    main()
