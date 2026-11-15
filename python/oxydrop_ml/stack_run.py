"""Hybrid (physics chain + CNN) P1 models fit inside nested CV, and test-time BatchNorm adaptation on the person's own baseline windows.

    python -m oxydrop_ml.stack_run run --out RUNS/stack/s0 --init ckpt/stageB_full.pt --recipe recipe.json --shard i/n     # CNN fits (inner and outer)
    python -m oxydrop_ml.stack_run report --runs RUNS/stack --out results/ml                                              # meta-learners, selection, tables

Protocol (everything is P1, Hoffman, leave-one-person-out; the outer person is never read before its fold is evaluated):
  For each outer person o (6) and each other Hoffman person q (5 inner people): fit the CNN on all rows without {o, q} (plus the usual inner-validation
  people for early stopping) and score q: these are the inner out-of-fold (OOF) CNN predictions of the training people. The outer CNN is fit without {o}
  only (the standard LOSO fold) and scores o. The v1 logistic chain (two ridge-logistic models on the 19 calibration-relative FUSION features, shipped C)
  is fit the same way: on the training people minus q for the inner OOF, on all training people for o.
  Candidates per outer fold (all use the label-free first minute of the person only):
     cnn            CNN logit
     cnn_bnA        CNN logit with BatchNorm statistics re-estimated on the person's own baseline windows (test-time adaptation), blend A in {0.5, 1}
     chain          v1 chain logit
     stack          logistic meta-learner on [CNN logit, chain logit]   (fit on the 5 inner people's OOF predictions)
     stack_bnA      same with the adapted CNN logit
  The candidate used for the outer person is chosen by leave-one-inner-person-out CV of the candidate on the inner OOF predictions (time-matched balanced
  accuracy), i.e. by inner folds only. All candidates are also reported (exploratory) for transparency.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import DATA_DIR
from . import metrics as M
from .train import HP, Pool, fit, fold_rows, predict, variant_hp, Norm, _emb_for_rows

ALPHAS = (0.0, 0.5, 1.0)


# ------------------------------------------------------------------------------------------------ test-time BatchNorm adaptation
@torch.no_grad()
def adapted_copy(net, pool: Pool, base_rows, norm: Norm, hp: HP, alpha, device='cpu'):
    """Copy of `net` whose BatchNorm running statistics are blended with the statistics of the person's own baseline windows:
    new = (1 - alpha) * trained + alpha * baseline. Label-free: only the first minute of the person's own recording is used."""
    if alpha <= 0 or len(base_rows) < 4:
        return net
    na = copy.deepcopy(net).to(device)
    bns = [m for m in na.modules() if isinstance(m, nn.BatchNorm1d)]
    orig = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = 1.0
    na.train()
    for m in na.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    x, s, dom = pool.inputs(base_rows, hp=hp)
    x, s = norm.apply(x, s, dom)
    na(torch.from_numpy(x).to(device), torch.from_numpy(dom).to(device), torch.from_numpy(s).to(device), None)
    for m, (om, ov) in zip(bns, orig):
        m.running_mean.copy_((1 - alpha) * om + alpha * m.running_mean)
        m.running_var.copy_((1 - alpha) * ov + alpha * m.running_var)
        m.momentum = 0.1
    na.eval()
    return na


def predict_alphas(net, pool: Pool, rows, norm: Norm, hp: HP, alphas=ALPHAS, device='cpu'):
    """dict alpha -> dict of predict() outputs, adapting per recording group (base_key) on that group's baseline windows."""
    rows = np.asarray(rows)
    out = {a: {k: np.zeros(len(rows)) for k in ('hr', 'spo2', 'p_low', 'p_sqi')} for a in alphas}
    for g in np.unique(pool.grp[rows]):
        sel = np.nonzero(pool.grp[rows] == g)[0]
        grp_rows = np.nonzero((pool.grp == g) & pool.in_base)[0]
        for a in alphas:
            n2 = adapted_copy(net, pool, grp_rows, norm, hp, a, device)
            o = predict(n2, pool, rows[sel], norm, hp, device)
            for k in out[a]:
                out[a][k][sel] = o[k]
    return out


# ------------------------------------------------------------------------------------------------ chain
def chain_logit(Xtr, ytr, Xte, C_hr=0.01, C_fus=0.1):
    """v1 chain as refit by FEAT-005 (baselines/chain_v1.py): mean of two class-balanced logistic regressions (dHR alone, 19 FUSION deltas); returns the logit of the mean probability."""
    p = 0.0
    for cols, c in (([0], C_hr), (list(range(19)), C_fus)):
        sc = StandardScaler().fit(Xtr[:, cols])
        lr = LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(sc.transform(Xtr[:, cols]), ytr)
        p = p + lr.predict_proba(sc.transform(Xte[:, cols]))[:, 1]
    p = np.clip(p / 2, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------ CNN fits
def jobs(pool: Pool):
    hof = pool.subjects_of('hoffman')
    out = []
    for o in hof:
        out.append(('outer', o, None))
        for q in hof:
            if q != o:
                out.append(('inner', o, q))
    return out


def run_job(pool: Pool, job, hp: HP, seed, device, out_dir, log=print):
    kind, o, q = job
    name = f'{kind}_{o}' + (f'_{q}' if q else '')
    cp = os.path.join(out_dir, name + '.csv')
    if os.path.exists(cp):
        return
    t0 = time.time()
    excl = [o] + ([q] if q else [])
    tr, va, vs = fold_rows(pool, excl, seed)
    fr = fit(pool, tr, va, hp, seed, device)
    target = q if q else o
    rows = np.nonzero((pool.subject == target) & pool.eval & pool.has_low)[0]
    pa = predict_alphas(fr.net, pool, rows, fr.norm, hp, ALPHAS, device)
    df = pd.DataFrame(dict(outer=o, inner=q or '', kind=kind, subject=pool.subject[rows], rec=pool.meta.rec.values[rows], t_start=pool.t[rows], low=pool.low[rows],
                           local=pool.local[rows], seed=seed))
    for a in ALPHAS:
        p = np.clip(pa[a]['p_low'], 1e-6, 1 - 1e-6)
        df[f'logit_a{int(a * 100)}'] = np.log(p / (1 - p))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(cp, index=False)
    log(f'{name} {time.time() - t0:.0f}s best_step {fr.best_step}')


# ------------------------------------------------------------------------------------------------ meta-learning and selection
def tm_bacc(t, y, score, thr=0.0):
    return M.time_matched(t, y, (score > thr).astype(int), score)['tm_bacc']


def meta_fit(F, y):
    sc = StandardScaler().fit(F)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000).fit(sc.transform(F), y)
    return lambda G: lr.decision_function(sc.transform(G))


def candidate_scores(inner_df, outer_df, chain_inner, chain_outer):
    """Per candidate: (score for outer rows, leave-one-inner-person-out CV TM bacc on the inner OOF predictions)."""
    cands = {}
    persons = inner_df.inner.unique()
    y_in, t_in = inner_df.low.values.astype(int), inner_df.t_start.values
    y_out = outer_df.low.values.astype(int)
    feats = {}
    for a in ALPHAS:
        feats[f'a{int(a * 100)}'] = (inner_df[f'logit_a{int(a * 100)}'].values, outer_df[f'logit_a{int(a * 100)}'].values)
    for tag, (fi, fo) in feats.items():
        name = 'cnn' if tag == 'a0' else f'cnn_bn{tag[1:]}'
        cands[name] = dict(inner=fi, outer=fo, thr=0.0, meta=False, F_in=fi[:, None], F_out=fo[:, None])
        sname = 'stack' if tag == 'a0' else f'stack_bn{tag[1:]}'
        cands[sname] = dict(F_in=np.c_[fi, chain_inner], F_out=np.c_[fo, chain_outer], meta=True)
    cands['chain'] = dict(inner=chain_inner, outer=chain_outer, thr=0.0, meta=False, F_in=chain_inner[:, None], F_out=chain_outer[:, None])
    res = {}
    for name, c in cands.items():
        if c['meta']:
            cv = np.zeros(len(y_in))
            for p in persons:
                te = (inner_df.inner.values == p)
                f = meta_fit(c['F_in'][~te], y_in[~te])
                cv[te] = f(c['F_in'][te])
            f = meta_fit(c['F_in'], y_in)
            res[name] = dict(out=f(c['F_out']), cv=tm_bacc(t_in, y_in, cv), cv_scores=cv)
        else:
            res[name] = dict(out=c['outer'], cv=tm_bacc(t_in, y_in, c['inner']), cv_scores=c['inner'])
    return res


def build_report(runs, hoffman_dir=None, B=1000, seed=0):
    from .windows import WindowSet
    ws = WindowSet('hoffman', hoffman_dir or os.path.join(DATA_DIR, 'hoffman'))
    X19 = np.load(os.path.join(ws.dir, 'chain19.npy'))
    meta = ws.meta
    files = sorted(glob.glob(os.path.join(runs, '*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    persons = sorted(df.outer.unique())
    ev = meta['eval'].values & meta.has_low.values
    ylab = meta.low.values
    rec_out, sel_log = [], []
    for o in persons:
        outer_df = df[(df.kind == 'outer') & (df.outer == o)].reset_index(drop=True)
        inner_df = df[(df.kind == 'inner') & (df.outer == o)].reset_index(drop=True)
        if outer_df.empty or inner_df.inner.nunique() < 2:
            continue
        subj = meta.subject.values
        tr_people = [p for p in persons if p != o]
        idx_tr = np.nonzero(ev & np.isin(subj, tr_people))[0]
        chain_out = chain_logit(X19[idx_tr], ylab[idx_tr].astype(int), X19[outer_df.local.values])
        chain_in = np.zeros(len(inner_df))
        for q in inner_df.inner.unique():
            tr_q = np.nonzero(ev & np.isin(subj, [p for p in tr_people if p != q]))[0]
            m = (inner_df.inner.values == q)
            chain_in[m] = chain_logit(X19[tr_q], ylab[tr_q].astype(int), X19[inner_df.local.values[m]])
        res = candidate_scores(inner_df, outer_df, chain_in, chain_out)
        best = max(res, key=lambda k: res[k]['cv'])
        for k, v in res.items():
            sel_log.append(dict(outer=o, candidate=k, inner_cv_tm_bacc=v['cv'], chosen=(k == best)))
            rec_out.append(pd.DataFrame(dict(outer=o, candidate=k, subject=outer_df.subject, rec=outer_df.rec, t_start=outer_df.t_start, low=outer_df.low, score=v['out'], chosen=(k == best))))
    allr = pd.concat(rec_out, ignore_index=True)
    sel = pd.DataFrame(sel_log)
    rows = []
    for k, g in list(allr.groupby('candidate')) + [('SELECTED_BY_INNER_CV', allr[allr.chosen])]:
        g = g.reset_index(drop=True)
        y, t, s, subj = g.low.values.astype(int), g.t_start.values, g.score.values, g.subject.values
        pooled = M.time_matched(t, y, (s > 0).astype(int), s)
        bt = M.person_bootstrap(subj, M.tm_stat(t, y, s, 0.0, 'tm_bacc'), B=B, seed=seed)
        per = [M.time_matched(t[subj == p], y[subj == p], (s[subj == p] > 0).astype(int), s[subj == p])['tm_bacc'] for p in np.unique(subj)]
        sm = M.summarize_folds(per)
        cl = M.classification_metrics(y, s, thr=0.0)
        rows.append(dict(candidate=k, tm_bacc=pooled['tm_bacc'], lo=bt['lo'], hi=bt['hi'], tm_auroc=pooled['tm_auroc'], per_person_mean=sm['mean'], per_person_sd=sm['sd'],
                         bacc=cl['bacc'], auroc=cl['auroc'], n_people=len(np.unique(subj))))
    return pd.DataFrame(rows), sel, allr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run'); r.add_argument('--out', required=True); r.add_argument('--init', required=True); r.add_argument('--recipe', default=None)
    r.add_argument('--seed', type=int, default=0); r.add_argument('--shard', default=None); r.add_argument('--threads', type=int, default=0); r.add_argument('--data', default=None)
    p = sp.add_parser('report'); p.add_argument('--runs', required=True); p.add_argument('--out', required=True); p.add_argument('--B', type=int, default=1000)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        if a.threads:
            torch.set_num_threads(a.threads)
        over = json.load(open(a.recipe)) if a.recipe else {}
        hp = variant_hp('full', **{**over, 'init': a.init})
        pool = Pool(root=a.data, seed=0)
        js = jobs(pool)
        i, n = (int(x) for x in a.shard.split('/')) if a.shard else (0, 1)
        for k, j in enumerate(js):
            if k % n == i:
                run_job(pool, j, hp, a.seed, 'cpu', a.out)
        return
    tab, sel, allr = build_report(a.runs, B=a.B)
    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(os.path.join(a.out, 'hybrid_p1.csv'), index=False)
    sel.to_csv(os.path.join(a.out, 'hybrid_p1_selection.csv'), index=False)
    allr.to_csv(os.path.join(a.out, 'scores', 'hybrid_p1_hoffman.csv'), index=False) if os.path.isdir(os.path.join(a.out, 'scores')) else None
    print(tab.round(3).to_string())


if __name__ == '__main__':
    main()
