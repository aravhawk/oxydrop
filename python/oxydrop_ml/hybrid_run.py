"""FEAT-008 fits: the CNN families P (plain), R (residual on the chain logit) and D (chain distillation) for the nested P1 protocol, and the report.

    python -m oxydrop_ml.hybrid_run run --families P R D --seeds 0 1 2 --out RUNS/hyb --init ckpt/stageB_full.pt --recipe recipe.json [--shard i/n] [--threads 1]
    python -m oxydrop_ml.hybrid_run final --families P R --seeds 0 1 2 --out RUNS/hyb_final --init ckpt/stageB_full.pt --recipe recipe.json     # all-people fits for the app
    python -m oxydrop_ml.hybrid_run report --runs RUNS/hyb --out results/ml           (see hybrid.py; heavy part is the bootstrap)

Jobs per (family, seed): 6 outer fits (one Hoffman person o held out; scores o, plus the 5 training people for the train-vs-test gap) and 30 inner fits
(o and q held out; scores q out of fold = the inner OOF prediction, and o as a fold-ensemble member). Several workers can share one output directory:
a job is claimed with an exclusive lock file, finished jobs are skipped. Each job writes `<out>/<family>_s<seed>/<kind>_<o>[_<q>].csv`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import distill as DS
from . import splits as S
from .train import HP, Pool, fit, fold_rows, variant_hp


def jobs_for(hof):
    """(kind, outer, inner) in a fixed order: for every outer person its outer fit, then the 5 inner fits."""
    out = []
    for o in hof:
        out.append(('outer', o, None))
        for q in hof:
            if q != o:
                out.append(('inner', o, q))
    return out


def job_name(job):
    kind, o, q = job
    return f'{kind}_{o}' + (f'_{q}' if q else '')


def family_hp(hp: HP, family):
    """Family L (frozen backbone, Amendment 4) trains a 73-weight head with its own learning rate; every other family uses the recipe."""
    return dataclasses.replace(hp, lr=DS.LR_L) if family == 'L' else hp


def rows_of(pool: Pool, people):
    return np.nonzero(np.isin(pool.subject, people) & (pool.ds == 'hoffman') & (pool.eval | pool.in_base))[0]


def frame(pool: Pool, rows, o, q, target_kind, seed, family, out):
    return pd.DataFrame(dict(outer=o, inner=q or '', target=target_kind, family=family, seed=seed, subject=pool.subject[rows], rec=pool.meta.rec.values[rows],
                             local=pool.local[rows], t_start=pool.t[rows], low=pool.low[rows], eval=pool.eval[rows], in_base=pool.in_base[rows],
                             hr_ref=pool.hr[rows], logit=out['logit'], hr=out['hr'], hr_argmax=out['hr_argmax'], p_sqi=out['p_sqi']))


def run_job(pool: Pool, X19, job, family, hp: HP, seed, device, out_dir, log=print):
    kind, o, q = job
    d = os.path.join(out_dir, f'{family}_s{seed}')
    os.makedirs(d, exist_ok=True)
    cp = os.path.join(d, job_name(job) + '.csv')
    lock = cp + '.lock'
    if os.path.exists(cp):
        return False
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    t0 = time.time()
    excl = [o] + ([q] if q else [])
    hof = pool.subjects_of('hoffman')
    tprime = [p for p in hof if p not in excl]
    tr, va, vs = fold_rows(pool, excl, seed)
    S.assert_no_leak(pool.subject, tr, np.nonzero(np.isin(pool.subject, excl))[0])
    prior = DS.prior_for_pool(pool, X19, tprime, excl) if family != 'P' else None
    hp = family_hp(hp, family)
    with DS.install(family, prior):
        fr = fit(pool, tr, va, hp, seed, device)
    parts = []
    target = q if q else o
    r = rows_of(pool, [target])
    parts.append(frame(pool, r, o, q, 'inner' if q else 'outer', seed, family, DS.predict_raw(fr.net, pool, r, fr.norm, hp, device)))
    if q:                                              # the inner fit excludes o too: it is a valid ensemble member for o
        r = rows_of(pool, [o])
        parts.append(frame(pool, r, o, q, 'member', seed, family, DS.predict_raw(fr.net, pool, r, fr.norm, hp, device)))
    else:                                              # in-sample rows of the training people for the train-vs-test gap
        r = rows_of(pool, tprime)
        parts.append(frame(pool, r, o, q, 'train', seed, family, DS.predict_raw(fr.net, pool, r, fr.norm, hp, device)))
    df = pd.concat(parts, ignore_index=True)
    df['best_step'] = fr.best_step
    df.to_csv(cp + '.tmp', index=False)
    os.replace(cp + '.tmp', cp)
    os.remove(lock)
    log(f'{family} s{seed} {job_name(job)} {time.time() - t0:.0f}s best_step {fr.best_step}')
    return True


def final_fit(pool: Pool, X19, family, hp: HP, seed, device, out_dir, log=print):
    """Deployable fit of one family on every Hoffman person (plus the other training sets): no outer test person. Same recipe as the nested fits.
    The checkpoint holds the phone input scale folded into the weights (consumes raw features), like train.py --final."""
    from .model import count_params
    from .train import fold_norm, save_ckpt
    cp = os.path.join(out_dir, f'{family}_s{seed}.pt')
    if os.path.exists(cp):
        return False
    t0 = time.time()
    hof = pool.subjects_of('hoffman')
    tr, va, vs = fold_rows(pool, [], seed)
    prior = DS.prior_for_pool(pool, X19, hof, []) if family != 'P' else None
    hp = family_hp(hp, family)
    with DS.install(family, prior):
        fr = fit(pool, tr, va, hp, seed, device)
    folded = fold_norm(fr.net, fr.norm, 0)

    class _F:
        pass
    ff = _F(); ff.net, ff.norm, ff.hp, ff.best_step = folded, fr.norm, fr.hp, fr.best_step
    os.makedirs(out_dir, exist_ok=True)
    sha = save_ckpt(cp, ff, {'folded_norm_domain': 0, 'val_subjects': vs, 'params_shipped': count_params(folded, include_aux=False), 'seed': seed, 'family': family})
    log(f'final {family} s{seed} {time.time() - t0:.0f}s best_step {fr.best_step} sha256 {sha}')
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--out', required=True); r.add_argument('--init', required=True); r.add_argument('--recipe', default=None)
    r.add_argument('--families', nargs='+', default=['P', 'R', 'D']); r.add_argument('--seeds', nargs='+', type=int, default=[0])
    r.add_argument('--shard', default=None); r.add_argument('--threads', type=int, default=0); r.add_argument('--data', default=None)
    r.add_argument('--steps', type=int, default=None, help='override the fine-tuning steps (smoke tests only)')
    f = sp.add_parser('final')
    f.add_argument('--out', required=True); f.add_argument('--init', required=True); f.add_argument('--recipe', default=None)
    f.add_argument('--families', nargs='+', default=['P', 'R']); f.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); f.add_argument('--threads', type=int, default=0)
    f.add_argument('--data', default=None)
    p = sp.add_parser('report')
    p.add_argument('--runs', required=True); p.add_argument('--out', required=True); p.add_argument('--B', type=int, default=2000)
    p.add_argument('--seeds', nargs='+', type=int, default=None); p.add_argument('--tag', default=''); p.add_argument('--families', nargs='+', default=['P', 'R', 'D'])
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        if a.threads:
            torch.set_num_threads(a.threads)
        over = json.load(open(a.recipe)) if a.recipe else {}
        if a.steps:
            over['steps'] = a.steps
        hp = variant_hp('full', **{**over, 'init': a.init})
        pool = Pool(root=a.data, seed=0)
        X19 = np.load(os.path.join(pool.ws['hoffman'].dir, 'chain19.npy'))
        js = jobs_for(pool.subjects_of('hoffman'))
        i, n = (int(x) for x in a.shard.split('/')) if a.shard else (0, 1)
        allj = [(f, s, j) for s in a.seeds for f in a.families for j in js]
        for k, (f, s, j) in enumerate(allj):
            if k % n == i:
                run_job(pool, X19, j, f, hp, s, 'cpu', a.out)
        return
    if a.cmd == 'final':
        if a.threads:
            torch.set_num_threads(a.threads)
        hp = variant_hp('full', **{**(json.load(open(a.recipe)) if a.recipe else {}), 'init': a.init})
        pool = Pool(root=a.data, seed=0)
        X19 = np.load(os.path.join(pool.ws['hoffman'].dir, 'chain19.npy'))
        for s in a.seeds:
            for fam in a.families:
                final_fit(pool, X19, fam, hp, s, 'cpu', a.out)
        return
    from . import hybrid as H
    H.report_cli(a)


if __name__ == '__main__':
    main()
