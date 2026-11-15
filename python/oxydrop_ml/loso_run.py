"""Leave-one-subject-out (and nested-selection) runs of the OxyNet recipe, plus the aggregation into the pre-registered P1/P2/P3 numbers.

    # P1: Hoffman, 6 folds (both hands of a person held out together)
    python -m oxydrop_ml.loso_run run --task p1 --variant full --seed 0 --init ckpt/stageB_full.pt --out runs/full_s0
    # P2/P3: MTHS (62) + BUT (38) leave-one-subject-out; --shard i/n splits the folds over processes
    python -m oxydrop_ml.loso_run run --task p23 --variant full --seed 0 --init ckpt/stageB_full.pt --out runs/full_s0 --shard 0/4
    # cheaper variant, labelled "not LOSO": 10 subject-grouped folds over the 100 MTHS+BUT people
    python -m oxydrop_ml.loso_run run --task p23k --variant no_aug --out runs/no_aug_s0
    # nested selection of the recipe (inner leave-one-Hoffman-person-out inside each outer P1 fold)
    python -m oxydrop_ml.loso_run nested --out runs/nested --shard 0/4
    python -m oxydrop_ml.loso_run summarize runs/full_s0 [runs/full_s1 ...]

Per fold the run writes `fold_<id>.csv` (per-window predictions of the held-out people) and `fold_<id>.json` (metrics, train-vs-test inputs, steps, seconds). Runs are
resumable: a fold with an existing JSON is skipped. Nothing here reads a test subject before the fold's model is trained: `train.fold_rows` removes every window of the
test people (and an inner-validation set drawn from the other people) from the training rows, and `fit` only sees those rows.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import metrics as M
from . import splits as S
from .train import HP, Pool, fit, fold_rows, predict, scores_frame, train_scores, variant_hp, save_ckpt


# ---------------------------------------------------------------------------------------------------------------- folds
def make_folds(pool: Pool, task: str, seed=0):
    """List of dict(id, test_subjects, protocol). p1: Hoffman LOSO. p23: MTHS and BUT LOSO. p23k: 10 grouped folds over the MTHS+BUT people (not LOSO)."""
    if task == 'p1':
        return [dict(id=s, test=[s], protocol='LOSO') for s in pool.subjects_of('hoffman')]
    subs = pool.subjects_of('mths') + pool.subjects_of('butppg')
    if task == 'p23':
        return [dict(id=s, test=[s], protocol='LOSO') for s in subs]
    if task == 'p23k':
        order = np.random.default_rng(1000 + seed).permutation(subs)
        return [dict(id=f'g{k}', test=sorted(order[k::10].tolist()), protocol='grouped10 (not LOSO: 10 subject-grouped folds to save compute)') for k in range(10)]
    raise ValueError(task)


def run_fold(pool: Pool, fold, hp: HP, seed, device, out_dir, log):
    fid = fold['id']
    jp, cp = os.path.join(out_dir, f'fold_{fid}.json'), os.path.join(out_dir, f'fold_{fid}.csv')
    if os.path.exists(jp):
        return None
    tr, va, vs = fold_rows(pool, fold['test'], seed)
    # hard guarantees (also covered by the unit test): no test subject in training, validation or scaling rows
    S.assert_no_leak(pool.subject, tr, np.nonzero(np.isin(pool.subject, fold['test']))[0])
    assert not (set(pool.subject[tr]) & set(fold['test'])) and not (set(pool.subject[va]) & set(fold['test']))
    fr = fit(pool, tr, va, hp, seed, device, log=log)
    rows = np.nonzero(np.isin(pool.subject, fold['test']))[0]
    o = predict(fr.net, pool, rows, fr.norm, hp, device)
    df = scores_frame(pool, rows, o, dict(fold=fid, variant=hp_name(hp), seed=seed))
    df = df[df.ds != 'openox']
    gap = train_scores(pool, fr, device, seed=seed)
    meta = dict(fold=fid, test=fold['test'], protocol=fold['protocol'], n_train=int(len(tr)), n_val=int(len(va)), val_subjects=vs, best_step=fr.best_step, seconds=fr.seconds,
                train=gap, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(cp, index=False)
    json.dump(meta, open(jp, 'w'), indent=1, default=float)
    return meta


def hp_name(hp):
    return hashlib.md5(json.dumps(hp.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------------------------------------------- aggregation
def load_run(dirs):
    """Concatenate the per-fold CSV/JSON of one or more run directories (one variant, several seeds or shards)."""
    S_, F_ = [], []
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        for jp in sorted(glob.glob(os.path.join(d, 'fold_*.json'))):
            j = json.load(open(jp)); j['dir'] = d
            F_.append(j)
            c = pd.read_csv(jp[:-5] + '.csv'); c['run_dir'] = d
            S_.append(c)
    return (pd.concat(S_, ignore_index=True) if S_ else pd.DataFrame()), F_


def p1_stats(df, B=1000, seed=0):
    d = df[(df.ds == 'hoffman') & df['eval'] & df.low.notna()].reset_index(drop=True)
    if not len(d):
        return {}
    y, t, s, subj = d.low.values.astype(int), d.t_start.values, d.p_low.values, d.subject.values
    pooled = M.time_matched(t, y, (s > 0.5).astype(int), s)
    boot = M.person_bootstrap(subj, M.tm_stat(t, y, s, 0.5, 'tm_bacc'), B=B, seed=seed)
    boot_auc = M.person_bootstrap(subj, M.tm_stat(t, y, s, 0.5, 'tm_auroc'), B=B, seed=seed)
    per = {p: M.time_matched(t[subj == p], y[subj == p], (s[subj == p] > 0.5).astype(int), s[subj == p])['tm_bacc'] for p in np.unique(subj)}
    return dict(p1=pooled['tm_bacc'], p1_lo=boot['lo'], p1_hi=boot['hi'], p1_auroc=pooled['tm_auroc'], p1_auroc_lo=boot_auc['lo'], p1_auroc_hi=boot_auc['hi'],
                p1_fold=per, n=len(d), n_people=len(per))


def p2_stats(df, B=1000, seed=0):
    d = df[df.ds.isin(['mths', 'butppg']) & df.hr_ref.notna()].reset_index(drop=True)
    if not len(d):
        return {}
    ref, pred, subj = d.hr_ref.values, d.hr_pred.values, d.subject.values
    stat = lambda i: M.hr_metrics(pred[i], ref[i])['within5_full']
    boot = M.person_bootstrap(subj, stat, B=B, seed=seed)
    h = M.hr_metrics(pred, ref)
    per = {p: M.hr_metrics(pred[subj == p], ref[subj == p])['within5_full'] for p in np.unique(subj)}
    out = dict(p2=h['within5_full'], p2_lo=boot['lo'], p2_hi=boot['hi'], p2_mae=h['mae'], p2_doubling=h['doubling'], p2_fold=per, n2=len(d), n_people2=len(per))
    for ds, key in (('mths', 'p2_mths'), ('butppg', 'p2_but')):
        q = d[d.ds == ds]
        if len(q):
            out[key] = M.hr_metrics(q.hr_pred.values, q.hr_ref.values)['within5_full']
    return out


def p3_stats(df, B=1000, seed=0):
    d = df[(df.ds == 'mths') & df['eval'] & df.spo2_ref.notna()].reset_index(drop=True)
    if not len(d):
        return {}
    ref, pred, subj = d.spo2_ref.values, d.spo2_pred.values, d.subject.values
    stat = lambda i: float(np.abs(pred[i] - ref[i]).mean())
    boot = M.person_bootstrap(subj, stat, B=B, seed=seed)
    per = {p: float(np.abs(pred[subj == p] - ref[subj == p]).mean()) for p in np.unique(subj)}
    return dict(p3=stat(np.arange(len(d))), p3_lo=boot['lo'], p3_hi=boot['hi'], p3_fold=per, n3=len(d), n_people3=len(per))


def gaps(folds, st):
    """Train-vs-test gap per pre-registered metric from the fold JSONs (train score of each fold's model on its own training rows vs its held-out score)."""
    out = {}
    for key, fk, hib in (('p1', 'p1_fold', True), ('p2', 'p2_fold', True), ('p3', 'p3_fold', False)):
        per = st.get(fk)
        if not per:
            continue
        tr, te = [], []
        for j in folds:
            if key not in j['train']:
                continue
            ids = [p for p in j['test'] if p in per and np.isfinite(per[p])]
            if not ids:
                continue
            te.append(float(np.mean([per[p] for p in ids]))); tr.append(j['train'][key])
        if tr:
            g = M.train_vs_test_gap(tr, te, higher_is_better=hib)
            out[key + '_train'] = g['train_mean']; out[key + '_test_foldmean'] = g['test_mean']; out[key + '_gap'] = g['gap']
            out[key + '_gap_lo'], out[key + '_gap_hi'] = g['gap_lo'], g['gap_hi']
    return out


def summarize(dirs, B=1000, seed=0):
    df, folds = load_run(dirs)
    if df.empty:
        return {}
    st = {}
    st.update(p1_stats(df, B, seed)); st.update(p2_stats(df, B, seed)); st.update(p3_stats(df, B, seed))
    for key in ('p1', 'p2', 'p3'):
        per = st.get(key + '_fold')
        if per:
            sm = M.summarize_folds(list(per.values()), seed=seed)
            st[key + '_fold_mean'], st[key + '_fold_sd'] = sm['mean'], sm['sd']
            st[key + '_fold_lo'], st[key + '_fold_hi'] = sm['lo'], sm['hi']
    st.update(gaps(folds, st))
    st['n_folds'] = len(folds)
    st['protocols'] = sorted({j['protocol'] for j in folds})
    st['seconds'] = float(sum(j['seconds'] for j in folds))
    st['seeds'] = sorted({int(j['seed']) for j in folds})
    return st


# ---------------------------------------------------------------------------------------------------------------- nested selection
CANDIDATES = {
    'A_base': dict(),                 # w_low 1.5, plain class-weighted BCE
    'B_focal': dict(focal=2.0),       # focal BCE (gamma 2)
    'C_lowhi': dict(w_low=3.0),       # low-oxygen loss weight 3
}
N_INNER = 2                           # inner held-out Hoffman people per outer fold (random 2 of the 5 remaining; compute budget)


def nested_selection(pool: Pool, base_over: dict, variant: str, out_dir, seed, shard, device, log):
    """For every outer P1 fold (a Hoffman person), an inner leave-one-Hoffman-person-out CV over N_INNER of the 5 remaining people scores each candidate recipe
    on the inner held-out person (TM balanced accuracy) and on one MTHS and one BUT validation group (HR within 5, SpO2 MAE) drawn from the training
    people only. The outer test person is never read. Selection score = mean of TM bacc, HR within-5 and (1 - MAE / 5), each on inner folds only.
    Writes nested_<outer>_<cand>_<inner>.json lines; `select_recipe` reads them."""
    os.makedirs(out_dir, exist_ok=True)
    hof = pool.subjects_of('hoffman')
    jobs = []
    for outer in hof:
        inner = [s for s in hof if s != outer]
        inner = sorted(np.random.default_rng(seed * 100 + hof.index(outer)).choice(inner, N_INNER, replace=False).tolist())
        for cand in CANDIDATES:
            for inn in inner:
                jobs.append((outer, cand, inn))
    for k, (outer, cand, inn) in enumerate(jobs):
        if shard and k % shard[1] != shard[0]:
            continue
        jp = os.path.join(out_dir, f'nested_{outer}_{cand}_{inn}.json')
        if os.path.exists(jp):
            continue
        hp = variant_hp(variant, **{**base_over, **CANDIDATES[cand]})
        test_inner = [outer, inn]                                 # outer person AND the inner validation person are both unseen in this training
        rng = np.random.default_rng(seed)
        others_m = [s for s in pool.subjects_of('mths')]
        others_b = [s for s in pool.subjects_of('butppg')]
        gm = rng.choice(others_m, 8, replace=False).tolist(); gb = rng.choice(others_b, 6, replace=False).tolist()
        excl = [outer, inn] + gm + gb
        tr, va, vs = fold_rows(pool, excl, seed)
        S.assert_no_leak(pool.subject, tr, np.nonzero(np.isin(pool.subject, excl))[0])
        t0 = time.time()
        fr = fit(pool, tr, va, hp, seed, device, log=None)
        r = np.nonzero(np.isin(pool.subject, [inn]) & pool.eval & pool.has_low)[0]
        o = predict(fr.net, pool, r, fr.norm, hp, device)
        tm = M.time_matched(pool.t[r], pool.low[r], (o['p_low'] > 0.5).astype(int), o['p_low'])['tm_bacc']
        rm = np.nonzero(np.isin(pool.subject, gm + gb) & pool.has_hr)[0]
        om = predict(fr.net, pool, rm, fr.norm, hp, device)
        w5 = M.hr_metrics(om['hr'], pool.hr[rm])['within5_full']
        rs = np.nonzero(np.isin(pool.subject, gm) & pool.has_spo2 & pool.eval)[0]
        mae = M.spo2_metrics(predict(fr.net, pool, rs, fr.norm, hp, device)['spo2'], pool.spo2[rs])['mae']
        rec = dict(outer=outer, cand=cand, inner=inn, tm_bacc=tm, hr_within5=w5, spo2_mae=mae, best_step=fr.best_step, seconds=time.time() - t0)
        json.dump(rec, open(jp, 'w'), default=float)
        log(f'nested {outer} {cand} {inn}: tm {tm:.3f} hr5 {w5:.3f} mae {mae:.2f} ({rec["seconds"]:.0f}s)')


def select_recipe(out_dir):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, 'nested_*.json'))]
    if not rows:
        return None, pd.DataFrame()
    d = pd.DataFrame(rows)
    d['score'] = (d.tm_bacc.fillna(0.5) + d.hr_within5 + (1 - d.spo2_mae.fillna(5) / 5.0).clip(0, 1)) / 3.0
    per_outer = d.groupby(['outer', 'cand']).agg(score=('score', 'mean'), tm=('tm_bacc', 'mean'), hr5=('hr_within5', 'mean'), mae=('spo2_mae', 'mean'), n_inner=('inner', 'count')).reset_index()
    chosen = per_outer.loc[per_outer.groupby('outer').score.idxmax()][['outer', 'cand', 'score']].rename(columns={'cand': 'chosen', 'score': 'chosen_inner_score'})
    glob_ = d.groupby('cand').score.mean().sort_values(ascending=False)
    return glob_.index[0], per_outer.merge(chosen, on='outer')


# ---------------------------------------------------------------------------------------------------------------- CLI
def parse_shard(s):
    if not s:
        return None
    i, n = s.split('/')
    return int(i), int(n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--task', choices=['p1', 'p23', 'p23k'], required=True); r.add_argument('--variant', default='full')
    r.add_argument('--seed', type=int, default=0); r.add_argument('--init', default=None); r.add_argument('--out', required=True); r.add_argument('--shard', default=None)
    r.add_argument('--hp', default=None, help='JSON HP overrides'); r.add_argument('--data', default=None); r.add_argument('--device', default='cpu'); r.add_argument('--threads', type=int, default=0)
    r.add_argument('--limit', type=int, default=0, help='run only the first N folds of this shard (smoke tests)')
    n = sub.add_parser('nested'); n.add_argument('--out', required=True); n.add_argument('--variant', default='full'); n.add_argument('--seed', type=int, default=0)
    n.add_argument('--init', default=None); n.add_argument('--shard', default=None); n.add_argument('--hp', default=None); n.add_argument('--data', default=None); n.add_argument('--device', default='cpu')
    n.add_argument('--threads', type=int, default=0)
    s = sub.add_parser('summarize'); s.add_argument('dirs', nargs='+'); s.add_argument('--json', default=None)
    sel = sub.add_parser('select'); sel.add_argument('dir')
    a = ap.parse_args(argv)
    if a.cmd == 'summarize':
        st = summarize(a.dirs)
        print(json.dumps({k: v for k, v in st.items() if not k.endswith('_fold')}, indent=1, default=float))
        if a.json:
            json.dump(st, open(a.json, 'w'), default=float)
        return
    if a.cmd == 'select':
        best, tab = select_recipe(a.dir)
        print('globally best candidate (mean inner score):', best); print(tab.to_string())
        return
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    over = json.loads(a.hp) if a.hp else {}
    if a.init:
        over['init'] = a.init
    hp = variant_hp(a.variant, **over)
    pool = Pool(root=a.data, seed=a.seed)
    log = lambda m: print(m, flush=True)
    sh = parse_shard(a.shard)
    if a.cmd == 'nested':
        nested_selection(pool, over, a.variant, a.out, a.seed, sh, a.device, log)
        return
    folds = make_folds(pool, a.task, a.seed)
    if sh:
        folds = [f for k, f in enumerate(folds) if k % sh[1] == sh[0]]
    if a.limit:
        folds = folds[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(variant=a.variant, hp=hp.to_dict(), seed=a.seed, task=a.task), open(os.path.join(a.out, f'config_{a.task}_{sh[0] if sh else 0}.json'), 'w'), indent=1, default=str)
    for k, f in enumerate(folds):
        t0 = time.time()
        m = run_fold(pool, f, hp, a.seed, a.device, a.out, None)
        log(f'[{k + 1}/{len(folds)}] fold {f["id"]} {"skipped" if m is None else f"done {time.time() - t0:.0f}s best_step {m["best_step"]} train {m["train"]}"}')


if __name__ == '__main__':
    main()
