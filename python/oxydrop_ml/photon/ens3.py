"""Round-3 candidate R3-ENSEMBLE, `ens3`: a nested-evaluated stacking ensemble of the top trained, null-passing members.

    # remote (Kaggle CPU kernel; trains members' inner folds, needs torch):
    cd python && python -m oxydrop_ml.photon.ens3 --inner real        # inner-OOF predictions of the members (writes results/photon/ens3_inner/*.npz)
    cd python && python -m oxydrop_ml.photon.ens3 --inner null 0      # same for the null-label pipelines of draw 0 (also 1, 2)
    # local (numpy + sklearn only, light):
    cd python && python -m oxydrop_ml.photon.ens3                     # combiner, score files ens3__L0..L5, ens3__X1, ens3_null / _d1_null / _d2_null

MEMBER POOL (trained round-2 models whose null controls pass and whose inner folds can be recomputed from data on disk):
    distill       PhotonDistill KD student (distill_student.py), 3 seeds; outer = the shipped 3-seed ensemble score files distill__*.csv
    oxyseq        OxySeq GRU (oxyseq.py); outer = oxyseq__*.csv (3-seed ensemble); inner OOF single seed 0 (as its own nested selection)
    distill_nokd  label-only student of the same family, seed 0; outer = distill_nokd__*.csv
    Excluded: photonnet and oxyssl (null controls fail: 0.618 / 0.643), distill_v1 (superseded variant of distill), round-3 models (no score files or
    inner folds on disk before the 20:15 CT L0 deadline).

NESTING (the held-out person never enters any step that produces its own prediction):
    For each outer held-out person p, every member's INNER out-of-fold predictions are recomputed on the 5 training people: for each inner person ip,
    the member is retrained on the other 4 people (its own fold-p hyperparameters: distill uses the lam/wd/epochs that its own nested selection chose
    for outer fold p, oxyseq re-runs its own inner selection of steps) and predicts ip. Outer-OOF scores of other people are never used to fit the
    combiner. Member choice (top K = 3 of the pool by inner-OOF TM bal. acc.), combiner regularisation C and the abstain threshold are chosen on those
    inner-OOF predictions only (combiner C and tau by leave-one-inner-person-out of the combiner itself).
COMBINER: class-balanced logistic regression on standardised [logit(score_m), logit(conf_m)] of the chosen members (conf = distill's abstain-head
    probability, oxyseq's quality probability). score = P(hypoxic); pred = score > 0.5; conf = |score - 0.5| (the combiner's own uncertainty);
    answer = conf >= tau, tau = flashgap.select_tau on the combiner's leave-one-inner-person-out predictions.
STRESS: the combiner is fitted once per outer fold on CLEAN inner-OOF predictions and applied to the members' outer scores at L0..L5 and X1.
NULLS: ens3_null, ens3_d1_null, ens3_d2_null: every member's whole pipeline with training labels permuted globally (each member's own permutation
    scheme, draw d), inner and outer, and the combiner fitted on the permuted inner labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
INNER = RES / 'ens3_inner'
LEVELS = [0, 1, 2, 3, 4, 5, 'X1']
POOL = ('distill', 'oxyseq', 'distill_nokd')
K = 3
GRID_C = (0.01, 0.1, 1.0)


def tag(lv):
    return lv if isinstance(lv, str) else f'L{lv}'


# ================================================================================================ remote: member inner folds
def _save(name, **arr):
    INNER.mkdir(parents=True, exist_ok=True)
    arr = {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else np.asarray(v)) for k, v in arr.items()}
    np.savez_compressed(INNER / f'{name}.npz', **arr)
    print('saved', name, {k: len(v) for k, v in arr.items() if hasattr(v, '__len__')}, flush=True)


def inner_distill(name, seeds, kd, pickles, null=None, log=print):
    """Inner-OOF clean predictions of the distill student family at each outer fold's own selected config (seed mean)."""
    import pickle
    from . import distill_student as DS
    from . import flashgap as F
    from .. import splits as SP
    DS._pack()
    mt = F.meta()
    y, t, sid, s2 = mt.y.values.astype(int), mt.t.values.astype(float), mt.sid.values.astype(str), mt.y_spo2.values.astype(float)
    conds_all = list(dict.fromkeys(sum([DS.train_conds(s) for s in seeds], [])))
    Xd, _ = DS.features(conds_all)
    acc = {}
    outer = {}
    for seed, pk in zip(seeds, pickles):
        conds = DS.train_conds(seed)
        folds = None
        if pk is not None and (DS.CACHE_DIR / 'models' / f'{pk}.pkl').exists():
            with open(DS.CACHE_DIR / 'models' / f'{pk}.pkl', 'rb') as fh:
                folds = pickle.load(fh)
        for fi, f in enumerate(SP.loso(sid)):
            t0 = time.time()
            tr = f.train_idx
            y_tr, s_tr = y[tr].copy(), s2[tr].copy()
            if null is not None:
                p = DS.null_perm(len(tr), f.test_group, null)
                y_tr, s_tr = y_tr[p], s_tr[p]
            if folds is not None:
                assert str(folds[fi]['sid']) == str(f.test_group)
                info = folds[fi]['info']
            else:                                            # no stored outer model (nokd nulls): run the member's own nested fit
                model, tau, info = DS.fit_fold(Xd, conds, tr, y_tr, s_tr, t[tr], sid[tr], dict(seed=seed, kd=kd, null=null))
                pr, _, ab = model.predict(Xd['clean'][f.test_idx])
                outer[f.test_group] = (f.test_idx, pr, ab)
            lam, wd, ep = info['lam'], info['wd'], info['epochs']
            g_tr = sid[tr]
            for ip in np.unique(g_tr):
                itr = np.nonzero(g_tr != ip)[0]; iva = np.nonzero(g_tr == ip)[0]
                Xs, ys, w, q = DS.soft_targets(Xd, conds, tr[itr], y_tr[itr])
                st = DS.train_student(Xs, ys, w, q, np.tile(s_tr[itr], len(conds)), lam, ep, seed, wd=wd)[ep]
                pr, _, ab = st.predict(Xd['clean'][tr[iva]])
                key = (str(f.test_group), str(ip))
                a = acc.setdefault(key, dict(idx=tr[iva], y=y_tr[iva], p=[], a=[]))
                a['p'].append(pr); a['a'].append(ab)
            log(f'{name} seed {seed} outer {f.test_group}: lam {lam} wd {wd:g} ep {ep} ({time.time() - t0:.0f} s)', flush=True)
    o, r, tt, yy, pp, aa = [], [], [], [], [], []
    for (op, ip), a in acc.items():
        o += [op] * len(a['idx']); r.append(mt.rec.values[a['idx']].astype(str)); tt.append(t[a['idx']]); yy.append(a['y'])
        pp.append(np.mean(a['p'], 0)); aa.append(np.mean(a['a'], 0))
    arr = dict(outer=np.array(o), rec=np.concatenate(r), t=np.concatenate(tt), y=np.concatenate(yy), score=np.concatenate(pp), conf=np.concatenate(aa))
    if outer:
        oi = np.concatenate([v[0] for v in outer.values()])
        arr.update(o_rec=mt.rec.values[oi].astype(str), o_t=t[oi], o_score=np.concatenate([v[1] for v in outer.values()]),
                   o_conf=np.concatenate([v[2] for v in outer.values()]))
    _save(name, **arr)


def inner_oxyseq(name, null=None, outer_fit=False, log=print):
    """Inner-OOF predictions of OxySeq exactly as its own nested(): inner leave-one-training-person-out, seed 0, steps chosen by inner TM."""
    from . import oxyseq as OS
    from .. import splits as S
    from ..baselines import common as BC
    OT = OS.OT
    wdf = OT.window_table()
    recs = OT.rec_table(OS.feats(0), wdf)
    sids = sorted({r['sid'] for r in recs.values()})
    out = dict(outer=[], rec=[], t=[], y=[], score=[], conf=[])
    oo = dict(o_rec=[], o_t=[], o_score=[], o_conf=[])
    for p in sids:
        t0 = time.time()
        tr = [r for r in recs.values() if r['sid'] != p]
        if null is not None:
            tr = OS.permute_labels(tr, null)
        rows_sid = np.concatenate([np.full(OT.eval_mask(r).sum(), r['sid']) for r in tr])
        rows_y = np.concatenate([(r['y'][OT.eval_mask(r)] < OS.THR).astype(int) for r in tr])
        inner = list(S.inner_group_folds(rows_sid, k=5, seed=0, balance=rows_y))
        oof = {c: [] for c in OS.CKPTS}
        Y, T, R = [], [], []
        for fi in inner:
            va = set(int(s) for s in fi.test_group)
            itr = [r for r in tr if r['sid'] not in va]; iva = [r for r in tr if r['sid'] in va]
            sc = OS.scaler(itr)
            st = OS.train([OS.tensors(sc, r) for r in itr], seed=0, tta=True)
            for r in iva:
                m = OT.eval_mask(r); td = OS.tensors(sc, r)
                for c in OS.CKPTS:
                    pr = OS.predict(st[c], td, True)
                    oof[c].append((pr['score'][m], pr['q'][m]))
                Y.append((r['y'][m] < OS.THR).astype(int)); T.append(r['t'][m]); R.append(np.full(int(m.sum()), r['rec']))
        Y, T, R = np.concatenate(Y), np.concatenate(T), np.concatenate(R)
        ch = {c: BC.choose_thr(np.concatenate([o[0] for o in oof[c]]), Y, T, 'tm_bacc', 'tune', 0.5) for c in OS.CKPTS}
        c = max(OS.CKPTS, key=lambda k: (ch[k][1], -k))
        out['outer'] += [str(p)] * len(Y); out['rec'].append(R.astype(str)); out['t'].append(T); out['y'].append(Y)
        out['score'].append(np.concatenate([o[0] for o in oof[c]])); out['conf'].append(np.concatenate([o[1] for o in oof[c]]))
        if outer_fit:                                       # null draws: the member's own outer fit (1 seed, as oxyseq's null runs)
            sc = OS.scaler(tr)
            st = OS.train([OS.tensors(sc, r) for r in tr], seed=0, ckpts=(c,), tta=True)[c]
            for r in [r for r in recs.values() if r['sid'] == p]:
                m = OT.eval_mask(r); pr = OS.predict(st, OS.tensors(sc, r), True)
                oo['o_rec'].append(np.full(int(m.sum()), r['rec']).astype(str)); oo['o_t'].append(r['t'][m])
                oo['o_score'].append(pr['score'][m]); oo['o_conf'].append(pr['q'][m])
        log(f'{name} outer {p}: steps {c} inner TM {ch[c][1]:.3f} ({time.time() - t0:.0f} s)', flush=True)
    arr = {k: (np.array(v) if k == 'outer' else np.concatenate(v)) for k, v in out.items()}
    if outer_fit:
        arr.update({k: np.concatenate(v) for k, v in oo.items()})
    _save(name, **arr)


def remote(kind, draw=None, member=None):
    import warnings
    warnings.filterwarnings('ignore')
    if kind == 'real':
        jobs = {'distill': lambda: inner_distill('distill', (0, 1, 2), True, ('distill_s0', 'distill_s1', 'distill_s2')),
                'distill_nokd': lambda: inner_distill('distill_nokd', (0,), False, ('distill_nokd',)),
                'oxyseq': lambda: inner_oxyseq('oxyseq')}
    else:
        d = int(draw)
        dpk = {0: 'distill_null', 1: 'distill_d1_null', 2: 'distill_d2_null'}[d]
        jobs = {'distill': lambda: inner_distill(f'distill_nulld{d}', (0,), True, (dpk,), null=d),
                'distill_nokd': lambda: inner_distill(f'distill_nokd_nulld{d}', (0,), False, (None,), null=d),
                'oxyseq': lambda: inner_oxyseq(f'oxyseq_nulld{d}', null=d, outer_fit=True)}
    for m, fn in jobs.items():
        if member is None or m == member:
            fn()


# ================================================================================================ local: combiner
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _tm(t, y, pred):
    from .. import metrics as M
    v = M.time_matched(np.asarray(t, float), np.asarray(y, int), np.asarray(pred, int))['tm_bacc']
    return float(v) if np.isfinite(v) else 0.5


def _feat(df, members):
    return np.column_stack([np.column_stack([_logit(df[f'{m}_score']), _logit(df[f'{m}_conf'])]) for m in members])


def _fit_lr(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(sc.transform(X), y)
    return sc, lr


def _pred_lr(m, X):
    return m[1].predict_proba(m[0].transform(X))[:, 1]


def inner_table(member_files, outer):
    """Rows of the 5 training people of outer fold `outer`, members' inner-OOF predictions joined on (rec, t)."""
    tab = None
    for m, z in member_files.items():
        k = z['outer'] == str(outer)
        d = pd.DataFrame({'rec': z['rec'][k], 't': z['t'][k].astype(float).round(3), f'{m}_y': z['y'][k], f'{m}_score': z['score'][k], f'{m}_conf': z['conf'][k]})
        tab = d if tab is None else tab.merge(d, on=['rec', 't'], how='inner')
    tab['sid'] = tab.rec.str[:6]
    return tab


def fit_outer(tab, pool, y_col):
    """Nested choices for one outer fold, from the inner table only: members (top K by inner TM), C (leave-one-inner-person-out), tau."""
    from . import flashgap as F
    y, t = tab[y_col].values.astype(int), tab.t.values
    rank = {m: _tm(t, y, tab[f'{m}_score'].values > 0.5) for m in pool}
    # oxyseq's decision threshold is not 0.5: rank by the TM of the member's own best single-feature LR would be leak-free too; we use AUROC-free TM of
    # a 1-d LR on its logit (inner, in-sample for a 2-parameter map), which treats members with shifted thresholds fairly.
    for m in pool:
        mm = _fit_lr(_logit(tab[f'{m}_score'].values)[:, None], y, 1.0)
        rank[m] = _tm(t, y, _pred_lr(mm, _logit(tab[f'{m}_score'].values)[:, None]) > 0.5)
    members = sorted(pool, key=lambda m: -rank[m])[:K]
    X = _feat(tab, members)
    people = np.unique(tab.sid.values)
    lo = {}
    for C in GRID_C:
        p = np.zeros(len(tab))
        for ip in people:
            k = tab.sid.values == ip
            p[k] = _pred_lr(_fit_lr(X[~k], y[~k], C), X[k])
        lo[C] = (_tm(t, y, p > 0.5), p)
    C = max(GRID_C, key=lambda c: (round(lo[c][0], 6), -c))           # ties: stronger regularisation
    p = lo[C][1]
    tau, cert, cov, acc = F.select_tau(np.abs(p - 0.5), (p > 0.5).astype(int), y)
    model = _fit_lr(X, y, C)
    coef = dict(zip([f'{m}:{k}' for m in members for k in ('logit', 'conf')], np.round(model[1].coef_[0] / model[0].scale_, 4).tolist()))
    return dict(members=members, rank={m: round(v, 4) for m, v in rank.items()}, C=C, inner_lopo_tm=round(lo[C][0], 4),
                tau=float(tau), tau_certified=bool(cert), inner_cov=round(float(cov), 3), inner_acc=round(float(acc), 3),
                coef=coef, intercept=round(float(model[1].intercept_[0] - (model[1].coef_[0] * model[0].mean_ / model[0].scale_).sum()), 4)), model


def outer_frame(outer_src, members, lv):
    """Members' outer (held-out) predictions at a level, joined with the contract columns."""
    base = None
    for m in members:
        src = outer_src[m]
        if isinstance(src, str):
            d = pd.read_csv(SCORES / f'{src}__{tag(lv)}.csv')
            d = d.rename(columns={'score': f'{m}_score', 'conf': f'{m}_conf'})[['rec', 't', 'sid', 'y_spo2', 'y', f'{m}_score', f'{m}_conf']]
        else:
            z = src
            d = pd.DataFrame({'rec': z['o_rec'], 't': z['o_t'], f'{m}_score': z['o_score'], f'{m}_conf': z['o_conf']})
        d['t'] = d.t.astype(float).round(3)
        if base is None:
            base = d
        else:
            base = base.merge(d.drop(columns=[c for c in ('sid', 'y_spo2', 'y') if c in d.columns and c in base.columns]), on=['rec', 't'], how='inner')
    return base


def combine(name, member_files, outer_src, levels, null=False, log=print):
    ref = pd.read_csv(SCORES / 'chain_v1__L0.csv')[['rec', 't', 'sid', 'y_spo2', 'y']]
    ref['t'] = ref.t.astype(float).round(3)
    info = []
    out = {lv: [] for lv in levels}
    for p in sorted(ref.sid.astype(str).unique()):
        tab = inner_table(member_files, p)
        assert not (tab.sid.values == p).any()
        ycol = f'{POOL[0]}_y'
        if not null:                                        # real labels agree across members; null labels are the distill member's permutation
            for m in member_files:
                assert (tab[f'{m}_y'].values == tab[ycol].values).all(), m
        fo, model = fit_outer(tab, list(member_files), ycol)
        fo.update(sid=p, n_inner=len(tab))
        info.append(fo)
        log(f'{name} outer {p}: members {fo["members"]} C {fo["C"]} inner LOPO TM {fo["inner_lopo_tm"]:.3f} tau {fo["tau"]:.3f} '
            f'({"cert" if fo["tau_certified"] else "NOT cert"}, cov {fo["inner_cov"]:.2f}) coef {fo["coef"]}')
        for lv in levels:
            d = outer_frame(outer_src, fo['members'], lv)
            d = d[d.rec.str[:6] == p].copy()
            if 'sid' not in d.columns:
                d = d.merge(ref, on=['rec', 't'], how='inner')
            s = _pred_lr(model, _feat(d, fo['members']))
            cf = np.abs(s - 0.5)
            out[lv].append(pd.DataFrame(dict(rec=d.rec.values, t=d.t.values, sid=d.rec.str[:6].values, y_spo2=d.y_spo2.values, y=d.y.values.astype(int),
                                             score=s, pred=(s > 0.5).astype(int), answer=cf >= fo['tau'], conf=cf)))
    SCORES.mkdir(parents=True, exist_ok=True)
    for lv in levels:
        df = pd.concat(out[lv], ignore_index=True)
        df = ref[['rec', 't']].merge(df, on=['rec', 't'], how='left')
        assert df.score.notna().all() and len(df) == len(ref), (name, lv, df.score.isna().sum())
        df['t'] = df.t.round().astype(int)
        p = SCORES / f'{name}__{tag(lv)}.csv'
        df.to_csv(p, index=False, float_format='%.6g')
        log(f'  {name} {tag(lv)}: TM {_tm(df.t, df.y, df.pred):.3f}, answered {df.answer.mean():.2f}')
    return info


def load_inner(names):
    return {m: dict(np.load(INNER / f'{f}.npz', allow_pickle=True)) for m, f in names.items()}


def local(levels=LEVELS, what=('real', 'null')):
    import warnings
    warnings.filterwarnings('ignore')
    log_info = {}
    if 'real' in what:
        mf = load_inner({m: m for m in POOL if (INNER / f'{m}.npz').exists()})
        log_info['ens3'] = combine('ens3', mf, {m: m for m in mf}, levels)
    if 'null' in what:
        for d, nm in enumerate(('ens3_null', 'ens3_d1_null', 'ens3_d2_null')):
            names = {'distill': f'distill_nulld{d}', 'distill_nokd': f'distill_nokd_nulld{d}', 'oxyseq': f'oxyseq_nulld{d}'}
            names = {m: f for m, f in names.items() if (INNER / f'{f}.npz').exists()}
            if 'distill' not in names:
                continue
            mf = load_inner(names)
            src = {'distill': {0: 'distill_null', 1: 'distill_d1_null', 2: 'distill_d2_null'}[d]}
            src.update({m: mf[m] for m in mf if m != 'distill'})
            log_info[nm] = combine(nm, mf, src, [0], null=True)
    p = RES / 'ens3_fold_info.json'
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(log_info)
    p.write_text(json.dumps(old, indent=1))
    return log_info


def run(level=0, seed=0):
    """CONTRACT entry point (combiner is deterministic; members carry their own seeds)."""
    local([level], ('real',))
    return pd.read_csv(SCORES / f'ens3__{tag(level)}.csv')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inner', nargs='*', default=None, help='remote: real | null <draw>')
    ap.add_argument('--member', default=None)
    ap.add_argument('--what', nargs='*', default=['real', 'null'])
    a = ap.parse_args(argv)
    if a.inner:
        remote(a.inner[0], a.inner[1] if len(a.inner) > 1 else None, a.member)
        return
    local(what=a.what)


if __name__ == '__main__':
    sys.exit(main())
