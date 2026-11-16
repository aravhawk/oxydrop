"""Baselines under the photon stress levels: the logistic chain v1 (shipped C, and C selected by nested inner CV) re-scored on degraded held-out recordings,
plus the existing clean (level 0) out-of-fold scores of the CNN and the hybrid converted to the common score-file format.

    cd python && python -m oxydrop_ml.photon.baselines_stress [--levels 0 1 2 3 4 5 X1] [--convert]

Protocol (leakage-free, same harness as baselines/p1.py):
  * leave-one-person-out, both hands together (splits.loso over the people of the 5,644 evaluated windows);
  * the chain is fit on CLEAN training people and is NOT refit per level; its C values (nested_C: the inner subject-grouped CV of `common.run_cls` on the training people)
    and the threshold (0.5, as shipped) never see the held-out person, degraded or not;
  * only the held-out person's two recordings are degraded (`stress.degrade` / `stress.degrade_x`, the whole recording, baseline minute included); the chain features
    are then recomputed from the degraded raw RGB with oxydrop.chain.deltas (hook: `chain_v1.hoffman_records` is replaced by the degraded recordings).
Score files: results/photon/scores/chain_v1__L*.csv, chain_v1_nestedC__L*.csv (and __X1), cnn_full__L0.csv, hybrid_final__L0.csv.
`answer` of chain_v1 is the shipped abstain band (|p - 0.5| >= 0.10); of chain_v1_nestedC the nested band chosen on the inner folds of the training people.
CNN / hybrid under stress: the repository holds no leave-one-person-out checkpoints (only the 12 final all-people fits in ml_cache/runs/hyb_final, which have seen every
test person), so scoring them on degraded recordings would need retraining and is not done here. Their level-0 out-of-fold files are converted so the clean table includes them.
"""
import argparse
import contextlib
import time
import warnings

import numpy as np
import pandas as pd

from .. import metrics as M
from .. import windows as W
from ..baselines import chain_v1, common as C
from . import stress as S
from .evalkit import SCORES, eval_windows


class Ctx:
    pass


def build_ctx():
    """The 5,644 evaluated Hoffman windows with the clean chain features (same rows and order as baselines/p1.build_ctx)."""
    ws = W.load('hoffman'); m = ws.meta
    sel = np.nonzero(m['eval'].values & m.has_low.values)[0]
    meta = m.iloc[sel].reset_index(drop=True)
    ctx = Ctx(); ctx.ws = ws; ctx.meta = meta
    ctx.y = meta.low.values.astype(int); ctx.t = meta.t_start.values.astype(float); ctx.subject = meta.subject.values.astype(str)
    X, valid, src = chain_v1.chain_features(ws, meta, 1)
    ctx.F = dict(fusion19_v1=X); ctx.valid = {1: valid}; ctx.feature_source = src
    return ctx


@contextlib.contextmanager
def degraded_records(level, seed=0):
    """Hook: make chain_v1.hoffman_records return the recordings degraded to `level` (an int of stress.LEVELS or an X-name)."""
    orig = chain_v1.hoffman_records

    def patched(ws):
        recs, src = orig(ws)
        if level == 0:
            return recs, src
        f = S.degrade_x if isinstance(level, str) else S.degrade
        return [(r, f(rgb, level, seed)) for r, rgb in recs], src + f' degraded to {level}'
    chain_v1.hoffman_records = patched
    try:
        yield
    finally:
        chain_v1.hoffman_records = orig


def degraded_features(ctx, level, seed=0):
    with degraded_records(level, seed):
        X, valid, _ = chain_v1.chain_features(ctx.ws, ctx.meta, 1)
    return X


def score_frame(ctx, rows, score, pred, answer):
    """Contract-format frame for `rows` (indices into ctx), sorted like windows.csv."""
    ref = eval_windows()
    m = ctx.meta.iloc[rows]
    d = pd.DataFrame(dict(rec=m.rec.str.replace('hof_', '', regex=False).values, t=m.t_start.astype(int).values, low=m.low.astype(int).values, score=score, pred=pred.astype(int), answer=answer.astype(bool)))
    out = ref.merge(d, on=['rec', 't'], how='left', validate='one_to_one')
    assert out.score.notna().all(), 'every evaluated window must be scored'
    assert (out.y.values == out.low.values).all(), 'labels of windows.csv and the shard differ'
    return out[['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer']]


def run(levels=None, log=print, out_dir=SCORES, seed=0):
    t0 = time.time()
    ctx = build_ctx()
    log(f'{len(ctx.y)} windows, {len(np.unique(ctx.subject))} people, chain features from {ctx.feature_source}')
    levels = S.LEVELS + S.X_LEVELS if levels is None else levels
    Xd = {l: (ctx.F['fusion19_v1'] if l == 0 else degraded_features(ctx, l, seed)) for l in levels}
    log(f'features for {len(levels)} levels done ({time.time() - t0:.0f}s)')
    spec1, spec2 = chain_v1.specs(ctx, 1)           # chain_v1 (shipped C), chain_v1_nestedC (25-point grid, inner CV)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec, name, band_kind in ((spec1, 'chain_v1', 'shipped'), (spec2, 'chain_v1_nestedC', 'nested')):
        folds = C.run_cls(ctx, spec)                 # selection of C on the clean training people only
        log(f'  {name}: folds done ({time.time() - t0:.0f}s), hp per fold: {[tuple(fo.hp.values()) for fo in folds]}')
        for l in levels:
            rows, sc, pr, an = [], [], [], []
            for fo in folds:
                m = spec.fit(fo.hp, fo.tr)                                     # clean training people
                s = chain_v1.ChainPred(m.m, Xd[l]).score(fo.te)                # degraded held-out person
                rows.append(fo.te); sc.append(s); pr.append(s > fo.thr)
                band = spec.shipped_band if band_kind == 'shipped' else fo.band
                an.append(np.abs(s - fo.thr) >= band)
            df = score_frame(ctx, np.concatenate(rows), np.concatenate(sc), np.concatenate(pr), np.concatenate(an))
            tag = f'L{l}' if isinstance(l, (int, np.integer)) else l
            df.to_csv(out_dir / f'{name}__{tag}.csv', index=False, float_format='%.6g')
            tm = M.time_matched(df.t.values, df.y.values, df.pred.values, df.score.values)['tm_bacc']
            log(f'  {name} {tag}: TM bal. acc. {tm:.3f}, answered {df.answer.mean():.2f}')
    return True


def convert_existing(out_dir=SCORES, log=print):
    """Level-0 files for the existing leave-one-person-out CNN (3-seed mean of p_low, threshold 0.5) and hybrid (prob, threshold 0.5)."""
    ref = eval_windows()
    out_dir.mkdir(parents=True, exist_ok=True)
    key = lambda d, rc, tc: d[rc].astype(str).str.replace('hof_', '', regex=False) + '|' + d[tc].astype(int).astype(str)
    ref['key'] = ref.rec + '|' + ref.t.astype(str)

    def finish(name, d, score, extra=None):
        o = ref.copy(); s = pd.Series(score.values, index=d['key'].values)
        o['score'] = s.reindex(o.key).values
        assert np.isfinite(o.score).all(), f'{name}: windows missing'
        o['pred'] = (o.score > 0.5).astype(int); o['answer'] = True; o['conf'] = (o.score - 0.5).abs()
        for k, v in (extra or {}).items():
            o[k] = pd.Series(v.values, index=d['key'].values).reindex(o.key).values
        o = o.drop(columns='key')
        o.to_csv(out_dir / f'{name}__L0.csv', index=False, float_format='%.6g')
        tm = M.time_matched(o.t.values, o.y.values, o.pred.values, o.score.values)['tm_bacc']
        log(f'  {name} L0: TM bal. acc. {tm:.3f}')

    c = pd.read_csv(C.SCORES / 'cnn_full_hoffman.csv'); c = c[c['eval'].astype(bool)].copy()
    c['key'] = key(c, 'rec', 't_start')
    g = c.groupby('key').agg(p=('p_low', 'mean'), sp=('spo2_pred', 'mean'))
    d = pd.DataFrame(dict(key=g.index)); finish('cnn_full', d, g.p, dict(spo2_hat=g.sp))
    h = pd.read_csv(C.SCORES / 'hybrid_final_hoffman.csv'); h = h[h['eval'].astype(bool)].copy()
    h['key'] = key(h, 'rec', 't_start')
    finish('hybrid_final', h, h.set_index('key').prob)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--levels', nargs='*', default=None, help='subset of 0-5 and X1 (default all)')
    ap.add_argument('--convert', action='store_true', help='only convert the existing CNN/hybrid level-0 files')
    ap.add_argument('--no-convert', action='store_true')
    a = ap.parse_args(argv)
    warnings.filterwarnings('ignore')
    if a.convert:
        convert_existing(); return
    lv = None if a.levels is None else [int(x) if x.isdigit() else x for x in a.levels]
    run(lv)
    if not a.no_convert:
        convert_existing()


if __name__ == '__main__':
    main()
