"""Control models for the photon stress test: how much of the robustness of a physics model is just the baseline-relative DC colour?

    cd python && python -m oxydrop_ml.photon.controls [--levels 0 1 2 3 4 5 X1]

Question. `twin_inverse_noresid` (physics inversion, no heart-rate input) is nearly flat under the simulated distant-flash stress while `chain_v1` collapses.
These controls ask whether a trivial class-balanced logistic regression on the same cue is equally robust, so the contribution of the physics model is reported honestly.
All inputs are the calibration-relative features of the shipped chain (`oxydrop.chain.deltas`): window feature minus the median over the windows that end within the
first 60 s of the SAME recording, computed from the (degraded) raw per-frame RGB.

    ctl_dc_lr      3 features: dc_R, dc_G, dc_B (mean over the 10 s window of ln(rgb + 1) per channel, minus the first-minute median).
    ctl_dc_ror_lr  9 features: the three DC values + ror_RG, ror_RB, ror_GB (ln ratio of the AC standard deviations, a ratio of ratios) + pi_R, pi_G, pi_B
                   (perfusion index: ln AC standard deviation per channel). The "optics features without physics".
    ctl_dc_hr_lr   4 features: the three DC values + dHR (heart-rate change vs the first minute, the chain's peak-based estimator).

Protocol (same harness as `baselines_stress.run`, which this module reuses for the loaders, the degraded features and the score-file writer):
  * leave-one-person-out over the 5,644 evaluated windows, both hands together (`splits.loso`, `common.run_cls`);
  * per fold the standardiser (StandardScaler) and the class-balanced logistic regression are fit on CLEAN training people only; C in {0.001, 0.01, 0.1, 1, 10} is
    selected by the inner person-grouped 5-fold CV of the training people (criterion: `metrics.time_matched` TM bal. acc.; fixed decision threshold 0.5);
  * the abstention half-width c is `common.choose_band(oof, y, 0.5)`: smallest c whose answered inner-OOF windows reach accuracy and bal. acc. >= 0.90 (BAND rule);
    answer = |p - 0.5| >= c;
  * the model is NOT refit per level: only the held-out person's recordings are degraded (`stress.degrade` / `stress.degrade_x`, whole recording, baseline minute
    included) and the features recomputed from them. The held-out labels are only read to score.
Null controls `<name>_null__L0.csv`: the held-out person's recordings are replaced by a constant colour (the recording's median RGB), the same pipeline then
sees no cue and must score ~0.50 TM.
Score files: results/photon/scores/ctl_*__L0..L5.csv, ctl_*__X1.csv, ctl_*_null__L0.csv.
"""
import argparse
import contextlib
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import metrics as M
from ..baselines import chain_v1, common as C
from . import baselines_stress as B
from . import stress as S
from .evalkit import SCORES

FUSION = ['hr', 'rmssd', 'sdnn', 'rr_amp', 'rr_ibi', 'pi_R', 'pi_G', 'pi_B', 'beat_cv', 'ror_RG', 'ror_RB', 'ror_GB',
          'sl_RG', 'sl_RB', 'sl30_RG', 'ror30_RG', 'dc_R', 'dc_G', 'dc_B']          # column order of chain_v1.chain_features (oxydrop.chain.FUSION)
DC = ['dc_R', 'dc_G', 'dc_B']
MODELS = {
    'ctl_dc_lr': DC,
    'ctl_dc_ror_lr': DC + ['ror_RG', 'ror_RB', 'ror_GB', 'pi_R', 'pi_G', 'pi_B'],
    'ctl_dc_hr_lr': DC + ['hr'],
}
GRID_C = [dict(C=c) for c in (0.001, 0.01, 0.1, 1.0, 10.0)]
THR = 0.5


class Pred:
    """Fitted standardiser + logistic regression; scores rows of the feature matrix X (which may be the degraded one)."""

    def __init__(self, scaler, lr, cols, X):
        self.scaler = scaler; self.lr = lr; self.cols = cols; self.X = X

    def score(self, rows):
        return self.lr.predict_proba(self.scaler.transform(self.X[rows][:, self.cols]))[:, 1]

    def on(self, X):
        return Pred(self.scaler, self.lr, self.cols, X)


def columns(names):
    return [FUSION.index(n) for n in names]


def make_spec(X, y, name, features):
    """common.Spec: class-balanced LR on the `features` columns of X; the standardiser is fit on the training rows handed to `fit` only."""
    cols = columns(features)

    def fit(hp, rows):
        sc = StandardScaler().fit(X[rows][:, cols])
        lr = LogisticRegression(C=hp['C'], class_weight='balanced', max_iter=2000).fit(sc.transform(X[rows][:, cols]), y[rows])
        return Pred(sc, lr, cols, X)
    return C.Spec(name, fit, GRID_C, thr_mode='fixed', fixed_thr=THR, select_on='tm_bacc', shipped_band=None,
                  note=f'class-balanced LR on {features}, C by inner person-grouped CV (TM bal. acc.), p>0.5, nested abstain band')


@contextlib.contextmanager
def constant_records():
    """Null control: chain_v1.hoffman_records returns every recording replaced by its median RGB (a constant colour)."""
    orig = chain_v1.hoffman_records

    def patched(ws):
        recs, src = orig(ws)
        return [(r, np.tile(np.median(rgb, 0), (len(rgb), 1))) for r, rgb in recs], src + ' replaced by a constant colour'
    chain_v1.hoffman_records = patched
    try:
        yield
    finally:
        chain_v1.hoffman_records = orig


def null_features(ctx):
    with constant_records():
        X, _, _ = chain_v1.chain_features(ctx.ws, ctx.meta, 1)
    return X


def level_tag(level):
    return f'L{level}' if isinstance(level, (int, np.integer)) else level


def score_levels(ctx, folds, spec, Xd):
    """{level: score frame}: models of each fold (clean training people) applied to the held-out person's features Xd[level]."""
    out = {}
    for lev, X in Xd.items():
        rows, sc, pr, an = [], [], [], []
        for fo in folds:
            s = spec.fit(fo.hp, fo.tr).on(X).score(fo.te)
            rows.append(fo.te); sc.append(s); pr.append(s > fo.thr); an.append(np.abs(s - fo.thr) >= fo.band)
        out[lev] = B.score_frame(ctx, np.concatenate(rows), np.concatenate(sc), np.concatenate(pr), np.concatenate(an))
    return out


def run_all(levels=None, log=print, out_dir=SCORES, seed=0, write=True):
    """Fit the three controls LOSO and score every level (+ X1) and the null control. Returns {model: {level: frame}} (null under key 'null')."""
    t0 = time.time()
    ctx = B.build_ctx()
    X0 = ctx.F['fusion19_v1']
    levels = S.LEVELS + S.X_LEVELS if levels is None else levels
    log(f'{len(ctx.y)} windows, {len(np.unique(ctx.subject))} people, chain features from {ctx.feature_source}')
    Xd = {l: (X0 if l == 0 else B.degraded_features(ctx, l, seed)) for l in levels}
    Xnull = null_features(ctx)
    log(f'features for {len(levels)} levels + null done ({time.time() - t0:.0f}s)')
    res = {}
    for name, feats in MODELS.items():
        spec = make_spec(X0, ctx.y, name, feats)
        folds = C.run_cls(ctx, spec)
        log(f'  {name}: folds done ({time.time() - t0:.0f}s), C per fold: {[fo.hp["C"] for fo in folds]}, band per fold: {[fo.band for fo in folds]}')
        frames = score_levels(ctx, folds, spec, Xd)
        frames['null'] = score_levels(ctx, folds, spec, {0: Xnull})[0]
        res[name] = frames
        for lev, df in frames.items():
            tag = f'{name}_null__L0' if lev == 'null' else f'{name}__{level_tag(lev)}'
            if write:
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / f'{tag}.csv', index=False, float_format='%.6g')
            tm = M.time_matched(df.t.values, df.y.values, df.pred.values, df.score.values)['tm_bacc']
            log(f'  {tag}: TM bal. acc. {tm:.3f}, answered {df.answer.mean():.2f}')
    return res


def run(level=0, seed=0, model='ctl_dc_lr'):
    """Contract entry point: the out-of-fold score frame of one control at one level (int of stress.LEVELS or 'X1')."""
    return run_all([level], log=lambda *a: None, seed=seed, write=False)[model][level]


TABLE_MODELS = ['chain_v1', 'twin_inverse_noresid', 'twin_inverse', 'ctl_dc_lr', 'ctl_dc_ror_lr', 'ctl_dc_hr_lr']


def level_table(ref_model='twin_inverse_noresid', B=2000):
    """Markdown tables from the written score files (run `python -m oxydrop_ml.photon.evalkit` first so TABLE.csv holds all models):
    TM bal. acc. [95 % person-bootstrap CI] per model and level, and the paired person-bootstrap difference of every other model to `ref_model` at the same level."""
    import pandas as pd
    from . import evalkit as E
    T = pd.read_csv(E.RES / 'TABLE.csv')
    levels = [str(l) for l in S.LEVELS] + S.X_LEVELS
    lab = lambda l: f'L{l}' if l.isdigit() else l
    T['lv'] = T.level.astype(str)
    out = ['| model | ' + ' | '.join(lab(l) for l in levels) + ' |', '|---|' + '---|' * len(levels)]
    for m in TABLE_MODELS:
        cells = []
        for l in levels:
            r = T[(T.model == m) & (T.lv == l)]
            cells.append(f'{r.tm_bacc.iloc[0]:.3f} [{r.tm_lo.iloc[0]:.2f}, {r.tm_hi.iloc[0]:.2f}]' if len(r) else '')
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    ref = E.eval_windows(); boot = E.Boot(ref, B=B)
    tm = {}
    for m in TABLE_MODELS:
        for l in levels:
            p = E.SCORES / f'{m}__{lab(l)}.csv'
            if p.exists():
                d = E.load_scores(p, ref); tm[(m, l)] = boot.tm(d.pred.values)
    out += ['', f'Paired person-bootstrap difference in TM bal. acc. to `{ref_model}` at the same level (model minus reference; [95 % CI]; * = CI excludes 0):', '',
            '| model | ' + ' | '.join(lab(l) for l in levels) + ' |', '|---|' + '---|' * len(levels)]
    for m in TABLE_MODELS:
        if m == ref_model:
            continue
        cells = []
        for l in levels:
            if (m, l) in tm and (ref_model, l) in tm:
                lo, hi = E.ci(tm[(m, l)][1] - tm[(ref_model, l)][1]); d = tm[(m, l)][0] - tm[(ref_model, l)][0]
                cells.append(f'{d:+.3f} [{lo:+.2f}, {hi:+.2f}]' + ('*' if lo > 0 or hi < 0 else ''))
            else:
                cells.append('')
        out.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--levels', nargs='*', default=None, help='subset of 0-5 and X1 (default all)')
    ap.add_argument('--table', action='store_true', help='print the level-by-level markdown tables from TABLE.csv and the score files (no fitting)')
    a = ap.parse_args(argv)
    warnings.filterwarnings('ignore')
    if a.table:
        print(level_table()); return
    lv = None if a.levels is None else [int(x) if x.isdigit() else x for x in a.levels]
    run_all(lv)


if __name__ == '__main__':
    main()
