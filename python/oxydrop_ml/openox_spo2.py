"""OpenOximetry contact-domain SaO2 test on the 34 held-out OpenOx people (FEAT-007, amendments/AMENDMENT_5_conformal_openox.md, Part B).

THIS IS A CONTACT RED/INFRARED FINGER-CLIP SENSOR TEST. It is not a phone-camera test and says nothing about phone SpO2 accuracy.

    # on a Daytona sandbox only (OpenOximetry is PhysioNet credentialed data: never Kaggle, never git; only aggregates leave the sandbox)
    python -m oxydrop_ml.openox_spo2 infer   --ckpt-dir ckpt --out runs/openox_spo2      # lags, labels, predictions of every model (row-level parquet stays outside git)
    python -m oxydrop_ml.openox_spo2 analyze --out runs/openox_spo2 --res results       # aggregates (CSV only)
    # on the host, from the committed aggregates
    python -m oxydrop_ml.openox_spo2 report  --res ../results/ml                        # OPENOX_SPO2.md

Who is evaluated. The 34 people of `train.openox_holdout` (deterministic, Monk-stratified quarter of the OpenOx people). They are removed before any sampling from SSL pretraining
(`ssl_full.pt`: n_excluded_subjects = 34), the stage-B warm-up and every supervised fit of the hybrid members (the `Pool` default). The scored models are the all-Hoffman fits of the
hybrid families R and P (3 seeds each, `hyb_final`) and `stageB_full.pt`; none has seen a held-out person. The output is the contact-domain SaO2 head `aux_spo2` (domain 1).
Nothing is fit or tuned on a held-out person: the label lag is estimated on the other OpenOx people (`data.openox.estimate_lag`), the train-mean and the calibrated ratio-of-ratios
are fit on the other OpenOx people, the textbook ratio-of-ratios has no parameter.
Label. `sao2` = the contact oximeter display mapped to arterial-blood-gas SaO2 with the per-encounter linear map fitted on that encounter's blood-gas draws, evaluated at the lag
(reference mean over [t - 10 + lag, t + lag]). Direct-ABG secondary label: windows that contain a blood-gas draw marker, one value per draw (lag independent).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import metrics as M

B_DEFAULT = 2000
B_AUC = 300                      # AUROC bootstraps are the slow ones (ranking of every window per resample)
LAG_SENS = list(range(-10, 61, 5))
MODELS = {'R': 'hybrid R members (3 seeds, mean), the deployed family', 'P': 'plain CNN P members (3 seeds, mean)', 'stageB': 'stage-B warm-up checkpoint (OpenOx only)'}
BASELINES = {'train_mean': 'mean SaO2 of the other OpenOx people', 'ror_textbook': 'textbook 110 - 25 R, R = exp(abs_ror_RG), clipped to [70, 100]',
             'ror_calibrated': 'OLS of SaO2 on abs_ror_RG fit on the other OpenOx people (clipped to [60, 100])'}
PRIMARY = 'R'
MONK_ORDER = ['1-3', '4-6', '7+']
ITA_ORDER = ['darkest tertile', 'mid tertile', 'lightest tertile']


# ------------------------------------------------------------------------------------------------ skin tone
def monk_group(m):
    """Monk 1-3 / 4-6 / 7+ of a person-level Monk value (median over recordings and hand sites), NaN -> 'unknown'."""
    if not np.isfinite(m):
        return 'unknown'
    r = int(np.floor(m + 0.5))
    return '1-3' if r <= 3 else ('4-6' if r <= 6 else '7+')


def ita_cuts(ita_by_person):
    """Tertile cut points of the person-level ITA over a set of people (label free)."""
    v = np.asarray([x for x in ita_by_person if np.isfinite(x)], float)
    return float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3))


def ita_group(x, cuts):
    if not np.isfinite(x):
        return 'unknown'
    return ITA_ORDER[0] if x <= cuts[0] else (ITA_ORDER[1] if x <= cuts[1] else ITA_ORDER[2])


def person_skin(meta):
    """Person-level skin fields: monk (median over the person's recordings), ita (mean)."""
    g = meta.groupby('subject').agg(monk=('monk', 'median'), ita=('ita', 'mean'))
    return g


# ------------------------------------------------------------------------------------------------ labels and baselines at a lag
def labels_at(meta, refs, lag):
    """SaO2 label of the rows of `meta` (needs subject, rec, t_start; one recording per call group) at `lag` seconds: reference series mean over [t_start + lag, t_start + lag + 10)
    (identical to `data.openox.relabel`, restricted to the given rows)."""
    from .data.openox import _win_mean
    out = np.full(len(meta), np.nan)
    rec = meta.rec.values
    ts = meta.t_start.values
    for r in pd.unique(rec):
        idx = np.nonzero(rec == r)[0]
        if r in refs.by:
            out[idx] = _win_mean(refs.by[r][2], ts[idx], lag)
    return out


def fit_baselines(train, ycol='y'):
    """Parameters of the two fitted baselines from the training people: train mean, and the OLS line SaO2 = a + b abs_ror_RG."""
    ok = np.isfinite(train[ycol].values)
    mean = float(train[ycol].values[ok].mean())
    x, y = train.abs_ror_RG.values, train[ycol].values
    k = ok & np.isfinite(x)
    b, a = np.polyfit(x[k], y[k], 1)
    return dict(train_mean=mean, a=float(a), b=float(b), n_train=int(ok.sum()), n_train_people=int(train.subject[ok].nunique()))


def baseline_predictions(ror, par):
    ror = np.asarray(ror, float)
    r = np.exp(ror)
    return dict(train_mean=np.full(len(ror), par['train_mean']), ror_textbook=np.clip(110.0 - 25.0 * r, 70.0, 100.0), ror_calibrated=np.clip(par['a'] + par['b'] * ror, 60.0, 100.0))


# ------------------------------------------------------------------------------------------------ models
def input_transform(norm, folded_domain):
    """Function (x, s, domain) -> inputs of a checkpoint. Final checkpoints have the scale of `folded_domain` folded into their weights (train.fold_norm), so the raw features of another
    domain are first rescaled by scale[folded] / scale[domain], which is exact because the fold is scale-only. Unfolded checkpoints (stage B) get x / scale[domain]."""
    xs = np.asarray(norm['x_scale'], np.float32); ss = np.asarray(norm['s_scale'], np.float32)

    def fx(x, s, dom):
        d = np.asarray(dom)
        if folded_domain is None:
            return x / xs[d][:, :, None], s / ss[d]
        return x * (xs[folded_domain][None, :, None] / xs[d][:, :, None]), s * (ss[folded_domain][None] / ss[d])
    return fx


def load_model(path):
    import torch
    from .model import OxyCfg, OxyNet
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = OxyNet(OxyCfg(**ck['cfg']))
    net.load_state_dict(ck['state'])
    net.eval()
    return net, input_transform(ck['norm'], ck.get('folded_norm_domain'))


def aux_predict(net, fx, x, s, dom):
    import torch
    with torch.no_grad():
        xx, ss = fx(x, s, dom)
        o = net(torch.from_numpy(np.ascontiguousarray(xx, dtype=np.float32)), torch.from_numpy(np.asarray(dom)), torch.from_numpy(np.ascontiguousarray(ss, dtype=np.float32)), None, with_aux=True)
    return o['aux_spo2'].float().numpy()


# ------------------------------------------------------------------------------------------------ inference (Daytona)
def infer(ckpt_dir, out_dir, bs=512, n_boot_lag=1000, hyb_dir=None, seeds=(0, 1, 2)):
    from . import windows as W
    from .data.openox import RefSeries, estimate_lag
    from .train import HP, Pool, openox_holdout
    os.makedirs(out_dir, exist_ok=True)
    ws = W.load('openox')
    meta = ws.meta
    hold = openox_holdout(meta)
    hold_set = set(hold)
    train_people = sorted(set(meta.subject) - hold_set)
    print('holdout', len(hold), 'train people', len(train_people), flush=True)
    refs = RefSeries(ws)
    lag_ppg = estimate_lag(train_people, ws=ws, method='ppg', n_boot=n_boot_lag)
    lag_abg = estimate_lag(train_people, ws=ws, method='abg', n_boot=n_boot_lag)
    lag_star = float(lag_ppg['lag']) if (lag_ppg['confident'] and np.isfinite(lag_ppg['lag'])) else 0.0
    print('lag ppg', {k: lag_ppg[k] for k in ('lag', 'n_enc', 'confident', 'ci')}, 'lag abg', {k: lag_abg[k] for k in ('lag', 'n_enc', 'ci')}, '-> lag*', lag_star, flush=True)
    json.dump(dict(ppg={k: lag_ppg[k] for k in ('lag', 'n_enc', 'confident', 'ci', 'profile')}, abg={k: lag_abg[k] for k in ('lag', 'n_enc', 'confident', 'ci', 'profile')}, lag_star=lag_star,
                   holdout=hold, n_train_people=len(train_people)), open(os.path.join(out_dir, 'lag.json'), 'w'), indent=1, default=float)
    # rows to score: held-out people, aligned, t >= 60 s, with an SaO2 label (continuous) or a blood-gas draw
    m = meta.copy()
    m['local'] = np.arange(len(m))
    h = m[m.subject.isin(hold_set) & m.aligned & m['eval'] & (m.has_sao2 | m.has_sao2_draw)].copy()
    tr = m[~m.subject.isin(hold_set) & m.aligned & m['eval'] & m.has_sao2].copy()
    for L in LAG_SENS + [lag_star]:
        h[f'y_lag{L:g}'] = labels_at(h, refs, L)
    h['y_lag0'] = labels_at(h, refs, 0.0); h['y_lagstar'] = labels_at(h, refs, lag_star)
    tr['y'] = labels_at(tr, refs, lag_star)
    par = fit_baselines(tr)
    print('baseline params', par, flush=True)
    bp = baseline_predictions(h.abs_ror_RG.values, par)
    # inference
    pool = Pool(('openox',), with_ox_all=True)
    assert sorted(pool.ox_holdout) == sorted(hold), 'pool and shard disagree on the held-out people'
    lut = np.full(len(meta), -1, np.int64)
    lut[pool.local] = np.arange(pool.N)
    rows = lut[h.local.values]
    assert (rows >= 0).all()
    hp = HP()
    models = {}
    for fam in ('R', 'P'):
        models[fam] = [load_model(os.path.join(ckpt_dir, f'{fam}_s{s}.pt')) for s in seeds]
    models['stageB'] = [load_model(os.path.join(ckpt_dir, 'stageB_full.pt'))]
    pred = {k: [] for k in models}
    sqi = []
    for a in range(0, len(rows), bs):
        r = rows[a:a + bs]
        x, s, dom = pool.inputs(r, hp=hp)
        for fam, ms in models.items():
            pred[fam].append(np.mean([aux_predict(n, fx, x, s, dom) for n, fx in ms], axis=0))
        if (a // bs) % 10 == 0:
            print('  infer', a, '/', len(rows), flush=True)
    for fam in models:
        h['pred_' + fam] = np.concatenate(pred[fam])
    for k, v in bp.items():
        h['pred_' + k] = v
    keep = ['subject', 'rec', 't_start', 'local', 'spo2_ref', 'sao2', 'sao2_draw', 'has_sao2', 'has_sao2_draw', 'monk', 'ita', 'skin_group', 'abs_ror_RG'] + [c for c in h.columns if c.startswith(('y_lag', 'pred_'))]
    h[keep].to_parquet(os.path.join(out_dir, 'preds.parquet'))
    json.dump(par, open(os.path.join(out_dir, 'baseline_params.json'), 'w'), indent=1)
    # all-people table for the non-learned analyses (textbook RoR and contact reference by skin tone)
    allp = m[m.aligned & m['eval'] & m.has_sao2][['subject', 'rec', 't_start', 'abs_ror_RG', 'monk', 'ita']].copy()
    allp['y'] = labels_at(allp, refs, lag_star)
    allp.to_parquet(os.path.join(out_dir, 'all_people.parquet'))
    dr = pd.read_parquet(os.path.join(ws.dir, 'draws.parquet'))
    rs = m.drop_duplicates('rec')[['rec', 'subject', 'monk', 'ita', 'aligned']]
    dr = dr.merge(rs, on='rec', how='left')
    dr.to_parquet(os.path.join(out_dir, 'draws_all.parquet'))
    pp = pd.DataFrame(dict(subject=sorted(set(m.subject)))).merge(person_skin(m).reset_index(), on='subject', how='left')
    pp['held_out'] = pp.subject.isin(hold_set)
    pp.to_parquet(os.path.join(out_dir, 'people.parquet'))
    print('inference done', len(h), 'windows', flush=True)


# ------------------------------------------------------------------------------------------------ metrics
def err_metrics(y, p, thr=90.0, auc=True):
    """MAE, bias (prediction minus SaO2), ARMS (root mean square error) over SaO2 70..100 and over all windows, AUROC for SaO2 < 90 with the predicted SpO2 as the (negated) score."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return dict(n=0, mae=np.nan, bias=np.nan, arms=np.nan, arms_70_100=np.nan, auroc_lt90=np.nan, sd_error=np.nan)
    e = p - y
    r = (y >= 70) & (y <= 100)
    return dict(n=int(len(y)), mae=float(np.abs(e).mean()), bias=float(e.mean()), sd_error=float(e.std(ddof=1)) if len(e) > 1 else np.nan, arms=float(np.sqrt((e ** 2).mean())),
                arms_70_100=float(np.sqrt((e[r] ** 2).mean())) if r.any() else np.nan, auroc_lt90=M.auroc((y < thr).astype(int), -p) if auc else np.nan)


def person_table(df, ycol, pcol):
    """Per-person metrics (mean over windows of the person)."""
    rows = []
    for s, g in df.groupby('subject'):
        m = err_metrics(g[ycol].values, g[pcol].values)
        m['subject'] = s
        rows.append(m)
    return pd.DataFrame(rows)


def boot_stat(df, ycol, pcol, key, B, seed=0):
    d = df[np.isfinite(df[ycol].values) & np.isfinite(df[pcol].values)]
    y, p, s = d[ycol].values, d[pcol].values, d.subject.values
    if len(np.unique(s)) < 2:
        return dict(point=err_metrics(y, p, auc=key == 'auroc_lt90')[key], lo=np.nan, hi=np.nan)
    auc = key == 'auroc_lt90'
    b = M.person_bootstrap(s, lambda idx: err_metrics(y[idx], p[idx], auc=auc)[key], B=min(B, B_AUC) if auc else B, seed=seed)
    return dict(point=b['point'], lo=b['lo'], hi=b['hi'])


def overall_rows(df, label, ycol, B):
    """Pooled metrics with person-bootstrap intervals, for every model and baseline on the same windows."""
    rows = []
    d = df[np.isfinite(df[ycol].values)]
    for name in list(MODELS) + list(BASELINES):
        pc = 'pred_' + name
        r = dict(label=label, model=name, n_windows=int(len(d)), n_people=int(d.subject.nunique()), prevalence_lt90=float((d[ycol] < 90).mean()))
        for k in ('mae', 'bias', 'arms', 'arms_70_100', 'auroc_lt90'):
            b = boot_stat(d, ycol, pc, k, B)
            r[k], r[k + '_lo'], r[k + '_hi'] = b['point'], b['lo'], b['hi']
        pt = person_table(d, ycol, pc)
        r['person_mae_mean'], r['person_mae_sd'] = float(pt.mae.mean()), float(pt.mae.std(ddof=1)) if len(pt) > 1 else np.nan
        rows.append(r)
    return rows


def paired_rows(df, label, ycol, B):
    """Paired person bootstrap of the MAE difference (model minus baseline; negative = model better)."""
    rows = []
    d = df[np.isfinite(df[ycol].values)]
    y, s = d[ycol].values, d.subject.values
    for m in MODELS:
        for b in list(BASELINES) + [k for k in MODELS if k != m]:
            pm, pb = d['pred_' + m].values, d['pred_' + b].values
            r = M.paired_person_bootstrap(s, lambda i: float(np.abs(pm[i] - y[i]).mean()), lambda i: float(np.abs(pb[i] - y[i]).mean()), B=B, seed=0, higher_is_better=False)
            # higher_is_better=False: diff is baseline minus model reported positive when the model is better -> convert to model minus baseline
            rows.append(dict(label=label, model=m, baseline=b, model_mae=r['a'], baseline_mae=r['b'], diff_model_minus_baseline=r['a'] - r['b'], lo=-r['hi'], hi=-r['lo'],
                             ))
    out = pd.DataFrame(rows)
    out['verdict'] = np.where(out.hi < 0, 'model better', np.where(out.lo > 0, 'model worse', 'tie'))
    return out.to_dict('records')


def group_frame(df, people_skin, cuts):
    d = df.copy()
    ps = people_skin.copy()
    ps['monk_group'] = ps.monk.map(monk_group)
    ps['ita_group'] = ps.ita.map(lambda v: ita_group(v, cuts))
    return d.merge(ps[['subject', 'monk_group', 'ita_group']], on='subject', how='left')


def skin_rows(df, label, ycol, B, models=None):
    """Per-group metrics (person-bootstrap CI) and dark-minus-light gaps (independent person bootstraps per group)."""
    rows, gaps = [], []
    models = models or (list(MODELS) + list(BASELINES))
    d = df[np.isfinite(df[ycol].values)]
    for scheme, col, order, dark, light in (('monk', 'monk_group', MONK_ORDER, '7+', '1-3'), ('ita', 'ita_group', ITA_ORDER, ITA_ORDER[0], ITA_ORDER[2])):
        for g in order + ['unknown']:
            sub = d[d[col] == g]
            if not len(sub):
                continue
            for name in models:
                pc = 'pred_' + name
                r = dict(label=label, scheme=scheme, group=g, model=name, n_people=int(sub.subject.nunique()), n_windows=int(len(sub)))
                for k in ('mae', 'bias', 'arms_70_100', 'auroc_lt90'):
                    b = boot_stat(sub, ycol, pc, k, B) if sub.subject.nunique() >= 2 else dict(point=err_metrics(sub[ycol].values, sub[pc].values)[k], lo=np.nan, hi=np.nan)
                    r[k], r[k + '_lo'], r[k + '_hi'] = b['point'], b['lo'], b['hi']
                rows.append(r)
        for name in models:
            pc = 'pred_' + name
            gd, gl = d[d[col] == dark], d[d[col] == light]
            for k in ('mae', 'bias'):
                nd, nl = gd.subject.nunique(), gl.subject.nunique()
                ok = nd >= 3 and nl >= 3
                if ok:
                    ed, el = gd[[ycol, pc, 'subject']].dropna(), gl[[ycol, pc, 'subject']].dropna()
                    bd = M.person_bootstrap(ed.subject.values, lambda i: err_metrics(ed[ycol].values[i], ed[pc].values[i], auc=False)[k], B=B, seed=1)
                    bl = M.person_bootstrap(el.subject.values, lambda i: err_metrics(el[ycol].values[i], el[pc].values[i], auc=False)[k], B=B, seed=2)
                    diff = bd['boot'] - bl['boot']
                    gaps.append(dict(label=label, scheme=scheme, model=name, metric=k, dark=dark, light=light, n_dark=nd, n_light=nl, dark_value=bd['point'], light_value=bl['point'],
                                     gap=bd['point'] - bl['point'], lo=float(np.nanpercentile(diff, 2.5)), hi=float(np.nanpercentile(diff, 97.5)), estimable=True))
                else:
                    gaps.append(dict(label=label, scheme=scheme, model=name, metric=k, dark=dark, light=light, n_dark=nd, n_light=nl, dark_value=np.nan, light_value=np.nan, gap=np.nan, lo=np.nan, hi=np.nan,
                                     estimable=False))
    return rows, gaps


def draw_level(df, B):
    """Direct-ABG secondary test: one row per blood-gas draw (mean prediction of the windows that contain the draw marker)."""
    d = df[df.has_sao2_draw & np.isfinite(df.sao2_draw.values)].copy()
    d['draw_key'] = d.rec.astype(str) + '|' + d.sao2_draw.round(4).astype(str)
    agg = {c: 'mean' for c in df.columns if c.startswith('pred_')}
    g = d.groupby('draw_key').agg(subject=('subject', 'first'), sao2_draw=('sao2_draw', 'first'), n_windows=('t_start', 'size'), **{c: (c, 'mean') for c in agg}).reset_index()
    return g


def contact_bias(draws, people, cuts, B):
    """SpO2 (contact display at the draw) minus ABG SaO2, per draw, by skin tone; person-level mean per person, group means with person bootstrap, dark-minus-light gaps. No model involved."""
    d = draws[np.isfinite(draws.sao2.values) & np.isfinite(draws.spo2_dev.values)].copy()
    d['bias'] = d.spo2_dev - d.sao2
    d = d.merge(people.assign(monk_group=people.monk.map(monk_group), ita_group=people.ita.map(lambda v: ita_group(v, cuts)))[['subject', 'monk_group', 'ita_group']], on='subject', how='left')
    rows, gaps = [], []
    for scheme, col, order, dark, light in (('monk', 'monk_group', MONK_ORDER, '7+', '1-3'), ('ita', 'ita_group', ITA_ORDER, ITA_ORDER[0], ITA_ORDER[2])):
        res = {}
        for g in order:
            s = d[d[col] == g]
            if not len(s):
                continue
            per = s.groupby('subject').bias.mean()
            b = M.person_bootstrap(s.subject.values, lambda i: float(s.bias.values[i].mean()), B=B, seed=3) if s.subject.nunique() >= 2 else dict(point=float(s.bias.mean()), lo=np.nan, hi=np.nan, boot=np.array([np.nan]))
            occult = ((s.spo2_dev >= 92) & (s.sao2 < 88)).sum() / max(int((s.sao2 < 88).sum()), 1)
            rows.append(dict(scheme=scheme, group=g, n_people=int(s.subject.nunique()), n_draws=int(len(s)), bias=b['point'], lo=b['lo'], hi=b['hi'], person_mean=float(per.mean()),
                             occult_share_of_sao2_lt88=float(occult), n_sao2_lt88=int((s.sao2 < 88).sum())))
            res[g] = (s, b)
        if dark in res and light in res and res[dark][0].subject.nunique() >= 3 and res[light][0].subject.nunique() >= 3:
            bd, bl = res[dark][1], res[light][1]
            diff = bd['boot'] - bl['boot']
            gaps.append(dict(scheme=scheme, dark=dark, light=light, n_dark=int(res[dark][0].subject.nunique()), n_light=int(res[light][0].subject.nunique()), gap=bd['point'] - bl['point'],
                             lo=float(np.nanpercentile(diff, 2.5)), hi=float(np.nanpercentile(diff, 97.5)), estimable=True))
        else:
            gaps.append(dict(scheme=scheme, dark=dark, light=light, n_dark=int(res[dark][0].subject.nunique()) if dark in res else 0, n_light=int(res[light][0].subject.nunique()) if light in res else 0,
                             gap=np.nan, lo=np.nan, hi=np.nan, estimable=False))
    return rows, gaps


def ror_all_people(allp, people, cuts, B):
    """Textbook ratio-of-ratios MAE by skin tone over ALL aligned OpenOx people (no fitted parameter, so no held-out requirement); descriptive."""
    d = allp[np.isfinite(allp.y.values) & np.isfinite(allp.abs_ror_RG.values)].copy()
    d['pred'] = baseline_predictions(d.abs_ror_RG.values, dict(train_mean=0, a=0, b=0))['ror_textbook']
    d = d.merge(people.assign(monk_group=people.monk.map(monk_group), ita_group=people.ita.map(lambda v: ita_group(v, cuts)))[['subject', 'monk_group', 'ita_group']], on='subject', how='left')
    rows = []
    for scheme, col, order in (('monk', 'monk_group', MONK_ORDER), ('ita', 'ita_group', ITA_ORDER)):
        for g in order:
            s = d[d[col] == g]
            if not len(s):
                continue
            b = M.person_bootstrap(s.subject.values, lambda i: err_metrics(s.y.values[i], s.pred.values[i], auc=False)['mae'], B=B, seed=4) if s.subject.nunique() >= 2 else dict(point=np.nan, lo=np.nan, hi=np.nan)
            bb = M.person_bootstrap(s.subject.values, lambda i: err_metrics(s.y.values[i], s.pred.values[i], auc=False)['bias'], B=B, seed=4) if s.subject.nunique() >= 2 else dict(point=np.nan, lo=np.nan, hi=np.nan)
            rows.append(dict(scheme=scheme, group=g, n_people=int(s.subject.nunique()), n_windows=int(len(s)), mae=b['point'], mae_lo=b['lo'], mae_hi=b['hi'], bias=bb['point'], bias_lo=bb['lo'], bias_hi=bb['hi']))
    return rows


def lag_rows(df, lagjson):
    rows = []
    for method in ('ppg', 'abg'):
        j = lagjson[method]
        rows.append(dict(kind='estimate', method=method, lag=j['lag'], n_encounters=j['n_enc'], confident=bool(j['confident']), ci_lo=(j['ci'] or [np.nan, np.nan])[0], ci_hi=(j['ci'] or [np.nan, np.nan])[1]))
    for L in sorted({int(c.replace('y_lag', '')) for c in df.columns if c.startswith('y_lag') and c != 'y_lagstar' and c.replace('y_lag', '').lstrip('-').isdigit()}):
        c = f'y_lag{L}'
        d = df[np.isfinite(df[c].values)]
        for name in list(MODELS) + list(BASELINES):
            m = err_metrics(d[c].values, d['pred_' + name].values)
            rows.append(dict(kind='sensitivity', method=name, lag=L, mae=m['mae'], bias=m['bias'], arms_70_100=m['arms_70_100'], auroc_lt90=m['auroc_lt90'], n=m['n']))
    return rows


def analyze(out_dir, res_dir, B=B_DEFAULT):
    df = pd.read_parquet(os.path.join(out_dir, 'preds.parquet'))
    lagj = json.load(open(os.path.join(out_dir, 'lag.json')))
    par = json.load(open(os.path.join(out_dir, 'baseline_params.json')))
    people = pd.read_parquet(os.path.join(out_dir, 'people.parquet'))
    allp = pd.read_parquet(os.path.join(out_dir, 'all_people.parquet'))
    draws = pd.read_parquet(os.path.join(out_dir, 'draws_all.parquet'))
    cuts = ita_cuts(people.ita.values)
    hold_people = people[people.held_out]
    dfg = group_frame(df, people, cuts)
    ov, pr, sk, gp = [], [], [], []
    for label, ycol in (('lag_star', 'y_lagstar'), ('lag_0', 'y_lag0')):
        ov += overall_rows(dfg, label, ycol, B)
        pr += paired_rows(dfg, label, ycol, B)
        r, g = skin_rows(dfg, label, ycol, B)
        sk += r; gp += g
    dd = draw_level(dfg, B)
    ddg = group_frame(dd, people, cuts)
    ddg['y_draw'] = ddg.sao2_draw
    ov += overall_rows(ddg, 'abg_draws', 'y_draw', B)
    pr += paired_rows(ddg, 'abg_draws', 'y_draw', B)
    r, g = skin_rows(ddg, 'abg_draws', 'y_draw', B)
    sk += r; gp += g
    cb, cg = contact_bias(draws, people, cuts, B)
    ra = ror_all_people(allp, people, cuts, B)
    # contact display against SaO2 on the held-out windows (device reference), lag-star label
    dev = dfg[np.isfinite(dfg.y_lagstar.values) & np.isfinite(dfg.spo2_ref.values)]
    devm = err_metrics(dev.y_lagstar.values, dev.spo2_ref.values)
    dv = draws[np.isfinite(draws.sao2.values)].copy()
    dv = dv.sort_values(['rec', 't0'])
    da = dv[dv.aligned.astype(bool)]

    def episodes(g):
        low = (g.sao2.values < 90).astype(int)
        return int(((low[1:] == 1) & (low[:-1] == 0)).sum() + (low[:1] == 1).sum()) if len(low) else 0
    ep_all = dv.groupby('rec').apply(episodes, include_groups=False)
    ep_al = da.groupby('rec').apply(episodes, include_groups=False)
    hd = dv[dv.subject.isin(set(people[people.held_out].subject))]
    counts = dict(draws_in_aligned_recordings=int(len(da)), draw_people_aligned=int(da.subject.nunique()), draw_encounters_aligned=int(da.rec.nunique()), draws_lt90_all=int((dv.sao2 < 90).sum()),
                  draws_lt90_aligned=int((da.sao2 < 90).sum()), episodes_lt90_all=int(ep_all.sum()), episodes_lt90_aligned=int(ep_al.sum()), people_with_lt90_all=int(dv[dv.sao2 < 90].subject.nunique()),
                  people_with_lt90_aligned=int(da[da.sao2 < 90].subject.nunique()), heldout_draw_people=int(hd[hd.aligned.astype(bool)].subject.nunique()), heldout_draws_aligned=int(hd.aligned.astype(bool).sum()),
                  heldout_draws_lt90_aligned=int(((hd.sao2 < 90) & hd.aligned.astype(bool)).sum()), heldout_people=int(len(hold_people)), heldout_people_with_monk=int(hold_people.monk.notna().sum()), heldout_people_scored=int(df.subject.nunique()),
                  heldout_people_with_sao2=int(df[df.has_sao2].subject.nunique()), windows_sao2=int(df.has_sao2.sum()), windows_draw=int(df.has_sao2_draw.sum()), draws_scored=int(len(dd)),
                  draws_in_data=int(draws.sao2.notna().sum()), draws_with_device_reading=int((draws.sao2.notna() & draws.spo2_dev.notna()).sum()), draw_encounters=int(draws[draws.sao2.notna()].rec.nunique()),
                  draw_people=int(draws[draws.sao2.notna()].subject.nunique()), openox_people=int(len(people)), aligned_people_all=int(allp.subject.nunique()), lag_star=lagj['lag_star'],
                  ita_cut_dark_mid=cuts[0], ita_cut_mid_light=cuts[1], train_people=int(par['n_train_people']), train_windows=int(par['n_train']), train_mean=par['train_mean'],
                  ror_cal_a=par['a'], ror_cal_b=par['b'], device_display_mae=devm['mae'], device_display_bias=devm['bias'], device_display_arms_70_100=devm['arms_70_100'],
                  device_display_auroc_lt90=devm['auroc_lt90'], device_display_n=devm['n'])
    os.makedirs(res_dir, exist_ok=True)
    for name, rows in (('overall', ov), ('paired', pr), ('skin', sk), ('gaps', gp), ('contact_bias', cb), ('contact_bias_gaps', cg), ('ror_all_people', ra), ('lag', lag_rows(df, lagj))):
        pd.DataFrame(rows).to_csv(os.path.join(res_dir, f'openox_spo2_{name}.csv'), index=False)
    pd.DataFrame([counts]).to_csv(os.path.join(res_dir, 'openox_spo2_counts.csv'), index=False)
    print('wrote aggregates to', res_dir, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    i = sp.add_parser('infer'); i.add_argument('--ckpt-dir', required=True); i.add_argument('--out', required=True)
    a_ = sp.add_parser('analyze'); a_.add_argument('--out', required=True); a_.add_argument('--res', required=True); a_.add_argument('--B', type=int, default=B_DEFAULT)
    r = sp.add_parser('report'); r.add_argument('--res', default='../results/ml')
    a = ap.parse_args(argv)
    if a.cmd == 'infer':
        infer(a.ckpt_dir, a.out)
    elif a.cmd == 'analyze':
        analyze(a.out, a.res, a.B)
    else:
        from .openox_spo2_md import report
        report(a.res)


if __name__ == '__main__':
    main()
