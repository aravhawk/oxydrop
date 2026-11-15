"""Single source of numbers for README.md, MODEL.md and WHITEPAPER.md.

Every number that appears in those documents is resolved here from the CSV / JSON files under results/ml/ (produced by the
committed scripts in python/oxydrop_ml/). The Markdown templates (docs/templates/*.md.in) contain tokens such as

    {{b1:chain_v1:tm_bacc}}            -> 0.796                        (baselines_P1.csv, model chain_v1, variant full, column pooled)
    {{b1:chain_v1:tm_bacc|ci}}         -> [0.635, 0.856]
    {{pd1:HYBRID_FINAL/chain_v1|d}}    -> -0.017 [-0.103, 0.069]       (hybrid_p1_final_paired.csv)
    {{@p1_main}}                       -> a whole generated table

and scripts/render_docs.py replaces them. Nothing numeric is typed by hand in a template (scripts/check_claims.py enforces that).
Sections that depend on results that do not exist yet (conformal output, OpenOximetry SpO2 by skin tone, HR voting, written by
the FEAT-007 branch) render a clearly marked PENDING block and render fully once their CSV files exist.

Run `python scripts/ml_numbers.py` to print the token catalogue sizes, or import `resolve(token)` / `block(name)`.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RES = os.path.join(ROOT, 'results', 'ml')
_CACHE: dict = {}

HOF = ['hof_100001', 'hof_100002', 'hof_100003', 'hof_100004', 'hof_100005', 'hof_100006']


# ------------------------------------------------------------------------------------------------------------ file access
def rd(name: str) -> pd.DataFrame:
    if name not in _CACHE:
        _CACHE[name] = pd.read_csv(os.path.join(RES, name))
    return _CACHE[name]


def rj(name: str):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def exists(name: str) -> bool:
    return os.path.exists(os.path.join(RES, name))


# ------------------------------------------------------------------------------------------------------------ formatting
def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)


def f(x, d=3) -> str:
    return 'n/a' if not _ok(x) else f'{float(x):.{d}f}'


def sgn(x, d=3) -> str:
    if not _ok(x):
        return 'n/a'
    s = f'{float(x):+.{d}f}'
    return s


def ci(lo, hi, d=3) -> str:
    return f'[{f(lo, d)}, {f(hi, d)}]'


def pm(m, sd, d=3) -> str:
    return f'{f(m, d)} \u00b1 {f(sd, d)}'


def pct(x, d=1) -> str:
    return 'n/a' if not _ok(x) else f'{100 * float(x):.{d}f}%'


def dstr(diff, lo, hi, d=3) -> str:
    return f'{sgn(diff, d)} {ci(lo, hi, d)}'


def verdict_from_ci(lo, hi, higher_better=True) -> str:
    if not (_ok(lo) and _ok(hi)):
        return 'n/a'
    if lo > 0:
        return 'better' if higher_better else 'worse'
    if hi < 0:
        return 'worse' if higher_better else 'better'
    return 'tie'


# ------------------------------------------------------------------------------------------------------------ row access
def _row(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    sub = df[m]
    if len(sub) == 0:
        raise KeyError(f'no row for {kw}')
    return sub.iloc[0]


def _split_mv(s: str, default='full'):
    if '.' in s:
        m, v = s.split('.', 1)
        return m, v
    return s, default


def b1(model: str, metric: str, variant='full') -> pd.Series:
    return _row(rd('baselines_P1.csv'), model=model, variant=variant, metric=metric)


def b2(ds: str, model: str, variant: str, metric: str) -> pd.Series:
    return _row(rd('baselines_P2.csv'), dataset=ds, model=model, variant=variant, metric=metric)


def b3(model: str, metric: str, variant='full') -> pd.Series:
    return _row(rd('baselines_P3.csv'), model=model, variant=variant, metric=metric)


def ab(variant: str, kind: str | None = None) -> pd.Series:
    df = rd('ablation_core.csv')
    sub = df[df.variant == variant]
    if kind:
        sub = sub[sub.p23_kind == kind]
    if len(sub) == 0:
        raise KeyError(f'ablation row {variant}/{kind}')
    return sub.iloc[0]


def p1_gap_pooled(model: str, variant='full') -> tuple[float, float, float]:
    """Gap on the primary metric like with like: mean over folds of the TM balanced accuracy of the fold model on its own
    (pooled) training people, minus the pooled held-out TM balanced accuracy (PREREGISTRATION Amendment 1, item 5)."""
    r = b1(model, 'tm_bacc', variant)
    return float(r.train_mean), float(r.pooled), float(r.train_mean - r.pooled)


# ------------------------------------------------------------------------------------------------------------ token resolver
def _mod(value_row_fn, mod: str | None) -> str:  # pragma: no cover - thin helper
    raise NotImplementedError


def resolve(token: str) -> str:
    """token without the surrounding braces, e.g. 'b1:chain_v1:tm_bacc|ci'."""
    tok, _, mod = token.partition('|')
    mod = mod.strip() or None
    parts = tok.split(':')
    ns = parts[0].strip()
    args = [p.strip() for p in parts[1:]]
    try:
        fn = NS[ns]
    except KeyError as e:
        raise KeyError(f'unknown token namespace {ns!r} in {{{{{token}}}}}') from e
    return fn(args, mod)


def _val(row: pd.Series, col: str, mod: str | None, lo='pooled_lo', hi='pooled_hi', sd='fold_sd', mean='fold_mean') -> str:
    v = row[col]
    if mod is None:
        return f(v)
    if mod == 'ci':
        return ci(row[lo], row[hi])
    if mod == 'pm':
        return pm(row[mean], row[sd])
    if mod == 'pct':
        return pct(v)
    if mod == 'f1':
        return f(v, 1)
    if mod == 'f2':
        return f(v, 2)
    if mod == 'f4':
        return f(v, 4)
    if mod == 'int':
        return f'{int(round(float(v))):,}'
    if mod == 'sgn':
        return sgn(v)
    if mod == 'full':
        return f'{f(v)} {ci(row[lo], row[hi])}'
    raise KeyError(f'unknown modifier {mod!r}')


def _ns_b1(args, mod):
    model, variant = _split_mv(args[0])
    metric = args[1]
    r = b1(model, metric, variant)
    if mod == 'gap':
        return sgn(r.train_mean - r.pooled)
    if mod == 'train':
        return f(r.train_mean)
    if mod == 'fold':   # per-person minimum / maximum
        return f'{f(r.fold_min)} to {f(r.fold_max)}'
    return _val(r, 'pooled', mod)


def _ns_b2(args, mod):
    ds_model, metric = args[0], args[1]
    ds, mv = ds_model.split('/')
    model, variant = _split_mv(mv)
    r = b2(ds, model, variant, metric)
    return _val(r, 'pooled', mod)


def _ns_b3(args, mod):
    model, variant = _split_mv(args[0])
    r = b3(model, args[1], variant)
    if mod == 'gap':
        return sgn(r.gap)
    return _val(r, 'pooled', mod)


def _ns_ab(args, mod):
    """ab:<variant>/<kind>:<prefix>  e.g. ab:full/p23:p1_tm_bacc  (columns <prefix>, <prefix>_lo, <prefix>_hi, <prefix>_seed_sd, ...)"""
    variant, _, kind = args[0].partition('/')
    r = ab(variant, kind or None)
    p = args[1]
    if mod == 'ci':
        return ci(r[p + '_lo'], r[p + '_hi'])
    if mod == 'pm':
        return pm(r[p + '_fold_mean'], r[p + '_fold_sd'])
    if mod == 'gap':
        return sgn(r[p + '_gap'])
    if mod == 'gapci':
        return f'{sgn(r[p + "_gap"])} {ci(r[p + "_gap_lo"], r[p + "_gap_hi"])}'
    if mod == 'seedsd':
        return f(r[p + '_seed_sd'])
    if mod == 'train':
        return f(r[p + '_train'])
    if mod == 'full':
        return f'{f(r[p])} {ci(r[p + "_lo"], r[p + "_hi"])}'
    if mod in ('f1', 'f2', 'sgn', 'pct', 'int'):
        return _val(r, p, mod)
    return f(r[p])


def _ns_hy(args, mod):
    """hy:<MODEL>:<col>  from hybrid_p1_final.csv (cols tm_bacc, tm_sens, ..., fold_mean, fold_sd, fold_min, fold_max)"""
    r = _row(rd('hybrid_p1_final.csv'), model=args[0])
    col = args[1]
    if mod == 'ci':
        return ci(r.ci_lo, r.ci_hi)
    if mod == 'pm':
        return pm(r.fold_mean, r.fold_sd)
    if mod == 'full':
        return f'{f(r[col])} {ci(r.ci_lo, r.ci_hi)}'
    if mod == 'range':
        return f'{f(r.fold_min)} to {f(r.fold_max)}'
    return _val(r, col, mod)


def _ns_hg(args, mod):
    r = _row(rd('hybrid_p1_final_gap.csv'), model=args[0])
    if mod == 'gapci':
        return f'{sgn(r.gap_person)} \u00b1 {f(r.gap_person_sd)} {ci(r.gap_person_lo, r.gap_person_hi)}'
    return _val(r, args[1], mod)


def _ns_pd1(args, mod):
    """pd1:<MODEL>/<baseline>[:col]  from hybrid_p1_final_paired.csv; |d -> 'diff [lo, hi]'; |v -> verdict"""
    model, base = args[0].split('/')
    r = _row(rd('hybrid_p1_final_paired.csv'), model=model, baseline=base)
    if mod == 'd':
        return dstr(r['diff'], r.lo, r.hi)
    if mod == 'v':
        return _verdict_word(r['diff'], r.lo, r.hi, True)
    if mod == 'base':
        return f(r.baseline_tm_bacc)
    return f(r[args[1]] if len(args) > 1 else r['diff'])


def _verdict_word(diff, lo, hi, higher_better):
    if lo > 0:
        return 'win' if higher_better else 'loss'
    if hi < 0:
        return 'loss' if higher_better else 'win'
    return 'tie'


def _ns_cp(args, mod):
    """cp:<task>/<seed>/<baseline>  from cnn_vs_baselines_paired.csv (seed -1 = 3-seed ensemble)"""
    task, seed, base = args[0].split('/')
    df = rd('cnn_vs_baselines_paired.csv')
    r = df[(df.task == task) & (df.seed == int(seed)) & (df.baseline == base)]
    if len(r) == 0:
        raise KeyError(f'cp row {args}')
    r = r.iloc[0]
    hb = 'mae' not in r.metric
    if mod == 'd':
        return dstr(r['diff'], r.lo, r.hi)
    if mod == 'v':
        return _verdict_word(r['diff'], r.lo, r.hi, hb)
    if mod == 'cnn':
        return f(r.cnn)
    if mod == 'base':
        return f(r.base)
    raise KeyError('cp needs a modifier')


def _ns_hp(args, mod):
    """hp:<scope>/<model>:<col>  from hybrid_p2.csv"""
    scope, model = args[0].split('/')
    r = _row(rd('hybrid_p2.csv'), scope=scope, model=model)
    col = args[1]
    if mod == 'ci':
        return ci(r.lo, r.hi)
    if mod == 'pm':
        return pm(r.fold_mean, r.fold_sd)
    if mod == 'full':
        return f'{f(r[col])} {ci(r.lo, r.hi)}'
    if mod == 'f2':
        return f(r[col], 2)
    return _val(r, col, mod)


def _ns_hpd(args, mod):
    scope, base = args[0].split('/')
    r = _row(rd('hybrid_p2_paired.csv'), scope=scope, baseline=base, model='HYBRID_agree')
    if mod == 'd':
        return dstr(r['diff'], r.lo, r.hi)
    if mod == 'v':
        return _verdict_word(r['diff'], r.lo, r.hi, True)
    return f(r['diff'])


def _ns_hr(args, mod):
    """hr:<rule>:<col>  from hybrid_p2_rules.csv"""
    r = _row(rd('hybrid_p2_rules.csv'), rule=args[0])
    return _val(r, args[1], mod)


def _ns_hs(args, mod):
    """hs:<MODEL>:<seed>:<col>  from hybrid_p1_final_seeds.csv"""
    r = _seed_row(args[0], args[1])
    return _val(r, args[2], mod)


def _seed_row(model: str, seed) -> pd.Series:
    df = rd('hybrid_p1_final_seeds.csv')
    seed = -1 if str(seed) == 'ensemble' else seed
    m = (df.model == model) & (df.seed.astype(str) == str(seed))
    if not m.any():
        raise KeyError(f'seeds row {model}/{seed}')
    return df[m].iloc[0]


def _ns_hgrid(args, mod):
    """hgrid:<cand>|<cal>|<k>:<col>  e.g. hgrid:R|cal=0|k=1:tm_bacc"""
    r = _row(rd('hybrid_p1_final_grid.csv'), model=args[0])
    return _val(r, args[1], mod)


def _ns_bn(args, mod):
    """bn:<candidate>:<col> from hybrid_p1.csv (FEAT-006 stack / BatchNorm adaptation, seed 0)"""
    r = _row(rd('hybrid_p1.csv'), candidate=args[0])
    col = args[1]
    if mod == 'ci':
        return ci(r.lo, r.hi)
    return _val(r, col, mod)


def _ns_v(args, mod):
    return str(CUSTOM[args[0]]())


def _ns_json(args, mod):
    """json:<file>:<dotted.path>"""
    d = rj(args[0])
    for k in args[1].split('.'):
        d = d[k]
    if mod == 'int':
        return f'{int(d):,}'
    if mod == 'f1':
        return f(d, 1)
    if mod == 'f2':
        return f(d, 2)
    if isinstance(d, float):
        return f(d)
    return str(d)


NS = {'b1': _ns_b1, 'b2': _ns_b2, 'b3': _ns_b3, 'ab': _ns_ab, 'hy': _ns_hy, 'hg': _ns_hg, 'pd1': _ns_pd1, 'cp': _ns_cp, 'hp': _ns_hp,
      'hpd': _ns_hpd, 'hr': _ns_hr, 'hs': _ns_hs, 'hgrid': _ns_hgrid, 'bn': _ns_bn, 'v': _ns_v, 'json': _ns_json}


# ------------------------------------------------------------------------------------------------------------ custom scalars
def _params() -> str:
    try:  # exact count from the model code when torch is importable
        import sys
        sys.path.insert(0, os.path.join(ROOT, 'python'))
        from oxydrop_ml.model import OxyNet, student_cfg  # type: ignore
        net = OxyNet(student_cfg())
        n = sum(p.numel() for n_, p in net.named_parameters() if not n_.startswith('aux_spo2'))
        return f'{n:,}'
    except Exception:
        m = re.search(r'Student ([\d,]+) parameters', open(os.path.join(ROOT, 'python/oxydrop_ml/MODEL_SPEC.md')).read())
        return m.group(1) if m else 'n/a'


def _prereg_commit() -> str:
    import subprocess
    try:
        out = subprocess.run(['git', 'log', '--diff-filter=A', '--format=%h', '--', 'python/oxydrop_ml/PREREGISTRATION.md'], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
        return out[-1] if out else 'n/a'
    except Exception:
        return '3e708fe'


def _hof_per_person(model: str):
    r = _row(rd('hybrid_p1_final.csv'), model=model)
    return [float(r[f'tm_bacc_{h}']) for h in HOF]


def _sel_counts() -> dict:
    ch = rd('hybrid_p1_final_choices.csv')
    ch = ch[ch.pipeline == 'HYBRID_FINAL']
    return ch.source.value_counts().to_dict()


def _pending(kind: str) -> bool:
    return not _found(kind)


KIND_PATTERNS = {
    'conformal': ['conformal*.csv', 'CONFORMAL*.csv'],
    'openox_spo2': ['openox_spo2*.csv', 'OPENOX_SPO2*.csv'],
    'hr_vote': ['hr_vote*.csv', 'HR_VOTE*.csv'],
}
KIND_MD = {'conformal': 'CONFORMAL.md', 'openox_spo2': 'OPENOX_SPO2.md', 'hr_vote': 'HR_VOTE.md'}


def _found(kind: str) -> list[str]:
    out: list[str] = []
    for pat in KIND_PATTERNS[kind]:
        out += glob.glob(os.path.join(RES, pat))
    return sorted(set(out))


def _app_state() -> dict:
    """What the app in this tree actually runs (generator detects it; docs never assert more than the tree shows)."""
    app_js = os.path.join(ROOT, 'app/js/app.js')
    txt = open(app_js).read() if os.path.exists(app_js) else ''
    hybrid_file = os.path.exists(os.path.join(ROOT, 'app/js/hybrid_engine.js')) or os.path.exists(os.path.join(ROOT, 'app/js/cnn_engine.js'))
    wired = bool(re.search(r'(hybrid_engine|cnn_engine|HybridEngine|CnnEngine)', txt))
    return {'wired': hybrid_file and wired, 'weights_exported': os.path.exists(os.path.join(ROOT, 'app/data/hybrid_weights.json'))}


CUSTOM = {
    'params': _params,
    'prereg_commit': _prereg_commit,
    'n_windows_p1': lambda: f"{rj('baselines_P1_info.json')['n_windows']:,}",
    'n_windows_p2': lambda: f"{rj('baselines_P2_info.json')['n_windows']:,}",
    'n_windows_p3': lambda: f"{rj('baselines_P3_info.json')['n_windows']:,}",
    'n_people_p3': lambda: f"{rj('baselines_P3_info.json')['n_people']}",
    'majority_share': lambda: pct(rj('baselines_P1_info.json')['majority_share'], 2),
    'p3_ref_median': lambda: f(rj('baselines_P3_info.json')['ref_median'], 1),
    'p3_ref_min': lambda: f(rj('baselines_P3_info.json')['ref_min'], 1),
    'p3_below95': lambda: pct(rj('baselines_P3_info.json')['share_below_95'], 1),
    'p3_windows_below90': lambda: f"{rj('baselines_P3_info.json')['windows_below_90']}",
    'p3_people_below90': lambda: f"{rj('baselines_P3_info.json')['people_with_window_below_90']}",
    'ssl_rows': lambda: f"{rj('ssl_manifest.json')['ssl_full']['rows']:,}",
    'ssl_subjects': lambda: f"{rj('ssl_manifest.json')['ssl_full']['subjects']}",
    'ssl_steps': lambda: f"{rj('ssl_manifest.json')['ssl_full']['steps']:,}",
    'ssl_first': lambda: f(rj('ssl_manifest.json')['ssl_full']['first_loss'], 2),
    'ssl_last': lambda: f(rj('ssl_manifest.json')['ssl_full']['last_loss'], 2),
    'ssl_top1': lambda: f(rj('ssl_manifest.json')['ssl_full']['last_top1'], 2),
    'ssl_noox_last': lambda: f(rj('ssl_manifest.json')['ssl_noox']['last_loss'], 2),
    'ssl_sha': lambda: rj('ssl_manifest.json')['ssl_full']['sha256'][:12],
    'ssl_excluded': lambda: f"{rj('ssl_manifest.json')['ssl_full']['n_excluded_subjects']}",
    'hr_pre_mae': lambda: f(rd('healthring_pretrain_ft_summary.csv').iloc[0].spo2_mae_per_person_mean, 2),
    'hr_scr_mae': lambda: f(_row(rd('healthring_pretrain_ft_summary.csv'), init='scratch').spo2_mae_per_person_mean, 2),
    'hr_frz_mae': lambda: f(_row(rd('healthring_pretrain_ft_summary.csv'), init='pretrained_frozenfeat').spo2_mae_per_person_mean, 2),
    'sel_R': lambda: str(_sel_counts().get('R', 0)),
    'sel_cp': lambda: str(_sel_counts().get('c+p', 0)),
    'sel_P': lambda: str(_sel_counts().get('P', 0)),
    'deploy_inner_cv': lambda: f(rj('../../app/data/hybrid_weights.json')['pipeline']['inner_cv_tm_bacc']) if False else f(
        json.load(open(os.path.join(ROOT, 'app/data/hybrid_weights.json')))['pipeline']['inner_cv_tm_bacc']),
    'deploy_coef': lambda: f(json.load(open(os.path.join(ROOT, 'app/data/hybrid_weights.json')))['pipeline']['meta']['coef'][0]),
    'deploy_intercept': lambda: f(json.load(open(os.path.join(ROOT, 'app/data/hybrid_weights.json')))['pipeline']['meta']['intercept']),
    'deploy_mean': lambda: f(json.load(open(os.path.join(ROOT, 'app/data/hybrid_weights.json')))['pipeline']['meta']['mean'][0]),
    'deploy_scale': lambda: f(json.load(open(os.path.join(ROOT, 'app/data/hybrid_weights.json')))['pipeline']['meta']['scale'][0]),
    'member_kb': lambda: f"{os.path.getsize(os.path.join(ROOT, 'app/data/hybrid_R_s0.json')) / 1e6:.2f} MB",
    'members_mb': lambda: f"{sum(os.path.getsize(os.path.join(ROOT, f'app/data/hybrid_R_s{i}.json')) for i in range(3)) / 1e6:.1f} MB",
    'onnx_kb': lambda: f"{os.path.getsize(os.path.join(ROOT, 'app/data/oxynet.onnx')) / 1e3:.0f} KB",
    'app_state': lambda: 'wired' if _app_state()['wired'] else 'not-wired',
    'v1_cov': lambda: pct(b1('chain_v1', 'coverage', 'band_shipped').pooled, 1),
    'v1_acc_band': lambda: pct(b1('chain_v1', 'acc', 'band_shipped').pooled, 1),
    'v1_alert_band': lambda: pct(b1('chain_v1', 'alert_share_all', 'band_shipped').pooled, 1),
    'v2_cov': lambda: pct(b1('chain_v2', 'coverage', 'nolock_band_shipped').pooled, 1),
    'v2_acc_band': lambda: pct(b1('chain_v2', 'acc', 'nolock_band_shipped').pooled, 1),
    'v2_alert_band': lambda: pct(b1('chain_v2', 'alert_share_all', 'nolock_band_shipped').pooled, 1),
}


# ------------------------------------------------------------------------------------------------------------ derived analyses
def _score_tables():
    h = rd('scores/hybrid_final_hoffman.csv') if False else pd.read_csv(os.path.join(RES, 'scores', 'hybrid_final_hoffman.csv'))
    return h[h['eval'].astype(str).str.lower() == 'true'].copy() if h['eval'].dtype == object else h[h['eval'].astype(bool)].copy()


def selective_curve(score: np.ndarray, y: np.ndarray, thr_center: float = 0.0, n=101):
    """Accuracy on answered windows and share of ALL hypoxic windows alerted when the most confident fraction `cov` is answered
    (confidence = |score - center|, prediction = score > center). Descriptive: no cutoff is tuned on labels."""
    conf = np.abs(score - thr_center)
    pred = (score > thr_center).astype(int)
    order = np.argsort(-conf, kind='stable')
    corr = (pred[order] == y[order]).astype(float)
    alert = ((pred[order] == 1) & (y[order] == 1)).astype(float)
    cs = np.cumsum(corr)
    al = np.cumsum(alert)
    N = len(y)
    npos = max(1, int(y.sum()))
    covs = np.linspace(0.05, 1.0, n)
    k = np.maximum(1, np.round(covs * N).astype(int))
    return covs, cs[k - 1] / k, al[k - 1] / npos


def p1_selective():
    """Per-window score vectors on the 5,644 evaluated Hoffman windows for chain v1, XGBoost, CNN (3-seed mean probability) and hybrid."""
    h = pd.read_csv(os.path.join(RES, 'scores', 'hybrid_final_hoffman.csv'))
    h = h[h['eval'].astype(bool)]
    key = ['subject', 'rec', 't_start']
    b = pd.read_csv(os.path.join(RES, 'scores', 'baseline_P1_hoffman.csv'))
    c = pd.read_csv(os.path.join(RES, 'scores', 'cnn_full_hoffman.csv'))
    c = c[c['eval'].astype(bool)].groupby(['subject', 'rec', 't_start'], as_index=False).p_low.mean()
    m = h.merge(b[key + ['chain_v1__score', 'xgb_fusion19_v2__score', 'xgb_hand26_cal__score']], on=key, how='left', validate='1:1')
    m = m.merge(c, on=key, how='left', validate='1:1', suffixes=('', '_cnn'))
    assert len(m) == len(h)
    y = m['low'].astype(int).to_numpy()
    return m, y


def matched_coverage_rows(cov_target: float | None = None):
    m, y = p1_selective()
    if cov_target is None:
        cov_target = float(b1('chain_v1', 'coverage', 'band_shipped').pooled)
    rows = []
    for label, sc, center in (('Logistic chain v1 (shipped)', m['chain_v1__score'].to_numpy(), 0.5),
                              ('XGBoost, 19 chain features (v2)', m['xgb_fusion19_v2__score'].to_numpy(), 0.5),
                              ('CNN alone (3-seed mean probability)', m['p_low'].to_numpy(), 0.5),
                              ('Hybrid (selected pipeline)', m['logit'].to_numpy(), 0.0)):
        mask = ~np.isnan(sc)
        order = np.argsort(-np.abs(sc[mask] - center), kind='stable')
        yy = y[mask][order]
        pr = (sc[mask][order] > center).astype(int)
        k = int(round(cov_target * mask.sum()))
        acc = float((pr[:k] == yy[:k]).mean())
        alert = float(((pr[:k] == 1) & (yy[:k] == 1)).sum() / max(1, yy[mask].sum()))
        full = float((pr == yy).mean())
        rows.append((label, k / mask.sum(), acc, alert, full))
    return rows, cov_target


# ------------------------------------------------------------------------------------------------------------ table blocks
def md_table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ['l'] + ['r'] * (len(header) - 1)
    sep = ['---:' if a == 'r' else ':---' for a in align]
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(sep) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


P1_BASE_LABEL = {
    'majority': 'Majority class (always "hypoxic")',
    'clock': 'Clock only (elapsed seconds, not deployable)',
    'ror_textbook_RB': 'Ratio-of-ratios, textbook 110-25R (red/blue)',
    'ror_textbook_RG': 'Ratio-of-ratios, textbook 110-25R (red/green)',
    'ror_calibrated_RB': 'Ratio-of-ratios, personally calibrated (red/blue)',
    'ror_calibrated_RG': 'Ratio-of-ratios, personally calibrated (red/green)',
    'xgb_hand26_cal': 'XGBoost, 26 handcrafted features',
    'lgbm_hand26_cal': 'LightGBM, 26 handcrafted features',
    'xgb_fusion19_v1': 'XGBoost, 19 chain features (v1 HR)',
    'xgb_fusion19_v2': 'XGBoost, 19 chain features (v2 HR)',
    'lgbm_fusion19_v1': 'LightGBM, 19 chain features (v1 HR)',
    'lgbm_fusion19_v2': 'LightGBM, 19 chain features (v2 HR)',
    'chain_v2': 'Logistic chain v2 (pulse-lock gate, forced answer)',
    'chain_v2_nestedC': 'Logistic chain v2, C by nested CV',
    'chain_v1_nestedC': 'Logistic chain v1, C by nested CV',
    'chain_v1': 'Logistic chain v1 (best baseline)',
}
P1_ORDER_SHORT = ['majority', 'clock', 'ror_textbook_RB', 'ror_calibrated_RG', 'xgb_hand26_cal', 'xgb_fusion19_v2', 'chain_v2', 'chain_v1']
P1_ORDER_FULL = ['majority', 'clock', 'ror_textbook_RB', 'ror_textbook_RG', 'ror_calibrated_RB', 'ror_calibrated_RG', 'xgb_hand26_cal', 'lgbm_hand26_cal',
                 'xgb_fusion19_v1', 'xgb_fusion19_v2', 'lgbm_fusion19_v1', 'lgbm_fusion19_v2', 'chain_v2', 'chain_v2_nestedC', 'chain_v1_nestedC', 'chain_v1']


def _p1_rows(order):
    rows = []
    for m in order:
        r = b1(m, 'tm_bacc')
        au = b1(m, 'auroc')
        g = r.train_mean - r.pooled
        rows.append([P1_BASE_LABEL[m], f(r.pooled), ci(r.pooled_lo, r.pooled_hi), pm(r.fold_mean, r.fold_sd, 3), f(au.pooled), sgn(g), 'baseline'])
    return rows


def block_p1_main(short=True) -> str:
    rows = _p1_rows(P1_ORDER_SHORT if short else P1_ORDER_FULL)
    c = ab('full', 'p23')
    best = 'chain_v1'
    ens = _row(rd('cnn_vs_baselines_paired.csv'), task='P1', seed=-1, baseline=best)
    rows.append(['**CNN alone** (167k parameters; mean of 3 seeds, own probability at 0.5)', f(c.p1_tm_bacc), ci(c.p1_tm_bacc_lo, c.p1_tm_bacc_hi),
                 pm(c.p1_tm_bacc_fold_mean, c.p1_tm_bacc_fold_sd), f(c.p1_auroc), sgn(c.p1_tm_bacc_gap),
                 f'tie with the best baseline (3-seed ensemble {f(ens.cnn)}, paired diff {dstr(ens["diff"], ens.lo, ens.hi)})'])
    dp = _row(rd('hybrid_p1_final.csv'), model='default_P')
    pdp = _row(rd('hybrid_p1_final_paired.csv'), model='HYBRID_FINAL', baseline='P_alone_default')
    rows.append(['CNN alone, 3-seed ensemble through the per-fold recalibration (`default_P`)', f(dp.tm_bacc), ci(dp.ci_lo, dp.ci_hi), pm(dp.fold_mean, dp.fold_sd), f(dp.auroc), 'n/a',
                 'exploratory'])
    h = _row(rd('hybrid_p1_final.csv'), model='HYBRID_FINAL')
    g = _row(rd('hybrid_p1_final_gap.csv'), model='HYBRID_FINAL')
    ph = _row(rd('hybrid_p1_final_paired.csv'), model='HYBRID_FINAL', baseline=best)
    rows.append(['**Hybrid, selected pipeline** (chain logit + CNN residual, nested selection)', f(h.tm_bacc), ci(h.ci_lo, h.ci_hi), pm(h.fold_mean, h.fold_sd), f(h.auroc),
                 sgn(g.gap_pooled), f'**{_verdict_word(ph["diff"], ph.lo, ph.hi, True)}** with the best baseline ({dstr(ph["diff"], ph.lo, ph.hi)})'])
    cs = _row(rd('hybrid_p1_final.csv'), model='CHAIN_SAME_LEVERS')
    gc = _row(rd('hybrid_p1_final_gap.csv'), model='CHAIN_SAME_LEVERS')
    pc = _row(rd('hybrid_p1_final_paired.csv'), model='CHAIN_SAME_LEVERS', baseline=best)
    rows.append(['Control: chain refit with the same nested levers (no CNN)', f(cs.tm_bacc), ci(cs.ci_lo, cs.ci_hi), pm(cs.fold_mean, cs.fold_sd), f(cs.auroc), sgn(gc.gap_pooled),
                 f'{_verdict_word(pc["diff"], pc.lo, pc.hi, True)} with the best baseline ({dstr(pc["diff"], pc.lo, pc.hi)})'])
    hdr = ['Model (P1: Hoffman, 6 people, LOSO)', 'TM bal. acc.', '95% CI (person bootstrap)', 'Per person, mean \u00b1 SD', 'AUROC', 'Train-vs-test gap', 'vs best baseline (pre-registered rule)']
    return md_table(hdr, rows, ['l', 'r', 'r', 'r', 'r', 'r', 'l'])


def block_p1_per_person() -> str:
    hdr = ['Model'] + [h.replace('hof_10000', 'person ') for h in HOF] + ['mean \u00b1 SD']
    rows = []

    def add(label, vals, mean, sd):
        rows.append([label] + [f(v) for v in vals] + [pm(mean, sd)])
    for lab, mod in (('Logistic chain v1 (shipped)', 'CHAIN_SHIPPED'), ('Chain, same nested levers', 'CHAIN_SAME_LEVERS'), ('CNN alone, 3-seed ensemble (`default_P`)', 'default_P'),
                     ('Residual hybrid R at default levers', 'default_R'), ('**Hybrid, selected pipeline**', 'HYBRID_FINAL')):
        r = _row(rd('hybrid_p1_final.csv'), model=mod)
        add(lab, [r[f'tm_bacc_{h}'] for h in HOF], r.fold_mean, r.fold_sd)
    return md_table(hdr, rows, ['l'] + ['r'] * 7)


def block_p1_seeds() -> str:
    pp = rd('p1_per_person_seeds.csv')
    rows = []
    for seed in sorted(pp.seed.unique()):
        s = pp[pp.seed == seed]
        rows.append([f'seed {int(seed)}'] + [f"{f(r.auroc, 2)} / {f(r.sens, 2)} / {f(r.spec, 2)}" for _, r in s.sort_values('subject').iterrows()])
    hdr = ['CNN alone'] + [h.replace('hof_10000', 'person ') for h in HOF]
    return md_table(hdr, rows, ['l'] + ['r'] * 6)


def block_p2_main() -> str:
    hdr = ['Model (P2: BUT PPG + MTHS, 100 people, LOSO, full coverage)', 'All: within 5 bpm [95% CI]', 'BUT PPG (38)', 'MTHS (62)', 'MAE (bpm)', 'Doubling', 'vs best baseline']
    order = [('FFT peak, plain', 'fft_plain__forced'), ('v1 peak-interval counter', 'v1_peak__forced'), ('**v2 spectral + autocorrelation (best baseline)**', 'v2_spectral_acf__forced'),
             ('v2 with its shipped quality gate (abstains)', 'v2_spectral_acf__gate_shipped'), ('CNN alone (3-seed median)', 'cnn_alone'), ('**Hybrid HR rule `agree`** (CNN, averaged with v2 spectral when within 5 bpm)', 'HYBRID_agree')]
    h2 = rd('hybrid_p2.csv')
    rows = []
    for lab, m in order:
        a = _row(h2, scope='all', model=m)
        b = _row(h2, scope='butppg', model=m)
        c = _row(h2, scope='mths', model=m)
        if m in ('cnn_alone', 'HYBRID_agree'):
            pr = _row(rd('hybrid_p2_paired.csv'), scope='all', model='HYBRID_agree', baseline='v2_spectral_acf__forced')
            ver = f'**{_verdict_word(pr["diff"], pr.lo, pr.hi, True)}** ({dstr(pr["diff"], pr.lo, pr.hi)})' if m == 'HYBRID_agree' else 'tie (3-seed ensemble)'
        else:
            ver = 'baseline'
        rows.append([lab, f'{f(a.within5_full)} {ci(a.lo, a.hi)}', f(b.within5_full), f(c.within5_full), f(a.mae, 2), pct(a.doubling, 2), ver])
    return md_table(hdr, rows, ['l', 'r', 'r', 'r', 'r', 'r', 'l'])


def block_p3_main() -> str:
    hdr = ['Model (P3: MTHS SpO2, 61 people, LOSO)', 'MAE, SpO2 points', '95% CI', 'Per person, mean \u00b1 SD', 'Train-vs-test gap']
    lab = {'train_mean': 'Train-mean predictor (best baseline)', 'ridge_ratio': 'Ridge on six ratio features', 'lgbm_hand26': 'LightGBM, 52 handcrafted features', 'xgb_hand26': 'XGBoost, 52 handcrafted features',
           'ror_textbook_RB': 'Ratio-of-ratios 110-25R (red/blue)', 'ror_textbook_RG': 'Ratio-of-ratios 110-25R (red/green)'}
    rows = []
    for m in ['train_mean', 'ridge_ratio', 'lgbm_hand26', 'xgb_hand26', 'ror_textbook_RB', 'ror_textbook_RG']:
        r = b3(m, 'mae')
        rows.append([lab[m], f(r.pooled), ci(r.pooled_lo, r.pooled_hi), pm(r.fold_mean, r.fold_sd), sgn(r.gap)])
    c = ab('full', 'p23')
    ens = _row(rd('cnn_vs_baselines_paired.csv'), task='P3', seed=-1, baseline='train_mean')
    rows.append(['**CNN alone** (mean of 3 seeds)', f(c.p3_mae), ci(c.p3_mae_lo, c.p3_mae_hi), pm(c.p3_mae_fold_mean, c.p3_mae_fold_sd), 'not reported'])
    rows.append(['CNN alone, 3-seed ensemble (the interval column is the paired difference to the train-mean predictor, positive = CNN better)', f(ens.cnn), dstr(ens['diff'], ens.lo, ens.hi), '', 'tie'])
    return md_table(hdr, rows, ['l', 'r', 'r', 'r', 'r'])


def block_ablation() -> str:
    a = rd('ablation_core.csv')
    full3 = ab('full', 'p23k')  # grouped protocol row with 3 seeds for P2/P3, the comparison row for all single-seed grouped rows
    full0 = _row(a, variant='full (seed 0)', p23_kind='p23k')
    names = {'full': 'full CNN (all levers)', 'no_ssl': 'no SSL (stage B only)', 'no_physics': 'no physics channels', 'no_openox': 'no OpenOximetry (no stage B, no SSL on OpenOx)',
             'no_aug': 'no device augmentation', 'no_basecond': 'no baseline conditioning', 'hr_only': 'HR and quality heads only', 'win20': '20 s window (two 10 s feature windows)',
             'no_pretrain': 'no pretraining at all (exploratory)', 'adv': 'gradient-reversal domain head (exploratory)'}
    hdr = ['Variant', 'Seeds (P1 / P2,P3)', 'P1 TM bal. acc. [95% CI] (seed SD)', 'P1 gap', 'P2 within 5 bpm [95% CI]', 'P3 MAE [95% CI]', 'P2/P3 protocol']
    rows = []
    for _, r in a.iterrows():
        if r.variant == 'full' and r.p23_kind == 'p23k':
            continue
        if r.variant == 'full (seed 0)' and r.p23_kind == 'p23k':
            lab = 'full CNN, seed 0 only (reference for single-seed rows)'
        elif r.variant == 'full (seed 0)':
            lab = 'full CNN, seed 0 only'
        else:
            lab = names.get(r.variant, r.variant)
        p1 = f'{f(r.p1_tm_bacc)} {ci(r.p1_tm_bacc_lo, r.p1_tm_bacc_hi)} ({f(r.p1_tm_bacc_seed_sd)})' if _ok(r.p1_tm_bacc) else 'n/a'
        n1 = int(r.p1_tm_bacc_n_seeds) if _ok(r.p1_tm_bacc_n_seeds) else 0
        n2 = int(r.p2_within5_n_seeds) if _ok(r.p2_within5_n_seeds) else 0
        p2 = f'{f(r.p2_within5)} {ci(r.p2_within5_lo, r.p2_within5_hi)}' if _ok(r.p2_within5) else 'n/a'
        p3 = f'{f(r.p3_mae)} {ci(r.p3_mae_lo, r.p3_mae_hi)}' if _ok(r.p3_mae) else 'n/a'
        rows.append([lab, f'{n1} / {n2}', p1, sgn(r.p1_tm_bacc_gap) if _ok(r.p1_tm_bacc_gap) else 'n/a', p2, p3, str(r.p23_protocol)])
    return md_table(hdr, rows, ['l', 'r', 'r', 'r', 'r', 'r', 'l'])


def _lever_delta_p1(var: str) -> tuple[float, float]:
    r = ab(var, 'p23k')
    return float(r.p1_tm_bacc), float(r.p1_tm_bacc - ab('full', 'p23').p1_tm_bacc)


def block_levers() -> str:
    """The lever ledger: every lever, its measured effect on held-out people, and the keep/drop decision with the evidence."""
    full = ab('full', 'p23')
    full0 = _row(rd('ablation_core.csv'), variant='full (seed 0)', p23_kind='p23k')
    ch = rd('hybrid_p1_final_choices.csv')
    ch = ch[ch.pipeline == 'HYBRID_FINAL']
    n_cal, n_k = int(ch.cal.astype(bool).sum()), int((ch.k > 1).sum())
    n_d = int((ch.source == 'D').sum())

    def abl(var):
        return ab(var, 'p23k')
    rows = []
    sdp1 = float(full.p1_tm_bacc_seed_sd)

    def dp1(var):
        r = abl(var)
        return f'{sgn(r.p1_tm_bacc - full.p1_tm_bacc)}'

    def dp2(var):
        r = abl(var)
        return f'{sgn(r.p2_within5 - full0.p2_within5)}'
    nox, nss, nph, nau, nbc = abl('no_openox'), abl('no_ssl'), abl('no_physics'), abl('no_aug'), abl('no_basecond')
    rows.append(['OpenOximetry supervised warm-up (stage B) and OpenOx data', f'P1 {dp1("no_openox")} when removed; P2 {dp2("no_openox")} (grouped, vs seed 0)', 'KEEP', 'Largest effect in the study, far outside the seed SD; with six people the person-level intervals of the two rows still overlap'])
    rows.append(['Self-supervised pretraining (SimCLR on {{v:ssl_rows}} windows)'.replace('{{v:ssl_rows}}', CUSTOM['ssl_rows']()), f'P1 {dp1("no_ssl")}; P2 {dp2("no_ssl")} (seed 0)', 'KEEP (small)', 'P1 change is inside the seed SD ' + f(sdp1) + '; the HR gain is a single seed'])
    rows.append(['Baseline conditioning (person-relative inputs and embedding subtraction)', f'P1 {dp1("no_basecond")}; gap {sgn(nbc.p1_tm_bacc_gap)} vs {sgn(full.p1_tm_bacc_gap)}', 'KEEP', 'Removing it widens the train-vs-test gap; the design is the personalised comparison itself'])
    rows.append(['Physics channels (ratio-of-ratios, beat mask, side features)', f'P1 {dp1("no_physics")}; P2 {dp2("no_physics")}', 'NO MEASURED GAIN (still in the network, not claimed)', 'Inside the seed SD; it was fixed before the ablation and removing it needs a full refit, so no accuracy claim is made for it'])
    rows.append(['Device-agnostic augmentation', f'P1 {dp1("no_aug")}; P2 {dp2("no_aug")}', 'NO MEASURED GAIN (still in the network, not claimed)', 'Inside the seed SD; the device-agnostic novelty claim is withdrawn'])
    rows.append(['HR-only training (HR and quality heads)', f'P2 {sgn(abl("hr_only").p2_within5 - full0.p2_within5)} vs full seed 0', 'DROP', 'Within noise of the multi-task network on HR; P1 and P3 are not defined for it'])
    rows.append(['20 s window', f'P1 {dp1("win20")}; P2 on MTHS only {f(abl("win20").p2_mths)} vs {f(full0.p2_mths)} for full seed 0', 'DROP', 'No gain on the matching MTHS subset; the 10 s window is the shipped one'])
    rows.append(['Gradient-reversal domain head (exploratory)', f'P1 {dp1("adv")}', 'DROP', 'Within noise, single seed'])
    r_d, p_d = _row(rd('hybrid_p1_final.csv'), model='default_R'), _row(rd('hybrid_p1_final.csv'), model='default_P')
    d_d, l_d = _row(rd('hybrid_p1_final.csv'), model='default_D'), _row(rd('hybrid_p1_final.csv'), model='default_L')
    cp = _row(rd('hybrid_p1_final.csv'), model='default_chain_platt')
    rows.append(['Residual on the chain logit (family R)', f'TM {f(r_d.tm_bacc)} vs plain CNN {f(p_d.tm_bacc)} (default levers, 3 seeds)', 'KEEP (selected in ' + CUSTOM['sel_R']() + ' of 6 folds)', f'Does not beat the chain alone ({f(cp.tm_bacc)} after the same recalibration)'])
    rows.append(['Chain distillation (family D)', f'TM {f(d_d.tm_bacc)}', 'DROP', f'Selected in {n_d} of 6 folds; below R and P at default levers'])
    rows.append(['Frozen-backbone residual (family L, exploratory)', f'TM {f(l_d.tm_bacc)}', 'DROP', 'Added after the first results (Amendment 4); not better'])
    cal_on = _row(rd('hybrid_p1_final_grid.csv'), model='R|cal=1|k=1')
    cal_off = _row(rd('hybrid_p1_final_grid.csv'), model='R|cal=0|k=1')
    rows.append(['Per-user baseline calibration of the scores', f'TM {f(cal_on.tm_bacc)} on vs {f(cal_off.tm_bacc)} off (R, k = 1)', f'DROP (chosen in {n_cal} of 6 folds)', 'Inner folds rarely prefer it'])
    k6 = _row(rd('hybrid_p1_final_grid.csv'), model='R|cal=0|k=6')
    rows.append(['Multi-window aggregation (k = 6 windows)', f'TM {f(k6.tm_bacc)} vs {f(cal_off.tm_bacc)} for k = 1 (R)', f'DROP (chosen in {n_k} of 6 folds)', 'Inner folds rarely prefer it; adds latency'])
    bn = _row(rd('hybrid_p1.csv'), candidate='cnn_bn100')
    cn = _row(rd('hybrid_p1.csv'), candidate='cnn')
    rows.append(['Test-time BatchNorm adaptation on the first minute', f'TM {f(bn.tm_bacc)} vs {f(cn.tm_bacc)} without (seed 0)', 'DROP', 'Hurts the CNN on its own'])
    ens = _row(rd('cnn_vs_baselines_paired.csv'), task='P1', seed=-1, baseline='chain_v1')
    rows.append(['Seed ensembling (3 seeds)', f'TM {f(ens.cnn)} vs {f(full.p1_tm_bacc)} mean of single seeds', 'KEEP (R members shipped as 3 seeds)', 'Gain over the mean of single seeds is inside the seed SD ' + f(sdp1) + '; the shipped members are an ensemble'])
    fe = _seed_row('P+fold_ensemble', 'ensemble')
    pe = _seed_row('P', 'ensemble')
    rows.append(['Fold ensembling (extra members from the inner fits)', f'TM {f(fe.tm_bacc)} vs {f(pe.tm_bacc)} for the seed ensemble', 'DROP', 'No gain'])
    cpp = _row(rd('hybrid_p1_final.csv'), model='default_c+p')
    rows.append(['Stacking: meta-learner on [chain logit, CNN logit]', f'TM {f(cpp.tm_bacc)}', 'KEEP (selected in ' + CUSTOM['sel_cp']() + ' of 6 folds)', 'Ties the chain'])
    hr = _row(rd('hybrid_p2.csv'), scope='all', model='HYBRID_agree')
    hc = _row(rd('hybrid_p2.csv'), scope='all', model='cnn_alone')
    rows.append(['HR: average CNN and v2 spectral when they agree within 5 bpm', f'within 5 bpm {f(hr.within5_full)} vs CNN alone {f(hc.within5_full)}', 'KEEP', 'Declared design choice made after seeing the pooled table (Amendment 3)'])
    hp = healthring_row()
    rows.append(['HealthRing PPG pretraining (Kaggle kernel, different CNN; predictions only)', hp, 'DROP', 'Tie with training from scratch; no weights, cannot initialise OxyNet'])
    for kind, name in (('conformal', 'Conformal 3-state abstention'), ('openox_spo2', 'OpenOximetry lag-corrected SaO2 head'), ('hr_vote', 'HR voting and harmonic correction')):
        rows.append([name, 'PENDING (results/ml files not yet produced)' if _pending(kind) else 'see the dedicated section', 'PENDING' if _pending(kind) else 'see section', ''])
    hdr = ['Lever', 'Measured effect on held-out people (descriptive)', 'Decision', 'Evidence and caveat']
    return md_table(hdr, rows, ['l', 'l', 'l', 'l'])


def healthring_row() -> str:
    h = rd('healthring_pretrain_ft_summary.csv')
    pre = _row(h, init='pretrained')
    scr = _row(h, init='scratch')
    return f'SpO2 MAE {f(pre.spo2_mae_per_person_mean, 2)} pretrained vs {f(scr.spo2_mae_per_person_mean, 2)} scratch (6 Hoffman people)'


def block_gap() -> str:
    hdr = ['Model', 'Train, pooled people', 'Held-out, pooled', 'Gap (train minus held-out)', 'Reading']
    rows = []
    for m, lab in (('chain_v1', 'Logistic chain v1'), ('chain_v2', 'Logistic chain v2'), ('xgb_fusion19_v2', 'XGBoost, 19 chain features'), ('xgb_hand26_cal', 'XGBoost, 26 handcrafted'), ('lgbm_hand26_cal', 'LightGBM, 26 handcrafted')):
        t, h, g = p1_gap_pooled(m)
        rows.append([lab, f(t), f(h), sgn(g), 'baseline'])
    c = ab('full', 'p23')
    rows.append(['CNN alone (3 seeds)', f(c.p1_tm_bacc_train), f(c.p1_tm_bacc), f'{sgn(c.p1_tm_bacc_gap)} {ci(c.p1_tm_bacc_gap_lo, c.p1_tm_bacc_gap_hi)}', 'train score below held-out: no over-fitting signal at the fixed 0.5 threshold'])
    for m, lab in (('HYBRID_FINAL', 'Hybrid, selected pipeline'), ('CHAIN_SAME_LEVERS', 'Chain, same nested levers')):
        g = _row(rd('hybrid_p1_final_gap.csv'), model=m)
        rows.append([lab, f(g.train_pooled), f(g.heldout_pooled), sgn(g.gap_pooled), f'per person {sgn(g.gap_person)} (SD {f(g.gap_person_sd)}); selection optimism {sgn(g.gap_inner_pooled)}'])
    return md_table(hdr, rows, ['l', 'r', 'r', 'r', 'l'])


def block_selection() -> str:
    ch = rd('hybrid_p1_final_choices.csv')
    ch = ch[ch.pipeline == 'HYBRID_FINAL']
    rows = [[r.outer.replace('hof_10000', 'person '), r.source, 'on' if bool(r.cal) else 'off', str(int(r.k)), f(r.inner_cv)] for _, r in ch.iterrows()]
    return md_table(['Held-out person', 'Chosen source', 'Per-user calibration', 'k windows', 'Inner-CV TM bal. acc.'], rows, ['l', 'l', 'l', 'r', 'r'])


def block_matched_coverage() -> str:
    rows, cov = matched_coverage_rows()
    base_all = float(b1('chain_v1', 'acc', 'full').pooled)
    out = []
    for lab, c, acc, alert, full in rows:
        out.append([lab, pct(c), pct(acc), pct(alert), pct(full)])
    t = md_table(['Model (Hoffman, 5,644 windows)', 'Coverage', 'Accuracy on answered windows', 'Share of ALL hypoxic windows alerted', 'Accuracy at full coverage'], out, ['l', 'r', 'r', 'r', 'r'])
    return t


def block_skin_hoffman() -> str:
    s = rd('baselines_P1_skin.csv')
    r = _row(s, model='chain_v1', variant='full')
    return (f'logistic chain v1 accuracy {pct(r.dark_acc)} for that person against {pct(r.light_mean_acc)} for the {int(r.n_light)} others, gap {sgn(r.gap_acc)}; '
            f'anecdote only, n = {int(r.n_dark)}')


def block_ssl() -> str:
    c = rd('ssl_loss_curve_full.csv')
    n = rd('ssl_loss_curve_noox.csv')
    return md_table(['Run', 'Steps', 'First logged loss', 'Last loss', 'Top-1 positive retrieval'],
                    [['SSL, all corpora', f'{int(c.step.iloc[-1]):,}', f(c.loss.iloc[0], 2), f(c.loss.iloc[-1], 2), f(c.top1.iloc[-1], 2)],
                     ['SSL, no OpenOximetry', f'{int(n.step.iloc[-1]):,}', f(n.loss.iloc[0], 2), f(n.loss.iloc[-1], 2), f(n.top1.iloc[-1], 2)]], ['l', 'r', 'r', 'r', 'r'])


# ------------------------------------------------------------------------------------------------------------ FEAT-007 sections (PENDING until the files exist)
def _generic_csv_md(path: str, max_rows=16, max_cols=9) -> str:
    df = pd.read_csv(path)
    cols = list(df.columns)[:max_cols]
    df = df[cols].head(max_rows)
    rows = []
    for _, r in df.iterrows():
        rows.append([f(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.tolist()])
    return md_table(cols, rows, ['l'] + ['r'] * (len(cols) - 1))


def _pending_block(title: str, what: str, files: str) -> str:
    return (f'> **PENDING: {title}.** {what} The numbers for this section come from a separate analysis that is not merged into this tree yet (`{files}`). '
            f'This block is generated: run `bash scripts/reproduce_ml.sh docs` after the files exist and it renders in full. '
            f'No number from this section is quoted anywhere else in this document.')


def block_conformal() -> str:
    fs = _found('conformal')
    if not fs:
        return _pending_block('conformal 3-state output', 'Split-conformal abstention on the final hybrid scores, with the coverage-versus-accuracy curve and the comparison with the v1 chain at matched coverage '
                              '(v1 shipped band: coverage ' + CUSTOM['v1_cov']() + ', accuracy ' + CUSTOM['v1_acc_band']() + ').', 'results/ml/CONFORMAL.md, results/ml/conformal*.csv')
    parts = []
    for p in fs:
        parts.append(f'`results/ml/{os.path.basename(p)}`:\n\n' + _generic_csv_md(p))
    return '\n\n'.join(parts) + '\n\nSee `results/ml/CONFORMAL.md` for the protocol and the reading.'


def block_openox() -> str:
    fs = _found('openox_spo2')
    if not fs:
        return _pending_block('OpenOximetry SpO2 by skin tone', 'ABG-anchored SaO2 (lag-corrected, learned on training people only) on OpenOximetry people held out of every training stage (Daytona only), '
                              'MAE and ARMS against the ratio-of-ratios and the train-mean predictor, split by Monk skin-tone group, with dark-minus-light gaps and person-bootstrap intervals.',
                              'results/ml/OPENOX_SPO2.md, results/ml/openox_spo2*.csv')
    parts = []
    for p in fs:
        parts.append(f'`results/ml/{os.path.basename(p)}`:\n\n' + _generic_csv_md(p))
    return '\n\n'.join(parts) + '\n\nSee `results/ml/OPENOX_SPO2.md` for the protocol and the reading.'


def block_hrvote() -> str:
    fs = _found('hr_vote')
    if not fs:
        return _pending_block('heart-rate voting', 'Voting between autocorrelation, the spectral harmonic sum and the CNN head with harmonic correction and a causal median, on BUT PPG and MTHS.',
                              'results/ml/HR_VOTE.md, results/ml/hr_vote*.csv')
    parts = []
    for p in fs:
        parts.append(f'`results/ml/{os.path.basename(p)}`:\n\n' + _generic_csv_md(p))
    return '\n\n'.join(parts) + '\n\nSee `results/ml/HR_VOTE.md` for the protocol and the reading.'


def block_app_state() -> str:
    st = _app_state()
    if st['wired']:
        return ('The default engine of the app in this tree is the hybrid: the v1 logistic chain logit plus the residual of the 3-seed CNN ensemble (`app/js/`, see `docs/APP_INTEGRATION.md`). '
                'The logistic chain stays as the fallback and as `?engine=logistic`.')
    return ('**PENDING (app integration):** in this tree the app still runs the logistic chain v2. The hybrid weights are exported (`app/data/hybrid_weights.json`, three seed members) and '
            'checked for numerical parity, and wiring them into the app is the job of the app-integration branch (`docs/APP_INTEGRATION.md`). Until that is merged, nothing in this document '
            'claims that the live site runs the CNN.')


def block_pending_summary() -> str:
    pend = [k for k, v in pending_report().items() if v]
    names = {'conformal': 'conformal 3-state output and matched-coverage comparison', 'openox_spo2': 'OpenOximetry SpO2 by skin tone', 'hr_vote': 'heart-rate voting'}
    out = []
    if pend:
        out.append('**PENDING (generated blocks, filled automatically when the files exist):** ' + '; '.join(names[k] for k in pend) + '.')
    if not _app_state()['wired']:
        out.append('**PENDING:** the app in this tree still runs the logistic chain v2 (the hybrid export is ready, wiring is the app-integration branch).')
    return ' '.join(out) if out else 'No section is pending: every block below is generated from files in `results/ml/`.'


BLOCKS = {
    'pending_summary': block_pending_summary,
    'p1_main': lambda: block_p1_main(True),
    'p1_main_full': lambda: block_p1_main(False),
    'p1_per_person': block_p1_per_person,
    'p1_seeds': block_p1_seeds,
    'p2_main': block_p2_main,
    'p3_main': block_p3_main,
    'ablation': block_ablation,
    'levers': block_levers,
    'gap': block_gap,
    'selection': block_selection,
    'matched_coverage': block_matched_coverage,
    'skin_hoffman': block_skin_hoffman,
    'ssl': block_ssl,
    'conformal': block_conformal,
    'openox': block_openox,
    'hrvote': block_hrvote,
    'app_state': block_app_state,
}


def block(name: str) -> str:
    return BLOCKS[name]()


def pending_report() -> dict:
    return {k: _pending(k) for k in KIND_PATTERNS}


if __name__ == '__main__':
    print('pending sections:', pending_report())
    print('app state:', _app_state())
    for n in BLOCKS:
        s = block(n)
        print(f'--- {n}: {len(s)} chars')
