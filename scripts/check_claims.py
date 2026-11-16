"""Check that README.md, MODEL.md, WHITEPAPER.md (and docs/WRITEUP.md) quote only numbers that come from results/ml/*.csv.

    python scripts/check_claims.py            # exit 1 on any failure

Checks
  1. the committed documents equal a fresh render of docs/templates/*.md.in (so every number was produced by scripts/ml_numbers.py);
  2. an INDEPENDENT recomputation of the headline numbers straight from the CSV files (own pandas code, not the template resolver) appears in
     README.md, MODEL.md and WHITEPAPER.md where it is quoted, with the same formatting in all three;
  3. templates contain no hand-typed result numbers: every decimal outside a {{token}}, a code span or a URL must be on a short allowlist of
     literature values and design constants;
  4. every 3-decimal number in the rendered documents is found in some results/ml file (or is a documented difference of two such values);
  5. no em dashes; retired phrases (tests/docs_claims.mjs) absent; WHITEPAPER.pdf has 4 to 6 pages and contains the headline number.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_docs  # noqa: E402
import ml_numbers as N  # noqa: E402

ROOT = N.ROOT
RES = N.RES
PHOTON = N.PHOTON
DOCS = ['README.md', 'MODEL.md', 'WHITEPAPER.md', 'docs/WRITEUP.md']
fails: list[str] = []
warns: list[str] = []


def fail(msg):
    fails.append(msg)
    print('FAIL', msg)


def read(p):
    return open(os.path.join(ROOT, p), encoding='utf-8').read()


def f3(x):
    return f'{float(x):.3f}'


def csv(name):
    return pd.read_csv(os.path.join(RES, name))


# ----------------------------------------------------------------------------------------------------------------------- 1. fresh render
def check_render():
    out, _ = render_docs.render_all()
    for dst, txt in out.items():
        if read(dst) != txt:
            fail(f'{dst} differs from a fresh render: run `python scripts/render_docs.py`')
    return out


# ----------------------------------------------------------------------------------------------------------------------- 2. independent recomputation
def independent_claims() -> list[tuple[str, str, list[str]]]:
    """(label, expected string, documents that must contain it)"""
    allthree = ['README.md', 'MODEL.md', 'WHITEPAPER.md']
    claims = []
    b = csv('baselines_P1.csv')
    r = b[(b.model == 'chain_v1') & (b.variant == 'full') & (b.metric == 'tm_bacc')].iloc[0]
    claims.append(('P1 best baseline chain_v1 TM bacc', f'{f3(r.pooled)} [{f3(r.pooled_lo)}, {f3(r.pooled_hi)}]', allthree))
    r = b[(b.model == 'clock') & (b.variant == 'full') & (b.metric == 'tm_bacc')].iloc[0]
    claims.append(('P1 clock TM bacc', f3(r.pooled), ['MODEL.md', 'WHITEPAPER.md']))
    r = b[(b.model == 'chain_v2') & (b.variant == 'full') & (b.metric == 'tm_bacc')].iloc[0]
    claims.append(('P1 chain_v2 TM bacc', f3(r.pooled), allthree[1:]))
    h = csv('hybrid_p1_final.csv').set_index('model')
    r = h.loc['HYBRID_FINAL']
    claims.append(('P1 hybrid', f'{f3(r.tm_bacc)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]', allthree))
    claims.append(('P1 chain same levers', f3(h.loc['CHAIN_SAME_LEVERS'].tm_bacc), allthree))
    claims.append(('P1 default_R', f3(h.loc['default_R'].tm_bacc), ['MODEL.md']))
    p = csv('hybrid_p1_final_paired.csv')
    r = p[(p.model == 'HYBRID_FINAL') & (p.baseline == 'chain_v1')].iloc[0]
    claims.append(('P1 hybrid minus chain_v1', f'{r["diff"]:+.3f} [{f3(r.lo)}, {f3(r.hi)}]', allthree))
    a = csv('ablation_core.csv')
    r = a[(a.variant == 'full') & (a.p23_kind == 'p23')].iloc[0]
    claims.append(('P1 CNN full (3-seed mean) with CI', f'{f3(r.p1_tm_bacc)}', allthree))
    claims.append(('P1 CNN full CI', f'[{f3(r.p1_tm_bacc_lo)}, {f3(r.p1_tm_bacc_hi)}]', ['MODEL.md', 'WHITEPAPER.md', 'README.md']))
    claims.append(('P3 CNN MAE', f3(r.p3_mae), allthree))
    r2 = a[(a.variant == 'no_openox') & (a.p23_kind == 'p23k')].iloc[0]
    claims.append(('P1 without OpenOx', f3(r2.p1_tm_bacc), allthree))
    r3 = a[(a.variant == 'no_pretrain') & (a.p23_kind == 'p23k')].iloc[0]
    claims.append(('P1 without any pretraining', f3(r3.p1_tm_bacc), ['WHITEPAPER.md']))
    c = csv('cnn_vs_baselines_paired.csv')
    r = c[(c.task == 'P1') & (c.seed == -1) & (c.baseline == 'chain_v1')].iloc[0]
    claims.append(('P1 CNN 3-seed ensemble', f3(r.cnn), allthree))
    claims.append(('P1 CNN ensemble minus chain', f'{r["diff"]:+.3f} [{f3(r.lo)}, {f3(r.hi)}]', ['MODEL.md', 'WHITEPAPER.md']))
    for s in (0, 1, 2):
        rr = c[(c.task == 'P1') & (c.seed == s) & (c.baseline == 'chain_v1')].iloc[0]
        claims.append((f'P1 CNN seed {s}', f3(rr.cnn), ['MODEL.md', 'WHITEPAPER.md']))
    h2 = csv('hybrid_p2.csv')
    r = h2[(h2.scope == 'all') & (h2.model == 'HYBRID_agree')].iloc[0]
    claims.append(('P2 hybrid within 5', f'{f3(r.within5_full)} [{f3(r.lo)}, {f3(r.hi)}]', allthree[1:]))
    claims.append(('P2 hybrid within 5 (README)', f3(r.within5_full), allthree))
    b2 = csv('baselines_P2.csv')
    r = h2[(h2.scope == 'all') & (h2.model == 'v2_spectral_acf__forced')].iloc[0]
    claims.append(('P2 best baseline v2 spectral', f'{f3(r.within5_full)} [{f3(r.lo)}, {f3(r.hi)}]', allthree))
    hp = csv('hybrid_p2_paired.csv')
    r = hp[(hp.scope == 'all') & (hp.model == 'HYBRID_agree') & (hp.baseline == 'v2_spectral_acf__forced')].iloc[0]
    claims.append(('P2 paired difference', f'{r["diff"]:+.3f} [{f3(r.lo)}, {f3(r.hi)}]', allthree))
    b3 = csv('baselines_P3.csv')
    r = b3[(b3.model == 'train_mean') & (b3.metric == 'mae')].iloc[0]
    claims.append(('P3 train-mean MAE', f3(r.pooled), allthree))
    g = csv('hybrid_p1_final_gap.csv').set_index('model')
    claims.append(('hybrid pooled gap', f'{g.loc["HYBRID_FINAL"].gap_pooled:+.3f}', ['MODEL.md', 'WHITEPAPER.md']))
    claims.append(('chain same levers pooled gap', f'{g.loc["CHAIN_SAME_LEVERS"].gap_pooled:+.3f}', ['MODEL.md', 'WHITEPAPER.md']))
    b1 = b[(b.model == 'chain_v1') & (b.variant == 'full') & (b.metric == 'tm_bacc')].iloc[0]
    claims.append(('v1 chain gap on pooled people', f'{b1.train_mean - b1.pooled:+.3f}', ['MODEL.md', 'WHITEPAPER.md']))
    sh = b[(b.model == 'chain_v1') & (b.variant == 'band_shipped')]
    cov = sh[sh.metric == 'coverage'].iloc[0].pooled
    acc = sh[sh.metric == 'acc'].iloc[0].pooled
    claims.append(('v1 shipped band coverage', f'{100 * cov:.1f}%', ['MODEL.md', 'WHITEPAPER.md']))
    claims.append(('v1 shipped band accuracy', f'{100 * acc:.1f}%', ['MODEL.md', 'WHITEPAPER.md']))
    # ---- conformal, OpenOximetry and HR voting (independent code, straight from the CSVs)
    cc = csv('conformal_curve.csv')
    mdwp = ['MODEL.md', 'WHITEPAPER.md']
    for sysname, docs_all in (('deployed_R', allthree), ('final', allthree), ('chain_v1', allthree)):
        r = cc[(cc.system == sysname) & (cc.method == 'mondrian') & ((cc.param - 0.10).abs() < 1e-9)].iloc[0]
        claims.append((f'conformal alpha 0.10 coverage ({sysname})', f'{100 * r.coverage:.1f}%', docs_all))
        claims.append((f'conformal alpha 0.10 accuracy on answered ({sysname})', f'{100 * r.acc:.1f}%', docs_all))
        claims.append((f'conformal alpha 0.10 hypoxic alerted ({sysname})', f'{100 * r.alert_share_all:.1f}%', docs_all))
        claims.append((f'conformal alpha 0.10 coverage with interval ({sysname})', f'{100 * r.coverage:.1f}% [{100 * r.coverage_lo:.1f}%, {100 * r.coverage_hi:.1f}%]', mdwp))
        claims.append((f'conformal alpha 0.10 accuracy with interval ({sysname})', f'{100 * r.acc:.1f}% [{100 * r.acc_lo:.1f}%, {100 * r.acc_hi:.1f}%]', mdwp))
        claims.append((f'conformal alpha 0.10 hypoxic called Normal ({sysname})', f'{100 * r.confident_miss:.1f}%', mdwp + (['README.md'] if sysname == 'deployed_R' else [])))
    cm = csv('conformal_matched.csv')
    for sysname in ('deployed_R', 'chain_v1', 'final'):
        for ref in ('v1', 'v2'):
            r = cm[(cm.system == sysname) & (cm.reference == ref) & (cm.variant == 'conformal_mondrian')].iloc[0]
            if sysname != 'final' or ref == 'v1':
                claims.append((f'matched coverage {ref} {sysname} accuracy', f'{100 * r.acc:.1f}%', allthree if (ref == 'v1' and sysname != 'final') else mdwp))
            claims.append((f'matched coverage {ref} {sysname} difference', f'{100 * r.diff_acc:+.1f} [{100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}]', mdwp + (['README.md'] if (ref == 'v1' and sysname == 'deployed_R') else [])))
    pp = csv('conformal_per_person.csv')
    r = pp[(pp.system == 'deployed_R') & (pp.method == 'mondrian') & ((pp.alpha - 0.1).abs() < 1e-9) & (pp.person == 'hof_100001')].iloc[0]
    claims.append(('person 1 hypoxic called Normal (deployed R)', f'{100 * r.confident_miss:.1f}%', allthree))
    # the shipped pipeline R and the weak-pulse guard
    claims.append(('P1 deployed R with interval', f'{f3(h.loc["default_R"].tm_bacc)} [{f3(h.loc["default_R"].ci_lo)}, {f3(h.loc["default_R"].ci_hi)}]', allthree))
    ood = csv('ood_ac_range.csv')
    for ds in ('mths', 'butppg'):
        claims.append((f'guard refusal share {ds}', f'{100 * ood[(ood.dataset == ds) & (ood.percentile == 0.1)].iloc[0].share_below_cut:.1f}%', allthree))
    hp_ = csv('hybrid_p2_paired.csv')
    r = hp_[(hp_.scope == 'butppg') & (hp_.model == 'HYBRID_agree') & (hp_.baseline == 'v2_spectral_acf__forced')].iloc[0]
    claims.append(('P2 paired difference BUT PPG', f'{r["diff"]:+.3f} [{f3(r.lo)}, {f3(r.hi)}]', allthree))
    rules = csv('hybrid_p2_rules.csv').set_index('rule')
    claims.append(('P2 Hoffman spectral within 5', f3(rules.loc['spec'].within5_full), allthree))
    claims.append(('P2 Hoffman agree within 5', f3(rules.loc['agree'].within5_full), allthree))
    ov = csv('openox_spo2_overall.csv')
    for mdl in ('R', 'ror_textbook'):
        r = ov[(ov.label == 'lag_star') & (ov.model == mdl)].iloc[0]
        claims.append((f'OpenOx SaO2 MAE {mdl}', f'{r.mae:.2f} [{r.mae_lo:.2f}, {r.mae_hi:.2f}]', allthree))
    r = ov[(ov.label == 'lag_star') & (ov.model == 'train_mean')].iloc[0]
    claims.append(('OpenOx SaO2 MAE train mean', f'{r.mae:.2f}', ['README.md', 'MODEL.md', 'WHITEPAPER.md']))
    pr = csv('openox_spo2_paired.csv')
    r = pr[(pr.label == 'lag_star') & (pr.model == 'R') & (pr.baseline == 'ror_textbook')].iloc[0]
    claims.append(('OpenOx CNN minus textbook', f'{r.diff_model_minus_baseline:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]', ['README.md', 'MODEL.md']))
    gp = csv('openox_spo2_gaps.csv')
    r = gp[(gp.label == 'lag_star') & (gp.scheme == 'ita') & (gp.model == 'R') & (gp.metric == 'mae')].iloc[0]
    claims.append(('OpenOx ITA MAE gap', f'{r.gap:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]', allthree))
    cb = csv('openox_spo2_contact_bias_gaps.csv').set_index('scheme')
    claims.append(('contact bias gap Monk', f'{cb.loc["monk"].gap:+.2f} [{cb.loc["monk"].lo:+.2f}, {cb.loc["monk"].hi:+.2f}]', allthree))
    claims.append(('contact bias gap ITA', f'{cb.loc["ita"].gap:+.2f} [{cb.loc["ita"].lo:+.2f}, {cb.loc["ita"].hi:+.2f}]', allthree))
    lg = csv('openox_spo2_lag.csv')
    r = lg[(lg.kind == 'estimate') & (lg.method == 'ppg')].iloc[0]
    claims.append(('OpenOx lag', f'[{int(r.ci_lo)}, {int(r.ci_hi)}]', ['README.md', 'MODEL.md', 'WHITEPAPER.md']))
    hv = csv('hr_vote_summary.csv')
    rule = 'vote+harmonic+octave+median5'
    for ds in ('butppg', 'mths'):
        r = hv[(hv.dataset == ds) & (hv.variant == rule) & (hv.scope == 'clean')].iloc[0]
        claims.append((f'HR vote clean MAE {ds}', f'{r.mae:.2f}', allthree))
    sb = csv('hr_vote_slow_band.csv').set_index('estimator')
    claims.append(('slow pulse MAE floor 51', f'{sb.loc["spec_band51"].mae:.1f}', allthree))
    claims.append(('slow pulse MAE floor 40', f'{sb.loc["spec_band40"].mae:.1f}', allthree))
    dep = open(os.path.join(RES, 'DEPLOY.md')).read()
    claims.append(('forward pass latency from DEPLOY.md', re.search(r'one OxyNet forward pass[^|]*\|\s*([\d.]+) ms', dep).group(1) + ' ms', ['MODEL.md', 'WHITEPAPER.md']))
    ssl = json.load(open(os.path.join(RES, 'ssl_manifest.json')))
    claims.append(('SSL rows', f"{ssl['ssl_full']['rows']:,}", ['MODEL.md', 'WHITEPAPER.md']))
    try:
        sys.path.insert(0, os.path.join(ROOT, 'python'))
        from oxydrop_ml.model import OxyNet, student_cfg  # type: ignore
        n = sum(p_.numel() for k, p_ in OxyNet(student_cfg()).named_parameters() if not k.startswith('aux_spo2'))
        claims.append(('parameter count from the model code', f'{n:,}', allthree))
    except Exception as e:  # torch missing: skip, not a failure
        warns.append(f'parameter count not re-derived ({type(e).__name__})')
    # ---- photon harness (round-2 shipped model PhotonDistill), straight from results/photon/TABLE.csv with own code
    t = pd.read_csv(os.path.join(PHOTON, 'TABLE.csv'), dtype={'level': str}).set_index(['model', 'level'])
    sd = lambda x: f'{x:+.3f}'
    d0, c0, dx, cx = t.loc[('distill', '0')], t.loc[('chain_v1', '0')], t.loc[('distill', 'X1')], t.loc[('chain_v1', 'X1')]
    claims.append(('photon: PhotonDistill L0 TM [CI]', f'{f3(d0.tm_bacc)} [{f3(d0.tm_lo)}, {f3(d0.tm_hi)}]', ['MODEL.md', 'WHITEPAPER.md']))
    claims.append(('photon: PhotonDistill L0 paired diff vs chain_v1', f'{sd(d0.diff_vs_chain_v1)} [{f3(d0.diff_lo)}, {f3(d0.diff_hi)}]', ['MODEL.md', 'WHITEPAPER.md']))
    claims.append(('photon: PhotonDistill loses by', f'loses to the logistic chain on clean light by {f3(abs(d0.diff_vs_chain_v1))}', ['MODEL.md', 'WHITEPAPER.md']))
    claims.append(('photon: chain_v1 L0 on the photon harness', f'{f3(c0.tm_bacc)} [{f3(c0.tm_lo)}, {f3(c0.tm_hi)}]', ['MODEL.md', 'WHITEPAPER.md']))
    claims.append(('photon: X1 pitch numbers', f'{f3(dx.tm_bacc)} against {f3(cx.tm_bacc)}', ['MODEL.md', 'WHITEPAPER.md']))
    for lv in ('4', '5', 'X1'):
        r = t.loc[('distill', lv)]
        claims.append((f'photon: PhotonDistill {lv} paired diff', f'{sd(r.diff_vs_chain_v1)} [{f3(r.diff_lo)}, {f3(r.diff_hi)}]', ['MODEL.md', 'WHITEPAPER.md']))
    return claims


def check_independent(rendered: dict):
    for label, expect, docs in independent_claims():
        for d in docs:
            if expect not in rendered.get(d, read(d)):
                fail(f'{d}: expected "{expect}" ({label}) not found')



# ----------------------------------------------------------------------------------------------------------------------- 2b. numbers quoted in the app text (app/index.html)
def check_app_text():
    """The results paragraph of the 'How it works' card is typed by hand in app/index.html. Every number in it is recomputed here from results/ml/*.csv and must appear verbatim,
    so the card cannot drift from the shipped model (review of 2026-11-15: it once quoted the per-person-selected pipeline instead of the shipped pipeline R)."""
    import html
    txt = html.unescape(re.sub(r'<[^>]+>', ' ', read('app/index.html')))
    txt = re.sub(r'\s+', ' ', txt)
    h = csv('hybrid_p1_final.csv').set_index('model')
    b = csv('baselines_P1.csv')
    r1 = b[(b.model == 'chain_v1') & (b.variant == 'full') & (b.metric == 'tm_bacc')].iloc[0]
    cc = csv('conformal_curve.csv')
    cf = lambda sysname: cc[(cc.system == sysname) & (cc.method == 'mondrian') & ((cc.param - 0.10).abs() < 1e-9)].iloc[0]
    d, c, fnl = cf('deployed_R'), cf('chain_v1'), cf('final')
    h2 = csv('hybrid_p2.csv')
    p2 = lambda scope, m: h2[(h2.scope == scope) & (h2.model == m)].iloc[0].within5_full
    rules = csv('hybrid_p2_rules.csv').set_index('rule')
    ood = csv('ood_ac_range.csv')
    guard = lambda ds: ood[(ood.dataset == ds) & (ood.percentile == 0.1)].iloc[0].share_below_cut
    ov = csv('openox_spo2_overall.csv')
    mae = lambda m: ov[(ov.label == 'lag_star') & (ov.model == m)].iloc[0].mae
    R = h.loc['default_R']; F = h.loc['HYBRID_FINAL']
    expect = [
        f'time-matched balanced accuracy is {R.tm_bacc:.3f} (95% interval {R.ci_lo:.3f} to {R.ci_hi:.3f}), against {r1.pooled:.3f} ({r1.pooled_lo:.3f} to {r1.pooled_hi:.3f}) for the logistic chain alone',
        f'the nested estimate for the whole selection procedure is {F.tm_bacc:.3f} ({F.ci_lo:.3f} to {F.ci_hi:.3f})',
        f'A chain refit with the same tuning and no network scores {h.loc["CHAIN_SAME_LEVERS"].tm_bacc:.3f}',
        f'the chain alone answers {100 * c.coverage:.1f}% of windows at {100 * c.acc:.1f}% accuracy, against {100 * d.coverage:.1f}% of windows at {100 * d.acc:.1f}% accuracy for the hybrid',
        f'the hybrid answers {100 * d.coverage:.1f}% of windows, {100 * d.acc:.1f}% of those answers are right, and {100 * d.alert_share_all:.1f}% of the windows with low oxygen raise an alert',
        f'within 5 bpm of the reference in {100 * p2("all", "HYBRID_agree"):.1f}% of windows on 100 people (BUT PPG and MTHS), against {100 * p2("all", "v2_spectral_acf__forced"):.1f}% for the spectral estimator',
        f'BUT PPG ({100 * p2("butppg", "HYBRID_agree"):.1f}% against {100 * p2("butppg", "v2_spectral_acf__forced"):.1f}%)',
        f'the spectral estimator alone does better ({100 * rules.loc["spec"].within5_full:.1f}% against {100 * rules.loc["agree"].within5_full:.1f}% for the rule)',
        f'nominally better ({100 * p2("mths", "v2_spectral_acf__forced"):.1f}% against {100 * p2("mths", "HYBRID_agree"):.1f}%)',
        f'mean error {mae("R"):.2f} against {mae("ror_textbook"):.2f} points',
        f'it would refuse {100 * guard("mths"):.1f}% (MTHS) and {100 * guard("butppg"):.1f}% (BUT PPG) of windows, against {100 * guard("hoffman"):.1f}% of the training windows',
    ]
    for e in expect:
        if e not in txt:
            fail(f'app/index.html: expected text not found (a number drifted from results/ml): "{e}"')
    # the shipped model is the fixed pipeline R: its numbers must be the ones in conformal.json's own metadata
    meta = json.load(open(os.path.join(ROOT, 'app/data/conformal.json'))).get('nested_evaluation') or {}
    for key, row in (('deployed_pipeline_R', d), ('chain_only_control', c), ('per_person_selected_hybrid', fnl)):
        m = meta.get(key, {})
        if abs(m.get('coverage', -1) - row.coverage) > 5e-5 or abs(m.get('accuracy_on_answered', -1) - row.acc) > 5e-5 or abs(m.get('share_of_hypoxic_windows_alerted', -1) - row.alert_share_all) > 5e-5:
            fail(f'app/data/conformal.json nested_evaluation.{key} differs from results/ml/conformal_curve.csv (rebuild with `python -m oxydrop_ml.export_conformal build`)')
    if '\u2014' in read('app/index.html'):
        fail('app/index.html contains an em dash')

# ----------------------------------------------------------------------------------------------------------------------- 3. hand-typed numbers in templates
ALLOW_DECIMALS = {
    '5.00',          # Hoffman 2022 SpO2 MAE (literature)
    '0.5', '0.7',    # design constants: fixed decision threshold, band-pass lower edge
    '0.2', '1.1', '1.0', '2.0', '1.5', '0.9', '0.8', '1.4',  # version strings, feature timing constants, band names
    '4.0',           # CC BY 4.0 / CC BY-NC-ND 4.0 licence versions
    '0.10',          # declared conformal miscoverage level alpha (Amendment 5, before any result)
    '11.6',          # million multiply-adds per window (python/oxydrop_ml/MODEL_SPEC.md, measured by model.py)
    '0.55',          # null-control ceiling of the round-2 rule (ROUND2.md rule 5, stated before any round-2 score)
}
TOKEN = re.compile(r'\{\{[^{}]+\}\}')
CODE = re.compile(r'```.*?```|`[^`\n]+`', re.S)
URL = re.compile(r'https?://\S+|\]\([^)]*\)|<!--.*?-->', re.S)


def check_templates():
    for tpl in glob.glob(os.path.join(ROOT, 'docs', 'templates', '*.md.in')):
        txt = open(tpl, encoding='utf-8').read()
        txt = TOKEN.sub(' ', URL.sub(' ', CODE.sub(' ', txt)))
        txt = re.sub(r'\bv\d+(\.\d+)*\b|\bdoi:\S+|\b\d+\.\d+\.\d+\b|arXiv:\S+', ' ', txt)       # version strings, dois, arXiv ids
        txt = re.sub(r'^#+ .*$', ' ', txt, flags=re.M)                                           # headings (section numbers)
        for m in re.finditer(r'(?<![\w.])(\d+\.\d+)(?![\w])', txt):
            if m.group(1) not in ALLOW_DECIMALS:
                line = txt[:m.start()].count('\n') + 1
                fail(f'{os.path.basename(tpl)} line ~{line}: hand-typed decimal {m.group(1)!r} (use a {{{{token}}}} or add it to ALLOW_DECIMALS with a reason)')
        for m in re.finditer(r'\b(\d{1,3}(?:\.\d+)?)\s?%', txt):
            if m.group(1) not in ('81', '79', '100', '95', '90', '10'):   # literature rates, 95% intervals, SpO2 < 90% definition, 10% = 1 - (1 - alpha) target
                line = txt[:m.start()].count('\n') + 1
                fail(f'{os.path.basename(tpl)} line ~{line}: hand-typed percentage {m.group(0)!r}')


# ----------------------------------------------------------------------------------------------------------------------- 4. traceability of every 3-decimal number
def value_set() -> set[str]:
    s: set[str] = set()
    for p in glob.glob(os.path.join(RES, '*.csv')) + glob.glob(os.path.join(PHOTON, '*.csv')):          # results/ml and the photon harness (TABLE, STRESS, NULLS)
        if p.endswith('CLAIMS_MANIFEST.csv'):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for col in df.select_dtypes(include=[np.number]).columns:
            v = df[col].dropna().to_numpy(dtype=float)
            for x in v:
                s.add(f'{x:.3f}')
                s.add(f'{abs(x):.3f}')
                s.add(f'{100 * x:.1f}')
    for p in glob.glob(os.path.join(RES, '*.json')) + [os.path.join(ROOT, 'app', 'data', 'hybrid_weights.json'), os.path.join(PHOTON, 'distill_weights.json')]:
        if not os.path.exists(p):
            continue
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, float):
                s.add(f'{o:.3f}')
                s.add(f'{o:.2f}')
        walk(json.load(open(p)))
    return s


def derived_set() -> set[str]:
    """Documented derived values: train-minus-held-out gaps and differences of two CSV values that the documents quote."""
    s: set[str] = set()
    for t in ('P1', 'P2', 'P3'):
        d = csv(f'baselines_{t}.csv')
        for _, r in d.iterrows():
            if pd.notna(r.train_mean) and pd.notna(r.pooled):
                s.add(f'{abs(r.train_mean - r.pooled):.3f}')
    a = csv('ablation_core.csv')
    full = a[(a.variant == 'full') & (a.p23_kind == 'p23')].iloc[0]
    full0 = a[(a.variant == 'full (seed 0)') & (a.p23_kind == 'p23k')].iloc[0]
    for _, r in a.iterrows():
        for col in ('p1_tm_bacc', 'p2_within5', 'p3_mae'):
            for ref in (full, full0):
                if pd.notna(r[col]) and pd.notna(ref[col]):
                    s.add(f'{abs(r[col] - ref[col]):.3f}')
    g = csv('hybrid_p1_final_gap.csv')
    for _, r in g.iterrows():
        for a_, b_ in (('train_pooled', 'heldout_pooled'), ('inner_cv_mean', 'heldout_pooled')):
            s.add(f'{abs(r[a_] - r[b_]):.3f}')
    h2 = csv('hybrid_p2.csv')
    r1 = h2[(h2.scope == 'all') & (h2.model == 'HYBRID_agree')].iloc[0]
    r2 = h2[(h2.scope == 'all') & (h2.model == 'cnn_alone')].iloc[0]
    s.add(f'{abs(r1.within5_full - r2.within5_full):.3f}')
    # photon harness: mean TM over the null draws of each round-2 model (ml_numbers.ph_null; the rule of ROUND2.md uses the mean)
    t = pd.read_csv(os.path.join(PHOTON, 'TABLE.csv'), dtype={'level': str})
    for m in set(t.model):
        for kind in ('null', 'blank'):
            d = N.ph_null(m, kind)
            if d:
                s.add(f'{np.mean([x for _, x in d]):.3f}')
    return s


def check_traceable(rendered: dict):
    vs = value_set() | derived_set()
    for d in ('README.md', 'MODEL.md', 'WHITEPAPER.md'):
        txt = rendered[d]
        txt = CODE.sub(' ', URL.sub(' ', txt))
        missing = sorted({m.group(1) for m in re.finditer(r'(?<![\w.])[+-]?(\d+\.\d{3})(?![\w.])', txt) if m.group(1) not in vs and m.group(1) not in ('0.000',)})
        if missing:
            fail(f'{d}: 3-decimal numbers not found in any results/ml file: {missing[:15]}')
    for d in sorted(glob.glob(os.path.join(ROOT, 'docs', 'submission', '*.md'))):          # hand-written submission pack: every 3-decimal number must trace too
        txt = CODE.sub(' ', URL.sub(' ', open(d, encoding='utf-8').read()))
        missing = sorted({m.group(1) for m in re.finditer(r'(?<![\w.])[+-]?(\d+\.\d{3})(?![\w.])', txt) if m.group(1) not in vs and m.group(1) not in ('0.000',)})
        if missing:
            fail(f'{os.path.relpath(d, ROOT)}: 3-decimal numbers not found in any results/ml or results/photon file: {missing[:15]}')


# ----------------------------------------------------------------------------------------------------------------------- 5. style, retired phrases, PDF
def check_style():
    for d in DOCS + ['DATA_LICENSES.md']:
        t = read(d)
        if '\u2014' in t:
            fail(f'{d}: contains an em dash')
    for d in glob.glob(os.path.join(ROOT, 'docs', 'submission', '*.md')):
        if '\u2014' in open(d, encoding='utf-8').read():
            fail(f'{os.path.relpath(d, ROOT)}: contains an em dash')
    for tpl in glob.glob(os.path.join(ROOT, 'docs', 'templates', '*.in')):
        if '\u2014' in open(tpl, encoding='utf-8').read():
            fail(f'{os.path.basename(tpl)}: contains an em dash')
    node = shutil.which('node')
    if node:
        r = subprocess.run([node, 'tests/docs_claims.mjs'], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            fail('tests/docs_claims.mjs failed:\n' + r.stdout[-800:])
    else:
        warns.append('node not found: tests/docs_claims.mjs not run')


def check_pdf():
    pdf = os.path.join(ROOT, 'WHITEPAPER.pdf')
    if not os.path.exists(pdf):
        fail('WHITEPAPER.pdf missing (bash scripts/build_whitepaper_pdf.sh)')
        return
    if not shutil.which('pdfinfo'):
        warns.append('pdfinfo not found: PDF page count not checked')
        return
    info = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    if not 4 <= pages <= 6:
        fail(f'WHITEPAPER.pdf has {pages} pages (need 4 to 6)')
    txt = subprocess.run(['pdftotext', pdf, '-'], capture_output=True, text=True).stdout
    h = csv('hybrid_p1_final.csv').set_index('model').loc['HYBRID_FINAL']
    for needle in (f3(h.tm_bacc), 'Abstract', 'not a medical device'):
        if needle not in txt:
            fail(f'WHITEPAPER.pdf looks stale: "{needle}" not found (rebuild with scripts/build_whitepaper_pdf.sh)')
    imgs = subprocess.run(['pdfimages', '-list', pdf], capture_output=True, text=True).stdout.strip().splitlines() if shutil.which('pdfimages') else []
    if imgs and len(imgs) < 4:
        fail('WHITEPAPER.pdf has fewer than 3 embedded images (figures missing?)')
    print(f'WHITEPAPER.pdf: {pages} pages')


def main():
    rendered = check_render()
    check_independent(rendered)
    check_app_text()
    check_templates()
    check_traceable(rendered)
    check_style()
    check_pdf()
    for w in warns:
        print('WARN', w)
    pend = N.pending_report()
    print('pending sections (rendered as PENDING blocks):', [k for k, v in pend.items() if v] or 'none')
    if fails:
        print(f'CHECK CLAIMS FAIL ({len(fails)})')
        return 1
    print('CHECK CLAIMS PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
