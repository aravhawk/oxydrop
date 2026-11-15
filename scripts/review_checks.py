"""Recompute the facts the independent review (REVIEW.md, PR #3) found missing or wrong, from committed files only.
Reads results/loPO_v1_windows.csv, results/loPO_v2_windows.csv, results/scores/*, results/timeline_notes.json, results/scores/exercise_fp.csv.
Writes results/review_checks.csv (name, value, note). Dependencies: numpy, pandas. Usage: python scripts/review_checks.py
Nothing here is a model result; it is arithmetic on saved out-of-fold scores, so it checks the write-up, not the training procedure."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
R = ROOT + '/results/'
BAND = 0.10
rows = []
def add(name, value, note=''):
    rows.append(dict(name=name, value=value, note=note))

v1 = pd.read_csv(R + 'loPO_v1_windows.csv'); v2 = pd.read_csv(R + 'loPO_v2_windows.csv')
assert (v1.rec == v2.rec).all() and (v1.t == v2.t).all() and (v1.y == v2.y).all()
y = v2.y.values; n = len(v2)
add('eval_windows', n, 't >= 60 s, label = reference SpO2 < 90')
add('hypoxic_windows', int(y.sum()))
add('majority_class_accuracy', round(max(y.mean(), 1 - y.mean()), 4), 'always answer "hypoxic": the missing baseline')

# v2 answers a window when the pulse locks and the probability is outside the abstain band; v1 scored on exactly those windows
ans = (v2.valid.values.astype(bool)) & (np.abs(v2.p.values - 0.5) >= BAND)
acc = lambda p, m: float(((p[m] > 0.5) == (y[m] == 1)).mean())
add('v2_answered_windows', int(ans.sum()), 'lock-gated + abstain band 0.10')
add('v2_coverage', round(ans.mean(), 4))
add('v2_accuracy_on_answered', round(acc(v2.p.values, ans), 4))
add('v1_accuracy_on_same_answered_windows', round(acc(v1.p.values, ans), 4), 'the selective gain over v1 on identical windows')
add('v1_accuracy_all_windows', round(acc(v1.p.values, np.ones(n, bool)), 4))

# alert yield: hypoxic windows that receive a "possible low oxygen" answer
alert = ans & (v2.p.values > 0.5)
add('hypoxic_windows_unanswered', int((y.astype(bool) & ~ans).sum()))
add('hypoxic_windows_alerted', int((y.astype(bool) & alert).sum()))
add('hypoxic_alert_rate_all_hypoxic_windows', round((y.astype(bool) & alert).sum() / y.sum(), 4), 'sensitivity counting abstentions as misses')
add('hypoxic_window_missed_given_answered', round(((y == 1) & ans & ~alert).sum() / max(((y == 1) & ans).sum(), 1), 4), 'the 2.4% figure: conditional on answering')
for sid_, g in v2.assign(ans=ans).groupby('sid'):
    add(f'person_{int(sid_) % 100}_answered_windows', int(g.ans.sum()), f'of {len(g)} windows')

# windows whose reference SpO2 is below 70 (the original paper excluded the lower range)
W = pd.read_csv(R + 'scores/windows.csv'); ev = W[W.t >= 60]
add('eval_windows_reference_below_70', int((ev.y < 70).sum()), 'window-mean reference SpO2 < 70 %')

# paired person bootstrap: TINY chain minus the clock-only baseline, accuracy at threshold 0.5 (same construction as make_eval.boot_ci, 500 draws, seed 1)
sid = W.sid.values; m = (W.t.values >= 60); yy = (W.y.values < 90).astype(int)
tiny = np.load(R + 'scores/tiny.npy'); clock = np.load(R + 'scores/time_only.npy')
corr = lambda s, idx: float(((s[idx] > .5) == yy[idx]).mean())
full = np.where(m)[0]
d0 = corr(tiny, full) - corr(clock, full)
rng = np.random.default_rng(1); ids = np.unique(sid); d = []
for _ in range(500):
    pick = rng.choice(ids, len(ids)); idx = np.concatenate([np.where(m & (sid == q))[0] for q in pick])
    d.append(corr(tiny, idx) - corr(clock, idx))
lo, hi = np.percentile(d, [2.5, 97.5])
add('tiny_minus_clock_accuracy_points', round(100 * d0, 2), 'saved out-of-fold scores')
add('tiny_minus_clock_ci_low_points', round(100 * lo, 2), 'paired person bootstrap, 500 draws, seed 1')
add('tiny_minus_clock_ci_high_points', round(100 * hi, 2))

# person 4 left hand: false alarms (alarm while reference >= 90) span
rec = W.rec.values; hit = m & (rec == '100004L') & (tiny > .5) & (yy == 0)
tt = W.t.values[hit]
add('person4_left_false_alarm_windows', int(hit.sum()))
add('person4_left_false_alarm_first_s', int(tt.min()))
add('person4_left_false_alarm_last_s', int(tt.max()), 'time of the window start')
add('person4_left_false_alarm_windows_after_400s', int((tt > 400).sum()))

# exercise false-alarm rate: pooled over 14 subjects and over the 13 with an extractable heart rate
E = pd.read_csv(R + 'scores/exercise_fp.csv')
nn = E.n_norm.values; fp = E.fp_rate_normoxic.values
ok = E.dhr_p50.notna().values   # a subject whose heart rate could not be extracted has no dHR and a 0.0 rate
add('exercise_subjects_total', len(E)); add('exercise_subjects_with_usable_heart_rate', int(ok.sum()))
add('exercise_fp_rate_all_subjects_pooled', round(float((fp * nn).sum() / nn.sum()), 4), 'the subject with no usable heart rate counts as never flagged, which lowers the headline')
add('exercise_fp_rate_usable_subjects_pooled', round(float((fp[ok] * nn[ok]).sum() / nn[ok].sum()), 4))

out = pd.DataFrame(rows); out.to_csv(R + 'review_checks.csv', index=False)
print(out.to_string(index=False))
