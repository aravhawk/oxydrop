"""Generate the HR parity fixture tests/fixtures/hr_parity.json: BUT PPG windows + synthetic edge cases, with the Python (hr.py) outputs.
Inputs are rounded to 6 significant digits BEFORE the reference is computed, so JS and Python see identical numbers.
Usage: python scripts/gen_hr_ref.py   (needs the BUT PPG dataset, see scripts/validate_hr.py)"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oxydrop import hr as HR  # noqa: E402
import validate_hr as V  # noqa: E402

N_CLIPS = 80      # 3 channels each -> 240 single-channel windows, 80 best-channel calls


def r6(a):
    return np.array([float(f'{v:.6g}') for v in np.ravel(a)]).reshape(np.shape(a))


def ref(x, f_lo=None):
    e = HR.estimate_hr(x, 30, f_lo)
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else (bool(v) if k == 'valid' else float(v))) for k, v in e.items()}


def edge_cases():
    rng = np.random.default_rng(7)
    t = np.arange(300) / 30.0
    cases = {}
    cases['flat'] = np.zeros(300)
    cases['too_short'] = np.sin(2 * np.pi * 1.2 * t[:200])
    for f in (0.85, 0.80, 1.0, 2.9, 3.0, 3.2):
        cases[f'sine_{f}Hz_edge'] = np.sin(2 * np.pi * f * t) + 0.02 * rng.standard_normal(300)
    # strong dicrotic notch: fundamental 75 bpm + 2nd harmonic at 1.1x (the 150 bpm field failure)
    for bpm in (50, 60, 75, 100):
        x = 2 * np.pi * bpm / 60 * t
        cases[f'notch_{bpm}bpm'] = np.sin(x) + 1.1 * np.sin(2 * x + 0.8) + 0.05 * rng.standard_normal(300)
    # slow drift (no peak in the autocorrelation search window) and low-frequency + noise
    cases['ramp'] = np.linspace(-1, 1, 300)
    cases['slow_sine_0.3Hz_plus_noise'] = np.sin(2 * np.pi * 0.3 * t) + 0.3 * rng.standard_normal(300)
    # search for windows whose in-band autocorrelation peaks are all negative (the case that used to raise StopIteration)
    found = 0
    for k in range(4000):
        x = np.cumsum(rng.standard_normal(300)) if k % 2 else np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * t + rng.uniform(0, 6)) + 0.2 * rng.standard_normal(300)
        x = r6(x)
        n = len(x); xc = x - x.mean(); m = 1 << int(np.ceil(np.log2(2 * n)))
        r = np.fft.irfft(np.abs(np.fft.rfft(xc, n=m)) ** 2)[:n]; r /= r[0]
        lo, hi = int(np.ceil(30 / 3.0)), min(int(np.floor(30 / HR.F_LO)), n - 2)
        seg = r[lo - 1:hi + 2]
        pk = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
        if pk and max(seg[i] for i in pk) < 0:
            cases[f'all_negative_acf_{found}'] = x; found += 1
            if found == 3:
                break
    assert found, 'no all-negative ACF window found'
    return {k: r6(v) for k, v in cases.items()}


def main():
    clips = V.load_but()
    sel = clips[::max(1, len(clips) // N_CLIPS)][:N_CLIPS]
    wins = []
    for rid, refhr, q, ef, mot, rgb in sel:
        ac = V.bandpass_from_rgb(rgb, mirror=True)
        wins.append(r6(ac[-300:]))
    out = dict(gate=dict(F_LO=HR.F_LO, SQI_MIN=HR.SQI_MIN, ACF_MIN=HR.ACF_MIN, AGREE=HR.AGREE), but=[], best=[], edge=[], f_lo_override=[])
    for w in wins:
        out['but'].append(dict(x=[w[:, c].tolist() for c in range(3)], ref=[ref(w[:, c]) for c in range(3)]))
        sd = w.std(0); primary = 1 if sd[1] > 0.3 * sd[0] else 0
        e, ch = HR.best_channel_estimate(w, primary)
        out['best'].append(dict(acs=[w[:, c].tolist() for c in range(3)], primary=primary, ch=int(ch),
                                ref={k: (None if (isinstance(v, float) and not np.isfinite(v)) else (bool(v) if k == 'valid' else float(v))) for k, v in e.items()}))
    for name, x in edge_cases().items():
        out['edge'].append(dict(name=name, x=x.tolist(), ref=ref(x)))
    for w in wins[:40]:   # a different band edge (the sweep values) must also agree
        for f in (0.65, 0.75):
            out['f_lo_override'].append(dict(f_lo=f, x=w[:, 1].tolist(), ref=ref(w[:, 1], f)))
    p = os.path.join(ROOT, 'tests', 'fixtures', 'hr_parity.json')
    json.dump(out, open(p, 'w'), separators=(',', ':'))
    print('wrote', p, os.path.getsize(p) // 1024, 'KiB;', len(out['but']) * 3, 'BUT windows,', len(out['best']), 'best-channel calls,', len(out['edge']), 'edge cases,', len(out['f_lo_override']), 'F_LO overrides')


if __name__ == '__main__':
    main()
