"""Ratio-of-ratios baselines.

Textbook rule: SpO2_hat = 110 - 25 R with R = (AC_R / DC_R) / (AC_x / DC_x), x = blue (RB) or green (RG). In the log domain the window AC std of ln(rgb + 1)
is AC/DC, so R = exp(rrRB) = exp(ln(sd_R / sd_B)) (the `rr*` features of xgb_hand.hand26). Clipped to [70, 100] like scripts/mths_spo2.py.
No parameter is fit, so the rule needs no selection.

P1 (hypoxia detection, SpO2 < 90): `ror_textbook_{RB,RG}` alerts when 110 - 25 R < 90 (R > 0.8). `ror_calibrated_{RB,RG}` uses the baseline-relative change of
ln R (the feature a Hoffman-style per-person calibrated ratio would use); its sign and threshold are fit on the training people only (the threshold
by the inner CV of common.run_cls).
"""
import numpy as np

from .. import metrics as M
from . import common as C
from .xgb_hand import FEATS26

COL = {n: i for i, n in enumerate(FEATS26)}


def textbook_spo2(rr):
    """110 - 25 R clipped to [70, 100] from the log AC ratio rr = ln R."""
    return np.clip(110.0 - 25.0 * np.exp(np.asarray(rr, float)), 70.0, 100.0)


class _Pred:
    def __init__(self, f, v=None):
        self.score = f
        if v is not None:
            self.valid = v


def specs(ctx):
    out = []
    for ch in ('RB', 'RG'):
        col = COL['rr' + ch]
        # score = R - 0.8: positive exactly when 110 - 25 R < 90
        out.append(C.Spec(f'ror_textbook_{ch}', lambda hp, rows, col=col: _Pred(lambda q: np.exp(ctx.F['hand26'][q, col]) - 0.8), thr_mode='fixed', fixed_thr=0.0, bands=False,
                          note=f'textbook SpO2 = 110 - 25 R (R from {ch}); alert when below 90; no fitted parameter'))

        def fit(hp, rows, col=col):
            x = ctx.F['hand26_cal'][rows, col]; y = ctx.y[rows]
            ok = np.isfinite(x)
            sign = 1.0 if M.auroc(y[ok], x[ok]) >= 0.5 else -1.0
            return _Pred(lambda q: sign * ctx.F['hand26_cal'][q, col])
        out.append(C.Spec(f'ror_calibrated_{ch}', fit, thr_mode='tune', bands=False,
                          note=f'baseline-relative change of ln R ({ch}); sign and threshold fit on training people only'))
    return out
