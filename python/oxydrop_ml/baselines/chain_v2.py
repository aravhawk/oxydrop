"""v2 logistic chain (python/oxydrop/chain_v2.py, robust-HR gated features) refit under the common LOSO harness.

Same two logistic regressions as v1 but fit on VALID windows only (pulse locked). Windows where the pulse is not locked are reported two ways:
`full` (forced answer: forward-filled features, the 'every window forced to answer' row of results/eval_v1_vs_v2.csv) and `nolock_abstain` (no answer).
Person 4 (resting HR 43 bpm, below the 51 bpm band edge of hr.py) never locks, so v2 abstains on all of that person's windows in the `nolock_*` variants.
The implementation is shared with v1: chain_v1.specs(ctx, version=2).
"""
from . import chain_v1


def specs(ctx):
    return chain_v1.specs(ctx, version=2)
