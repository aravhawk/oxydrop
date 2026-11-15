"""Subject-grouped splits for the pre-registered protocol (python/oxydrop_ml/PREREGISTRATION.md).

* The group is the PERSON (`subject` column of the shards): both Hoffman hands, all BUT clips and every OpenOx encounter of one person share a group.
* `loso`: leave-one-subject-out. `pooled_loso`: when a subject of dataset D is held out, training uses every other subject of every dataset.
* `inner_group_folds`: subject-grouped k-fold on the outer TRAINING subjects only (nested CV for every selection: hyperparameters, thresholds,
  abstention cutoffs, lag offsets, ensemble weights, conformal quantiles, lever choice).
* No window-level split exists anywhere. Every generator asserts that no group is on both sides of a split.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    test_group: object          # the held-out subject (LOSO) or a tuple of subjects (inner folds)
    train_idx: np.ndarray       # row indices into the arrays the split was built from
    test_idx: np.ndarray


def assert_no_leak(groups, train_idx, test_idx):
    """Raise AssertionError if any group appears on both sides, or if an index is on both sides."""
    groups = np.asarray(groups)
    a = set(groups[np.asarray(train_idx)].tolist()); b = set(groups[np.asarray(test_idx)].tolist())
    both = a & b
    assert not both, f'subject leakage: {sorted(both)[:5]} on both sides'
    assert not (set(np.asarray(train_idx).tolist()) & set(np.asarray(test_idx).tolist())), 'row on both sides'


def loso(groups, eval_mask=None):
    """Leave-one-subject-out. groups: array [n] of subject ids. Yields Fold(test_group, train_idx, test_idx) for each unique subject (sorted).
    eval_mask: optional bool [n]; only rows with eval_mask True are in test_idx (training keeps every row of the other subjects). Subjects with
    no evaluable row are skipped."""
    groups = np.asarray(groups)
    em = np.ones(len(groups), bool) if eval_mask is None else np.asarray(eval_mask, bool)
    for g in sorted(np.unique(groups).tolist()):
        te = np.nonzero((groups == g) & em)[0]
        if not len(te):
            continue
        tr = np.nonzero(groups != g)[0]
        assert_no_leak(groups, tr, te)
        yield Fold(g, tr, te)


def pooled_loso(groups, datasets, test_dataset, eval_mask=None, train_datasets=None):
    """Pooled-training protocol. groups [n] subject ids (globally unique across datasets, e.g. 'hof_100001'), datasets [n] dataset ids.
    For each subject of `test_dataset`, test = that subject's rows (eval rows only if eval_mask is given), train = every row of every OTHER subject,
    of all datasets (or only `train_datasets`). A person never appears in two datasets, but the assertion checks groups, not datasets."""
    groups = np.asarray(groups); datasets = np.asarray(datasets)
    em = np.ones(len(groups), bool) if eval_mask is None else np.asarray(eval_mask, bool)
    allow = np.ones(len(groups), bool) if train_datasets is None else np.isin(datasets, list(train_datasets))
    for g in sorted(np.unique(groups[datasets == test_dataset]).tolist()):
        te = np.nonzero((groups == g) & (datasets == test_dataset) & em)[0]
        if not len(te):
            continue
        tr = np.nonzero((groups != g) & allow)[0]
        assert_no_leak(groups, tr, te)
        yield Fold(g, tr, te)


def inner_group_folds(train_groups, k=5, seed=0, balance=None):
    """Subject-grouped k-fold over the given training rows (nested CV). train_groups [m]: the group of each training row (so pass
    `groups[train_idx]`). Yields Fold(test_group=tuple of validation subjects, train_idx, test_idx) with indices into train_groups.
    Subjects are shuffled with `seed` and dealt round-robin; if `balance` (an array [m] of 0/1, e.g. a hypoxic indicator) is given, subjects are
    first ordered by their share of positives so each fold gets a similar mix. k is capped at the number of subjects."""
    tg = np.asarray(train_groups)
    ug = np.unique(tg)
    k = int(min(k, len(ug)))
    assert k >= 2, 'need at least 2 training subjects for inner CV'
    rng = np.random.default_rng(seed)
    order = rng.permutation(ug)
    if balance is not None:
        b = np.asarray(balance, float)
        share = {g: np.nanmean(b[tg == g]) if (tg == g).any() else 0.0 for g in ug}
        order = np.array(sorted(order.tolist(), key=lambda g: -(share[g] if np.isfinite(share[g]) else 0.0)))
    for f in range(k):
        va_groups = order[f::k]
        va = np.nonzero(np.isin(tg, va_groups))[0]
        tr = np.nonzero(~np.isin(tg, va_groups))[0]
        assert_no_leak(tg, tr, va)
        yield Fold(tuple(va_groups.tolist()), tr, va)


def split_summary(folds):
    """List of dict(test_group, n_train, n_test) for logging."""
    return [dict(test_group=f.test_group, n_train=len(f.train_idx), n_test=len(f.test_idx)) for f in folds]


def pretrain_allowed(pretrain_subjects, outer_test_subjects):
    """SSL rule (prereg 5): pretraining may not use the unlabeled signal of an outer test subject. Returns a boolean mask over
    `pretrain_subjects` that is True where the subject is NOT in `outer_test_subjects`."""
    return ~np.isin(np.asarray(pretrain_subjects), list(outer_test_subjects))
