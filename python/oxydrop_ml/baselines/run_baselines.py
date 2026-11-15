"""Run the honest baselines under the common LOSO harness and write results/ml/baselines_{P1,P2,P3}.csv (+ per-person tables, per-window scores, BASELINES.md).

    cd python && python -m oxydrop_ml.baselines.run_baselines --all            # P1 + P2 + P3 + report (about 10 minutes on the Surface host)
    python -m oxydrop_ml.baselines.run_baselines --p1 [--only chain_v1,xgb]    # one task (--only filters model names by substring)
    python -m oxydrop_ml.baselines.run_baselines --report                      # rebuild results/ml/BASELINES.md from the CSVs

Needs the shards of oxydrop_ml.data.{hoffman,mths,butppg} in the cache (python -m oxydrop_ml.data.<ds> --build) and the Hoffman repository for the chain features
(otherwise they are rebuilt from the shard). `--boot B` sets the number of person-bootstrap draws (default 1000, the pre-registered minimum).
"""
import argparse
import warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--all', action='store_true'); ap.add_argument('--p1', action='store_true'); ap.add_argument('--p2', action='store_true'); ap.add_argument('--p3', action='store_true')
    ap.add_argument('--report', action='store_true'); ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--only', default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings('ignore')
    only = a.only.split(',') if a.only else None
    if not (a.all or a.p1 or a.p2 or a.p3 or a.report):
        ap.error('choose --all, --p1, --p2, --p3 or --report')
    if a.all or a.p1:
        from . import p1
        p1.run(B=a.boot, only=only)
    if a.all or a.p2:
        from . import p2
        p2.run(B=a.boot)
    if a.all or a.p3:
        from . import p3
        p3.run(B=a.boot, only=only)
    if a.all or a.report:
        from . import report
        report.write()


if __name__ == '__main__':
    main()
