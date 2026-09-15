"""Run `python run_study.py --help` for the reproducible study interface."""
import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    data = sub.add_parser("download")
    data.add_argument("--output", required=True)
    data.add_argument("--as-of", required=True)
    data.add_argument("--legacy-data", default="data/Data.CSV")
    run = sub.add_parser("run")
    run.add_argument("--configuration", choices=["modern", "legacy"], default="modern")
    run.add_argument("--data", default="data/snapshots/2026-09-14-v2/monthly.csv")
    run.add_argument("--output", required=True)
    run.add_argument("--start", default="2018-01-31")
    run.add_argument("--end", default="2026-08-31")
    run.add_argument("--windows", nargs="+", type=int, default=[571,120])
    run.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    run.add_argument("--models", nargs="+")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--limit", type=int, help="Pilot forecasts per model/seed/window; rerun without limit to resume")
    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    args = p.parse_args()
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = str(getattr(args, "threads", 2))
    if args.command == "download":
        from qrcstudy.data import download
        download(args.output, args.as_of, args.legacy_data)
    elif args.command == "report":
        from qrcstudy.report import report
        report(args.run)
    elif args.configuration == "legacy":
        from qrcstudy.legacy import run_legacy
        run_legacy(args.output)
    else:
        from qrcstudy.run import run_modern
        from qrcstudy.models import MODELS
        models = args.models or MODELS
        if set(models)-set(MODELS):
            p.error("Unknown model")
        run_modern(args.data, args.output, args.start, args.end, args.windows, args.seeds, models, args.workers, args.threads, args.limit)


if __name__ == "__main__":
    main()
