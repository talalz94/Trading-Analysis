"""
python -m quant.news update              refresh the feed (+ FRED, JBlanked when keys are in .env)
python -m quant.news import-archive      the 2007-2025 archive, once (~68 MB download, ~1.4 MB kept)
python -m quant.news backfill-jb [FROM]  JBlanked month by month (one request per 5 minutes)
python -m quant.news show [--days N] [--past N] [--level high] [--cur USD,EUR]
python -m quant.news status              what each source last did
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


def _show(args):
    from .calendar import load_events
    from .catalog import DEFAULT_COVERAGE
    now = pd.Timestamp.now(tz="UTC")
    if args.cur:
        cov = {c.strip().upper(): args.level or "medium" for c in args.cur.split(",")}
    elif args.level:
        cov = {"*": args.level}
    else:
        cov = DEFAULT_COVERAGE
    ev = load_events(now - pd.Timedelta(days=args.past), now + pd.Timedelta(days=args.days),
                     coverage=cov)
    if ev.empty:
        print("nothing in range -- run `python -m quant.news update` first?")
        return
    tz = args.tz
    for e in ev.itertuples():
        when = pd.Timestamp(e.t).tz_convert(tz).strftime("%a %d %b %H:%M")
        nums = "  ".join(f"{k} {v}" for k, v in (("A", e.actual), ("F", e.forecast),
                                                   ("P", e.previous)) if v)
        mood = f"  -> {e.sentiment} for {e.currency}" if e.sentiment else ""
        time_note = "" if e.time_q in ("exact", "official") else f"  ({e.time_q} time)"
        print(f"{when}  {e.currency}  {e.level:<8} {e.title:<34} {nums}{mood}{time_note}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quant.news", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update")
    a = sub.add_parser("import-archive")
    a.add_argument("--csv", help="a CSV already downloaded (else it is downloaded and deleted)")
    b = sub.add_parser("backfill-jb")
    b.add_argument("start", nargs="?", default="2025-04-01")
    s = sub.add_parser("show")
    s.add_argument("--days", type=float, default=7)
    s.add_argument("--past", type=float, default=0)
    s.add_argument("--level", choices=["low", "medium", "high", "critical"])
    s.add_argument("--cur", help="currencies, e.g. USD,EUR")
    s.add_argument("--tz", default="Asia/Karachi")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    from . import update
    from .calendar import NewsStore
    st = NewsStore()
    if args.cmd == "update":
        print("\n".join(update.update_all(st)))
    elif args.cmd == "import-archive":
        print(update.import_archive(st, args.csv, keep_download=bool(args.csv)))
    elif args.cmd == "backfill-jb":
        if not update.keys()["jblanked"]:
            sys.exit("JBLANKED_API_KEY is not in .env")
        print(f"stored {update.backfill_jb(st, args.start)} rows")
    elif args.cmd == "show":
        _show(args)
    elif args.cmd == "status":
        m = st.meta()
        for k in sorted(m):
            if k != "feed_weeks":
                print(f"{k:22s} {m[k]}")
        for n in NewsStore.FILES:
            df = st.read(n)
            print(f"{n + '.parquet':22s} {'-' if df is None else f'{len(df):,} rows'}")


if __name__ == "__main__":
    main()
