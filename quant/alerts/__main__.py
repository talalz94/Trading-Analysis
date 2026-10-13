"""
    python -m quant.alerts list         every instrument, timeframe, pattern, kind and setup you can use
    python -m quant.alerts check        what WOULD have fired in the last 48h -> preview.html; sends nothing
    python -m quant.alerts test-email   one short email, to prove the .env settings work
    python -m quant.alerts tick         one pass of the monitor, then exit
    python -m quant.alerts run          the monitor itself (what the scheduled task starts)
    python -m quant.alerts status       is it running, when did it last email, what is queued
    python -m quant.alerts news         the covered calendar for the next --days, with reminders
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, DEFAULT_STATE, load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m quant.alerts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "check", "test-email", "tick", "run", "status",
                                        "news"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--hours", type=float, default=48, help="check: how far back to look")
    ap.add_argument("--days", type=float, default=7, help="news: how far ahead to look")
    ap.add_argument("--write-docs", action="store_true",
                    help="list: rewrite docs/ALERTS_CATALOGUE.md instead of printing")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):           # a cp1252 console cannot print the bolt
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    if a.command == "list":
        from .catalogue import catalogue, write_docs
        print(f"wrote {write_docs()}" if a.write_docs else catalogue())
        return 0

    if a.command == "status":
        from .runner import status
        print(status(a.state))
        return 0

    if a.command == "news":
        import pandas as pd
        from .news import LiveNews, upcoming_table
        cfg = load_config(a.config)
        if not cfg.news.on:
            print("NEWS is off (or missing) in the config")
            return 0
        now = pd.Timestamp.now(tz="UTC")
        live = LiveNews(a.state / "news")
        for k, v in live.refresh(now).items():
            print(f"  {k}: {v}")
        print(f"times in {cfg.news.timezone}; results "
              + ("on" if live.results_available() and cfg.news.results else
                 "OFF (no JBLANKED_API_KEY in .env)" if cfg.news.results else "off"))
        rows = upcoming_table(live, cfg, now, days=a.days)
        print("\n".join(rows) if rows else f"nothing covered in the next {a.days:g} days")
        return 0

    if a.command == "check":
        from .engine import preview
        from .feeds import Feed
        cfg = load_config(a.config)
        events, subject, text, html, errors = preview(cfg, Feed(), hours=a.hours)
        for m, e in errors.items():
            print(f"  {m}: feed failed -- {e}")       # already "Type: message"
        if not events:
            print(f"nothing would have fired in the last {a.hours:g} hours")
            return 0
        a.state.mkdir(parents=True, exist_ok=True)
        out = a.state / "preview.html"
        out.write_text(f"<!doctype html><meta charset='utf-8'><title>{subject}</title>"
                       f"<p><b>{subject}</b></p>{html}", encoding="utf-8")
        print(subject, "", text, "", f"preview written to {out}", sep="\n")
        return 0

    from .notify import NotConfigured, SmtpNotifier
    try:
        notifier = SmtpNotifier.from_env()
    except NotConfigured as e:
        print(f"email is not set up: {e}", file=sys.stderr)
        return 2

    if a.command == "test-email":
        notifier.send("[Alerts] test -- the email settings work",
                      "If you can read this, quant alerts can reach you.",
                      "<p>If you can read this, <b>quant alerts</b> can reach you.</p>")
        print(f"sent to {', '.join(notifier.to)}")
        return 0

    from .feeds import Feed
    from .runner import acquire_lock, log_report, run_forever, setup_logging
    if a.command == "run":
        return run_forever(a.config, a.state, notifier, Feed())

    from .engine import tick
    from .store import State
    lock = acquire_lock(a.state / "alerts.lock")
    if lock is None:
        print("the monitor is running; it ticks every minute already")
        return 1
    setup_logging(a.state, console=True)
    log_report(tick(load_config(a.config), State(a.state), notifier, Feed()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
