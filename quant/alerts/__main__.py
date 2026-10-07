"""
    python -m quant.alerts check        what WOULD have fired in the last 48h -> preview.html; sends nothing
    python -m quant.alerts test-email   one short email, to prove the .env settings work
    python -m quant.alerts tick         one pass of the monitor, then exit
    python -m quant.alerts run          the monitor itself (what the scheduled task starts)
    python -m quant.alerts status       is it running, when did it last email, what is queued
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, DEFAULT_STATE, load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m quant.alerts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "test-email", "tick", "run", "status"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--hours", type=float, default=48, help="check: how far back to look")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):           # a cp1252 console cannot print the bolt
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    if a.command == "status":
        from .runner import status
        print(status(a.state))
        return 0

    if a.command == "check":
        from .engine import preview
        from .feeds import Feed
        cfg = load_config(a.config)
        events, subject, text, html, errors = preview(cfg, Feed(), hours=a.hours)
        for m, e in errors.items():
            print(f"  {m}: feed failed -- {e}")
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
