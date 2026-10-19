"""
The monitor loop: one tick a minute, forever, and only ever one copy of it.

Windows Task Scheduler starts it at logon and re-starts it every 5 minutes (scripts/
install_alerts_task.ps1). A second copy finds the lock held and exits at once, so the schedule is a
watchdog, not a duplicate: if the loop dies, the next trigger brings it back, and its first tick
catches up on whatever closed while it was gone.

The config is re-read every tick; a broken edit is logged and the last good config kept, so a typo
cannot stop the monitor.
"""
from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from .config import ConfigError, load_config

log = logging.getLogger("quant.alerts")


def acquire_lock(path: Path):
    """An open file holding an exclusive OS lock, or None if another process holds it. The OS
    drops the lock when the process dies, however it dies -- no stale lock file to clean up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)                                        # who holds it, for the one that is turned away
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def lock_holder(path: Path) -> str:
    """The pid written by whoever holds the lock, or '?'."""
    try:
        return Path(path).read_text().strip() or "?"
    except OSError:
        return "?"


def setup_logging(state_dir: Path, *, console: bool = False):
    state_dir.mkdir(parents=True, exist_ok=True)
    if log.handlers:
        return
    h = RotatingFileHandler(state_dir / "alerts.log", maxBytes=1_000_000, backupCount=3,
                            encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    if console:
        log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)


def log_report(r):
    if r.started:
        log.info("watching (from now on): %s", "; ".join(r.started))
    if r.fetched or r.events or r.queued or r.sent:
        log.info("fetched %s | %d new events | %d emails queued, %d sent",
                 ",".join(r.fetched) or "-", r.events, r.queued, r.sent)
    for k, v in r.errors.items():
        log.warning("%s: %s", k, v)


def run_forever(config_path: Path, state_dir: Path, notifier, feed, *, offset_s: int = 20):
    from .engine import tick
    from .store import State
    from .news import LiveNews
    lock = acquire_lock(state_dir / "alerts.lock")
    if lock is None:                                   # already running: the watchdog's no-op --
        print(f"quant alerts: another monitor (pid {lock_holder(state_dir / 'alerts.lock')}) "
              "is already running; this copy exits", flush=True)       # said, not silent
        return 0
    state = None
    try:
        setup_logging(state_dir)
        log.info("monitor started (pid %d)", os.getpid())
        state, cfg = State(state_dir), None
        news = LiveNews(state_dir / "news")             # one calendar, kept across ticks
        while True:
            try:
                cfg = load_config(config_path)
            except ConfigError as e:
                log.error("config not loaded, %s: %s", "keeping the last good one" if cfg else
                          "waiting for a valid one", e)
            if cfg is not None:
                try:
                    log_report(tick(cfg, state, notifier, feed, news=news))
                except Exception:
                    log.exception("tick failed")
            now = time.time()                          # next tick: offset_s past the next minute
            nxt = (now // 60) * 60 + offset_s
            time.sleep(max(1.0, nxt + 60 - now if nxt <= now else nxt - now))
    finally:
        if state is not None:
            state.close()
        lock.close()


def status(state_dir: Path) -> str:
    from .store import State
    if not (state_dir / "alerts.db").exists():
        return "never run: no state yet"
    s = State(state_dir)
    now = pd.Timestamp.now(tz="UTC")
    beat = s.get("heartbeat")
    age = "" if beat is None else f" ({(now - pd.Timestamp(beat)).total_seconds() / 60:.0f} min ago)"
    lines = [f"heartbeat   {beat}{age}", f"last email  {s.get('last_email')}"]
    lock = acquire_lock(state_dir / "alerts.lock")
    lines.append("monitor     " + ("NOT running" if lock else "running"))
    if lock:
        lock.close()
    q = s.db.execute("SELECT status, COUNT(*) FROM outbox GROUP BY status").fetchall()
    lines.append("outbox      " + (", ".join(f"{n} {k}" for k, n in q) or "empty"))
    q = s.db.execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()
    lines.append("events      " + (", ".join(f"{n} {k}" for k, n in q) or "none yet"))
    err = s.db.execute("SELECT key, value FROM meta WHERE key LIKE 'feed_error:%'").fetchall()
    lines += [f"{k:<11} {v}" for k, v in err]
    if (state_dir / "news").is_dir():
        from quant.news.calendar import NewsStore
        m = NewsStore(state_dir / "news").meta()
        ok = m.get("feed_ok")
        ago = "" if ok is None else f" ({(now - pd.Timestamp(ok)).total_seconds() / 60:.0f} min ago)"
        lines.append(f"news feed   last updated {ok}{ago}")
        for k in ("feed_error", "jb_ok", "fred_ok", "jb_zone"):
            if m.get(k):
                lines.append(f"news        {k} {m[k]}")
    lines.append("cursors")
    for r in s.db.execute("SELECT * FROM cursors ORDER BY market, tf, alert"):
        lines.append(f"  {r['market']:<8} {r['tf']:>4}  {r['alert']:<28} last judged bar {r['last_bar']}")
    s.close()
    return "\n".join(lines)
