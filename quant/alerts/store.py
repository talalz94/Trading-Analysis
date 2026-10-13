"""
Everything the monitor must remember across a crash, a reboot or a week switched off.

  state/alerts.db           SQLite (WAL): cursors, events, outbox, meta -- see SCHEMA
  state/bars/<market>.parquet   the recent 5m bars of each market, rewritten atomically

The rules that make a restart safe:
- an EVENT's id is a hash of (alert, market, timeframe, bar, what), and inserting it twice is a
  no-op -- so re-evaluating a bar after a crash can never send its alert again;
- events and the cursor that covers them are written in ONE transaction;
- an email is written to the OUTBOX before anything is sent, and marked sent only after the SMTP
  server accepted it -- a failed or interrupted send is retried, with backoff, on the next tick;
- a bar file is written to a temporary name and renamed over the old one, so it is either the old
  file or the new one, never half of each.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from quant.data.base import TIME_COL

SCHEMA = """
CREATE TABLE IF NOT EXISTS cursors (
    alert TEXT, market TEXT, tf TEXT,
    last_bar TEXT,                       -- open time of the last bar this alert has judged
    PRIMARY KEY (alert, market, tf));
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    alert TEXT, category TEXT, priority TEXT, market TEXT, tf TEXT,
    bar_time TEXT, close_time TEXT,      -- the bar's open / close (UTC ISO)
    what TEXT, side TEXT, price REAL, detail TEXT,
    created TEXT, late INTEGER DEFAULT 0,
    status TEXT DEFAULT 'new',           -- new -> queued -> sent
    outbox_id INTEGER);
CREATE INDEX IF NOT EXISTS events_status ON events(status);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, subject TEXT, text TEXT, html TEXT,
    attempts INTEGER DEFAULT 0, next_try TEXT, sent_at TEXT, last_error TEXT,
    status TEXT DEFAULT 'pending');      -- pending -> sent | failed
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").isoformat()


def ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    return None if s is None else pd.Timestamp(s).tz_convert("UTC")


def event_id(alert, market, tf, bar_time, what) -> str:
    return hashlib.sha1(f"{alert}|{market}|{tf}|{iso(bar_time)}|{what}".encode()).hexdigest()


class State:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "bars").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / "alerts.db"), timeout=30,
                                  isolation_level=None)          # explicit BEGIN/COMMIT below
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # ── transactions ───────────────────────────────────────────────────────────────────────
    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, kind, *_):
        self.db.execute("ROLLBACK" if kind else "COMMIT")

    # ── meta ───────────────────────────────────────────────────────────────────────────────
    def get(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if r is None else r[0]

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    # ── cursors ────────────────────────────────────────────────────────────────────────────
    def cursor(self, alert, market, tf) -> Optional[pd.Timestamp]:
        r = self.db.execute("SELECT last_bar FROM cursors WHERE alert=? AND market=? AND tf=?",
                            (alert, market, tf)).fetchone()
        return None if r is None else ts(r[0])

    def keep_cursors(self, keep):
        """Delete every cursor whose (alert, market, tf) is not in `keep`."""
        gone = [tuple(r) for r in self.db.execute("SELECT alert, market, tf FROM cursors")
                if tuple(r) not in keep]
        self.db.executemany("DELETE FROM cursors WHERE alert=? AND market=? AND tf=?", gone)
        return len(gone)

    def set_cursor(self, alert, market, tf, bar):
        self.db.execute("INSERT OR REPLACE INTO cursors VALUES (?, ?, ?, ?)",
                        (alert, market, tf, iso(bar)))

    # ── events ─────────────────────────────────────────────────────────────────────────────
    def add_event(self, *, alert, category, priority, market, tf, bar_time, close_time, what,
                  side, price, detail, created, late, status="new") -> bool:
        """True if new; False if this exact event was already recorded. `status="skipped"`
        records an event that must never be mailed (a news reminder overtaken by a later one)."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO events (id, alert, category, priority, market, tf, bar_time,"
            " close_time, what, side, price, detail, created, late, status) VALUES"
            " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (event_id(alert, market, tf, bar_time, what), alert, category, priority, market, tf,
             iso(bar_time), iso(close_time), what, side, float(price), json.dumps(detail),
             iso(created), int(late), status))
        return cur.rowcount == 1

    def last_event_bar(self, alert, market, tf) -> Optional[pd.Timestamp]:
        r = self.db.execute("SELECT MAX(bar_time) FROM events WHERE alert=? AND market=? AND tf=?",
                            (alert, market, tf)).fetchone()
        return ts(r[0]) if r and r[0] else None

    def pending(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM events WHERE status='new' ORDER BY close_time, market, tf")]

    # ── outbox ─────────────────────────────────────────────────────────────────────────────
    def queue_email(self, subject, text, html, event_ids, now) -> int:
        cur = self.db.execute("INSERT INTO outbox (created, subject, text, html, next_try)"
                              " VALUES (?,?,?,?,?)", (iso(now), subject, text, html, iso(now)))
        oid = cur.lastrowid
        self.db.executemany("UPDATE events SET status='queued', outbox_id=? WHERE id=?",
                            [(oid, e) for e in event_ids])
        return oid

    def due_emails(self, now):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM outbox WHERE status='pending' AND next_try<=? ORDER BY id", (iso(now),))]

    def mark_sent(self, oid, now):
        self.db.execute("UPDATE outbox SET status='sent', sent_at=? WHERE id=?", (iso(now), oid))
        self.db.execute("UPDATE events SET status='sent' WHERE outbox_id=?", (oid,))

    def mark_failed(self, oid, now, error, *, max_attempts=20):
        r = self.db.execute("SELECT attempts FROM outbox WHERE id=?", (oid,)).fetchone()
        n = r[0] + 1
        wait = pd.Timedelta(minutes=min(2 ** (n - 1), 30))          # 1, 2, 4 ... 30 minutes
        status = "failed" if n >= max_attempts else "pending"
        self.db.execute("UPDATE outbox SET attempts=?, next_try=?, last_error=?, status=?"
                        " WHERE id=?", (n, iso(now + wait), str(error)[:500], status, oid))

    # ── 5m bars ────────────────────────────────────────────────────────────────────────────
    def _bar_path(self, market) -> Path:
        return self.root / "bars" / (re.sub(r"[^A-Za-z0-9_-]+", "_", market) + ".parquet")

    def bars(self, market) -> pd.DataFrame:
        p = self._bar_path(market)
        if not p.exists():
            return pd.DataFrame(columns=[TIME_COL, "open", "high", "low", "close", "volume"])
        return pd.read_parquet(p)

    def save_bars(self, market, df: pd.DataFrame):
        p = self._bar_path(market)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
