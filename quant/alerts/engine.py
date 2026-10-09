"""
One TICK of the monitor: fetch what is due, judge the bars that closed, queue and send the email.

A tick is idempotent and resumable -- run it every minute, or once after a week switched off, and it
does the same right thing:

1. A market is FETCHED only when one of its timeframes has a bar due to close (15m is the finest
   here, so a market is polled four times an hour, not sixty). Only new bars are requested.
2. A timeframe bar is JUDGED once it is FINAL: closed for `settle_seconds`, and either the 5m data
   reaches its end or the feed has answered for `close_grace_minutes` past it without (a weekend,
   a holiday). The forming bar is never judged, so nothing repaints.
3. Each alert keeps its own CURSOR per market and timeframe: the last bar it judged. The first time
   it sees a pair it starts from NOW -- it never mails you the history it was switched on into.
4. Found more than `catchup_max_age_minutes` after its bar closed (the PC was off), an event is
   LATE: still reported, in a "while offline" section, never as urgent.
5. Urgent events are emailed on the tick that finds them; everything else waits for the digest,
   at most one email per `digest_minutes`. Any email carries EVERY pending event, so the markets
   that fired on the same bar close arrive together.
"""
from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from quant.data.base import TIME_COL

from .compose import compose
from .config import TF_MINUTES, AlertsConfig
from .detectors import REGISTRY
from .feeds import BAR
from .store import State, iso, ts


@dataclass
class TickReport:
    fetched: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    started: List[str] = field(default_factory=list)       # pairs seen for the first time
    events: int = 0
    queued: int = 0
    sent: int = 0


def _tf(tf) -> pd.Timedelta:
    return pd.Timedelta(minutes=TF_MINUTES[tf])


def history_days(cfg: AlertsConfig) -> int:
    """Calendar days of 5m bars to keep: the longest warm-up, x7/5 for weekends, plus slack."""
    need = max(((REGISTRY[a.kind].warmup_bars(a.params) + 5) * TF_MINUTES[tf]
                for a in cfg.alerts for tf in a.tfs), default=0)
    return math.ceil(need / 1440 * 7 / 5) + 3


def tf_bars(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """5m bars -> `tf` bars, buckets on UTC boundaries (4h = 00, 04, 08 ... UTC)."""
    if bars.empty:
        return bars.copy()
    if tf == "5m":
        return bars.reset_index(drop=True)
    key = bars[TIME_COL].dt.floor(f"{TF_MINUTES[tf]}min")
    g = bars.groupby(key, sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum()})
    out.index.name = TIME_COL
    return out.reset_index()


def final_bars(bars: pd.DataFrame, tf: str, now: pd.Timestamp, d) -> pd.DataFrame:
    """The `tf` bars that will not change any more (see the module docstring, step 2). Called
    only on bars the feed has just returned, so `now` is also when the feed last answered."""
    f = tf_bars(bars, tf)
    if f.empty:
        return f
    end = f[TIME_COL] + _tf(tf)
    covered = (bars[TIME_COL].iloc[-1] + BAR >= end) | (
        now >= end + pd.Timedelta(minutes=d.close_grace_minutes))
    ok = (now >= end + pd.Timedelta(seconds=d.settle_seconds)) & covered
    return f[ok.to_numpy()].reset_index(drop=True)


def judge(alert, frame: pd.DataFrame, after: Optional[pd.Timestamp], tf: str):
    """Run the alert's detector over `frame` (final bars only) and return the hits on bars that
    opened after `after`, each as (row of frame, hit dict). Only the warm-up the detector needs is
    passed in, so the cost does not grow with the stored history."""
    det = REGISTRY[alert.kind]
    first = 0 if after is None else int((frame[TIME_COL] > after).to_numpy().argmax())
    lo = max(0, first - det.warmup_bars(alert.params))
    win = frame.iloc[lo:].reset_index(drop=True)
    win["t"] = win[TIME_COL]                         # the research code's time column (UTC)
    hits = []
    for h in det.evaluate(win, alert.params, tf):
        row = win.iloc[h["i"]]
        if after is None or row[TIME_COL] > after:
            hits.append((row, h))
    hits.sort(key=lambda rh: rh[0][TIME_COL])
    return hits


def _due(cfg, state, now) -> List[str]:
    """Markets with a bar due: some pair's next bar has closed, or the pair is new -- unless the
    market is backing off (its feed failed, or had nothing new: a weekend)."""
    settle = pd.Timedelta(seconds=cfg.delivery.settle_seconds)
    due = []
    for (m, tf), alerts in cfg.pairs().items():
        wait = ts(state.get(f"wait_until:{m}"))
        if wait is not None and now < wait:
            continue
        for a in alerts:
            c = state.cursor(a.name, m, tf)
            if c is None or now >= c + 2 * _tf(tf) + settle:
                due.append(m)
                break
    return list(dict.fromkeys(due))


def _backoff(state, m, now, why: Optional[str]):
    """After a failed or empty poll, wait 1, 2, 4 ... 15 minutes before asking again; reset on
    the first poll that brings bars. Bounds a dead feed AND a shut market to ~4 polls an hour."""
    if why is None:
        state.put(f"misses:{m}", 0)
        state.put(f"wait_until:{m}", iso(now))
        return
    n = int(state.get(f"misses:{m}", 0)) + 1
    state.put(f"misses:{m}", n)
    state.put(f"wait_until:{m}", iso(now + pd.Timedelta(minutes=min(2 ** (n - 1), 15))))


def refresh_bars(cfg, state, feed, markets, now, report) -> List[str]:
    """Fetch the new 5m bars of `markets` into the bar store. Returns the markets that answered."""
    keep = pd.Timedelta(days=history_days(cfg))
    starts, stored = {}, {}
    for m in markets:
        b = state.bars(m)
        stored[m] = b
        recent = not b.empty and b[TIME_COL].iloc[-1] >= now - keep
        starts[m] = (b[TIME_COL].iloc[-1] - 2 * BAR) if recent else (now - keep).floor("D")
    got = feed.fetch([cfg.markets[m] for m in markets], starts, now)
    ok = []
    for m, new in got.items():
        if isinstance(new, Exception):
            report.errors[m] = f"{type(new).__name__}: {new}"
            state.put(f"feed_error:{m}", f"{iso(now)} {report.errors[m]}")
            _backoff(state, m, now, "failed")
            continue
        old = stored[m]
        before = None if old.empty else old[TIME_COL].iloc[-1]
        if not new.empty:                            # an empty answer is not merged: its columns
            b = pd.concat([old, new]) if not old.empty else new        # carry no dtypes
            b = b.drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL)
            b = b[b[TIME_COL] >= now - keep].reset_index(drop=True)
            state.save_bars(m, b)
        grew = not new.empty and (before is None or new[TIME_COL].iloc[-1] > before)
        _backoff(state, m, now, None if grew else "nothing new")
        state.put(f"last_ok_poll:{m}", iso(now))
        ok.append(m)
        report.fetched.append(m)
    return ok


def detect(cfg, state, markets, now, report):
    """Judge every new final bar of `markets`; record events and advance cursors."""
    d = cfg.delivery
    late_after = pd.Timedelta(minutes=d.catchup_max_age_minutes)
    for (m, tf), alerts in cfg.pairs().items():
        if m not in markets:
            continue
        frame = final_bars(state.bars(m), tf, now, d)
        if frame.empty:
            continue
        last = frame[TIME_COL].iloc[-1]
        for a in alerts:
            c = state.cursor(a.name, m, tf)
            if c is None:                              # first sight: start from now, mail nothing
                with state:
                    state.set_cursor(a.name, m, tf, last)
                report.started.append(f"{a.name} / {m} / {tf}")
                continue
            if last <= c:
                continue
            try:
                hits = judge(a, frame, c, tf)
            except Exception as e:                     # one broken alert must not stop the rest;
                report.errors[f"{a.name}/{m}/{tf}"] = f"{type(e).__name__}: {e}"
                continue                               # its cursor stays, so it is retried
            cool = a.cooldown_bars * _tf(tf)
            prev = state.last_event_bar(a.name, m, tf)
            with state:
                for row, h in hits:
                    bar = row[TIME_COL]
                    if cool and prev is not None and bar - prev < cool:
                        continue
                    close_time = bar + _tf(tf)
                    if state.add_event(
                            alert=a.name, category=a.category, priority=a.priority, market=m,
                            tf=tf, bar_time=bar, close_time=close_time, what=h["what"],
                            side=h.get("side", ""), price=row["close"],
                            detail={k: v for k, v in h.items() if k not in ("i", "what", "side")}
                            | {"tv": cfg.markets[m].tv},
                            created=now, late=now - close_time > late_after):
                        report.events += 1
                        prev = bar
                state.set_cursor(a.name, m, tf, last)


def deliver(cfg, state, notifier, now, report):
    """Queue an email if one is due, then send everything in the outbox that is due."""
    pending = state.pending()
    if pending:
        urgent = any(e["priority"] == "urgent" and not e["late"] for e in pending)
        last = ts(state.get("last_email"))
        digest = last is None or now - last >= pd.Timedelta(minutes=cfg.delivery.digest_minutes)
        if urgent or digest:
            subject, text, html = compose(pending, cfg, now, host=socket.gethostname())
            with state:
                state.queue_email(subject, text, html, [e["id"] for e in pending], now)
                state.put("last_email", iso(now))
            report.queued += 1
    for mail in state.due_emails(now):
        try:
            notifier.send(mail["subject"], mail["text"], mail["html"])
        except Exception as e:
            state.mark_failed(mail["id"], now, f"{type(e).__name__}: {e}")
            report.errors[f"email {mail['id']}"] = f"{type(e).__name__}: {e}"
            break                                      # the server is down; do not hammer it
        with state:
            state.mark_sent(mail["id"], now)
        report.sent += 1


def tick(cfg: AlertsConfig, state: State, notifier, feed, now=None) -> TickReport:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    report = TickReport()
    # an alert switched OFF (or a market / timeframe taken out of it) loses its cursor, so
    # switching it back on starts from that moment instead of mailing everything in between.
    # A PC that was off keeps its cursors -- that gap IS caught up, as "while offline".
    with state:
        state.keep_cursors({(a.name, m, tf) for (m, tf), al in cfg.pairs().items() for a in al})
    due = _due(cfg, state, now)
    if due:
        ok = refresh_bars(cfg, state, feed, due, now, report)
        detect(cfg, state, ok, now, report)
    deliver(cfg, state, notifier, now, report)
    state.put("heartbeat", iso(now))
    return report


def preview(cfg: AlertsConfig, feed, *, hours: float = 48, now=None):
    """What WOULD have fired over the last `hours`, as the email it would have sent -- without
    touching the monitor's state. Returns (events, subject, text, html, feed errors)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    keep = pd.Timedelta(days=history_days(cfg))
    start = (now - keep).floor("D")
    got = feed.fetch(list(cfg.markets.values()), {m: start for m in cfg.markets}, now)
    since = now - pd.Timedelta(hours=hours)
    events = []
    for (m, tf), alerts in cfg.pairs().items():
        bars = got.get(m)
        if bars is None or isinstance(bars, Exception) or bars.empty:
            continue
        frame = final_bars(bars, tf, now, cfg.delivery)
        for a in alerts:
            prev, cool = None, a.cooldown_bars * _tf(tf)
            for row, h in judge(a, frame, None, tf):
                bar = row[TIME_COL]
                if cool and prev is not None and bar - prev < cool:
                    continue
                prev = bar
                if bar + _tf(tf) < since:
                    continue
                events.append(dict(alert=a.name, category=a.category, priority=a.priority,
                                   market=m, tf=tf, bar_time=iso(bar),
                                   close_time=iso(bar + _tf(tf)), what=h["what"],
                                   side=h.get("side", ""), price=float(row["close"]),
                                   detail=json.dumps({"tv": cfg.markets[m].tv,
                                                      "note": h.get("note", "")}), late=0))
    errors = {m: f"{type(v).__name__}: {v}" for m, v in got.items() if isinstance(v, Exception)}
    if not events:
        return events, None, None, None, errors
    subject, text, html = compose(events, cfg, now, host=socket.gethostname(),
                                  title=f"Preview: the last {hours:g} hours")
    return events, subject, text, html, errors
