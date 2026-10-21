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
6. With `every_minutes` (DELIVERY) the clock decides instead: one email per boundary (15 = :00 :15
   :30 :45 UTC), sent once that boundary's bars are final and carrying everything found since --
   urgent alerts and news included. A timeframe finer than the boundary is fetched and judged only
   there, so five 1m candles through a line in one quarter hour arrive in ONE email.
7. A 1m alert reads 1m bars, fetched and stored apart from the 5m ones (`base_of`); every other
   timeframe is still resampled from 5m.
"""
from __future__ import annotations

import json
import math
import socket
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from quant.data.base import TIME_COL

from .compose import compose
from .config import TF_MINUTES, AlertsConfig, base_of
from .detectors import REGISTRY
from .feeds import BASE_BAR
from .store import State, iso, ts


@dataclass
class TickReport:
    fetched: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    started: List[str] = field(default_factory=list)       # pairs seen for the first time
    events: int = 0
    queued: int = 0
    sent: int = 0
    after: List[Callable] = field(default_factory=list)   # news work to run once candles are out


def _tf(tf) -> pd.Timedelta:
    return pd.Timedelta(minutes=TF_MINUTES[tf])


def history_days(cfg: AlertsConfig, base: str = "5m") -> int:
    """Calendar days of `base` bars to keep: the longest warm-up of the timeframes built from them,
    x7/5 for weekends, plus slack."""
    need = max(((REGISTRY[a.kind].warmup_bars(a.params) + 5) * TF_MINUTES[tf]
                for a in cfg.alerts for tf in a.tfs if base_of(tf) == base), default=0)
    return math.ceil(need / 1440 * 7 / 5) + 3


def tf_bars(bars: pd.DataFrame, tf: str, base: str = "5m") -> pd.DataFrame:
    """`base` bars -> `tf` bars, buckets on UTC boundaries (4h = 00, 04, 08 ... UTC)."""
    if bars.empty:
        return bars.copy()
    if tf == base:
        return bars.reset_index(drop=True)
    key = bars[TIME_COL].dt.floor(f"{TF_MINUTES[tf]}min")
    g = bars.groupby(key, sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum()})
    out.index.name = TIME_COL
    return out.reset_index()


def final_bars(bars: pd.DataFrame, tf: str, now: pd.Timestamp, d, base: str = "5m") -> pd.DataFrame:
    """The `tf` bars that will not change any more (see the module docstring, step 2). Called
    only on bars the feed has just returned, so `now` is also when the feed last answered."""
    f = tf_bars(bars, tf, base)
    if f.empty:
        return f
    end = f[TIME_COL] + _tf(tf)
    covered = (bars[TIME_COL].iloc[-1] + BASE_BAR[base] >= end) | (
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


def _feed_key(m: str, base: str) -> str:
    """How a market's FEED is named in the state: the 5m one keeps its old plain name."""
    return m if base == "5m" else f"{m}@{base}"


def _due(cfg, state, now) -> List[Tuple[str, str]]:
    """(market, base) feeds with a bar due: some pair's next bar has closed, or the pair is new --
    unless that feed is backing off (it failed, or had nothing new: a weekend). With
    `every_minutes`, a bar finer than the boundary waits for the boundary it closes before."""
    settle = pd.Timedelta(seconds=cfg.delivery.settle_seconds)
    cad = cfg.delivery.every_minutes
    due = []
    for (m, tf), alerts in cfg.pairs().items():
        base = base_of(tf)
        if (m, base) in due:
            continue
        wait = ts(state.get(f"wait_until:{_feed_key(m, base)}"))
        if wait is not None and now < wait:
            continue
        for a in alerts:
            c = state.cursor(a.name, m, tf)
            if c is None:
                due.append((m, base))
                break
            closes = c + 2 * _tf(tf)
            if cad:
                closes = closes.ceil(f"{cad}min")
            if now >= closes + settle:
                due.append((m, base))
                break
    return due


def _backoff(state, m, now, why: Optional[str]):
    """After a failed or empty poll, wait 1, 2, 4 ... 15 minutes before asking again; reset on
    the first poll that brings bars. Bounds a dead feed AND a shut market to ~4 polls an hour.
    `m` is the feed's key (_feed_key)."""
    if why is None:
        state.put(f"misses:{m}", 0)
        state.put(f"wait_until:{m}", iso(now))
        return
    n = int(state.get(f"misses:{m}", 0)) + 1
    state.put(f"misses:{m}", n)
    state.put(f"wait_until:{m}", iso(now + pd.Timedelta(minutes=min(2 ** (n - 1), 15))))


def refresh_bars(cfg, state, feed, due, now, report) -> List[Tuple[str, str]]:
    """Fetch the new bars of each (market, base) in `due` into the bar store. Returns the feeds
    that answered."""
    ok = []
    for base in dict.fromkeys(b for _, b in due):
        markets = [m for m, b in due if b == base]
        keep = pd.Timedelta(days=history_days(cfg, base))
        bar = BASE_BAR[base]
        starts, stored, deeper = {}, {}, {}
        want = (now - keep).floor("D")
        for m in markets:
            b = state.bars(m, base)
            stored[m] = b
            recent = not b.empty and b[TIME_COL].iloc[-1] >= now - keep
            starts[m] = (b[TIME_COL].iloc[-1] - 2 * bar) if recent else want
            # a longer warm-up (a new alert, a new release) wants history the store never had:
            # fetched ONCE from where it should start. Marked once it is in, so a market with
            # nothing older to give (a young listing) is not asked for months every tick
            mark = state.get(f"history_from:{_feed_key(m, base)}")
            if recent and b[TIME_COL].iloc[0] > want + pd.Timedelta(days=3) and (
                    mark is None or pd.Timestamp(mark) > want + pd.Timedelta(days=3)):
                starts[m] = deeper[m] = want
        ms = [cfg.markets[m] for m in markets]
        # the 5m call is made exactly as it always was, so any feed written for it still works
        got = (feed.fetch(ms, starts, now) if base == "5m"
               else feed.fetch(ms, starts, now, base=base))
        for m, new in got.items():
            key = _feed_key(m, base)
            label = m if base == "5m" else f"{m} {base}"
            if isinstance(new, Exception):
                report.errors[label] = f"{type(new).__name__}: {new}"
                state.put(f"feed_error:{key}", f"{iso(now)} {report.errors[label]}")
                _backoff(state, key, now, "failed")
                continue
            old = stored[m]
            before = None if old.empty else old[TIME_COL].iloc[-1]
            if not new.empty:                        # an empty answer is not merged: its columns
                b = pd.concat([old, new]) if not old.empty else new    # carry no dtypes
                b = b.drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL)
                b = b[b[TIME_COL] >= now - keep].reset_index(drop=True)
                state.save_bars(m, b, base)
            if m in deeper:
                state.put(f"history_from:{key}", iso(deeper[m]))
            grew = not new.empty and (before is None or new[TIME_COL].iloc[-1] > before)
            _backoff(state, key, now, None if grew else "nothing new")
            state.put(f"last_ok_poll:{key}", iso(now))
            ok.append((m, base))
            report.fetched.append(label)
    return ok


def detect(cfg, state, fed, now, report, news_note=None):
    """Judge every new final bar of the (market, base) feeds in `fed`; record events and advance
    cursors. `news_note` (from the news step) adds "news: USD CPI m/m in 12 min" to an alert
    closing near news."""
    d = cfg.delivery
    late_after = pd.Timedelta(minutes=d.catchup_max_age_minutes)
    for (m, tf), alerts in cfg.pairs().items():
        base = base_of(tf)
        if (m, base) not in fed:
            continue
        frame = final_bars(state.bars(m, base), tf, now, d, base)
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
                    detail = ({k: v for k, v in h.items() if k not in ("i", "what", "side")}
                              | {"tv": cfg.markets[m].tv})
                    nn = news_note(close_time) if news_note else ""
                    if nn:
                        detail["note"] = f"{detail['note']} · {nn}" if detail.get("note") else nn
                    if state.add_event(
                            alert=a.name, category=a.category, priority=a.priority, market=m,
                            tf=tf, bar_time=bar, close_time=close_time, what=h["what"],
                            side=h.get("side", ""), price=row["close"], detail=detail,
                            created=now, late=now - close_time > late_after):
                        report.events += 1
                        prev = bar
                state.set_cursor(a.name, m, tf, last)


def _email_due(cfg, state, now, pending) -> bool:
    """Is an email due now? Without `every_minutes`: an urgent event, or the digest is up. With it:
    the first look after a boundary's bars are final, once per boundary -- whether or not anything
    is waiting, so an event found later in the quarter hour waits for the next boundary."""
    d = cfg.delivery
    if d.every_minutes:
        slot = now.floor(f"{d.every_minutes}min")
        if now < slot + pd.Timedelta(seconds=d.settle_seconds) or state.get("email_slot") == iso(slot):
            return False
        state.put("email_slot", iso(slot))
        return bool(pending)
    if not pending:
        return False
    urgent = any(e["priority"] == "urgent" and not e["late"] for e in pending)
    last = ts(state.get("last_email"))
    return urgent or last is None or now - last >= pd.Timedelta(minutes=d.digest_minutes)


def deliver(cfg, state, notifier, now, report):
    """Queue an email if one is due, then send everything in the outbox that is due."""
    pending = state.pending()
    if pending or cfg.delivery.every_minutes:
        if _email_due(cfg, state, now, pending):
            subject, text, html = compose(pending, cfg, now, host=socket.gethostname())
            images = None
            if cfg.delivery.charts:                    # pictures of the top rows; never blocks
                from .snapshot import attach, email_charts, pack
                charts, err = email_charts(pending, cfg, state.bars, now)
                if err:
                    report.errors["charts"] = err
                html, images = attach(html, charts), pack(charts)
            with state:
                state.queue_email(subject, text, html, [e["id"] for e in pending], now, images)
                state.put("last_email", iso(now))
            report.queued += 1
    for mail in state.due_emails(now):
        try:
            if mail.get("images"):
                from .snapshot import unpack
                notifier.send(mail["subject"], mail["text"], mail["html"],
                              images=unpack(mail["images"]))
            else:                                      # exactly the call it always was
                notifier.send(mail["subject"], mail["text"], mail["html"])
        except Exception as e:
            state.mark_failed(mail["id"], now, f"{type(e).__name__}: {e}")
            report.errors[f"email {mail['id']}"] = f"{type(e).__name__}: {e}"
            break                                      # the server is down; do not hammer it
        with state:
            state.mark_sent(mail["id"], now)
        report.sent += 1


def tick(cfg: AlertsConfig, state: State, notifier, feed, now=None, news=None) -> TickReport:
    """`news` is the calendar the NEWS block reads (quant.alerts.news.LiveNews, kept by the
    runner across ticks); built here when the block is on and none is given."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    report = TickReport()
    note = None
    if cfg.news.on:
        from .news import LiveNews, news_step
        news = news or LiveNews(state.root / "news")
        try:
            # no transaction around it: it fetches over the network, and every write it makes is
            # an INSERT OR IGNORE that is safe to repeat after a crash
            note = news_step(cfg, state, news, now, report)
        except Exception as e:                         # news must never stop the candle alerts
            report.errors["news"] = f"{type(e).__name__}: {e}"
    # an alert switched OFF (or a market / timeframe taken out of it) loses its cursor, so
    # switching it back on starts from that moment instead of mailing everything in between.
    # A PC that was off keeps its cursors -- that gap IS caught up, as "while offline".
    with state:
        state.keep_cursors({(a.name, m, tf) for (m, tf), al in cfg.pairs().items() for a in al})
    due = _due(cfg, state, now)
    if due:
        ok = refresh_bars(cfg, state, feed, due, now, report)
        detect(cfg, state, ok, now, report, news_note=note)
    every = cfg.delivery.every_minutes
    if not every:
        deliver(cfg, state, notifier, now, report)
    if report.after:
        # news results still waiting for their number: asked only now that the candle emails are
        # out, and whatever came in is sent in this same tick
        for fn in report.after:
            try:
                fn()
            except Exception as e:
                report.errors["news results"] = f"{type(e).__name__}: {e}"
    if every or report.after:
        # on a clock, ONE email per boundary: the results asked for above ride in it
        deliver(cfg, state, notifier, now, report)
    if cfg.news.on and hasattr(news, "background"):
        # the slow fetches nothing in this tick waits for -- after the candle emails are out
        try:
            report.errors.update(news.background(now) or {})
        except Exception as e:
            report.errors["news background"] = f"{type(e).__name__}: {e}"
    state.put("heartbeat", iso(now))
    return report


def preview(cfg: AlertsConfig, feed, *, hours: float = 48, now=None):
    """What WOULD have fired over the last `hours`, as the email it would have sent -- without
    touching the monitor's state. Returns (events, subject, text, html, feed errors)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    got, errors = {}, {}
    for base in dict.fromkeys(base_of(tf) for _, tf in cfg.pairs()):
        keep = pd.Timedelta(days=history_days(cfg, base))
        start = (now - keep).floor("D")
        names = list(dict.fromkeys(m for m, tf in cfg.pairs() if base_of(tf) == base))
        ms = [cfg.markets[m] for m in names]          # only the markets this base is needed for
        got[base] = (feed.fetch(ms, {m: start for m in names}, now) if base == "5m"
                     else feed.fetch(ms, {m: start for m in names}, now, base=base))
        errors.update({(m if base == "5m" else f"{m} {base}"): f"{type(v).__name__}: {v}"
                       for m, v in got[base].items() if isinstance(v, Exception)})
    since = now - pd.Timedelta(hours=hours)
    events = []
    for (m, tf), alerts in cfg.pairs().items():
        base = base_of(tf)
        bars = got[base].get(m)
        if bars is None or isinstance(bars, Exception) or bars.empty:
            continue
        frame = final_bars(bars, tf, now, cfg.delivery, base)
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
    if not events:
        return events, None, None, None, errors
    subject, text, html = compose(events, cfg, now, host=socket.gethostname(),
                                  title=f"Preview: the last {hours:g} hours")
    if cfg.delivery.charts:                            # in a local file the pictures are inline
        from .snapshot import attach, email_charts
        charts, err = email_charts(events, cfg, lambda m, b: got[b][m], now)
        if err:
            errors["charts"] = err
        html = attach(html, charts, inline="data")
    return events, subject, text, html, errors
