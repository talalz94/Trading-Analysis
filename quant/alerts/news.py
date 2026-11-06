"""
NEWS IN THE ALERTS -- notices driven by the CLOCK and the calendar, not by candles.

What it sends (each switchable in the NEWS block of alerts/alerts_config.py):

  week ahead    once a week (default Sunday 20:00 your time): the coming 7 days' high and
                critical releases. Waits for Forex Factory to publish the new week, up to 11 h.
  day ahead     every evening (default 20:00): tomorrow's covered releases, with forecast and
                previous. Nothing to report -> no email.
  reminders     N minutes before a release (default 60 and 15 for critical, 15 for high),
                urgent. Releases at the same minute in one currency are ONE reminder.
  results       the actual against the forecast, the surprise and what it means for the currency,
                once JBlanked has the number (needs JBLANKED_API_KEY; polled every 5 minutes, and
                only while a covered release is waiting for its number).
  headlines     the result as the squawk feeds TYPE it, seconds after the release
                (quant/news/headlines.py): FinancialJuice's feed first, its Telegram channel and
                Walter Bloomberg's when the feed fails or has nothing yet, the first lines quoted
                word for word with their source. Asked every tick inside the release's window
                (10 min; 2 h for the BOJ), once per release. A US number from the agencies that
                the headline already printed is not mailed again; one that differs is.
  breaking      critical news that is NOT on the calendar (quant/news/headlines.py): a TOP line
                (an ordered departure, an embassy closed, an emergency central-bank move,
                intervention, war declared) goes at once; a BREAKING line (attacks on Gulf
                targets, Hormuz shut, new tariffs, an exchange hack...) only when Gold, USDJPY or
                BTC moves sharply from 10 min before it to 15 min after, read off the monitor's
                own 1m candles. One per story an hour, at most 8 a day. Measured first:
                experiments/breaking_news_check.py.
  instant       reminders, results and moves go the moment they are due, not at the next
                DELIVERY boundary.
  moved         a covered high/critical release that changed its time or left the calendar.
  feed          the calendar has not updated for 6 hours -- reminders may be out of date.
  warn_alerts   a CANDLE alert closing within N minutes of a high release says so in its note.

The rules that keep it trustworthy -- each tested:
  - every notice is an event in the same table as the candle alerts, with an id hashed from what it
    is ABOUT (the release time, the currency, which reminder), so a restart, a re-run or a second
    look can never send it twice;
  - a reminder is only sent BEFORE its release. After an outage only the latest one still ahead
    goes out ("in 3 min"), and the earlier ones are recorded as skipped, never sent late;
  - a digest older than its window (6 h for the day, 12 h for the week) is not sent at all: Monday's
    "tomorrow" email arriving on Tuesday is worse than none;
  - times are shown in YOUR zone and in New York, because the calendar speaks New York.
"""
from __future__ import annotations

import html as _html
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from quant.logging_utils import redact
from quant.news.catalog import LEVELS, RANK, parse_coverage

ALERT = "news"                       # the alert name every notice is stored under (reserved)
CATEGORY = "News"
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
LEVEL_MARK = {"critical": "[critical]", "high": "[high]", "medium": "[medium]", "low": "[low]",
              "none": ""}
LEVEL_COLOUR = {"critical": "#b91c1c", "high": "#dc2626", "medium": "#ea580c", "low": "#ca8a04",
                "none": "#6b7280"}
DAY_WINDOW = pd.Timedelta(hours=6)
WEEK_WINDOW = pd.Timedelta(hours=12)
WEEK_WAIT = pd.Timedelta(hours=11)   # how long the week-ahead waits for the new week's calendar
DAY_WAIT = pd.Timedelta(hours=5)
STALE = pd.Timedelta(hours=6)
RESULT_WINDOW = pd.Timedelta(hours=3)   # how long a missing number is asked for (FRED can post
                                         # an hour after the release)
RESULT_FRESH = pd.Timedelta(minutes=60)  # a number older than this goes out only if THIS monitor
                                         # was already waiting for it -- never after an outage
RESULT_SETTLE = pd.Timedelta(minutes=20)  # send a partial result after this if the rest never come
BREAKING_LOOKBACK = pd.Timedelta(minutes=15)   # a line older than this when first seen is not
                                               # news any more (a first poll's backlog, an outage)
BREAKING_SEEN = pd.Timedelta(hours=3)          # how long a line's id is remembered
HEADLINE_LINES = 3                       # lines quoted per release: the first is not always THE
                                         # number (on GDP day FinancialJuice led with PCE)


# ══ CONFIG ═══════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class NewsConfig:
    on: bool = False
    currencies: Dict[str, str] = field(default_factory=dict)
    timezone: str = "Asia/Karachi"
    week_ahead: Optional[Tuple[int, str]] = (6, "20:00")
    day_ahead: Optional[str] = "20:00"
    remind_before: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    results: Optional[str] = "high"
    warn_alerts: int = 30
    headlines: Tuple[str, ...] = ()                # quant.news.headlines sources, first = main
    instant: bool = False
    breaking: Tuple[str, ...] = ()                 # the same sources, for unscheduled news


SETTINGS = {
    "on": "True = news notices on",
    "currencies": "{currency: lowest level kept}, e.g. {'USD': 'medium', 'EUR': 'high'}; '*' = "
                  "every other currency",
    "timezone": "the zone your emails show times in, e.g. 'Asia/Karachi'",
    "week_ahead": "'Sun 20:00' -- the week's high/critical news, once a week; None = off",
    "day_ahead": "'20:00' -- tomorrow's covered news, every evening; None = off",
    "remind_before": "{level: minutes before}, e.g. {'critical': (60, 15), 'high': (15,)}",
    "results": "lowest level whose actual-vs-forecast is emailed (needs JBLANKED_API_KEY); None = off",
    "warn_alerts": "a candle alert within this many minutes of high news says so; 0 = off",
    "headlines": "the result from news headlines, in order of trust: ('financialjuice', "
                 "'financialjuice_telegram', 'walterbloomberg') -- the next is asked when one "
                 "fails or has nothing yet; None = off",
    "instant": "True = reminders, results and moves go at once, not at the next email boundary",
    "breaking": "critical news NOT on the calendar, from these sources in order of trust: "
                "('financialjuice', 'financialjuice_telegram'); a TOP line goes at once, a BREAKING "
                "line only when an alert market moves sharply; None = off",
}


class NewsConfigError(ValueError):
    pass


def _hhmm(s, what) -> str:
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(s)):
        raise NewsConfigError(f"NEWS {what}: expected 'HH:MM', got {s!r}")
    return str(s)


def parse_news(d: Optional[dict]) -> NewsConfig:
    """The config file's NEWS dict -> NewsConfig, every mistake refused by name."""
    if not d:
        return NewsConfig()
    d = dict(d)
    bad = sorted(set(d) - set(SETTINGS))
    if bad:
        raise NewsConfigError(f"NEWS has unknown settings {bad}; known: {sorted(SETTINGS)}")
    try:
        cov = parse_coverage(d.get("currencies"))
    except ValueError as e:
        raise NewsConfigError(str(e)) from None
    tz = str(d.get("timezone", "Asia/Karachi"))
    try:
        pd.Timestamp("2026-01-01").tz_localize(tz)
    except Exception:
        raise NewsConfigError(f"NEWS timezone: unknown zone {tz!r} (e.g. 'Asia/Karachi')") from None
    wa = d.get("week_ahead", "Sun 20:00")
    if wa is not None:
        m = re.fullmatch(r"\s*([A-Za-z]{3})\w*\s+(\S+)\s*", str(wa))
        if not m or m.group(1).lower() not in _DAYS:
            raise NewsConfigError(f"NEWS week_ahead: expected like 'Sun 20:00', got {wa!r}")
        wa = (_DAYS.index(m.group(1).lower()), _hhmm(m.group(2), "week_ahead"))
    da = d.get("day_ahead", "20:00")
    da = None if da is None else _hhmm(da, "day_ahead")
    rb = d.get("remind_before", {"critical": (60, 15), "high": (15,)}) or {}
    rem = {}
    for lv, mins in dict(rb).items():
        if lv not in RANK or lv == "none":
            raise NewsConfigError(f"NEWS remind_before: level must be one of {LEVELS[1:]}, "
                                  f"got {lv!r}")
        mins = (mins,) if isinstance(mins, (int, float)) else tuple(mins)
        if not mins or any(not (isinstance(x, int) and 0 < x <= 2880) for x in mins):
            raise NewsConfigError(f"NEWS remind_before[{lv!r}]: whole minutes 1..2880, got {mins}")
        rem[lv] = tuple(sorted(set(mins), reverse=True))
    res = d.get("results", "high")
    if res is not None and res not in RANK:
        raise NewsConfigError(f"NEWS results: a level of {LEVELS} or None, got {res!r}")
    warn = d.get("warn_alerts", 30)
    if not isinstance(warn, int) or warn < 0:
        raise NewsConfigError(f"NEWS warn_alerts: whole minutes >= 0, got {warn!r}")
    from quant.news.headlines import SOURCES
    hl = d.get("headlines") or ()
    hl = (hl,) if isinstance(hl, str) else tuple(hl)
    bad = [h for h in hl if h not in SOURCES]
    if bad or len(set(hl)) != len(hl):
        raise NewsConfigError(f"NEWS headlines: each of {sorted(SOURCES)} at most once, in order of "
                              f"trust, or None; got {d.get('headlines')!r}")
    inst = d.get("instant", False)
    if not isinstance(inst, bool):
        raise NewsConfigError(f"NEWS instant: True or False, got {inst!r}")
    br = d.get("breaking") or ()
    br = (br,) if isinstance(br, str) else tuple(br)
    if [b for b in br if b not in SOURCES] or len(set(br)) != len(br):
        raise NewsConfigError(f"NEWS breaking: each of {sorted(SOURCES)} at most once, in order of "
                              f"trust, or None; got {d.get('breaking')!r}")
    return NewsConfig(on=bool(d.get("on", True)), currencies=cov, timezone=tz, week_ahead=wa,
                      day_ahead=da, remind_before=rem, results=res, warn_alerts=warn,
                      headlines=hl, instant=inst, breaking=br)


# ══ THE CALENDAR THE MONITOR READS ═══════════════════════════════════════════════════════════
class LiveNews:
    """The monitor's own small calendar store (alerts/state/news): Forex Factory's feed every 30
    minutes, FRED twice a day (official US dates beyond this week, when its key is set), the US
    actuals from the agencies, and JBlanked only while a covered release waits for its number.

    Two speeds. `refresh` runs BEFORE the candle checks and does only what this tick's notices
    need: the feed. `background` runs AFTER the candle emails are out and does the slow,
    unhurried fetches (FRED's dates, the agencies' history, the Fed's calendar), so a daily sweep
    of twenty requests never holds up a candle alert."""

    def __init__(self, root):
        from quant.news.calendar import NewsStore
        self.store = NewsStore(root)
        self.fresh = False                 # did THIS tick bring a new full fetch of the feed?
        self._lines: Dict[str, object] = {}    # this tick's headlines per source (or its error)
        self.archive = self.store.root / "headlines.sqlite"   # every line read, for later study
        self._archive_warned = False

    def refresh(self, now) -> Dict[str, str]:
        from quant.news import update
        errors, self.fresh = {}, False
        self._lines = {}                   # a new tick: each source may be asked once again
        before = self.store.meta().get("feed_ok")
        try:
            update.update_feed(self.store, now)
            self.fresh = self.store.meta().get("feed_ok") != before
        except Exception as e:
            errors["news feed"] = redact(f"{type(e).__name__}: {e}")
            self.store.set_meta(feed_error=f"{now.isoformat()} {e}")
        return errors

    def background(self, now) -> Dict[str, str]:
        from quant.news import update
        errors = {}
        # the Fed's calendar FIRST: the agencies' history reads its decision days for the rate
        try:
            update.update_fomc(self.store, now)
        except Exception as e:
            errors["news fomc"] = redact(f"{type(e).__name__}: {e}")
        key = update.keys()["fred"]
        if key:
            try:
                update.update_fred(self.store, key, now)
            except Exception as e:
                errors["news fred"] = redact(f"{type(e).__name__}: {e}")
            try:                           # the recent history: baselines + charts' actuals
                update.update_official(self.store, key, now,
                                       start=(now - pd.Timedelta(days=150)).date())
            except Exception as e:
                errors["news official"] = redact(f"{type(e).__name__}: {e}")
        return errors

    def poll_results(self, now, pending=None) -> Dict[str, str]:
        """The agencies for US numbers (FRED key), JBlanked for the rest when it has credits."""
        from quant.news import update
        k, errors = update.keys(), {}
        if k["fred"] and pending is not None and len(pending):
            try:
                update.poll_official(self.store, k["fred"], pending, now)
            except Exception as e:
                errors["news official"] = redact(f"{type(e).__name__}: {e}")
        if k["jblanked"] and update.jb_has_credits(self.store, now):
            try:
                update.update_jb(self.store, k["jblanked"], now, kind="week")
            except Exception as e:
                errors["news results"] = redact(f"{type(e).__name__}: {e}")
        return errors

    def headlines(self, name: str) -> List[dict]:
        """One headline source's latest lines (quant.news.headlines), through the monitor's
        shared connections -- one certificate store per process, which matters on the server.
        Raises when the source fails, so the next one is asked."""
        from quant.news import headlines as H
        from .feeds import session

        answer = {"status": 0}                  # 0 = no HTTP answer at all (a timeout)

        def get(url, headers, timeout):
            r = session().get(url, headers=headers, timeout=timeout)
            answer["status"] = r.status_code
            return r.status_code, r.content
        # ONE download per source per tick, shared by the results and the breaking news:
        # FinancialJuice answers a second request within a minute with HTTP 429. Each download
        # is kept in the archive (alerts/state/news/headlines.sqlite) for later study
        if name not in self._lines:
            import time
            t0, seen = time.monotonic(), pd.Timestamp.now(tz="UTC")
            try:
                self._lines[name] = H.fetch(name, get)
            except Exception as e:
                self._lines[name] = e
            got = self._lines[name]
            failed = isinstance(got, Exception)
            kept = H.archive(self.archive, name, None if failed else got, seen,
                             int((time.monotonic() - t0) * 1000), answer["status"],
                             redact(f"{type(got).__name__}: {got}") if failed else None)
            if kept is None and not self._archive_warned:
                import logging
                self._archive_warned = True        # once a run: the alerts go on regardless
                logging.getLogger("quant.alerts").warning(
                    "headline archive %s could not be written; alerts unaffected", self.archive)
        got = self._lines[name]
        if isinstance(got, Exception):
            raise got
        return got

    def results_available(self) -> bool:
        """US numbers need the FRED key (it holds the history the live numbers are checked
        against); JBlanked counts only when its key has credits."""
        from quant.news import update
        k = update.keys()
        return bool(k["fred"]) or (bool(k["jblanked"]) and update.jb_has_credits(self.store))

    def can_resolve(self, currency: str, title: str) -> bool:
        """Is there a source that will ever give this release's number?"""
        from quant.news import update
        from quant.news.official import specs_by_title
        if currency == "USD" and title in specs_by_title() and update.keys()["fred"]:
            return True
        return bool(update.keys()["jblanked"]) and update.jb_has_credits(self.store)

    def events(self, start, end) -> pd.DataFrame:
        from quant.news.calendar import load_events
        return load_events(start, end, store=self.store)

    def covered_until(self) -> Optional[pd.Timestamp]:
        weeks = self.store.meta().get("feed_weeks") or {}
        if not weeks:
            return None
        return (pd.Timestamp(max(weeks)).tz_localize("America/New_York")
                + pd.Timedelta(days=7)).tz_convert("UTC")

    def feed_age(self, now) -> Optional[pd.Timedelta]:
        t = self.store.meta().get("feed_ok")
        return None if t is None else now - pd.Timestamp(t)


class FakeNews:
    """A calendar for tests: fixed events, a fixed coverage end, and a record of every poll."""

    def __init__(self, events: pd.DataFrame, covered_until=None, results=True, fresh=True,
                 age=pd.Timedelta(0), unresolvable=(), lines=None):
        self.ev = events
        self.lines = dict(lines or {})     # headline source -> its lines, or an Exception to raise
        self.asked = []                    # every headline source asked, in order
        self.unresolvable = set(unresolvable)
        self.pending = []
        self.until = None if covered_until is None else pd.Timestamp(covered_until, tz="UTC")
        self.results = results
        self.fresh = fresh
        self.age = age
        self.polls = []

    def refresh(self, now):
        return {}

    def background(self, now):
        self.background_at = now
        return {}

    def poll_results(self, now, pending=None):
        self.polls.append(now)
        self.pending.append(pending)
        return {}

    def headlines(self, name):
        self.asked.append(name)
        v = self.lines.get(name, [])
        if isinstance(v, Exception):
            raise v
        return v

    def results_available(self):
        return self.results

    def can_resolve(self, currency, title):
        return title not in self.unresolvable

    def events(self, start, end):
        e = self.ev
        return e[(e["t"] >= start) & (e["t"] < end)].reset_index(drop=True)

    def covered_until(self):
        return self.until

    def feed_age(self, now):
        return self.age


# ══ WORDING ══════════════════════════════════════════════════════════════════════════════════
def _local(t, tz) -> str:
    return pd.Timestamp(t).tz_convert(tz).strftime("%a %d %b %H:%M")


def _ny(t) -> str:
    return pd.Timestamp(t).tz_convert("America/New_York").strftime("%H:%M")


def _tzname(tz, t) -> str:
    return pd.Timestamp(t).tz_convert(tz).strftime("%Z")


def _nums(e) -> str:
    parts = [f"{k} {v}" for k, v in (("actual", e.get("actual")), ("forecast", e.get("forecast")),
                                     ("previous", e.get("previous"))) if v not in (None, "")]
    return ", ".join(parts)


def _text(v):
    """A printed value, or None -- a missing number arrives as NaN and must not print as 'nan'."""
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) == "" else str(v)


def _item(e) -> dict:
    """What a notice keeps about one release (JSON-safe)."""
    z = e.get("surprise_z")
    return dict(t=pd.Timestamp(e["t"]).isoformat(), currency=e["currency"], title=e["title"],
                level=e["level"], time_q=e.get("time_q", "exact"),
                actual=_text(e.get("actual")), forecast=_text(e.get("forecast")),
                previous=_text(e.get("previous")), sentiment=_text(e.get("sentiment")) or "",
                z=None if z is None or pd.isna(z) else round(float(z), 2))


def _item_line(i, tz, *, with_day=True) -> str:
    when = _local(i["t"], tz) if with_day else pd.Timestamp(i["t"]).tz_convert(tz).strftime("%H:%M")
    if i.get("time_q") in ("date", "allday"):
        when = pd.Timestamp(i["t"]).tz_convert(tz).strftime("%a %d %b") + " (time TBA)"
    nums = _nums(i)
    return (f"{when}  {i['currency']} {LEVEL_MARK.get(i['level'], '')} {i['title']}"
            + (f"  ({nums})" if nums else ""))


def span(minutes: int) -> str:
    """Minutes as a person says them: 15 min · 1 h · 2 h 30 min · 12 h · 1 day · 1 day 2 h."""
    m = max(0, int(minutes))
    if m < 60:
        return f"{m} min"
    if m < 1440:
        h, r = divmod(m, 60)
        return f"{h} h" + (f" {r} min" if r else "")
    d, rest = divmod(m, 1440)
    h = rest // 60
    return f"{d} day{'s' * (d > 1)}" + (f" {h} h" if h else "")


def _render(kind: str, items: List[dict], tz: str, now, **kw) -> dict:
    """Subject, text lines and an HTML block for one notice."""
    esc = _html.escape
    first = items[0] if items else None
    cur = first["currency"] if first else ""
    zone = _tzname(tz, now)
    if kind == "remind":
        t = pd.Timestamp(first["t"])
        mins = max(0, int(round((t - now).total_seconds() / 60)))
        names = ", ".join(i["title"] for i in items)
        head = (f"{cur} {names} in {span(mins)} -- {_local(t, tz)} {zone} "
                f"({_ny(t)} New York)")
        subject = f"{cur} {items[0]['title']}{' +' + str(len(items) - 1) if len(items) > 1 else ''}" \
                  f" in {span(mins)}"
        lines = [head] + [f"    {i['title']}: {_nums(i) or 'no forecast'}" for i in items]
    elif kind == "result":
        t = pd.Timestamp(first["t"])
        lines = [f"{cur} released {_local(t, tz)} {zone}:"]
        for i in items:
            mood = (f" -> {i['sentiment']} for {cur}" if i["sentiment"] else "")
            z = f", {i['z']:+.1f} sigma" if i.get("z") is not None and i["sentiment"] else ""
            lines.append(f"    {i['title']}: {_nums(i)}{mood}{z}")
        if kw.get("note"):
            lines.append(f"    ({kw['note']})")
        moods = {i["sentiment"] for i in items if i["sentiment"]}
        subject = (f"{cur} {first['title']}: {first.get('actual') or '?'} vs "
                   f"{first.get('forecast') or '?'}"
                   + (f" ({moods.pop()})" if len(moods) == 1 else ""))
    elif kind == "breaking":
        c, mv = kw["line"], kw.get("move")
        t = pd.Timestamp(c["t"])
        lines = [f"BREAKING {_local(t, tz)} {zone} -- {c['source']}:", f"    {c['text']}"]
        if mv:
            lines.append(f"    {mv['market']} {mv['move_pct']:+.2f}% in 5 min from "
                         f"{pd.Timestamp(mv['start']).tz_convert(tz):%H:%M}")
        lines.append("    (as the source typed it -- news that is not on the calendar)")
        subject = f"BREAKING: {c['text']}" + (f" · {mv['market']} {mv['move_pct']:+.2f}%"
                                              if mv else "")
        subject = subject if len(subject) <= 150 else subject[:147] + "..."
    elif kind == "headline":
        t = pd.Timestamp(first["t"])
        lines = [f"{cur} released {_local(t, tz)} {zone} -- the first headlines:"]
        lines += [f"    {x['after_s']:+.0f} s  {x['source']}: {x['text']}" for x in kw["lines"]]
        fc = [f"{i['title']} {i['forecast']}" for i in items if i.get("forecast")]
        if fc:
            lines.append("    calendar forecast: " + ", ".join(fc))
        lines.append("    (as the source typed it -- not the official number)")
        # the subject names the group's FIRST title (its most important), so it quotes the line
        # about THAT one: on 29 Oct the first line was PCE's and GDP's came 2 s later
        from quant.news.headlines import release_words
        words, _ = release_words(cur, first["title"])
        top = next((x for x in kw["lines"] if words and re.search(words, x["text"], re.I)),
                   kw["lines"][0])
        subject = f"{cur} {first['title']}: {top['text']}"
        subject = subject if len(subject) <= 140 else subject[:137] + "..."
    elif kind in ("day", "week"):
        label = kw["label"]
        subject = label + f" -- {len(items)} release{'s' * (len(items) != 1)}"
        lines = [label + f" ({zone}):"]
        last_day = None
        for i in items:
            d = pd.Timestamp(i["t"]).tz_convert(tz).strftime("%a %d %b")
            if kind == "week" and d != last_day:
                lines.append(f"  {d}")
                last_day = d
            lines.append("    " + _item_line(i, tz, with_day=False))
        if kw.get("warning"):
            lines.append("  ! " + kw["warning"])
    elif kind == "moved":
        old = kw["old"]
        subject = f"{cur} {first['title']} moved to {_local(first['t'], tz)}"
        lines = [f"{cur} {first['title']} MOVED: {_local(old, tz)} -> {_local(first['t'], tz)} "
                 f"{zone}"]
    elif kind == "removed":
        subject = f"{cur} {first['title']} is no longer on the calendar"
        lines = [f"{cur} {first['title']} ({_local(first['t'], tz)} {zone}) is no longer on the "
                 f"calendar -- cancelled, or renamed"]
    else:                                  # feed
        subject = "the news calendar is not updating"
        lines = [kw["message"]]
    body = "<br>".join(esc(x).replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;") for x in lines)
    colour = (LEVEL_COLOUR["critical"] if kind == "breaking" else
              LEVEL_COLOUR.get(first["level"] if first else "none", "#111827"))
    html = f'<div style="border-left:3px solid {colour};padding:2px 8px;margin:4px 0;">{body}</div>'
    out = dict(kind=kind, subject=subject, text=lines, html=html, items=items)
    if kind == "headline":
        out["lines"] = kw["lines"]                 # what the official number is checked against
    return out


# ══ THE STEP ═════════════════════════════════════════════════════════════════════════════════
def _add(state, *, tf, market, bar_time, what, priority, now, notice, status="new",
         side="") -> bool:
    return state.add_event(alert=ALERT, category=CATEGORY, priority=priority, market=market,
                           tf=tf, bar_time=bar_time, close_time=now, what=what, side=side,
                           price=0.0, detail={"news": notice}, created=now, late=False,
                           status=status)


def reminder_offsets(nc, level: str) -> Tuple[int, ...]:
    """Minutes-before for a release of `level`: the nearest configured level AT OR BELOW it, so
    {"high": (15,)} also reminds of critical releases, and never of medium ones."""
    for lv in LEVELS[RANK[level]::-1]:
        if lv in nc.remind_before:
            return nc.remind_before[lv]
    return ()


def _groups(ev: pd.DataFrame):
    """Releases at the same minute in the same currency, most important first inside."""
    if ev.empty:
        return []
    ev = ev.assign(_rk=ev["level"].map(RANK)).sort_values(["t", "currency", "_rk"],
                                                          ascending=[True, True, False])
    return [(t, cur, g) for (t, cur), g in ev.groupby(["t", "currency"], sort=True)]


def _headline_results(nc, state, news, now, report, timed, pri):
    """Queue the headline result of every covered release inside its window that has none yet
    (quant.news.headlines). The sources are asked by tick() AFTER the candle emails of this tick
    are out, in order of trust: each next one only for the releases the last could not answer --
    it failed, or has no line about them yet. A release's first lines go at once, and once."""
    from quant.news import headlines as H
    frame = timed[(timed["level"].map(RANK) >= RANK[nc.results]) & (timed["t"] <= now)]
    due = []
    for t, cur, g in _groups(frame):
        w = H.window(cur, g["title"])
        if w is None or now > t + w[2] + pd.Timedelta(minutes=1):
            continue                       # no rule names it, or its window is over
        if (state.find_event(ALERT, cur, "headline", t, "result")
                or state.find_event(ALERT, cur, "result", t, "result")):
            continue                       # its headline went -- or the official number did first
        due.append((t, cur, g, w))
    if not due:
        return

    def ask_then_send():
        # the official results run between the queueing and this: one that went this very tick
        # leaves nothing for a headline to add
        left = [d for d in due if not state.find_event(ALERT, d[1], "result", d[0], "result")]
        for name in nc.headlines:
            if not left:
                break
            try:
                got = news.headlines(name)
            except Exception as e:                  # the next source is asked instead
                report.errors[f"news headlines {name}"] = f"{type(e).__name__}: {e}"
                continue
            label, rest = H.SOURCES[name].label, []
            for t, cur, g, (words, since, until) in left:
                m = H.matching(got, t, words, since, until)
                if not m:
                    rest.append((t, cur, g, (words, since, until)))
                    continue
                quoted = [dict(source=label, text=H.clean(x["title"])[:240], id=x["id"],
                               after_s=(x["at"] - t).total_seconds()) for x in m[:HEADLINE_LINES]]
                items = [_item(e) for e in g.to_dict("records")]
                report.events += _add(state, tf="headline", market=cur, bar_time=t,
                                      what="result", priority=pri, now=now,
                                      notice=_render("headline", items, nc.timezone, now,
                                                     lines=quoted))
            left = rest
    report.after.append(ask_then_send)


def _recent_moves(cfg, state, now) -> List[dict]:
    """The sharp moves of the alert markets whose 5 minutes are complete, started within the last
    half hour, from the monitor's own 1m candles (a market with no 1m candles or no measured floor
    confirms nothing). The usual move is the day before's, so a day and a half is read."""
    from quant.data.base import TIME_COL
    from quant.news import headlines as H
    out = []
    for name, m in cfg.markets.items():
        floor = H.move_floor(m.symbol)
        b = state.bars(name, "1m") if floor else None
        if b is None or not len(b):
            continue
        t = pd.to_datetime(b[TIME_COL], utc=True)
        b = b[(t >= now - pd.Timedelta(hours=36)) & (t + pd.Timedelta(minutes=1) <= now)]
        mv = H.sharp_moves(b.rename(columns={TIME_COL: "open_time"}), floor)
        for r in mv.itertuples():
            s = pd.Timestamp(r.start)
            if s >= now - pd.Timedelta(minutes=30) and s + H.MOVE_KNOWN <= now:
                out.append(dict(market=name, start=s, move_pct=float(r.move_pct)))
    return sorted(out, key=lambda x: x["start"])


def _breaking_news(cfg, state, news, now, report, pri):
    """Queue the breaking-news check (quant/news/headlines.py tiers) for after this tick's candle
    emails. The first source that answers is read -- the next only when it fails. A TOP line goes
    at once; a BREAKING line waits, at most 15 min + 5, for a sharp move from 10 min before it to
    15 min after; one per story an hour, at most BREAKING_PER_DAY a day (the rest are recorded,
    status 'skipped'). A line already older than BREAKING_LOOKBACK when first seen is not sent."""
    from quant.news import headlines as H
    nc = cfg.news

    def check():
        got = label = None
        for name in nc.breaking:
            try:
                got, label = news.headlines(name), H.SOURCES[name].label
                break
            except Exception as e:                  # the next source is asked instead
                report.errors[f"news breaking {name}"] = f"{type(e).__name__}: {e}"
        if got is None:
            return
        seen = json.loads(state.get("news_breaking_seen") or "{}")       # line id -> printed
        waiting = json.loads(state.get("news_breaking_waiting") or "[]")
        sent = json.loads(state.get("news_breaking_sent") or "[]")       # [{at, text}]
        for x in got:
            if not x.get("published") or x["id"] in seen:
                continue
            t = pd.Timestamp(x["published"])
            seen[x["id"]] = t.isoformat()
            text = H.clean(x["title"])
            tier = H.breaking_tier(text)
            if tier and t >= now - BREAKING_LOOKBACK:
                waiting.append(dict(t=t.isoformat(), text=text[:240], tier=tier[0], rule=tier[1],
                                    source=label))
        moves = _recent_moves(cfg, state, now) if any(w["tier"] == 2 for w in waiting) else []
        still = []
        for w in sorted(waiting, key=lambda w: w["t"]):
            t, move = pd.Timestamp(w["t"]), None
            if w["tier"] == 2:
                ok = [m for m in moves if t - H.MOVE_BEFORE <= m["start"] <= t + H.MOVE_AFTER]
                if not ok:
                    if now <= t + H.MOVE_AFTER + H.MOVE_KNOWN:
                        still.append(w)             # its move may still come
                    continue
                move = ok[0]
            if any(H.same_story(w["text"], s["text"]) and now - pd.Timestamp(s["at"]) <= H.STORY_GAP
                   for s in sent):
                continue                            # the same story went within the hour
            today = sum(pd.Timestamp(s["at"]).floor("D") == now.floor("D") for s in sent)
            status = "new" if today < H.BREAKING_PER_DAY else "skipped"
            report.events += _add(state, tf="breaking", market=w["source"], bar_time=t,
                                  what="breaking", priority=pri, now=now, status=status,
                                  notice=_render("breaking", [], cfg.news.timezone, now, line=w,
                                                 move=move))
            if status == "new":
                sent.append(dict(at=now.isoformat(), text=w["text"]))
        keep = now - BREAKING_SEEN
        state.put("news_breaking_seen", json.dumps({k: v for k, v in seen.items()
                                                    if pd.Timestamp(v) >= keep}))
        state.put("news_breaking_waiting", json.dumps(still))
        state.put("news_breaking_sent", json.dumps([s for s in sent if pd.Timestamp(s["at"])
                                                    >= now - pd.Timedelta(days=1)]))
    report.after.append(check)


def _against_headline(state, cur, t, items) -> Tuple[str, Optional[str]]:
    """(status, note) for the official result of a release whose headline result already went:
    skipped when the headline printed every number, else sent with a word on why."""
    from quant.news.headlines import agrees
    hl = state.find_event(ALERT, cur, "headline", t, "result")
    if hl is None:
        return "new", None
    lines = notice_of(hl).get("lines") or []
    said = [agrees(i["actual"], cur, i["title"], lines) for i in items]
    if said and all(v is True for v in said):
        return "skipped", None             # nothing new: the headline already printed it
    return "new", ("the official number -- it differs from the headline"
                   if False in said else "the official number")


def news_step(cfg, state, news, now, report) -> Optional[Callable]:
    """One tick of news. Returns `note(close_time) -> str` for candle alerts (warn_alerts), or
    None when that is off."""
    from quant.news.calendar import select
    nc = cfg.news
    report.errors.update(news.refresh(now))
    tz = nc.timezone
    horizon = news.events(now - RESULT_WINDOW - pd.Timedelta(hours=1), now + pd.Timedelta(days=8))
    ev = select(horizon, nc.currencies) if len(horizon) else horizon
    timed = ev[ev["time_q"].isin(["exact", "official", "typical"])] if len(ev) else ev
    sent = 0
    pri = "instant" if nc.instant else "urgent"    # instant: never waits for the email clock

    # ── reminders ─────────────────────────────────────────────────────────────────────────
    if nc.remind_before and len(timed):
        # on a clock (DELIVERY every_minutes) a reminder is due from the boundary at or BEFORE its
        # time, so it rides in that boundary's email -- early by up to a quarter hour, never late
        every = getattr(cfg.delivery, "every_minutes", 0)

        def due_at(t, k):
            at = t - pd.Timedelta(minutes=k)
            return at.floor(f"{every}min") if every else at

        for t, cur, g in _groups(timed[timed["t"] > now]):
            offs = reminder_offsets(nc, max(g["level"], key=RANK.get))
            if not offs:
                continue
            due = [k for k in offs if due_at(t, k) <= now]
            if not due:
                continue
            keep = min(due)                        # the latest reminder still ahead of the news
            items = [_item(e) for e in g.to_dict("records")]
            for k in due:
                new = _add(state, tf=f"T-{k}", market=cur, bar_time=t, what="reminder",
                           priority=pri, now=now, notice=_render("remind", items, tz, now),
                           status="new" if k == keep else "skipped")
                sent += new and k == keep

    # ── results from headlines ────────────────────────────────────────────────────────────
    if nc.results and nc.headlines and len(timed):
        _headline_results(nc, state, news, now, report, timed, pri)

    # ── breaking news, not on the calendar ─────────────────────────────────────────────────
    if nc.breaking:
        _breaking_news(cfg, state, news, now, report, pri)

    # ── results ───────────────────────────────────────────────────────────────────────────
    if nc.results and len(timed) and news.results_available():
        def results(frame) -> bool:
            """Send every group whose numbers are in; True if some group is still waiting."""
            nonlocal sent
            if not len(frame):
                return False
            frame = frame[(frame["level"].map(RANK) >= RANK[nc.results]) & (frame["t"] <= now)
                          & (frame["t"] > now - RESULT_WINDOW)]
            waiting = False
            for t, cur, g in _groups(frame):
                gk = f"{cur}|{t.isoformat()}"
                if now - t > RESULT_FRESH and gk not in was_waiting:
                    continue                       # the PC was off when it came out: stale
                have = g["actual"].notna()
                # a release no source will ever answer (ISM: no free official feed) is not
                # waited for -- the rest of its group goes as soon as THEY are in
                can = g["title"].map(lambda ti, c=cur: news.can_resolve(c, ti)).to_numpy(bool)
                done = (have | ~can).all()
                if have.any() and (done or now - t >= RESULT_SETTLE):
                    items = [_item(e) for e in g[have].to_dict("records")]
                    worse = sum(i["sentiment"] == "worse" for i in items)
                    better = sum(i["sentiment"] == "better" for i in items)
                    status, note = (_against_headline(state, cur, t, items) if nc.headlines
                                    else ("new", None))
                    new = _add(state, tf="result", market=cur, bar_time=t, what="result",
                               priority=pri, now=now, status=status,
                               side="bull" if better > worse else "bear" if worse > better else "",
                               notice=_render("result", items, tz, now, note=note))
                    sent += new and status == "new"
                if not done:
                    waiting = True
                    pending.append(g[~have & can])
                    still.add(gk)
            return waiting

        was_waiting = set(json.loads(state.get("news_waiting") or "[]"))
        pending: list = []
        still: set = set()
        if results(timed):
            asked = pd.concat(pending, ignore_index=True)

            def poll_then_send():
                """Ask the sources, then send what came in -- run by tick() AFTER this tick's
                candle emails are out: at a release the agencies can take 20-40 s to answer
                (14 Oct 2026: the BLS answered 503 after 12 s), and that is exactly when a
                candle alert must not wait."""
                report.errors.update(news.poll_results(now, asked))
                again = news.events(now - RESULT_WINDOW, now + pd.Timedelta(minutes=1))
                still.clear()
                results(select(again, nc.currencies) if len(again) else again)
                state.put("news_waiting", json.dumps(sorted(still)))

            report.after.append(poll_then_send)
        state.put("news_waiting", json.dumps(sorted(still)))

    # ── day ahead ─────────────────────────────────────────────────────────────────────────
    local = now.tz_convert(tz)
    until = news.covered_until()
    if nc.day_ahead:
        hh, mm = map(int, nc.day_ahead.split(":"))
        slot = local.normalize() + pd.Timedelta(hours=hh, minutes=mm)
        if slot > local:                   # the window runs past midnight: 20:00 + 6 h = 02:00,
            slot -= pd.Timedelta(days=1)   # so at 01:00 the live slot is YESTERDAY's
        if local < slot + DAY_WINDOW:
            a = slot.normalize() + pd.Timedelta(days=1)
            b = a + pd.Timedelta(days=1)
            known = until is not None and until >= b.tz_convert("UTC")
            if known or local >= slot + DAY_WAIT:
                items = ev[(ev["t"] >= a.tz_convert("UTC")) & (ev["t"] < b.tz_convert("UTC"))] \
                    if len(ev) else ev
                if len(items):
                    notice = _render("day", [_item(e) for e in items.to_dict("records")], tz,
                                     now, label=f"Tomorrow, {a:%a %d %b}",
                                     warning=None if known else
                                     "the calendar for that day is not complete yet")
                    sent += _add(state, tf="day", market="calendar",
                                 bar_time=slot.tz_convert("UTC"), what="day ahead",
                                 priority="normal", now=now, notice=notice)

    # ── week ahead ────────────────────────────────────────────────────────────────────────
    if nc.week_ahead:
        dow, hhmm = nc.week_ahead
        hh, mm = map(int, hhmm.split(":"))
        back = (local.dayofweek - dow) % 7
        slot = local.normalize() - pd.Timedelta(days=back) + pd.Timedelta(hours=hh, minutes=mm)
        if slot > local:
            slot -= pd.Timedelta(days=7)
        if local < slot + WEEK_WINDOW:
            end = slot + pd.Timedelta(days=7)
            known = until is not None and until >= (slot + pd.Timedelta(days=6)).tz_convert("UTC")
            if known or local >= slot + WEEK_WAIT:
                wk = ev[(ev["t"] >= slot.tz_convert("UTC")) & (ev["t"] < end.tz_convert("UTC"))
                        & (ev["level"].map(RANK) >= RANK["high"])] if len(ev) else ev
                if len(wk):
                    notice = _render("week", [_item(e) for e in wk.to_dict("records")], tz, now,
                                     label=f"The week ahead, from {slot:%a %d %b}",
                                     warning=None if known else
                                     "Forex Factory has not published the whole week yet")
                    sent += _add(state, tf="week", market="calendar",
                                 bar_time=slot.tz_convert("UTC"), what="week ahead",
                                 priority="normal", now=now, notice=notice)

    # ── moved / removed ───────────────────────────────────────────────────────────────────
    if news.fresh and len(timed):
        ahead = timed[timed["t"] > now]
        watch = ahead[(ahead["t"] < now + pd.Timedelta(days=7))
                      & (ahead["level"].map(RANK) >= RANK["high"])]
        cur_map = {r["key"]: r for r in watch.to_dict("records")}
        # every future release, any level: a release pushed past the week or downgraded is
        # still on the calendar, and must not be reported as gone
        all_map = {r["key"]: r for r in ahead.to_dict("records")}
        old = json.loads(state.get("news_upcoming") or "{}")
        for k, o in old.items():
            ot = pd.Timestamp(o["t"])
            if ot <= now:
                continue
            if k in all_map:
                r = all_map[k]
                if pd.Timestamp(r["t"]) != ot:
                    sent += _add(state, tf="moved", market=r["currency"], bar_time=r["t"], what=k,
                                 priority=pri if r["level"] == "critical" else "normal",
                                 now=now, notice=_render("moved", [_item(r)], tz, now, old=ot))
                continue
            twin = [r for kk, r in all_map.items() if kk not in old
                    and r["currency"] == o["currency"] and r["title"] == o["title"]
                    and abs(pd.Timestamp(r["t"]) - ot) <= pd.Timedelta(days=3)]
            if twin:
                r = twin[0]
                sent += _add(state, tf="moved", market=r["currency"], bar_time=r["t"], what=k,
                             priority=pri if r["level"] == "critical" else "normal",
                             now=now, notice=_render("moved", [_item(r)], tz, now, old=ot))
            elif ot < now + pd.Timedelta(hours=72):
                gone = dict(o, t=ot)
                sent += _add(state, tf="removed", market=o["currency"], bar_time=ot, what=k,
                             priority="normal", now=now,
                             notice=_render("removed", [_item(gone)], tz, now))
        snap = {k: dict(t=pd.Timestamp(r["t"]).isoformat(), currency=r["currency"],
                        title=r["title"], level=r["level"]) for k, r in cur_map.items()}
        state.put("news_upcoming", json.dumps(snap))

    # ── a calendar that stopped updating ──────────────────────────────────────────────────
    age = news.feed_age(now)
    if age is not None and age > STALE:
        msg = (f"The news calendar has not updated for {age.total_seconds() / 3600:.0f} hours "
               f"(last error: {state.get('news_error') or report.errors.get('news feed') or '?'}). "
               f"Reminders still go out from what it last had.")
        sent += _add(state, tf="feed", market="calendar", bar_time=now.floor("D"), what="stale",
                     priority="normal", now=now, notice=_render("feed", [], tz, now, message=msg))
    if report.errors.get("news feed"):
        state.put("news_error", report.errors["news feed"])
    report.events += sent

    if not nc.warn_alerts or not len(timed):
        return None
    hi = timed[timed["level"].map(RANK) >= RANK["high"]]
    if not len(hi):
        return None
    w = pd.Timedelta(minutes=nc.warn_alerts)

    def note(close_time) -> str:
        c = pd.Timestamp(close_time)
        near = hi[(hi["t"] >= c - w) & (hi["t"] <= c + w)]
        if not len(near):
            return ""
        e = near.iloc[(near["t"] - c).abs().argmin()]
        m = int(round((e["t"] - c).total_seconds() / 60))
        rel = f"in {span(m)}" if m > 0 else ("now" if m == 0 else f"{span(-m)} ago")
        return f"news: {e['currency']} {e['title']} {rel}"
    return note


# ══ THE EMAIL SECTION ════════════════════════════════════════════════════════════════════════
def is_news(e) -> bool:
    return e.get("alert") == ALERT


def notice_of(e) -> dict:
    return json.loads(e["detail"] or "{}").get("news", {})


def news_text(events) -> List[str]:
    out = []
    for e in events:
        n = notice_of(e)
        mark = "!" if e["priority"] == "urgent" else " "
        lines = n.get("text") or [e["what"]]
        out.append(f"{mark} {lines[0]}")
        out += [f"  {x}" for x in lines[1:]]
    return out


def news_html(events) -> str:
    return "".join(notice_of(e).get("html", "") for e in events)


def news_subject(events) -> str:
    n = notice_of(events[0]).get("subject", "news")
    return n + (f" +{len(events) - 1} more" if len(events) > 1 else "")


def upcoming_table(news, cfg, now, days: float = 7) -> List[str]:
    """For `python -m quant.alerts news`: the covered calendar and when each reminder will go."""
    from quant.news.calendar import select
    nc = cfg.news
    ev = news.events(now, now + pd.Timedelta(days=days))
    ev = select(ev, nc.currencies) if len(ev) else ev
    lines = []
    for t, cur, g in _groups(ev):
        offs = reminder_offsets(nc, max(g["level"], key=RANK.get))
        rem = ", ".join(f"{span(k)} before" for k in offs) or "no reminder"
        for i, e in enumerate(g.to_dict("records")):
            lines.append(_item_line(_item(e), nc.timezone) + (f"   [{rem}]" if i == 0 else ""))
    return lines
