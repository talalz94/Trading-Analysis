"""
HEADLINE PROBE -- a five-trading-day test of free headline sources, BEFORE any of them is wired
into the alerts. It records every headline each source publishes -- with the source's own time
and the moment we first saw it -- and every download, so experiments/headline_report.py can say
per source: how fast, how complete around sharp moves, how noisy, how reliable.

Why a test first: on 21 Oct 2026 Gold fell 34 points from 18:44 UTC on a US "ordered departure"
headline. FinancialJuice's free RSS had it at 18:47:50, Forex Factory (through Google News) at
18:48:38, investingLive at 19:13 -- one story, three sources, 25 minutes apart. A week of that,
for every story, decides which sources the headlines layer uses.

Runs on the alerts server as its own service (scripts/install_headline_probe.sh): its own
database (experiments/results/headline_probe/, git-ignored), never the alerts state; about six
small downloads a minute, conditional (ETag / Last-Modified) where the site supports it.

    python experiments/headline_probe.py --once      one round of every source, printed; stores nothing
    python experiments/headline_probe.py --days 7    the test: polls until 7 days after its FIRST start
    python experiments/headline_probe.py --status    what has been recorded so far
"""
from __future__ import annotations

import argparse
import html
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import quote

import pandas as pd

DB = Path(__file__).resolve().parent / "results" / "headline_probe" / "probe.sqlite"
UA = "Mozilla/5.0 (compatible; quant-headline-probe/1.0; personal research, low rate)"
TIMEOUT = 20.0
MAX_BACKOFF = 900.0                  # a failing source is retried at most every 15 minutes


def _gnews(q: str) -> str:
    return ("https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
            + quote(q + " when:1d"))


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                        # "rss" | "telegram"
    url: str
    every: int                       # seconds between polls
    group: str                       # squawk | aggregator | official


SOURCES = [
    # squawk-style: one-line headlines, minutes after the wire
    Source("fj_rss", "rss", "https://www.financialjuice.com/feed.ashx?xy=rss", 60, "squawk"),
    Source("fj_telegram", "telegram", "https://t.me/s/FinancialJuice", 60, "squawk"),
    Source("walter_telegram", "telegram", "https://t.me/s/WalterBloomberg", 60, "squawk"),
    Source("investinglive", "rss", "https://investinglive.com/feed/", 60, "squawk"),
    # Google News searches: every outlet, Forex Factory's headlines included, without scraping
    Source("gnews_security", "rss", _gnews('(war OR attack OR missile OR strike OR embassy OR '
                                           'evacuation OR "ordered departure" OR ceasefire OR '
                                           'Houthi OR Iran OR Israel)'), 300, "aggregator"),
    Source("gnews_centralbanks", "rss", _gnews('("Federal Reserve" OR FOMC OR "Fed chair" OR '
                                               '"Bank of Japan" OR BoJ OR "yen intervention")'),
           300, "aggregator"),
    Source("gnews_trade", "rss", _gnews('(tariff OR tariffs OR "US sanctions" OR "trade deal" OR '
                                        '"trade war")'), 300, "aggregator"),
    Source("gnews_markets", "rss", _gnews('("gold price" OR bitcoin OR "dollar yen" OR '
                                          '"oil prices")'), 300, "aggregator"),
    # official: slow, but they ARE the source
    Source("fed", "rss", "https://www.federalreserve.gov/feeds/press_all.xml", 300, "official"),
    Source("state_press", "rss", "https://www.state.gov/rss-feed/press-releases/feed/", 300,
           "official"),
    Source("state_travel", "rss", "https://travel.state.gov/_res/rss/TAsTWs.xml", 600, "official"),
    Source("whitehouse", "rss", "https://www.whitehouse.gov/news/feed/", 600, "official"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    source TEXT, item_id TEXT, title TEXT, link TEXT,
    published TEXT,                  -- the source's own time (UTC ISO), NULL when it gives none
    first_seen TEXT,                 -- when this probe first saw it (UTC ISO)
    backlog INTEGER,                 -- 1 = already there on the source's first poll: no timing
    PRIMARY KEY (source, item_id));
CREATE TABLE IF NOT EXISTS polls (
    source TEXT, at TEXT, ok INTEGER, status INTEGER, n INTEGER, new INTEGER, ms INTEGER,
    error TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").isoformat()


def _when(s: Optional[str]) -> Optional[str]:
    """An RSS date -> UTC ISO, or None. A date with no time of day (travel.state.gov prints
    "Wed, 21 Oct 2026") is None: midnight would be a made-up time, and timing is the point."""
    if not s or not re.search(r"\d{1,2}:\d{2}", s):
        return None
    try:
        return iso(pd.Timestamp(parsedate_to_datetime(s.strip())))
    except (TypeError, ValueError):
        try:
            return iso(pd.Timestamp(s.strip()))
        except (TypeError, ValueError):
            return None


def parse_rss(body) -> List[dict]:
    """RSS 2.0 -> [{id, title, link, published}]; the id is the guid, else the link. Takes the
    raw BYTES: the Fed's feed starts with a byte-order mark and declares no charset, so decoding
    it as text first turned the mark into "ï»¿" and the parse failed on line 1."""
    root = ET.fromstring(body if isinstance(body, bytes) else body.encode("utf-8"))
    out = []
    for it in root.iter("item"):
        title = html.unescape(" ".join((it.findtext("title") or "").split()))
        link = (it.findtext("link") or "").strip()
        gid = (it.findtext("guid") or "").strip() or link or title
        date = it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date")
        if title:
            out.append(dict(id=gid, title=title, link=link, published=_when(date)))
    return out


_TG_POST = re.compile(r'data-post="([^"]+)"')
_TG_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
_TG_TIME = re.compile(r'<time[^>]*datetime="([^"]+)"')


def parse_telegram(body) -> List[dict]:
    """A public channel's web page (t.me/s/<channel>) -> [{id, title, link, published}]. Each
    message carries the time TELEGRAM received it, so polling once a minute does not blur it."""
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    out = []
    marks = list(_TG_POST.finditer(text))
    for k, m in enumerate(marks):
        chunk = text[m.start(): marks[k + 1].start() if k + 1 < len(marks) else len(text)]
        body, when = _TG_TEXT.search(chunk), _TG_TIME.search(chunk)
        if not body:
            continue                                     # a photo or a sticker with no words
        words = re.sub(r"<br\s*/?>", " ", body.group(1))
        words = html.unescape(re.sub(r"<[^>]+>", "", words))
        words = " ".join(words.split())
        if words:
            out.append(dict(id=m.group(1), title=words, link=f"https://t.me/{m.group(1)}",
                            published=iso(pd.Timestamp(when.group(1))) if when else None))
    return out


PARSERS = {"rss": parse_rss, "telegram": parse_telegram}


_SESSION = None


def http_get(url: str, headers: Dict[str, str]):
    """-> (status, body BYTES, response headers). The one network call: tests replace it.
    ONE session for the whole run: a bare requests.get builds a new TLS context per call and
    re-reads the certificate bundle -- 24 of 29 CPU-seconds of a round of ten feeds (measured).
    The session reads it once and keeps connections open between polls."""
    global _SESSION
    import requests
    if _SESSION is None:
        _SESSION = requests.Session()
    r = _SESSION.get(url, headers=headers, timeout=TIMEOUT)
    return r.status_code, r.content, r.headers


class Probe:
    def __init__(self, db_path, sources=SOURCES, fetch: Callable = http_get,
                 clock: Callable = lambda: pd.Timestamp.now(tz="UTC")):
        self.db = sqlite3.connect(str(db_path))
        self.db.executescript(SCHEMA)
        self.sources, self.fetch, self.clock = list(sources), fetch, clock
        self.fails: Dict[str, int] = {}
        self.cache: Dict[str, Dict[str, str]] = {}           # ETag / Last-Modified per source
        self.due: Dict[str, pd.Timestamp] = {}

    def meta(self, key, value=None):
        if value is not None:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))
            self.db.commit()
        r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else None

    def poll(self, s: Source) -> dict:
        """One download of one source; stores what is new. Never raises."""
        now = self.clock()
        first = self.db.execute("SELECT 1 FROM polls WHERE source = ? AND ok = 1 LIMIT 1",
                                (s.name,)).fetchone() is None
        t0 = time.monotonic()
        status, n, new, err = 0, 0, 0, None
        try:
            status, body, hdr = self.fetch(s.url, {"User-Agent": UA, **self.cache.get(s.name, {})})
            if status == 304:                                # unchanged since the last look
                pass
            elif status != 200:
                raise RuntimeError(f"HTTP {status}")
            else:
                cond = {k: hdr[h] for k, h in (("If-None-Match", "ETag"),
                                               ("If-Modified-Since", "Last-Modified"))
                        if hdr and hdr.get(h)}
                self.cache[s.name] = cond
                items = PARSERS[s.kind](body)
                n = len(items)
                for it in items:
                    cur = self.db.execute(
                        "INSERT OR IGNORE INTO items VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (s.name, it["id"], it["title"], it["link"], it["published"], iso(now),
                         int(first)))
                    new += cur.rowcount
            ok = True
        except Exception as e:                               # this source, this time, only
            ok, err = False, f"{type(e).__name__}: {e}"[:300]
        ms = int((time.monotonic() - t0) * 1000)
        self.db.execute("INSERT INTO polls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (s.name, iso(now), int(ok), status, n, new, ms, err))
        self.db.commit()
        self.fails[s.name] = 0 if ok else self.fails.get(s.name, 0) + 1
        wait = s.every if ok else min(s.every * 2 ** self.fails[s.name], MAX_BACKOFF)
        self.due[s.name] = now + pd.Timedelta(seconds=wait)
        return dict(source=s.name, ok=ok, status=status, n=n, new=new, ms=ms, error=err)

    def run(self, days: float, sleep: Callable = time.sleep, log: Callable = print):
        """Poll every source on its own schedule until `days` after the FIRST start (a restart
        does not extend the test)."""
        start = pd.Timestamp(self.meta("start") or self.meta("start", iso(self.clock())))
        end = start + pd.Timedelta(days=days)
        now = self.clock()
        for k, s in enumerate(self.sources):                 # staggered: not all at once
            self.due.setdefault(s.name, now + pd.Timedelta(seconds=3 * k))
        log(f"headline probe: {len(self.sources)} sources until {end:%Y-%m-%d %H:%M} UTC")
        last_report = now
        while (now := self.clock()) < end:
            for s in self.sources:
                if self.due[s.name] <= now:
                    r = self.poll(s)
                    if not r["ok"] and self.fails[s.name] in (1, 5, 20):
                        log(f"{s.name}: {r['error']} (failure {self.fails[s.name]})")
            if now - last_report >= pd.Timedelta(hours=6):
                log(status_text(self.db))
                last_report = now
            nxt = min(self.due.values())
            sleep(max(1.0, min(30.0, (nxt - self.clock()).total_seconds())))
        log("headline probe: test window over, stopping")


def status_text(db) -> str:
    rows = db.execute("""
        SELECT p.source, COUNT(*), SUM(p.ok), MAX(CASE WHEN p.ok = 1 THEN p.at END),
               (SELECT COUNT(*) FROM items i WHERE i.source = p.source AND i.backlog = 0),
               (SELECT error FROM polls q WHERE q.source = p.source AND q.ok = 0
                ORDER BY q.at DESC LIMIT 1)
        FROM polls p GROUP BY p.source ORDER BY p.source""").fetchall()
    out = [f"{'source':20} {'polls':>6} {'ok':>6} {'new items':>9}  last ok           last error"]
    for src, n, ok, last, items, err in rows:
        out.append(f"{src:20} {n:6d} {ok:6d} {items:9d}  {(last or '-')[5:16]:16}  "
                   f"{(err or '')[:60]}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--once", action="store_true", help="one round of every source, printed")
    ap.add_argument("--days", type=float, default=7.0, help="test length from the first start")
    ap.add_argument("--status", action="store_true", help="what has been recorded so far")
    ap.add_argument("--db", default=str(DB))
    a = ap.parse_args(argv)
    if a.status:
        if not Path(a.db).is_file():
            print("no probe database yet:", a.db)
            return
        db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        start = db.execute("SELECT value FROM meta WHERE key = 'start'").fetchone()
        print("started", start[0][:16] if start else "?", "UTC")
        print(status_text(db))
        return
    if a.once:
        p = Probe(":memory:")
        for s in p.sources:
            r = p.poll(s)
            newest = p.db.execute("SELECT MAX(published), COUNT(*) FROM items WHERE source = ?",
                                  (s.name,)).fetchone()
            top = p.db.execute("SELECT title FROM items WHERE source = ? ORDER BY published DESC "
                               "LIMIT 1", (s.name,)).fetchone()
            print(f"{s.name:20} {'OK ' if r['ok'] else 'FAIL'} {r['n']:4d} items  "
                  f"newest {(newest[0] or '-')[5:16]:11}  {r['ms']:5d} ms  "
                  f"{r['error'] or (top[0][:70] if top else '')}")
        return
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    Probe(a.db).run(a.days)


if __name__ == "__main__":
    main()
