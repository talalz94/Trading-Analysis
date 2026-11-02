"""
HEADLINES -- the first word on a release, from free squawk feeds: what the alerts quote as a
release's RESULT seconds after it is out, before any official number.

Measured before it was wired in (experiments/headline_probe.py, headline_report.py --critical and
telegram_check.py, 20 Oct - 1 Nov 2026: the 9 critical releases, UK CPI and the Gulf story):

  financialjuice           FinancialJuice's RSS feed. All 9, 0-8 s after the release by its own
                           clock (51 s once, New Zealand), the number in every decision headline.
                           99.1% of once-a-minute polls answered, the longest outage 7 min.
                           Cloudflare does not cache it (cf-cache-status: DYNAMIC).
  financialjuice_telegram  the same desk's Telegram channel, on Telegram's servers: all 11, ~40 s
                           behind the feed -- the backup for when the feed fails.
  walterbloomberg          an independent desk: 6 of 11, 9-20 s on US and European releases, none
                           on New Zealand, Japan or the UK.

Every line is something a person typed, not an official number: the alerts quote it word for
word and name the source, and for US releases the agencies' number follows as the check.

A line belongs to a release when it is printed inside the release's WINDOW (30 s before its time
to 10 min after; 2 h for the BOJ, which announces when its meeting ends) and uses the release's
WORDS (RELEASE_WORDS, per currency and title). A release no rule names gets no headline: a guess
would only add noise. The first matching line is not always the number asked for -- on GDP day
FinancialJuice led with the PCE line -- so lines are ranked (a printed "Actual", then any line
with a number) and the alerts quote the first few.

The readers take BYTES (the Fed's feed starts with a byte-order mark and declares no charset) and
are shared with the test that measured them (experiments/headline_probe.py imports them).
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .values import parse_value

UA = "Mozilla/5.0 (compatible; quant-alerts/1.0; personal use, low rate)"
TIMEOUT = 8.0                       # seconds per download: a release is when it must not hang


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                       # "rss" | "telegram"
    url: str
    label: str                      # how an email names it


SOURCES: Dict[str, Source] = {s.name: s for s in (
    Source("financialjuice", "rss", "https://www.financialjuice.com/feed.ashx?xy=rss",
           "FinancialJuice"),
    Source("financialjuice_telegram", "telegram", "https://t.me/s/FinancialJuice",
           "FinancialJuice (Telegram)"),
    Source("walterbloomberg", "telegram", "https://t.me/s/WalterBloomberg", "Walter Bloomberg"),
)}
DEFAULT_ORDER = ("financialjuice", "financialjuice_telegram", "walterbloomberg")

# How long around its scheduled time a line still counts for a release: from 30 s BEFORE (a
# source's clock runs a little ahead) to 10 minutes after, unless its rule gives more minutes.
WINDOW = (pd.Timedelta(seconds=-30), pd.Timedelta(minutes=10))

RELEASE_WORDS = [  # (currency, release title) -> the words a headline about it uses [, minutes]
    ("USD", r"federal funds|fomc", r"\bfed\b|fomc|federal reserve|powell|warsh|fed funds|"
                                   r"rates? (?:decision|unchanged)"),
    ("USD", r"gdp", r"\bgdp\b"),
    ("USD", r"pce", r"\bpce\b"),
    ("USD", r"cpi", r"\bcpi\b"),
    ("USD", r"claims", r"jobless|claims"),           # before payrolls: "Unemployment Claims"
    ("USD", r"adp", r"\badp\b"),                     # before payrolls: "ADP Non-Farm ..."
    # the jobs report's three numbers each get THEIR words, so the official payrolls figure is
    # checked against the payrolls line and not the unemployment rate typed a second earlier
    ("USD", r"non-farm", r"payroll|non-?farm|\bnfp\b"),
    ("USD", r"unemployment", r"unemployment"),
    ("USD", r"hourly", r"hourly earnings|average earnings"),
    ("USD", r"ppi", r"\bppi\b"),
    ("USD", r"retail sales", r"retail sales"),
    ("USD", r"\bism\b", r"\bism\b"),
    ("USD", r"pmi", r"\bpmi\b"),
    ("USD", r"jolts", r"jolts|job openings"),
    ("USD", r"confidence|sentiment", r"confidence|sentiment"),
    ("USD", r"durable", r"durable"),
    ("EUR", r"pmi", r"(?:euro ?zone|euro area|\bez\b|german|france|french).*\bpmi\b"),
    ("GBP", r"pmi", r"\buk\b.*\bpmi\b"),
    ("EUR", r"refinancing|monetary policy|ecb", r"\becb\b|lagarde|deposit (?:facility )?rate|"
                                                r"refinancing"),
    ("EUR", r"cpi", r"(?:euro ?zone|euro area|\bez\b).*(?:cpi|hicp|inflation)|\bhicp\b"),
    ("GBP", r"bank rate|monetary policy", r"\bboe\b|bank of england|bailey"),
    ("GBP", r"cpi", r"\buk\b.*(?:cpi|inflation)"),
    ("GBP", r"budget", r"budget|reeves"),
    ("JPY", r"boj|policy rate|monetary policy", r"\bboj\b|bank of japan|ueda", 120),  # no set time
    ("CAD", r"overnight rate|boc", r"\bboc\b|bank of canada|macklem"),
    ("CAD", r"cpi|employment", r"canad"),
    ("AUD", r"cash rate|rba", r"\brba\b|reserve bank of australia|bullock"),
    ("AUD", r"cpi|employment", r"australia|aussie|trimmed mean"),
    ("NZD", r"cash rate|rbnz", r"\brbnz\b|reserve bank of new zealand|official cash rate|\bocr\b"),
    ("CHF", r"snb", r"\bsnb\b|swiss national bank"),
]


# ── reading a source ────────────────────────────────────────────────────────────────────────────
def iso(t) -> str:
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


def fetch(name: str, get: Callable) -> List[dict]:
    """One source's latest lines. `get(url, headers, timeout)` -> (status, body bytes). Raises
    when the download fails or the page holds no lines at all -- a page that suddenly parses to
    nothing has changed or is blocked, and the next source must be asked."""
    s = SOURCES[name]
    status, body = get(s.url, {"User-Agent": UA}, TIMEOUT)
    if status != 200:
        raise IOError(f"{s.label}: HTTP {status}")
    lines = PARSERS[s.kind](body)
    if not lines:
        raise ValueError(f"{s.label}: the page holds no headlines")
    return lines


# ── which lines are about a release ─────────────────────────────────────────────────────────────
def release_words(currency: str, title: str) -> tuple:
    """(the headline words for a release, how many minutes after its scheduled time a headline
    still counts) from RELEASE_WORDS -- (None, 0) when no rule names it. The window is 10 minutes
    unless the rule gives more: the BOJ announces when its meeting ends, not at a set time."""
    for rule in RELEASE_WORDS:
        if rule[0] == currency and re.search(rule[1], str(title), re.I):
            return rule[2], (rule[3] if len(rule) > 3 else WINDOW[1].total_seconds() / 60)
    return None, 0             # no headline rule: a guess would only add noise


def window(currency: str, titles) -> Optional[Tuple[str, pd.Timedelta, pd.Timedelta]]:
    """(words, from, until) for one release MOMENT -- the titles of one currency at one minute;
    None when no rule names any of them."""
    pats, minutes = [], 0.0
    for ti in titles:
        w, m = release_words(currency, ti)
        if w is not None and w not in pats:
            pats.append(w)
            minutes = max(minutes, m)
    if not pats:
        return None
    return "|".join(pats), WINDOW[0], pd.Timedelta(minutes=minutes)


_ACTUAL = re.compile(r"\bactual\b", re.I)


def matching(lines: List[dict], t, words: str, since: pd.Timedelta,
             until: pd.Timedelta) -> List[dict]:
    """The lines printed inside [t + since, t + until] that use `words`, best first: a printed
    result ("... Actual 2.5% (Forecast ...)"), then any line with a number, then the rest -- each
    group in time order. Every line gains `at`, its printed time. A line with no time is left out:
    it cannot be placed in the window."""
    t = pd.Timestamp(t)
    out = []
    for x in lines:
        if not x.get("published"):
            continue
        at = pd.Timestamp(x["published"])
        if t + since <= at <= t + until and re.search(words, x["title"], re.I):
            out.append(dict(x, at=at))

    def rank(x):
        s = x["title"]
        return (0 if _ACTUAL.search(s) else 1 if re.search(r"\d", s) else 2, x["at"])
    return sorted(out, key=rank)


_JUNK_HEAD = re.compile(r"^(?:financialjuice\s*:\s*|[^\w(]*breaking\s*:\s*|[*\s]+|[^\w\s(]+\s*)+",
                        re.I)
_JUNK_TAIL = re.compile(r"(?:\s*\|\s*fj|\s*\$[A-Z]+|\s*\(@\w+\))+\s*$", re.I)


def clean(text: str) -> str:
    """A line as a person reads it: no "FinancialJuice:" or "‼ BREAKING:" in front, no "$MACRO|FJ"
    or "(@WalterBloomberg)" behind. The words in between are left exactly as printed."""
    s = _JUNK_TAIL.sub("", _JUNK_HEAD.sub("", text.strip()))
    return s.strip() or text.strip()


# ── does a headline print the official number? ──────────────────────────────────────────────────
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:\s*%|[KMBT]\b)?", re.I)


def headline_number(text: str) -> Optional[float]:
    """The number a line gives as the RESULT, in its printed units: the one after "Actual" when it
    has one ("Actual 245K (Forecast 180K)" -> 245000), else its first ("GDP +3.4% (consensus
    +3.2%)" -> 3.4). A quarter's name is not a number ("Q3" -> skipped). None when it prints none."""
    m = _ACTUAL.search(text)
    rest = text[m.end():] if m else text
    n = _NUM.search(rest)
    if not n:
        return None
    v = parse_value(n.group(0).replace(" ", "").upper())
    return None if v != v else v                          # NaN -> None


_CUES = (("core", r"\bcore\b"), ("price", r"\bprices?\b"))


def _about(title: str, text: str) -> bool:
    """Is a line about THIS title rather than its neighbour typed beside it? A core title wants
    "core" and a headline one refuses it (CPI m/m is not Core CPI m/m); likewise "price" (GDP is
    not the GDP price index); m/m wants a monthly line, y/y a yearly one."""
    t = title.lower()
    for word, rx in _CUES:
        if bool(re.search(rx, text, re.I)) != (word in t):
            return False
    if "m/m" in t and not re.search(r"\bm/?o/?m\b|\bm/m\b|month", text, re.I):
        return False
    if "y/y" in t and not re.search(r"\by/?o/?y\b|\by/y\b|year|annual", text, re.I):
        return False
    return True


def agrees(actual, currency: str, title: str, lines: List[dict]) -> Optional[bool]:
    """Did a headline about THIS title print the official `actual`? True (one of the lines about it
    gives that number as its result); False (they give others); None (no line about it gives a
    number: on GDP day the PCE line says nothing about GDP). "About it" = its words, and its
    neighbours' lines left out (_about) -- one release day prints several numbers in a row."""
    words, _ = release_words(currency, title)
    want = parse_value(actual) if actual is not None else float("nan")
    if words is None or want != want:
        return None
    got = []
    for x in lines:
        text = x.get("text") or x.get("title") or ""
        if re.search(words, text, re.I) and _about(title, text):
            n = headline_number(text)
            if n is not None:
                got.append(n)
    if not got:
        return None
    return any(abs(n - want) <= 1e-9 * max(1.0, abs(want)) for n in got)
