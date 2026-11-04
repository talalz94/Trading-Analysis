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


# ── breaking news: critical lines that are NOT on the calendar ────────────────────────────────
# Measured before it was wired in (experiments/breaking_news_check.py, FinancialJuice's 2,880 posts
# of 19 Oct - 2 Nov 2026, a fortnight of war around Iran and the Gulf):
#   * a WORD LIST alone sent 4.6 a day, a third of them routine Russia/Ukraine strikes and
#     politicians' statements -- a list cannot tell a new event from talk about an old one;
#   * a SHARP MOVE alone sent 4.2 a day at the old floor, and at 1.5x the floor it missed the
#     21 Oct Gulf story (gold fell steadily, not in one 5-minute lurch);
#   * WORDS + MOVE sent 0.5 a day and caught it.
# So TWO tiers. A TOP line goes at once: the few that are critical whatever the market does (an
# ordered departure, an embassy closed, an emergency central-bank move, intervention, war declared).
# A BREAKING line goes only with a sharp move on an alert market from 10 min before it to 15 min
# after (sharp_moves). Talk ("could", "mulls", analysts), reposts (ICYMI) and printed results are
# neither, and Russia/Ukraine strikes (daily; none moved Gold, USDJPY or BTC in the fortnight) are
# not BREAKING -- a TOP line about them still goes.
_PLACES = (r"(?:iran\w*|israel\w*|saudi\w*|uae|emirat\w*|qatar\w*|kuwait\w*|bahrain\w*|oman\w*|"
           r"gulf|lebanon|hezbollah|syria\w*|iraq\w*|yemen\w*|houthi\w*|riyadh|tehran|tel aviv|"
           r"jerusalem|dubai|abu dhabi|doha|(?:super)?tankers?|vessels?|ships?|oil ?fields?|"
           r"refiner\w*|aramco|pipelines?|ports?|oil (?:terminal|facilit|platform|installation)\w*|"
           r"bases?|embass\w*|consulates?|taiwan\w*|nato|us forces|us troops)")
_HORMUZ_SHUT = r"(?:closed|closes|closing|closure|shut\w*|block\w*|attack\w*|mine[sd]?|seiz\w*)"
_FIRE = r"(?:explo(?:d|sion)\w*|blasts?|caught fire|ablaze|ignites?)"
TOP = {
    "evacuation": r"\b(?:ordered departure|(?:evacuat\w*|withdraw\w*|pull\w* out) (?:\w+ ){0,4}"
                  r"(?:us |american )?(?:embass\w*|diplomat\w*|personnel|citizens|nationals|"
                  r"troops|forces|staff)|(?:troops|diplomats|personnel|citizens|nationals|staff)"
                  r"\b.{0,30}\bevacuat\w*|embass(?:y|ies) (?:\w+ ){0,3}(?:clos\w*|attacked|"
                  r"evacuat\w*|suspend\w*))",
    "war declared": r"\b(?:declares? war|declaration of war|martial law|coup\b|nuclear (?:test|"
                    r"strike|attack|explosion|detonation))",
    "central bank": r"\b(?:(?:fed|fomc|ecb|boj|boe|snb|rba|boc|rbnz|central bank)\b.{0,40}"
                    r"\bemergency\b|emergency (?:rate cut|rate hike|fomc|rate decision)|"
                    r"inter-?meeting (?:rate )?(?:cut|hike|decision)|"
                    r"intervened? in (?:the )?(?:fx|currency|yen|forex|foreign exchange)|"
                    r"(?:conducted|confirms?|confirmed|carried out) (?:\w+ ){0,2}intervention|"
                    r"rate check)",
}
BREAKING = {
    # "hit" alone is a price word ("gold hit a record high as Gulf investors bought"): a weapon
    # must do the hitting
    "attack": r"\b(?:(?:attacks?|attacked|targeted|targets|struck|strikes?|bombed|shelled|"
              r"seized)\b(?: \S+){0,4} " + _PLACES + r"\b|"
              r"(?:missiles?|drones?|rockets?|projectiles?)\b(?: \S+){0,2} hit\w*\b|"
              r"missiles?\b.{0,40}\b(?:fired|launched)\b.{0,40}\b(?:at|on|toward|towards|into)\b|"
              r"(?:fired|launched)\b.{0,20}\bmissiles?\b.{0,20}\b(?:at|on|toward|towards|into)\b|"
              r"intercept\w*\b.{0,40}\b(?:missiles?|drones?)\b|"
              + _FIRE + r"\b.{0,40}\b" + _PLACES + r"\b|"
              + _PLACES + r"\b.{0,40}\b(?:" + _FIRE[3:-1] + r"|hit\w* (?:a |naval |sea )?mines?)|"
              r"invad\w*|invasion|war (?:on|with|against)|"
              r"military (?:action|operation|strikes?) (?:on|against|in))",
    "hormuz": r"\b(?:hormuz\b.{0,40}\b" + _HORMUZ_SHUT + r"|" + _HORMUZ_SHUT + r"\b.{0,40}\bhormuz|"
              r"(?:imposes?|declares?|begins?|announces?|starts?) (?:a |naval )?blockade|"
              r"blockade of)",
    "de-escalation": r"\b(?:ceasefire (?:agreed|reached|announced|deal|takes effect|begins|holds)|"
                     r"peace (?:deal|agreement|proposal|plan)|truce|"
                     r"(?:reopen\w*|opening) (?:\w+ ){0,3}(?:strait|hormuz)|"
                     r"lift\w* (?:the )?blockade)",
    "policy": r"\b(?:(?:imposes?|announces?|slaps?|raises?|hikes?|doubles?) (?:[\w%$.]+ ){0,3}"
              r"tariffs?|new tariffs?|tariffs? of \d+%|government shutdown|debt ceiling|"
              r"defaults? on)",
    "stress": r"\b(?:bank (?:failure|collapse|run)|fdic (?:seiz\w*|takes over)|circuit breakers?|"
              r"trading (?:halt\w*|suspended)|flash crash)",
    "crypto": r"\b(?:(?:bitcoin|btc|crypto)\b.{0,40}\b(?:ban(?:s|ned)?|strategic reserve|"
              r"etf (?:approv\w*|reject\w*))|(?:exchange|binance|coinbase|tether|okx|bybit)\b"
              r".{0,40}\b(?:hack\w*|exploit\w*|insolven\w*|halts? withdrawals))",
}
_RU_UA = re.compile(r"\b(?:ukrain\w*|russia\w*|kyiv|kiev|kremlin|zelensk\w*|putin|moscow|crimea|"
                    r"donbas|kherson|zaporizh\w*|oleshky|kursk|belgorod|bryansk|voronezh|lipetsk|"
                    r"sheremetyevo|kharkiv|odesa|odessa)\b", re.I)
_NOT_BREAKING = re.compile(r"\bactual\b.*\bforecast\b|\b(?:icymi|analysts?|preview|explainer|"
                          r"opinion|poll|survey|mulls?|considers?|could|might)\b", re.I)


def breaking_tier(text: str) -> Optional[Tuple[int, str]]:
    """(1, rule) for a TOP line, sent at once; (2, rule) for a BREAKING line, sent only with a
    sharp move; None for everything else."""
    if _NOT_BREAKING.search(text):
        return None
    for name, rx in TOP.items():
        if re.search(rx, text, re.I):
            return 1, name
    if _RU_UA.search(text) and not re.search(r"\bnato\b", text, re.I):
        return None
    for name, rx in BREAKING.items():
        if re.search(rx, text, re.I):
            return 2, name
    return None


def breaking_rule(text: str) -> Optional[str]:
    """The rule a line meets in either tier ("evacuation", "attack", ...), or None."""
    t = breaking_tier(text)
    return None if t is None else t[1]


# the market's own word: a sharp move is a 5-minute return of at least the market's floor AND
# MOVE_K times the median 5-minute move of the day before (the headline report's definition)
MOVE_FLOOR = {"XAU": 0.0025, "BTC": 0.005, "ETH": 0.008}   # by symbol; a 6-letter FX pair: 0.12%
MOVE_FX_FLOOR = 0.0012
MOVE_K = 6.0
MOVE_BEFORE = pd.Timedelta(minutes=10)     # a sharp move this long before a BREAKING line confirms it
MOVE_AFTER = pd.Timedelta(minutes=15)      # ... or this long after it
MOVE_KNOWN = pd.Timedelta(minutes=5)       # a move is known 5 min after it starts
STORY_GAP = pd.Timedelta(minutes=60)       # the same story again within this is not sent again
BREAKING_PER_DAY = 8                       # whatever happens, no more than this a day


def move_floor(symbol: str) -> Optional[float]:
    """A market's sharp-move floor from its symbol (XAUUSD 0.25%, USDJPY 0.12%, BTCUSDT 0.5%), or
    None for one this has no measured floor for -- it then confirms nothing."""
    s = str(symbol).upper().replace("/", "")
    for k, v in MOVE_FLOOR.items():
        if s.startswith(k):
            return v
    return MOVE_FX_FLOOR if re.fullmatch(r"[A-Z]{6}", s) else None


def sharp_moves(bars: pd.DataFrame, floor: float, k: float = MOVE_K) -> pd.DataFrame:
    """Sharp moves in 1m candles [open_time, close]: start (the close the 5 minutes run from) and
    move_pct. One move per 30 minutes (its first minute). A move is KNOWN 5 minutes after start."""
    import numpy as np
    b = bars.sort_values("open_time").reset_index(drop=True)
    c = b["close"].to_numpy(float)
    r5 = np.full(len(c), np.nan)
    r5[:-5] = c[5:] / c[:-5] - 1                         # from this candle's close, 5 later
    typical = pd.Series(np.abs(r5)).shift(1).rolling(1440, min_periods=240).median().to_numpy()
    hit = np.abs(r5) >= np.maximum(floor, k * np.nan_to_num(typical, nan=np.inf))
    out, until = [], None
    for i in np.flatnonzero(hit):
        t = b["open_time"].iat[i] + pd.Timedelta(minutes=1)    # the move starts at this close
        if until is not None and t < until:
            continue
        out.append(dict(start=t, move_pct=round(100 * r5[i], 3)))
        until = t + pd.Timedelta(minutes=30)
    return pd.DataFrame(out, columns=["start", "move_pct"])


_STOP = frozenset("""the and for with from that this have has had will said says say over after
into about amid near than then they their there were been being also more most some such only
says, said: says: reuters bloomberg statement source sources official officials report reports""".split())


def story_words(text: str) -> frozenset:
    """A line's content words, for telling one story from another."""
    return frozenset(w for w in re.findall(r"[a-z0-9]+", text.lower())
                     if len(w) >= 4 and w not in _STOP)


def same_story(a: str, b: str) -> bool:
    """Two lines about one story: 3+ content words shared AND half the shorter line's."""
    x, y = story_words(a), story_words(b)
    n = len(x & y)
    return n >= 3 and n >= 0.5 * min(len(x), len(y))


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
