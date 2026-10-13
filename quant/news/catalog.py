"""
What the code KNOWS about economic events, as opposed to what a source says about them.

Every source here is Forex Factory's calendar seen through a different window (its weekly feed, a
scraped archive, JBlanked's API), so titles are Forex Factory's own: "CPI m/m", "Non-Farm
Employment Change". Four pieces of knowledge sit on top of them:

  CRITICALITY   Forex Factory's impact (low / medium / high), with "critical" above high for the
                handful of releases that move every market -- CPI, payrolls, the central-bank
                decision. Critical is only ever an UPGRADE of a high-impact event, so a source that
                calls an event medium can never be overruled into critical.
  HOME ZONE     the time zone an event is scheduled in. A US release is "8:30 New York" in winter
                and in summer; the same instant in UTC moves by an hour twice a year.
  OFFICIAL TIME the published release time of the US events the research needs, by era: the Fed
                announced at 14:15 until 2012 and at 14:00 since. Used only where a source has the
                DATE but not the time.
  DIRECTION     whether a HIGHER actual is good (+1) or bad (-1) for the currency, or neither (0).
                These are Forex Factory's own "usual effect" rules, written out here so the live
                monitor can judge a release without the archive. tests/test_news.py checks them
                against the archive's rules wherever the archive is present.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import pandas as pd

LEVELS = ("none", "low", "medium", "high", "critical")       # rank = position
RANK = {lv: i for i, lv in enumerate(LEVELS)}

CURRENCIES = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY")
HOME_TZ = {"USD": "America/New_York", "CAD": "America/Toronto", "EUR": "Europe/Berlin",
           "GBP": "Europe/London", "CHF": "Europe/Zurich", "JPY": "Asia/Tokyo",
           "AUD": "Australia/Sydney", "NZD": "Pacific/Auckland", "CNY": "Asia/Shanghai"}

# the default COVERAGE: which currencies, from which level up. USD from medium because every
# market watched here is quoted in dollars; the other majors only when it is big.
DEFAULT_COVERAGE = {"USD": "medium", "EUR": "high", "GBP": "high", "JPY": "high"}


def impact_level(raw) -> str:
    """A source's impact label -> one of LEVELS (never "critical": that is ours to decide)."""
    s = str(raw or "").strip().lower()
    for lv in ("high", "medium", "low"):
        if s.startswith(lv):                       # "High", "High Impact Expected"
            return lv
    return "none"                                  # "Holiday", "Non-Economic", blank


# ── CRITICAL: the releases that move everything. Exact titles, per currency. ──────────────────
CRITICAL = {
    "USD": ("CPI m/m", "CPI y/y", "Core CPI m/m", "Core CPI y/y",
            "Non-Farm Employment Change", "Unemployment Rate", "Average Hourly Earnings m/m",
            "Federal Funds Rate", "FOMC Statement", "FOMC Press Conference",
            "FOMC Economic Projections", "Core PCE Price Index m/m", "PPI m/m", "Core PPI m/m",
            "Retail Sales m/m", "Core Retail Sales m/m", "Advance GDP q/q"),
    "EUR": ("Main Refinancing Rate", "Monetary Policy Statement", "ECB Press Conference",
            "CPI Flash Estimate y/y", "Core CPI Flash Estimate y/y"),
    "GBP": ("Official Bank Rate", "Monetary Policy Summary", "CPI y/y"),
    "JPY": ("BOJ Policy Rate", "Monetary Policy Statement", "BOJ Press Conference"),
    "CAD": ("Overnight Rate", "BOC Rate Statement", "CPI m/m", "Employment Change",
            "Unemployment Rate"),
    "AUD": ("Cash Rate", "RBA Rate Statement", "CPI q/q", "Employment Change"),
    "NZD": ("Official Cash Rate", "RBNZ Rate Statement"),
    "CHF": ("SNB Policy Rate",),
}
_CRIT = {(c, t.lower()) for c, ts in CRITICAL.items() for t in ts}


def criticality(currency: str, title: str, impact: str) -> str:
    """impact (a LEVELS name) upgraded to "critical" for the releases in CRITICAL."""
    if impact == "high" and (currency, str(title).strip().lower()) in _CRIT:
        return "critical"
    return impact


# ── OFFICIAL US RELEASE TIMES (New York), by era ──────────────────────────────────────────────
# Each entry is ((from_date, "HH:MM"), ...): the time in force from that date. Before the first
# date there is no official time and the event keeps whatever its sources say.
_830 = (("1990-01-01", "08:30"),)
_1000 = (("1990-01-01", "10:00"),)
US_OFFICIAL_TIMES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # BLS, BEA and Census release at 8:30 ET -- the "8:30 red folder"
    **{t: _830 for t in (
        "CPI m/m", "CPI y/y", "Core CPI m/m", "Core CPI y/y", "PPI m/m", "PPI y/y",
        "Core PPI m/m", "Core PPI y/y", "Non-Farm Employment Change", "Unemployment Rate",
        "Average Hourly Earnings m/m", "Retail Sales m/m", "Core Retail Sales m/m",
        "Advance GDP q/q", "Prelim GDP q/q", "Final GDP q/q", "Advance GDP Price Index q/q",
        "Prelim GDP Price Index q/q", "Final GDP Price Index q/q", "Core PCE Price Index m/m",
        "Core PCE Price Index y/y", "PCE Price Index m/m", "PCE Price Index y/y",
        "Personal Spending m/m", "Personal Income m/m", "Unemployment Claims",
        "Durable Goods Orders m/m", "Core Durable Goods Orders m/m", "Empire State Manufacturing Index",
        "Philly Fed Manufacturing Index", "Building Permits", "Housing Starts", "Trade Balance",
        "Employment Cost Index q/q", "Import Prices m/m", "Prelim Nonfarm Productivity q/q",
        "Prelim Unit Labor Costs q/q")},
    **{t: _1000 for t in (
        "ISM Manufacturing PMI", "ISM Services PMI", "JOLTS Job Openings",
        "CB Consumer Confidence", "Prelim UoM Consumer Sentiment",
        "Revised UoM Consumer Sentiment", "New Home Sales", "Existing Home Sales",
        "Pending Home Sales m/m", "Factory Orders m/m")},
    "ADP Non-Farm Employment Change": (("2006-01-01", "08:15"),),
    # the FOMC has announced at 2:00 pm since January 2013 (2:15 pm before, 12:30 on the 2011-12
    # press-conference days -- too irregular to state, so no official time before 2013)
    "Federal Funds Rate": (("2013-01-01", "14:00"),),
    "FOMC Statement": (("2013-01-01", "14:00"),),
    "FOMC Economic Projections": (("2013-01-01", "14:00"),),
    "FOMC Press Conference": (("2013-01-01", "14:30"),),
    "FOMC Meeting Minutes": (("2005-01-01", "14:00"),),
}


def official_time(currency: str, title: str, day) -> Optional[str]:
    """The official "HH:MM" (New York) for a US event on `day`, or None."""
    if currency != "USD":
        return None
    eras = US_OFFICIAL_TIMES.get(str(title).strip())
    if not eras:
        return None
    d = str(pd.Timestamp(day).date())
    hhmm = None
    for since, t in eras:
        if d >= since:
            hhmm = t
    return hhmm


# ── DIRECTION: does a higher actual help the currency? Forex Factory's "usual effect" ─────────
# Everything not listed is +1 ("'Actual' greater than 'Forecast' is good for currency").
LOWER_IS_BETTER = re.compile(
    r"unemployment rate|unemployment claims|jobless claims|unemployment change|claimant count|"
    r"challenger job cuts|inventories|natural gas storage|foreign currency reserves|"
    r"public sector net borrowing|asset purchase facility|productivity|mortgage delinquencies",
    re.I)
NO_DIRECTION = re.compile(r"bond auction|refinancing operation|speaks|testifies|holiday|"
                          r"press conference|meeting minutes|statement|projections|election|"
                          r"summit|report|bulletin", re.I)


def direction(title: str) -> int:
    """+1 a higher actual is good for the currency, -1 a lower one is, 0 no consistent effect.

    Speeches, statements and minutes are 0 here: Forex Factory rates them "more hawkish than
    expected is good for currency", which is a judgement, not a number to compare."""
    t = str(title)
    if NO_DIRECTION.search(t) and not re.search(r"rate|balance", t, re.I):
        return 0
    if LOWER_IS_BETTER.search(t):
        return -1
    return 1


# ── CATEGORY: what kind of news, for filters and for the email ────────────────────────────────
_CATEGORY_RULES = (
    ("holiday", r"holiday"),
    ("central bank", r"funds rate|cash rate|bank rate|policy rate|refinancing rate|overnight rate|"
                     r"deposit rate|interest rate|rate statement|monetary policy|fomc|\becb\b|\bboj\b|"
                     r"\bboe\b|\bmpc\b|\bsnb\b|\brba\b|\brbnz\b|\bboc\b|fed chair|federal reserve|"
                     r"asset purchase|beige book|press conference|economic projections"),
    ("inflation", r"\bcpi\b|\bppi\b|\bpce\b|inflation|price index|prices"),
    ("jobs", r"employment|unemployment|payroll|claims|claimant|earnings|wage|job|labor|labour|"
             r"\badp\b|jolts"),
    ("growth", r"\bgdp\b|retail sales|industrial|manufacturing|production|\bpmi\b|\bism\b|"
               r"durable goods|factory orders|business|economic|activity"),
    ("consumer", r"sentiment|confidence|consumer|spending|credit"),
    ("housing", r"home|housing|building|construction|mortgage|house price"),
    ("trade", r"trade balance|current account|exports|imports|capital flows|tic "),
    ("energy", r"crude|oil|natural gas|inventories"),
    ("bonds", r"bond auction|bill auction|note auction|yield"),
)
_CATEGORY_RX = tuple((name, re.compile(rx, re.I)) for name, rx in _CATEGORY_RULES)


def category(title: str) -> str:
    t = str(title)
    for name, rx in _CATEGORY_RX:
        if rx.search(t):
            return name
    return "other"


def parse_coverage(spec) -> Dict[str, str]:
    """{"USD": "medium", "EUR": "high", "*": "critical"} -> validated {currency: level}.

    `"*"` covers every currency not named. A level is the LOWEST criticality kept."""
    if spec is None:
        spec = DEFAULT_COVERAGE
    if isinstance(spec, str):                     # "high" = every currency from high up
        spec = {"*": spec}
    out = {}
    for cur, lv in dict(spec).items():
        cur = str(cur).upper()
        if cur != "*" and cur not in CURRENCIES:
            raise ValueError(f"news coverage: unknown currency {cur!r}; one of {CURRENCIES} or '*'")
        if lv not in RANK:
            raise ValueError(f"news coverage {cur}: level must be one of {LEVELS}, got {lv!r}")
        out[cur] = lv
    return out
