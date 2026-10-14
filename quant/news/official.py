"""
US ACTUALS FROM THE AGENCIES THAT PUBLISH THEM -- the permitted, free and exact source of the number
behind every Forex Factory "actual" for the US releases that move markets.

    history   FRED's archive of every series AS PUBLISHED ON EACH DAY (ALFRED). A release's number is
              computed from the data as it stood on its own release day -- payrolls compare against
              the previous month AS REVISED IN THAT SAME RELEASE, CPI m/m uses that day's seasonal
              factors -- never from today's revised history. One request per series returns every
              snapshot needed (output_type=2 + vintage_dates), so the whole history is ~20 requests.
    live      at the release minute: the Bureau of Labor Statistics' own API for CPI, PPI, payrolls,
              unemployment and hourly earnings (seconds after 8:30, no key); FRED for retail sales,
              PCE, GDP, claims, durable goods, JOLTS and the regional Fed surveys (when FRED posts,
              usually within the hour); the Fed's statement page for the rate decision at 14:00.
              A number counts as NEW only when its reference period is later than the last one
              published -- an unchanged reply can never be mistaken for the release.

Every value is formatted the way Forex Factory prints it ("0.4%", "206K", "7.60M", "4.50%"), rounded
half-up from the published levels the way the agencies round, so the merge can lay it beside the
feed's forecast. The mapping was checked against the archive's actuals, release by release
(validate()); docs/NEWS.md has the result.

Requests: FRED allows 120 a minute (each is spaced); BLS without a key allows 25 a day per IP, so a
day's polls are counted and capped (BLS_API_KEY in .env lifts it to 500).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .catalog import HOME_TZ, official_time
from .sources import ROW_COLS, SourceError, _fred, _get, _rows

NY = HOME_TZ["USD"]


@dataclass(frozen=True)
class Spec:
    title: str                 # Forex Factory's title
    fred: str                  # FRED series (history; live too when `bls` is None)
    how: str                   # pch m/m %, pc1 y/y %, chg change, lin level, gdp, fed
    dec: int                   # decimals as Forex Factory prints them
    unit: str = ""             # "%" | "K" | "M" | ""
    div: float = 1.0           # value / div before printing (claims in persons -> K)
    bls: Optional[str] = None  # the BLS series for the live number
    vintage: bool = True       # True: read as of each release day (the series is revised)


SPECS: Tuple[Spec, ...] = (
    # Bureau of Labor Statistics -- live from its own API
    Spec("CPI m/m", "CPIAUCSL", "pch", 1, "%", bls="CUSR0000SA0"),
    Spec("Core CPI m/m", "CPILFESL", "pch", 1, "%", bls="CUSR0000SA0L1E"),
    Spec("CPI y/y", "CPIAUCNS", "pc1", 1, "%", bls="CUUR0000SA0", vintage=False),
    Spec("Core CPI y/y", "CPILFENS", "pc1", 1, "%", bls="CUUR0000SA0L1E", vintage=False),
    Spec("PPI m/m", "PPIFIS", "pch", 1, "%", bls="WPSFD4"),
    Spec("Core PPI m/m", "PPIFES", "pch", 1, "%", bls="WPSFD49104"),
    Spec("Non-Farm Employment Change", "PAYEMS", "chg", 0, "K", bls="CES0000000001"),
    Spec("Unemployment Rate", "UNRATE", "lin", 1, "%", bls="LNS14000000", vintage=False),
    Spec("Average Hourly Earnings m/m", "CES0500000003", "pch", 1, "%", bls="CES0500000003"),
    # through FRED
    Spec("Retail Sales m/m", "RSAFS", "pch", 1, "%"),
    Spec("Core Retail Sales m/m", "RSFSXMV", "pch", 1, "%"),
    Spec("Core PCE Price Index m/m", "PCEPILFE", "pch", 1, "%"),
    Spec("Personal Spending m/m", "PCE", "pch", 1, "%"),
    Spec("GDP q/q", "A191RL1Q225SBEA", "gdp", 1, "%"),       # -> Advance / Prelim / Final
    Spec("Unemployment Claims", "ICSA", "lin", 0, "K", 1000.0, vintage=False),
    Spec("Durable Goods Orders m/m", "DGORDER", "pch", 1, "%"),
    Spec("Core Durable Goods Orders m/m", "ADXTNO", "pch", 1, "%"),
    Spec("JOLTS Job Openings", "JTSJOL", "lin", 2, "M", 1000.0, vintage=False),
    Spec("Philly Fed Manufacturing Index", "GACDFSA066MSFRBPHI", "lin", 1, vintage=False),
    Spec("Empire State Manufacturing Index", "GACDISA066MSFRBNY", "lin", 1, vintage=False),
    # the Fed -- history from FRED's target range, live from the statement
    Spec("Federal Funds Rate", "DFEDTARU", "fed", 2, "%"),
)
GDP_STAGES = ("Advance", "Prelim", "Final")


def specs_by_title() -> Dict[str, Spec]:
    out = {}
    for s in SPECS:
        if s.how == "gdp":
            out.update({f"{st} GDP q/q": s for st in GDP_STAGES})
        else:
            out[s.title] = s
    return out


# ══ NUMBERS AS FOREX FACTORY PRINTS THEM ═════════════════════════════════════════════════════
def round_half_up(x: float, dec: int) -> float:
    q = Decimal(1).scaleb(-dec)
    return float(Decimal(repr(round(float(x), 10))).quantize(q, rounding=ROUND_HALF_UP))


def fmt(v: float, spec: Spec) -> Optional[str]:
    if v is None or not np.isfinite(v):
        return None
    r = round_half_up(v / spec.div, spec.dec)
    if r == 0:
        r = 0.0                                   # never print "-0.0%"
    return f"{r:.{spec.dec}f}{spec.unit}"


def compute(levels: pd.Series, how: str) -> Tuple[Optional[pd.Timestamp], float]:
    """The latest observation of `levels` (index = observation date, ascending, as one vintage saw
    it) -> (its period, the release's number)."""
    s = levels.dropna()
    if not len(s):
        return None, np.nan
    p, v = s.index[-1], float(s.iloc[-1])
    if how in ("lin", "gdp", "fed"):
        return p, v
    if how in ("pch", "chg"):
        prev = s[s.index < p]
        if not len(prev):
            return p, np.nan
        b = float(prev.iloc[-1])
        return p, (v - b) if how == "chg" else (v / b - 1.0) * 100.0
    if how == "pc1":
        ago = p - pd.DateOffset(years=1)
        if ago not in s.index:
            return p, np.nan
        return p, (v / float(s[ago]) - 1.0) * 100.0
    raise ValueError(how)


# ══ HISTORY ══════════════════════════════════════════════════════════════════════════════════
def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan                             # FRED writes "." for a missing value


def first_prints(key: str, series: str, start) -> pd.DataFrame:
    """Each observation's first published value and the day it was published. A day shared by
    more than 3 observations is the start of FRED's vintage history, not a release."""
    js = _fred("series/observations", key, series_id=series, output_type=4,
               observation_start=str(pd.Timestamp(start).date()),
               realtime_start="1776-07-04", realtime_end="9999-12-31")
    df = pd.DataFrame([dict(period=pd.Timestamp(o["date"]), day=o["realtime_start"],
                            value=_num(o["value"])) for o in js.get("observations", [])],
                      columns=["period", "day", "value"])
    if not len(df):
        return df
    n = df["day"].value_counts()
    return df[df["day"].map(n) <= 3].reset_index(drop=True)


def vintages(key: str, series: str, days: List[str], obs_start,
             chunk: int = 40) -> Dict[str, pd.Series]:
    """{day: the series as it stood on that day} -- one request per `chunk` days."""
    out: Dict[str, pd.Series] = {}
    for k in range(0, len(days), chunk):
        part = days[k:k + chunk]
        js = _fred("series/observations", key, series_id=series, output_type=2,
                   observation_start=str(pd.Timestamp(obs_start).date()),
                   vintage_dates=",".join(part))
        obs = js.get("observations", [])
        for d in part:
            col = f"{series}_{d.replace('-', '')}"
            s = pd.Series({pd.Timestamp(o["date"]): _num(o.get(col)) for o in obs if col in o},
                          dtype=float).sort_index()
            out[d] = s
    return out


def _row(title: str, day: str, value: float, spec: Spec, period) -> Optional[dict]:
    """A release row at its official New York time. A WEEKEND day is not a release day -- it is
    FRED posting late (the Philly Fed survey of 17 Nov 2022 is stamped Sunday the 27th) -- so the
    number is kept with its date only, and the calendar ties it to the real release
    (calendar._align_official)."""
    hhmm = official_time("USD", title, day) or ("14:00" if spec.how == "fed" else None)
    txt = fmt(value, spec)
    if hhmm is None or txt is None:
        return None
    weekend = pd.Timestamp(day).dayofweek >= 5
    t = pd.Timestamp(f"{day} {'12:00' if weekend else hhmm}").tz_localize(NY).tz_convert("UTC")
    # no impact: that is Forex Factory's rating, and the calendar takes it from there
    return dict(t=t, time_q="date" if weekend else "official", currency="USD", title=title,
                impact=None, actual=txt, forecast=None, previous=None, source="official",
                period=pd.Timestamp(period), series=spec.fred)


def history(key: str, start="2022-01-01", *, specs: Iterable[Spec] = SPECS,
            fomc_days: Iterable[str] = (), gdp_days: Optional[Dict[str, str]] = None,
            after: Optional[Dict[str, str]] = None, workers: int = 4,
            log=None) -> pd.DataFrame:
    """Every spec's releases since `start`, each as published that day. `after` {FRED series:
    last release day stored} fetches only what is newer (an incremental update); `fomc_days` are
    the Fed's decision days (the store's fomc rows); `gdp_days` {day: "Prelim GDP q/q" ...} the
    GDP release days from FRED's release calendar (the store's fred rows).

    `workers` series are fetched at once. FRED's limit (120 requests a minute) is kept by the
    pacer every request goes through (sources._fred), not by sleeping here, so the workers only
    hide the network's latency: ~98 s to ~25 s for 150 days.

    GDP NEEDS THOSE DAYS. Its 2nd and 3rd estimates revise a quarter already printed, and FRED
    records a revision only when a value CHANGES: the 2nd estimates of 27 Nov 2024 and 27 Feb 2025
    repeated the 1st (2.8%, 2.3%) and left no trace, so counting FRED's revision days put every
    later estimate one stage off. Without `gdp_days` that fallback is still used."""
    from concurrent.futures import ThreadPoolExecutor
    after = after or {}
    specs = list(specs)
    obs_start = pd.Timestamp(start) - pd.DateOffset(months=15)

    def one(spec: Spec) -> List[dict]:
        return _spec_history(key, spec, start, obs_start, after.get(spec.fred), fomc_days,
                             gdp_days)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        parts = list(pool.map(one, specs))          # in order; the first failure is raised
    rec = []
    for spec, part in zip(specs, parts):
        rec += part
        if log:
            log(f"  {spec.title}: {len(part)} releases")
    return pd.DataFrame(rec, columns=ROW_COLS + ["period", "series"])


def _spec_history(key, spec: Spec, start, obs_start, done, fomc_days, gdp_days) -> List[dict]:
    """One spec's releases since `start` (after `done`, its last stored day)."""
    rec: List[dict] = []
    if spec.how == "fed":
        return _fed_history(key, spec, fomc_days, start, done)
    if spec.how == "gdp" and gdp_days:
        days = sorted(d for d in gdp_days if d >= str(pd.Timestamp(start).date())
                      and (done is None or d > done))
        vs = vintages(key, spec.fred, days, obs_start) if days else {}
        for d in days:
            per, v = compute(vs[d], "lin")
            r = _row(gdp_days[d], d, v, spec, per)
            if r:
                rec.append(r)
        return rec
    if spec.how == "gdp":
        # fallback: count FRED's revision days per quarter (misses an unchanged estimate)
        vd = _fred("series/vintagedates", key, series_id=spec.fred, limit=10000)
        days_all = [d for d in vd.get("vintage_dates", []) if d >= str(pd.Timestamp(start).date())]
        fp = pd.DataFrame({"day": days_all})
    else:
        fp = first_prints(key, spec.fred, obs_start)
        fp = fp[fp["day"] >= str(pd.Timestamp(start).date())]
        if done:
            fp = fp[fp["day"] > done]
    if not len(fp) or (done and not (fp["day"] > done).any()):
        return rec
    if not spec.vintage:
        allfp = first_prints(key, spec.fred, obs_start - pd.DateOffset(years=1)) \
            if spec.how == "pc1" else None
        base = (allfp if allfp is not None else fp).set_index("period")["value"].sort_index()
        # two months on one day (BLS catching up after the 2025 shutdown): the release is
        # the LATER month, as every calendar prints it
        fp = fp.sort_values("period").drop_duplicates("day", keep="last")
        for p, d in zip(fp["period"], fp["day"]):
            per, v = compute(base[base.index <= p], spec.how)
            r = _row(spec.title, d, v, spec, per)
            if r:
                rec.append(r)
        return rec
    days = sorted(set(fp["day"]))
    vs = vintages(key, spec.fred, days, obs_start)
    seen: Dict[pd.Timestamp, int] = {}
    for d in days:
        per, v = compute(vs[d], "lin" if spec.how == "gdp" else spec.how)
        title = spec.title
        if spec.how == "gdp":
            k = seen.get(per, 0)
            seen[per] = k + 1
            if k >= len(GDP_STAGES):
                continue                          # an annual revision, not an estimate
            title = f"{GDP_STAGES[k]} GDP q/q"
        r = _row(title, d, v, spec, per)
        if r and not (done and d <= done):
            rec.append(r)
    return rec


def _fed_history(key, spec, fomc_days, start, done) -> List[dict]:
    """The upper bound of the target range in force the day AFTER each decision."""
    days = sorted(d for d in fomc_days if d >= str(pd.Timestamp(start).date())
                  and (done is None or d > done))
    if not days:
        return []
    js = _fred("series/observations", key, series_id=spec.fred,
               observation_start=str((pd.Timestamp(days[0]) - pd.Timedelta(days=5)).date()))
    s = pd.Series({pd.Timestamp(o["date"]): _num(o["value"]) for o in js.get("observations", [])},
                  dtype=float).sort_index()
    out = []
    for d in days:
        nxt = s[s.index > pd.Timestamp(d)]
        if len(nxt):
            r = _row(spec.title, d, float(nxt.iloc[0]), spec, pd.Timestamp(d))
            if r:
                out.append(r)
    return out


# ══ LIVE ═════════════════════════════════════════════════════════════════════════════════════
BLS_V1 = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
FED_STATEMENT = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm"


def bls_levels(series: List[str], year: int, key: Optional[str] = None,
               timeout: float = 30) -> Dict[str, pd.Series]:
    """The BLS series' monthly levels over the last two years and this one, as published right
    now. TWO back, not one: December's y/y comes out in January and needs the December before."""
    import requests
    body = {"seriesid": list(series), "startyear": str(year - 2), "endyear": str(year)}
    if key:
        body["registrationkey"] = key
    r = requests.post(BLS_V2 if key else BLS_V1, json=body, timeout=timeout,
                      headers={"Content-Type": "application/json"})
    if r.status_code != 200:
        raise SourceError(f"BLS: HTTP {r.status_code}")
    js = r.json()
    if js.get("status") != "REQUEST_SUCCEEDED":
        raise SourceError(f"BLS: {js.get('status')} {' '.join(js.get('message') or [])[:160]}")
    out = {}
    for s in js.get("Results", {}).get("series", []):
        pts = {pd.Timestamp(f"{x['year']}-{x['period'][1:]}-01"): _num(x["value"])
               for x in s.get("data", []) if re.fullmatch(r"M(0[1-9]|1[0-2])", x["period"])}
        out[s["seriesID"]] = pd.Series(pts, dtype=float).sort_index()
    return out


def fred_now(key: str, series: str, obs_start, timeout: float = 30) -> pd.Series:
    """The series as FRED has it now."""
    js = _fred("series/observations", key, series_id=series, timeout=timeout,
               observation_start=str(pd.Timestamp(obs_start).date()))
    return pd.Series({pd.Timestamp(o["date"]): _num(o["value"])
                      for o in js.get("observations", [])}, dtype=float).sort_index()


def fred_posted(key: str, series: str, day: str, timeout: float = 30) -> bool:
    """Did FRED post a new version of `series` on `day` (the New York release day)?

    Its observations CANNOT say: FRED's default real-time period is today, so every value comes
    back with today as its realtime_start, whatever day it was published. Trusting that let the
    previous release through -- 14 Oct 2026, the BLS API answered 503 at 8:30, and August's CPI
    (0.4% m/m) was taken for September's (0.6%), its meaning reversed on three of four lines.
    The series' list of vintage dates is the record of when it changed."""
    try:
        js = _fred("series/vintagedates", key, series_id=series, realtime_start=day,
                   timeout=timeout)
    except SourceError as e:
        if "after today" in str(e):     # FRED's own date has not reached the release day yet
            return False
        raise
    return day in js.get("vintage_dates", [])


_FRACTION = re.compile(r"(\d+)(?:-(\d)/(\d))?|(\d)/(\d)")


def _rate(txt: str) -> float:
    """'4-1/2' -> 4.5, '1/4' -> 0.25, '4.50' -> 4.5"""
    txt = txt.strip()
    if re.fullmatch(r"\d+(\.\d+)?", txt):
        return float(txt)
    m = re.fullmatch(r"(\d+)-(\d)/(\d)", txt)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.fullmatch(r"(\d)/(\d)", txt)
    if m:
        return int(m.group(1)) / int(m.group(2))
    raise ValueError(txt)


def parse_fed_statement(html: str) -> Optional[float]:
    """The upper bound of the federal funds target range stated in an FOMC statement."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    # a cut or a hike names its size first: "...federal funds rate by 1/2 percentage point to
    # 4-3/4 to 5 percent" (18 Sep 2024)
    m = re.search(r"target range for the federal funds rate "
                  r"(?:by [\d./-]+ percentage points? )?(?:at|to|of) "
                  r"([\d./-]+) (?:to|and) ([\d./-]+) percent", text)
    return _rate(m.group(2)) if m else None


def fed_decision(day, timeout: float = 20) -> Optional[float]:
    r = _get(FED_STATEMENT.format(ymd=pd.Timestamp(day).strftime("%Y%m%d")), timeout=timeout,
             headers={"User-Agent": "Mozilla/5.0 (quant-news)"})
    return parse_fed_statement(r.text) if r.status_code == 200 else None


LIVE_TIMEOUT = 12          # seconds per request at the release minute
LIVE_BUDGET = 25           # seconds for one whole poll: the alerts' candle checks wait behind it


def live(key: Optional[str], pending: pd.DataFrame, baseline: Dict[str, pd.Timestamp], *,
         bls_key: Optional[str] = None, bls_ok: bool = True,
         budget: float = LIVE_BUDGET) -> Tuple[pd.DataFrame, List[str]]:
    """Look for the numbers of `pending` releases (rows with t, title) that have come out.
    `baseline` {title: the last period already published} -- a reply whose latest period is not
    LATER than that is the old number, not the release, from the BLS and from FRED alike (FRED
    must also have posted on the release day: fred_posted). Returns (new rows, notes). Stops
    asking once `budget` seconds have gone; what is left is asked for on the next poll."""
    deadline = time.monotonic() + budget
    by_title = specs_by_title()
    rec, notes = [], []
    pend = [(r["title"], pd.Timestamp(r["t"])) for r in pending.to_dict("records")
            if r["title"] in by_title]
    if not pend:
        return pd.DataFrame(columns=ROW_COLS + ["period", "series"]), notes
    year = max(t for _, t in pend).tz_convert(NY).year
    bls_ids = sorted({by_title[ti].bls for ti, _ in pend if by_title[ti].bls})
    levels: Dict[str, pd.Series] = {}
    if bls_ids and bls_ok:
        try:
            levels = bls_levels(bls_ids, year, bls_key, timeout=LIVE_TIMEOUT)
        except Exception as e:
            notes.append(f"BLS: {type(e).__name__}: {e}")
    def one(title: str, t: pd.Timestamp):
        """-> (row or None, note or None, skipped for time)."""
        spec = by_title[title]
        day = str(t.tz_convert(NY).date())
        if time.monotonic() > deadline:
            return None, None, True
        try:
            if spec.how == "fed":
                v, per = fed_decision(day, timeout=LIVE_TIMEOUT), pd.Timestamp(day)
                if v is None:
                    return None, None, False
            elif spec.bls and spec.bls in levels:
                base = baseline.get(title)
                if base is None:
                    return None, f"{title}: no baseline period -- not accepted", False
                per, v = compute(levels[spec.bls], spec.how)
                if per is None or not per > pd.Timestamp(base):
                    return None, None, False      # still the previous number
            elif key:
                # BOTH tests, like the BLS path: FRED changed the series on the release day, AND
                # what it now holds is a later period than the last one published (per title, so
                # each GDP stage moves on a quarter too). A day's vintage alone could be a revision
                base = baseline.get(title)
                if base is None:
                    return None, f"{title}: no baseline period -- not accepted", False
                if not fred_posted(key, spec.fred, day, timeout=LIVE_TIMEOUT):
                    return None, None, False      # FRED has not posted this release yet
                s = fred_now(key, spec.fred, t - pd.DateOffset(months=15), timeout=LIVE_TIMEOUT)
                per, v = compute(s, "lin" if spec.how == "gdp" else spec.how)
                if per is None or not per > pd.Timestamp(base):
                    return None, None, False      # posted today, but still the old period
            else:
                return None, None, False
        except Exception as e:
            return None, f"{title}: {type(e).__name__}: {e}", False
        r = _row(title, day, v, spec, per)
        if r:
            r["t"] = t
        return r, None, False

    # the titles are asked at once (FRED's pacer still spaces the requests): CPI's four, one
    # after another, took 25 s on 14 Oct 2026 and left the y/y pair for the next poll
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, len(pend))) as pool:
        got = list(pool.map(lambda x: one(*x), pend))
    rec = [r for r, _, _ in got if r]
    notes += [n for _, n, _ in got if n]
    if any(late for _, _, late in got):
        notes.append("time budget spent -- the rest next poll")
    return pd.DataFrame(rec, columns=ROW_COLS + ["period", "series"]), notes


# ══ HOW GOOD IS IT ═══════════════════════════════════════════════════════════════════════════
def validate(official: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Per title: releases on the same New York day in both, and how many print the SAME actual."""
    day = lambda s: pd.to_datetime(s, utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    o = official.assign(day=day(official["t"]))
    a = archive[archive["currency"] == "USD"].assign(day=day(archive["t"]))
    a = a[(a["day"] >= o["day"].min()) & (a["day"] <= o["day"].max())]
    rows = []
    for title, g in o.groupby("title"):
        ga = a[(a["title"] == title) & a["actual"].notna()]
        m = g.merge(ga[["day", "actual"]], on="day", suffixes=("", "_ff"))
        same = (m["actual"] == m["actual_ff"])
        rows.append(dict(title=title, official=len(g), archive=len(ga), both=len(m),
                         same=int(same.sum()),
                         match=round(float(same.mean()), 3) if len(m) else np.nan,
                         dates_missing=len(set(ga["day"]) - set(g["day"])),
                         differ=", ".join(f"{d} {x}/{y}" for d, x, y in
                                          m.loc[~same, ["day", "actual", "actual_ff"]]
                                          .head(4).itertuples(index=False))))
    out = pd.DataFrame(rows, columns=["title", "official", "archive", "both", "same", "match",
                                      "dates_missing", "differ"])
    return out.sort_values("match").reset_index(drop=True)
