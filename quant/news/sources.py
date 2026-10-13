"""
The four windows onto the calendar. Each returns ROWS in one shape, and none of them decides
anything -- `calendar.build_events` merges them, field by field, by how much each is trusted.

ROW SHAPE: t (UTC) · time_q · currency · title · impact · actual · forecast · previous · source

time_q says how the time was obtained, best first:
  exact     the source states the instant (the weekly feed; a timed archive row)
  official  the source gave the DATE, the time is the US official one (catalog.US_OFFICIAL_TIMES)
  typical   the source gave the DATE, the time is the same event's nearest timed release
  date      the date is known and the time is not; t is NOON in the home zone, so the date is
            stable under any conversion -- and nothing time-critical uses such a row
  allday    a holiday

  FF FEED    https://nfs.faireconomy.media/ff_calendar_thisweek.json -- Forex Factory's own
             export: THIS week, every currency, exact times with their offset, impact, forecast,
             previous. No actual. No key. Asked at most every `min_interval` (it rate-limits).
             The calendar PAGES are behind a Cloudflare bot check and are not scraped.
  ARCHIVE    a third party's scrape of Forex Factory 2007-01 .. 2025-04 (Hugging Face,
             Tropstan/Forex_Factory_Calendar), research only, never committed. Two faults, both
             measured and both repaired here:
               - it was scraped in TEHRAN time and stamped with Iran's daylight saving after Iran
                 abolished it in 2022, so 2023-25 times sit an hour early. The wall time is
                 re-read with the true zone, which puts claims back on 8:30 and ISM on 10:00.
               - 30% of rows carry midnight: Forex Factory prints a time only on the first row of
                 a group, and the scrape was re-sorted afterwards. Their DATE is right (24 of 24
                 CPI dates 2023-24 checked); their time comes from catalog or history.
  JBLANKED   jblanked.com's API (free key, one request per 5 minutes): Forex Factory's calendar
             WITH actuals, by date range. Its timestamps ("2024.02.08 15:30:00") carry NO zone.
             Its own client's offsets (GMT = 3, EST = 7) point at MetaTrader broker time, i.e.
             New York + 7 h, which follows US daylight saving -- but that is a guess from
             constants, so the rows are stored RAW (`wall`) and the zone is decided when the
             calendar is built, from all the evidence (calendar.calibrate_jb_zone).
  FRED       St. Louis Fed (free key): the OFFICIAL release dates of US releases, past and
             scheduled. Dates only -- the time is the official one.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .catalog import HOME_TZ, impact_level, official_time

ROW_COLS = ["t", "time_q", "currency", "title", "impact", "actual", "forecast", "previous",
            "source"]
TIME_Q = ("exact", "official", "typical", "date", "allday")
UA = "quant-news/1.0 (+personal research; contact via repo owner)"


class SourceError(RuntimeError):
    pass


def _blank_to_none(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _rows(df: pd.DataFrame) -> pd.DataFrame:
    for c in ROW_COLS:
        if c not in df.columns:
            df[c] = None
    out = df[ROW_COLS].copy()
    out["t"] = pd.to_datetime(out["t"], utc=True)
    for c in ("actual", "forecast", "previous"):
        out[c] = out[c].map(_blank_to_none).astype("object")
    return out.reset_index(drop=True)


def _get(url, *, params=None, headers=None, timeout=20):
    import requests
    h = {"User-Agent": UA}
    h.update(headers or {})
    r = requests.get(url, params=params, headers=h, timeout=timeout)
    return r


# ══ FOREX FACTORY WEEKLY FEED ════════════════════════════════════════════════════════════════
FF_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def fetch_ff_week(*, timeout: float = 20) -> list:
    r = _get(FF_FEED_URL, timeout=timeout)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or "json" not in ctype:
        # the feed answers a too-frequent caller with an HTML page, not an error code
        raise SourceError(f"Forex Factory feed: HTTP {r.status_code} {ctype or ''} "
                          f"{r.text[:120]!r}".strip())
    data = r.json()
    if not isinstance(data, list):
        raise SourceError("Forex Factory feed: expected a list of events")
    return data


def parse_ff_week(items: Iterable[dict]) -> pd.DataFrame:
    """The feed's events -> rows. Its `date` carries the New York offset, so it is exact."""
    rec = []
    for e in items:
        try:
            t = pd.Timestamp(e["date"]).tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        imp = str(e.get("impact", "")).strip().lower()
        allday = imp == "holiday"
        rec.append(dict(t=t, time_q="allday" if allday else "exact",
                        currency=str(e.get("country", "")).upper().strip(),
                        title=" ".join(str(e.get("title", "")).split()),
                        impact=impact_level(e.get("impact")),
                        actual=e.get("actual"), forecast=e.get("forecast"),
                        previous=e.get("previous"), source="ff_feed"))
    return _rows(pd.DataFrame(rec, columns=ROW_COLS))


# ══ THE ARCHIVE (research only) ══════════════════════════════════════════════════════════════
ARCHIVE_URL = ("https://huggingface.co/datasets/Tropstan/Forex_Factory_Calendar/resolve/main/"
               "forex_factory_cache.csv")
ARCHIVE_ZONE = "Asia/Tehran"          # the zone it was scraped in -- see the module docstring


def download_archive(dest: str, *, timeout: float = 600) -> str:
    """Stream the archive CSV (~68 MB) to `dest`."""
    import requests
    with requests.get(ARCHIVE_URL, stream=True, timeout=timeout, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


def _home_instant(dates: pd.Series, hhmm: pd.Series, zones: pd.Series) -> pd.Series:
    """date + "HH:MM" read in each row's home zone -> UTC. Rows are grouped by zone, so this is
    one vectorised localisation per zone, not one per row."""
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns, UTC]")
    wall = pd.to_datetime(dates.astype(str) + " " + hhmm.astype(str), errors="coerce")
    for z, idx in zones.groupby(zones).groups.items():
        w = wall.loc[idx]
        loc = w.dt.tz_localize(z, ambiguous="NaT", nonexistent="shift_forward")
        out.loc[idx] = loc.dt.tz_convert("UTC")
    return out


def parse_archive(raw: pd.DataFrame) -> pd.DataFrame:
    """The scraped CSV (DateTime, Currency, Impact, Event, Actual, Forecast, Previous[, Detail])
    -> rows, every time repaired as described in the module docstring."""
    s = raw["DateTime"].astype(str)
    day, wall = s.str[:10], s.str[11:19]
    df = pd.DataFrame({
        "currency": raw["Currency"].astype(str).str.upper().str.strip(),
        "title": raw["Event"].astype(str).map(lambda x: " ".join(x.split())),
        "impact": raw["Impact"].map(impact_level),
        "actual": raw["Actual"], "forecast": raw["Forecast"], "previous": raw["Previous"],
        "source": "ff_archive"})
    allday = wall.eq("23:59:59") | raw["Impact"].astype(str).str.startswith("Non-Econ") & \
        raw["Event"].astype(str).str.contains("Holiday", case=False)
    timed = ~wall.isin(["00:00:00", "23:59:59"])
    zone = df["currency"].map(HOME_TZ).fillna("UTC")

    # 1. timed rows: the wall time as written, in the zone it was REALLY in
    t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    w = pd.to_datetime(s[timed].str[:19], errors="coerce")
    # the hour Tehran's clocks went BACK happens twice (2 rows in 83k). The FIRST pass, still
    # daylight time, is taken: it puts the FOMC press conference of 21 Sep 2022 on its official
    # 14:30 New York, where the second pass would put it at 15:30
    t[timed] = w.dt.tz_localize(ARCHIVE_ZONE, ambiguous=np.ones(len(w), bool),
                                nonexistent="shift_forward").dt.tz_convert("UTC")
    timed = timed & t.notna()                 # an unparseable stamp is treated as untimed
    q = pd.Series(np.where(timed, "exact", None), index=df.index, dtype="object")

    # 2. untimed rows: the DATE is the archive's (a Tehran date); the time comes from the
    #    official US schedule, else from the same event's nearest timed release (home zone).
    need = ~timed & ~allday
    off = pd.Series([official_time(c, ti, d) for c, ti, d in
                     zip(df["currency"][need], df["title"][need], day[need])],
                    index=df.index[need], dtype="object")
    home_t = t.dt.tz_convert("UTC")
    # nearest timed sibling's HOME wall time
    sib = pd.Series(None, index=df.index, dtype="object")
    if timed.any():
        hw = pd.Series(index=df.index, dtype="object")
        for z, idx in zone[timed].groupby(zone[timed]).groups.items():
            hw.loc[idx] = home_t.loc[idx].dt.tz_convert(z).dt.strftime("%H:%M")
        order = pd.to_datetime(day, errors="coerce")
        key = df["currency"] + "|" + df["title"]
        frame = pd.DataFrame({"key": key, "d": order, "hw": hw.where(timed)})
        frame = frame.sort_values(["key", "d"], kind="stable")
        fw = frame.groupby("key")["hw"].ffill()
        bw = frame.groupby("key")["hw"].bfill()
        fd = frame["d"].where(frame["hw"].notna()).groupby(frame["key"]).ffill()
        bd = frame["d"].where(frame["hw"].notna()).groupby(frame["key"]).bfill()
        use_f = (frame["d"] - fd).abs() <= (bd - frame["d"]).abs()
        pick = np.where(fw.notna() & (use_f | bw.isna()), fw, bw)
        sib.loc[frame.index] = pick
    hhmm = off.reindex(df.index)
    from_official = need & hhmm.notna()
    hhmm = hhmm.where(from_official, sib.where(need))
    has = need & hhmm.notna()
    if has.any():
        # the archive's date is a TEHRAN date: pick the home-zone day whose instant falls on it
        cand = []
        base = pd.to_datetime(day[has])
        for shift in (0, -1, 1):
            d = (base + pd.Timedelta(days=shift)).dt.strftime("%Y-%m-%d")
            inst = _home_instant(d, hhmm[has], zone[has])
            ok = inst.dt.tz_convert(ARCHIVE_ZONE).dt.strftime("%Y-%m-%d") == day[has]
            cand.append(inst.where(ok))
        best = cand[0].fillna(cand[1]).fillna(cand[2])
        best = best.fillna(_home_instant(day[has], hhmm[has], zone[has]))
        t[has] = best
        q[has & from_official] = "official"
        q[has & ~from_official] = "typical"
    # 3. no time anywhere: noon at home on the archive's date
    rest = (need & ~has) | allday
    if rest.any():
        t[rest] = _home_instant(day[rest], pd.Series("12:00", index=day[rest].index), zone[rest])
        q[need & ~has] = "date"
        q[allday] = "allday"
    df["t"], df["time_q"] = t, q
    return _rows(df)


def archive_directions(raw: pd.DataFrame) -> pd.DataFrame:
    """Forex Factory's own "usual effect" per (currency, title) -> +1 / -1 / 0, from the archive's
    Detail column. Used to CHECK catalog.direction, and preferred to it where present."""
    if "Detail" not in raw.columns:
        return pd.DataFrame(columns=["currency", "title", "direction"])
    ue = raw["Detail"].astype(str).str.extract(r"Usual Effect:?\s*([^;|\n]*)", expand=False)
    d = pd.Series(np.select([ue.str.contains("greater than", na=False),
                             ue.str.contains("less than", na=False),
                             ue.str.contains("hawkish", case=False, na=False)],
                            [1, -1, 0], default=0), index=raw.index)
    df = pd.DataFrame({"currency": raw["Currency"].astype(str).str.upper().str.strip(),
                       "title": raw["Event"].astype(str).map(lambda x: " ".join(x.split())),
                       "direction": d, "has": ue.notna()})
    df = df[df["has"]]
    # the LATEST rule wins: 9 of 602 events changed their rule over 18 years
    return df.groupby(["currency", "title"], as_index=False)["direction"].last()


# ══ JBLANKED (Forex Factory with actuals) ════════════════════════════════════════════════════
JB_BASE = "https://www.jblanked.com/news/api"
JB_MIN_INTERVAL_S = 300                       # the free tier: one request every 5 minutes


def fetch_jb(key: str, *, kind: str = "range", start=None, end=None, source="forex-factory",
             timeout: float = 30) -> list:
    """`kind` today | week | range (with `start`/`end` dates). One request -- the caller owns the
    rate limit (JB_MIN_INTERVAL_S)."""
    if not key:
        raise SourceError("JBlanked: no API key (JBLANKED_API_KEY in .env)")
    url = f"{JB_BASE}/{source}/calendar/{kind}/"
    params = {}
    if kind == "range":
        params = {"from": str(pd.Timestamp(start).date()), "to": str(pd.Timestamp(end).date())}
    r = _get(url, params=params, timeout=timeout,
             headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json"})
    if r.status_code != 200:
        why = r.text[:160]
        if "credits" in why.lower():
            why += " -- this JBlanked endpoint needs paid credits (jblanked.com/api/billing)"
        raise SourceError(f"JBlanked {kind}: HTTP {r.status_code} {why!r}")
    data = r.json()
    if isinstance(data, dict):                 # tolerate a wrapped list
        for k in ("data", "results", "events"):
            if isinstance(data.get(k), list):
                return data[k]
        raise SourceError(f"JBlanked {kind}: unexpected payload keys {sorted(data)[:8]}")
    return data


def _field(e: dict, *names):
    low = {str(k).lower(): v for k, v in e.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _num_text(v) -> Optional[str]:
    """JBlanked sends numbers; keep them as text so every source parses through one path."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (int, float)):
        return repr(float(v)).rstrip("0").rstrip(".") if float(v) != 0 else "0"
    return _blank_to_none(v)


JB_RAW_COLS = ["wall", "currency", "title", "impact", "actual", "forecast", "previous",
               "source"]
# MetaTrader broker time: New York wall time + 7 h. Not an IANA zone, so it is named here and
# handled by `localize_wall`. It follows US daylight saving, which no fixed offset and no European
# zone does: the weeks when US and EU clocks change on different Sundays tell them apart.
NY7 = "America/New_York+7"


def localize_wall(wall, zone: str) -> pd.Series:
    """Naive wall times read in `zone` (an IANA name or NY7) -> UTC. The repeated hour of a
    clock change takes its first pass; a skipped one is shifted forward."""
    w = pd.to_datetime(pd.Series(wall))
    if zone == NY7:
        w, zone = w - pd.Timedelta(hours=7), "America/New_York"
    loc = w.dt.tz_localize(zone, ambiguous=np.ones(len(w), bool), nonexistent="shift_forward")
    return loc.dt.tz_convert("UTC")


def parse_jb(items: Iterable[dict]) -> pd.DataFrame:
    """JBlanked's events -> RAW rows (JB_RAW_COLS): the naive `wall` time exactly as sent, no
    zone applied. `jb_rows` turns them into calendar rows once the zone is known."""
    rec = []
    for e in items:
        raw_t = _field(e, "Date", "date", "time")
        try:
            wall = pd.Timestamp(str(raw_t).replace(".", "-", 2))
        except (ValueError, TypeError):
            continue
        if wall is pd.NaT or wall.tzinfo is not None:
            continue
        rec.append(dict(wall=wall, currency=str(_field(e, "Currency") or "").upper().strip(),
                        title=" ".join(str(_field(e, "Name", "title", "event") or "").split()),
                        impact=impact_level(_field(e, "Impact")),
                        actual=_num_text(_field(e, "Actual")),
                        forecast=_num_text(_field(e, "Forecast")),
                        previous=_num_text(_field(e, "Previous")), source="jblanked"))
    out = pd.DataFrame(rec, columns=JB_RAW_COLS)
    out["wall"] = pd.to_datetime(out["wall"])
    return out


def jb_rows(raw: pd.DataFrame, zone: Optional[str]) -> pd.DataFrame:
    """Raw JBlanked rows -> calendar rows. With a zone their time is `exact`; without one they
    carry their DATE only (noon at home, time_q "date"), so they can still lend an actual to
    the same release from another source but never decide its time."""
    if raw is None or not len(raw):
        return _rows(pd.DataFrame(columns=ROW_COLS))
    df = raw.copy()
    if zone:
        df["t"] = localize_wall(df["wall"], zone).to_numpy()
        df["time_q"] = "exact"
    else:
        zones = df["currency"].map(HOME_TZ).fillna("UTC")
        day = pd.to_datetime(df["wall"]).dt.strftime("%Y-%m-%d")
        df["t"] = _home_instant(day, pd.Series("12:00", index=df.index), zones)
        df["time_q"] = "date"
    return _rows(df)


# ══ FRED (official US release dates) ═════════════════════════════════════════════════════════
FRED_BASE = "https://api.stlouisfed.org/fred"
# FRED release (by NAME, resolved to an id at run time so a wrong id cannot creep in) -> the
# series whose FIRST PRINT is that release's headline, and the Forex Factory titles it carries.
#
# WHY FIRST PRINTS. FRED's release calendar lists every day FRED's data for a release CHANGED,
# not only the headline publication: CPI's February seasonal-factor revisions, retail sales' April
# annual revision, payrolls' August benchmark preview ... about one phantom date a year per
# release, each of which would have become a "CPI at 8:30" critical event. A series observation's
# first realtime_start (output_type=4) is the day it was FIRST published: exactly one per period,
# the headline day. Checked 2022-2025 against the archive: identical dates.
# The FOMC is not here: FRED's "FOMC Press Release" changes almost every business day (3,894
# dates since 2014). Rate decisions come from the Fed's own calendar (fomc_rows).
FRED_RELEASES: Dict[str, tuple] = {
    "Consumer Price Index": (("CPIAUCSL",),
                             ("CPI m/m", "Core CPI m/m", "CPI y/y", "Core CPI y/y")),
    "Producer Price Index": (("PPIFIS", "PPIFGS"), ("PPI m/m", "Core PPI m/m")),
    "Employment Situation": (("PAYEMS",), ("Non-Farm Employment Change", "Unemployment Rate",
                                           "Average Hourly Earnings m/m")),
    "Advance Monthly Sales for Retail and Food Services": (("RSAFS",), ("Retail Sales m/m",
                                                                        "Core Retail Sales m/m")),
    "Personal Income and Outlays": (("PCEPI",), ("Core PCE Price Index m/m",
                                                 "Personal Spending m/m")),
    "Gross Domestic Product": (("GDP",), ("GDP q/q",)),
}
_GDP_BY_MONTH = {1: "Advance", 4: "Advance", 7: "Advance", 10: "Advance",
                 2: "Prelim", 5: "Prelim", 8: "Prelim", 11: "Prelim",
                 3: "Final", 6: "Final", 9: "Final", 12: "Final"}


def _fred(path, key, timeout=30, **params):
    if not key:
        raise SourceError("FRED: no API key (FRED_API_KEY in .env)")
    p = {"api_key": key, "file_type": "json", **params}
    r = _get(f"{FRED_BASE}/{path}", params=p, timeout=timeout)
    if r.status_code != 200:
        # never echo the URL: it carries the key
        raise SourceError(f"FRED {path}: HTTP {r.status_code} {r.text[:160]!r}")
    return r.json()


def fred_release_ids(key: str) -> Dict[str, int]:
    out, offset = {}, 0
    while True:
        js = _fred("releases", key, limit=1000, offset=offset)
        rel = js.get("releases", [])
        out.update({r["name"].strip(): int(r["id"]) for r in rel})
        offset += len(rel)
        if not rel or offset >= int(js.get("count", 0)):
            return out


def fred_release_dates(key: str, release_id: int) -> List[str]:
    """FRED's release calendar, past and SCHEDULED (see FRED_RELEASES on why the past part is
    not trusted for headline days)."""
    out, offset = [], 0
    while True:
        js = _fred("release/dates", key, release_id=release_id, limit=10000, offset=offset,
                   sort_order="asc", include_release_dates_with_no_data="true",
                   realtime_start="2000-01-01", realtime_end="9999-12-31")
        rd = js.get("release_dates", [])
        out += [r["date"] for r in rd]
        offset += len(rd)
        if not rd or offset >= int(js.get("count", 0)):
            return sorted(set(out))


def fred_first_prints(key: str, series: str, start: str = "1999-01-01") -> List[str]:
    """The day each observation of `series` was FIRST published. A date shared by more than 3
    observations is the start of FRED's vintage history, not a publication, and is dropped (two
    months published together after the 2025 shutdown share a date legitimately)."""
    # output_type=4 reads the WHOLE publication history, so the real-time window must span it
    # (the default, today only, is refused: "No vintage dates exist for the specified period")
    js = _fred("series/observations", key, series_id=series, output_type=4,
               observation_start=start, realtime_start="1776-07-04", realtime_end="9999-12-31")
    days = pd.Series([o["realtime_start"] for o in js.get("observations", [])])
    if days.empty:
        return []
    n = days.value_counts()
    return sorted(set(days[days.map(n) <= 3]))


def fred_vintage_dates(key: str, series: str) -> List[str]:
    js = _fred("series/vintagedates", key, series_id=series, limit=10000)
    return sorted(set(js.get("vintage_dates", [])))


def _one_per_month(future: List[str], past: List[str]) -> List[str]:
    """A monthly release's scheduled dates, one per month: where FRED schedules two, the one
    nearest the release's usual day of that month (from its first prints) is the headline."""
    if not future:
        return []
    hist = pd.to_datetime(pd.Series(past)) if past else pd.Series([], dtype="datetime64[ns]")
    usual = hist.groupby(hist.dt.month).apply(lambda d: d.dt.day.median()) if len(hist) else {}
    out = []
    f = pd.to_datetime(pd.Series(future))
    for _, g in f.groupby(f.dt.to_period("M")):
        if len(g) == 1:
            out.append(g.iloc[0])
            continue
        want = usual.get(g.iloc[0].month, 15) if len(usual) else 15
        out.append(g.iloc[(g.dt.day - want).abs().argmin()])
    return [d.strftime("%Y-%m-%d") for d in out]


def fred_headline_dates(key: str, name: str, release_id: int, series: tuple) -> Dict[str, str]:
    """{date: GDP stage or ""} -- the release's headline days: first prints for the past, the
    official schedule (one per month) after the last first print."""
    past = sorted(set(d for s in series for d in fred_first_prints(key, s)))
    sched = fred_release_dates(key, release_id)
    last = max(past) if past else "1999-12-31"
    future = [d for d in sched if d > last]
    out = {d: "" for d in past}
    if name == "Gross Domestic Product":
        out = {d: "Advance" for d in past}
        vint = fred_vintage_dates(key, series[0])
        for a in past:                      # the next two vintages are the 2nd and 3rd estimates
            nxt = [v for v in vint if a < v <= str((pd.Timestamp(a) + pd.Timedelta(days=80))
                                                  .date())][:2]
            for stage, v in zip(("Prelim", "Final"), nxt):
                out.setdefault(v, stage)
        out.update({d: _GDP_BY_MONTH[pd.Timestamp(d).month] for d in future})
    else:
        out.update({d: "" for d in _one_per_month(future, past)})
    return out


def fred_rows(key: str, *, pause: float = 0.6) -> pd.DataFrame:
    """Every FRED_RELEASES headline day as rows at its official New York time."""
    ids = fred_release_ids(key)
    low = {k.lower(): v for k, v in ids.items()}
    missing = [n for n in FRED_RELEASES if n.lower() not in low]
    if missing:
        raise SourceError(f"FRED has no release named {missing}; update FRED_RELEASES")
    rec = []
    for name, (series, titles) in FRED_RELEASES.items():
        for d, stage in fred_headline_dates(key, name, low[name.lower()], series).items():
            day = pd.Timestamp(d)
            for title in titles:
                if title == "GDP q/q":
                    title = f"{stage} GDP q/q"
                hhmm = official_time("USD", title, day)
                if hhmm is None:
                    continue
                t = pd.Timestamp(f"{d} {hhmm}").tz_localize(HOME_TZ["USD"]).tz_convert("UTC")
                rec.append(dict(t=t, time_q="official", currency="USD", title=title,
                                impact="high", source="fred"))
        time.sleep(pause)                      # FRED allows 120 requests a minute
    return _rows(pd.DataFrame(rec, columns=ROW_COLS))


# ══ THE FED'S OWN FOMC CALENDAR ══════════════════════════════════════════════════════════════
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"), start=1)}
_MON3 = {k[:3]: v for k, v in _MONTHS.items()}


def fetch_fomc(*, timeout: float = 30) -> str:
    r = _get(FOMC_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (quant-news)"})
    if r.status_code != 200 or "FOMC Meetings" not in r.text:
        raise SourceError(f"Fed FOMC calendar: HTTP {r.status_code}")
    return r.text


def parse_fomc(html: str) -> pd.DataFrame:
    """The Fed's meeting calendar (federalreserve.gov, about five years back and two ahead) ->
    rows at the official New York times: the rate decision and statement at 14:00 on the LAST
    day of the meeting, the press conference at 14:30, and the economic projections at 14:00 for
    the meetings marked *. A notation vote or a cancelled meeting is not a decision day."""
    import re
    rec = []
    sections = re.split(r"(\d{4}) FOMC Meetings", html)
    for k in range(1, len(sections) - 1, 2):
        year, body = int(sections[k]), sections[k + 1]
        months = re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>', body)
        dates = re.findall(r'fomc-meeting__date[^>]*>([^<]+)</div>', body)
        for mon, day in zip(months, dates):
            txt = " ".join(day.split())
            if re.search(r"notation|cancel", txt, re.I):
                continue
            m = re.match(r"(\d{1,2})(?:-(\d{1,2}))?(\*)?", txt)
            if not m:
                continue
            last = int(m.group(2) or m.group(1))
            mon_last = mon.split("/")[-1].strip().lower()
            month = _MONTHS.get(mon_last) or _MON3.get(mon_last[:3])
            if month is None:
                continue
            d = f"{year}-{month:02d}-{last:02d}"
            for title, hhmm in (("Federal Funds Rate", "14:00"), ("FOMC Statement", "14:00"),
                                ("FOMC Press Conference", "14:30")) + (
                    (("FOMC Economic Projections", "14:00"),) if m.group(3) else ()):
                t = pd.Timestamp(f"{d} {hhmm}").tz_localize("America/New_York").tz_convert("UTC")
                rec.append(dict(t=t, time_q="official", currency="USD", title=title,
                                impact="high", source="fed"))
    out = pd.DataFrame(rec, columns=ROW_COLS).drop_duplicates(["t", "title"])
    return _rows(out.sort_values("t"))
