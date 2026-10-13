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
             WITH actuals, by date range. Its timestamps carry no zone; `jb_zone` holds the one
             found by calibrating against the feed (calendar.calibrate_jb_zone).
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
        raise SourceError(f"JBlanked {kind}: HTTP {r.status_code} {r.text[:160]!r}")
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


def parse_jb(items: Iterable[dict], zone: Optional[str]) -> pd.DataFrame:
    """JBlanked rows -> rows. `zone` is the time zone its naive timestamps are in (found by
    calibration); without one the rows carry their date only (time_q "date")."""
    rec = []
    for e in items:
        raw_t = _field(e, "Date", "date", "time")
        try:
            wall = pd.Timestamp(str(raw_t).replace(".", "-", 2))
        except (ValueError, TypeError):
            continue
        cur = str(_field(e, "Currency") or "").upper().strip()
        if zone:
            t = wall.tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward")
            t = t.tz_convert("UTC") if t is not pd.NaT else pd.NaT
            q = "exact"
        else:
            t = (pd.Timestamp(wall.date()).tz_localize(HOME_TZ.get(cur, "UTC"))
                 + pd.Timedelta(hours=12)).tz_convert("UTC")
            q = "date"
        if t is pd.NaT:
            continue
        rec.append(dict(t=t, time_q=q, currency=cur,
                        title=" ".join(str(_field(e, "Name", "title", "event") or "").split()),
                        impact=impact_level(_field(e, "Impact")),
                        actual=_num_text(_field(e, "Actual")),
                        forecast=_num_text(_field(e, "Forecast")),
                        previous=_num_text(_field(e, "Previous")), source="jblanked"))
    return _rows(pd.DataFrame(rec, columns=ROW_COLS))


def jb_walls(items: Iterable[dict]) -> pd.DataFrame:
    """(currency, title, naive wall time) of JBlanked rows -- what calibration compares."""
    rec = []
    for e in items:
        try:
            wall = pd.Timestamp(str(_field(e, "Date", "date", "time")).replace(".", "-", 2))
        except (ValueError, TypeError):
            continue
        rec.append(dict(currency=str(_field(e, "Currency") or "").upper().strip(),
                        title=" ".join(str(_field(e, "Name", "title", "event") or "").split()),
                        wall=wall))
    return pd.DataFrame(rec, columns=["currency", "title", "wall"])


# ══ FRED (official US release dates) ═════════════════════════════════════════════════════════
FRED_BASE = "https://api.stlouisfed.org/fred"
# FRED release (by NAME, resolved to an id at run time so a wrong id cannot creep in) -> the
# Forex Factory titles it publishes, all high impact. GDP is three titles by month: the first
# estimate comes the month after a quarter ends, the second a month later, the third after that.
FRED_RELEASES: Dict[str, tuple] = {
    "Consumer Price Index": ("CPI m/m", "Core CPI m/m", "CPI y/y", "Core CPI y/y"),
    "Producer Price Index": ("PPI m/m", "Core PPI m/m"),
    "Employment Situation": ("Non-Farm Employment Change", "Unemployment Rate",
                             "Average Hourly Earnings m/m"),
    "Advance Monthly Sales for Retail and Food Services": ("Retail Sales m/m",
                                                           "Core Retail Sales m/m"),
    "Personal Income and Outlays": ("Core PCE Price Index m/m", "Personal Spending m/m"),
    "Gross Domestic Product": ("GDP q/q",),
    "FOMC Press Release": ("Federal Funds Rate", "FOMC Statement"),
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


def fred_rows(key: str, *, pause: float = 0.6) -> pd.DataFrame:
    """Every FRED_RELEASES date as rows at its official New York time."""
    ids = fred_release_ids(key)
    low = {k.lower(): v for k, v in ids.items()}
    missing = [n for n in FRED_RELEASES if n.lower() not in low]
    if missing:
        raise SourceError(f"FRED has no release named {missing}; update FRED_RELEASES")
    rec = []
    for name, titles in FRED_RELEASES.items():
        for d in fred_release_dates(key, low[name.lower()]):
            day = pd.Timestamp(d)
            for title in titles:
                if title == "GDP q/q":
                    title = f"{_GDP_BY_MONTH[day.month]} GDP q/q"
                hhmm = official_time("USD", title, day)
                if hhmm is None:
                    continue
                t = pd.Timestamp(f"{d} {hhmm}").tz_localize(HOME_TZ["USD"]).tz_convert("UTC")
                rec.append(dict(t=t, time_q="official", currency="USD", title=title,
                                impact="high", source="fred"))
        time.sleep(pause)                      # FRED allows 120 requests a minute
    return _rows(pd.DataFrame(rec, columns=ROW_COLS))
