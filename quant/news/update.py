"""
Bring the store up to date -- each source on its own schedule, each failure reported, never fatal.

    python -m quant.news update            feed (+ JBlanked, + FRED, when their keys are in .env)
    python -m quant.news import-archive    the 2007-2025 archive, once (68 MB download -> ~1 MB)
    python -m quant.news backfill-jb       JBlanked month by month, one request per 5 minutes

    python -m quant.news official          US actuals from the agencies (history + validation)

Keys come from the environment (.env, git-ignored): FRED_API_KEY, JBLANKED_API_KEY, and optionally
BLS_API_KEY (lifts the Bureau of Labor Statistics' 25-requests-a-day limit to 500).
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import sources
from .calendar import NewsStore, build_events

FEED_MIN_INTERVAL = pd.Timedelta(minutes=30)       # the feed rate-limits eager callers
FRED_MIN_INTERVAL = pd.Timedelta(hours=12)         # release dates change rarely
FOMC_MIN_INTERVAL = pd.Timedelta(hours=12)         # the Fed's calendar, the same
NO_CREDITS_PAUSE = pd.Timedelta(hours=24)          # JBlanked said "requires credits": ask again
                                                   # tomorrow, not every 5 minutes
OFFICIAL_MIN_INTERVAL = pd.Timedelta(hours=6)      # the agencies' history: new releases only
OFFICIAL_FULL_INTERVAL = pd.Timedelta(hours=24)    # ...and every series once a day, whatever
                                                   # the calendar says (it may have missed one)
RETRY_AFTER_FAILURE = pd.Timedelta(minutes=30)     # a source that failed is not asked again
                                                   # every minute while it is down
FEED_RETRY = pd.Timedelta(minutes=5)               # ...except the feed, which reminders need
OFFICIAL_LIVE_GAP = pd.Timedelta(seconds=45)       # between live polls while a number is due
OFFICIAL_LIVE_WINDOW = pd.Timedelta(hours=3)       # FRED can post an hour after the release
BLS_DAILY_CAP = (20, 450)                          # without / with BLS_API_KEY (BLS: 25 / 500)


def keys() -> Dict[str, Optional[str]]:
    import quant.config  # noqa: F401  -- loads .env into the environment
    return {"fred": os.environ.get("FRED_API_KEY") or None,
            "jblanked": os.environ.get("JBLANKED_API_KEY") or None,
            "bls": os.environ.get("BLS_API_KEY") or None}


def _ago(meta, name, now) -> Optional[pd.Timedelta]:
    t = meta.get(f"{name}_ok")
    return None if t is None else now - pd.Timestamp(t)


def _backing_off(meta, name, now, wait=RETRY_AFTER_FAILURE) -> bool:
    t = meta.get(f"{name}_failed")
    return t is not None and now - pd.Timestamp(t) < wait


def _failing(store: NewsStore, name: str, now, fn):
    """fn(), remembering a failure so the next attempt waits RETRY_AFTER_FAILURE."""
    try:
        return fn()
    except Exception:
        store.set_meta(**{f"{name}_failed": now.isoformat()})
        raise


def update_feed(store: NewsStore, now=None, *, force: bool = False) -> str:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    meta = store.meta()
    ago = _ago(meta, "feed", now)
    if not force and ago is not None and ago < FEED_MIN_INTERVAL:
        return f"feed: fresh ({ago.seconds // 60} min old)"
    # the reminders need the feed, so it is retried sooner than the others -- but not every
    # minute: the alerts' candle checks wait behind it, up to its 20 s timeout each time
    if not force and _backing_off(meta, "feed", now, FEED_RETRY):
        return "feed: backing off after a failed fetch"
    rows = _failing(store, "feed", now, lambda: sources.parse_ff_week(sources.fetch_ff_week()))
    n = store.log_feed(rows, now)
    store.set_meta(feed_ok=now.isoformat(), feed_events=len(rows))
    return f"feed: {len(rows)} events this week, {n} new or changed"


def update_jb(store: NewsStore, key: str, now=None, *, kind: str = "week", days_back: int = 3,
              days_ahead: int = 7) -> str:
    """One request: JBlanked's calendar, stored RAW (its zone is decided when the calendar is
    built, from everything the store knows). `kind="week"` (this week, with the actuals out so
    far) is what a FREE key may ask for; "range" (days_back .. days_ahead) needs paid credits."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    last = store.meta().get("jb_request")
    if last and now - pd.Timestamp(last) < pd.Timedelta(seconds=sources.JB_MIN_INTERVAL_S):
        return "jblanked: waiting (one request per 5 minutes)"
    store.set_meta(jb_request=now.isoformat())
    try:
        items = sources.fetch_jb(key, kind=kind, start=now - pd.Timedelta(days=days_back),
                                 end=now + pd.Timedelta(days=days_ahead))
    except sources.SourceError as e:
        if "credits" in str(e).lower():
            store.set_meta(jb_no_credits=now.isoformat())
        raise
    store.set_meta(jb_no_credits=None)
    raw = sources.parse_jb(items)
    store.merge_jb(raw)
    store.set_meta(jb_ok=now.isoformat(), jb_last_count=len(raw))
    return f"jblanked: {len(raw)} events"


def jb_has_credits(store: NewsStore, now=None) -> bool:
    """False for a day after JBlanked last answered "requires credits"."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    t = store.meta().get("jb_no_credits")
    return t is None or now - pd.Timestamp(t) >= NO_CREDITS_PAUSE


def update_fomc(store: NewsStore, now=None, *, force: bool = False) -> str:
    """The Fed's own meeting calendar (no key). A page that parses to fewer than 20 rows has
    changed its layout, and is refused rather than allowed to empty the store."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    meta = store.meta()
    ago = _ago(meta, "fomc", now)
    if not force and ago is not None and ago < FOMC_MIN_INTERVAL:
        return "fomc: fresh"
    if not force and _backing_off(meta, "fomc", now):
        return "fomc: backing off after a failed fetch"
    rows = _failing(store, "fomc", now, lambda: sources.parse_fomc(sources.fetch_fomc()))
    if len(rows) < 20:
        store.set_meta(fomc_failed=now.isoformat())
        raise sources.SourceError(f"the Fed's calendar parsed to only {len(rows)} rows -- has "
                                  f"its page layout changed? ({sources.FOMC_URL})")
    store.write("fomc", rows)
    store.set_meta(fomc_ok=now.isoformat())
    n = int((rows["title"] == "Federal Funds Rate").sum())
    return f"fomc: {n} rate decisions {rows['t'].min():%Y} .. {rows['t'].max():%Y}"


# ── US actuals from the agencies (quant/news/official.py) ─────────────────────────────────────
def _ny_day(t) -> pd.Series:
    return pd.to_datetime(pd.Series(t), utc=True).dt.tz_convert("America/New_York") \
        .dt.strftime("%Y-%m-%d")


def _official_days(store: NewsStore, now):
    """The Fed's decision days and FRED's GDP release days {day: title}, up to today."""
    today = str(now.tz_convert("America/New_York").date())
    fomc = store.read("fomc")
    fdays = [] if fomc is None else sorted(set(_ny_day(fomc.loc[fomc["title"] ==
                                                                "Federal Funds Rate", "t"])))
    fred = store.read("fred")
    gdp = {}
    if fred is not None and len(fred):
        g = fred[fred["title"].str.endswith("GDP q/q")]
        gdp = {d: ti for d, ti in zip(_ny_day(g["t"]), g["title"]) if d <= today}
    return [d for d in fdays if d <= today], gdp


def _merge_official(store: NewsStore, rows: pd.DataFrame, via: str) -> int:
    """Upsert by (title, New York day): the newer row wins. `via` = "history" | "live" records how
    it was found, because the history resumes from the last HISTORY row only -- a number caught
    live after weeks offline must not make the history skip the releases in between."""
    if rows is None or not len(rows):
        return 0
    rows = rows.assign(via=via)
    old = store.read("official")
    both = pd.concat([x for x in (rows, old) if x is not None and len(x)], ignore_index=True)
    k = both["title"] + "|" + _ny_day(both["t"]).to_numpy()
    out = both[~k.duplicated(keep="first")].sort_values("t").reset_index(drop=True)
    store.write("official", out)
    return len(rows)


def _due_specs(store: NewsStore, after: dict, now) -> list:
    """The series with a release in the calendar since the last day stored for them -- the rest
    have nothing new, so asking FRED about them would only cost time. A series never stored is
    always due."""
    from . import official
    ev = store.read("events")
    if ev is None or not len(ev):
        return list(official.SPECS)
    us = ev[(ev["currency"] == "USD") & (pd.to_datetime(ev["t"], utc=True) <= now)]
    days = us.assign(day=_ny_day(us["t"]).to_numpy()).groupby("title")["day"].max().to_dict()
    by = official.specs_by_title()
    out = []
    for spec in official.SPECS:
        done = after.get(spec.fred)
        latest = max((days.get(ti, "") for ti, s in by.items() if s is spec), default="")
        if done is None or latest > done:
            out.append(spec)
    return out


def update_official(store: NewsStore, key: str, now=None, *, start="2022-01-01",
                    force: bool = False, refetch: bool = False, log=None) -> str:
    """The agencies' history since `start`, incremental: only releases newer than the last one
    stored per series are fetched, and between the daily full sweeps only the series the calendar
    says have released since (~25 requests the first time and once a day, ~0-3 otherwise)."""
    from . import official
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    meta = store.meta()
    ago = _ago(meta, "official", now)
    if not force and ago is not None and ago < OFFICIAL_MIN_INTERVAL:
        return "official: fresh"
    if not force and _backing_off(meta, "official", now):
        return "official: backing off after a failed fetch"
    full_ago = _ago(meta, "official_full", now)
    full = force or full_ago is None or full_ago >= OFFICIAL_FULL_INTERVAL
    old = store.read("official")
    after = {}
    # refetch: ignore what is stored and read everything since `start` again -- the way to reach
    # further back, since the history otherwise resumes from the newest stored release
    if old is not None and len(old) and not refetch:
        h = old[old["via"].fillna("history") != "live"] if "via" in old.columns else old
        last = h.assign(day=_ny_day(h["t"]).to_numpy()).groupby("series")["day"].max()
        after = last.to_dict()
    specs = list(official.SPECS) if full else _due_specs(store, after, now)
    n = 0
    if specs:
        fdays, gdp = _official_days(store, now)
        rows = _failing(store, "official", now, lambda: official.history(
            key, start, specs=specs, fomc_days=fdays, gdp_days=gdp, after=after, log=log))
        n = _merge_official(store, rows, "history")
    store.set_meta(official_ok=now.isoformat(),
                   **({"official_full_ok": now.isoformat()} if full else {}))
    what = "every series" if full else f"{len(specs)} series due"
    return f"official: {n} new US releases with their actual ({what})"


def _baseline(store: NewsStore, key: Optional[str], title: str, t) -> Optional[pd.Timestamp]:
    """The last period already published before the release at `t`: from the stored history,
    else from FRED's first prints. None = unknown, and then a BLS number is not accepted."""
    from . import official
    old = store.read("official")
    if old is not None and len(old) and "period" in old.columns:
        prev = old[(old["title"] == title) & (pd.to_datetime(old["t"], utc=True) < t)]
        if len(prev):
            return pd.Timestamp(prev.sort_values("t")["period"].iloc[-1])
    spec = official.specs_by_title().get(title)
    if key and spec is not None:
        day = str(pd.Timestamp(t).tz_convert("America/New_York").date())
        fp = official.first_prints(key, spec.fred, pd.Timestamp(t) - pd.DateOffset(years=2))
        fp = fp[fp["day"] < day]
        if len(fp):
            return pd.Timestamp(fp["period"].max())
    return None


def poll_official(store: NewsStore, key: Optional[str], pending: pd.DataFrame, now=None) -> str:
    """Ask the agencies for the numbers of `pending` US releases (t, title) that are out but have
    none yet. At most once every OFFICIAL_LIVE_GAP; BLS requests counted per day and capped."""
    from . import official
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    if pending is None or not len(pending):
        return "official live: nothing waiting"
    titles = official.specs_by_title()
    pend = pending[(pending["currency"] == "USD") & pending["title"].isin(titles)
                   & (pd.to_datetime(pending["t"], utc=True) <= now)
                   & (pd.to_datetime(pending["t"], utc=True) > now - OFFICIAL_LIVE_WINDOW)]
    if not len(pend):
        return "official live: nothing waiting"
    m = store.meta()
    last = m.get("official_live")
    if last and now - pd.Timestamp(last) < OFFICIAL_LIVE_GAP:
        return "official live: waiting between polls"
    k = keys()
    day = str(now.date())
    used = int((m.get("bls_count") or {}).get(day, 0))
    cap = BLS_DAILY_CAP[1 if k["bls"] else 0]
    need_bls = any(titles[ti].bls for ti in pend["title"])
    base = {ti: _baseline(store, key, ti, pd.Timestamp(t)) for ti, t in zip(pend["title"], pend["t"])}
    rows, notes = official.live(key, pend[["t", "title"]], base, bls_key=k["bls"],
                                bls_ok=used < cap)
    store.set_meta(official_live=now.isoformat(),
                   bls_count={day: used + (1 if need_bls and used < cap else 0)})
    n = _merge_official(store, rows, "live")
    return (f"official live: {n} of {len(pend)} numbers in"
            + (f" ({'; '.join(notes)})" if notes else ""))


def update_fred(store: NewsStore, key: str, now=None, *, force: bool = False) -> str:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    meta = store.meta()
    ago = _ago(meta, "fred", now)
    if not force and ago is not None and ago < FRED_MIN_INTERVAL:
        return "fred: fresh"
    if not force and _backing_off(meta, "fred", now):
        return "fred: backing off after a failed fetch"
    rows = _failing(store, "fred", now, lambda: sources.fred_rows(key))
    store.write("fred", rows)
    store.set_meta(fred_ok=now.isoformat())
    return f"fred: {len(rows)} official US release times"


def update_all(store: Optional[NewsStore] = None, now=None, *, rebuild: bool = True) -> list:
    """Every source that can be asked, each in its own try: one being down never stops another.
    Returns one line per source."""
    store = store or NewsStore()
    k = keys()
    out = []
    steps = [("feed", lambda: update_feed(store, now)), ("fomc", lambda: update_fomc(store, now))]
    if k["fred"]:
        steps.append(("fred", lambda: update_fred(store, k["fred"], now)))
    else:
        out.append("fred: skipped (no FRED_API_KEY in .env)")
    if k["fred"]:
        steps.append(("official", lambda: update_official(store, k["fred"], now)))
        def _live():
            ev = store.read("events")
            if ev is None or not len(ev):
                return "official live: nothing waiting"
            t = pd.to_datetime(ev["t"], utc=True)
            nw = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
            return poll_official(store, k["fred"], ev[ev["actual"].isna() & (t <= nw)
                                                     & (t > nw - OFFICIAL_LIVE_WINDOW)], now)
        steps.append(("official live", _live))
    if k["jblanked"] and not jb_has_credits(store, now):
        out.append("jblanked: skipped (the key has no credits -- jblanked.com/api/billing)")
    elif k["jblanked"]:
        steps.append(("jblanked", lambda: update_jb(store, k["jblanked"], now)))
    else:
        out.append("jblanked: skipped (no JBLANKED_API_KEY in .env)")
    for name, fn in steps:
        try:
            out.append(fn())
        except Exception as e:                       # report, keep going
            out.append(f"{name}: FAILED {type(e).__name__}: {e}")
            store.set_meta(**{f"{name}_error": f"{pd.Timestamp.now(tz='UTC').isoformat()} {e}"})
    if rebuild:
        ev = build_events(store)
        out.append(f"calendar: {len(ev):,} events")
    return out


def import_archive(store: Optional[NewsStore] = None, csv_path=None, *,
                   keep_download: bool = False) -> str:
    """Download (unless `csv_path` is given), repair and store the archive. The CSV is deleted
    afterwards unless `keep_download`: the repaired copy is all that is ever read."""
    store = store or NewsStore()
    tmp = None
    if csv_path is None:
        tmp = Path(tempfile.mkdtemp(prefix="ffarchive_")) / "ff_archive.csv"
        sources.download_archive(str(tmp))
        csv_path = tmp
    t0 = time.time()
    raw = pd.read_csv(csv_path)
    rows = sources.parse_archive(raw)
    store.write("archive", rows)
    store.write("directions", sources.archive_directions(raw))
    q = rows["time_q"].value_counts().to_dict()
    if tmp is not None and not keep_download:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()
    store.set_meta(archive_rows=len(rows), archive_imported=pd.Timestamp.now(tz="UTC").isoformat())
    build_events(store)
    return (f"archive: {len(rows):,} events {rows['t'].min():%Y-%m-%d} .. "
            f"{rows['t'].max():%Y-%m-%d} in {time.time() - t0:.0f}s; times {q}; "
            f"{store.path('archive').stat().st_size / 1e6:.1f} MB")


def backfill_jb(store: Optional[NewsStore] = None, start="2025-04-01", end=None, *,
                step_days: int = 30, log=print) -> int:
    """JBlanked month by month from `start`, one request per 5 minutes (the free tier).
    Resumable: months already stored are skipped. Returns rows stored."""
    store = store or NewsStore()
    key = keys()["jblanked"]
    # NOTE: the date-range endpoint needs paid JBlanked credits (a free key gets HTTP 401
    # "This endpoint requires credits"); the free key covers only this week (update_jb)
    end = pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end, tz="UTC")
    a = pd.Timestamp(start, tz="UTC")
    done = set(store.meta().get("jb_backfilled", []))
    total = 0
    while a < end:
        b = min(a + pd.Timedelta(days=step_days), end)
        tag = f"{a:%Y-%m-%d}"
        if tag not in done:
            wait = sources.JB_MIN_INTERVAL_S
            last = store.meta().get("jb_request")
            if last:
                left = wait - (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last)).total_seconds()
                if left > 0:
                    log(f"waiting {left:.0f}s for the JBlanked rate limit")
                    time.sleep(left + 1)
            store.set_meta(jb_request=pd.Timestamp.now(tz="UTC").isoformat())
            items = sources.fetch_jb(key, kind="range", start=a, end=b)
            rows = sources.parse_jb(items)
            store.merge_jb(rows, prefer="old")
            total += len(rows)
            done.add(tag)
            store.set_meta(jb_backfilled=sorted(done))
            log(f"{tag} .. {b:%Y-%m-%d}: {len(rows)} events")
        a = b
    build_events(store)
    return total
