"""
Bring the store up to date -- each source on its own schedule, each failure reported, never fatal.

    python -m quant.news update            feed (+ JBlanked, + FRED, when their keys are in .env)
    python -m quant.news import-archive    the 2007-2025 archive, once (68 MB download -> ~1 MB)
    python -m quant.news backfill-jb       JBlanked month by month, one request per 5 minutes

Keys come from the environment (.env, git-ignored): FRED_API_KEY, JBLANKED_API_KEY.
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


def keys() -> Dict[str, Optional[str]]:
    import quant.config  # noqa: F401  -- loads .env into the environment
    return {"fred": os.environ.get("FRED_API_KEY") or None,
            "jblanked": os.environ.get("JBLANKED_API_KEY") or None}


def _ago(meta, name, now) -> Optional[pd.Timedelta]:
    t = meta.get(f"{name}_ok")
    return None if t is None else now - pd.Timestamp(t)


def update_feed(store: NewsStore, now=None, *, force: bool = False) -> str:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    ago = _ago(store.meta(), "feed", now)
    if not force and ago is not None and ago < FEED_MIN_INTERVAL:
        return f"feed: fresh ({ago.seconds // 60} min old)"
    rows = sources.parse_ff_week(sources.fetch_ff_week())
    n = store.log_feed(rows, now)
    store.set_meta(feed_ok=now.isoformat(), feed_events=len(rows))
    return f"feed: {len(rows)} events this week, {n} new or changed"


def update_jb(store: NewsStore, key: str, now=None, *, days_back: int = 3,
              days_ahead: int = 7) -> str:
    """One request: JBlanked's calendar around now. Calibrates its zone against the feed first
    if that has not been done."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    last = store.meta().get("jb_request")
    if last and now - pd.Timestamp(last) < pd.Timedelta(seconds=sources.JB_MIN_INTERVAL_S):
        return "jblanked: waiting (one request per 5 minutes)"
    store.set_meta(jb_request=now.isoformat())
    items = sources.fetch_jb(key, kind="range", start=now - pd.Timedelta(days=days_back),
                             end=now + pd.Timedelta(days=days_ahead))
    zone = calibrate(store, items)
    rows = sources.parse_jb(items, zone)
    if len(rows):
        store.merge_into("jblanked", rows)
    store.set_meta(jb_ok=now.isoformat())
    return f"jblanked: {len(rows)} events (zone {zone or 'UNKNOWN -- dates only'})"


def calibrate(store: NewsStore, items) -> Optional[str]:
    """JBlanked's zone: the stored one, or found now against the feed log's exact times."""
    zone = store.meta().get("jb_zone")
    if zone:
        return zone
    log = store.read("feed_log")
    if log is None or not len(log):
        return None
    from .calendar import calibrate_jb_zone
    zone = calibrate_jb_zone(sources.jb_walls(items), log)
    if zone:
        store.set_meta(jb_zone=zone)
    return zone


def update_fred(store: NewsStore, key: str, now=None, *, force: bool = False) -> str:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    ago = _ago(store.meta(), "fred", now)
    if not force and ago is not None and ago < FRED_MIN_INTERVAL:
        return "fred: fresh"
    rows = sources.fred_rows(key)
    store.write("fred", rows)
    store.set_meta(fred_ok=now.isoformat())
    return f"fred: {len(rows)} official US release times"


def update_all(store: Optional[NewsStore] = None, now=None, *, rebuild: bool = True) -> list:
    """Every source that can be asked, each in its own try: one being down never stops another.
    Returns one line per source."""
    store = store or NewsStore()
    k = keys()
    out = []
    steps = [("feed", lambda: update_feed(store, now))]
    if k["fred"]:
        steps.append(("fred", lambda: update_fred(store, k["fred"], now)))
    else:
        out.append("fred: skipped (no FRED_API_KEY in .env)")
    if k["jblanked"]:
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
            rows = sources.parse_jb(items, calibrate(store, items))
            if len(rows):
                store.merge_into("jblanked", rows, prefer="old")
            total += len(rows)
            done.add(tag)
            store.set_meta(jb_backfilled=sorted(done))
            log(f"{tag} .. {b:%Y-%m-%d}: {len(rows)} events")
        a = b
    build_events(store)
    return total
