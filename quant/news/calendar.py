"""
ONE calendar from four sources: store them, merge them, read them back.

    data/news/                    (git-ignored, like every other cache here)
      archive.parquet             the scraped archive, repaired (sources.parse_archive)  ~1 MB
      directions.parquet          Forex Factory's "usual effect" per event, from the archive
      feed_log.parquet            EVERY distinct observation of the weekly feed: a row is added
                                  only when an event's time, impact, forecast or previous changes,
                                  so forecast revisions and reschedules stay on record. Each row
                                  also carries `last_seen`, and meta.json the latest fetch of each
                                  feed week: an event missing from its week's latest fetch was
                                  MOVED or CANCELLED, and is left out of the calendar
      jblanked.parquet            JBlanked RAW: its naive wall times as sent. The zone is decided
                                  at every build from all the evidence (calibrate_jb_zone) and
                                  recorded in meta.json as jb_zone / jb_zone_evidence
      fred.parquet                official US release dates
      events.parquet              the merged calendar (rebuilt when any input is newer)
      meta.json                   when each source was last asked, and JBlanked's zone

THE MERGE. Rows describe the same release when they share (currency, title, day in the
currency's home zone). Each field is then taken from the most trusted source that HAS it:
  time      the best time_q (exact > official > typical > date), then feed > JBlanked > archive
  impact    feed > JBlanked > archive > FRED
  forecast  the feed's LAST observation made BEFORE the release, else the archive, else
            JBlanked -- a forecast revised after the number came out is not what the market
            expected, and JBlanked's bare numbers carry no unit
  actual    JBlanked > archive (the feed has none); a JBlanked actual is put in the printed
            forecast's units ("254" or "254000" beside "140K" both become "254K")
  previous  feed > archive > JBlanked
A field no source has stays empty; nothing is filled in by guessing.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import catalog
from .catalog import HOME_TZ, RANK, criticality, parse_coverage
from .sources import JB_RAW_COLS, NY7, ROW_COLS, TIME_Q, jb_rows, localize_wall
from .values import parse_values, sentiment, surprise_z

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "news"

EVENT_COLS = ["t", "time_q", "currency", "title", "impact", "level", "category",
              "actual", "forecast", "previous", "actual_v", "forecast_v", "previous_v",
              "surprise", "surprise_z", "direction", "sentiment", "source", "key"]
_SRC_RANK = {"ff_feed": 0, "jblanked": 1, "ff_archive": 2, "fred": 3, "fed": 4}
_TQ_RANK = {q: i for i, q in enumerate(TIME_Q)}


class NewsStore:
    """The files above, written atomically (temp file + rename), read with a small cache."""

    FILES = ("archive", "directions", "feed_log", "jblanked", "fred", "fomc", "events")

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self._cache: Dict[str, tuple] = {}

    def path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def read(self, name: str) -> Optional[pd.DataFrame]:
        p = self.path(name)
        if not p.is_file():
            return None
        mt = p.stat().st_mtime_ns
        hit = self._cache.get(name)
        if hit and hit[0] == mt:
            return hit[1]
        df = pd.read_parquet(p)
        self._cache[name] = (mt, df)
        return df

    def write(self, name: str, df: pd.DataFrame):
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(name)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
        self._cache.pop(name, None)

    # ── meta ───────────────────────────────────────────────────────────────────────────────
    def meta(self) -> dict:
        p = self.root / "meta.json"
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def set_meta(self, **kv):
        self.root.mkdir(parents=True, exist_ok=True)
        m = self.meta()
        m.update(kv)
        p = self.root / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, p)

    # ── the feed log ───────────────────────────────────────────────────────────────────────
    def log_feed(self, rows: pd.DataFrame, fetched_at, *, min_events: int = 20) -> int:
        """Record one fetch of the weekly feed: append the rows that differ from the latest
        observation of the same event, and mark every event present as seen now. Returns how
        many rows were new. A fetch with fewer than `min_events` events is logged but NOT taken
        as the week's word on what exists -- a truncated answer must not cancel the rest."""
        now = _utc(fetched_at)                    # a naive time is UTC, as everywhere here
        rows = rows.copy()
        rows["fetched_at"] = now
        rows["key"] = event_keys(rows.drop(columns=["fetched_at"]))
        rows["last_seen"] = now
        old = self.read("feed_log")
        cmp = ["t", "impact", "forecast", "previous", "actual"]
        if old is not None and len(old):
            old = old.copy()
            last = old.sort_values("fetched_at").groupby("key").tail(1).set_index("key")
            mine = rows.set_index("key")[cmp]
            prev = last.reindex(mine.index)[cmp]
            same = (mine.astype(str).values == prev.astype(str).values).all(axis=1)
            new = rows[~same]
            old.loc[old["key"].isin(rows["key"]), "last_seen"] = now
            out = pd.concat([old, new], ignore_index=True)
        else:
            new, out = rows, rows
        self.write("feed_log", out)
        if len(rows) >= min_events:
            weeks = dict(self.meta().get("feed_weeks", {}))
            for w in feed_week(rows["t"]).unique():
                weeks[str(w)] = now.isoformat()
            self.set_meta(feed_weeks=weeks)
        return len(new)

    def merge_jb(self, raw: pd.DataFrame, *, prefer: str = "new") -> int:
        """Upsert raw JBlanked rows by (currency, title, wall time). Returns the stored count."""
        old = self.read("jblanked")
        if old is not None and len(old) and "wall" not in old.columns:
            old = None                                # a pre-raw file: superseded, not merged
        both = [raw, old] if prefer == "new" else [old, raw]
        out = pd.concat([b for b in both if b is not None and len(b)], ignore_index=True)
        if not len(out):
            return 0
        k = out["currency"] + "|" + out["title"].str.lower() + "|" + out["wall"].astype(str)
        out = out[~k.duplicated(keep="first")].sort_values("wall").reset_index(drop=True)
        self.write("jblanked", out[JB_RAW_COLS])
        return len(out)

    def merge_into(self, name: str, rows: pd.DataFrame, *, prefer: str = "new"):
        """Upsert `rows` into a store file by event key; `prefer` new or old on a clash."""
        rows = rows.copy()
        rows["key"] = event_keys(rows)
        old = self.read(name)
        if old is not None and len(old):
            both = [rows, old] if prefer == "new" else [old, rows]
            out = pd.concat(both, ignore_index=True).drop_duplicates("key", keep="first")
        else:
            out = rows
        self.write(name, out.sort_values("t").reset_index(drop=True))


def _home_day(df: pd.DataFrame) -> pd.Series:
    """Each row's date in its currency's home zone, as YYYY-MM-DD."""
    out = pd.Series("", index=df.index, dtype="object")
    zone = df["currency"].map(HOME_TZ).fillna("UTC")
    t = pd.to_datetime(df["t"], utc=True)
    for z, idx in zone.groupby(zone).groups.items():
        local = t.loc[idx].dt.tz_convert(z).dt.tz_localize(None).to_numpy()
        out.loc[idx] = local.astype("datetime64[D]").astype(str)      # strftime is 10x slower
    return out


def event_keys(df: pd.DataFrame) -> pd.Series:
    """currency|title|home-day: the identity two sources must share to be the same release.

    The same title twice on one day (a speaker in the morning and again at night) is numbered
    in time order WITHIN one source and one fetch -- "...|2026-10-14#1" for the second -- so
    each source numbers its own rows the same way and they still meet."""
    if not len(df):
        return pd.Series([], dtype="object")
    title = df["title"].astype(str).str.lower().str.split().str.join(" ")
    base = df["currency"].astype(str) + "|" + title + "|" + _home_day(df)
    by = [base] + [df[c].astype(str) for c in ("source", "fetched_at") if c in df.columns]
    order = np.argsort(pd.to_datetime(df["t"], utc=True).to_numpy(), kind="stable")
    srt = df.index[order]
    n = pd.Series(0, index=df.index)
    n.loc[srt] = pd.Series(np.arange(len(df)), index=srt).groupby(
        [b.loc[srt].fillna("").to_numpy() for b in by], dropna=False).cumcount().to_numpy()
    n = n.astype(int)
    return base.where(n == 0, base + "#" + n.astype(str))


def feed_week(t) -> pd.Series:
    """The Forex Factory week a time belongs to: its Sunday, in New York."""
    et = pd.to_datetime(pd.Series(t), utc=True).dt.tz_convert("America/New_York")
    day = et.dt.tz_localize(None).dt.normalize()
    return (day - pd.to_timedelta((et.dt.dayofweek + 1) % 7, unit="D")).dt.strftime("%Y-%m-%d")


# ══ THE MERGE ════════════════════════════════════════════════════════════════════════════════
def merge_rows(rows: pd.DataFrame, directions: Optional[pd.DataFrame] = None,
               feed_weeks: Optional[dict] = None) -> pd.DataFrame:
    """All sources' rows (feed rows carrying `fetched_at` / `last_seen`) -> the calendar.

    `feed_weeks` {week: latest fetch}: a feed event not seen in its week's latest fetch was moved
    or cancelled, so its rows are dropped before anything is merged."""
    if rows is None or not len(rows):
        return pd.DataFrame(columns=EVENT_COLS)
    r = rows.copy().reset_index(drop=True)
    r["t"] = pd.to_datetime(r["t"], utc=True)
    if feed_weeks and "last_seen" in r:
        feed = r["source"] == "ff_feed"
        latest = pd.to_datetime(feed_week(r["t"]).map(feed_weeks), utc=True)
        gone = feed & latest.notna() & (pd.to_datetime(r["last_seen"], utc=True) < latest)
        r = r[~gone].reset_index(drop=True)
    # numbered within each source and, for the feed, within each fetch (see event_keys)
    tag = r["source"].astype(str)
    if "fetched_at" in r:
        tag = tag + r["fetched_at"].astype(str)
    r["key"] = event_keys(pd.DataFrame({"t": r["t"], "currency": r["currency"],
                                        "title": r["title"], "source": tag}))
    r["src"] = r["source"].map(_SRC_RANK).fillna(9)
    r["tq"] = r["time_q"].map(_TQ_RANK).fillna(9)

    # a feed row observed AFTER its own release cannot speak for the forecast -- it may already
    # be the next period's, or revised -- so only pre-release observations feed `forecast`
    if "fetched_at" in r:
        fa = pd.to_datetime(r["fetched_at"], utc=True)
        late = (r["source"] == "ff_feed") & fa.notna() & (fa > r["t"])
        r.loc[late, "forecast"] = None
        # newest observation first among feed rows
        r["obs"] = fa.fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    else:
        r["obs"] = pd.Timestamp("1970-01-01", tz="UTC")

    # time: one row's (t, time_q) together, never mixed
    tsel = r.sort_values(["key", "tq", "src", "obs"], ascending=[True, True, True, False])
    first = tsel.drop_duplicates("key", keep="first").set_index("key")
    ev = first[["t", "time_q", "currency", "title"]].copy()

    def pick(col, order):
        rank = r["source"].map({s: i for i, s in enumerate(order)}).fillna(99)
        sub = r.assign(_rk=rank).sort_values(["key", "_rk", "obs"],
                                             ascending=[True, True, False])
        sub = sub[sub[col].notna() & sub["_rk"].lt(99)]
        return sub.drop_duplicates("key", keep="first").set_index("key")[col]

    ev["impact"] = pick("impact", ["ff_feed", "jblanked", "ff_archive", "fred", "fed"])
    ev["actual"] = pick("actual", ["jblanked", "ff_archive"])
    # Forex Factory's PRINTED values first ("245K", "0.3%"); JBlanked's bare numbers only where
    # nothing printed exists -- they carry no unit (see _rescale_jb)
    ev["forecast"] = pick("forecast", ["ff_feed", "ff_archive", "jblanked"])
    ev["previous"] = pick("previous", ["ff_feed", "ff_archive", "jblanked"])
    _jb = r[r["source"] == "jblanked"].drop_duplicates("key").set_index("key")
    if len(_jb):
        ev = _rescale_jb(ev, _jb)
    # which sources spoke, as a bit mask summed over DISTINCT (key, source) -- a Python join per
    # key cost 6 s on the archive
    bit = r["source"].map({s: 1 << i for s, i in _SRC_RANK.items()}).fillna(0).astype(int)
    mask = r.assign(_b=bit).drop_duplicates(["key", "source"]).groupby("key")["_b"].sum()
    names = sorted(_SRC_RANK, key=_SRC_RANK.get)
    ev["source"] = mask.map(lambda m: "+".join(n for i, n in enumerate(names) if m >> i & 1))
    ev = ev.reset_index()
    return finish(ev, directions)


def _decimals(s) -> int:
    m = re.search(r"\.(\d+)", str(s or ""))
    return len(m.group(1)) if m else 0


def _rescale_jb(ev: pd.DataFrame, jb: pd.DataFrame) -> pd.DataFrame:
    """JBlanked sends bare numbers; Forex Factory prints units ("245K", "0.3%"). An actual
    taken from JBlanked is put in the units of the forecast it will be compared with: the scale
    is the ratio of the PRINTED forecast (or previous) to JBlanked's own number for it, which must
    be a power of 1000 -- so it is right whether JBlanked sends 254 or 254000 for "254K". With no
    printed number to compare against, its actual is left as sent."""
    from .values import _MULT
    ev = ev.copy()                                     # indexed by key, and returned so
    keys = ev.index.intersection(jb.index)
    for k in keys:
        a_jb = parse_values([jb.at[k, "actual"]])[0]
        if not np.isfinite(a_jb) or ev.at[k, "actual"] != jb.at[k, "actual"]:
            continue                                   # the actual came from somewhere else
        scale, ref_txt = np.nan, None
        for col in ("forecast", "previous"):
            shown, own = ev.at[k, col], jb.at[k, col]
            if shown is None or own is None or (isinstance(shown, float) and np.isnan(shown)):
                continue
            s_v, o_v = parse_values([shown])[0], parse_values([own])[0]
            if np.isfinite(s_v) and np.isfinite(o_v) and o_v != 0 and s_v != 0:
                lg = np.log(abs(s_v / o_v)) / np.log(1000.0)
                if abs(lg - round(lg)) < 0.02 and s_v / o_v > 0:
                    scale, ref_txt = 1000.0 ** round(lg), str(shown)
                    break
        if not np.isfinite(scale):
            continue
        m = re.search(r"(%+|[KkMBT])\s*$", ref_txt)
        unit = (m.group(1)[:1] if m else "")
        div = _MULT.get(unit.upper(), 1.0) if unit != "%" else 1.0
        v = a_jb * scale / div
        dec = max(_decimals(ref_txt), _decimals(ev.at[k, "previous"]))
        txt = f"{v:.{dec}f}"
        if abs(float(txt) - v) > 1e-9:               # the printed precision would lose digits
            txt = f"{v:.3f}".rstrip("0").rstrip(".")
        ev.at[k, "actual"] = txt + unit
    return ev


def finish(ev: pd.DataFrame, directions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Derived columns: level, category, numbers, surprise, its z-score, direction, sentiment."""
    if not len(ev):                             # an empty calendar is a calendar, not an error
        return pd.DataFrame(columns=EVENT_COLS)
    ev = ev.copy()
    ev["impact"] = ev["impact"].fillna("none")
    ev["level"] = [criticality(c, ti, im) for c, ti, im in
                   zip(ev["currency"], ev["title"], ev["impact"])]
    # per DISTINCT title / value, then mapped: ~1,500 titles and a few thousand values among
    # 80,000 rows, and a regex per row was most of the build time
    titles = pd.Index(ev["title"].unique())
    ev["category"] = ev["title"].map(dict(zip(titles, map(catalog.category, titles))))
    for c in ("actual", "forecast", "previous"):
        u = pd.unique(ev[c].dropna())
        ev[f"{c}_v"] = ev[c].map(dict(zip(u, parse_values(u)))).astype(float)
    ev["surprise"] = ev["actual_v"] - ev["forecast_v"]
    ev = ev.sort_values("t", kind="stable").reset_index(drop=True)
    ev["surprise_z"] = surprise_z(ev)
    d = ev["title"].map(dict(zip(titles, map(catalog.direction, titles)))).astype(int)
    if directions is not None and len(directions):
        ffd = directions.set_index(["currency", "title"])["direction"]
        known = pd.MultiIndex.from_arrays([ev["currency"], ev["title"]])
        got = ffd.reindex(known)
        d = pd.Series(np.where(got.notna(), got, d), index=ev.index).astype(int)
    ev["direction"] = d
    ev["sentiment"] = sentiment(ev["surprise"], ev["direction"])
    if "key" not in ev:
        ev["key"] = event_keys(ev)
    return ev[EVENT_COLS]


def build_events(store: Optional[NewsStore] = None) -> pd.DataFrame:
    """Merge every source the store holds and save the result as events.parquet. JBlanked's
    zone is decided here, every time, from everything the store knows (calibrate_jb_zone)."""
    store = store or NewsStore()
    parts = []
    for name in ("archive", "fred", "fomc"):
        df = store.read(name)
        if df is not None and len(df):
            parts.append(df[[c for c in ROW_COLS if c in df.columns]])
    log = store.read("feed_log")
    if log is not None and len(log):
        parts.append(log[ROW_COLS + ["fetched_at", "last_seen"]])
    raw = store.read("jblanked")
    if raw is not None and len(raw) and "wall" in raw.columns:
        ref = pd.concat([p[p["time_q"] == "exact"][["t", "currency", "title", "time_q"]]
                         for p in parts if len(p)], ignore_index=True) if parts else None
        zone, why = calibrate_jb_zone(raw, ref, explain=True)
        store.set_meta(jb_zone=zone, jb_zone_evidence=why)
        parts.append(jb_rows(raw, zone))
    rows = pd.concat(parts, ignore_index=True) if parts else None
    ev = merge_rows(rows, store.read("directions"), store.meta().get("feed_weeks"))
    store.write("events", ev)
    return ev


def _stale(store: NewsStore) -> bool:
    out = store.path("events")
    if not out.is_file():
        return True
    built = out.stat().st_mtime_ns
    return any(store.path(n).is_file() and store.path(n).stat().st_mtime_ns > built
               for n in ("archive", "directions", "feed_log", "jblanked", "fred", "fomc"))


def load_events(start=None, end=None, *, coverage=None, store: Optional[NewsStore] = None,
                timed_only: bool = False) -> pd.DataFrame:
    """The merged calendar between `start` and `end` (UTC), optionally narrowed by `coverage`
    ({"USD": "medium", "EUR": "high"}, see catalog.parse_coverage). Rebuilt first if a source
    file is newer than it. `timed_only` drops rows whose time is not known (date / allday)."""
    store = store or NewsStore()
    if _stale(store):
        if not any(store.exists(n) for n in ("archive", "feed_log", "jblanked", "fred", "fomc")):
            return pd.DataFrame(columns=EVENT_COLS)
        ev = build_events(store)
    else:
        ev = store.read("events")
    if start is not None:
        ev = ev[ev["t"] >= _utc(start)]
    if end is not None:
        ev = ev[ev["t"] < _utc(end)]
    if coverage is not None:
        ev = select(ev, coverage)
    if timed_only:
        ev = ev[ev["time_q"].isin(("exact", "official", "typical"))]
    return ev.reset_index(drop=True)


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def select(events: pd.DataFrame, coverage) -> pd.DataFrame:
    """Rows whose currency is covered at or above its level; "*" covers every currency not
    named, and a currency neither named nor covered by "*" is left out."""
    cov = parse_coverage(coverage)
    floor = {c: RANK[lv] for c, lv in cov.items()}
    need = events["currency"].map(lambda c: floor.get(c, floor.get("*", 99)))
    return events[events["level"].map(RANK) >= need].reset_index(drop=True)


# ══ JBLANKED'S ZONE ══════════════════════════════════════════════════════════════════════════
JB_ZONE_CANDIDATES = (NY7, "Europe/Athens", "Etc/GMT-3", "Etc/GMT-2", "Europe/Moscow", "UTC",
                      "America/New_York", "Asia/Karachi")
# a year of walls at a release hour, every day: two candidates that convert ANY of them
# differently are different zones, however well they agree on the data at hand
_PROBE = pd.Series(pd.date_range("2025-01-01 15:30", "2025-12-31 15:30", freq="D"))


def _official_reference(jb: pd.DataFrame) -> pd.DataFrame:
    """The official New York time of every US release JBlanked lists that has one -- ground
    truth in EVERY season, from JBlanked's own dates (its wall date is the New York date for
    every release between 7 a.m. and 5 p.m. there)."""
    from .catalog import official_time
    us = jb[jb["currency"] == "USD"]
    rec = []
    for title, wall in zip(us["title"], us["wall"]):
        d = pd.Timestamp(wall).date()
        hhmm = official_time("USD", title, d)
        if hhmm:
            rec.append(dict(t=pd.Timestamp(f"{d} {hhmm}").tz_localize("America/New_York")
                            .tz_convert("UTC"), currency="USD", title=title, time_q="exact"))
    return pd.DataFrame(rec, columns=["t", "currency", "title", "time_q"])


def calibrate_jb_zone(jb: pd.DataFrame, reference: Optional[pd.DataFrame] = None, *,
                      min_matches: int = 5, min_agree: float = 0.9, explain: bool = False):
    """Which zone JBlanked's naive `wall` times are in.

    Every candidate converts the walls; a conversion AGREES when it lands on the exact time of the
    same (currency, title) in the reference -- the weekly feed, the archive's timed rows, and the
    official US release times on JBlanked's own dates (any season). The answer must
      - match at least `min_matches` releases and agree on `min_agree` of them, and
      - be UNIQUE: if another candidate scores within 2 points and would convert some day of the
        year differently (Athens and broker time agree except in the weeks when US and EU clocks
        change on different Sundays), the data cannot tell them apart yet, and the answer is None.
    None leaves JBlanked's rows dated but untimed, which is safe: the feed or the official time
    decides the time, and JBlanked lends only the number. `explain=True` returns (zone, why)."""
    def done(z, why):
        return (z, why) if explain else z
    if jb is None or not len(jb):
        return done(None, "no JBlanked rows")
    refs = [_official_reference(jb)]
    if reference is not None and len(reference):
        refs.append(reference[reference["time_q"] == "exact"][["t", "currency", "title"]])
    refs = [x for x in refs if len(x)]
    if not refs:
        return done(None, "nothing to compare with")
    ref = pd.concat(refs, ignore_index=True)
    ref = ref.assign(_k=ref["currency"] + "|" + ref["title"].str.lower(),
                     t=pd.to_datetime(ref["t"], utc=True))
    ref = ref.drop_duplicates(["_k", "t"])
    jbk = jb.assign(_k=jb["currency"] + "|" + jb["title"].str.lower())
    jbk = jbk[jbk["_k"].isin(set(ref["_k"]))]
    score = {}
    for z in JB_ZONE_CANDIDATES:
        t = localize_wall(jbk["wall"], z)
        m = pd.merge_asof(jbk.assign(t=t.to_numpy()).dropna(subset=["t"]).sort_values("t"),
                          ref[["t", "_k"]].sort_values("t").rename(columns={"t": "t_ref"}),
                          left_on="t", right_on="t_ref", by="_k",
                          tolerance=pd.Timedelta(hours=14), direction="nearest")
        m = m.dropna(subset=["t_ref"])
        if len(m) >= min_matches:
            score[z] = (float((m["t"] == m["t_ref"]).mean()), len(m))
    if not score:
        return done(None, f"fewer than {min_matches} releases to compare")
    best = max(score, key=lambda z: (score[z][0], score[z][1]))
    share, n = score[best]
    if share < min_agree:
        return done(None, f"best {best} agrees on only {share:.0%} of {n}")
    rivals = [z for z, (s, _) in score.items() if z != best and s >= share - 0.02
              and not localize_wall(_PROBE, z).equals(localize_wall(_PROBE, best))]
    if rivals:
        return done(None, f"{best} and {rivals} agree on all {n} releases so far; they differ "
                          f"in some weeks of the year, so the data cannot choose yet")
    return done(best, f"{best}: {share:.1%} of {n} releases on the reference minute")
