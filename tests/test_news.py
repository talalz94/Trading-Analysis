"""
quant.news -- the economic calendar. Every rule is checked on rows built by hand, so the answer is
known before the code runs: the archive's broken clock is repaired to the official minute, sources
are merged by trust and never mixed within a field, a cancelled event leaves the calendar, and no
column on a bar grid knows a number before it was released.
"""
import numpy as np
import pandas as pd
import pytest

from quant.news import catalog, sources
from quant.news.calendar import (NewsStore, build_events, calibrate_jb_zone, event_keys,
                                 feed_week, load_events, merge_rows, select)
from quant.news.features import news_blackout, news_features
from quant.news.values import parse_values, sentiment, surprise_z

NY = "America/New_York"


def et(ts):
    return pd.Timestamp(ts).tz_convert(NY).strftime("%Y-%m-%d %H:%M")


# ── values ──────────────────────────────────────────────────────────────────────────────────
def test_printed_values_parse_with_their_units_and_non_numbers_stay_empty():
    got = parse_values(["0.3%", "-0.1%", "245K", "-1.2B", "51.2", "4.35|2.3", "12k", "9.9%%",
                        "<0.1%", "0-0-9", "Pass", None, ""])
    want = [0.3, -0.1, 245e3, -1.2e9, 51.2, 4.35, 12e3, 9.9]
    assert np.allclose(got[:8], want)
    assert np.isnan(got[8:]).all()


def test_sentiment_is_for_the_currency_and_respects_lower_is_better():
    # CPI above forecast is good for the currency; unemployment above forecast is bad
    s = sentiment([0.1, 0.1, 0.0, np.nan, 0.2], [1, -1, 1, 1, 0])
    assert s.tolist() == ["better", "worse", "as expected", "", ""]
    assert catalog.direction("Unemployment Rate") == -1
    assert catalog.direction("Unemployment Claims") == -1
    assert catalog.direction("CPI m/m") == 1
    assert catalog.direction("Federal Funds Rate") == 1
    assert catalog.direction("Trade Balance") == 1
    assert catalog.direction("10-y Bond Auction") == 0
    assert catalog.direction("Fed Chair Powell Speaks") == 0


def test_direction_rules_match_forex_factorys_own_wherever_the_archive_is_present():
    d = NewsStore().read("directions")
    if d is None:
        pytest.skip("archive not imported (python -m quant.news import-archive)")
    ev = NewsStore().read("events")
    valued = set(ev.loc[ev["actual_v"].notna(), "title"])
    d = d[d["title"].isin(valued) & (d["direction"] != 0)]
    mine = d["title"].map(catalog.direction)
    assert len(d) > 300 and (mine == d["direction"]).all(), d[mine != d["direction"]]


def test_surprise_z_never_knows_a_later_release():
    rng = np.random.default_rng(1)
    n = 40
    ev = pd.DataFrame({"currency": "USD", "title": "CPI m/m",
                       "t": pd.date_range("2020-01-10", periods=n, freq="30D", tz="UTC"),
                       "surprise": rng.normal(0, 0.1, n)})
    full = surprise_z(ev)
    assert np.isnan(full[:8]).all() and np.isfinite(full[8:]).all()
    for k in range(9, n):
        cut = surprise_z(ev.iloc[:k])
        assert np.allclose(cut, full[:k], equal_nan=True)
    # scaled by the PAST: release 8 is its surprise over the rms of releases 0..7
    assert np.isclose(full[8], ev.surprise[8] / np.sqrt((ev.surprise[:8] ** 2).mean()))


# ── catalog ─────────────────────────────────────────────────────────────────────────────────
def test_critical_is_only_ever_an_upgrade_of_high():
    assert catalog.criticality("USD", "CPI m/m", "high") == "critical"
    assert catalog.criticality("USD", "CPI m/m", "medium") == "medium"
    assert catalog.criticality("EUR", "CPI m/m", "high") == "high"        # not on EUR's list
    assert catalog.criticality("USD", "ISM Manufacturing PMI", "high") == "high"


def test_official_time_follows_the_era():
    assert catalog.official_time("USD", "Federal Funds Rate", "2012-12-12") is None
    assert catalog.official_time("USD", "Federal Funds Rate", "2013-01-30") == "14:00"
    assert catalog.official_time("USD", "CPI m/m", "2024-07-11") == "08:30"
    assert catalog.official_time("EUR", "CPI m/m", "2024-07-11") is None


def test_coverage_is_validated_and_star_covers_the_rest():
    with pytest.raises(ValueError, match="unknown currency"):
        catalog.parse_coverage({"XYZ": "high"})
    with pytest.raises(ValueError, match="level"):
        catalog.parse_coverage({"USD": "red"})
    ev = pd.DataFrame({"currency": ["USD", "USD", "EUR", "EUR", "AUD"],
                       "level": ["medium", "low", "high", "medium", "critical"]})
    assert select(ev, {"USD": "medium", "EUR": "high"}).index.size == 2
    kept = select(ev, {"USD": "medium", "*": "critical"})
    assert kept["currency"].tolist() == ["USD", "AUD"]


# ── the archive's clock ─────────────────────────────────────────────────────────────────────
def raw(rows):
    return pd.DataFrame(rows, columns=["DateTime", "Currency", "Impact", "Event", "Actual",
                                       "Forecast", "Previous"])


HI, MED = "High Impact Expected", "Medium Impact Expected"


def test_archive_times_are_reread_in_the_zone_they_were_really_in():
    # Iran had no daylight saving after 2022, but the scrape stamped +04:30 anyway: claims at
    # 16:00 Tehran are 12:30 UTC = 8:30 New York -- read with the stamp they would be 7:30
    r = sources.parse_archive(raw([
        ("2024-07-11T16:00:00+04:30", "USD", MED, "Unemployment Claims", "222K", "236K", "239K"),
        ("2021-07-08T17:00:00+04:30", "USD", MED, "Unemployment Claims", "373K", "350K", "364K"),
    ]))
    assert [et(t) for t in r["t"]] == ["2024-07-11 08:30", "2021-07-08 08:30"]
    assert (r["time_q"] == "exact").all()


def test_untimed_archive_rows_get_the_official_time_then_their_history_then_nothing():
    r = sources.parse_archive(raw([
        # official US time on the archive's own date
        ("2024-07-11T00:00:00+04:30", "USD", HI, "CPI m/m", "-0.1%", "0.1%", "0.0%"),
        # no official time: the same event's nearest TIMED release, in its home zone
        ("2024-03-05T13:00:00+03:30", "EUR", MED, "Made Up Index", "1", "1", "1"),
        ("2024-04-05T00:00:00+04:30", "EUR", MED, "Made Up Index", "2", "1", "1"),
        # nothing to go on: the date only, at noon at home
        ("2024-04-09T00:00:00+04:30", "JPY", MED, "Never Timed", None, None, None),
        # a holiday
        ("2024-04-10T23:59:59+04:30", "USD", "Non-Economic", "Bank Holiday", None, None, None),
    ]))
    assert et(r.t[0]) == "2024-07-11 08:30" and r.time_q[0] == "official"
    home = [pd.Timestamp(t).tz_convert("Europe/Berlin").strftime("%H:%M") for t in r.t[1:3]]
    assert home == ["10:30", "10:30"] and r.time_q.tolist()[1:3] == ["exact", "typical"]
    assert pd.Timestamp(r.t[2]).tz_convert("Europe/Berlin").day == 5
    assert r.time_q[3] == "date"
    assert pd.Timestamp(r.t[3]).tz_convert("Asia/Tokyo").strftime("%d %H:%M") == "09 12:00"
    assert r.time_q[4] == "allday"


def test_forex_factory_feed_times_carry_their_offset():
    r = sources.parse_ff_week([
        {"title": "CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00",
         "impact": "High", "forecast": "0.6%", "previous": "0.4%"},
        {"title": "Bank Holiday", "country": "JPY", "date": "2026-10-11T19:00:00-04:00",
         "impact": "Holiday", "forecast": "", "previous": ""},
        {"title": "broken", "country": "USD", "date": "not a date", "impact": "Low"},
    ])
    assert len(r) == 2
    assert r.t[0] == pd.Timestamp("2026-10-14 12:30", tz="UTC") and r.time_q[0] == "exact"
    assert r.impact.tolist() == ["high", "none"] and r.time_q[1] == "allday"
    assert r.forecast[1] is None


# ── merging ─────────────────────────────────────────────────────────────────────────────────
def row(t, src, *, title="CPI m/m", q="exact", impact="high", actual=None, forecast=None,
        previous=None, cur="USD", **kw):
    return dict(t=pd.Timestamp(t, tz="UTC"), time_q=q, currency=cur, title=title, impact=impact,
                actual=actual, forecast=forecast, previous=previous, source=src, **kw)


def test_one_release_from_several_sources_shares_a_key_and_a_repeat_is_numbered():
    df = pd.DataFrame([row("2024-07-11 12:30", "ff_archive"), row("2024-07-11 16:00", "fred",
                                                                    q="official"),
                       row("2024-07-11 13:00", "ff_feed", title="Fed Chair Powell Speaks"),
                       row("2024-07-11 20:00", "ff_feed", title="Fed Chair Powell Speaks")])
    k = event_keys(df)
    assert k[0] == k[1] == "USD|cpi m/m|2024-07-11"
    assert k[2] == "USD|fed chair powell speaks|2024-07-11" and k[3] == k[2] + "#1"


def test_each_field_comes_from_the_most_trusted_source_that_has_it():
    rows = pd.DataFrame([
        row("2024-07-11 12:00", "ff_archive", q="typical", actual="-0.1%", forecast="0.2%",
            previous="0.0%"),
        row("2024-07-11 12:30", "ff_feed", forecast="0.1%", previous="0.0%",
            fetched_at=pd.Timestamp("2024-07-10", tz="UTC")),
        # observed AFTER the release: its forecast is not what the market expected
        row("2024-07-11 12:30", "ff_feed", forecast="0.9%", previous="0.0%",
            fetched_at=pd.Timestamp("2024-07-12", tz="UTC")),
        row("2024-07-11 16:00", "fred", q="official", impact="high"),
    ])
    ev = merge_rows(rows)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e.t == pd.Timestamp("2024-07-11 12:30", tz="UTC") and e.time_q == "exact"
    assert e.forecast == "0.1%" and e.actual == "-0.1%"
    assert e.level == "critical" and e.category == "inflation"
    assert np.isclose(e.surprise, -0.2) and e.sentiment == "worse"
    assert e.source == "ff_feed+ff_archive+fred"


def test_an_official_time_beats_a_typical_one_and_jblanked_actual_beats_the_archive():
    rows = pd.DataFrame([
        row("2023-05-10 11:00", "ff_archive", q="typical", actual="0.4%", forecast="0.4%"),
        row("2023-05-10 12:30", "fred", q="official"),
        row("2023-05-10 12:30", "jblanked", actual="0.5%"),
    ])
    e = merge_rows(rows).iloc[0]
    assert e.t == pd.Timestamp("2023-05-10 12:30", tz="UTC") and e.time_q == "exact"
    assert e.actual == "0.5%" and e.forecast == "0.4%"


def test_a_moved_or_cancelled_feed_event_leaves_the_calendar(tmp_path):
    st = NewsStore(tmp_path)
    week = [row(f"2026-10-1{d} 12:30", "ff_feed", title=f"Filler {i}", impact="low")
            for i, d in enumerate([2, 3, 4, 5, 6] * 4)]
    cpi = row("2026-10-14 12:30", "ff_feed", forecast="0.6%")
    st.log_feed(pd.DataFrame(week + [cpi]).drop(columns="source").assign(source="ff_feed"),
                "2026-10-12 10:00")
    moved = row("2026-10-15 12:30", "ff_feed", forecast="0.6%")
    st.log_feed(pd.DataFrame(week + [moved]), "2026-10-13 10:00")
    ev = build_events(st)
    cpis = ev[ev.title == "CPI m/m"]
    assert cpis["t"].tolist() == [pd.Timestamp("2026-10-15 12:30", tz="UTC")]
    # a short, possibly truncated answer is logged but cancels nothing
    st.log_feed(pd.DataFrame([moved])[:0].reindex(columns=sources.ROW_COLS), "2026-10-13 11:00")
    st.log_feed(pd.DataFrame(week[:3]), "2026-10-13 12:00")
    assert len(build_events(st)) == len(ev)


def test_the_feed_log_keeps_only_changes(tmp_path):
    st = NewsStore(tmp_path)
    a = pd.DataFrame([row("2026-10-14 12:30", "ff_feed", forecast="0.6%")])
    assert st.log_feed(a, "2026-10-12 10:00", min_events=1) == 1
    assert st.log_feed(a, "2026-10-12 11:00", min_events=1) == 0
    b = a.assign(forecast="0.5%")                         # a revised forecast is kept
    assert st.log_feed(b, "2026-10-12 12:00", min_events=1) == 1
    assert len(st.read("feed_log")) == 2
    assert feed_week(pd.Series([pd.Timestamp("2026-10-11 23:30", tz="UTC")]))[0] == "2026-10-11"


def test_load_events_rebuilds_when_a_source_is_newer(tmp_path):
    st = NewsStore(tmp_path)
    assert load_events(store=st).empty
    st.write("fred", sources._rows(pd.DataFrame([row("2024-07-11 12:30", "fred",
                                                     q="official")])))
    ev = load_events(store=st)
    assert len(ev) == 1
    st.write("fred", sources._rows(pd.DataFrame([row("2024-07-11 12:30", "fred", q="official"),
                                                 row("2024-08-14 12:30", "fred",
                                                     q="official")])))
    assert len(load_events(store=st)) == 2
    assert len(load_events("2024-08-01", store=st)) == 1


# ── on a bar grid ───────────────────────────────────────────────────────────────────────────
def bars(start="2024-07-11 12:00", n=12, freq="5min"):
    t = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})


def calendar_one(t="2024-07-11 12:30", z=1.5):
    ev = merge_rows(pd.DataFrame([row(t, "ff_archive")]))
    ev["surprise_z"] = z
    return ev


def test_minutes_are_counted_from_the_bar_close():
    f = news_features(bars(), calendar_one())
    # the 12:20 bar closes at 12:25: CPI in 5 minutes. The 12:25 bar closes AT the release.
    assert f["news_next_min"].iloc[4] == 5
    assert f["news_prev_min"].iloc[5] == 0 and np.isinf(f["news_next_min"].iloc[5])
    assert f["news_day"].all()


def test_a_release_is_known_only_from_the_first_bar_closing_after_it():
    f = news_features(bars(), calendar_one())
    sz = f["news_last_surprise_z"]
    # the 12:25 bar closes at exactly 12:30 -- the number is not out yet at that instant
    assert np.isnan(sz.iloc[:6]).all() and (sz.iloc[6:] == 1.5).all()
    for k in range(1, 12):                   # truncation changes nothing already computed
        cut = news_features(bars(n=k), calendar_one())
        assert cut["news_last_surprise_z"].equals(sz.iloc[:k]) or k < 2


def test_blackout_covers_before_and_after():
    b = news_blackout(bars(n=24), calendar_one(), before=10, after=15)
    closes = bars(n=24)["open_time"] + pd.Timedelta(minutes=5)
    want = (closes >= pd.Timestamp("2024-07-11 12:20", tz="UTC")) & \
           (closes <= pd.Timestamp("2024-07-11 12:45", tz="UTC"))
    assert b.tolist() == want.tolist()


def test_coverage_decides_what_the_grid_sees():
    ev = merge_rows(pd.DataFrame([row("2024-07-11 12:30", "ff_archive", cur="EUR",
                                      title="German Thing", impact="medium")]))
    f = news_features(bars(), ev, coverage={"EUR": "high"})
    assert np.isinf(f["news_next_min"]).all() and not f["news_day"].any()


# ── JBlanked's zone and FRED ────────────────────────────────────────────────────────────────
def test_jblanked_zone_is_found_only_when_the_evidence_is_convincing():
    ref = pd.DataFrame([row(t, "ff_feed", title=f"E{i}") for i, t in enumerate(
        ["2026-01-14 13:30", "2026-01-15 13:30", "2026-07-14 12:30", "2026-07-15 12:30",
         "2026-07-16 14:00", "2026-01-16 15:00"])])
    walls = pd.DataFrame({"currency": "USD", "title": ref.title,
                          "wall": ref.t.dt.tz_convert("Europe/Athens").dt.tz_localize(None)})
    assert calibrate_jb_zone(walls, ref) == "Europe/Athens"
    assert calibrate_jb_zone(walls.iloc[:3], ref) is None          # too few to trust
    noisy = walls.assign(wall=walls.wall + pd.to_timedelta([0, 1, 2, 3, 4, 5], unit="h"))
    assert calibrate_jb_zone(noisy, ref) is None                    # nothing agrees


def test_fred_dates_become_official_times_with_daylight_saving(monkeypatch):
    def fake(path, key, timeout=30, **p):
        if path == "releases":
            return {"count": len(sources.FRED_RELEASES),
                    "releases": [{"name": n, "id": i} for i, n in enumerate(sources.FRED_RELEASES)]}
        rid = p["release_id"]
        name = list(sources.FRED_RELEASES)[rid]
        days = {"Consumer Price Index": ["2024-01-11", "2024-07-11"],
                "Gross Domestic Product": ["2024-07-25", "2024-08-29", "2024-09-26"]}.get(name, [])
        return {"count": len(days), "release_dates": [{"date": d} for d in days]}
    monkeypatch.setattr(sources, "_fred", fake)
    r = sources.fred_rows("k", pause=0)
    cpi = r[r.title == "CPI m/m"]
    assert [et(t) for t in cpi.t] == ["2024-01-11 08:30", "2024-07-11 08:30"]
    assert cpi.t.iloc[0].hour == 13 and cpi.t.iloc[1].hour == 12              # EST vs EDT
    gdp = r[r.title.str.contains("GDP")]
    assert gdp.title.tolist() == ["Advance GDP q/q", "Prelim GDP q/q", "Final GDP q/q"]
    assert (r.time_q == "official").all() and (r.source == "fred").all()


def test_a_missing_fred_release_is_named_not_skipped(monkeypatch):
    monkeypatch.setattr(sources, "_fred", lambda path, key, **p: {
        "count": 1, "releases": [{"name": "Consumer Price Index", "id": 10}]})
    with pytest.raises(sources.SourceError, match="no release named"):
        sources.fred_rows("k", pause=0)


def test_one_source_failing_does_not_stop_the_others(tmp_path, monkeypatch):
    from quant.news import update
    monkeypatch.setattr(update, "keys", lambda: {"fred": "x", "jblanked": None})
    monkeypatch.setattr(sources, "fetch_ff_week", lambda **k: [
        {"title": "CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00",
         "impact": "High", "forecast": "0.6%", "previous": "0.4%"}])
    def boom(*a, **k):
        raise sources.SourceError("FRED down")
    monkeypatch.setattr(sources, "fred_rows", boom)
    out = update.update_all(NewsStore(tmp_path), pd.Timestamp("2026-10-13 10:00", tz="UTC"))
    assert any(s.startswith("feed: 1 events") for s in out)
    assert any("fred: FAILED" in s for s in out)
    assert any("jblanked: skipped" in s for s in out)
    assert len(load_events(store=NewsStore(tmp_path))) == 1
