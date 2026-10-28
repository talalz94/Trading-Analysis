"""
exness/levels.py on the stand-in runtime (tests/indie_standin.py), held to the research code:
quant.indicators.levels.period_levels for the previous day / week / month, session_windows for the
sessions, and Python's zoneinfo for the daylight-saving rules Indie has to spell out by hand.
"""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import indie_standin as rt
from quant.indicators.levels import period_levels, session_windows

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ROOT / "exness" / "levels.py"
DATA = ROOT / "tests" / "data"
ZONES = {1: "America/New_York", 2: "Europe/London", 3: "Asia/Tokyo", 4: "Australia/Sydney"}
SESSIONS = {"Sydney": ("Australia/Sydney", "07:00", "16:00"),     # notebooks/nb23_config.py
            "Tokyo": ("Asia/Tokyo", "09:00", "18:00"),
            "London": ("Europe/London", "08:00", "17:00"),
            "New York": ("America/New_York", "08:00", "17:00")}
ALL = dict(show_sydney=True, show_month=True, days_kept=10_000, sessions_kept=10_000)


def _bare(monkeypatch, bar_secs=300, **settings):
    """An indicator object to call the calendar methods on."""
    mod = rt.load(LEVELS, monkeypatch)
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv").iloc[:2]
    return rt.run(mod, bars, bar_secs=bar_secs, **settings)


def test_the_clock_rules_are_zoneinfos_hour_by_hour(monkeypatch):
    ctx = _bare(monkeypatch)
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    hours = range(t0, t0 + 5 * 366 * 86400, 3600)                 # 2024 to the end of 2028
    for zone, name in ZONES.items():
        z = ZoneInfo(name)
        for t in hours:
            want = datetime.fromtimestamp(t, z).utcoffset().total_seconds()
            assert ctx._offset(zone, float(t)) == want, (name, datetime.fromtimestamp(t, z))
    for t in hours[::24]:
        d = datetime.fromtimestamp(t, timezone.utc)
        z = t // 86400
        assert ctx._civil(z) == [d.year, d.month, d.day] and ctx._weekday(z) == d.weekday()
        assert ctx._days(d.year, d.month, d.day) == z


def test_every_session_opens_and_closes_when_session_windows_says(monkeypatch):
    ctx = _bare(monkeypatch)
    want = session_windows(SESSIONS, "2024-01-01", "2029-01-01")
    names = ["Sydney", "Tokyo", "London", "New York"]
    for s, name in enumerate(names):
        w = want[want["name"] == name]
        tz = SESSIONS[name][0]
        for o, c in zip(w["open"], w["close"]):
            day = int(pd.Timestamp(o).tz_convert(tz).tz_localize(None).normalize().value
                      // 86_400_000_000_000)
            assert ctx._window(s, day) == [o.timestamp(), c.timestamp()], (name, o)


def _segments(ctx, colour, style):
    return sorted((s.point_a.time, s.point_b.time, s.point_a.price)
                  for s in ctx.chart.items(rt.LineSegment)
                  if repr(s.color) == repr(colour) and s.line_style == style)


def _want_periods(bars, period, day_start):
    shift = pd.Timedelta(hours=7) if day_start.startswith("17:00") else pd.Timedelta(0)
    tz = "UTC" if day_start == "UTC midnight" else "America/New_York"
    f = bars.assign(open_time=bars["open_time"] + shift)
    p = period_levels(f, period, tz=tz)
    out = []
    for r in p.itertuples():
        a, b = (r.start - shift).timestamp(), (r.end - shift).timestamp()
        out += [(a, b, r.high), (a, b, r.low)]
    return sorted(out)


@pytest.mark.parametrize("day_start", ["UTC midnight", "New York midnight",
                                       "17:00 New York (FX day)"])
def test_previous_day_and_week_are_period_levels_on_fx_candles(monkeypatch, day_start):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(LEVELS, monkeypatch)
    ctx = rt.run(mod, bars, bar_secs=300, day_start=day_start, **ALL)
    for p, period in enumerate("DW"):
        want = _want_periods(bars, period, day_start)
        assert len(want) >= (20 if period == "D" else 4)
        assert _segments(ctx, ctx._cols[p], "dashed") == want, period


def _synthetic(start, days, minutes=60, seed=3):
    """A market open around the clock (crypto-like), for months and both clock changes."""
    t = pd.date_range(start, periods=days * 1440 // minutes, freq=f"{minutes}min", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.3, len(t)))
    o = np.r_[100.0, c[:-1]]
    w = np.abs(rng.normal(0, 0.2, len(t)))
    return pd.DataFrame(dict(open_time=t, open=o, close=c, high=np.maximum(o, c) + w,
                             low=np.minimum(o, c) - w))


@pytest.mark.parametrize("day_start", ["UTC midnight", "New York midnight",
                                       "17:00 New York (FX day)"])
def test_day_week_and_month_across_both_clock_changes_on_a_24h_market(monkeypatch, day_start):
    bars = _synthetic("2026-01-25 23:00", 130)                        # into June, both changes
    ctx = rt.run(rt.load(LEVELS, monkeypatch), bars, bar_secs=3600, day_start=day_start, **ALL)
    for p, period in enumerate("DWM"):
        want = _want_periods(bars, period, day_start)
        got = _segments(ctx, ctx._cols[p], "dashed")
        if period == "M":
            want = want[-6:]                                          # 3 months are kept
        assert got == want and len(want) >= 6, period


def _want_sessions(bars):
    w = session_windows(SESSIONS, bars["open_time"].iloc[0], bars["open_time"].iloc[-1]
                        + pd.Timedelta(minutes=5))
    out = {}
    for r in w.itertuples():
        inside = bars[(bars["open_time"] >= r.open) & (bars["open_time"] < r.close)]
        if len(inside):
            a, b = r.open.timestamp(), r.close.timestamp()
            out.setdefault(r.name, []).extend([(a, b, inside["high"].max()),
                                               (a, b, inside["low"].min())])
    return {k: sorted(v) for k, v in out.items()}


@pytest.mark.parametrize("name", ["usdjpy_5m_2026-10-22.csv", "24h market through March"])
def test_each_sessions_high_and_low_are_its_candles_extremes(monkeypatch, name):
    # FX candles, and a market open around the clock across the US (8 Mar) and UK (29 Mar)
    # clock changes, when London and New York open an hour apart from their usual UTC times
    bars = (rt.read_bars(DATA / name) if name.endswith(".csv")
            else _synthetic("2026-03-02", 35, minutes=5))
    ctx = rt.run(rt.load(LEVELS, monkeypatch), bars, bar_secs=300, **ALL)
    want = _want_sessions(bars)
    for s, sname in enumerate(["Sydney", "Tokyo", "London", "New York"]):
        assert _segments(ctx, ctx._cols[3 + s], "solid") == want.get(sname, []), sname
    assert sum(map(len, want.values())) >= 8


def test_a_session_is_carried_on_until_it_opens_again_and_shows_its_high_so_far(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(LEVELS, monkeypatch)
    ctx = rt.run(mod, bars, bar_secs=300, **ALL)
    t = (bars["open_time"].astype("int64") // 10**9).to_numpy(float)
    london = _segments(ctx, ctx._cols[5], "solid")
    carried = _segments(ctx, ctx._cols[5](0.7), "dotted")
    opens = sorted({a for a, b, y in london})
    for (a, b, y), (a2, b2, y2) in zip(london, carried):
        assert a2 == b and y2 == y                                   # carried from its close ...
        nxt = [o for o in opens if o > a]
        if nxt:
            assert b2 == nxt[0]                                       # ... to the next open
    # cut the candles in the middle of a London session: its lines are the high SO FAR
    w = session_windows({"London": SESSIONS["London"]}, bars["open_time"].iloc[0],
                        bars["open_time"].iloc[-1])
    o, c = w["open"].iloc[5], w["close"].iloc[5]
    k = int(np.searchsorted(t, (o + (c - o) / 2).timestamp()))
    part = bars.iloc[:k]
    ctx = rt.run(mod, part, bar_secs=300, **ALL)
    so_far = part[part["open_time"] >= o]
    last = _segments(ctx, ctx._cols[5], "solid")[-2:]
    assert sorted(y for _, _, y in last) == [so_far["low"].min(), so_far["high"].max()]


def test_live_candles_reruns_and_the_drawing_limit(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")[:3000]
    mod = rt.load(LEVELS, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300, show_sydney=True, show_month=True)
    ctx = rt.run(mod, bars, bar_secs=300, show_sydney=True, show_month=True, realtime_last=40,
                 rerun=range(0, len(bars), 7))
    assert rt.state(ctx) == rt.state(plain)
    assert ctx.chart.peak <= 90 and ctx.chart.forming == 0
    # what is kept: 10 days (2 lines each), 5 of each session (4 lines each with carry-on)
    seg = ctx._seg_owner.get()
    assert seg.count(0) == 20 and all(seg.count(10 + s) == 20 for s in range(4))
    assert len(ctx.chart.items(rt.LineSegment)) == len(seg)


def test_sessions_only_where_candles_fit_inside_the_hour_and_periods_above_the_chart(
        monkeypatch):
    # a 4h candle straddles a session's open and close, so its high could be from outside the
    # session; on a daily chart the previous day is just the candle before
    mod = rt.load(LEVELS, monkeypatch)
    hourly = _synthetic("2026-01-25 23:00", 130)
    owners = lambda c: set(c._seg_owner.get())
    one_h = rt.run(mod, hourly, bar_secs=3600, **ALL)
    assert {10, 11, 12, 13} <= owners(one_h) and {0, 1, 2} <= owners(one_h)
    four_h = rt.run(mod, rt.htf_bars(hourly, 240), bar_secs=4 * 3600, **ALL)
    assert owners(four_h) == {0, 1, 2}                          # no sessions
    daily = rt.run(mod, rt.htf_bars(hourly, 1440), bar_secs=86400, **ALL)
    assert owners(daily) == {1, 2}                              # no day levels, no sessions
    # what is drawn above the chart is still period_levels'
    want = _want_periods(rt.htf_bars(hourly, 1440), "W", "UTC midnight")
    assert _segments(daily, daily._cols[1], "dashed") == want
