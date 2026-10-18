"""
The chart layers added 2026-10-18 -- previous period highs and lows, market sessions, candlestick
markers, and S/R drawn as bands coloured by role -- plus notebook 23, which uses them.

The promises: a previous day's / week's high is exactly what the existing per-bar helpers say, a
partly-loaded period is never a source, and Monday's "yesterday" is Friday; a session opens at its
own market's local time through daylight saving, and Sydney's Monday is Sunday in UTC; an S/R zone
that is broken and held beyond ENDS there instead of running to the right edge, and its colour
changes at the close that changed its role; every new layer is APPENDED (the chart without it is
untouched), its dropdown touches only its own traces, and the levels rewind while the timetable
does not; a setting that cannot mean what it says is refused by name.
"""
from __future__ import annotations

import ast
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant.indicators import (add_prev_day_levels, add_week_levels, period_levels,
                              session_windows)
from quant.research.chart_layers import chart_layers
from quant.research.workspace import NOTEBOOKS
from quant.viz import pattern_figure
from test_notebook_config import SETTING_RE


def bars(start, n, freq="1h", price=None, seed=3):
    t = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    c = (100 + np.random.default_rng(seed).normal(0, 0.5, n).cumsum()) if price is None \
        else np.asarray(price, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open_time": t, "t": t, "open": o, "high": np.maximum(o, c) + 0.2,
                         "low": np.minimum(o, c) - 0.2, "close": c, "volume": 1.0})


def per_bar(df, tb):
    m = pd.merge_asof(df[["open_time"]], tb.rename(columns={"start": "open_time"}),
                      on="open_time")
    inside = (m["open_time"] < m["end"]).to_numpy()
    return np.where(inside, m["high"], np.nan), np.where(inside, m["low"], np.nan)


# ── period levels ───────────────────────────────────────────────────────────────────────────
def test_previous_day_and_week_levels_are_what_the_per_bar_helpers_say():
    df = bars("2026-03-02", 24 * 40)                  # starts on a Monday at midnight: whole
    h, low = per_bar(df, period_levels(df, "D"))
    ref = add_prev_day_levels(df, time_col="open_time")
    assert np.array_equal(h, ref["pd_high"].to_numpy(), equal_nan=True)
    assert np.array_equal(low, ref["pd_low"].to_numpy(), equal_nan=True)
    hw, _ = per_bar(df, period_levels(df, "W"))
    refw = add_week_levels(df, time_col="open_time")["wk_high"].to_numpy()
    both = ~np.isnan(hw) & ~np.isnan(refw)
    assert both.sum() > 24 * 30 and np.array_equal(hw[both], refw[both])


@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_a_level_uses_only_the_bars_of_the_period_before_it(tz):
    # in New York's zone too: a day there runs midnight to midnight NEW YORK time, so its high is
    # the high of the bars between those instants -- not of a UTC day with New York labels on it
    df = bars("2026-03-02", 24 * 75)
    for p in ("D", "W", "M"):
        for r in period_levels(df, p, tz=tz).itertuples():
            src = df[(df["open_time"] >= r.source_start) & (df["open_time"] < r.start)]
            assert len(src) and src["high"].max() == r.high and src["low"].min() == r.low, p
            assert (src["open_time"] < r.start).all()          # nothing from the period itself


def test_a_partly_loaded_period_is_never_a_source():
    df = bars("2026-03-02 13:00", 24 * 10)            # the first day is loaded from 13:00
    tb = period_levels(df, "D")
    assert tb["source_start"].min() == pd.Timestamp("2026-03-03", tz="UTC")


def test_mondays_previous_day_is_friday_when_the_market_shuts_at_the_weekend():
    df = bars("2026-03-02", 24 * 14)
    df = df[df["open_time"].dt.dayofweek < 5].reset_index(drop=True)
    tb = period_levels(df, "D")
    mon = tb[tb["start"] == pd.Timestamp("2026-03-09", tz="UTC")].iloc[0]
    assert mon["source_start"] == pd.Timestamp("2026-03-06", tz="UTC")   # Friday


def test_a_day_starts_at_midnight_in_the_zone_asked_for():
    df = bars("2026-03-02", 24 * 20)
    ny = period_levels(df, "D", tz="America/New_York")
    starts = pd.to_datetime(ny["start"], utc=True)
    assert set(starts.dt.tz_convert("America/New_York").dt.hour) == {0}
    # the clocks change on 8 March: midnight is 05:00 UTC before and 04:00 UTC after
    assert {5, 4} <= set(starts.dt.hour)


# ── sessions ────────────────────────────────────────────────────────────────────────────────
def test_a_session_opens_at_its_own_markets_time_through_daylight_saving():
    w = session_windows({"London": ("Europe/London", "08:00", "17:00")},
                        "2026-10-20", "2026-10-30")
    o = pd.to_datetime(w["open"], utc=True)
    assert {(t.day, t.hour) for t in o} >= {(23, 7), (26, 8)}    # BST ends on 25 October
    assert set(o.dt.tz_convert("Europe/London").dt.hour) == {8}
    assert (pd.to_datetime(w["close"]) - pd.to_datetime(w["open"])).eq(pd.Timedelta(hours=9)).all()


def test_sydneys_monday_is_sunday_in_utc_and_no_session_opens_on_a_saturday():
    w = session_windows({"Sydney": ("Australia/Sydney", "07:00", "16:00")},
                        "2026-10-23", "2026-10-28")
    local = pd.to_datetime(w["open"], utc=True).dt.tz_convert("Australia/Sydney")
    assert set(local.dt.dayofweek) <= {0, 1, 2, 3, 4}
    assert (pd.to_datetime(w["open"], utc=True).dt.dayofweek == 6).any()    # Sunday UTC
    night = session_windows({"X": ("UTC", "22:00", "06:00")}, "2026-10-26", "2026-10-28")
    assert (pd.to_datetime(night["close"]) - pd.to_datetime(night["open"])
            ).eq(pd.Timedelta(hours=8)).all()


# ── S/R: a zone that breaks ends there, and changes colour at the close that broke it ────────
def _sr_market():
    """Six swings between 100 and 110, then a break above 110 that holds at 120."""
    x = np.arange(240)
    p = list(105 + 5 * np.sin(2 * np.pi * x / 40)) + [116, 119] + [120] * 58
    return bars("2026-03-02", len(p), freq="5min", price=p)


def _sr_figure(df, sr_break_bars=3, **kw):
    return pattern_figure(df, patterns=[], sr_zones=True, sr_sensitivity="normal",
                          zones_left=3, zones_right=3, cluster_pct=0.5, min_touches=2,
                          sr_anchor=True, sr_break_bars=sr_break_bars, max_zones=10,
                          chart_tf="5m", emas=(), line_picker=False, zone_lookback_bars=0, **kw)


def _sr_traces(fig):
    return [tr for tr in fig.data if (tr.legendgroup or "") == "S/R zones"
            and tr.mode == "lines"]


def _segments(xs, ys):
    seg, cur = [], []
    for x, y in zip(list(xs), list(ys)):
        if x is None:
            if cur:
                seg.append(cur)
            cur = []
        else:
            cur.append((pd.Timestamp(x), float(y)))
    if cur:
        seg.append(cur)
    return seg


def test_a_zone_broken_and_held_beyond_ends_where_it_broke():
    df = _sr_market()
    last = df["t"].iloc[-1]
    (line,) = _sr_traces(_sr_figure(df))
    segs = {round(s[0][1]): s for s in _segments(line.x, line.y)}
    near = {k: v for k, v in segs.items() if 108 <= k <= 112}
    assert near, segs.keys()
    for s in near.values():                          # the resistance at ~110: retired after the break
        assert max(t for t, _ in s) < last - pd.Timedelta(hours=3)
    lows = [s for k, s in segs.items() if 98 <= k <= 102]
    assert lows and any(max(t for t, _ in s) == last for s in lows)   # the support is still live


@pytest.mark.parametrize("rewind", [False, True])
def test_a_zone_from_the_warm_up_starts_at_the_view_and_a_dead_one_is_not_drawn(rewind):
    # With rewind on, a revision that ended in the warm-up was drawn from the first visible candle
    # BACKWARDS to its end: lines that all stopped at the view start, as if never carried forward.
    x = np.arange(240)                    # _sr_market's break, then swings 101-121 to look at
    p = (list(105 + 5 * np.sin(2 * np.pi * x / 40)) + [116, 119] + [120] * 18
         + list(111 + 10 * np.cos(2 * np.pi * np.arange(80) / 40)))
    df = bars("2026-03-02", len(p), freq="5min", price=p)
    vs = df["t"].iloc[250]                                 # after the ~110 zone was broken
    (line,) = _sr_traces(_sr_figure(df, view_start=vs, rewind=rewind))
    segs = _segments(line.x, line.y)
    assert segs
    for s in segs:
        assert min(t for t, _ in s) >= vs and s[0][0] < s[-1][0]       # inside, and forwards
    assert not [s for s in segs if 108 <= s[0][1] <= 112]              # broken before the view
    live = [s for s in segs if 98 <= s[0][1] <= 102]
    assert live and min(s[0][0] for s in live) == vs                  # carried in from the warm-up


def test_a_band_takes_the_zones_tolerance_and_its_colour_changes_at_the_close():
    df = _sr_market()
    fig = _sr_figure(df, sr_style="band", sr_roles=True)
    sup, res = _sr_traces(fig)
    assert sup.fill == "toself" and res.fill == "toself"
    assert sup.fillcolor.startswith("rgba(68,147,248") and res.fillcolor.startswith("rgba(240,136")
    res_boxes = [s for s in _segments(res.x, res.y) if 105 < np.mean([y for _, y in s]) < 115]
    assert res_boxes
    lo, hi = min(y for _, y in res_boxes[0]), max(y for _, y in res_boxes[0])
    assert hi - lo >= 0.5 / 100 * 110 * 0.99           # at least the clustering tolerance
    # the ~110 zone turns SUPPORT once price closes above it -- at that bar's CLOSE
    sup_110 = [s for s in _segments(sup.x, sup.y) if 105 < np.mean([y for _, y in s]) < 115]
    assert sup_110
    first_above = df.loc[df["close"] > 110.5, "t"].iloc[0]
    assert min(t for t, _ in sup_110[0]) == first_above + pd.Timedelta(minutes=5)


def test_break_bars_can_be_set_per_timeframe():
    df = _sr_market()
    never = _sr_figure(df.copy(), sr_break_bars={"1h": 3, "*": 0})       # 5m falls to "*" = 0
    (line,) = _sr_traces(never)
    ends = [max(t for t, _ in s) for s in _segments(line.x, line.y)
            if 108 <= np.mean([y for _, y in s]) <= 112]
    assert ends and max(ends) == df["t"].iloc[-1]                       # never retired


# ── the market layers only APPEND, and their menus touch only their own traces ─────────────
def _with_layers(df, **kw):
    lv = {p: period_levels(df, p) for p in ("D", "W")}
    w = session_windows({"London": ("Europe/London", "08:00", "17:00"),
                         "New York": ("America/New_York", "08:00", "17:00")},
                        df["t"].iloc[0], df["t"].iloc[-1])
    return pattern_figure(df, patterns=[], sr_zones=False, emas=(), line_picker=False,
                          period_levels=lv, levels_show=("D",), sessions=w,
                          session_open_lines=("London",), candle_patterns=["engulfing_bull"],
                          rewind=True, **kw)


def test_the_market_layers_are_appended_and_change_nothing_before_them():
    df = bars("2026-03-02", 24 * 20, freq="15min")
    base = pattern_figure(df, patterns=[], sr_zones=False, emas=(), line_picker=False,
                          rewind=True)
    full = _with_layers(df)
    n = len(base.data)
    from plotly.io import to_json
    assert [to_json(t) for t in full.data[:n]] == [to_json(t) for t in base.data]
    added = {t.meta for t in full.data[n:]}
    assert added == {"levels", "sessions", "candles"}
    rw = set(full.layout.meta["rewind"])
    for i, t in enumerate(full.data[n:], start=n):
        assert (i in rw) == (t.meta in ("levels", "candles")), t.meta   # the timetable stays


def test_each_footer_menu_addresses_only_its_own_traces():
    df = bars("2026-03-02", 24 * 20, freq="15min")
    fig = _with_layers(df)
    for menu in fig.layout.updatemenus:
        first = menu.buttons[0].label
        want = "levels" if first.startswith("Levels") else \
            "sessions" if first.startswith("Sessions") else None
        if want is None:
            continue
        for b in menu.buttons:
            idx = b.args[1]
            assert idx and {fig.data[i].meta for i in idx} == {want}, b.label
    shown = {t.legendgroup for t in fig.data if t.meta == "levels" and t.visible is True}
    assert shown == {"lvl:D"}


def test_open_lines_start_hidden_on_a_long_chart_and_shown_on_a_short_one():
    long_ = _with_layers(bars("2026-03-02", 24 * 4 * 20, freq="15min"))
    short = _with_layers(bars("2026-03-02", 24 * 4 * 3, freq="15min"))

    def line_vis(fig):
        return [t.visible for t in fig.data if t.meta == "sessions" and t.mode == "lines"
                and t.line.dash == "dot"]

    assert line_vis(long_) == ["legendonly"] and line_vis(short) == [True]


# ── the settings ────────────────────────────────────────────────────────────────────────────
def test_absent_settings_ask_for_nothing_and_bad_ones_are_refused_by_name():
    kw = dict(symbol="X", source="binance", view_start="2026-03-10")
    assert chart_layers(SimpleNamespace(), **kw) == {}
    df = bars("2026-01-01", 24 * 80)
    with pytest.raises(ValueError, match="'Q'"):
        chart_layers(SimpleNamespace(PERIOD_LEVELS=("D", "Q")), level_frame=df, **kw)
    with pytest.raises(ValueError, match="SR_STYLE"):
        chart_layers(SimpleNamespace(SR_STYLE="bands"), **kw)
    sess = {"London": ("Europe/London", "08:00", "17:00")}
    with pytest.raises(ValueError, match="Tokyo"):
        chart_layers(SimpleNamespace(SESSIONS=sess, SESSION_OPEN_LINES=("Tokyo",)), **kw)
    with pytest.raises(ValueError, match="London"):
        chart_layers(SimpleNamespace(SESSIONS={"London": ("Europe/London", "08:00")}), **kw)
    got = chart_layers(SimpleNamespace(PERIOD_LEVELS=("D", "W"), PERIOD_LEVELS_ON=("W", "Y"),
                                       CHART_PATTERNS="all", SR_STYLE="band"),
                       level_frame=df, **kw)
    assert set(got["period_levels"]) == {"D", "W"} and got["levels_show"] == ("W",)
    assert got["patterns"] is None and got["sr_style"] == "band"


def test_an_unknown_candlestick_pattern_is_refused_by_name():
    from quant.viz.markets import add_candle_markers
    import plotly.graph_objects as go
    with pytest.raises(ValueError, match="hamer"):
        add_candle_markers(go.Figure(), bars("2026-03-02", 50), ["hamer"])


# ── notebook 23 ─────────────────────────────────────────────────────────────────────────────
NB23 = NOTEBOOKS / "23_break_confirmation_news.ipynb"
CFG23 = NOTEBOOKS / "nb23_config.py"


def test_notebook_23_states_no_setting_and_defines_no_function():
    bad = []
    for c in json.loads(NB23.read_text(encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        bad += ["assigns: " + line.strip() for line in src.splitlines()
                if re.match(SETTING_RE, line)]
        bad += ["defines: " + n.name for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not bad, bad
    assert "nb23_config" in "".join("".join(c["source"])
                                    for c in json.loads(NB23.read_text(encoding="utf-8"))["cells"])


def test_notebook_23_config_has_what_the_lab_reads_and_stays_a_control_panel():
    b = {}
    exec(compile(CFG23.read_text(encoding="utf-8"), str(CFG23), "exec"), b)
    need = ["SYMBOL", "TF", "VIEW_START", "END", "WARMUP_START", "LINE_TFS", "TRENDLINE",
            "TL_CLOSE_N", "ENTRY_RULES", "STOP_N", "TP_RR", "COSTS", "COMPARE", "FVG_MITIGATION",
            "NEWS", "CHART_PATTERNS", "CANDLE_PATTERNS", "PERIOD_LEVELS", "SESSIONS",
            "SR_STYLE", "SR_ROLES", "SR_BREAK_BARS"]
    assert not [k for k in need if k not in b]
    lines = CFG23.read_text(encoding="utf-8").splitlines()
    comment = sum(1 for line in lines if line.lstrip().startswith("#"))
    assert len(lines) < 240 and comment / len(lines) < 0.35
    # the settings it offers are ones the chart accepts
    chart_layers(SimpleNamespace(**{k: v for k, v in b.items() if k.isupper()}),
                 symbol=b["SYMBOL"], source=b["SOURCE"], view_start=b["VIEW_START"],
                 level_frame=bars("2025-01-01", 24 * 600))


def test_the_chart_guide_names_every_pattern_the_code_has():
    from quant.indicators.candle_patterns import ALL_PATTERNS as CANDLES
    from quant.indicators.chart_patterns import ALL_PATTERNS as CHARTS
    guide = (NOTEBOOKS.parent / "docs" / "CHART_GUIDE.md").read_text(encoding="utf-8")
    missing = [n for n in list(CHARTS) + list(CANDLES) if f"`{n}`" not in guide]
    assert not missing, missing
