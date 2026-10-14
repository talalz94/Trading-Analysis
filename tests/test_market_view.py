"""
The market view (quant/research/market_view.py, notebooks/market_view.ipynb) -- on a fake market
with a known clock, so every test states which candles existed. The promises: the window is the last
DAYS of TRADING counted in candles and ends at the latest CLOSED one; a candle still forming is
dropped from every timeframe (a line fitted on it would repaint); a shut weekend makes the window
reach back past it rather than show it empty; settings that cannot mean what they say are refused
by name; OPEN_WITH switches on exactly the layers it names; the chart has every timeframe's layers,
no trades, and the news strip; the notebook states and defines nothing; and another machine's feed
log merges into this one without losing or doubling an observation.
"""
from __future__ import annotations

import ast
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quant.data as qdata
import quant.data.loader as qloader
from quant.data.timeframe import tf_minutes
from quant.research import market_view
from quant.research.market_view import instrument, open_with
from quant.research.workspace import NOTEBOOKS, market_view_config
from test_notebook_config import SETTING_RE

NOW = pd.Timestamp("2026-10-14 21:30:20", tz="UTC")     # 20 s into the 21:30 candle


def cfg(**kw):
    base = dict(INSTRUMENT="X", INSTRUMENTS={"X": dict(symbol="XUSD", source="fake")},
                TF="1m", DAYS=0.5, TF_WARMUP_BARS=120,
                SHOW_TRENDLINES=True, LINE_TFS=("1m", "5m", "15m"),
                FVG_DRAW=True, FVG_TFS=("1m", "5m", "15m"),
                SHOW_SR_ZONES=True, SR_TFS=("1m", "15m"),
                TRENDLINE=dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                               min_span_bars=40, max_age_bars=500, break_sustain_bars=1),
                FVG_MITIGATION="ce", FVG_MIN_SIZE_ATR=0.5, FVG_MAX_AGE_BARS=1000,
                FVG_ATR_PERIOD=14, NEWS=None, NEWS_UPDATE=False,
                OPEN_WITH=dict(trendlines=("15m",), gaps=("5m", "15m"), zones=("15m",)),
                EMAS=(), REWIND=False, HEIGHT=600)
    base.update(kw)
    return SimpleNamespace(**base)


def market(now=NOW, shut=()):
    """Every 1m candle from 10 days before `now` up to the one forming at `now`, minus the `shut`
    spans; coarser candles resampled from them. The forming candle IS returned, as the real
    sources return it."""
    t = pd.date_range(now.floor("1min") - pd.Timedelta(days=10), now.floor("1min"), freq="1min")
    for a, b in shut:
        t = t[(t < pd.Timestamp(a, tz="UTC")) | (t >= pd.Timestamp(b, tz="UTC"))]
    c = 100 + np.random.default_rng(7).normal(0, 0.05, len(t)).cumsum()
    o = np.r_[c[0], c[:-1]]
    m1 = pd.DataFrame({"open_time": t, "open": o, "high": np.maximum(o, c) + 0.03,
                       "low": np.minimum(o, c) - 0.03, "close": c, "volume": 1.0})
    asked = []

    def get(symbol, tf, start, end=None, **kw):
        asked.append((tf, start, end))
        if tf == "1m":
            d = m1
        else:
            d = (m1.set_index("open_time").resample(f"{tf_minutes(tf)}min", label="left",
                                                     closed="left")
                 .agg(dict(open="first", high="max", low="min", close="last", volume="sum"))
                 .dropna().reset_index())
        d = d[d["open_time"] >= pd.Timestamp(start, tz="UTC")].reset_index(drop=True)
        return d.assign(t=d["open_time"])

    return get, asked


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        get, asked = market(**kw)
        monkeypatch.setattr(qdata, "get_ohlcv", get)
        monkeypatch.setattr(qloader, "get_ohlcv", get)
        return asked
    return install


# ── the window ──────────────────────────────────────────────────────────────────────────────
def test_the_window_is_the_last_days_of_candles_ending_at_the_latest_closed_one(fake):
    fake()
    v = market_view(cfg(), now=NOW)
    assert v.latest == pd.Timestamp("2026-10-14 21:29", tz="UTC")     # 21:30 is still forming
    assert len(v.window) == 720                                         # half a day of 1m
    assert v.view_start == pd.Timestamp("2026-10-14 09:30", tz="UTC")
    assert len(v.df) == 720 + 120                                       # + the warm-up
    for tf, f in v.frames.items():                                      # every frame: closed only
        shut = f["open_time"].iloc[-1] + pd.Timedelta(minutes=tf_minutes(tf))
        assert shut <= NOW, tf
    # 21:15-21:30 closed at 21:30; the 21:30 bar is still forming, and so is the 21:00 hour
    assert v.frames["15m"]["open_time"].iloc[-1] == pd.Timestamp("2026-10-14 21:15", tz="UTC")
    assert v.frames["5m"]["open_time"].iloc[-1] == pd.Timestamp("2026-10-14 21:25", tz="UTC")


def test_a_shut_weekend_makes_the_window_reach_back_past_it(fake):
    # gold shuts Friday 21:00 to Sunday 22:00: on Monday morning the last half day of TRADING
    # starts on Friday, not on Sunday night with a few candles in it
    now = pd.Timestamp("2026-10-12 03:00:30", tz="UTC")
    fake(now=now, shut=[("2026-10-09 21:00", "2026-10-11 22:00")])
    v = market_view(cfg(), now=now)
    assert len(v.window) == 720
    assert v.view_start < pd.Timestamp("2026-10-09 21:00", tz="UTC")
    assert v.latest == pd.Timestamp("2026-10-12 02:59", tz="UTC")


# ── settings that cannot mean what they say ─────────────────────────────────────────────────
def test_settings_that_cannot_mean_what_they_say_are_refused_by_name(fake):
    fake()
    with pytest.raises(KeyError, match="NOPE.*X"):
        market_view(cfg(INSTRUMENT="NOPE"), now=NOW)
    with pytest.raises(ValueError, match="FVG_TFS.*'1m'"):
        market_view(cfg(FVG_TFS=("5m", "15m")), now=NOW)     # the 1m layer is drawn regardless
    market_view(cfg(FVG_DRAW=False, FVG_TFS=("5m",)), now=NOW)  # ...unless the feature is off
    with pytest.raises(KeyError, match="lines"):
        open_with(cfg(OPEN_WITH=dict(lines=("1h",))))
    with pytest.raises(Exception, match="7x"):
        market_view(cfg(LINE_TFS=("1m", "7x")), now=NOW)


def test_open_with_names_each_layer_the_way_the_chart_does():
    keys = set(open_with(cfg(OPEN_WITH=dict(trendlines=("1m", "1h"), gaps=("1m", "4h"),
                                            zones=("1m", "15m")))))
    assert keys == {"support_trendline", "resistance_trendline",
                    "1h::support trendline", "1h::resistance trendline",
                    "fvg_bull", "fvg_bear", "4h::FVG bullish", "4h::FVG bearish",
                    "sr_zones", "15m::S/R zones"}


# ── the chart ───────────────────────────────────────────────────────────────────────────────
def test_the_chart_has_every_layer_no_trades_and_opens_on_what_open_with_names(fake, monkeypatch):
    fake()
    import quant.news as qnews
    news = pd.DataFrame([dict(t=pd.Timestamp("2026-10-14 12:30", tz="UTC"), time_q="official",
                              currency="USD", title="CPI m/m", impact="high", level="critical",
                              actual="0.6%", forecast="0.6%", previous="0.4%",
                              sentiment="as expected", surprise_z=np.nan)])
    monkeypatch.setattr(qnews, "load_events", lambda *a, **k: news)
    v = market_view(cfg(NEWS={"USD": "high"}), now=NOW)
    fig = v.figure()
    groups = {tr.legendgroup for tr in fig.data if tr.legendgroup}
    for want in ("FVG bullish", "5m::FVG bullish", "15m::FVG bearish", "S/R zones",
                 "news:critical"):
        assert want in groups, want
    assert not any("trade" in (tr.name or "").lower() for tr in fig.data)
    on = {tr.legendgroup for tr in fig.data if tr.visible is True and tr.legendgroup
          and not tr.legendgroup.startswith(("i#", "news"))}
    assert on <= {"15m::support trendline", "15m::resistance trendline", "5m::FVG bullish",
                  "5m::FVG bearish", "15m::FVG bullish", "15m::FVG bearish", "15m::S/R zones"}
    assert {"5m::FVG bullish", "15m::FVG bearish"} <= on
    assert "XUSD" in fig.layout.title.text and "last 0.5 days" in fig.layout.title.text


def test_each_layer_gets_exactly_the_timeframes_its_list_names(fake, monkeypatch):
    # random prices form few zones, so the WIRING is checked: which frames go to which layer
    import quant.viz as qviz
    fake()
    seen = {}
    monkeypatch.setattr(qviz, "pattern_figure", lambda df, **kw: seen.update(kw) or "fig")
    v = market_view(cfg(LINE_TFS=("1m", "5m", "1h"), FVG_TFS=("1m", "15m"),
                        SR_TFS=("1m", "15m")), now=NOW)
    v.figure()
    assert set(seen["tf_frames"]) == {"5m", "1h"}                       # fitted for lines
    assert seen["tf_fvgs"] == {"5m": False, "1h": False}                # ...but no gaps of theirs
    assert set(seen["fvg_frames"]) == {"15m"}
    assert set(seen["sr_frames"]) == {"15m"}
    assert seen.get("trades") is None and seen["patterns"] == []
    assert seen["start_visible"] == open_with(v.cfg)
    v2 = market_view(cfg(SHOW_TRENDLINES=False, LINE_TFS=(), SHOW_SR_ZONES=False), now=NOW)
    v2.figure()
    assert seen["tf_frames"] == {} and seen["sr_frames"] == {} and not seen["trendlines"]


def test_the_shipped_config_names_real_instruments():
    from quant.data.dukascopy import _instrument
    c = market_view_config()
    assert c.INSTRUMENT in c.INSTRUMENTS
    for name, spec in c.INSTRUMENTS.items():
        assert spec["source"] in ("binance", "dukascopy"), name
        if spec["source"] == "dukascopy":
            _instrument(spec["symbol"])                       # an unknown code raises here
    assert instrument(c)[0] == c.INSTRUMENT
    for need in ("NASDAQ", "GOLD", "USDJPY", "BTC"):
        assert need in c.INSTRUMENTS
    for lst in ("LINE_TFS", "FVG_TFS", "SR_TFS"):
        assert getattr(c, lst) == ("1m", "5m", "15m", "1h", "4h")
    assert c.TF == "1m" and c.DAYS == 2 and c.NEWS


# ── the notebook ────────────────────────────────────────────────────────────────────────────
def test_the_notebook_states_no_setting_and_defines_no_function():
    nb = json.loads((NOTEBOOKS / "market_view.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    bad = ["assigns: " + line.strip() for line in src.splitlines() if re.match(SETTING_RE, line)]
    bad += ["defines: " + n.name for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not bad, bad
    assert "market_view_config()" in src and "market_view(" in src
    assert "fvg_lab" not in src and "run_arms" not in src         # it never trades


# ── another machine's feed log ──────────────────────────────────────────────────────────────
def _fetch(store, when, forecast):
    from quant.news.calendar import NewsStore  # noqa: F401  (the store type, for readers)
    rows = pd.DataFrame([dict(t=pd.Timestamp("2026-10-14 12:30", tz="UTC") + pd.Timedelta(minutes=i),
                              time_q="exact", currency="USD", title=f"T{i}", impact="high",
                              actual=None, forecast=forecast, previous="0.1%", source="ff_feed")
                         for i in range(25)])
    store.log_feed(rows, pd.Timestamp(when, tz="UTC"))


def test_another_machines_feed_log_merges_without_losing_or_doubling(tmp_path):
    from quant.news.calendar import NewsStore
    pc, cloud = NewsStore(tmp_path / "pc"), NewsStore(tmp_path / "cloud")
    _fetch(pc, "2026-10-12 08:00", "0.2%")
    _fetch(cloud, "2026-10-12 08:00", "0.2%")                 # the same observation, both sides
    _fetch(cloud, "2026-10-13 08:00", "0.3%")                 # a revised forecast the PC missed
    n_pc = len(pc.read("feed_log"))
    added = pc.import_feed_log(cloud.read("feed_log"))
    log = pc.read("feed_log")
    assert added == 25 and len(log) == n_pc + 25
    assert pc.import_feed_log(cloud.read("feed_log")) == 0    # importing twice changes nothing
    # EVERY row of an event carries its latest sighting, as log_feed keeps it: a row left at the
    # PC's older sighting would read as "dropped from its week's feed" and be thrown away
    assert (log["last_seen"] == pd.Timestamp("2026-10-13 08:00", tz="UTC")).all()
    assert pc.meta()["feed_weeks"]["2026-10-11"] == "2026-10-13T08:00:00+00:00"
    latest = log.sort_values("fetched_at").groupby("key").tail(1)
    assert set(latest["forecast"]) == {"0.3%"}                # the newest pre-release word
