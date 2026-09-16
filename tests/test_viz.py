"""Viz smoke tests: responsive chart constructs and (when available) actually resamples."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.viz import ResearchChart, equity_chart, price_chart, _HAVE_RESAMPLER


def _df(n=20000):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"t": t, "open_time": t, "open": close, "high": close * 1.001,
                        "low": close * 0.999, "close": close, "volume": rng.random(n)})


def test_research_chart_builds_and_resamples():
    df = _df()
    ch = ResearchChart(df, candles=True)
    ch.add_ema(50)
    fig = ch.show()
    assert len(fig.data) >= 2  # close line + ema + candles
    if _HAVE_RESAMPLER:
        # line traces should be decimated to far fewer than n points
        line_lens = [len(t.x) for t in fig.data if t.type in ("scattergl", "scatter") and t.x is not None]
        assert line_lens and max(line_lens) < len(df)


def test_price_chart_with_trades():
    df = _df(5000)
    trades = pd.DataFrame({
        "side": ["long", "long"], "entry_time": [df["t"].iloc[100], df["t"].iloc[200]],
        "exit_time": [df["t"].iloc[150], df["t"].iloc[250]],
        "entry_price": [df["close"].iloc[100], df["close"].iloc[200]],
        "exit_price": [df["close"].iloc[150], df["close"].iloc[250]],
        "pnl": [5.0, -3.0],
    })
    fig = price_chart(df, emas=[20, 50], trades=trades, candles=True)
    assert len(fig.data) >= 3


def test_equity_chart_builds():
    df = _df(8000)
    eq = pd.DataFrame({"t": df["t"], "equity": np.linspace(10000, 11000, len(df)),
                      "drawdown": np.zeros(len(df))})
    fig = equity_chart(eq)
    assert len(fig.data) >= 1


def _mtf_fixture():
    import numpy as np, pandas as pd
    n = 900
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": idx, "t": idx, "open": np.r_[close[0], close[:-1]],
                         "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0})


def test_mtf_ema_lines_drawn_whether_or_not_the_frame_already_has_them():
    """Regression: when the scenario's `mtf_emas` param already built `{tf}__ema_{p}`, the chart used
    to re-run build_mtf, hit pandas' _x/_y suffixing, and silently draw NOTHING."""
    import numpy as np
    from quant import scenarios as sc
    from quant.viz import scenario_figure
    df = _mtf_fixture()
    base = {"grad_max": None, "prev_low_n": 20, "confirm_red": 2}
    for extra in ({}, {"mtf_emas": ("5min", [20, 50])}):
        run = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                            params={**base, **extra}, cfg={"spread": 0.0})
        fig = scenario_figure(run, mtf_ema=("5min", [20, 50]))
        lines = [t for t in fig.data if t.name and "5min" in t.name]
        assert len(lines) == 2, f"expected 2 HTF EMA lines, got {[t.name for t in lines]}"
        for t in lines:
            v = np.asarray(t.y, dtype=float)
            assert np.isfinite(v).sum() > 100, f"{t.name} is empty/all-NaN"


def test_mtf_ema_missing_column_raises_instead_of_silently_skipping():
    """A column that cannot be built must be loud, not an invisible line."""
    import pytest as _pytest
    from quant.viz.studies import _add_mtf_emas
    import plotly.graph_objects as go
    df = _mtf_fixture()
    fig = go.Figure()
    with _pytest.raises(Exception):
        _add_mtf_emas(fig, df.drop(columns=["open_time"]), df["t"], ("5min", 20), None)


# ---------------------------------------------------------------------------
# S/R ZONES, PER TIMEFRAME
#
# A level clustered from 1h pivots is a different level from one clustered from 5m pivots, so each
# frame is clustered in its OWN bar space and only becomes a chart coordinate when a bar index
# turns into a timestamp. These check that the layers stay separate, that the chart's own layer is
# untouched by the addition, and that the controls exist.
# ---------------------------------------------------------------------------
def _ohlc(n=9000, seed=4, freq="5min"):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    close = 30000 + 900 * np.sin(2 * np.pi * i / 400) + np.cumsum(rng.normal(0, 6, n))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 14, n))
    t = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_,
                         "high": np.maximum(open_, close) + wick,
                         "low": np.minimum(open_, close) - wick, "close": close, "volume": 1.0})


def _sr_traces(fig):
    return [t for t in fig.data if "S/R" in (t.name or "")]


def test_each_timeframe_gets_its_own_sr_layer():
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc()
    frames = {"15m": resample_ohlcv(base, "15min"), "1h": resample_ohlcv(base, "60min")}
    fig = pattern_figure(base, patterns=[], sr_zones=True, sr_frames=frames, chart_tf="5m",
                         emas=())
    names = [t.name for t in _sr_traces(fig)]
    assert any(n.startswith("15m") for n in names), names
    assert any(n.startswith("1h") for n in names), names
    assert any(n == n.lstrip("0123456789hm ") and "·" not in n for n in names), \
        "the chart's own layer is missing"


def test_weight_encodes_the_timeframe():
    """Colour is already spent on direction elsewhere on this chart, so S/R uses WEIGHT — the same
    convention the trendlines use. The coarsest layer must be the thickest, and no two may tie."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc()
    frames = {"15m": resample_ohlcv(base, "15min"), "1h": resample_ohlcv(base, "60min"),
              "4h": resample_ohlcv(base, "240min")}
    fig = pattern_figure(base, patterns=[], sr_zones=True, sr_frames=frames, chart_tf="5m",
                         emas=())
    w = {t.name.split(" ·")[0]: t.line.width for t in _sr_traces(fig) if t.line.width}
    assert len(set(w.values())) == len(w), f"two layers share a width: {w}"
    assert w["4h"] > w["1h"] > w["15m"], w


def test_adding_sr_layers_leaves_the_chart_own_layer_alone():
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc()
    one = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=())
    many = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=(),
                          sr_frames={"1h": resample_ohlcv(base, "60min")})
    own_one = [t for t in _sr_traces(one) if "·" not in t.name]
    own_many = [t for t in _sr_traces(many) if "·" not in t.name]
    assert own_one and own_many
    assert own_one[0].name == own_many[0].name, "the chart's own zone count changed"
    assert own_one[0].line.color == own_many[0].line.color
    assert list(own_one[0].y) == list(own_many[0].y), "the chart's own levels moved"


def test_the_chart_carries_an_sr_timeframe_button_row():
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc()
    fig = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=(),
                         sr_frames={"1h": resample_ohlcv(base, "60min")})
    rows = [m for m in (fig.layout.updatemenus or ()) if m.type == "buttons"
            and any("S/R" in str(b.label) for b in m.buttons)]
    assert len(rows) == 1, "no S/R timeframe row"
    labels = [b.label for b in rows[0].buttons]
    assert labels[0] == "S/R: all" and labels[-1] == "none" and "1h" in labels, labels
    # isolating a layer needs a PER-TRACE visibility list; one style value could not express it
    iso = rows[0].buttons[1]
    assert isinstance(iso.args[0]["visible"], list)
    assert len(iso.args[0]["visible"]) == len(iso.args[1])
    assert any(v is True for v in iso.args[0]["visible"])
    assert any(v != True for v in iso.args[0]["visible"]), "the button hid nothing"


def test_a_zone_is_named_and_hoverable():
    """`5m·Z3`, the same shape as `5m·S7` for a trendline — one name the hover, the legend and the
    dropdown all share. And hover lives on a midpoint marker too, because a horizontal line is a
    one-pixel target."""
    from quant.viz.studies import pattern_figure

    base = _ohlc()
    fig = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=())
    hov = [t for t in fig.data if t.customdata is not None
           and any("S/R" in str(c) for c in t.customdata)]
    assert hov, "no S/R hover anywhere"
    assert any("·Z" in str(c) for t in hov for c in t.customdata)
    assert any(t.mode == "markers" for t in hov), "no midpoint marker to hover"


def test_no_sr_frames_means_exactly_the_old_behaviour():
    from quant.viz.studies import pattern_figure

    base = _ohlc()
    off = pattern_figure(base, patterns=[], sr_zones=False, chart_tf="5m", emas=())
    on = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=())
    sr_ix = {i for i, t in enumerate(on.data)
             if "S/R" in (t.name or "") or "S/R" in (t.legendgroup or "")
             or "·Z" in (t.name or "")}
    sr_ix |= {i + 1 for i in sr_ix if i + 1 < len(on.data)
              and (on.data[i + 1].legendgroup or "") == (on.data[i].legendgroup or "")}
    rest = [t.name for i, t in enumerate(on.data) if i not in sr_ix]
    assert rest == [t.name for t in off.data]


# ---------------------------------------------------------------------------
# CHART CONTROLS: what autoscale may do, what a picker button may touch, and
# how wide a menu label may be. Each of these was a visible bug first.
# ---------------------------------------------------------------------------
def test_autoscale_cannot_escape_the_window():
    """`range` only sets the OPENING view; the autoscale button rescales to the DATA. A trendline
    fitted through pivots before the window, or a gap born in the warm-up and still open, dragged
    the chart back months. `minallowed` bounds the axis itself, so no layer can pull the view off
    the period that was asked for -- and no geometry has to be clipped to achieve it, so a line
    that genuinely starts earlier keeps its real slope."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc(n=6000)
    view = str(base["t"].iloc[4000])
    fig = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=(),
                         fvgs=True, view_start=view,
                         sr_frames={"1h": resample_ohlcv(base, "60min")})
    floor = fig.layout.xaxis.autorangeoptions.minallowed
    assert floor is not None, "autoscale is unbounded on the left"
    # epoch MILLISECONDS -- see test_the_autorange_floor_is_epoch_ms_not_a_date_string
    assert pd.Timestamp(floor, unit="ms", tz="UTC") == pd.Timestamp(view)


def test_a_gap_box_starts_at_the_window_not_at_its_birth():
    from quant.viz.studies import pattern_figure

    base = _ohlc(n=6000)
    view = pd.Timestamp(str(base["t"].iloc[4000]))
    fig = pattern_figure(base, patterns=[], sr_zones=False, chart_tf="5m", emas=(),
                         fvgs=True, view_start=str(view))
    boxes = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
    assert boxes
    for t in boxes:
        xs = pd.to_datetime(pd.Series([v for v in t.x if v is not None]), utc=True).dropna()
        assert xs.min() >= view, f"{t.name} is drawn from before the window"


def test_no_picker_button_touches_a_background_layer():
    """Choosing a trade used to re-show every gap and level you had just switched off, because each
    button carried a visibility vector as long as the whole figure. The gaps, levels and EMAs have
    their own controls; the pickers address only the traces they own."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc(n=6000)
    fig = pattern_figure(base, patterns=[], sr_zones=True, chart_tf="5m", emas=(20, 50),
                         fvgs=True, sr_frames={"1h": resample_ohlcv(base, "60min")},
                         view_start=str(base["t"].iloc[3000]))
    bg = {i for i, t in enumerate(fig.data)
          if not (t.legendgroup or "").startswith("i#")
          and ((t.legendgroup or "").startswith(("ema", "mtfema"))
               or getattr(t, "meta", None) == "fvg"
               or (t.legendgroup or "").endswith("S/R zones"))}
    assert bg, "the fixture drew no background layers — it is not testing anything"
    checked = 0
    for m in fig.layout.updatemenus:
        for b in (m.buttons or ()):
            if b.method != "update":
                continue
            checked += 1
            args = list(b.args or ())
            assert len(args) > 2, f"{b.label!r} carries a whole-figure vector"
            assert not (set(args[2]) & bg), f"{b.label!r} reaches a background trace"
    assert checked, "no update buttons found"


def test_menu_labels_cannot_grow_wide_enough_to_overlap_the_next_menu():
    """A plotly dropdown is as wide as its widest row and the menus sit at fixed x positions, so one
    60-character occurrence label pushed its menu straight over the one beside it."""
    from quant.viz.studies import _short, pattern_figure

    assert _short("x" * 80).endswith("…") and len(_short("x" * 80)) <= 34
    assert _short("short one") == "short one"

    base = _ohlc(n=6000)
    fig = pattern_figure(base, patterns=None, sr_zones=True, chart_tf="5m", emas=(),
                         view_start=str(base["t"].iloc[3000]))
    rows = []
    for m in fig.layout.updatemenus:
        for b in (m.buttons or ()):
            rows.append(str(b.label))
            args = list(b.args or ())
            lay = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
            for k in ("updatemenus[1].buttons", "updatemenus[2].buttons"):
                rows += [str(c["label"]) for c in lay.get(k, [])]
    assert rows
    longest = max(rows, key=len)
    assert len(longest) <= 42, f"{longest!r} is {len(longest)} chars"


def test_an_empty_pattern_list_means_none_not_all():
    """`patterns=None` is every pattern; `patterns=[]` used to mean the same, so every notebook
    asking for trendlines only was detecting and drawing the whole 22-pattern library -- most of the
    trace count, most of the dropdown rows, and the slowest call in the figure."""
    from quant.viz.studies import pattern_figure

    base = _ohlc(n=6000)
    kw = dict(sr_zones=False, trendlines=True, chart_tf="5m", emas=())
    none = pattern_figure(base, patterns=[], **kw)
    every = pattern_figure(base, patterns=None, **kw)

    # tested against the actual pattern names, not a guess at what a pattern trace looks like --
    # the trendline instances are also named (`5m·S0`) and a loose filter catches those instead
    from quant.indicators import CHART_PATTERNS
    # FULL pattern names, not their words -- "support trendline" shares a word with a pattern and
    # a word-level filter flagged the trendline layer as a pattern
    names = {str(n).replace("_", " ") for n in CHART_PATTERNS}

    def _pats(fig):
        return [str(t.name) for t in fig.data
                if t.name and any(n in str(t.name) for n in names)]

    assert _pats(none) == [], _pats(none)[:5]
    assert _pats(every), "the fixture produced no patterns — it is not testing anything"
    assert len(none.data) < len(every.data)


def test_the_autorange_floor_is_epoch_ms_not_a_date_string():
    """plotly.js cannot compare a date STRING against a date axis here. Every string form
    ("2026-06-01", with or without a time or a "+00:00" offset) made autorange yield nothing and the
    axis fell back to its EMPTY default of 2000-01-01..2001-01-01 -- so pressing Autoscale threw the
    chart 26 years into the past. Measured in a real browser; the number works, the string does not."""
    from quant.viz.studies import pattern_figure

    base = _ohlc(n=6000)
    view = pd.Timestamp(str(base["t"].iloc[3000]))
    fig = pattern_figure(base, patterns=[], sr_zones=False, chart_tf="5m", emas=(),
                         view_start=str(view))
    floor = fig.layout.xaxis.autorangeoptions.minallowed
    assert isinstance(floor, int), f"{floor!r} is {type(floor).__name__}, not epoch ms"
    assert floor == view.value // 1_000_000


def test_a_timeframe_button_row_opens_on_what_is_actually_drawn():
    """A row highlighting "all" while every layer is legendonly is lying about the chart."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base = _ohlc(n=6000)
    kw = dict(patterns=[], chart_tf="5m", emas=(), fvgs=True, sr_zones=True,
              view_start=str(base["t"].iloc[3000]),
              fvg_frames={"1h": resample_ohlcv(base, "60min")},
              sr_frames={"1h": resample_ohlcv(base, "60min")})

    def rows(fig):
        return {m.buttons[0].label: m.buttons[m.active].label
                for m in (fig.layout.updatemenus or ())
                if m.type == "buttons" and m.active is not None and m.active >= 0
                and str(m.buttons[0].label).endswith(": all")}

    off = rows(pattern_figure(base, start_visible=(), **kw))
    assert off and all(v == "none" for v in off.values()), off

    on = rows(pattern_figure(base, start_visible=("fvg_bull", "fvg_bear", "sr_zones"), **kw))
    # the chart's own layers are on but the 1h ones are not, so neither end is true
    assert all(v != "all" for v in on.values()), on
