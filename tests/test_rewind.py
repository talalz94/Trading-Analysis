"""
REWIND — the point-in-time view, and the two ways of getting one.

The browser tool is JavaScript and is verified by driving a real chart (done once, by hand, in a
browser: 345 rewindable traces, 0 left past the cut, candles and EMAs still running, "live"
restoring exact point counts). What these tests hold is the PYTHON CONTRACT the tool depends on —
which traces are offered to it, what the flag changes about the drawing, and that `AS_OF` is a
genuine truncation rather than a filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.viz import AS_OF_JS
from quant.viz.studies import pattern_figure


def _ohlc(n=6000, seed=4, freq="5min"):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    close = 30000 + 900 * np.sin(2 * np.pi * i / 400) + np.cumsum(rng.normal(0, 6, n))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 14, n))
    t = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_,
                         "high": np.maximum(open_, close) + wick,
                         "low": np.minimum(open_, close) - wick, "close": close, "volume": 1.0})


def _fig(base, **kw):
    from quant.data import resample_ohlcv
    return pattern_figure(base, patterns=[], sr_zones=True, trendlines=True, fvgs=True,
                          chart_tf="5m", emas=(50,), view_start=str(base["t"].iloc[3000]),
                          sr_frames={"1h": resample_ohlcv(base, "60min")},
                          fvg_frames={"1h": resample_ohlcv(base, "60min")}, **kw)


# ---------------------------------------------------------------------------
# what the flag offers the tool
# ---------------------------------------------------------------------------
def test_rewind_off_changes_nothing():
    """Notebook 19 must behave exactly as it did, so the flag has to be inert when off."""
    base = _ohlc()
    off = _fig(base)
    assert off.layout.meta is None
    on = _fig(base, rewind=True)
    assert (on.layout.meta or {}).get("rewind"), "nothing offered to the tool"


def test_everything_with_a_birth_is_rewindable_and_the_clock_is_not():
    """A deny-list, not an allow-list: naming the rewindable kinds missed the per-trade traces,
    which carry neither a name nor a legendgroup. Two exclusions, both deliberate — the candles are
    the clock (you want to see what happened next) and an EMA is context with no birth."""
    base = _ohlc()
    fig = _fig(base, rewind=True)
    rw = set(fig.layout.meta["rewind"])
    for i, t in enumerate(fig.data):
        g = t.legendgroup or ""
        excluded = t.type == "candlestick" or g.startswith(("ema", "mtfema"))
        assert (i not in rw) == excluded, f"trace {i} ({t.name!r}, {g!r}) on the wrong side"
    assert len(rw) > 20


def test_the_indices_are_valid_and_carry_clippable_geometry():
    base = _ohlc()
    fig = _fig(base, rewind=True)
    rw = fig.layout.meta["rewind"]
    assert len(set(rw)) == len(rw), "duplicate indices"
    assert all(0 <= i < len(fig.data) for i in rw)
    # every one must have an x the clip can walk
    assert all(getattr(fig.data[i], "x", None) is not None for i in rw)


# ---------------------------------------------------------------------------
# the one thing rewinding changes about the drawing
# ---------------------------------------------------------------------------
def test_rewind_keeps_every_sr_revision_instead_of_only_the_last():
    """A zone's level is nudged as later pivots merge into it, and the normal chart draws only the
    FINAL one -- today's level painted across the whole window, which is a quiet lookahead when the
    point is to see what was known earlier. Keeping each revision lets the clip pick the right one
    with no special-casing."""
    base = _ohlc()
    off, on = _fig(base), _fig(base, rewind=True)

    def segs(fig):
        return {t.name.split(" (")[0]: sum(1 for v in t.x if v is None)
                for t in fig.data if t.name and "S/R zones" in t.name}

    a, b = segs(off), segs(on)
    assert a and b
    assert any(b[k] > a[k] for k in a), f"no extra revisions kept: {a} vs {b}"
    assert all(b[k] >= a[k] for k in a), "rewind lost a zone"


def test_a_revision_stops_where_the_next_one_starts():
    """Only the last revision may run to the right edge; an earlier one ending there would draw
    several levels for one zone across the same bars."""
    base = _ohlc()
    fig = _fig(base, rewind=True)
    t = [x for x in fig.data if x.name and x.name.startswith("S/R zones")][0]
    xs = list(t.x)
    ends = [pd.Timestamp(str(xs[i + 1])) for i in range(0, len(xs) - 2, 3)]
    right = max(ends)
    assert sum(1 for e in ends if e == right) < len(ends), "every revision runs to the right edge"


# ---------------------------------------------------------------------------
# AS_OF: a truncation, not a filter
# ---------------------------------------------------------------------------
def test_as_of_stops_the_data_rather_than_hiding_it():
    """AS_OF is an earlier END, not a filter over a full run -- the detectors never see past it, so
    there is nothing to disbelieve. Checked on the source because the alternative is a live fetch."""
    import importlib
    import inspect

    # `quant.research` re-exports the FUNCTION `fvg_lab`, which shadows the submodule of the same
    # name, so `import quant.research.fvg_lab as fl` hands back the function
    fl = importlib.import_module("quant.research.fvg_lab")

    assert fl._get(object(), "AS_OF", "fallback") == "fallback"
    src = inspect.getsource(fl.fvg_lab)
    assert 'end = _get(c, "AS_OF", None) or c.END' in src
    assert "end=c.END" not in src, "a loader call still pins the real END, so AS_OF would leak"
    assert src.count("end=end") >= 3, "not every frame load follows the truncated end"


# ---------------------------------------------------------------------------
# the script itself
# ---------------------------------------------------------------------------
def test_the_script_reads_the_figure_and_bails_out_when_there_is_nothing_to_do():
    assert "layout.meta" in AS_OF_JS and "m.rewind" in AS_OF_JS
    assert "if (!rewindable().length) return;" in AS_OF_JS, \
        "the script must be inert on a figure built without rewind=True"
    # plotly.py 6 base64-encodes numeric columns; reading gd.data[i].y directly threw on the trade
    # markers, so the decode has to go through _fullData
    assert "_fullData" in AS_OF_JS and "_inputArray" in AS_OF_JS


def test_the_script_is_small_enough_to_inline_everywhere():
    assert len(AS_OF_JS) < 12000, f"{len(AS_OF_JS)} bytes of injected script"


def test_the_cut_comes_from_the_points_own_data_value_not_the_formatted_click():
    """The bug this replaces, found by clicking a real chart on a UTC+5 machine: plotly hands the
    click back a formatted, TIMEZONE-NAIVE string ("2026-06-06 06:10"), and `new Date()` reads that
    as LOCAL time -- so the chart was cut five hours early and everything in between silently
    vanished instead of being clipped. The stored value carries its offset and parses correctly."""
    assert "fd.x[p.pointIndex]" in AS_OF_JS, "the cut must come from the stored x, not ev.points[].x"
    assert "curveNumber" in AS_OF_JS
    # the marker has the same problem in reverse: plotly renders a Date in the browser's zone
    assert "x0: rawX, x1: rawX" in AS_OF_JS, "the as-of line must be drawn at the raw x"
    assert "new Date(cut), x1: new Date(cut)" not in AS_OF_JS
    assert "UTC" in AS_OF_JS, "the readout should say which zone it is quoting"


def test_a_gap_box_is_clamped_not_truncated():
    """The bug: a gap is the closed polygon [x0,x1,x1,x0,x0] x [bot,bot,top,top,bot], and clipping
    it like a polyline keeps only the first edge -- the bottom -- so every rewound gap collapsed
    into a flat line. Clamping each vertex back to the cut keeps a box of the gap's REAL height and
    only shortens it, which is what the tool is for: the height never depends on where you click."""
    assert "function clipBox(" in AS_OF_JS
    assert "o.box ? clipBox(" in AS_OF_JS, "the two clips must be dispatched on the trace's fill"
    # read from gd.data, NOT _fullData: plotly only computes full defaults for traces it has
    # drawn, and every isolatable gap is built visible:false -- so _fullData[i].fill was undefined
    # for all of them and each one rewound into a degenerate sliver
    assert "spec.fill === 'toself'" in AS_OF_JS, "a filled polygon has to be recognised as one"
    assert "var spec = gd.data[i]" in AS_OF_JS, "fill must come from the trace spec"
    # the line path must still INTERPOLATE, or a clipped trendline would tilt
    assert "q.y + f * (p.y - q.y)" in AS_OF_JS
