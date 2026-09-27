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


@pytest.mark.parametrize("sr", [dict(sr_sensitivity=None), dict(sr_sensitivity="loose")],
                         ids=["legacy-engine", "tuned-engine"])
def test_a_revision_stops_where_the_next_one_starts(sr):
    """Only the last revision may run to the right edge; an earlier one ending there would draw
    several levels for one zone across the same bars.

    Run on BOTH zone engines. The tuned "normal" draws nothing on this synthetic sine wave -- 4
    touches inside a 0.08% band is a strict ask of smooth data -- so the test named no settings,
    found no S/R trace and failed on an empty list when the defaults changed. The property belongs
    to the rewind drawing, not to any one setting, so it must hold for each engine."""
    base = _ohlc()
    fig = _fig(base, rewind=True, **sr)
    found = [x for x in fig.data if x.name and x.name.startswith("S/R zones")]
    assert found, "the fixture must draw an S/R layer or this test checks nothing"
    t = found[0]
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
    """A budget on the CODE, not on the prose.

    The point of the cap is that the tool stays a script and never grows into a bundle. Counting
    comments against it just taxes the explanations, and every one of them here is a browser trap
    that cost a real debugging session to find -- which is exactly the kind of thing that should be
    written down beside the line it explains.
    """
    import re

    code = re.sub(r"^\s*//.*$", "", AS_OF_JS, flags=re.M)
    code = re.sub(r"\n\s*\n", "\n", code)
    assert len(code) < 11000, f"{len(code)} bytes of actual code"
    assert len(AS_OF_JS) < 18000, f"{len(AS_OF_JS)} bytes injected"


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


def test_the_geometry_is_read_from_the_trace_spec_and_the_hidden_traces_carry_it():
    """The bug behind "after a rewind I cannot see any trade artefacts again".

    plotly expands `_fullData`, and attaches the decoded `_inputArray` to a base64 column, only for
    traces it has actually DRAWN. Every isolatable gap and every per-trade entry/stop/target group
    is built `visible=False`, so for exactly the traces the pickers exist to show there is no
    readable geometry there at all. Snapshotting through `_fullData` therefore stored EMPTY arrays
    for them, and because `live` restores from the snapshot it wrote that emptiness back for good:
    one rewind, and every trade the dropdown could offer drew nothing for the rest of the session.

    The spec is the one copy that is always present, so the tool reads `gd.data` first and decodes
    plotly.py's base64 itself. This holds the PYTHON half of that contract — that the hidden traces
    really do carry their geometry in the spec.
    """
    base = _ohlc()
    fig = _fig(base, rewind=True)
    rw = fig.layout.meta["rewind"]

    hidden = [i for i in rw if fig.data[i].visible is False]
    assert hidden, "no visible=False traces built — this test is not exercising the case"
    for i in hidden:
        t = fig.data[i]
        assert t.x is not None and len(t.x), f"trace {i} ({t.name!r}) has no x in the spec"
        assert t.y is not None and len(t.y), f"trace {i} ({t.name!r}) has no y in the spec"

    assert "var spec = gd.data[i]" in AS_OF_JS, "the spec must be read first"
    assert "gd._fullData && gd._fullData[i]) || null" in AS_OF_JS, "_fullData is only the fallback"
    assert "function pick(spec, full, key)" in AS_OF_JS
    # and it must decode the base64 itself, since _inputArray is absent for the same traces
    assert "function fromB64(" in AS_OF_JS and "atob(" in AS_OF_JS
    assert "if (v.bdata)" in AS_OF_JS, "a raw {dtype, bdata} column has to be decodable"
    # a trace it could not read must be left alone rather than blanked
    assert "if (!x.length) { return; }" in AS_OF_JS
    assert "filter(function (i) { return !!ORIG[i]; })" in AS_OF_JS


def test_the_cut_is_written_as_a_string_never_as_a_date_object():
    """The bug that made a rewound gap run PAST the yellow line, on a machine that is not UTC.

    The clipped vertices were `new Date(cut)`. plotly formats a Date in the BROWSER'S timezone,
    while every other x on the trace is an ISO string carrying "+00:00" -- so on a UTC+5 machine
    each clamped edge was painted five hours to the right of the cut. A gap clipped at 07:15 was
    drawn out to 12:15: past the marker, and past its own real end.

    The trace DATA was correct the whole time (`new Date(x).getTime()` round-trips), which is why
    measuring the arrays found nothing and only looking at the rendered chart showed it. Same trap
    as the as-of line itself, one function over.
    """
    assert "new Date(cut), y: q.y" not in AS_OF_JS, "an interpolated endpoint must not be a Date"
    assert "p.t > cut ? new Date(cut)" not in AS_OF_JS, "a clamped vertex must not be a Date"
    assert "var cutX = rawX;" in AS_OF_JS, "the cut's x comes from the clicked point's own value"
    assert "cutX instanceof Date" in AS_OF_JS, "and is stringified if plotly handed back a Date"
    assert "clipBox(o.x, o.y, o.cd, cut, cutX)" in AS_OF_JS
    assert AS_OF_JS.count("new Date(cut).toISOString()") == 1, "the fallback must be explicit UTC"


def test_a_naive_timestamp_is_read_as_utc_the_way_plotly_reads_it():
    """Not every column plotly is given is in the same format, which is the part that bites.

    The candles carry "+00:00"; the trade-marker traces are TIMEZONE-NAIVE ("2026-06-06 08:25").
    `new Date()` reads a naive string as LOCAL, so on a UTC+5 machine an 08:25 entry compared as
    03:25 -- it sorted BEFORE a 07:15 cut, and a trade that had not happened yet stayed on the
    chart. plotly.js reads a naive string as UTC, so the tool matches plotly, not the browser.
    """
    assert "z += 'Z';" in AS_OF_JS, "a naive string must be pinned to UTC before parsing"
    assert "Date.parse(z)" in AS_OF_JS
    assert "typeof v === 'string'" in AS_OF_JS


def test_a_marker_is_truncated_never_interpolated():
    """A marker is a fact, not a path -- and interpolating one INVENTED A TRADE.

    Every long entry lives in a single marker trace, so the generic line clip drew a path between
    two entries and put a point on the cut wherever a later entry existed: at 07:15 the chart showed
    a "trade" at 61,633, a price no trade was ever taken at, beside a real entry from 08:25 that had
    not happened yet. There is nothing to interpolate along, so the points after the cut simply go.
    """
    assert "function clipPts(" in AS_OF_JS
    assert "o.pts ? clipPts(" in AS_OF_JS, "marker traces must be dispatched to their own clip"
    assert "mode.indexOf('markers') >= 0 && mode.indexOf('lines') < 0" in AS_OF_JS
    # clipPts must not interpolate: no y arithmetic anywhere in it
    body = AS_OF_JS[AS_OF_JS.index("function clipPts("):AS_OF_JS.index("function clip(")]
    assert "q.y" not in body and "f *" not in body, "clipPts must not compute a y"
