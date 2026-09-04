"""
Multi-pane study charts: price candles on top, oscillator studies below — TradingView-style,
with a crosshair (spikelines) and a toggle-able legend.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Squeeze-Momentum colours (green/yellow/red/purple — the common SQZMOM rendering)
_SQZ_COLORS = {
    "up_rising": "#00C853", "up_falling": "#C6C630",
    "down_falling": "#FF5252", "down_rising": "#9C27B0",
}
_SQZ_DOT = {"on": "#000000", "off": "#B2B5BE", "no": "#2196F3"}


# A full-height vertical crosshair that spans ALL stacked panes (Plotly's built-in x-spike only
# covers the hovered subplot). Injected into the exported HTML via `open_in_browser(extra_js=...)`.
CROSSHAIR_JS = """
setTimeout(function(){
  var gd = document.querySelector('.plotly-graph-div');
  if(!gd) return;
  gd.style.position = 'relative';
  var line = document.createElement('div');
  line.style.cssText = 'position:absolute;left:0;width:0;border-left:1px dashed #8a8a8a;'
                     + 'pointer-events:none;display:none;z-index:10;';
  gd.appendChild(line);
  gd.addEventListener('mousemove', function(e){
    var fl = gd._fullLayout || {};
    var mt = (fl.margin && fl.margin.t) || 60;
    var mb = (fl.margin && fl.margin.b) || 40;
    var r = gd.getBoundingClientRect();
    var x = e.clientX - r.left;
    if (x < ((fl.margin&&fl.margin.l)||50) || x > r.width - ((fl.margin&&fl.margin.r)||30)) {
      line.style.display='none'; return;
    }
    line.style.left = x + 'px';
    line.style.top = mt + 'px';
    line.style.height = (gd.clientHeight - mt - mb) + 'px';
    line.style.display = 'block';
  });
  gd.addEventListener('mouseleave', function(){ line.style.display='none'; });
}, 400);
"""


def _apply_crosshair(fig):
    """Horizontal spike on the hovered pane (read the value off the axis) + a shared vertical line
    across all panes supplied by CROSSHAIR_JS. The built-in x-spike is off (it can't span panes)."""
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=True, spikesnap="cursor", spikethickness=1,
                     spikedash="dot", spikecolor="#888")
    fig.update_layout(hovermode="x", spikedistance=-1, hoverdistance=100)


def squeeze_momentum_figure(df: pd.DataFrame, *, time_col: str = "t",
                            title: str = "Squeeze Momentum [LazyBear]", height: int = 800,
                            emas: Optional[Sequence[int]] = None,
                            bb_length: int = 20, bb_mult: float = 2.0, kc_length: int = 20,
                            kc_mult: float = 1.5, use_tr: bool = True):
    """Price + Squeeze-Momentum only (kept for back-compat). See `studies_figure` for +WaveTrend."""
    return studies_figure(df, time_col=time_col, title=title, height=height, emas=emas,
                          bb_length=bb_length, bb_mult=bb_mult, kc_length=kc_length, kc_mult=kc_mult,
                          use_tr=use_tr, wavetrend=False)


def studies_figure(df: pd.DataFrame, *, time_col: str = "t",
                   title: str = "Studies", height: int = 900, emas: Optional[Sequence[int]] = None,
                   squeeze: bool = True, wavetrend: bool = True,
                   bb_length: int = 20, bb_mult: float = 2.0, kc_length: int = 20, kc_mult: float = 1.5,
                   use_tr: bool = True,
                   wt_channel: int = 10, wt_average: int = 21,
                   wt_ob1: float = 60, wt_ob2: float = 53, wt_os1: float = -60, wt_os2: float = -53):
    """Price candles (+ EMAs) on top, then a Squeeze-Momentum pane and/or a WaveTrend pane below.

    Shared x-axis, a cursor crosshair, and a clickable legend (toggle any element on/off). Returns
    (fig, prepared_df).
    """
    from ..indicators import add_squeeze_momentum, add_wavetrend, ema as _ema
    d = df.reset_index(drop=True)
    if squeeze:
        d = add_squeeze_momentum(d, bb_length, bb_mult, kc_length, kc_mult, use_tr)
    if wavetrend:
        d = add_wavetrend(d, wt_channel, wt_average, wt_ob1, wt_ob2, wt_os1, wt_os2)
    x = d[time_col]

    # ---- panes ----
    titles = [""]
    heights = [0.5]
    rows_map = {}
    r = 1
    if squeeze:
        r += 1; rows_map["sqz"] = r; heights.append(0.25)
        titles.append(f"SQZMOM_LB  {bb_length} {int(bb_mult)} {kc_length} {kc_mult}")
    if wavetrend:
        r += 1; rows_map["wt"] = r; heights.append(0.25)
        titles.append(f"WT_CROSS_LB  {wt_channel} {wt_average} {int(wt_ob1)} {int(wt_ob2)} {int(wt_os1)} {int(wt_os2)}")
    # normalise heights
    tot = sum(heights); heights = [h / tot for h in heights]
    fig = make_subplots(rows=r, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.04, subplot_titles=titles)

    # ---- row 1: price ----
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price", legendgroup="price",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
                  row=1, col=1)
    for p in (emas or []):
        fig.add_trace(go.Scatter(x=x, y=_ema(d["close"], p), name=f"EMA{p}", mode="lines",
                                 line=dict(width=1)))

    # ---- Squeeze Momentum pane ----
    if squeeze:
        rr = rows_map["sqz"]
        state = d["sqz_state"].to_numpy()
        first = True
        for st, color in _SQZ_COLORS.items():
            m = state == st
            if m.any():
                fig.add_trace(go.Bar(x=x[m], y=d["sqz_mom"].to_numpy()[m], marker_color=color,
                                     marker_line_width=0, name="Squeeze Mom", legendgroup="sqz",
                                     showlegend=first), row=rr, col=1)
                first = False
        dot = np.where(d["sqz_on"], _SQZ_DOT["on"], np.where(d["sqz_off"], _SQZ_DOT["off"], _SQZ_DOT["no"]))
        fig.add_trace(go.Scatter(x=x, y=np.zeros(len(x)), mode="markers", name="Squeeze state",
                                 legendgroup="sqzdot",
                                 marker=dict(color=dot, size=4, symbol="x"),
                                 hovertext=np.where(d["sqz_on"], "squeeze ON",
                                            np.where(d["sqz_off"], "squeeze off", "no squeeze")),
                                 hoverinfo="text"), row=rr, col=1)
        fig.add_hline(y=0, line_width=1, line_color="#888", row=rr, col=1)

    # ---- WaveTrend pane ----
    if wavetrend:
        rr = rows_map["wt"]
        for lvl, col_ in [(wt_ob1, "#ef5350"), (wt_ob2, "#ef9a9a"), (0, "#888"),
                          (wt_os2, "#a5d6a7"), (wt_os1, "#26a69a")]:
            fig.add_hline(y=lvl, line_width=1,
                          line_dash=("dot" if lvl in (wt_ob2, wt_os2) else "solid"),
                          line_color=col_, row=rr, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["wt_diff"], name="WT diff", legendgroup="wtdiff",
                                 mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(33,150,243,0.35)"), row=rr, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["wt1"], name="WT1", mode="lines",
                                 line=dict(width=1.4, color="#26a69a")), row=rr, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["wt2"], name="WT2", mode="lines",
                                 line=dict(width=1.4, color="#ef5350")), row=rr, col=1)
        cu = d["wt_cross_up"].to_numpy(); cd = d["wt_cross_down"].to_numpy()
        yb = d["wt2"].to_numpy()
        fig.add_trace(go.Scatter(x=x[cu], y=yb[cu], mode="markers", name="WT cross",
                                 legendgroup="wtx", marker=dict(color="#2e7d32", size=6)),
                      row=rr, col=1)
        fig.add_trace(go.Scatter(x=x[cd], y=yb[cd], mode="markers", name="WT cross",
                                 legendgroup="wtx", showlegend=False,
                                 marker=dict(color="#c62828", size=6)), row=rr, col=1)

    fig.update_layout(title=title, template="plotly_white", height=height, dragmode="pan",
                      showlegend=True, bargap=0.15, margin=dict(l=50, r=30, t=60, b=40),
                      legend=dict(orientation="h", y=1.03, yanchor="bottom"))
    fig.update_xaxes(rangeslider_visible=False)
    _apply_crosshair(fig)
    return fig, d


# Trade-annotation colours for the scenario chart
_EXIT_STYLE = {
    "take_profit": ("#2e7d32", "star", "TP"),
    "stop_loss": ("#c62828", "x", "stop"),
    "alt_exit": ("#6a1b9a", "hexagram", "secondary bail"),
    "signal": ("#616161", "diamond", "signal exit"),
    "forced_close_end": ("#9e9e9e", "circle", "forced close"),
    "margin_call": ("#000000", "x-thin", "margin call"),
}


def _add_atr_bands(fig, d, x, line_col, mults, atr_period=14):
    """EMA ± k×ATR envelopes — the visual form of the 'entry is N ATR below the EMA' measure.
    A trade sitting outside the outer band is one that chased an already-extended move."""
    from ..indicators import add_atr
    atr = add_atr(d, atr_period)[f"atr_{int(atr_period)}"]
    base = d[line_col]
    shades = ["#90a4ae", "#78909c", "#607d8b", "#455a64", "#37474f"]
    for k, m in enumerate(mults):
        col = shades[min(k, len(shades) - 1)]
        for sign in (-1, 1):
            fig.add_trace(go.Scatter(
                x=x, y=base + sign * m * atr, mode="lines",
                name=f"EMA ±{m}·ATR", legendgroup=f"atr{m}", showlegend=(sign == -1),
                line=dict(width=0.8, color=col, dash="dot"), opacity=0.55,
                hovertemplate=f"{'-' if sign < 0 else '+'}{m} ATR: %{{y:.2f}}<extra></extra>"),
                row=1, col=1)


_MTF_EMA_SHAPES = ("('1h', 200)  |  ('1h', [50, 200])  |  [('1h', 200), ('4h', 200)]")


def _norm_mtf_ema(spec):
    """Accept any of:  ('5min', 200)  |  ('5min', [100, 200])  |  [('5min',100), ('15min',200)]
    and return a flat list of (timeframe, period) pairs.

    Every entry needs a PERIOD as well as a timeframe. A bare timeframe is rejected explicitly
    rather than recursed into: a `str` is not a tuple or list, so it used to fall through to the
    loop below, iterate its own characters, and recurse on each one forever — `["4h", "1h"]` (the
    shape `PATTERN_TFS` takes) died 3,000 frames deep in a RecursionError that named nothing.
    """
    if spec is None or (not isinstance(spec, str) and not spec):
        return []
    if isinstance(spec, str):
        raise ValueError(
            f"mtf_emas got the bare timeframe {spec!r}; each entry needs a PERIOD too. "
            f"Write one of:  {_MTF_EMA_SHAPES}   (unlike `pattern_tfs`, which is timeframes alone)")
    # a single (tf, period-or-list) pair
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        tf, per = spec
        pers = per if isinstance(per, (list, tuple)) else [per]
        out = []
        for pp in pers:
            try:
                out.append((tf, int(pp)))
            except (TypeError, ValueError):
                raise ValueError(
                    f"mtf_emas: {pp!r} is not an EMA period. Got the pair ({tf!r}, {per!r}); "
                    f"if you meant two timeframes, each needs its own period — "
                    f"{_MTF_EMA_SHAPES}") from None
        return out
    out = []
    for item in spec:
        out.extend(_norm_mtf_ema(item))
    return out


def _add_mtf_emas(fig, d, x, spec, row, atr_period=14):
    """Draw one or more HIGHER-timeframe EMAs on the base chart (line only unless `row` is given).

    If the prepared frame already carries a `{tf}__ema_{p}` column (the scenario's `mtf_emas` param
    builds the same thing so entry rules can use it), reuse it. Re-running `build_mtf` over a frame
    that already has those columns makes pandas suffix the collision to `_x`/`_y`, which used to make
    the lines silently vanish.
    """
    pairs = _norm_mtf_ema(spec)
    if not pairs:
        return
    cols = {}
    missing = {}
    for tf, per in pairs:
        col = f"{tf}__ema_{int(per)}"
        if col in d.columns:
            cols[(tf, per)] = d[col]
        else:
            missing.setdefault(tf, []).append(per)
    if missing:
        from ..data import build_mtf
        from ..indicators.overlays import add_emas
        merged = build_mtf(d, {tf: (lambda ps: (lambda z: add_emas(z, ps)))(sorted(set(ps)))
                               for tf, ps in missing.items()})
        for tf, pers in missing.items():
            for per in pers:
                col = f"{tf}__ema_{int(per)}"
                if col not in merged.columns:
                    raise KeyError(
                        f"could not build {col!r} for the chart — got {sorted(merged.columns)[-6:]}")
                cols[(tf, per)] = merged[col]

    shades = ["#1565c0", "#6a1b9a", "#00838f", "#ef6c00", "#4527a0"]
    first = None
    for k, (tf, per) in enumerate(pairs):
        series = cols[(tf, per)]
        first = first or (series, f"{tf} EMA{int(per)}")
        fig.add_trace(go.Scatter(
            x=x, y=series, mode="lines", name=f"EMA{int(per)} ({tf})",
            legendgroup=f"mtf{tf}{per}", line=dict(width=1.7, color=shades[k % len(shades)]),
            hovertemplate=f"EMA{int(per)} {tf}: %{{y:.2f}}<extra></extra>"))
    if row is not None and first:
        _add_mtf_distance_pane(fig, d, x, first[0], first[1], row, atr_period)


def _add_mtf_distance_pane(fig, d, x, line, label, row, atr_period=14):
    from ..indicators import add_atr
    atr = add_atr(d, atr_period)[f"atr_{int(atr_period)}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = ((d["close"] - line) / atr).to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=x, y=np.where(dv >= 0, dv, np.nan), mode="lines",
                             name=f"above {label}", legendgroup="mtfdist",
                             line=dict(width=1.2, color="#26a69a"), connectgaps=False),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=np.where(dv < 0, dv, np.nan), mode="lines",
                             name=f"below {label}", legendgroup="mtfdist", showlegend=False,
                             line=dict(width=1.2, color="#ef5350"), connectgaps=False),
                  row=row, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#888", row=row, col=1)


def _add_mtf_ema(fig, d, x, tf, period, row, atr_period=14):
    """A HIGHER-timeframe EMA drawn on the base chart, plus its own pane.

    Uses `build_mtf`, which shifts each HTF feature forward one HTF bar before the as-of merge — so
    the 5m EMA value shown on a 1m bar is one that was actually CLOSED and knowable at that moment.
    (Resampling and merging naively would leak the rest of the 5m candle into earlier 1m bars.)

    The pane shows price distance from that EMA in ATR units: above 0 = trading above the HTF trend,
    below 0 = under it. That is the number an MTF bias filter actually keys on.
    """
    from ..data import build_mtf
    from ..indicators import add_atr
    from ..indicators.overlays import add_emas
    col = f"{tf}__ema_{int(period)}"
    merged = build_mtf(d, {tf: lambda z: add_emas(z, [int(period)])})
    if col not in merged.columns:
        return None
    line = merged[col]
    fig.add_trace(go.Scatter(x=x, y=line, mode="lines", name=f"EMA{int(period)} ({tf})",
                             legendgroup="mtfema",
                             line=dict(width=1.8, color="#1565c0"),
                             hovertemplate=f"EMA{int(period)} {tf}: %{{y:.2f}}<extra></extra>"),
                  row=1, col=1)
    if row is None:
        return merged
    atr = add_atr(d, atr_period)[f"atr_{int(atr_period)}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = (d["close"] - line) / atr
    dv = dist.to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=x, y=np.where(dv >= 0, dv, np.nan), mode="lines",
                             name=f"above {tf} EMA", legendgroup="mtfdist",
                             line=dict(width=1.2, color="#26a69a"), connectgaps=False),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=np.where(dv < 0, dv, np.nan), mode="lines",
                             name=f"below {tf} EMA", legendgroup="mtfdist", showlegend=False,
                             line=dict(width=1.2, color="#ef5350"), connectgaps=False),
                  row=row, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#888", row=row, col=1)
    return merged


def _add_structure(fig, d, x, left, right, *, method="fractal", atr_mult=2.5):
    """HH / HL / LH / LL pivot labels + a trend ribbon (green = HH&HL, red = LH&LL, grey = range).

    `method='zigzag'` uses the ATR-scaled directional-change swings — adaptive, and a fraction of the
    fixed fractal's lag. Its labels are drawn at the TRUE extreme bar (that's where the swing is),
    while the hover shows how many bars later it was actually confirmed, so the delay stays visible.
    """
    from ..indicators import add_trend_structure, add_zigzag_structure
    if method == "zigzag":
        s = add_zigzag_structure(d, atr_mult=atr_mult)
        lab = s["zz_label"].to_numpy()
        px = s["zz_price"].to_numpy()
        piv = s["zz_pivot_i"].to_numpy()
        lg = s["zz_lag"].to_numpy()
        trend_col = "zz_trend"
    else:
        s = add_trend_structure(d, left=left, right=right)
        lab = s["struct_label"].to_numpy()
        px = s["struct_price"].to_numpy()
        piv = None
        lg = np.full(len(d), float(right))
        trend_col = "struct_trend"

    m = np.array([v is not None for v in lab])
    if m.any():
        colors = {"HH": "#2e7d32", "HL": "#66bb6a", "LH": "#ef5350", "LL": "#c62828"}
        # place the label on the bar the swing actually happened (zigzag knows it; fractal implies it)
        idx = piv[m] if piv is not None else (np.flatnonzero(m) - right)
        idx = np.clip(idx, 0, len(d) - 1)
        xs = x.iloc[idx]
        txt = [f"{v} @ {p:.2f} (confirmed +{int(l)} bars)"
               for v, p, l in zip(lab[m], px[m], np.nan_to_num(lg[m]))]
        fig.add_trace(go.Scatter(
            x=xs, y=px[m], mode="markers+text",
            name=f"Structure ({method})", legendgroup="struct",
            text=[str(v) for v in lab[m]], textposition="top center",
            textfont=dict(size=9, color=[colors.get(str(v), "#666") for v in lab[m]]),
            marker=dict(size=5, color=[colors.get(str(v), "#666") for v in lab[m]]),
            hovertext=txt, hoverinfo="text"))
    # trend ribbon along the bottom of the price pane
    tr = s[trend_col].to_numpy()
    lo = float(np.nanmin(d["low"])) if len(d) else 0.0
    span = (float(np.nanmax(d["high"])) - lo) or 1.0
    ybase = lo - span * 0.03
    for val, colr, nm in [(1, "#2e7d32", "up (HH+HL)"), (-1, "#c62828", "down (LH+LL)"),
                          (0, "#bdbdbd", "sideways")]:
        sel = tr == val
        if sel.any():
            fig.add_trace(go.Scatter(
                x=x[sel], y=np.full(int(sel.sum()), ybase), mode="markers",
                name=f"Trend: {nm}", legendgroup="trendribbon",
                marker=dict(size=3, color=colr, symbol="square"),
                hoverinfo="name"))
    return s


# ---------------------------------------------------------------------------
# Pattern overlays (toggleable from any figure)
# ---------------------------------------------------------------------------
#: higher-timeframe EMA colours — deliberately unlike the chart's own EMAs, which use the
#: template defaults, so you can tell at a glance which timeframe a line belongs to
_MTF_EMA_SHADES = ["#42a5f5", "#ab47bc", "#26c6da", "#ffa726", "#7e57c2", "#ec407a"]


def pattern_figure(df, *, patterns=None, time_col: str = "t", title=None, height: int = 820,
                   sr_zones: bool = True, zones_left: int = 10, zones_right: int = 10,
                   cluster_pct: float = 0.15, min_touches: int = 2, max_zones: int = 40,
                   extend: int = 12, start_visible=(), detect_kwargs=None, tuned: bool = False,
                   per_pattern=None, trendlines: bool = False, trendline_kwargs=None,
                   emas=(20, 50, 100, 200), view_start=None, line_picker: bool = True,
                   zone_lookback_bars: int = 2000, tf_frames=None, tf_patterns=None,
                   tf_trendlines=None, max_picker_instances: int = 60,
                   max_shapes_per_tf: int = 40, y_from_price: bool = True,
                   mtf_emas=None):
    """A clean price chart with CHART-PATTERN SHAPES and S/R zones, each toggleable on its own.

    Unlike `add_pattern_overlays`, which drops a marker at the confirmation bar, this draws the
    GEOMETRY: the pivots the shape is built from, the upper and lower trendlines through them, the
    neckline that had to break, and the measured-move target. That is what makes a triangle look like
    a triangle rather than a dot.

    Every pattern gets ONE legend entry controlling all of its drawing, so you can bring shapes in
    and out one at a time. `start_visible` names the ones that should be on when the chart opens;
    everything else starts in `legendonly`. S/R zones are their own toggle.

    `emas` overlays moving averages, one legend toggle each. It DEFAULTS to (20, 50, 100, 200)
    rather than to nothing on purpose: a caller that has not been updated still gets them. Pass
    `emas=()` for a bare chart.

    `zone_lookback_bars` is how far back S/R zones are built from, counted back from the start of
    the view. Warm-up helps most indicators and HURTS this one: the tracker keeps the most-touched
    levels, and in a trending market those are all old. Measured on BTC 5m running 66k to 82k, a
    two-month warm-up left **1 of 40 zones anywhere near the price on screen**; a 2,000-bar lookback
    leaves 39. A level from 20% ago is history, not support.

    `view_start` sets where the x-axis OPENS while leaving the data intact. Indicators that need
    history — S/R zones, trendlines, long EMAs — are only as good as the bars before them, so load
    a warm-up period before the range you care about and pass its end here. Without it the left edge
    of the chart shows zones and lines built from a handful of bars, which is not what they would
    have looked like at the time.

    `line_picker` adds a dropdown that ISOLATES one trendline at a time. Twenty overlapping lines
    under one legend toggle is unreadable, and one legend entry per line is worse; the dropdown keeps
    the legend to two entries and still lets you inspect any single line.

    `trendlines=True` adds standalone TRENDLINES — single fitted support and resistance lines with
    their touches and their break markers — on their own two legend toggles, independent of the
    shapes. `trendline_kwargs` is passed straight to `add_trendlines`.

    `tuned=True` draws every pattern at its own calibrated settings (`TUNED_SPECS`) instead of one
    shared configuration — which needs several detection passes, because those settings disagree on
    `atr_mult` and that sizes the zigzag for the whole pass. `per_pattern={name: {...}}` does the
    same with settings you supply. Both override `detect_kwargs` for pattern detection.

    MULTI-TIMEFRAME. `tf_frames={"1h": df_1h, "4h": df_4h}` draws patterns and trendlines detected
    on OTHER timeframes onto this chart. Each extra frame is detected and drawn in its OWN bar
    space — its slopes are per-bar-of-that-timeframe and its indices index that frame — and lands
    here through the only coordinate the two share, the timestamp. `quant.data.tf_frames` loads
    them with equal history in BARS, which is what detection needs. Three things make it readable:

    * **Events are stamped at the higher timeframe's bar CLOSE**, not its open. Bars carry their
      open time everywhere in this project, so a 4h pattern confirming on the bar stamped 12:00 is
      not knowable until 16:00 — drawing it at 12:00 would be a four-hour lookahead on a 15m chart.
      Geometry (line ends on the left, touch pivots) stays at the open, where that candle sits.
    * **Weight encodes timeframe**: coarser draws heavier, finer draws thinner and semi-transparent.
      Colour still means DIRECTION. Only the chart's own timeframe is visible when the chart opens.
    * The timeframe folds into the pattern family, so `4h - rising wedge` is one entry in the
      existing family dropdown rather than a third cascading menu. `max_picker_instances` bounds
      how many single-instance entries the extra timeframes may add between them, and
      `max_shapes_per_tf` bounds how many shapes each pattern draws. `tf_trendlines` follows
      `trendlines` unless you set it explicitly.

    `y_from_price` scales the y-axis to the CANDLES on screen rather than letting plotly autorange
    over everything drawn. A pattern's boundaries are only as tall as the pattern, and a shape that
    began before the window can be far taller than it: a 1h falling wedge starting at 86,000 and
    converging at 51,800 stretched a 60k-76k chart out to 50k-85k, shrinking the price action to a
    third of the pane. The chart is there to show price; an annotation should not rescale it.

    `mtf_emas` draws HIGHER-TIMEFRAME EMAs, taking the same spec as `scenario_figure`:
    `("1h", 200)`, `("1h", [50, 200])`, or `[("1h", 200), ("4h", 200)]`. A frame already loaded in
    `tf_frames` is reused (it has the deeper history); otherwise the chart frame is resampled. Each
    line is a STEP that changes only at its own bar's close, because that is the only moment the
    value is final — an interpolated 1h EMA shows a number nobody could have read at the time. They
    are ON by default; `"-1h_ema200"` in `start_visible` turns one off.

    S/R zones stay on the chart frame only: a 4h zone and a 15m zone are the same
    horizontal line at the same price, so drawing both adds clutter and no information.

    Nothing here is forward-looking: shapes are drawn from the bar the pattern CONFIRMED, using only
    pivots that had already been confirmed by then.
    """
    import numpy as np
    import plotly.graph_objects as go
    from ..indicators import chart_pattern_shapes, sr_zone_table
    from ..indicators.chart_patterns import chart_pattern_shapes_tuned, tuned_specs
    from ..indicators.chart_patterns import PATTERN_DIRECTION as _PD
    from ..data.timeframe import bar_interval, tf_minutes

    d = df.reset_index(drop=True)
    t = d[time_col].to_numpy() if time_col in d.columns else np.arange(len(d))

    def _view_index(frame, tcol):
        """First bar of `frame` at or after `view_start`; 0 when no view was asked for."""
        if view_start is None or tcol not in frame.columns:
            return 0
        vs = pd.to_datetime(view_start)
        if vs.tzinfo is None:
            vs = vs.tz_localize("UTC")
        # compare as UTC-naive on both sides; the column is tz-aware and a bare date string is not
        tt = pd.to_datetime(frame[tcol])
        tt = tt.dt.tz_convert("UTC").dt.tz_localize(None) if tt.dt.tz is not None else tt
        i = int(np.searchsorted(tt.to_numpy(), np.datetime64(vs.tz_localize(None))))
        return max(0, min(i, len(frame) - 1))

    # WARM-UP IS COMPUTED, NOT DRAWN. Indicators need the history; the browser does not. Rendering
    # all 20,000 bars to look at 2,600 made the chart crawl, so everything below is drawn from
    # `v0` onward while still being CALCULATED on the whole frame.
    v0 = _view_index(d, time_col)
    dv = d.iloc[v0:]
    tv = t[v0:]
    # the price band actually on screen — used to drop levels that are no longer anywhere near price
    _lo, _hi = float(dv["low"].min()), float(dv["high"].max())
    _pad = 0.15 * max(_hi - _lo, 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=tv, open=dv["open"], high=dv["high"], low=dv["low"], close=dv["close"],
        name="price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False))

    for p in (emas or ()):
        from ..indicators import ema as _ema
        fig.add_trace(go.Scattergl(
            x=tv, y=_ema(d["close"], int(p)).to_numpy()[v0:], mode="lines", name=f"EMA {p}",
            legendgroup=f"ema{p}", line=dict(width=1.4),
            visible=False if f"-ema{p}" in start_visible else True,   # ON by default
            hovertemplate=f"EMA {p}: %{{y:.2f}}<extra></extra>"))

    if mtf_emas:
        # HIGHER-TIMEFRAME EMAs. Drawn as a STEP that changes only at each HTF bar's close, because
        # that is the only moment the value is final — a 1h EMA interpolated across the hour shows a
        # number nobody could have read at the time. Same rule as every other event on this chart.
        from ..indicators.overlays import ema as _ema_of
        for k, (etf, per) in enumerate(_norm_mtf_ema(mtf_emas)):
            src = (tf_frames or {}).get(etf)
            if src is None:
                # no frame was loaded for it, so resample the chart's own — cheap, but its history
                # is capped by the chart frame's start, which a long period may not have enough of
                from ..data.timeframe import resample_ohlcv
                src = resample_ohlcv(d, f"{tf_minutes(etf)}min", time_col=time_col)
            src = src.reset_index(drop=True)
            ey = _ema_of(src["close"], int(per)).to_numpy()
            # a tz-aware column comes back as dtype=object, which will not take a timedelta64, so
            # the shift is done in pandas and only then handed to plotly
            ex = pd.to_datetime(src[time_col]) + bar_interval(src, time_col=time_col)
            i0 = max(0, int(ex.searchsorted(pd.Timestamp(str(tv[0])).tz_localize(None)
                                            if ex.dt.tz is None else pd.Timestamp(str(tv[0])))) - 1)
            ex = ex.to_numpy()      # one step back, so the line enters from the left edge
            label = f"{etf} EMA {int(per)}"
            if not np.isfinite(ey[i0:]).any():
                # an EMA needs `per` bars of its OWN timeframe. Resampling a three-week 15m chart
                # gives 126 4h bars, so a 4h EMA200 comes out entirely NaN and draws as nothing at
                # all — a silently missing line looks exactly like a line you forgot to turn on.
                import warnings as _w
                _w.warn(f"{label} has no value over this window: {len(src)} {etf} bars is fewer "
                        f"than the {int(per)} it needs. Load {etf} through quant.data.tf_frames "
                        f"(or widen the chart's warm-up) so it has the history.", stacklevel=2)
                continue
            fig.add_trace(go.Scattergl(
                x=ex[i0:], y=ey[i0:], mode="lines", name=label,
                legendgroup=f"mtfema{etf}{per}", line_shape="hv",
                line=dict(width=2.0, color=_MTF_EMA_SHADES[k % len(_MTF_EMA_SHADES)]),
                visible=False if f"-{etf}_ema{int(per)}" in start_visible else True,
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"))

    if sr_zones:
        # built from a BOUNDED history ending at the view, not the whole warm-up — see the docstring
        z0 = max(0, v0 - int(zone_lookback_bars)) if zone_lookback_bars else 0
        zt = sr_zone_table(d.iloc[z0:].reset_index(drop=True), left=zones_left, right=zones_right,
                           cluster_pct=cluster_pct, max_zones=max_zones, time_col=time_col)
        if len(zt):
            zt = zt.copy()
            zt[["start_i", "end_i"]] += z0
        zt = zt[zt["touches"] >= min_touches]
        # A zone is a level price is respecting NOW. Two filters, both about relevance rather than
        # correctness: the segment has to reach the visible window at all, and its price has to be
        # near the prices on screen. Without the second, a long warm-up in a trending market fills
        # the chart with levels from 20% below — 767 segments, none of them next to price.
        if len(zt):
            # ONE line per zone at its CURRENT level. A zone emits a row per price revision, so
            # drawing every row stacks a zone's whole history of slightly-different levels on the
            # chart — 1,366 segments for 40 zones. The last revision is the level in force.
            zt = zt.sort_values("end_i").groupby("zone", as_index=False).tail(1)
            zt = zt[zt["price"].between(_lo - _pad, _hi + _pad)]
        xs, ys, widths = [], [], []
        for r in zt.itertuples():
            # drawn across the whole visible window: a level established earlier is still the level
            x_from = tv[0] if r.start_i < v0 else r.start_t
            xs += [x_from, tv[-1], None]
            ys += [r.price, r.price, None]
            widths.append(r.touches)
        if xs:
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines", name=f"S/R zones ({len(zt)})",
                legendgroup="sr", line=dict(color="#8e9aa6", width=2),
                opacity=0.55, hoverinfo="skip",
                visible=True if "sr_zones" in start_visible else "legendonly"))

    def _detect(frame, want):
        """One detection pass for a frame, honouring tuned / per-pattern / shared settings."""
        if tuned or per_pattern:
            spec = per_pattern or tuned_specs()
            nm = list(want) if want else list(spec)
            return nm, chart_pattern_shapes_tuned(frame, specs=spec, patterns=nm,
                                                  **(detect_kwargs or {}))
        nm = list(want) if want else list(_PD)
        return nm, chart_pattern_shapes(frame, patterns=nm, **(detect_kwargs or {}))

    picker = []      # (family, label, first_trace_index, sort_key) for every isolatable instance
    fam_lg = {}      # family label -> the legendgroup that family's grouped traces carry
    tf_fams = {}     # timeframe label (None = this chart) -> its family labels, in drawing order

    def _draw_layer(dl, *, tf, want, draw_lines, width, dotsz, alpha, picker_budget):
        """Draw ONE timeframe's shapes and trendlines. Returns how much picker budget it used.

        Everything works in `dl`'s own bar space, and only becomes a chart coordinate at the moment
        a bar index is turned into a timestamp — the one coordinate the two frames share. That is
        why no index remapping is needed and why a slope fitted on 4h bars stays correct here.
        """
        tl_ = dl[time_col].to_numpy() if time_col in dl.columns else np.arange(len(dl))
        vl = _view_index(dl, time_col)
        n = len(dl)
        chart_tf = tf is None
        # An EVENT read off bar i of a HIGHER timeframe is only knowable when that bar CLOSES.
        # Zero for the chart's own frame, where the bar you are looking at IS the bar it happened on.
        step = np.timedelta64(0, "ns")
        if not chart_tf and time_col in dl.columns and n > 1:
            step = np.timedelta64(bar_interval(dl, time_col=time_col))

        def ev(i):
            """x-coordinate of an EVENT on bar `i`: this frame's bar CLOSE."""
            return tl_[int(i)] + step

        # plotly leaves `opacity` unset for None, which keeps the chart layer's JSON exactly as it
        # was before extra timeframes existed — a full-opacity layer should not serialise differently
        alpha = None if (alpha is None or alpha >= 1.0) else alpha
        pref = "" if chart_tf else f"{tf} · "
        lgp = "" if chart_tf else f"{tf}::"
        used = 0

        def _register(label, legendgroup):
            fam_lg[label] = legendgroup
            fams = tf_fams.setdefault(tf, [])
            if label not in fams:
                fams.append(label)

        names_, shapes = _detect(dl, want)
        if len(shapes) and vl > 0:
            # a shape that confirmed entirely inside the warm-up is history: it is what taught the
            # detector, not something to draw over the window you asked to look at
            shapes = shapes[shapes["confirm_i"] >= vl]
        for name in names_:
            sub = shapes[shapes["pattern"] == name] if len(shapes) else shapes
            if not len(sub):
                continue
            if max_shapes_per_tf and len(sub) > int(max_shapes_per_tf):
                sub = sub.sort_values("confirm_i").tail(int(max_shapes_per_tf))
            up = _PD.get(name, 0) > 0
            colr = "#2ecc71" if up else ("#e74c3c" if _PD.get(name, 0) < 0 else "#f0a30a")
            key = name if chart_tf else f"{tf}::{name}"
            vis = True if key in start_visible else "legendonly"
            lx, ly = [], []          # the two fitted boundaries, extended across the shape
            mx, my = [], []          # the pivots those lines are fitted through
            nx, ny = [], []          # the boundary that broke
            tx, ty = [], []          # measured-move target
            sx, sy = [], []          # protective stop
            for r in sub.itertuples():
                a, b = int(r.start_i), min(int(r.confirm_i) + extend, n - 1)
                xa, xb = tl_[a], ev(b)
                # Draw the LINE, not the segment between its two end pivots: evaluate the fit across
                # the whole pattern. Level shapes (H&S, double tops) have no fitted boundaries, only
                # a neckline. CONVERGING shapes are drawn all the way to their APEX, where the two
                # boundaries meet. Stopping at the break left a triangle as two lines heading toward
                # each other without ever closing, so it did not read as a triangle at all. Only
                # converging pairs have an apex ahead of them; channels are parallel and broadening
                # shapes met theirs in the past, so both are unaffected. The apex is bounded because
                # a pattern already expires at `max_apex_frac` (0.75) of the way there.
                xe = b
                du, dlo = r.upper_slope, r.lower_slope
                if du == du and dlo == dlo and abs(du - dlo) > 1e-12:
                    apex = (r.lower_intercept - r.upper_intercept) / (du - dlo)
                    if b < apex <= b + 4 * max(b - a, 1):
                        xe = int(min(apex, n - 1))
                xb2 = ev(xe)
                for sl, ic in ((r.upper_slope, r.upper_intercept),
                               (r.lower_slope, r.lower_intercept)):
                    if sl == sl and ic == ic:                       # not NaN
                        lx += [xa, xb2, None]
                        ly += [sl * r.start_i + ic, sl * xe + ic, None]
                # the pivots the lines were FITTED THROUGH, not every swing in the search window
                mx += list(r.touch_t) + [None]
                my += list(r.touch_price) + [None]
                nx += [xa, xb, None]
                ny += [r.boundary_slope * r.start_i + r.boundary_intercept,
                       r.boundary_slope * b + r.boundary_intercept, None]
                tx += [ev(r.confirm_i), xb, None]
                ty += [r.target, r.target, None]
                sx += [ev(r.confirm_i), xb, None]
                sy += [r.stop, r.stop, None]
            label = pref + name.replace("_", " ")
            _register(label, lgp + name)
            fig.add_trace(go.Scattergl(
                x=lx or nx, y=ly or ny, mode="lines", name=f"{label} ({len(sub)})",
                legendgroup=lgp + name, visible=vis, opacity=alpha,
                line=dict(color=colr, width=round(width, 2)),
                hovertemplate=f"{label}<extra></extra>"))
            fig.add_trace(go.Scattergl(
                x=mx, y=my, mode="markers", legendgroup=lgp + name, visible=vis, showlegend=False,
                opacity=alpha,
                marker=dict(size=dotsz, color=colr, line=dict(width=1, color="#111")),
                hovertemplate=f"{label} touch<extra></extra>"))
            for X, Y, colr2, dash, wid in ((nx, ny, "#f0a30a", "solid", width - 0.6),
                                           (tx, ty, colr, "dot", width - 1.2),
                                           (sx, sy, "#888888", "dot", width - 1.4)):
                if X:
                    fig.add_trace(go.Scattergl(
                        x=X, y=Y, mode="lines", legendgroup=lgp + name, visible=vis,
                        showlegend=False, opacity=alpha,
                        line=dict(color=colr2, width=max(0.8, round(wid, 2)), dash=dash),
                        hoverinfo="skip"))
            # WHERE IT CONFIRMED. A shape is only a signal from the bar price closed through its
            # boundary, and that bar is often far to the right of the shape itself — the median wait
            # on BTC 15m is 15 bars and a double top waits about 42. Marking it makes the lag visible
            # instead of leaving the shape looking as though it fired when it formed.
            fig.add_trace(go.Scattergl(
                x=[ev(r.confirm_i) for r in sub.itertuples()],
                y=[float(r.entry) for r in sub.itertuples()],
                mode="markers", legendgroup=lgp + name, visible=vis, showlegend=False,
                opacity=alpha,
                marker=dict(size=dotsz + 6, color=colr, symbol="star",
                            line=dict(width=1.5, color="#fff")),
                customdata=[[int(r.confirm_i) - int(max(r.pivot_i))] for r in sub.itertuples()],
                hovertemplate=(f"{label} CONFIRMED here<br>entry %{{y:.2f}}"
                               "<br>%{customdata[0]} bars after its last pivot<extra></extra>")))

            if line_picker and (picker_budget is None or used < picker_budget):
                # scope the picker to shapes that CONFIRM inside the opening view. Warm-up history
                # carries just as many again, and an instance list you have to scroll past a hundred
                # entries to use is no better than the overlap it replaces.
                _keep = sub
                if view_start is not None:
                    _vs0 = pd.to_datetime(view_start)
                    if _vs0.tzinfo is None:
                        _vs0 = _vs0.tz_localize("UTC")
                    _keep = sub[[pd.to_datetime(str(ev(r.confirm_i))).tz_convert("UTC") >= _vs0
                                 for r in sub.itertuples()]]
                if picker_budget is not None and len(_keep) > max(0, picker_budget - used):
                    # keep the MOST RECENT: those are the ones near the right edge you would inspect
                    _keep = _keep.sort_values("confirm_i").tail(max(0, picker_budget - used))
                for r in _keep.itertuples():
                    a2, b2 = int(r.start_i), min(int(r.confirm_i) + extend, n - 1)
                    ix, iy = [], []
                    for sl, ic in ((r.upper_slope, r.upper_intercept),
                                   (r.lower_slope, r.lower_intercept)):
                        if sl == sl and ic == ic:
                            ix += [tl_[a2], ev(b2), None]
                            iy += [sl * a2 + ic, sl * b2 + ic, None]
                    if not ix:
                        ix = [tl_[a2], ev(b2), None]
                        iy = [r.boundary_slope * a2 + r.boundary_intercept,
                              r.boundary_slope * b2 + r.boundary_intercept, None]
                    picker.append((label,
                                   f"{label} @ {pd.to_datetime(str(tl_[a2])):%b-%d %H:%M}"
                                   f"  (confirms {pd.to_datetime(str(ev(r.confirm_i))):%b-%d %H:%M})",
                                   len(fig.data), pd.Timestamp(str(tl_[a2])).value))
                    used += 1
                    fig.add_trace(go.Scattergl(x=ix, y=iy, mode="lines", visible=False,
                                               showlegend=False,
                                               line=dict(color=colr, width=round(width + 0.6, 2)),
                                               hovertemplate=f"{label}<extra></extra>"))
                    fig.add_trace(go.Scattergl(
                        x=list(r.touch_t) + [ev(r.confirm_i)],
                        y=list(r.touch_price) + [float(r.entry)],
                        mode="markers", visible=False, showlegend=False,
                        marker=dict(size=[dotsz + 2] * len(r.touch_t) + [dotsz + 8], color=colr,
                                    symbol=["circle"] * len(r.touch_t) + ["star"],
                                    line=dict(width=1.5, color="#fff")),
                        hovertemplate="touch / CONFIRMED<extra></extra>"))

        if draw_lines:
            from ..indicators.trendlines import trendline_table
            tl = trendline_table(dl, **(trendline_kwargs or {}))
            if len(tl) and vl > 0:
                tl = tl[tl["end_i"] >= vl]        # lines that died before the window opened
            for kind, base_label, colr in ((-1, "support trendline", "#2ecc71"),
                                           (+1, "resistance trendline", "#e74c3c")):
                sub = tl[tl["kind"] == kind] if len(tl) else tl
                if not len(sub):
                    continue
                if max_shapes_per_tf and len(sub) > int(max_shapes_per_tf):
                    # the same cap the shapes get: a 1m layer offered 80 resistance lines over nine
                    # days, which is a mesh rather than a chart. The most recent are the live ones.
                    sub = sub.sort_values("end_i").tail(int(max_shapes_per_tf))
                label = pref + base_label
                _register(label, lgp + base_label)
                key = (base_label.split()[0] + "_trendline") if chart_tf else f"{tf}::{base_label}"
                vis = True if key in start_visible else "legendonly"
                lx, ly, dx, dy, bx, by = [], [], [], [], [], []
                per_line = []      # one entry per line, for the picker
                for r in sub.itertuples():
                    # from the FIRST TOUCH, not from born_i: the line is fitted through pivots that
                    # all sit before it became known, so drawing from born_i left every touch dot
                    # outside the segment. All 36 lines in the sample had this.
                    lx += [tl_[int(r.start_i)], ev(r.end_i), None]
                    ly += [r.y_start, r.y_end, None]
                    dx += list(r.touch_t) + [None]
                    dy += list(r.touch_price) + [None]
                    if r.broke:
                        bx.append(ev(r.end_i)); by.append(r.y_end)
                    per_line.append((
                        f"{label} @ {pd.to_datetime(str(tl_[int(r.start_i)])):%b-%d %H:%M}"
                        f"  ({int(r.touches)} touches, {'broke' if r.broke else 'faded'})",
                        [tl_[int(r.start_i)], ev(r.end_i)], [r.y_start, r.y_end],
                        list(r.touch_t), list(r.touch_price), int(r.start_i), label,
                        int(r.born_i), int(r.end_i), bool(r.broke)))
                fig.add_trace(go.Scattergl(
                    x=lx, y=ly, mode="lines",
                    name=f"{label} ({len(sub)}, {int(sub.broke.sum())} broke)",
                    legendgroup=lgp + base_label, visible=vis, opacity=alpha,
                    line=dict(color=colr, width=max(0.8, round(width - 0.6, 2)), dash="dash"),
                    hovertemplate=f"{label}<extra></extra>"))
                fig.add_trace(go.Scattergl(
                    x=dx, y=dy, mode="markers", legendgroup=lgp + base_label, visible=vis,
                    showlegend=False, opacity=alpha,
                    marker=dict(size=max(3, dotsz - 1), color=colr,
                                line=dict(width=1, color="#111")),
                    hovertemplate=f"{label} touch<extra></extra>"))
                if bx:
                    fig.add_trace(go.Scattergl(
                        x=bx, y=by, mode="markers", legendgroup=lgp + base_label, visible=vis,
                        showlegend=False, opacity=alpha,
                        marker=dict(size=dotsz + 4, color=colr, symbol="x", line=dict(width=1.5)),
                        hovertemplate=f"{label} BREAK<extra></extra>"))
                if line_picker and (picker_budget is None or used < picker_budget):
                    # only offer lines that are actually VISIBLE in the opening view. Warm-up history
                    # can carry a hundred more, and a dropdown of 130 entries is no more usable than
                    # the overlapping lines it was meant to untangle.
                    if view_start is not None:
                        vs = pd.to_datetime(view_start)
                        # bar timestamps are tz-aware; a bare date string is not
                        if vs.tzinfo is None:
                            vs = vs.tz_localize("UTC")
                        per_line = [pl for pl in per_line
                                    if pd.to_datetime(str(pl[1][1])).tz_convert("UTC") >= vs]
                    if picker_budget is not None:
                        room = max(0, picker_budget - used)
                        per_line = per_line[-room:] if len(per_line) > room else per_line
                    for lbl, xs, ys, txs, tys, si, fam, bi, ei, broke in per_line:
                        picker.append((fam, lbl, len(fig.data), pd.Timestamp(str(tl_[si])).value))
                        used += 1
                        # same dashed style as the grouped view, just thicker — isolating a line
                        # should not make it look like a different kind of object
                        fig.add_trace(go.Scattergl(
                            x=xs, y=ys, mode="lines", visible=False, showlegend=False,
                            line=dict(color=colr, width=round(width + 0.4, 2), dash="dash"),
                            hovertemplate=f"{lbl}<extra></extra>"))

                        def _y_at(i, _ys=ys, _si=si, _ei=ei):
                            """Height of the drawn segment at bar `i`, by linear interpolation."""
                            if _ei == _si:
                                return _ys[0]
                            return _ys[0] + (_ys[1] - _ys[0]) * (i - _si) / (_ei - _si)

                        # touches, then the bar the line became KNOWN, then the break. The isolated
                        # view was drawing only the touches, so the two events that matter — when it
                        # became tradable and when it failed — were visible in the grouped view and
                        # nowhere else.
                        mx2 = list(txs) + [ev(bi)]
                        my2 = list(tys) + [float(_y_at(bi))]
                        syms = ["circle"] * len(txs) + ["diamond"]
                        szs = [dotsz + 2] * len(txs) + [dotsz + 6]
                        txt = ["touch"] * len(txs) + ["line CONFIRMED here (last pivot in place)"]
                        if broke:
                            mx2.append(ev(ei)); my2.append(float(_y_at(ei)))
                            syms.append("x"); szs.append(dotsz + 9); txt.append("BROKE here")
                        fig.add_trace(go.Scattergl(
                            x=mx2, y=my2, mode="markers", visible=False, showlegend=False,
                            marker=dict(size=szs, color=colr, symbol=syms,
                                        line=dict(width=1.6, color="#fff")),
                            text=txt, hovertemplate="%{text}<extra></extra>"))
        return used

    # ---- the chart's own timeframe, then any extra ones ----------------------------------
    _draw_layer(d, tf=None, want=patterns, draw_lines=trendlines,
                width=2.6, dotsz=7, alpha=1.0, picker_budget=None)

    if tf_frames:
        base_min = max(1.0, bar_interval(d, time_col=time_col) / pd.Timedelta(minutes=1))
        # coarsest first: a 4h line is the structure a 5m shape sits inside
        order = sorted(tf_frames, key=tf_minutes, reverse=True)
        # Split the picker budget between the extra timeframes and hand back what the quiet ones do
        # not use. Left uncapped, a 1m layer alone contributed 335 instance entries and ~9.8 MB of
        # dropdown JSON: every button carries a visibility vector as long as the trace list, so the
        # cost is buttons x traces and bounding the buttons is what keeps the chart openable.
        share = max(1, int(max_picker_instances) // max(1, len(order)))
        budget = {k: share for k in order}
        budget[order[0]] += max(0, int(max_picker_instances) - share * len(order))
        for pos, tf in enumerate(order):
            dl = tf_frames[tf].reset_index(drop=True)
            rank = float(np.log2(max(tf_minutes(tf), 1) / base_min))
            if rank >= 0:      # coarser than the chart: heavier, fully opaque — this is structure
                width, alpha = min(4.4, 2.6 + 0.55 * rank), 1.0
                dotsz = min(11, int(round(7 + 1.2 * rank)))
            else:              # finer: thinner and semi-transparent — this is detail, not structure
                width, alpha = max(1.2, 2.6 + 0.45 * rank), 0.55
                dotsz = max(4, int(round(7 + 1.2 * rank)))
            spent = _draw_layer(
                dl, tf=tf, want=(tf_patterns or {}).get(tf, patterns),
                draw_lines=trendlines if tf_trendlines is None else tf_trendlines,
                width=width, dotsz=dotsz, alpha=alpha,
                picker_budget=budget[tf])
            if budget[tf] > spent and pos + 1 < len(order):
                budget[order[pos + 1]] += budget[tf] - spent     # rolls on to the next timeframe

    # ---- TWO dropdowns: pick a family, then pick one instance inside it ------------------
    # A single flat list does not scale: with warm-up history it reached 130 entries, which is no
    # more readable than the overlapping lines it was meant to untangle. The first menu narrows to
    # one family (S/R zones, support trendlines, rising wedges...), the second isolates a single
    # occurrence within it. Instances are listed EARLIEST FIRST so the order matches the chart.
    # With extra timeframes the family carries its timeframe (`4h · rising wedge`) and the list is
    # grouped under per-timeframe headers rather than growing a THIRD cascading menu — a third menu
    # nests multiplicatively (a timeframe button would carry the family list, whose buttons each
    # carry an instance list), which is exactly what made the naive version unopenable.
    if picker:
        base = [tr.visible if tr.visible is not None else True for tr in fig.data]
        owned = {i for _, _, i, _ in picker} | {i + 1 for _, _, i, _ in picker}
        fam_groups = {}
        for fam, lbl, idx, sortk in sorted(picker, key=lambda z: z[3]):
            fam_groups.setdefault(fam, []).append((lbl, idx))

        def _grouped(keep=None):
            """Restore the normal grouped drawing, optionally showing only `keep` legendgroups."""
            v = list(base)
            for i in owned:
                v[i] = False
            if keep is not None:
                for j, tr in enumerate(fig.data):
                    g = getattr(tr, "legendgroup", "") or ""
                    if g and g != "price" and not g.startswith("ema"):
                        v[j] = (g in keep)
            return v

        def _isolate(first):
            v = list(base)
            for i in owned:
                v[i] = False
            for j, tr in enumerate(fig.data):
                g = getattr(tr, "legendgroup", "") or ""
                if g and g not in ("price",) and not g.startswith("ema"):
                    v[j] = False          # everything else out of the way
            v[first] = True
            v[first + 1] = True
            return v

        def _inst_buttons(fams, instances=True):
            """The instance menu's contents for a given set of families.

            `instances=False` lists the families only. Every button carries a visibility vector as
            long as the whole trace list, so the menu costs buttons x traces; listing an instance in
            the initial menu AND under its timeframe AND under its family tripled the dominant term
            for nothing. With several timeframes loaded a flat list of every instance is unusable
            anyway, so the wide selections show families and the family itself shows its instances.
            """
            keep = {fam_lg[fams[0]]} if len(fams) == 1 else None
            out = [dict(label="▸ all in this group", method="update",
                        args=[{"visible": _grouped(keep)}])]
            for fam in fams:
                if len(fams) > 1:
                    out.append(dict(label=f"── {fam.upper()}"
                                          f"{'' if instances else f'  ({len(fam_groups[fam])})'} ──",
                                    method="update",
                                    args=[{"visible": _grouped({fam_lg[fam]})}]))
                if not instances:
                    continue
                for lbl, idx in fam_groups[fam]:
                    out.append(dict(label=("    " if len(fams) > 1 else "") + lbl,
                                    method="update", args=[{"visible": _isolate(idx)}]))
            return out

        multi_tf = len(tf_fams) > 1
        def _wide_buttons(fams):
            """Menu 2 for a selection spanning several families."""
            return _inst_buttons(fams, instances=not multi_tf)

        # CASCADING: picking a family REWRITES the instance menu, so the second dropdown only ever
        # lists what belongs to the first one's selection. Plotly has no native cascade, but a
        # button's `update` carries a layout patch alongside the trace patch, and a menu's buttons
        # are a layout property — so the first menu can rebuild the second.
        all_fams = list(fam_groups)
        fam_buttons = [dict(label="▸ everything", method="update",
                            args=[{"visible": _grouped(None)},
                                  {"updatemenus[1].buttons": _wide_buttons(all_fams),
                                   "updatemenus[1].active": 0}])]
        if sr_zones:
            fam_buttons.append(dict(
                label="S/R zones", method="update",
                args=[{"visible": _grouped({"sr"})},
                      {"updatemenus[1].buttons": [dict(
                          label="(S/R zones have no individual entries)", method="update",
                          args=[{"visible": _grouped({"sr"})}])],
                       "updatemenus[1].active": 0}]))

        def _fam_button(fam):
            return dict(label=f"{fam}  ({len(fam_groups[fam])})", method="update",
                        args=[{"visible": _grouped({fam_lg[fam]})},
                              {"updatemenus[1].buttons": _inst_buttons([fam]),
                               "updatemenus[1].active": 0}])

        if multi_tf:
            # one section per timeframe, coarsest first, each opening with an "all of it" entry
            for tf in sorted(tf_fams, key=lambda k: (0, 0) if k is None else (1, -tf_minutes(k))):
                fams = [f for f in tf_fams[tf] if f in fam_groups]
                if not fams:
                    continue
                fam_buttons.append(dict(
                    label=f"━━ {'this chart' if tf is None else tf}"
                          f"  —  all patterns ({len(fams)})",
                    method="update",
                    args=[{"visible": _grouped({fam_lg[f] for f in fams})},
                          {"updatemenus[1].buttons": _wide_buttons(fams),
                           "updatemenus[1].active": 0}]))
                for fam in fams:
                    fam_buttons.append(_fam_button(fam))
        else:
            for fam in all_fams:
                fam_buttons.append(_fam_button(fam))

        menu = dict(direction="down", showactive=True, yanchor="top", y=1.045,
                    bgcolor="rgba(28,32,38,0.96)", bordercolor="#4a5058", borderwidth=1,
                    font=dict(size=11, color="#dfe3e8"), pad=dict(l=2, r=2, t=2, b=2))
        fig.update_layout(updatemenus=[
            dict(buttons=fam_buttons, x=0.0, xanchor="left", **menu),
            dict(buttons=_wide_buttons(all_fams), x=0.185, xanchor="left", **menu)])

    if view_start is not None:
        # the data before this is WARM-UP: it feeds the zones, lines and EMAs but is not shown
        _vs = pd.to_datetime(view_start)
        if _vs.tzinfo is None:
            _vs = _vs.tz_localize("UTC")
        fig.update_xaxes(range=[_vs, t[-1]])

    if y_from_price:
        # scale to the candles, not to whatever geometry hangs off them — see the docstring
        _ypad = 0.04 * max(_hi - _lo, 1e-9)
        fig.update_yaxes(range=[_lo - _ypad, _hi + _ypad])

    fig.update_layout(
        title=title or "Chart patterns — click the legend to show/hide each one",
        height=height, template="plotly_dark", xaxis_rangeslider_visible=False,
        hovermode="x unified", legend=dict(orientation="v", x=1.01, y=1,
                                           bgcolor="rgba(0,0,0,0.35)"),
        margin=dict(l=50, r=210, t=104, b=40),
        title_x=0.5, title_xanchor="center", title_y=0.985, title_yanchor="top")
    fig.update_yaxes(side="right")
    return fig


def add_pattern_overlays(fig, df, *, time_col: str = "t", row: int = 1, col: int = 1,
                         candles=True, charts=True, candle_names=None,
                         chart_names=None, cdl_prefix: str = "cdl", pat_prefix: str = "pat",
                         show_levels: bool = True, max_labels: int = 400):
    """Draw detected patterns onto an existing price figure.

    Candlestick hits become small triangles under (bullish) or over (bearish) the bar, grouped into
    ONE legend entry per pattern so each can be toggled on and off individually. Chart patterns get a
    marker at the confirmation bar plus, with `show_levels`, their entry / stop / target as short
    horizontal segments — the geometry the trade would actually use.

    `candle_names` / `chart_names` restrict what is drawn; by default everything present is drawn,
    which on a long series is a lot, so `max_labels` caps each pattern's markers to the most recent
    ones.
    """
    import numpy as np
    import plotly.graph_objects as go
    from ..indicators.candle_patterns import PATTERN_DIRECTION as _CDL_DIR
    from ..indicators.chart_patterns import PATTERN_DIRECTION as _PAT_DIR

    t = df[time_col].to_numpy() if time_col in df.columns else np.arange(len(df))
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    pad = (hi - lo)
    pad = np.where(np.isfinite(pad) & (pad > 0), pad, 1.0) * 0.6

    if candles:
        names = candle_names or [n for n in _CDL_DIR if f"{cdl_prefix}_{n}" in df.columns]
        for name in names:
            c = f"{cdl_prefix}_{name}"
            if c not in df.columns:
                continue
            idx = np.flatnonzero(df[c].to_numpy(bool))
            if not len(idx):
                continue
            idx = idx[-max_labels:]
            d = _CDL_DIR.get(name, 0)
            up = d > 0
            fig.add_trace(go.Scattergl(
                x=t[idx], y=(lo[idx] - pad[idx]) if up else (hi[idx] + pad[idx]),
                mode="markers", name=f"cdl {name}",
                legendgroup=f"cdl_{name}", visible="legendonly",
                marker=dict(symbol="triangle-up" if up else "triangle-down", size=8,
                            color="#2ecc71" if up else ("#e74c3c" if d < 0 else "#f1c40f")),
                hovertemplate=f"{name}<extra></extra>"), row=row, col=col)

    if charts and f"{pat_prefix}_signal" in df.columns:
        names = chart_names or [n for n in _PAT_DIR if f"{pat_prefix}_{n}" in df.columns]
        entry = df.get(f"{pat_prefix}_entry")
        stop = df.get(f"{pat_prefix}_stop")
        target = df.get(f"{pat_prefix}_target")
        for name in names:
            c = f"{pat_prefix}_{name}"
            if c not in df.columns:
                continue
            idx = np.flatnonzero(df[c].to_numpy(bool))
            if not len(idx):
                continue
            idx = idx[-max_labels:]
            d = _PAT_DIR.get(name, 0)
            up = d > 0
            fig.add_trace(go.Scattergl(
                x=t[idx], y=(lo[idx] - 2 * pad[idx]) if up else (hi[idx] + 2 * pad[idx]),
                mode="markers+text", name=f"pat {name}",
                legendgroup=f"pat_{name}", visible="legendonly",
                text=[name.replace("_", " ")] * len(idx), textposition="bottom center",
                textfont=dict(size=9),
                marker=dict(symbol="star", size=11,
                            color="#2ecc71" if up else "#e74c3c"),
                hovertemplate=f"{name}<extra></extra>"), row=row, col=col)
            if show_levels and entry is not None:
                seg_x, seg_y, tgt_x, tgt_y, stp_x, stp_y = [], [], [], [], [], []
                span = max(len(df) // 200, 5)
                for i in idx:
                    j = min(i + span, len(df) - 1)
                    for xs, ys, src in ((seg_x, seg_y, entry), (tgt_x, tgt_y, target),
                                        (stp_x, stp_y, stop)):
                        v = float(src.to_numpy(float)[i])
                        xs += [t[i], t[j], None]
                        ys += [v, v, None]
                for xs, ys, lab, colr, dash in (
                        (seg_x, seg_y, "entry", "#f39c12", "solid"),
                        (tgt_x, tgt_y, "target", "#2ecc71", "dot"),
                        (stp_x, stp_y, "stop", "#e74c3c", "dot")):
                    fig.add_trace(go.Scattergl(
                        x=xs, y=ys, mode="lines", name=f"pat {name} {lab}",
                        legendgroup=f"pat_{name}", visible="legendonly", showlegend=False,
                        line=dict(color=colr, width=1, dash=dash),
                        hoverinfo="skip"), row=row, col=col)
    return fig


def scenario_figure(run, *, time_col: str = "t", zones_left: int = 10, zones_right: int = 10,
                    cluster_pct: float = 0.15, title: Optional[str] = None, height: int = 900,
                    show_zones: bool = True, grad_band: Optional[float] = None,
                    atr_bands: Optional[Sequence[float]] = None, atr_period: int = 14,
                    rsi: bool = False, rsi_period: int = 14, rsi_divergence=True,
                    div_left: int = 10, div_right: int = 2,
                    structure: bool = False, struct_lr: int = 10,
                    struct_method: str = "zigzag", struct_atr_mult: float = 2.5,
                    mtf_ema=None, mtf_ema_pane: bool = False):
    """Fully-annotated scenario chart — everything needed to validate a setup by eye against the
    TradingView screenshots, on ONE shared-x figure:

      * price candles + the smoothed-EMA trend line (`{grad}_line`, red dotted, like the screenshot);
      * S/R **zones** as horizontal bands (support green / resistance red, thicker = more touches),
        drawn causally via `sr_zone_table` — so you see the zone price traps in and then breaks;
      * every trade: **entry** (red ▼ at the break), the **stop** level (dotted line to the exit),
        and the **exit** (coloured by reason — TP / stop / signal);
      * a **gradient pane** below with the `grad` oscillator, the 0 line and a ±threshold band, so you
        can see it turn down before entries fire.

    `run` is a `quant.scenarios.ScenarioRun` (uses its `.df`, `.trades`, `.params`). Crosshair +
    toggle-able legend included. Returns the plotly figure (feed to `open_in_browser(..., extra_js=
    CROSSHAIR_JS)`).
    """
    from ..indicators import sr_zone_table

    d = run.df.reset_index(drop=True)
    p = run.params
    pre = p.get("grad_prefix", "grad")
    line_col, grad_col = f"{pre}_line", pre
    x = d[time_col]
    band = grad_band if grad_band is not None else p.get("grad_max")
    scen = getattr(getattr(run, "scenario", None), "key", "scenario")
    sec = getattr(run, "secondary", None)
    secs = list(sec) if isinstance(sec, (list, tuple)) else ([sec] if sec is not None else [])
    # Name every active piece — including whether a secondary rule was running, so a chart can never
    # be mistaken for one produced with a different configuration.
    pieces = " / ".join(x for x in [getattr(getattr(run, "stop", None), "key", None),
                                    getattr(getattr(run, "tp", None), "key", None),
                                    (" + ".join(s.key for s in secs) if secs
                                     else "no secondary")] if x)
    title = title or f"{scen}  ({pieces})  |  {p.get('entry_mode', '')}  {len(run.trades)} trades"

    # panes: price on top, then the gradient, then (optionally) RSI and a higher-TF EMA pane
    _has_extra = bool(rsi) or bool(mtf_ema and mtf_ema_pane)
    heights = [0.62 if _has_extra else 0.72, 0.20 if _has_extra else 0.28]
    titles = ["", f"EMA gradient ({grad_col})"]
    rsi_row = mtf_row = None
    if rsi:
        heights.append(0.18)
        _dm = "confirmed" if rsi_divergence is True else (rsi_divergence or "")
        titles.append(f"RSI({rsi_period})" + (f" + {_dm} divergence" if _dm else ""))
        rsi_row = len(heights)
    _mtf_pairs = _norm_mtf_ema(mtf_ema)
    if _mtf_pairs and mtf_ema_pane:
        _tf, _per = _mtf_pairs[0]
        heights.append(0.18)
        titles.append(f"distance from {_tf} EMA{int(_per)}  (ATR units, higher-TF bias)")
        mtf_row = len(heights)
    tot = sum(heights)
    heights = [h / tot for h in heights]
    fig = make_subplots(rows=len(heights), cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.045, subplot_titles=titles)

    # ---- row 1: price + trend line ----
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price", legendgroup="price",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
                  row=1, col=1)
    if atr_bands:
        _add_atr_bands(fig, d, x, line_col, list(atr_bands), atr_period)
    if structure:
        _add_structure(fig, d, x, struct_lr, struct_lr,
                       method=struct_method, atr_mult=struct_atr_mult)
    if _mtf_pairs:
        _add_mtf_emas(fig, d, x, _mtf_pairs, mtf_row, atr_period)
    if line_col in d.columns:
        fig.add_trace(go.Scatter(x=x, y=d[line_col], name="EMA line", mode="lines",
                                 line=dict(width=1.4, color="#d32f2f", dash="dot")))

    # ---- S/R zones (causal segments) ----
    if show_zones and {"sr_support", "sr_resistance"}.issubset(d.columns):
        zt = sr_zone_table(d, left=zones_left, right=zones_right, cluster_pct=cluster_pct,
                           time_col=time_col)
        close = d["close"].to_numpy()
        seg = {"sup": [[], []], "res": [[], []]}   # x, y with None separators
        for _, z in zt.iterrows():
            si = int(z["start_i"])
            kind = "sup" if z["price"] <= close[si] else "res"   # below price = support
            seg[kind][0] += [z["start_t"], z["end_t"], None]
            seg[kind][1] += [z["price"], z["price"], None]
        for kind, color, label in [("sup", "#26a69a", "Support"), ("res", "#ef5350", "Resistance")]:
            if seg[kind][0]:
                fig.add_trace(go.Scatter(x=seg[kind][0], y=seg[kind][1], mode="lines", name=label,
                                         legendgroup=kind, line=dict(width=2, color=color),
                                         opacity=0.55, hoverinfo="y"))

    # ---- trades: entries, stop levels, exits ----
    tr = run.trades
    if len(tr):
        fig.add_trace(go.Scatter(x=tr["entry_time"], y=tr["entry_price"], mode="markers",
                                 name="Entry (short)", legendgroup="entry",
                                 marker=dict(symbol="triangle-down", color="#c62828", size=11,
                                             line=dict(width=1, color="#7f0000")),
                                 hovertext=[f"#{i} entry {ep:.2f}" for i, ep in
                                            zip(tr.index, tr["entry_price"])],
                                 hoverinfo="text"))
        # stop levels: one dotted segment per trade from entry to exit
        sx, sy = [], []
        for _, t in tr.iterrows():
            if t.get("stop_price") and t["stop_price"] == t["stop_price"]:
                sx += [t["entry_time"], t["exit_time"], None]
                sy += [t["stop_price"], t["stop_price"], None]
        if sx:
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="Stop", legendgroup="stop",
                                     line=dict(width=1, color="#c62828", dash="dot"),
                                     opacity=0.7, hoverinfo="skip"))
        # LEVEL B: the level this trade actually broke, held for the trade's life. Own legend group
        # so it toggles independently of the stop lines.
        if "break_level" in d.columns:
            bx, by, btxt = [], [], []
            lv = d["break_level"].to_numpy()
            for i, t in tr.iterrows():
                b = lv[int(t["entry_i"])]
                if b == b:                                  # not NaN
                    bx += [t["entry_time"], t["exit_time"], None]
                    by += [b, b, None]
                    btxt += [f"#{i} level B {b:.2f}"] * 2 + [""]
            if bx:
                fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", name="Level B (broken)",
                                         legendgroup="breaklevel",
                                         line=dict(width=1.6, color="#f9a825", dash="dash"),
                                         opacity=0.9, hovertext=btxt, hoverinfo="text"),
                              row=1, col=1)
        # Secondary rule: mark the bar each trade ARMED on, so "did it arm, and when?" is visible.
        # Mirrors the engine: arming only counts from the bar AFTER entry, up to the exit.
        if secs:
            try:
                arm = np.logical_or.reduce([np.asarray(s.arm(p)(d), bool) for s in secs])
            except Exception:
                arm = None
            if arm is not None:
                ax, ay, atxt = [], [], []
                for i, t in tr.iterrows():
                    ei, xi = int(t["entry_i"]), int(t["exit_i"])
                    w = np.flatnonzero(arm[ei + 1:xi + 1])
                    if len(w):
                        j = ei + 1 + int(w[0])
                        ax.append(d[time_col].iloc[j]); ay.append(d["high"].iloc[j])
                        atxt.append(f"#{i} armed @ {d[time_col].iloc[j]:%H:%M}")
                if ax:
                    fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers", name="Secondary armed",
                                             legendgroup="armed",
                                             marker=dict(symbol="triangle-up-open", size=11,
                                                         color="#6a1b9a",
                                                         line=dict(width=2, color="#6a1b9a")),
                                             hovertext=atxt, hoverinfo="text"))

        # exits coloured by reason
        for reason, (color, symbol, label) in _EXIT_STYLE.items():
            m = tr["close_reason"] == reason
            if m.any():
                sub = tr[m]
                fig.add_trace(go.Scatter(x=sub["exit_time"], y=sub["exit_price"], mode="markers",
                                         name=f"Exit: {label}", legendgroup="exit",
                                         marker=dict(symbol=symbol, color=color, size=9),
                                         hovertext=[f"#{i} {reason} {xp:.2f}  pnl {pn:.2f}"
                                                    for i, xp, pn in
                                                    zip(sub.index, sub["exit_price"], sub["pnl"])],
                                         hoverinfo="text"))

    # ---- row 2: gradient oscillator ----
    if grad_col in d.columns:
        g = d[grad_col].to_numpy()
        fig.add_trace(go.Scatter(x=x, y=np.where(g >= 0, g, np.nan), mode="lines", name="grad +",
                                 legendgroup="grad", line=dict(width=1.2, color="#26a69a"),
                                 connectgaps=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=np.where(g < 0, g, np.nan), mode="lines", name="grad -",
                                 legendgroup="grad", line=dict(width=1.2, color="#ef5350"),
                                 connectgaps=False, showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_width=1, line_color="#888", row=2, col=1)
        if band is not None:
            for lvl in (band, -abs(band)):
                fig.add_hline(y=lvl, line_width=1, line_dash="dash", line_color="#9c27b0",
                              row=2, col=1)

    # ---- RSI pane (+ divergence drawn on BOTH panes, so price and momentum line up) ----
    if rsi:
        from ..indicators import add_rsi, rsi_divergence_table
        rr = add_rsi(d, rsi_period)[f"rsi_{int(rsi_period)}"]
        for lvl, colr, dash in [(70, "#ef5350", "dash"), (50, "#9e9e9e", "dot"),
                                (30, "#26a69a", "dash")]:
            fig.add_hline(y=lvl, line_width=1, line_dash=dash, line_color=colr,
                          row=rsi_row, col=1)
        fig.add_trace(go.Scatter(x=x, y=rr, mode="lines", name=f"RSI({rsi_period})",
                                 legendgroup="rsi", line=dict(width=1.2, color="#7b1fa2")),
                      row=rsi_row, col=1)
        # 'confirmed' (lag == div_right) | 'realtime' (zero-lag, provisional) | 'both'
        div_mode = "confirmed" if rsi_divergence is True else (rsi_divergence or "")
        if div_mode in ("confirmed", "both"):
            tab = rsi_divergence_table(d, rsi_period, left=div_left, right=div_right)
            for kind, colr, nm in [("bear", "#c62828", "Bearish div"),
                                   ("bull", "#2e7d32", "Bullish div")]:
                sub = tab[tab.kind == kind] if len(tab) else tab
                if not len(sub):
                    continue
                px_x, px_y, rs_x, rs_y = [], [], [], []
                for _, z in sub.iterrows():
                    i0, i1 = int(z.i0), int(z.i1)
                    px_x += [x.iloc[i0], x.iloc[i1], None]
                    px_y += [z.px0, z.px1, None]
                    rs_x += [x.iloc[i0], x.iloc[i1], None]
                    rs_y += [z.rsi0, z.rsi1, None]
                fig.add_trace(go.Scatter(x=px_x, y=px_y, mode="lines+markers", name=nm,
                                         legendgroup=f"div{kind}",
                                         line=dict(width=1.6, color=colr),
                                         marker=dict(size=5, color=colr),
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=rs_x, y=rs_y, mode="lines+markers", name=nm,
                                         legendgroup=f"div{kind}", showlegend=False,
                                         line=dict(width=1.6, color=colr, dash="dot"),
                                         marker=dict(size=5, color=colr),
                                         hoverinfo="skip"), row=rsi_row, col=1)

        if div_mode in ("realtime", "both"):
            # zero-lag PROVISIONAL signals — hollow markers, deliberately distinct from the
            # confirmed lines. ~half are never ratified by a real pivot; see SESSION_NOTES.
            from ..indicators import add_rsi_divergence_realtime
            rtd = add_rsi_divergence_realtime(d, rsi_period, left=div_left)
            for col_flag, colr, nm in [("rsi_rtdiv_bear", "#c62828", "Bearish div (provisional)"),
                                       ("rsi_rtdiv_bull", "#2e7d32", "Bullish div (provisional)")]:
                mk = rtd[col_flag].to_numpy()
                if not mk.any():
                    continue
                yv = d["high"].to_numpy() if "bear" in col_flag else d["low"].to_numpy()
                fig.add_trace(go.Scatter(
                    x=x[mk], y=yv[mk], mode="markers", name=nm, legendgroup=f"rt{col_flag}",
                    marker=dict(symbol="circle-open", size=8, color=colr, line=dict(width=1.5)),
                    hovertemplate=nm + " (zero-lag) %{y:.2f}<extra></extra>"))

    # Title on the top line, legend on its own line beneath it — with enough top margin for both.
    # (A shared band made the title and the legend entries overlap.)
    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", y=0.985, yanchor="top",
                                 font=dict(size=15)),
                      template="plotly_white", height=height, dragmode="pan",
                      showlegend=True, margin=dict(l=50, r=30, t=110, b=40),
                      legend=dict(orientation="h", y=1.015, yanchor="bottom", x=0, xanchor="left"))
    fig.update_xaxes(rangeslider_visible=False)
    _apply_crosshair(fig)
    return fig


def zoom_to_trade(fig, run, trade_i: int, *, pad_bars: int = 60, time_col: str = "t"):
    """Zoom a `scenario_figure` to a single trade (entry→exit ± `pad_bars`) so you can inspect the
    break, the zone, the stop and the exit up close. Mutates and returns the figure."""
    d = run.df.reset_index(drop=True)
    t = run.trades.loc[trade_i]
    ei, xi = int(t["entry_i"]), int(t["exit_i"])
    lo_i = max(0, ei - pad_bars)
    hi_i = min(len(d) - 1, xi + pad_bars)
    x0, x1 = d[time_col].iloc[lo_i], d[time_col].iloc[hi_i]
    win = d.iloc[lo_i:hi_i + 1]
    pad = (win["high"].max() - win["low"].min()) * 0.08
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[win["low"].min() - pad, win["high"].max() + pad])
    fig.update_layout(title_text=f"trade #{trade_i}  {t['side']}  "
                                 f"{t['entry_price']:.2f} → {t['exit_price']:.2f}  "
                                 f"({t['close_reason']}, pnl {t['pnl']:.2f})")
    return fig


def pattern_contact_sheet(df, *, patterns=None, worst: int = 1, best: int = 1,
                          time_col: str = "t", tuned: bool = True, per_pattern=None,
                          detect_kwargs=None, pad_frac: float = 0.25, cols: int = 4,
                          panel_height: int = 300):
    """One panel per pattern occurrence — its BEST and WORST example, side by side.

    Every detector bug in this library so far was found by looking at a chart: dots that were not
    touches, "symmetrical" triangles with 17:1 slopes, a parallel channel called a wedge. This makes
    that inspection systematic instead of lucky. Shapes are ranked by CONTAINMENT — the share of bars
    between the shape's first pivot and its break that sit inside its two boundaries — because a
    shape price keeps escaping is not the shape it claims to be, whatever its boundaries do.

    Read the worst column first. That is where a definitional bug shows up, and if the worst example
    of every pattern looks reasonable, the library is behaving.

    Level shapes (head-and-shoulders, double and triple tops) have no fitted boundaries, so they are
    ranked by span and shown for completeness rather than scored.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ..indicators import chart_pattern_shapes, add_atr
    from ..indicators.chart_patterns import chart_pattern_shapes_tuned, tuned_specs

    d = df.reset_index(drop=True)
    if "atr_14" not in d.columns:
        d = add_atr(d, 14)
    t = d[time_col].to_numpy() if time_col in d.columns else np.arange(len(d))
    hi, lo = d["high"].to_numpy(float), d["low"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    atr_med = float(np.nanmedian(atr))

    if tuned or per_pattern:
        spec = per_pattern or tuned_specs()
        names = list(patterns) if patterns else list(spec)
        shapes = chart_pattern_shapes_tuned(d, specs=spec, patterns=names, **(detect_kwargs or {}))
    else:
        shapes = chart_pattern_shapes(d, patterns=patterns, **(detect_kwargs or {}))
    if not len(shapes):
        return go.Figure().update_layout(title="no patterns found")

    def containment(r) -> float:
        if not (np.isfinite(r.upper_slope) and np.isfinite(r.lower_slope)):
            return np.nan
        a, b = int(min(r.pivot_i)), int(r.confirm_i)
        if b <= a:
            return np.nan
        x = np.arange(a, b + 1)
        tol = 0.6 * (atr[b] if np.isfinite(atr[b]) else atr_med)
        u = r.upper_slope * x + r.upper_intercept
        l = r.lower_slope * x + r.lower_intercept
        return float(((hi[a:b + 1] <= u + tol) & (lo[a:b + 1] >= l - tol)).mean())

    shapes = shapes.copy()
    shapes["containment"] = [containment(r) for r in shapes.itertuples()]

    picks = []
    for name, g in shapes.groupby("pattern"):
        g = g.sort_values("containment", na_position="last")
        lvl = g["containment"].isna().all()
        if lvl:                                   # level shapes: widest and narrowest instead
            g = g.sort_values("span_bars")
        for r in list(g.head(worst).itertuples()):
            picks.append((name, "worst" if not lvl else "narrowest", r))
        for r in list(g.tail(best).itertuples()):
            picks.append((name, "best" if not lvl else "widest", r))

    n = len(picks)
    rows = (n + cols - 1) // cols
    titles = [f"{nm.replace('_', ' ')} — {tag}"
              + (f" ({r.containment:.2f})" if np.isfinite(r.containment) else "")
              for nm, tag, r in picks]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.06, horizontal_spacing=0.04)

    for k, (nm, tag, r) in enumerate(picks):
        rr, cc = k // cols + 1, k % cols + 1
        a, b = int(min(r.pivot_i)), int(r.confirm_i)
        pad = max(int((b - a) * pad_frac), 5)
        s0, s1 = max(a - pad, 0), min(b + pad, len(d) - 1)
        w = d.iloc[s0:s1 + 1]
        fig.add_trace(go.Candlestick(
            x=w[time_col], open=w["open"], high=w["high"], low=w["low"], close=w["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            showlegend=False, name=""), row=rr, col=cc)
        colr = "#f0a30a"
        if np.isfinite(r.upper_slope):
            x = np.arange(a, b + 1)
            for sl, ic in ((r.upper_slope, r.upper_intercept), (r.lower_slope, r.lower_intercept)):
                fig.add_trace(go.Scatter(x=t[x], y=sl * x + ic, mode="lines", showlegend=False,
                                         line=dict(color=colr, width=2.2)), row=rr, col=cc)
        fig.add_trace(go.Scatter(x=list(r.touch_t), y=list(r.touch_price), mode="markers",
                                 showlegend=False,
                                 marker=dict(size=7, color=colr, line=dict(width=1, color="#111"))),
                      row=rr, col=cc)
        fig.add_vline(x=t[b], line=dict(color="#8e9aa6", width=1, dash="dot"), row=rr, col=cc)

    fig.update_xaxes(rangeslider_visible=False, showticklabels=False)
    fig.update_layout(height=panel_height * rows, template="plotly_dark", showlegend=False,
                      title="Pattern contact sheet — worst and best example of each, by containment",
                      margin=dict(l=40, r=20, t=80, b=30))
    for ann in fig.layout.annotations:
        ann.font.size = 11
    return fig
