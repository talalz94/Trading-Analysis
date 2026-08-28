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


def _norm_mtf_ema(spec):
    """Accept any of:  ('5min', 200)  |  ('5min', [100, 200])  |  [('5min',100), ('15min',200)]
    and return a flat list of (timeframe, period) pairs."""
    if not spec:
        return []
    # a single (tf, period-or-list) pair
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        tf, per = spec
        pers = per if isinstance(per, (list, tuple)) else [per]
        return [(tf, int(p)) for p in pers]
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
def pattern_figure(df, *, patterns=None, time_col: str = "t", title=None, height: int = 820,
                   sr_zones: bool = True, zones_left: int = 10, zones_right: int = 10,
                   cluster_pct: float = 0.15, min_touches: int = 2, max_zones: int = 40,
                   extend: int = 12, start_visible=(), detect_kwargs=None, tuned: bool = False,
                   per_pattern=None):
    """A clean price chart with CHART-PATTERN SHAPES and S/R zones, each toggleable on its own.

    Unlike `add_pattern_overlays`, which drops a marker at the confirmation bar, this draws the
    GEOMETRY: the pivots the shape is built from, the upper and lower trendlines through them, the
    neckline that had to break, and the measured-move target. That is what makes a triangle look like
    a triangle rather than a dot.

    Every pattern gets ONE legend entry controlling all of its drawing, so you can bring shapes in
    and out one at a time. `start_visible` names the ones that should be on when the chart opens;
    everything else starts in `legendonly`. S/R zones are their own toggle.

    `tuned=True` draws every pattern at its own calibrated settings (`TUNED_SPECS`) instead of one
    shared configuration — which needs several detection passes, because those settings disagree on
    `atr_mult` and that sizes the zigzag for the whole pass. `per_pattern={name: {...}}` does the
    same with settings you supply. Both override `detect_kwargs` for pattern detection.

    Nothing here is forward-looking: shapes are drawn from the bar the pattern CONFIRMED, using only
    pivots that had already been confirmed by then.
    """
    import numpy as np
    import plotly.graph_objects as go
    from ..indicators import chart_pattern_shapes, sr_zone_table
    from ..indicators.chart_patterns import chart_pattern_shapes_tuned, tuned_specs
    from ..indicators.chart_patterns import PATTERN_DIRECTION as _PD

    d = df.reset_index(drop=True)
    t = d[time_col].to_numpy() if time_col in d.columns else np.arange(len(d))
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=t, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False))

    if sr_zones:
        zt = sr_zone_table(d, left=zones_left, right=zones_right, cluster_pct=cluster_pct,
                           max_zones=max_zones, time_col=time_col)
        zt = zt[zt["touches"] >= min_touches]
        xs, ys, widths = [], [], []
        for r in zt.itertuples():
            xs += [r.start_t, r.end_t, None]
            ys += [r.price, r.price, None]
            widths.append(r.touches)
        if xs:
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="lines", name=f"S/R zones ({len(zt)})",
                legendgroup="sr", line=dict(color="#8e9aa6", width=2),
                opacity=0.55, hoverinfo="skip",
                visible=True if "sr_zones" in start_visible else "legendonly"))

    if tuned or per_pattern:
        spec = per_pattern or tuned_specs()
        names = list(patterns) if patterns else list(spec)
        shapes = chart_pattern_shapes_tuned(d, specs=spec, patterns=names,
                                            **(detect_kwargs or {}))
    else:
        names = list(patterns) if patterns else list(_PD)
        shapes = chart_pattern_shapes(d, patterns=names, **(detect_kwargs or {}))
    n = len(d)
    for name in names:
        sub = shapes[shapes["pattern"] == name] if len(shapes) else shapes
        if not len(sub):
            continue
        up = _PD.get(name, 0) > 0
        colr = "#2ecc71" if up else ("#e74c3c" if _PD.get(name, 0) < 0 else "#f0a30a")
        vis = True if name in start_visible else "legendonly"
        lx, ly = [], []          # the two fitted boundaries, extended across the shape
        mx, my = [], []          # the pivots those lines are fitted through
        nx, ny = [], []          # the boundary that broke
        tx, ty = [], []          # measured-move target
        sx, sy = [], []          # protective stop
        for r in sub.itertuples():
            a, b = int(r.start_i), min(int(r.confirm_i) + extend, n - 1)
            xa, xb = t[a], t[b]
            # Draw the LINE, not the segment between its two end pivots: evaluate the fit across the
            # whole pattern. Level shapes (H&S, double tops) have no fitted boundaries, only a neckline.
            for sl, ic in ((r.upper_slope, r.upper_intercept), (r.lower_slope, r.lower_intercept)):
                if sl == sl and ic == ic:                       # not NaN
                    lx += [xa, xb, None]
                    ly += [sl * r.start_i + ic, sl * (b) + ic, None]
            # the pivots the lines were FITTED THROUGH, not every swing in the search window
            mx += list(r.touch_t) + [None]
            my += list(r.touch_price) + [None]
            nx += [xa, xb, None]
            ny += [r.boundary_slope * r.start_i + r.boundary_intercept,
                   r.boundary_slope * b + r.boundary_intercept, None]
            tx += [t[int(r.confirm_i)], xb, None]
            ty += [r.target, r.target, None]
            sx += [t[int(r.confirm_i)], xb, None]
            sy += [r.stop, r.stop, None]
        label = name.replace("_", " ")
        fig.add_trace(go.Scattergl(
            x=lx or nx, y=ly or ny, mode="lines", name=f"{label} ({len(sub)})",
            legendgroup=name, visible=vis,
            line=dict(color=colr, width=2.6),
            hovertemplate=f"{label}<extra></extra>"))
        fig.add_trace(go.Scattergl(
            x=mx, y=my, mode="markers", legendgroup=name, visible=vis, showlegend=False,
            marker=dict(size=7, color=colr, line=dict(width=1, color="#111")),
            hovertemplate=f"{label} touch<extra></extra>"))
        for X, Y, colr2, dash, wid in ((nx, ny, "#f0a30a", "solid", 2.0),
                                       (tx, ty, colr, "dot", 1.4),
                                       (sx, sy, "#888888", "dot", 1.2)):
            if X:
                fig.add_trace(go.Scattergl(
                    x=X, y=Y, mode="lines", legendgroup=name, visible=vis, showlegend=False,
                    line=dict(color=colr2, width=wid, dash=dash), hoverinfo="skip"))

    fig.update_layout(
        title=title or "Chart patterns — click the legend to show/hide each one",
        height=height, template="plotly_dark", xaxis_rangeslider_visible=False,
        hovermode="x unified", legend=dict(orientation="v", x=1.01, y=1,
                                           bgcolor="rgba(0,0,0,0.35)"),
        margin=dict(l=50, r=210, t=60, b=40))
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
