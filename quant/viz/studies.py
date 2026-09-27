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


def _target_path(r, x0, x1, rr, ref_label):
    """A trade's target as segments of (x, y, one label per point): the original level, stepping
    to where it was MOVED if the tp-move rule moved it, plus the second target the trade carried."""
    lab = f"{rr:g}R" if rr else ""
    out = []
    tm = getattr(r, "target_moved_time", None)
    nt = getattr(r, "target_moved_to", np.nan)
    if tm is not None and pd.notna(tm) and nt == nt:
        out.append(([x0, tm, tm, x1], [r.target, r.target, nt, nt],
                    [lab, lab, "moved to", "moved to"]))
    else:
        out.append(([x0, x1], [r.target, r.target], [lab, lab]))
    rt = getattr(r, "ref_target", np.nan)
    if rt == rt:
        out.append(([x0, x1], [rt, rt], [ref_label, ref_label]))
    return out


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


_MTF_EMA_SHAPES = ("('1h', 200)  |  ('1h', [50, 200])  |  ('1h', 50, 100, 200)  |  "
                   "[('1h', 200), ('4h', 200)]")


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
    # one timeframe followed by its period(s): ('1h', 200), ('1h', [50, 200]) and the loose
    # ('1h', 50, 100, 200) are all the same thing, so all three are accepted
    if isinstance(spec, (tuple, list)) and len(spec) >= 2 and isinstance(spec[0], str):
        tf, rest = spec[0], spec[1:]
        pers = []
        for chunk in rest:
            pers.extend(chunk if isinstance(chunk, (list, tuple)) else [chunk])
        out = []
        for pp in pers:
            try:
                out.append((tf, int(pp)))
            except (TypeError, ValueError):
                raise ValueError(
                    f"mtf_emas: {pp!r} is not an EMA period. Got {tuple(spec)!r}; if you meant "
                    f"several timeframes, each needs its own period — {_MTF_EMA_SHAPES}") from None
        return out
    out = []
    for item in spec:
        out.extend(_norm_mtf_ema(item))
    return out


def mtf_ema_timeframes(spec, *, chart_tf=None):
    """The timeframes an `mtf_emas` spec refers to, minus the chart's own.

    Each of these needs a REAL frame. Resampling the chart frame only works downward: asking for a
    1m EMA on a 5m chart resamples 11,521 bars into 11,521 bars and hands back the 5m EMA under a 1m
    name — which is exactly what happened, identical on 100% of bars and 310 points from the truth.
    """
    out = []
    for tf, _ in _norm_mtf_ema(spec):
        if chart_tf and str(tf).lower() == str(chart_tf).lower():
            continue
        if tf not in out:
            out.append(tf)
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
#: higher-timeframe EMA colours — deliberately unlike the chart's own EMAs, which use the template
#: defaults. One hue per TIMEFRAME rather than one per line: asking for 50/100/200 on two
#: timeframes is six lines, and six unrelated colours make it impossible to see which three belong
#: together. Within a timeframe the PERIOD sets the weight, so the slow line reads as the slow line.
#: "no timeframe chosen" — distinct from None, which is the chart's OWN timeframe
_SENTINEL = object()

_MTF_EMA_SHADES = ["#42a5f5", "#ab47bc", "#26c6da", "#ffa726", "#7e57c2", "#ec407a"]
_FVG_COL = {"bull": "#26a69a", "bear": "#ef5350"}
# S/R reads as STRUCTURE, so it stays neutral and lets WEIGHT carry the timeframe -- the same
# convention the trendlines use. Coarsest is brightest, because that is the level that matters.
_SR_SHADES = ["#e3e8ee", "#c2cad3", "#a3acb6", "#8e9aa6", "#77818b", "#636c75"]


def _short(label, n=34):
    """Menu labels, capped. A plotly dropdown is as wide as its widest row and the menus sit at
    fixed x positions, so one long occurrence label ("5m.triple_top#0 @ Jun-04 04:20 (confirms
    Jun-05 06:15)") pushed its menu straight over the next one. The identity is at the FRONT of
    every label here, so trimming the tail costs nothing."""
    label = str(label)
    return label if len(label) <= n else label[:n - 1].rstrip() + "\u2026"


def _row_active(fig, idx, n_buttons):
    """Which button in an `all | ...timeframes... | none` row reflects the CURRENT state.

    A row that opens on "all" while every layer is `legendonly` is simply lying about the chart.
    Mixed states highlight nothing (-1) rather than pick a button that is not true.
    """
    vis = [fig.data[i].visible for i in idx]
    if all(v is True for v in vis):
        return 0
    if not any(v is True for v in vis):
        return n_buttons - 1          # the trailing "none"
    return -1


def _rgba(hexc, alpha):
    """#rrggbb -> an rgba() string. A box needs a fill far more transparent than its border."""
    h = hexc.lstrip("#")
    return "rgba(%d,%d,%d,%s)" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def pattern_figure(df, *, patterns=None, time_col: str = "t", title=None, height: int = 820,
                   sr_zones: bool = True, zones_left=None, zones_right=None,
                   cluster_pct=None, min_touches=None, max_zones: int = 40,
                   sr_sensitivity="normal", sr_break_bars=None, sr_anchor=None,
                   extend: int = 12, start_visible=(), detect_kwargs=None, tuned: bool = False,
                   per_pattern=None, trendlines: bool = False, trendline_kwargs=None,
                   emas=(20, 50, 100, 200), view_start=None, line_picker: bool = True,
                   zone_lookback_bars: int = 2000, tf_frames=None, tf_patterns=None,
                   tf_trendlines=None, max_picker_instances: int = 60,
                   max_shapes_per_tf: int = 40, y_from_price: bool = True,
                   mtf_emas=None, ema_frames=None, emas_on: bool = True,
                   legend_instances=None, trades=None, trade_tp_rr=None,
                   chart_tf=None, max_trade_picker: int = 40,
                   ref_target_label: str = "second target", exit_labels: bool = False,
                   fvgs: bool = False, fvg_kwargs=None, fvg_frames=None, tf_fvgs=None,
                   max_fvg_picker: int = 6, fvg_extend: bool = True,
                   fvg_show=("bull", "bear"),
                   sr_frames=None, tf_sr=None, max_sr_picker: int = 6,
                   rewind: bool = False, rejected=None, max_missed_picker: int = 8):
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

    `rejected` is the table from `quant.research.rejected_setups` — every trendline break that did
    NOT become a trade, with the reason. One legend entry per reason, all starting `legendonly`, and
    added after the button rows so no menu payload covers them: the chart opens exactly as it does
    without them, and no existing control can disturb them.

    `sr_sensitivity` is the S/R knob: "tight" | "normal" | "loose", resolved PER LAYER from the
    tuned table in `quant.indicators.sr_params`, so a 4h layer gets 4h settings on a 5m chart.
    "normal" is the forward-validated best (experiments/sr_zone_search.py). `None` is LEGACY
    and reproduces the pre-2026-09-27 zones exactly. `zones_left`, `zones_right`,
    `cluster_pct`, `min_touches`, `sr_break_bars` and `sr_anchor` override the table when
    given, so an explicit setting is never silently replaced.

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

    `legend_instances` PINS chosen families so that every occurrence of them becomes its own
    permanent legend entry, e.g. `["15m::resistance_trendline", "5m::resistance_trendline",
    "1m::channel_up"]`, or `"all"`, or a bare timeframe like `"5m"` for all of its families. This is
    the only way to build a selection ACROSS families — one 15m line plus two 5m lines plus a few 1m
    shapes — because a plotly dropdown is single-choice and its buttons are stateless, so each press
    replaces the selection rather than adding to it. The legend has no such limit: entries toggle
    independently, accumulate, and it scrolls. Pinned entries start off, and the dropdowns are then a
    bulk show/hide over them rather than the way you choose. Pin only what you are working on: every
    occurrence of every family would be a legend hundreds of rows long.

    `trades` overlays a backtest's trades — `SimResult.trades`, or that frame with a `trigger`
    column added. Four legend groups, because the questions are different: the entry marker (side,
    R, close reason, what triggered it), the entry level, the stop, and the exit marker coloured by
    outcome. Stop and target run from entry to exit, so the box you see IS the trade.
    `trade_tp_rr` draws the target at that many R; omit it and no target line is drawn, since the
    engine records the stop per trade but not the target.

    `mtf_emas` draws HIGHER-TIMEFRAME EMAs, taking the same spec as `scenario_figure`:
    `("1h", 200)`, `("1h", [50, 200])`, or `[("1h", 200), ("4h", 200)]`. A frame already loaded in
    `tf_frames` is reused (it has the deeper history); otherwise the chart frame is resampled. Each
    line is a STEP that changes only at its own bar's close, because that is the only moment the
    value is final — an interpolated 1h EMA shows a number nobody could have read at the time. They
    are ON by default; `"-1h_ema200"` in `start_visible` turns one off, and `emas_on=False` starts
    them all in the legend but off the chart. `ema_frames={"1m": df_1m}` supplies the history for a
    timeframe that is not among `tf_frames` — needed for any timeframe FINER than the chart, which
    is refused rather than resampled, because upsampling returns the chart's own EMA under another
    name. Hue is the TIMEFRAME and
    weight is the PERIOD within it: `[("5m", 50, 100, 200), ("15m", 50, 100, 200)]` is six lines, and
    six unrelated colours would hide which three belong together.

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

    def _tf_name_of(frame):
        """'1m', '15m', '4h' — read off the bars, so no caller has to tell us."""
        mins = bar_interval(frame, time_col=time_col) / pd.Timedelta(minutes=1)
        if not mins or mins != mins:
            return "chart"
        mins = int(round(mins))
        for div, unit in ((1440, "d"), (60, "h"), (1, "m")):
            if mins >= div and mins % div == 0:
                return f"{mins // div}{unit}"
        return f"{mins}m"

    base_tf = chart_tf or _tf_name_of(d)
    line_ref = {}    # "5m·S7" -> the first trace index of that line's isolatable pair

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
            visible=(False if f"-ema{p}" in start_visible
                     else (True if emas_on else "legendonly")),
            hovertemplate=f"EMA {p}: %{{y:.2f}}<extra></extra>"))

    base_min = max(1.0, bar_interval(d, time_col=time_col) / pd.Timedelta(minutes=1))

    if mtf_emas:
        # HIGHER-TIMEFRAME EMAs. Drawn as a STEP that changes only at each HTF bar's close, because
        # that is the only moment the value is final — a 1h EMA interpolated across the hour shows a
        # number nobody could have read at the time. Same rule as every other event on this chart.
        from ..indicators.overlays import ema as _ema_of
        _pairs = _norm_mtf_ema(mtf_emas)
        _tf_order = list(dict.fromkeys(tf for tf, _ in _pairs))
        _by_tf = {tf: sorted({p for t, p in _pairs if t == tf}) for tf in _tf_order}
        for etf, per in _pairs:
            src = (ema_frames or {}).get(etf)
            if src is None:
                src = (tf_frames or {}).get(etf)   # `or` on a DataFrame is ambiguous
            if src is None:
                if tf_minutes(etf) < base_min:
                    # RESAMPLING ONLY WORKS DOWNWARD. Upsampling a 5m frame to 1m cannot invent the
                    # bars that were never loaded: it returns the same rows and the "1m" EMA comes
                    # out byte-identical to the 5m one -- 310 points from the real answer, with
                    # nothing on the chart to say so. Refuse instead.
                    raise ValueError(
                        f"mtf_emas asks for a {etf} EMA on a {base_tf} chart, but {etf} is FINER "
                        f"than the chart and no {etf} frame was given. Resampling would silently "
                        f"return the {base_tf} EMA. Load it — add {etf!r} to the frames you pass as "
                        f"`ema_frames` (or to `tf_frames`).")
                # coarser: resampling the chart frame is legitimate, though its history is capped by
                # the chart frame's start, which a long period may not have enough of
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
            # hue says which timeframe, weight says which period within it
            _ranks = _by_tf[etf]
            _r = _ranks.index(per) / max(1, len(_ranks) - 1)
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
                line=dict(width=round(1.3 + 1.2 * _r, 2),
                          color=_MTF_EMA_SHADES[_tf_order.index(etf) % len(_MTF_EMA_SHADES)]),
                opacity=None if _r >= 1.0 else round(0.65 + 0.35 * _r, 2),
                visible=(False if f"-{etf}_ema{int(per)}" in start_visible
                         else (True if emas_on else "legendonly")),
                hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"))

    def _detect(frame, want):
        """One detection pass for a frame, honouring tuned / per-pattern / shared settings.

        AN EMPTY LIST MEANS NONE. `want=None` is "every pattern"; `want=[]` used to mean the same
        thing, so every notebook asking for trendlines-only (`patterns=[]`) was quietly detecting
        and drawing the whole 22-pattern library -- most of the trace count, most of the dropdown,
        and the 4.3 s `_best_line` bill. Skipped on an empty frame so the columns still line up.
        """
        if want is not None and len(want) == 0:
            return [], chart_pattern_shapes(frame.iloc[:0], **(detect_kwargs or {}))
        if tuned or per_pattern:
            spec = per_pattern or tuned_specs()
            nm = list(want) if want else list(spec)
            return nm, chart_pattern_shapes_tuned(frame, specs=spec, patterns=nm,
                                                  **(detect_kwargs or {}))
        nm = list(want) if want else list(_PD)
        return nm, chart_pattern_shapes(frame, patterns=nm, **(detect_kwargs or {}))

    # families whose every occurrence gets a permanent legend entry — resolved to the `tf::name`
    # keys used below, so a bare timeframe or a bare pattern name both work
    _pin_all = isinstance(legend_instances, str) and legend_instances.lower() == "all"

    def _norm(x):
        # patterns are named with underscores and trendlines with a space, and nobody should have
        # to remember which; "5m::resistance trendline" and "5m::resistance_trendline" are one key
        return str(x).strip().lower().replace(" ", "_").replace("·", "::").replace("__", "_")

    _pin = set()
    if legend_instances and not _pin_all:
        raw = [legend_instances] if isinstance(legend_instances, str) else list(legend_instances)
        _pin = {_norm(k) for k in raw}

    def _pinned(tf, name):
        """Is this family pinned? A BARE name means the chart's own timeframe, matching how
        `patterns=` and `start_visible=` name things; `*::name` means the family on every one."""
        if _pin_all:
            return True
        keys = {_norm(f"*::{name}")}
        keys.add(_norm(name) if tf is None else _norm(f"{tf}::{name}"))
        if tf is not None:
            keys.add(_norm(tf))
        return bool(_pin & keys)

    # Lines named as the CAUSE of a trade escape both caps. The caps exist to keep the chart
    # openable, but a trade whose line was capped away cannot show you what it broke, which is the
    # one question the trade picker is there to answer.
    _wanted = set()
    if trades is not None and len(trades) and "cause" in getattr(trades, "columns", ()):
        for c in trades["cause"].astype(str):
            _wanted.update(x for x in c.split("+") if x and x != "?")

    TRADE_TRACES = 4    # entry, stop, target, markers — one trace each so each has
                        # its own colour, which is the whole point
    picker = []      # (family, label, first_trace_index, sort_key) for every isolatable instance
    trade_picker = []   # (tid, row, first_trace_index, [cause tags], stop_walked)
    pinned_traces = set()
    n_pin = {}       # per-family counter, so two lines born on the same bar get distinct labels
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

        # THE CAP IS DECIDED BEFORE ANYTHING IS DRAWN, and then bounds the drawing and the dropdown
        # equally. Trimming only the dropdown is what produced a legend reading "(11)" beside a menu
        # offering nine, with two lines on the chart that could not be isolated. Counting the
        # families first is what lets one budget be shared without starving whichever came last.
        lines_tbl = None
        n_fams = int(shapes["pattern"].nunique()) if len(shapes) else 0
        if draw_lines:
            from ..indicators.trendlines import trendline_table
            lines_tbl = trendline_table(dl, **(trendline_kwargs or {}))
            if len(lines_tbl) and vl > 0:
                lines_tbl = lines_tbl[lines_tbl["end_i"] >= vl]   # died before the window opened
            n_fams += int(lines_tbl["kind"].nunique()) if len(lines_tbl) else 0
        cap = int(max_shapes_per_tf) if max_shapes_per_tf else 10 ** 6
        if picker_budget is not None and n_fams:
            cap = max(1, min(cap, -(-int(picker_budget) // n_fams)))

        for name in names_:
            sub = shapes[shapes["pattern"] == name] if len(shapes) else shapes
            if not len(sub):
                continue
            if len(sub) > cap:
                sub = sub.sort_values("confirm_i").tail(cap)     # the most recent are the live ones
            up = _PD.get(name, 0) > 0
            colr = "#2ecc71" if up else ("#e74c3c" if _PD.get(name, 0) < 0 else "#f0a30a")
            key = name if chart_tf else f"{tf}::{name}"
            vis = True if key in start_visible else "legendonly"
            lx, ly = [], []          # the two fitted boundaries, extended across the shape
            mx, my = [], []          # the pivots those lines are fitted through
            nx, ny = [], []          # the boundary that broke
            tx, ty = [], []          # measured-move target
            sx, sy = [], []          # protective stop
            tf_nm = tf or base_tf
            for k_, r in enumerate(sub.itertuples()):
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
                customdata=[[int(r.confirm_i) - int(max(r.pivot_i)), f"{tf_nm}·{name}#{k_}"]
                            for k_, r in enumerate(sub.itertuples())],
                hovertemplate=("<b>%{customdata[1]}</b> CONFIRMED here<br>entry %{y:.2f}"
                               "<br>%{customdata[0]} bars after its last pivot<extra></extra>")))

            if line_picker:
                # EVERY shape drawn above gets an entry. `sub` was already capped, so the count in
                # the legend and the length of the dropdown are the same number by construction.
                for _k, r in enumerate(sub.itertuples()):
                    ptag = f"{tf_nm}·{name}#{_k}"
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
                    short = f"{pd.to_datetime(str(tl_[a2])):%b-%d %H:%M}"
                    line_ref[ptag] = len(fig.data)
                    picker.append((label,
                                   f"{ptag}  @ {short}"
                                   f"  (confirms {pd.to_datetime(str(ev(r.confirm_i))):%b-%d %H:%M})",
                                   len(fig.data), pd.Timestamp(str(tl_[a2])).value))
                    used += 1
                    # the two traces share a legendgroup so that ONE legend entry moves both. The
                    # entry only appears once a family is selected — see the menu block below.
                    grp = f"i#{len(fig.data)}"
                    pin = _pinned(tf, name)
                    if pin:
                        pinned_traces.update((len(fig.data), len(fig.data) + 1))
                        n_pin[label] = n_pin.get(label, 0) + 1
                    fig.add_trace(go.Scattergl(
                        x=ix, y=iy, mode="lines",
                        # pinned entries are LISTED but off, so the legend is the selection
                        visible="legendonly" if pin else False, showlegend=pin,
                        # the same id-first naming the trendlines use, so a legend row, a hover
                        # and a trade's `cause` all say the same word for the same object
                        name=f"{ptag} · {short}" if pin else short,
                        legendgroup=grp,
                        line=dict(color=colr, width=round(width + 0.6, 2)),
                        hovertemplate=f"<b>{ptag}</b>  {label}<extra></extra>"))
                    fig.add_trace(go.Scattergl(
                        x=list(r.touch_t) + [ev(r.confirm_i)],
                        y=list(r.touch_price) + [float(r.entry)],
                        mode="markers", visible="legendonly" if pin else False,
                        showlegend=False, legendgroup=grp,
                        marker=dict(size=[dotsz + 2] * len(r.touch_t) + [dotsz + 8], color=colr,
                                    symbol=["circle"] * len(r.touch_t) + ["star"],
                                    line=dict(width=1.5, color="#fff")),
                        hovertemplate="touch / CONFIRMED<extra></extra>"))

        if draw_lines:
            tl = lines_tbl if lines_tbl is not None else pd.DataFrame(columns=["kind"])
            for kind, base_label, colr in ((-1, "support trendline", "#2ecc71"),
                                           (+1, "resistance trendline", "#e74c3c")):
                sub = tl[tl["kind"] == kind] if len(tl) else tl
                if not len(sub):
                    continue
                tf_nm0 = tf or base_tf
                if len(sub) > cap:
                    # the same cap the shapes get: a 1m layer offered 80 resistance lines over nine
                    # days, which is a mesh rather than a chart. The most recent are the live ones —
                    # plus any line a trade actually broke, which is never dropped.
                    want = sub[[f"{tf_nm0}·{i}" in _wanted for i in sub["id"]]]
                    sub = pd.concat([sub.sort_values("end_i").tail(cap), want]).drop_duplicates(
                        subset=["id"]).sort_values("end_i")
                label = pref + base_label
                _register(label, lgp + base_label)
                key = (base_label.split()[0] + "_trendline") if chart_tf else f"{tf}::{base_label}"
                vis = True if key in start_visible else "legendonly"
                lx, ly, dx, dy, bx, by, lcd, dcd = [], [], [], [], [], [], [], []
                per_line = []      # one entry per line, for the picker
                tf_nm = tf or base_tf
                for r in sub.itertuples():
                    tag = f"{tf_nm}·{getattr(r, 'id', '?')}"    # the name everything else uses
                    # from the FIRST TOUCH, not from born_i: the line is fitted through pivots that
                    # all sit before it became known, so drawing from born_i left every touch dot
                    # outside the segment. All 36 lines in the sample had this.
                    si, sy = int(r.start_i), r.y_start
                    if si < vl:
                        # CLIPPED TO THE WINDOW, with y INTERPOLATED along the fitted line. A line
                        # can legitimately start ten days before the period you asked to see, and
                        # drawn from there it made autoscale rescale to all ten. Moving x without
                        # recomputing y would have tilted the line, which is worse than the bug.
                        si, sy = vl, float(r.slope) * vl + float(r.intercept)
                    lx += [tl_[si], ev(r.end_i), None]
                    ly += [sy, r.y_end, None]
                    lcd += [tag, tag, tag]
                    # touches outside the window are not drawn: they are the pivots that DEFINED
                    # the line, off-screen by definition, and each one dragged the axis back too
                    _left = tl_[vl] if vl > 0 else None
                    _tt = [(a_, b_) for a_, b_ in zip(r.touch_t, r.touch_price)
                           if _left is None or a_ >= _left]
                    dx += [a_ for a_, _ in _tt] + [None]
                    dy += [b_ for _, b_ in _tt] + [None]
                    dcd += [tag] * (len(_tt) + 1)
                    if r.broke:
                        bx.append(ev(r.end_i)); by.append(r.y_end)
                    per_line.append((
                        f"{label} @ {pd.to_datetime(str(tl_[si])):%b-%d %H:%M}"
                        f"  ({int(r.touches)} touches, {'broke' if r.broke else 'faded'})",
                        # the ISOLATABLE copy is clipped identically -- otherwise picking the
                        # line out of the dropdown put the off-screen geometry straight back
                        [tl_[si], ev(r.end_i)], [sy, r.y_end],
                        [a_ for a_, _ in _tt], [b_ for _, b_ in _tt], si, label,
                        int(r.born_i), int(r.end_i), bool(r.broke), tag, int(r.touches)))
                fig.add_trace(go.Scattergl(
                    x=lx, y=ly, mode="lines",
                    name=f"{label} ({len(sub)}, {int(sub.broke.sum())} broke)",
                    legendgroup=lgp + base_label, visible=vis, opacity=alpha,
                    line=dict(color=colr, width=max(0.8, round(width - 0.6, 2)), dash="dash"),
                    customdata=lcd,
                    hovertemplate=f"<b>%{{customdata}}</b>  {label}<extra></extra>"))
                fig.add_trace(go.Scattergl(
                    x=dx, y=dy, mode="markers", legendgroup=lgp + base_label, visible=vis,
                    showlegend=False, opacity=alpha,
                    marker=dict(size=max(3, dotsz - 1), color=colr,
                                line=dict(width=1, color="#111")),
                    customdata=dcd,
                    hovertemplate=f"<b>%{{customdata}}</b> touch<extra></extra>"))
                if bx:
                    fig.add_trace(go.Scattergl(
                        x=bx, y=by, mode="markers", legendgroup=lgp + base_label, visible=vis,
                        showlegend=False, opacity=alpha,
                        marker=dict(size=dotsz + 4, color=colr, symbol="x", line=dict(width=1.5)),
                        hovertemplate=f"{label} BREAK<extra></extra>"))
                if line_picker or (_wanted & {p[10] for p in per_line}):
                    # one entry per line DRAWN, for the same reason as the shapes above
                    for lbl, xs, ys, txs, tys, si, fam, bi, ei, broke, tag, ntouch in per_line:
                        if not line_picker and tag not in _wanted:
                            continue
                        lbl = _short(f"{tag}  {lbl.split(' @ ', 1)[-1]}")
                        line_ref[tag] = len(fig.data)
                        picker.append((fam, lbl, len(fig.data), pd.Timestamp(str(tl_[si])).value))
                        used += 1
                        grp = f"i#{len(fig.data)}"
                        pin = _pinned(tf, base_label)
                        if pin:
                            pinned_traces.update((len(fig.data), len(fig.data) + 1))
                            n_pin[label] = n_pin.get(label, 0) + 1
                        _when = f"{pd.to_datetime(str(tl_[si])):%b-%d %H:%M}"
                        # same dashed style as the grouped view, just thicker — isolating a line
                        # should not make it look like a different kind of object
                        fig.add_trace(go.Scattergl(
                            x=xs, y=ys, mode="lines",
                            visible="legendonly" if pin else False, showlegend=pin,
                            # id first, then what it is, then when — a legend row has to be
                            # readable on its own, and the id is what ties it to a trade
                            name=(f"{tag} {base_label.split()[0]} · {_when}" if pin
                                  else f"{tag}  {_when} ({'broke' if broke else 'faded'})"),
                            legendgroup=grp,
                            line=dict(color=colr, width=round(width + 0.4, 2), dash="dash"),
                            hovertemplate=(f"<b>{tag}</b>  {label}<br>{ntouch} touches, "
                                           f"{'broke' if broke else 'faded'}<extra></extra>")))

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
                            x=mx2, y=my2, mode="markers",
                            visible="legendonly" if pin else False, showlegend=False,
                            legendgroup=grp,
                            marker=dict(size=szs, color=colr, symbol=syms,
                                        line=dict(width=1.6, color="#fff")),
                            text=txt, hovertemplate="%{text}<extra></extra>"))
        return used


    # ---- FAIR VALUE GAPS -----------------------------------------------------------------
    def _draw_fvgs(dl, *, tf, alpha, width, budget):
        """One timeframe's gaps as filled boxes. Returns the picker budget used.

        ONE TRACE PAIR FOR THE WHOLE LAYER, not one per gap: a `fill="toself"` Scattergl takes any
        number of `None`-separated polygons, so four hundred boxes cost the same two traces as one.
        That matters more here than anywhere else on this chart -- a 1m layer over a week produces
        thousands of gaps, and every picker button carries a visibility vector as long as the trace
        list, so the trace count is what decides whether the chart opens at all.

        Isolatable entries come out of the same budget the shapes and trendlines use, capped again
        by `max_fvg_picker`, and the cap is applied BEFORE drawing so the legend count and the
        dropdown can never disagree -- the bug that left a legend reading (11) beside a menu of 9.
        """
        from ..indicators.fvg import fvg_table

        used = 0
        tl_ = dl[time_col].to_numpy() if time_col in dl.columns else np.arange(len(dl))
        vl = _view_index(dl, time_col)
        n = len(dl)
        own = tf is None
        # An event read off bar i of a HIGHER timeframe is only knowable when that bar CLOSES, so a
        # gap is drawn from its third candle's close -- never from the middle candle where it
        # visually sits, which would be a two-bar lookahead drawn on the chart.
        step = np.timedelta64(0, "ns")
        bar = np.timedelta64(0, "ns")
        if time_col in dl.columns and n > 1:
            bar = np.timedelta64(bar_interval(dl, time_col=time_col))
            if not own:
                step = bar

        try:
            gt = fvg_table(dl, **(fvg_kwargs or {}))
        except Exception as exc:          # a layer with too few bars must not take the chart down
            import warnings as _w
            _w.warn("fvg: %s layer skipped -- %s" % (tf or base_tf, exc), stacklevel=2)
            return 0
        if not len(gt):
            return 0
        # a gap that both formed AND died inside the warm-up is history: it taught the detector
        # something, but it is not on the window you asked to look at. One that formed there and is
        # still open is very much live, so it stays.
        # `end_i`, not `dead_i`: a gap that EXPIRED before the window opens is not on screen at
        # all, and keying off mitigation alone kept every one of them
        _end = (gt["end_i"] if "end_i" in gt.columns else gt["dead_i"]).to_numpy(np.int64)
        gt = gt[(gt["born_i"].to_numpy(np.int64) >= vl) | (_end < 0) | (_end >= vl)]
        if not len(gt):
            return 0
        alpha_l = None if (alpha is None or alpha >= 1.0) else alpha
        pref = "" if own else "%s · " % tf      # the same separator every other layer uses
        lgp = "" if own else "%s::" % tf
        tf_nm = tf or base_tf
        right = tv[-1] if len(tv) else None

        for side in fvg_show:
            sub = gt[gt["kind"] > 0] if side == "bull" else gt[gt["kind"] < 0]
            if not len(sub):
                continue
            base_label = "FVG %sish" % side
            label = pref + base_label
            key = ("fvg_%s" % side) if own else "%s::%s" % (tf, base_label)
            vis = True if key in start_visible else "legendonly"
            colr = _FVG_COL[side]
            # The same registration `_draw_layer` does for a shape family. The key is the
            # PREFIXED label: "FVG bullish" is the name on four different timeframes at once, so
            # keying by it made every layer's menu show the 4h layer's counts.
            fam_lg[label] = lgp + base_label
            if label not in tf_fams.setdefault(tf, []):
                tf_fams[tf].append(label)

            # THE CAP, decided before a single box is drawn, and applied PER TIMEFRAME PER
            # DIRECTION rather than from one shared pool -- a single pool was drained entirely by
            # the coarsest layer, leaving the 5m gaps you were actually looking at un-isolatable.
            # Biggest first: if only six of four hundred gaps get an entry, they should be the six
            # a strategy might care about, not the six that happened to form earliest.
            room = max(0, int(budget))
            pick = sub.sort_values("size_atr", ascending=False).head(room) if room else sub.iloc[:0]
            pick_ids = set(pick["id"])

            xs, ys, hx, hy, hc = [], [], [], [], []
            per_gap = []
            left = tv[0] if len(tv) else None
            for r in sub.itertuples():
                x0 = tl_[int(r.born_i)] + step
                # A gap that formed in the WARM-UP and is still open is worth drawing -- but from
                # the left edge of the window, not from its birth. Drawn from birth, a 4h gap
                # reached 131 days back and autoscale rescaled the chart to all of it.
                if left is not None and x0 < left:
                    x0 = left
                _e = int(getattr(r, "end_i", r.dead_i))
                if _e >= 0:
                    # filled OR expired -- either way this is the bar it stopped being live, which
                    # is the bar the counts and the veto stop seeing it
                    x1 = tl_[_e] + step
                else:
                    x1 = right if (fvg_extend and right is not None) else x0 + bar * 3
                if x1 is None or x1 <= x0:
                    x1 = x0 + bar
                xs += [x0, x1, x1, x0, x0, None]
                ys += [r.bot, r.bot, r.top, r.top, r.bot, None]
                tag = "\u00b7".join((tf_nm, str(r.id)))   # `5m\u00b7FB12`, like `5m\u00b7S7`
                _why = str(getattr(r, "end_reason", "filled" if r.dead_i >= 0 else "open"))
                if _e < 0:
                    st = "still open"
                elif _why == "expired":
                    st = "expired %s — never filled, too old to count" % pd.to_datetime(
                        str(tl_[_e])).strftime("%b-%d %H:%M")
                else:
                    st = "filled %s" % pd.to_datetime(str(tl_[_e])).strftime("%b-%d %H:%M")
                txt = (f"<b>{tag}</b>  {side}ish FVG<br>"
                       f"{r.bot:,.2f} - {r.top:,.2f}  ({r.size_atr:.2f} ATR)<br>{st}")
                hx.append(x0 + (x1 - x0) / 2)
                hy.append(r.mid)
                hc.append(txt)
                if r.id in pick_ids:
                    per_gap.append((tag, label, [x0, x1, x1, x0, x0],
                                    [r.bot, r.bot, r.top, r.top, r.bot], hx[-1], r.mid, txt,
                                    pd.Timestamp(str(x0)).value))

            _mine = fvg_layer_ix.setdefault(tf_nm, [])
            _mine.append(len(fig.data))
            _mine.append(len(fig.data) + 1)
            fig.add_trace(go.Scattergl(
                x=xs, y=ys, fill="toself", mode="lines",
                name="%s (%d, %d open)" % (label, len(sub),
                                           int(((sub["end_i"] if "end_i" in sub.columns
                                                 else sub["dead_i"]) < 0).sum())),
                legendgroup=lgp + base_label, visible=vis, opacity=alpha_l, meta="fvg",
                line=dict(color=colr, width=max(0.5, round(width - 1.8, 2))),
                fillcolor=_rgba(colr, 0.13), hoverinfo="skip"))
            # hover lives on an INVISIBLE centroid marker: a filled polygon reports its outline, so
            # hovering the middle of a box -- where the cursor actually goes -- said nothing.
            fig.add_trace(go.Scattergl(
                x=hx, y=hy, mode="markers", legendgroup=lgp + base_label, visible=vis,
                showlegend=False, opacity=alpha_l, meta="fvg",
                marker=dict(size=max(6, int(width * 3)), color=colr, opacity=0.001),
                customdata=hc, hovertemplate="%{customdata}<extra></extra>"))

            for tag, fam, bx, by, cx, cy, txt, sort_key in per_gap:
                line_ref[tag] = len(fig.data)
                picker.append((fam, _short("%s  %s" % (tag, txt.split("<br>")[1])),
                               len(fig.data), sort_key))
                used += 1
                grp = "i#%d" % len(fig.data)
                fig.add_trace(go.Scattergl(
                    x=bx, y=by, fill="toself", mode="lines", name=tag,
                    legendgroup=grp, visible=False, showlegend=True, meta="fvg",
                    line=dict(color=colr, width=2.0), fillcolor=_rgba(colr, 0.22),
                    hoverinfo="skip"))
                fig.add_trace(go.Scattergl(
                    x=[cx], y=[cy], mode="markers", legendgroup=grp, visible=False,
                    showlegend=False, meta="fvg",
                    marker=dict(size=14, color=colr, opacity=0.001),
                    customdata=[txt], hovertemplate="%{customdata}<extra></extra>"))
        return used

    fvg_layer_ix = {}   # timeframe name -> the trace indices of its boxes, for the toggle row

    if fvgs:
        # EVERY gap layer, before a single line is drawn. Two reasons it is one block here rather
        # than a call inside each layer: plotly paints in trace order, so a box added later would
        # sit ON TOP of the trendline it is supposed to be background for; and the picker budget is
        # easier to reason about spent in one place.
        #
        # `fvg_frames` is for timeframes you want GAPS from but not patterns or trendlines from. It
        # is kept separate from `tf_frames` deliberately: tf_frames is what gets FITTED and, in the
        # trading notebook, what gets TRADED -- adding a frame there just to see its gaps would
        # quietly start trading its lines.
        _fvg_layers = {None: d}
        for _k, _f in list((tf_frames or {}).items()) + list((fvg_frames or {}).items()):
            if _k not in _fvg_layers and (tf_fvgs is None or tf_fvgs.get(_k, True)):
                _fvg_layers[_k] = _f.reset_index(drop=True)
        _fvg_budget = int(max_fvg_picker)      # per layer, per direction
        for _k in sorted(_fvg_layers, key=lambda k: -tf_minutes(k) if k else -base_min):
            _rank = 0.0 if _k is None else float(np.log2(max(tf_minutes(_k), 1) / base_min))
            _w = min(4.4, 2.6 + 0.55 * _rank) if _rank >= 0 else max(1.2, 2.6 + 0.45 * _rank)
            _draw_fvgs(_fvg_layers[_k], tf=_k, width=_w,
                       alpha=1.0 if _rank >= 0 else 0.55, budget=_fvg_budget)

    sr_layer_ix = {}    # timeframe name -> its trace indices, for the toggle row

    def _draw_sr(dl, *, tf, width, alpha, shade, budget):
        """One timeframe's S/R zones. Returns the picker budget used.

        A level fitted on 1h bars is a different level from one fitted on 5m bars, so each frame is
        clustered in ITS OWN bar space and only becomes a chart coordinate when a bar index turns
        into a timestamp -- the same reasoning the pattern and trendline layers use.
        """
        own = tf is None
        lv = _view_index(dl, time_col)
        tl_ = dl[time_col].to_numpy() if time_col in dl.columns else np.arange(len(dl))
        # a zone read off bar i of a HIGHER timeframe is only knowable when that bar CLOSES
        step = np.timedelta64(0, "ns")
        if not own and time_col in dl.columns and len(dl) > 1:
            step = np.timedelta64(bar_interval(dl, time_col=time_col))

        # built from a BOUNDED history ending at the view, not the whole warm-up — see the
        # docstring. The lookback is in THIS frame's bars, so a 4h layer gets 4h of depth per bar.
        z0 = max(0, lv - int(zone_lookback_bars)) if zone_lookback_bars else 0
        # THE KNOB, resolved for THIS layer's timeframe: a 4h layer on a 5m chart is tuned as 4h
        from ..indicators import sr_params
        _prm = sr_params(tf or chart_tf or '5m', sr_sensitivity, left=zones_left,
                         right=zones_right, cluster_pct=cluster_pct, min_touches=min_touches,
                         break_bars=sr_break_bars, anchor=sr_anchor)
        zt = sr_zone_table(dl.iloc[z0:].reset_index(drop=True), left=_prm['left'],
                           right=_prm['right'], cluster_pct=_prm['cluster_pct'],
                           max_zones=max_zones, time_col=time_col,
                           break_bars=_prm['break_bars'], anchor=_prm['anchor'])
        if len(zt):
            zt = zt.copy()
            zt[["start_i", "end_i"]] += z0
        zt = zt[zt["touches"] >= _prm['min_touches']]
        # A zone is a level price is respecting NOW. Two filters, both about relevance rather than
        # correctness: the segment has to reach the visible window at all, and its price has to be
        # near the prices on screen. Without the second, a long warm-up in a trending market fills
        # the chart with levels from 20% below — 767 segments, none of them next to price.
        if len(zt):
            # ONE line per zone at its CURRENT level. A zone emits a row per price revision, so
            # drawing every row stacks a zone's whole history of slightly-different levels on the
            # chart — 1,366 segments for 40 zones. The last revision is the level in force.
            #
            # EXCEPT WHEN REWINDING. "The level in force" means TODAY's, painted across the whole
            # window — a quiet lookahead when the point is to see what was known at some past bar.
            # Keeping every revision as its own segment lets the rewind clip pick the one that was
            # live then, with no special-casing in the tool.
            if not rewind:
                zt = zt.sort_values("end_i").groupby("zone", as_index=False).tail(1)
            zt = zt[zt["price"].between(_lo - _pad, _hi + _pad)]
        if not len(zt):
            return 0

        tf_nm = tf or base_tf
        pref = "" if own else "%s \u00b7 " % tf
        lgp = "" if own else "%s::" % tf
        base_label = "S/R zones"
        label = pref + base_label
        key = "sr_zones" if own else "%s::%s" % (tf, base_label)
        vis = True if key in start_visible else "legendonly"
        alpha_l = 0.55 if (alpha is None or alpha >= 1.0) else round(0.55 * alpha, 2)
        fam_lg[label] = lgp + base_label
        if label not in tf_fams.setdefault(tf, []):
            tf_fams[tf].append(label)

        room = max(0, int(budget))
        pick = set(zt.sort_values("touches", ascending=False).head(room)["zone"]) if room else set()

        xs, ys, cd, hx, hy, hc = [], [], [], [], [], []
        per_zone = []
        for r in zt.itertuples():
            # drawn across the whole visible window: a level established earlier is still the level
            x_from = tv[0] if r.start_i < lv else (tl_[int(r.start_i)] + step)
            # a REVISION ends where the next one begins; only the last one runs to the right edge
            x_to = tv[-1]
            if rewind and int(r.end_i) < len(tl_) - 1:
                x_to = min(tv[-1], tl_[int(r.end_i)] + step)
            tag = "\u00b7".join((tf_nm, "Z%d" % int(r.zone)))
            txt = ("<b>%s</b>  S/R<br>%s<br>%d touches" % (tag, format(r.price, ",.2f"),
                                                           int(r.touches)))
            xs += [x_from, x_to, None]
            ys += [r.price, r.price, None]
            cd += [txt, txt, txt]
            hx.append(tv[len(tv) // 2] if len(tv) else x_from)
            hy.append(r.price)
            hc.append(txt)
            if r.zone in pick:
                per_zone.append((tag, label, [x_from, x_to], [r.price, r.price], txt,
                                 pd.Timestamp(str(x_from)).value, int(r.touches)))

        _mine = sr_layer_ix.setdefault(tf_nm, [])
        _mine += [len(fig.data), len(fig.data) + 1]
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines", name="%s (%d)" % (label, len(zt)),
            legendgroup=lgp + base_label, line=dict(color=shade, width=width),
            opacity=alpha_l, customdata=cd, visible=vis,
            hovertemplate="%{customdata}<extra></extra>"))
        # A horizontal line only reports a hit when the cursor is ON it. The midpoint marker gives
        # each zone somewhere to hover that is not a one-pixel target.
        fig.add_trace(go.Scattergl(
            x=hx, y=hy, mode="markers", legendgroup=lgp + base_label, visible=vis,
            showlegend=False, marker=dict(size=10, color=shade, opacity=0.001),
            customdata=hc, hovertemplate="%{customdata}<extra></extra>"))

        used = 0
        for tag, fam, bx, by, txt, sort_key, _t in per_zone:
            line_ref[tag] = len(fig.data)
            picker.append((fam, "%s  %s" % (tag, txt.split("<br>")[1]), len(fig.data), sort_key))
            used += 1
            grp = "i#%d" % len(fig.data)
            fig.add_trace(go.Scattergl(
                x=bx, y=by, mode="lines", name=tag, legendgroup=grp, visible=False,
                showlegend=True, line=dict(color=shade, width=max(2.0, width + 0.8)),
                customdata=[txt, txt], hovertemplate="%{customdata}<extra></extra>"))
            fig.add_trace(go.Scattergl(
                x=[bx[0] + (bx[1] - bx[0]) / 2], y=[by[0]], mode="markers", legendgroup=grp,
                visible=False, showlegend=False,
                marker=dict(size=12, color=shade, opacity=0.001),
                customdata=[txt], hovertemplate="%{customdata}<extra></extra>"))
        return used

    if sr_zones:
        # EVERY zone layer. `sr_frames` is for timeframes you want LEVELS from but not patterns or
        # trendlines from -- kept out of tf_frames on purpose, since that is what gets fitted and,
        # in the trading notebooks, traded.
        _sr_layers = {None: d}
        for _k, _f in (sr_frames or {}).items():
            if _k not in _sr_layers and (tf_sr is None or tf_sr.get(_k, True)):
                _sr_layers[_k] = _f.reset_index(drop=True)
        _order = sorted(_sr_layers, key=lambda k: -tf_minutes(k) if k else -base_min)
        # THE CHART'S OWN LAYER ALWAYS GETS THE SAME SHADE, whatever else is loaded. Indexing by
        # position meant it was the brightest when drawn alone and a different grey the moment a 4h
        # layer was added -- the same levels changing appearance because of an unrelated setting.
        # Anchored on the chart instead: coarser layers step brighter from it, finer ones darker.
        _anchor = _order.index(None)
        for _n, _k in enumerate(_order):
            _rank = 0.0 if _k is None else float(
                np.log2(max(tf_minutes(_k), 1) / base_min))
            # the cap has to clear the coarsest layer or the top two timeframes come out the
            # same weight, which is the whole point of weighting them
            _w = min(4.4, 2.0 + 0.42 * _rank) if _rank >= 0 else max(1.0, 2.0 + 0.4 * _rank)
            _draw_sr(_sr_layers[_k], tf=_k, width=round(_w, 2),
                     alpha=1.0 if _rank >= 0 else 0.7,
                     shade=_SR_SHADES[min(max(3 - (_anchor - _n), 0), len(_SR_SHADES) - 1)],
                     budget=max_sr_picker)

    # ---- the chart's own timeframe, then any extra ones ----------------------------------
    _draw_layer(d, tf=None, want=patterns, draw_lines=trendlines,
                width=2.6, dotsz=7, alpha=1.0, picker_budget=None)

    if tf_frames:
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


    if trades is not None and len(trades):
        # ---- TRADES: entry, stop, target, exit -------------------------------------------
        # Drawn as four legend groups rather than one, because the questions are different: "where
        # did it get in", "how much was risked", "what was it aiming at", "how did it end". Each
        # trade's stop and target run from its entry to its exit, so the box you see IS the trade.
        tq = trades.copy()
        for c in ("entry_time", "exit_time", "target_moved_time"):
            if c in tq.columns:
                tq[c] = pd.to_datetime(tq[c])
        risk = (tq["entry_price"] - tq["stop_price"]).abs()
        with np.errstate(invalid="ignore", divide="ignore"):
            tq["R"] = np.where(risk > 0, tq["pnl"] / (risk * tq["qty"]), np.nan)
        sgn = np.where(tq["side"].to_numpy() == "long", 1.0, -1.0)
        tq["target"] = tq["entry_price"] + sgn * float(trade_tp_rr or 0.0) * risk

        # A TRADE ID, and it goes on everything the trade owns. Four separate concatenated traces
        # meant the stop you were hovering could belong to any of forty-four trades; naming them all
        # `#07` is what lets you read one trade off the chart instead of inferring it from position.
        if "tid" not in tq.columns:
            tq = tq.sort_values("entry_time").reset_index(drop=True)
            tq["tid"] = [f"#{i + 1:02d}" for i in range(len(tq))]
        tq["cause"] = tq["cause"] if "cause" in tq.columns else ""

        ex, ey, sx, sy, gx, gy = [], [], [], [], [], []
        ecd, scd, gcd = [], [], []
        for r in tq.itertuples():
            x0, x1 = r.entry_time, (r.exit_time if pd.notna(r.exit_time) else tv[-1])
            ex += [x0, x1, None]; ey += [r.entry_price, r.entry_price, None]
            sx += [x0, x1, None]; sy += [r.stop_price, r.stop_price, None]
            ecd += [[r.tid, r.side, r.cause]] * 3
            scd += [[r.tid, r.side, r.cause]] * 3
            if trade_tp_rr:
                for sx_, sy_, lab_ in _target_path(r, x0, x1, trade_tp_rr, ref_target_label):
                    gx += sx_ + [None]; gy += sy_ + [None]
                    gcd += [[r.tid, r.side, r.cause, v] for v in lab_ + [""]]

        for X, Y, CD, nm, what, colr, dash, wid, grp in (
                (ex, ey, ecd, "trade entry level", "entry", "#90a4ae", "solid", 1.0, "trades"),
                (sx, sy, scd, "trade stop", "STOP", "#ef5350", "dot", 1.2, "trade_stop"),
                (gx, gy, gcd, "trade target", "TARGET %{customdata[3]}", "#26a69a", "dot", 1.2,
                 "trade_target")):
            if X:
                fig.add_trace(go.Scattergl(
                    x=X, y=Y, mode="lines", name=nm, legendgroup=grp,
                    line=dict(color=colr, width=wid, dash=dash), opacity=0.75,
                    customdata=CD,
                    hovertemplate=("<b>%{customdata[0]}</b> " + what + " %{y:.2f}"
                                   "<br>%{customdata[1]} off %{customdata[2]}<extra></extra>"),
                    visible=True if grp in start_visible or "trades" in start_visible
                    else "legendonly"))

        _extra = [c for c in ("trigger",) if c in tq.columns]
        # EXIT LABELS: a short tag drawn on every exit marker and a sentence in every hover, when
        # asked for AND the trades carry them (quant.research.exit_labels). Off, nothing changes.
        _lab = bool(exit_labels) and "exit_tag" in tq.columns
        _why_col = tq["exit_why"] if _lab else tq["close_reason"]
        for side, sym, colr in (("long", "triangle-up", "#26a69a"),
                                ("short", "triangle-down", "#ef5350")):
            g = tq[tq["side"] == side]
            if not len(g):
                continue
            cd = np.column_stack([g["R"].to_numpy(), g["pnl"].to_numpy(),
                                  _why_col.loc[g.index].to_numpy(),
                                  g["cause"].to_numpy(), g["tid"].to_numpy()])
            fig.add_trace(go.Scattergl(
                x=g["entry_time"], y=g["entry_price"], mode="markers",
                name=f"{side} entries ({len(g)})", legendgroup="trades",
                marker=dict(symbol=sym, size=11, color=colr,
                            line=dict(width=1.2, color="#fff")),
                customdata=cd,
                visible=True if "trades" in start_visible else "legendonly",
                hovertemplate=("<b>%{customdata[4]}</b>  " + side.upper()
                               + " entry %{y:.2f}<br>%{customdata[0]:.2f}R"
                               "  (%{customdata[1]:.2f})<br>closed: %{customdata[2]}"
                               "<br>broke %{customdata[3]}<extra></extra>")))

        done = tq[tq["exit_time"].notna()]
        if len(done):
            win = done["pnl"] > 0
            _cd = [_why_col.loc[done.index].to_numpy(), done["R"].to_numpy(), done["tid"].to_numpy()]
            _txt = {}
            if _lab:
                # THE TAG IS READ FROM customdata, never from `text`: rewind drops the markers after
                # its cut and filters customdata with them, but leaves `text` alone -- a text array
                # would slide every label onto the wrong trade after the first rewind
                _cd.append(done["exit_tag"].to_numpy())
                _txt = dict(texttemplate="%{customdata[3]}", textposition="top center",
                            textfont=dict(size=10, color=list(np.where(win, "#26a69a",
                                                                       "#ef5350"))))
            fig.add_trace(go.Scattergl(
                x=done["exit_time"], y=done["exit_price"],
                mode="markers+text" if _lab else "markers",
                name=f"exits ({int(win.sum())}W / {int((~win).sum())}L)", legendgroup="trades",
                marker=dict(symbol="x", size=9,
                            color=np.where(win, "#26a69a", "#ef5350"),
                            line=dict(width=1.4)),
                customdata=np.column_stack(_cd),
                visible=True if "trades" in start_visible else "legendonly",
                hovertemplate=("<b>%{customdata[2]}</b> exit %{y:.2f}<br>%{customdata[0]}"
                               "<br>%{customdata[1]:.2f}R<extra></extra>"), **_txt))

        # ONE TRACE PAIR PER TRADE, hidden, so a single trade can be pulled out of the pile along
        # with the line that caused it. The concatenated traces above are the overview; this is the
        # only way to answer "show me just #23 and the line it broke".
        keep = tq if max_trade_picker <= 0 or len(tq) <= max_trade_picker \
            else tq.tail(int(max_trade_picker))
        for r in keep.itertuples():
            x0, x1 = r.entry_time, (r.exit_time if pd.notna(r.exit_time) else tv[-1])
            up = r.side == "long"
            colr = "#26a69a" if r.pnl > 0 else "#ef5350"
            # DID THE STOP MOVE? `stop_price` is the stop set at ENTRY. With a breakeven rung the
            # trade can be closed by a stop that has since walked to entry, and then the exit sits
            # nowhere near the red line — which reads as a broken chart until you know. Measured
            # against the original risk so it is a fact about this trade, not a threshold.
            _risk = abs(r.entry_price - r.stop_price)
            _walked = (r.close_reason == "stop_loss" and _risk > 0
                       and abs(r.exit_price - r.stop_price) > 0.25 * _risk)
            trade_picker.append((r.tid, r, len(fig.data),
                                 [c for c in str(r.cause).split("+") if c in line_ref], _walked))

            # ONE TRACE PER LEVEL, in the SAME colours the grouped view uses. Drawing all three in
            # one trace meant one colour for all of them, so a losing trade showed two red lines
            # and no way to tell the stop from the entry.
            for yv, nm, colr2, dash in ((r.entry_price, "entry", "#90a4ae", "solid"),
                                        (r.stop_price, "STOP (set at entry)", "#ef5350", "dot")):
                fig.add_trace(go.Scattergl(
                    x=[x0, x1] if yv == yv else [], y=[yv, yv] if yv == yv else [],
                    mode="lines", visible=False, showlegend=False,
                    line=dict(color=colr2, width=1.6, dash=dash),
                    hovertemplate=f"<b>{r.tid}</b> {nm} {yv:,.2f}<extra></extra>"))
            # THE TARGET, as the trade actually carried it: a step where it was MOVED closer, and
            # the FVG exit level beside it when the trade had one. Still ONE trace, because every
            # picker addresses a trade's traces by a fixed count (TRADE_TRACES).
            tx, ty, tt = [], [], []
            yv = getattr(r, "target", np.nan)
            if yv == yv:
                for sx_, sy_, lab_ in _target_path(r, x0, x1, trade_tp_rr, ref_target_label):
                    tx += sx_ + [None]; ty += sy_ + [None]
                    tt += [f"{a} {v:,.2f}" for a, v in zip(lab_, sy_)] + [""]
            fig.add_trace(go.Scattergl(
                x=tx, y=ty, text=tt, mode="lines", visible=False, showlegend=False,
                line=dict(color="#26a69a", width=1.6, dash="dot"),
                hovertemplate=f"<b>{r.tid}</b> TARGET %{{text}}<extra></extra>"))

            _why = str(r.exit_why) if _lab else str(r.close_reason)
            if _walked and not _lab:
                _why += " — but the stop had walked to ~breakeven, so the exit is at entry"
            fig.add_trace(go.Scattergl(
                x=[x0, x1], y=[r.entry_price, r.exit_price], mode="markers+lines",
                visible=False, showlegend=False,
                line=dict(color=colr, width=1.0, dash="dot"),
                marker=dict(size=[13, 11], color=colr,
                            symbol=["triangle-up" if up else "triangle-down", "x"],
                            line=dict(width=1.4, color="#fff")),
                text=[f"{r.tid} {r.side} entry {r.entry_price:,.2f} · broke {r.cause}",
                      f"{r.tid} exit {r.exit_price:,.2f} · {_why} · {r.R:.2f}R"],
                hovertemplate="%{text}<extra></extra>"))

    # ---- THREE dropdowns: timeframe -> family -> one instance ---------------------------
    # Each menu narrows the next, because a flat list does not scale and, worse, plotly menus do
    # not SCROLL: they render every button and clip whatever runs past the plot area. Thirty
    # families was 570px of dropdown in a 900px figure, so the last few were simply unreachable.
    # Cascading keeps every menu short by construction — one timeframe's families, one family's
    # instances — rather than relying on the figure being tall enough.
    #
    # MULTI-SELECT lives in the LEGEND, not here: a plotly dropdown is single-choice by nature.
    # Picking a family promotes its instances to individual legend entries, so you can then tick
    # exactly the two channel-ups you want and leave everything else off, EMAs included.
    # ---- THE MISSED SETUPS, one pickable entry each ------------------------------------------
    # The reason-grouped layer further down is the overview; this is the single-setup view, built
    # the same way a trade is: the LINE it would have broken plus a marker at the break, hidden
    # until the dropdown asks for it. Added HERE, before the menus, precisely so the pickers own
    # them -- isolating a trade must hide these, and isolating one of these must hide the trades.
    # (The reason layer is the opposite case and is appended after the menus on purpose.)
    MISS_TRACES = 3      # the line, the bar it became KNOWN, and the break
    # THE SAME COLOURS THE TRADED LINES USE. A short breaks a SUPPORT and a long a RESISTANCE, so
    # the line a missed setup would have broken is drawn exactly as it would have been had it
    # traded -- a muted grey read as chart furniture and was invisible against the candles.
    _MISS_LINE = {"short": "#2ecc71", "long": "#e74c3c"}

    def _lag_note(r):
        """WHY a line is not tradeable the moment price last touched it.

        The obvious reading -- "price bounced off it three times, so it is confirmed" -- is right
        about the touches and wrong about the timing. The third touch is only a TOUCH once the
        zigzag has called that bar a pivot, and it cannot do that until price has moved away by
        `atr_mult` ATR: a low is not a swing low until something proves it turned. So the line
        becomes tradeable some bars AFTER the touch, and price is necessarily far from it by then.
        That distance is not a problem with the confirmation -- it IS the confirmation.
        """
        lt = getattr(r, "last_touch_t", None)
        if lt is None or pd.isna(lt) or pd.isna(getattr(r, "born_t", None)):
            return ""
        mins = (pd.Timestamp(r.born_t) - pd.Timestamp(lt)).total_seconds() / 60
        if mins <= 0:
            return ""
        return (f"<br>last touch {pd.Timestamp(lt):%b-%d %H:%M} — tradeable {mins:.0f} min later,"
                "<br>once price moved far enough away to confirm that pivot")
    _MISS_COL = {"entry rule": "#7a828c", "not confirmed": "#d9a441",
                 "cancelled": "#c96a4a", "already open": "#5b7fa6"}
    missed_picker = []          # (label, row, first_trace_index)
    if rejected is not None and len(rejected) and trades is not None:
        _per = {}
        for _r in rejected.itertuples():
            # CAPPED PER REASON, not from one shared pool: "entry rule" outnumbers the others
            # three to one here and would drain a shared cap before the interesting ones appeared.
            if _per.get(_r.reason, 0) >= max_missed_picker:
                continue
            _per[_r.reason] = _per.get(_r.reason, 0) + 1
            _g = f"miss#{len(missed_picker)}"
            _c = _MISS_COL.get(_r.reason, "#8b93a0")
            _idx = len(fig.data)
            _ln = getattr(_r, "line", "") or "?"
            _has = not (pd.isna(getattr(_r, "x0", None)) or pd.isna(getattr(_r, "y0", None)))
            fig.add_trace(go.Scattergl(
                x=[_r.x0, _r.x1] if _has else [], y=[_r.y0, _r.y1] if _has else [],
                mode="lines", name=f"{_ln} (missed)", legendgroup=_g, visible=False,
                showlegend=False,
                line=dict(color=_MISS_LINE.get(str(_r.side), "#2ecc71"), width=2.0),
                hovertemplate=f"{_ln}<br>the line this setup would have broken<extra></extra>"))
            # WHEN IT BECAME KNOWN. A line is fitted through pivots that all sit BEFORE it can be
            # traded -- its last pivot has to confirm first -- so "when was this line formed" has
            # two honest answers and the gap between them can be a day. 5m-S53's first touch is
            # 06-08 05:40; it only became tradeable at 06-09 12:00 and broke 75 minutes later.
            _bt = getattr(_r, "born_t", None)
            _by = None
            if _has and _bt is not None and not pd.isna(_bt):
                _span = (pd.Timestamp(_r.x1) - pd.Timestamp(_r.x0)).total_seconds()
                if _span > 0:
                    _f = (pd.Timestamp(_bt) - pd.Timestamp(_r.x0)).total_seconds() / _span
                    _by = _r.y0 + _f * (_r.y1 - _r.y0)
            fig.add_trace(go.Scattergl(
                x=[_bt] if _by is not None else [], y=[_by] if _by is not None else [],
                mode="markers", name=f"{_ln} known", legendgroup=_g, visible=False,
                showlegend=False,
                marker=dict(symbol="circle-open", size=11,
                            line=dict(width=2.2, color=_MISS_LINE.get(str(_r.side), "#2ecc71"))),
                hovertemplate=(
                    f"{_ln} became TRADEABLE here"
                    f"<br>fitted on {getattr(_r, 'touches', 0)} touches, the first at "
                    f"{pd.Timestamp(_r.x0):%b-%d %H:%M}"
                    + _lag_note(_r) +
                    "<br>%{x|%b-%d %H:%M}<extra></extra>")))
            fig.add_trace(go.Scattergl(
                x=[_r.t], y=[_r.price], mode="markers", name=f"{_ln} break", legendgroup=_g,
                visible=False, showlegend=False,
                marker=dict(symbol="x-thin", size=13, color=_c, line=dict(width=2.2, color=_c)),
                hovertemplate=(f"{_ln}  {_r.side}  BROKE here"
                               f"<br>NO TRADE: {_r.reason}"
                               "<br>%{x|%b-%d %H:%M}  %{y:,.2f}<extra></extra>")))
            missed_picker.append((f"{_ln} {_r.side[0].upper()}", _r, _idx))

    menus = []

    if picker or trade_picker:
        base = [tr.visible if tr.visible is not None else True for tr in fig.data]
        # BACKGROUND = the bulk EMA, gap and level layers. Not their isolatable instances, which
        # live in `i#` groups and ARE picker-owned -- otherwise selecting one gap from the dropdown
        # would stop working. A button that does not address a trace leaves it exactly as the user
        # set it, which is the whole point.
        _bg = {i for i, tr in enumerate(fig.data)
               if not (getattr(tr, "legendgroup", "") or "").startswith("i#")
               and ((getattr(tr, "legendgroup", "") or "").startswith(("ema", "mtfema"))
                    or getattr(tr, "meta", None) == "fvg"
                    or (getattr(tr, "legendgroup", "") or "").endswith("S/R zones"))}
        ctl = [i for i in range(len(fig.data)) if i not in _bg]

        def _upd(data, layout=None):
            """A button payload addressing ONLY the traces the pickers own."""
            return [{k: ([v[i] for i in ctl] if isinstance(v, list) else v)
                     for k, v in data.items()}, layout or {}, ctl]

        owned = ({i for _, _, i, _ in picker} | {i + 1 for _, _, i, _ in picker}
                 | {z[2] + k for z in trade_picker for k in range(TRADE_TRACES)}
                 | {z[2] + k for z in missed_picker for k in range(MISS_TRACES)})
        fam_groups = {}
        for fam, lbl, idx, sortk in sorted(picker, key=lambda z: z[3]):
            fam_groups.setdefault(fam, []).append((lbl, idx))
        # instance traces are hidden from the legend until a family is chosen
        base_showlegend = [tr.showlegend if tr.showlegend is not None else True for tr in fig.data]

        def _legend_for(fams=None):
            """Which instance traces appear as their own legend entries."""
            want = set()
            for fam in (fams or ()):
                want |= {i for _, i in fam_groups.get(fam, ())}
            v = list(base_showlegend)
            for entries in fam_groups.values():
                for _, i in entries:
                    # a PINNED entry stays in the legend whatever the menus do — it is the only
                    # selection that survives across families, so a menu press must not erase it
                    v[i] = (i in want) or (i in pinned_traces)
                    v[i + 1] = False          # its markers ride the same legendgroup
            return v

        def _grouped(keep=None):
            """The normal grouped drawing, optionally showing only `keep` legendgroups."""
            v = list(base)
            for i in owned:
                v[i] = False
            if keep is not None:
                for j, tr in enumerate(fig.data):
                    g = getattr(tr, "legendgroup", "") or ""
                    if g and g != "price" and not g.startswith(("ema", "mtfema", "i#")):
                        v[j] = (g in keep)
            return v

        def _family(fam):
            """One family, with every instance of it ON and individually toggleable."""
            v = _grouped({fam_lg[fam]})
            for _, i in fam_groups[fam]:
                v[i] = v[i + 1] = True
            return v

        def _isolate(first):
            v = list(base)
            for i in owned:
                v[i] = False
            for j, tr in enumerate(fig.data):
                g = getattr(tr, "legendgroup", "") or ""
                if g and g != "price" and not g.startswith(("ema", "mtfema")):
                    v[j] = False          # everything else out of the way
            v[first] = v[first + 1] = True
            return v

        def _tf_sort(k):
            return (0, 0) if k is None else (1, -tf_minutes(k))

        def _tf_name(k):
            return "this chart" if k is None else k

        live_tfs = [k for k in sorted(tf_fams, key=_tf_sort)
                    if any(f in fam_groups for f in tf_fams[k])]
        fams_of = {k: [f for f in tf_fams[k] if f in fam_groups] for k in live_tfs}

        # ---- menu 2: the instances of one family --------------------------------------
        def _inst_buttons(fam=None):
            if fam is None:
                return [dict(label="▸ pick a pattern to list its occurrences", method="update",
                             args=_upd({"visible": _grouped(None)}))]
            out = [dict(label=f"▸ all {len(fam_groups[fam])} — tick them in the legend",
                        method="update",
                        args=_upd({"visible": _family(fam), "showlegend": _legend_for([fam])}))]
            for lbl, idx in fam_groups[fam]:
                # capped HERE rather than at each site that builds a label, so every occurrence row
                # is bounded whatever produced it -- a pattern, a trendline, a gap or a level
                out.append(dict(label=_short(lbl), method="update",
                                args=_upd({"visible": _isolate(idx)})))
            return out

        # ---- menu 1: the families of one timeframe ------------------------------------
        # With several timeframes loaded the cross-timeframe list ran to 51 rows — 969px of
        # dropdown in a plot area of 760 — so it was clipped exactly like the flat list it replaced.
        # A menu that cannot show its own last entry is not a menu, so the wide selection prompts
        # for a timeframe instead. Nothing is lost: with no timeframe chosen every family is a
        # legend entry already, which is also the only way to tick several at once.
        multi_tf = len(live_tfs) > 1

        def _fam_buttons(tf=_SENTINEL):
            wide = tf is _SENTINEL
            fams = [] if (wide and multi_tf) else (
                [f for k in live_tfs for f in fams_of[k]] if wide else fams_of.get(tf, []))
            head = ("▸ every pattern" if wide else f"▸ every {_tf_name(tf)} pattern")
            keep = {fam_lg[f] for f in fams_of.get(tf, ())} if not wide else None
            out = [dict(label=head, method="update",
                        args=_upd({"visible": _grouped(keep), "showlegend": _legend_for(None)},
                                  {"updatemenus[2].buttons": _inst_buttons(None),
                                   "updatemenus[2].active": 0}))]
            if wide and multi_tf:
                out.append(dict(label="── pick a timeframe on the left ──", method="update",
                                args=_upd({"visible": _grouped(None),
                                          "showlegend": _legend_for(None)},
                                         {"updatemenus[2].buttons": _inst_buttons(None),
                                          "updatemenus[2].active": 0})))
            for fam in fams:
                # the label drops the timeframe prefix once a timeframe is selected — it is already
                # named by the menu to the left, and repeating it just makes every row longer
                short = fam.split(" · ", 1)[-1] if (not wide and tf is not None) else fam
                out.append(dict(label=_short(f"{short}  ({len(fam_groups[fam])})"), method="update",
                                args=_upd({"visible": _family(fam), "showlegend": _legend_for([fam])},
                                          {"updatemenus[2].buttons": _inst_buttons(fam),
                                           "updatemenus[2].active": 0})))
            return out

        # ---- menu 0: the timeframe ----------------------------------------------------
        tf_buttons = [dict(label="▸ all timeframes", method="update",
                           args=_upd({"visible": _grouped(None), "showlegend": _legend_for(None)},
                                     {"updatemenus[1].buttons": _fam_buttons(),
                                      "updatemenus[1].active": 0,
                                      "updatemenus[2].buttons": _inst_buttons(None),
                                      "updatemenus[2].active": 0}))]
        for k in live_tfs:
            n_inst = sum(len(fam_groups[f]) for f in fams_of[k])
            tf_buttons.append(dict(
                label=f"{_tf_name(k)}   ({len(fams_of[k])} patterns, {n_inst})", method="update",
                args=_upd({"visible": _grouped({fam_lg[f] for f in fams_of[k]}),
                           "showlegend": _legend_for(None)},
                          {"updatemenus[1].buttons": _fam_buttons(k),
                           "updatemenus[1].active": 0,
                           "updatemenus[2].buttons": _inst_buttons(None),
                           "updatemenus[2].active": 0})))

        menu = dict(direction="down", showactive=True, yanchor="top", y=1.045,
                    bgcolor="rgba(28,32,38,0.96)", bordercolor="#4a5058", borderwidth=1,
                    font=dict(size=11, color="#dfe3e8"), pad=dict(l=2, r=2, t=2, b=2))
        # WHERE THE DROPDOWNS GO. Five of them on one row is about 1,050px of header in a plot
        # 890px wide, and plotly does not wrap -- they simply overlap, which they have been doing.
        # With a MISSED half there are still five, so they move onto the three rows the layer
        # buttons already occupy, using the empty right-hand half of each:
        #
        #   row 3 (top)   S/R: ...        |  traded/missed toggle  |  the list
        #   row 2         FVGs: ...       |  timeframe             |  family
        #   row 1                         |  occurrence            |          EMAs (right)
        #
        # Gated on `missed_picker` so every notebook without it keeps the layout it has today.
        _wide = bool(missed_picker)
        _cas = ([dict(buttons=tf_buttons, x=0.40, xanchor="left", **dict(menu, y=1.105)),
                 dict(buttons=_fam_buttons(), x=0.65, xanchor="left", **dict(menu, y=1.105)),
                 dict(buttons=_inst_buttons(None), x=0.40, xanchor="left", **menu)]
                if _wide else
                [dict(buttons=tf_buttons, x=0.0, xanchor="left", **menu),
                 dict(buttons=_fam_buttons(), x=0.155, xanchor="left", **menu),
                 dict(buttons=_inst_buttons(None), x=0.40, xanchor="left", **menu)])
        menus += _cas

        if trade_picker:
            # A TRADE and everything it owns, in one click -- its entry, stop, target and exit, plus
            # the trendline it broke. Grouped by side and outcome first, because forty-four rows is
            # 836px of dropdown and plotly menus clip rather than scroll.
            def _isolate_trade(first, causes):
                v = list(base)
                for i in owned:
                    v[i] = False
                for j, tr in enumerate(fig.data):
                    g = getattr(tr, "legendgroup", "") or ""
                    if g and g != "price" and not g.startswith(("ema", "mtfema")):
                        v[j] = False
                for k in range(TRADE_TRACES):
                    v[first + k] = True
                for tag in causes:               # ...and the line that caused it
                    k = line_ref[tag]
                    v[k] = v[k + 1] = True
                return v

            def _all_trades():
                v = list(base)
                for i in owned:
                    v[i] = False
                for j, tr in enumerate(fig.data):
                    g = getattr(tr, "legendgroup", "") or ""
                    if g.startswith("trade"):
                        v[j] = True
                return v

            def _row(tid, r, idx, causes, walked=False):
                sd = "L" if r.side == "long" else "S"
                got = "".join(w[0] for w in str(r.close_reason).split("_"))
                if walked:
                    got = "sl@be"     # stopped, but on a stop that had walked to breakeven
                return dict(label=f"{tid} {sd} {r.cause or '?'}  {r.R:+.2f}R {got}",
                            method="update", args=_upd({"visible": _isolate_trade(idx, causes)}))

            # No "all trades" group: every trade is in longs or shorts, and each group's first row
            # already restores all of them. Listing all forty again cost 0.5 MB for nothing, and the
            # menus are the dominant term in this figure's size.
            groups = {
                "longs": lambda z: z[1].side == "long",
                "shorts": lambda z: z[1].side == "short",
                "winners": lambda z: z[1].pnl > 0,
                "losers": lambda z: z[1].pnl <= 0,
            }

            def _trade_buttons(which):
                rows = [z for z in trade_picker if groups[which](z)]
                out = [dict(label=f"▸ show all {len(trade_picker)} trades again", method="update",
                            args=_upd({"visible": _all_trades()}))]
                out += [_row(*z) for z in rows]
                return out

            # THE SETUPS THAT DID NOT TRADE, in the same two menus. One toggle picks executed or
            # missed, the list beside it fills with whichever you chose -- rather than a third
            # dropdown, which would crowd the row AND shift `updatemenus[4]`, the index every
            # cascade payload above addresses the list by.
            def _isolate_missed(first):
                v = list(base)
                for i in owned:
                    v[i] = False
                for j, tr in enumerate(fig.data):
                    g = getattr(tr, "legendgroup", "") or ""
                    if g and g != "price" and not g.startswith(("ema", "mtfema")):
                        v[j] = False
                for k in range(MISS_TRACES):
                    v[first + k] = True
                return v

            def _missed_buttons(why):
                # NOT "show all N trades again": that is the executed list's reset row, and reading
                # it at the top of a MISSED list says the opposite of what has been selected.
                out = [dict(label="▸ clear — back to the trades", method="update",
                            args=_upd({"visible": _all_trades()}))]
                out += [dict(label=f"{lbl}  ·  {r.t:%b-%d %H:%M}", method="update",
                             args=_upd({"visible": _isolate_missed(i)}))
                        for lbl, r, i in missed_picker if r.reason == why]
                return out

            grp_buttons = []
            for g in groups:
                n = sum(1 for z in trade_picker if groups[g](z))
                grp_buttons.append(dict(
                    # the "traded" prefix only earns its place when there is a MISSED half to tell
                    # it apart from; without one it is noise on every older notebook's chart
                    label=(f"traded · {g}  ({n})" if missed_picker else f"{g}  ({n})"),
                    method="update",
                    args=_upd({"visible": _all_trades()},
                              {"updatemenus[4].buttons": _trade_buttons(g),
                               "updatemenus[4].active": 0})))
            for why in sorted({r.reason for _, r, _ in missed_picker}):
                n = sum(1 for _, r, _ in missed_picker if r.reason == why)
                grp_buttons.append(dict(
                    label=f"MISSED · {why}  ({n})", method="update",
                    args=_upd({"visible": _all_trades()},
                              {"updatemenus[4].buttons": _missed_buttons(why),
                               "updatemenus[4].active": 0})))
            # 41 rows is 780px of dropdown in a plot area of about 760, and plotly menus clip
            # rather than scroll, so a long list opens on a prompt and the groups on the left are
            # what make every trade reachable.
            def _all_rows():
                out = [dict(label=f"▸ show all {len(trade_picker)} trades again", method="update",
                            args=_upd({"visible": _all_trades()}))]
                return out + [_row(*z) for z in trade_picker]

            opening = (_all_rows() if len(trade_picker) <= 30
                       else [dict(label=f"▸ show all {len(trade_picker)} trades again", method="update",
                                  args=_upd({"visible": _all_trades()})),
                             dict(label="── pick a group on the left ──", method="update",
                                  args=_upd({"visible": _all_trades()}))])
            _tm = dict(menu, y=1.165) if missed_picker else menu
            menus.append(dict(buttons=grp_buttons,
                              x=0.40 if missed_picker else 0.60, xanchor="left", **_tm))
            menus.append(dict(buttons=opening,
                              x=0.68 if missed_picker else 0.74, xanchor="left", **_tm))

    # ONE TIMEFRAME'S GAPS AT A TIME. The legend already toggles each layer, but it lists them as
    # "4h . FVG bullish" / "4h . FVG bearish" -- two clicks per timeframe and six rows to hunt
    # through. These buttons isolate a whole timeframe in one press, both directions together.
    #
    # `restyle` takes a LIST of visibilities alongside a list of trace indices, so one button can
    # show one layer and hide the rest -- which a single style value could not express. Cost is
    # buttons x gap traces, and there are two gap traces per layer, so it is negligible.
    if fvg_layer_ix and len(fvg_layer_ix) > 1:
        _all_ix = [i for v in fvg_layer_ix.values() for i in v]
        _fvg_btn = [dict(label="FVGs: all", method="restyle",
                         args=[{"visible": [True] * len(_all_ix)}, _all_ix])]
        for _tf in sorted(fvg_layer_ix, key=lambda k: -tf_minutes(k)):
            _fvg_btn.append(dict(
                label=_tf, method="restyle",
                args=[{"visible": [i in set(fvg_layer_ix[_tf]) for i in _all_ix]}, _all_ix]))
        _fvg_btn.append(dict(label="none", method="restyle",
                             args=[{"visible": ["legendonly"] * len(_all_ix)}, _all_ix]))
        menus.append(dict(
            type="buttons", direction="right", showactive=True,
            active=_row_active(fig, _all_ix, len(_fvg_btn)),
            x=0.0, xanchor="left", y=1.105, yanchor="top",
            bgcolor="rgba(28,32,38,0.96)", bordercolor="#4a5058", borderwidth=1,
            font=dict(size=11, color="#dfe3e8"), pad=dict(l=2, r=2, t=2, b=2),
            buttons=_fvg_btn))

    # ONE TIMEFRAME'S LEVELS AT A TIME, the same control the gaps get and for the same reason:
    # the legend lists them as "4h . S/R zones", one row per timeframe, and hunting through rows is
    # what the row of buttons replaces.
    if sr_layer_ix and len(sr_layer_ix) > 1:
        _all_sr = [i for v in sr_layer_ix.values() for i in v]
        _sr_btn = [dict(label="S/R: all", method="restyle",
                        args=[{"visible": [True] * len(_all_sr)}, _all_sr])]
        for _tf in sorted(sr_layer_ix, key=lambda k: -tf_minutes(k)):
            _sr_btn.append(dict(
                label=_tf, method="restyle",
                args=[{"visible": [i in set(sr_layer_ix[_tf]) for i in _all_sr]}, _all_sr]))
        _sr_btn.append(dict(label="none", method="restyle",
                            args=[{"visible": ["legendonly"] * len(_all_sr)}, _all_sr]))
        menus.append(dict(
            type="buttons", direction="right", showactive=True,
            active=_row_active(fig, _all_sr, len(_sr_btn)),
            x=0.0, xanchor="left", y=1.165, yanchor="top",
            bgcolor="rgba(28,32,38,0.96)", bordercolor="#4a5058", borderwidth=1,
            font=dict(size=11, color="#dfe3e8"), pad=dict(l=2, r=2, t=2, b=2),
            buttons=_sr_btn))

    # ALL THE EMAs AT ONCE. Plotly's own way to do this is double-clicking a legend entry,
    # which ISOLATES it -- and isolating an EMA hides the trade and the line you were looking
    # at. `itemdoubleclick` below defuses that; this is the control that does what the
    # double-click was being used for. Two buttons, each restyling only the EMA traces, so they
    # cost nothing and cannot disturb anything else on the chart.
    _ema_ix = [j for j, tr in enumerate(fig.data)
               if (getattr(tr, "legendgroup", "") or "").startswith(("ema", "mtfema"))]
    if _ema_ix:
        menus.append(dict(
            type="buttons", direction="right", showactive=False,
            x=1.0, xanchor="right", y=1.045, yanchor="top",
            bgcolor="rgba(28,32,38,0.96)", bordercolor="#4a5058", borderwidth=1,
            font=dict(size=11, color="#dfe3e8"), pad=dict(l=2, r=2, t=2, b=2),
            buttons=[dict(label=f"EMAs on ({len(_ema_ix)})", method="restyle",
                          args=[{"visible": True}, _ema_ix]),
                     dict(label="off", method="restyle",
                          args=[{"visible": "legendonly"}, _ema_ix])]))

    if menus:
        fig.update_layout(updatemenus=menus)

    # ---- THE SETUPS THAT DID NOT TRADE -------------------------------------------------------
    # A filter can only be judged against what it REMOVED, and the chart shows only what survived:
    # a rule that deletes a good setup looks exactly like one that deletes nothing, because both
    # just leave a shorter list of trades. `quant.research.rejected_setups` labels each break with
    # why it was turned down; this puts them back on the chart so a lost winner is something you
    # can SEE.
    #
    # ADDED AFTER THE MENUS, DELIBERATELY. Every button row carries a visibility payload built
    # against the trace list as it stood when the menu was made, so traces appended here are in no
    # payload at all -- pressing `FVGs: 4h` or isolating one trade cannot touch this layer, and the
    # chart opens byte-identical to before because every one of these starts `legendonly`.
    if rejected is not None and len(rejected):
        _RJ = {"entry rule": "#7a828c", "not confirmed": "#d9a441",
               "cancelled": "#c96a4a", "already open": "#5b7fa6"}
        for _why, _col in _RJ.items():
            _r = rejected[rejected["reason"] == _why]
            if not len(_r):
                continue
            fig.add_trace(go.Scattergl(
                x=list(_r["t"]), y=list(_r["price"]), mode="markers",
                name=f"skipped · {_why} ({len(_r)})", legendgroup=f"rej:{_why}",
                visible="legendonly", meta="rejected",
                marker=dict(symbol="x-thin", size=9, color=_col,
                            line=dict(width=1.6, color=_col)),
                customdata=[f"{w.side} {w.tf}  ·  skipped: {w.reason}" for w in _r.itertuples()],
                hovertemplate="%{customdata}<br>%{x|%b-%d %H:%M}  %{y:,.2f}<extra></extra>"))

    if rewind:
        # WHICH TRACES THE REWIND MAY CLIP: everything with a birth — gaps, levels, trendlines and
        # trades. Not the candles (they are the clock), not the EMAs (a step line has no birth and
        # clipping it would just truncate the chart's own context). Stored as indices on the layout
        # so the browser script needs no second payload and the figure carries its own tool.
        # A DENY-LIST, not an allow-list. Naming the rewindable kinds missed the per-trade
        # entry/stop/target traces, which carry neither a name nor a legendgroup -- and the next
        # layer added would have been missed too. Two things are excluded and both for a reason:
        #   candles  are the clock, and you want to SEE what happened after the bar you picked
        #   EMAs     are context with no birth; a truncated step line reads as a broken indicator
        _rw = [_i for _i, _t in enumerate(fig.data)
               if _t.type != "candlestick"
               and not (getattr(_t, "legendgroup", "") or "").startswith(("ema", "mtfema"))]
        fig.update_layout(meta={"rewind": _rw})

    if view_start is not None:
        # the data before this is WARM-UP: it feeds the zones, lines and EMAs but is not shown
        _vs = pd.to_datetime(view_start)
        if _vs.tzinfo is None:
            _vs = _vs.tz_localize("UTC")
        # AUTOSCALE MUST NOT ESCAPE THE WINDOW. `range` only sets the opening view; the
        # autoscale button rescales to the DATA, and a trendline fitted through pivots ten days
        # before the window -- or a 4h gap born 131 days before it -- dragged the chart back to
        # all of it. `minallowed` bounds the axis itself, so it holds for every layer without
        # clipping any geometry: a line that genuinely starts before the window keeps its real
        # slope, it just cannot pull the view off the period you asked for.
        fig.update_xaxes(range=[_vs, t[-1]])
        try:
            # Belt: bounds the AXIS, so no layer can pull the view left however it is drawn.
            #
            # EPOCH MILLISECONDS, NOT A DATE STRING. plotly.js cannot compare a date STRING against
            # a date axis here: every string form ("2026-06-01", with or without a time or an
            # offset) made autorange yield nothing at all, and the axis fell back to its empty
            # default of 2000-01-01..2001-01-01 -- so pressing Autoscale threw the chart 26 years
            # into the past. The number is compared correctly and pins the edge exactly.
            #
            # Added in plotly 5.19; an older kernel must degrade quietly, and the geometry clipping
            # above is what holds the line there.
            fig.update_xaxes(autorangeoptions=dict(
                minallowed=int(pd.Timestamp(_vs).value // 1_000_000)))
        except (ValueError, TypeError):
            pass

    if y_from_price:
        # scale to the candles, not to whatever geometry hangs off them — see the docstring
        _ypad = 0.04 * max(_hi - _lo, 1e-9)
        fig.update_yaxes(range=[_lo - _ypad, _hi + _ypad])

    fig.update_layout(
        title=title or "Chart patterns — click the legend to show/hide each one",
        height=height, template="plotly_dark", xaxis_rangeslider_visible=False,
        hovermode="x unified",
        # A double-click on a legend entry normally ISOLATES it, hiding everything else -- which
        # silently undoes the trade picker: pick a trade, double-click an EMA, and the trade and the
        # line that caused it vanish. "toggle" makes a double-click do what a single click does.
        legend=dict(orientation="v", x=1.01, y=1, bgcolor="rgba(0,0,0,0.35)",
                    itemdoubleclick="toggle"),
        # The centred title shares the very top of the container with the TOP menu row. With the
        # layer buttons alone that row is short and left-hand, so the title cleared it; once the
        # trade menus move up there, they sit straight under it. More headroom pushes the rows
        # (paper coords, so they follow the plot area) clear of the title.
        margin=dict(l=50, r=210, t=104 + (66 if missed_picker else 0), b=40),
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
