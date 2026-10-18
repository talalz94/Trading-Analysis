"""
MARKET HOURS, PERIOD LEVELS AND CANDLESTICK MARKERS ON THE CHART.

    fig = pattern_figure(df, period_levels={"D": tbl_d, "W": tbl_w}, levels_show=("D",),
                         sessions=windows, session_open_lines=("London", "New York"))

  period levels   the PREVIOUS day / week / month / year high and low, as dotted steps across the
                  period they apply to (quant.indicators.period_levels), one legend entry per
                  period. Only the CURRENT period's pair carries a label -- PDH / PDL, PWH / PWL,
                  PMH / PML, PYH / PYL -- at the right edge, so the chart names the levels that
                  matter now and not every one it has ever drawn.
  sessions        a slim ribbon per market along the TOP edge (Sydney, Tokyo, London, New York by
                  default), each bar the hours that market is open in its own clock, with the name
                  at the left; optional faint vertical lines at each OPEN, full height, so "London
                  opened here" can be read against the candles. On an overlay axis fixed at [0, 1],
                  like the news strip, so the ribbon stays put however the price axis moves.
  candlestick     a small triangle under (bullish) or over (bearish) the bar a named pattern
  markers         completes on, one legend entry per pattern.

Footer dropdowns, beside the News one: `Levels: off | day | week | ... | all` and
`Sessions: on | + opens | off`. Each addresses ONLY its own traces, by index, so no existing
control moves; the legend toggles each period or session on its own.

REWIND. A previous period's high is known from the first moment of the period it is drawn over,
and a candle pattern only once its bar closed, so both are in the rewind list. Session hours are a
timetable, known in advance, so they stay (as the news schedule does).

Added AFTER everything else on the chart: a figure built without these is exactly what it was.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .news import _next_axis

# label, short name, colour, width -- one hue per period; dotted, so they never read as S/R bands
PERIOD_STYLE = {"D": ("day", "PD", "#8b949e", 1.2), "W": ("week", "PW", "#bc8cff", 1.5),
                "M": ("month", "PM", "#e3b341", 1.8), "Y": ("year", "PY", "#39c5cf", 2.1)}
PERIOD_ORDER = ("Y", "M", "W", "D")                 # coarsest drawn first, the day on top
LEVEL_MENU = (("off", ()), ("D", ("D",)), ("W", ("W",)), ("D+W", ("D", "W")),
              ("M", ("M",)), ("Y", ("Y",)), ("all", ("D", "W", "M", "Y")))
OPEN_LINES_MAX_SPAN = pd.Timedelta(days=7)          # longer, and the open lines start hidden
SESSION_COLOURS = ("#7ee787", "#ff9bce", "#79c0ff", "#ffa657", "#d2a8ff", "#56d4dd")
SESSION_TOP, SESSION_STEP = 0.988, 0.016           # the first ribbon row and the gap between rows
MENU_Y = -0.035                                     # the footer row the News dropdown sits on
MENU_Y2 = -0.085                                    # the row under it, when News is there


def _xs(times, x_tz):
    """Timestamps in the chart's x zone (the news strip's convention)."""
    x = pd.DatetimeIndex(pd.to_datetime(list(times), utc=True))
    return list(x.tz_convert(x_tz) if x_tz is not None else x.tz_convert("UTC").tz_localize(None))


def _utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _rewind(fig, idx):
    meta = dict(fig.layout.meta or {}) if isinstance(fig.layout.meta, dict) else {}
    meta["rewind"] = list(meta.get("rewind", [])) + list(idx)
    fig.update_layout(meta=meta)


def _menu(fig, buttons, *, x, active, y=MENU_Y):
    """A footer dropdown, appended LAST so every menu addressed by index keeps its index."""
    menu = dict(type="dropdown", direction="up", showactive=True, active=active, x=x,
                xanchor="right", y=y, yanchor="top", bgcolor="rgba(28,32,38,0.96)",
                bordercolor="#4a5058", borderwidth=1, font=dict(size=11, color="#dfe3e8"),
                pad=dict(l=2, r=2, t=2, b=2), buttons=buttons)
    fig.update_layout(updatemenus=list(fig.layout.updatemenus or ()) + [menu])
    b = fig.layout.margin.b if fig.layout.margin and fig.layout.margin.b is not None else 40
    fig.update_layout(margin=dict(b=max(b, 68 if y > -0.06 else 102)))
    return menu


# ── period levels ────────────────────────────────────────────────────────────────────────────
def add_period_levels(fig, tables: Dict[str, pd.DataFrame], *, t_min, t_max, x_tz=None,
                      show: Sequence[str] = ("D",), rewind: bool = False,
                      menu_x: Optional[float] = None, menu_y: float = MENU_Y) -> dict:
    """Draw `tables` {"D": period_levels(...), ...} between t_min and t_max. Returns the trace
    indices per period and the menu."""
    import plotly.graph_objects as go
    lo, hi = _utc(t_min), _utc(t_max)
    out = {"traces": {}, "menu": None}
    for p in PERIOD_ORDER:
        tb = tables.get(p)
        if tb is None or not len(tb):
            continue
        tb = tb[(pd.to_datetime(tb["end"], utc=True) > lo)
                & (pd.to_datetime(tb["start"], utc=True) < hi)]
        if not len(tb):
            continue
        word, short, colour, width = PERIOD_STYLE[p]
        x0 = [max(_utc(s), lo) for s in tb["start"]]
        x1 = [min(_utc(e), hi) for e in tb["end"]]
        xs, ys, cd = [], [], []
        for a, b, h, l, src in zip(_xs(x0, x_tz), _xs(x1, x_tz), tb["high"], tb["low"],
                                   tb["source_start"]):
            when = pd.Timestamp(src).strftime("%a %d %b %Y" if p in ("D", "W") else
                                              "%b %Y" if p == "M" else "%Y")
            for v, side in ((h, "high"), (l, "low")):
                txt = f"<b>previous {word} {side}</b>  {v:,.2f}<br>{word} of {when}"
                xs += [a, b, None]
                ys += [v, v, None]
                cd += [txt, txt, txt]
        vis = True if p in show else "legendonly"
        idx = []
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines", name=f"prev {word} high / low", legendgroup=f"lvl:{p}",
            meta="levels", visible=vis, customdata=cd,
            hovertemplate="%{customdata}<extra></extra>",
            line=dict(color=colour, width=width, dash="dot")))
        idx.append(len(fig.data) - 1)
        # a label on the CURRENT period's pair only -- the levels in force at the right edge
        last = tb.iloc[-1]
        fig.add_trace(go.Scatter(
            x=_xs([min(_utc(last["end"]), hi)] * 2, x_tz), y=[last["high"], last["low"]],
            mode="text", text=[f"{short}H", f"{short}L"], textposition="middle left",
            textfont=dict(size=10, color=colour), legendgroup=f"lvl:{p}", showlegend=False,
            meta="levels", visible=vis, hoverinfo="skip"))
        idx.append(len(fig.data) - 1)
        out["traces"][p] = idx
    if not out["traces"]:
        return out
    if rewind:
        _rewind(fig, [i for v in out["traces"].values() for i in v])
    have = list(out["traces"])
    all_ix = [i for p in have for i in out["traces"][p]]
    buttons, active = [], 0
    for n, (label, sel) in enumerate(LEVEL_MENU):
        if any(p not in have for p in sel):
            continue
        if tuple(sorted(sel)) == tuple(sorted(p for p in show if p in have)):
            active = len(buttons)
        buttons.append(dict(
            label="Levels: " + label, method="restyle",
            args=[{"visible": [True if p in sel else "legendonly"
                               for p in have for _ in out["traces"][p]]}, all_ix]))
    out["menu"] = _menu(fig, buttons, x=1.0 if menu_x is None else menu_x, active=active,
                        y=menu_y)
    return out


# ── sessions ─────────────────────────────────────────────────────────────────────────────────
def add_sessions(fig, windows: pd.DataFrame, *, t_min, t_max, x_tz=None, show: bool = True,
                 open_lines: Sequence[str] = (), lines_show: Optional[bool] = None,
                 menu_x: Optional[float] = None, menu_y: float = MENU_Y,
                 colours: Optional[Dict[str, str]] = None) -> dict:
    """Draw `windows` (quant.indicators.session_windows) as a ribbon per market along the top,
    plus a dotted line at every open of the markets named in `open_lines`. The lines start
    visible on a chart of a week or less (`lines_show=None`); on seven weeks of 5m candles two
    lines a day are a curtain, so there they wait for the `+ opens` button."""
    import plotly.graph_objects as go
    lo, hi = _utc(t_min), _utc(t_max)
    out = {"traces": {}, "lines": {}, "menu": None}
    if lines_show is None:
        lines_show = (hi - lo) <= OPEN_LINES_MAX_SPAN
    if windows is None or not len(windows):
        return out
    w = windows[(pd.to_datetime(windows["close"], utc=True) > lo)
                & (pd.to_datetime(windows["open"], utc=True) < hi)]
    if not len(w):
        return out
    n_axis = _next_axis(fig)
    yref = f"y{n_axis}"
    fig.update_layout(**{f"yaxis{n_axis}": dict(overlaying="y", range=[0, 1], fixedrange=True,
                                                 visible=False, showgrid=False)})
    names = list(dict.fromkeys(windows["name"]))       # the config's order, top to bottom
    for k, name in enumerate(names):
        g = w[w["name"] == name]
        if not len(g):
            continue
        colour = (colours or {}).get(name, SESSION_COLOURS[k % len(SESSION_COLOURS)])
        y = SESSION_TOP - k * SESSION_STEP
        a = [max(_utc(o), lo) for o in g["open"]]
        b = [min(_utc(c), hi) for c in g["close"]]
        xa, xb = _xs(a, x_tz), _xs(b, x_tz)
        xs, ys = [], []
        for p, q in zip(xa, xb):
            xs += [p, q, None]
            ys += [y, y, None]
        vis = True if show else "legendonly"
        idx = []
        fig.add_trace(go.Scatter(
            x=xs, y=ys, yaxis=yref, mode="lines", name=name, legendgroup=f"sess:{name}",
            meta="sessions", visible=vis, hoverinfo="skip",
            line=dict(color=colour, width=5), opacity=0.75))
        idx.append(len(fig.data) - 1)
        # hover lives on the OPENS: a ribbon segment would answer for every candle under it
        tips = [f"<b>{name} opens</b>  {r.local_open} {r.tz.split('/')[-1].replace('_', ' ')}"
                f"<br>{_utc(r.open):%a %d %b %H:%M} UTC · closes {_utc(r.close):%H:%M} UTC"
                for r in g.itertuples()]
        opens = [o for o in g["open"] if lo <= _utc(o) <= hi]
        if opens:
            fig.add_trace(go.Scatter(
                x=_xs(opens, x_tz), y=[y] * len(opens), yaxis=yref, mode="markers",
                legendgroup=f"sess:{name}", showlegend=False, meta="sessions", visible=vis,
                customdata=[t for o, t in zip(g["open"], tips) if lo <= _utc(o) <= hi],
                hovertemplate="%{customdata}<extra></extra>",
                marker=dict(size=7, color=colour, symbol="triangle-right",
                            line=dict(width=0))))
            idx.append(len(fig.data) - 1)
        # the market's name at the left end of its row, toggled with it
        fig.add_trace(go.Scatter(
            x=_xs([lo], x_tz), y=[y], yaxis=yref, mode="text", text=[f" {name}"],
            textposition="middle right", textfont=dict(size=9, color=colour),
            legendgroup=f"sess:{name}", showlegend=False, meta="sessions", visible=vis,
            hoverinfo="skip"))
        idx.append(len(fig.data) - 1)
        out["traces"][name] = idx
        if name in open_lines and opens:
            lx, ly = [], []
            for x in _xs(opens, x_tz):
                lx += [x, x, None]
                ly += [0, y - 0.006, None]
            fig.add_trace(go.Scatter(
                x=lx, y=ly, yaxis=yref, mode="lines", legendgroup=f"sess:{name}",
                showlegend=False, meta="sessions",
                visible=vis if lines_show else "legendonly", hoverinfo="skip",
                line=dict(color=colour, width=1, dash="dot"), opacity=0.35))
            out["lines"][name] = len(fig.data) - 1
    if not out["traces"]:
        return out
    ribbon = [i for v in out["traces"].values() for i in v]
    lines = list(out["lines"].values())
    every = ribbon + lines
    buttons = [dict(label="Sessions: on", method="restyle",
                    args=[{"visible": [True] * len(ribbon) + ["legendonly"] * len(lines)},
                          every])]
    if lines:
        buttons.append(dict(label="Sessions: + opens", method="restyle",
                            args=[{"visible": [True] * len(every)}, every]))
    buttons.append(dict(label="Sessions: off", method="restyle",
                        args=[{"visible": ["legendonly"] * len(every)}, every]))
    active = (1 if lines and lines_show else 0) if show else len(buttons) - 1
    out["menu"] = _menu(fig, buttons, x=0.84 if menu_x is None else menu_x, active=active,
                        y=menu_y)
    return out


# ── candlestick markers ──────────────────────────────────────────────────────────────────────
def add_candle_markers(fig, df: pd.DataFrame, names: Sequence[str], *, time_col: str = "t",
                       start=None, show: bool = True, rewind: bool = False) -> dict:
    """A triangle under (bullish) or over (bearish) every bar a named candlestick pattern
    completes on, from `start` on. Patterns are detected on `df` here, causally (each flagged on
    its LAST candle, quant.indicators.candle_patterns)."""
    import plotly.graph_objects as go
    from ..indicators.candle_patterns import ALL_PATTERNS, PATTERN_DIRECTION, add_candle_patterns
    names = list(ALL_PATTERNS) if names == "all" else list(names)
    bad = sorted(set(names) - set(ALL_PATTERNS))
    if bad:
        raise ValueError(f"unknown candlestick patterns {bad} -- docs/CHART_GUIDE.md lists them")
    out = {"traces": {}}
    if not names or df is None or not len(df):
        return out
    d = add_candle_patterns(df.reset_index(drop=True), patterns=names)
    if start is not None:
        tt = pd.to_datetime(d[time_col])
        s = pd.Timestamp(start)
        if tt.dt.tz is not None and s.tzinfo is None:
            s = s.tz_localize("UTC")
        d = d[(tt >= s).to_numpy()]
    hi_, lo_ = d["high"].to_numpy(float), d["low"].to_numpy(float)
    pad = np.nanmedian(hi_ - lo_) * 0.9 if len(d) else 0.0
    for name in names:
        hit = d[f"cdl_{name}"].to_numpy(bool)
        if not hit.any():
            continue
        dirn = PATTERN_DIRECTION.get(name, 0)
        up = dirn > 0
        fig.add_trace(go.Scattergl(
            x=d[time_col].to_numpy()[hit], y=(lo_[hit] - pad) if up else (hi_[hit] + pad),
            mode="markers", name=f"candle · {name.replace('_', ' ')} ({int(hit.sum())})",
            legendgroup=f"cdl:{name}", meta="candles", visible=True if show else "legendonly",
            hovertemplate=f"{name.replace('_', ' ')}<extra></extra>",
            marker=dict(symbol="triangle-up" if up else "triangle-down", size=8,
                        color="#2ecc71" if up else ("#e74c3c" if dirn < 0 else "#f1c40f"),
                        line=dict(width=0))))
        out["traces"][name] = len(fig.data) - 1
    if rewind and out["traces"]:
        _rewind(fig, out["traces"].values())
    return out
