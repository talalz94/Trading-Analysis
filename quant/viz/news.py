"""
NEWS ON THE CHART -- a TradingView-style strip of economic releases along the bottom edge.

    from quant.news import load_events
    fig = pattern_figure(df, news=load_events(start, end, coverage={"USD": "medium"}),
                         news_show="high")

What it draws, each level its own trace so the dropdown and the legend can switch it:

  markers    one per release minute and level, pinned to the BOTTOM of the plot on an overlay
             axis of fixed range [0, 1] -- so they stay on the edge however the price axis is
             zoomed or panned. Colour and size are the level: critical (dark red, largest),
             high (red), medium (orange), low (yellow); holidays grey.
  lines      a faint dotted vertical line at each critical and high release, on the same toggle
  dropdown   "News: critical | high+ | medium+ | all | off" under the chart, bottom right
             (the rewind control owns bottom left). Every button addresses ONLY the news traces,
             by index, so no existing control moves and none of them can disturb the strip.

Hover gives the release in your zone and in New York, its forecast and previous, and -- once
released -- the actual, the surprise and what it means for the currency.

REWIND. A release's TIME is known weeks ahead, its ACTUAL only from the moment it is out. So the
schedule markers are left out of the rewind (they stay, as they would on a real calendar) and the
actuals live on a second, invisible marker per release that IS in the rewind list: clicking a
candle hides every actual that was not yet out at that candle, and nothing else.

Added AFTER everything else on the chart, so a figure built without `news=` is exactly what it was.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LEVEL_STYLE = {   # colour, marker size
    "critical": ("#b71c1c", 13), "high": ("#e53935", 11), "medium": ("#fb8c00", 9),
    "low": ("#fdd835", 7), "none": ("#9e9e9e", 7)}
DRAW_ORDER = ("none", "low", "medium", "high", "critical")      # most important drawn on top
SHOW = {"critical": ("critical",), "high": ("critical", "high"),
        "medium": ("critical", "high", "medium"), "all": DRAW_ORDER, "off": ()}
MENU_LABELS = (("critical", "News: critical"), ("high", "high+"), ("medium", "medium+"),
               ("all", "all"), ("off", "off"))
Y_MARK, Y_RESULT = 0.022, 0.022


def _next_axis(fig) -> int:
    """The first unused y-axis number (yaxis2, yaxis3 ...)."""
    used = {1}
    for k in fig.layout.to_plotly_json():
        if k.startswith("yaxis"):
            used.add(int(k[5:] or 1))
    for tr in fig.data:
        ya = getattr(tr, "yaxis", None)
        if ya:
            used.add(int(ya[1:] or 1))
    return max(used) + 1


def _fmt(e, tz) -> str:
    t = pd.Timestamp(e["t"])
    when = t.tz_convert(tz).strftime("%a %d %b %H:%M %Z") if tz else t.strftime("%a %d %b %H:%M")
    ny = t.tz_convert("America/New_York").strftime("%H:%M")
    return f"{when} · {ny} New York"


def _line(e, with_actual: bool) -> str:
    bits = []
    if with_actual and e.get("actual"):
        bits.append(f"<b>A {e['actual']}</b>")
    for k, lab in (("forecast", "F"), ("previous", "P")):
        if e.get(k):
            bits.append(f"{lab} {e[k]}")
    s = f"{e['currency']} <b>{e['title']}</b>"
    if e.get("time_q") not in ("exact", "official"):
        s += " <i>(time approx.)</i>"
    if bits:
        s += "  " + " · ".join(bits)
    if with_actual and e.get("sentiment"):
        z = e.get("surprise_z")
        zz = f", {z:+.1f}σ" if z is not None and np.isfinite(z) else ""
        s += f"  → {e['sentiment']} for {e['currency']}{zz}"
    return s


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else v


def add_news_strip(fig, events: pd.DataFrame, *, t_min, t_max, show: str = "high",
                   tz: Optional[str] = "Asia/Karachi", x_tz=None, rewind: bool = False) -> dict:
    """Append the news strip to `fig` (see the module docstring). `t_min`/`t_max` bound the
    releases drawn (the chart's own span); `x_tz` is the zone of the chart's x values, so a
    release lands on its candle; `tz` is the zone the hover reads in. Returns the trace indices
    by level and the menu added."""
    import plotly.graph_objects as go
    if show not in SHOW:
        raise ValueError(f"news_show must be one of {list(SHOW)}, got {show!r}")
    ev = events.copy()
    ev["t"] = pd.to_datetime(ev["t"], utc=True)
    lo, hi = pd.Timestamp(t_min), pd.Timestamp(t_max)
    lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo.tz_convert("UTC")
    hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi.tz_convert("UTC")
    ev = ev[(ev["t"] >= lo) & (ev["t"] <= hi)]
    ev = ev[ev["time_q"].isin(["exact", "official", "typical", "allday"])]
    out = {"markers": {}, "results": {}, "lines": {}, "menu": None}
    if ev.empty:
        return out
    for c in ("actual", "forecast", "previous", "sentiment"):
        ev[c] = ev[c].map(_clean)

    n_axis = _next_axis(fig)
    yref = f"y{n_axis}"
    fig.update_layout(**{f"yaxis{n_axis}": dict(overlaying="y", range=[0, 1], fixedrange=True,
                                                 visible=False, showgrid=False)})

    def xs(times):
        x = pd.DatetimeIndex(times)
        return list(x.tz_convert(x_tz) if x_tz is not None else x.tz_convert("UTC")
                    .tz_localize(None))

    start_on = set(SHOW[show])
    for lv in DRAW_ORDER:
        g = ev[ev["level"] == lv]
        if g.empty:
            continue
        colour, size = LEVEL_STYLE[lv]
        vis = True if lv in start_on else "legendonly"
        groups = [(t, grp.to_dict("records")) for t, grp in g.groupby("t", sort=True)]
        times = [t for t, _ in groups]
        sched = ["<br>".join([_fmt(r[0], tz)] + [_line(e, False) for e in r]) for _, r in groups]
        name = {"none": "holidays"}.get(lv, lv)
        fig.add_trace(go.Scatter(
            x=xs(times), y=[Y_MARK] * len(times), yaxis=yref, mode="markers",
            name=f"news · {name} ({len(g)})", legendgroup=f"news:{lv}", meta="news",
            visible=vis, customdata=sched, hovertemplate="%{customdata}<extra></extra>",
            marker=dict(size=size, color=colour, symbol="circle",
                        line=dict(width=1, color="rgba(255,255,255,0.75)"))))
        out["markers"][lv] = len(fig.data) - 1
        if lv in ("critical", "high"):
            lx, ly = [], []
            for x in xs(times):
                lx += [x, x, None]
                ly += [0, 1, None]
            fig.add_trace(go.Scatter(
                x=lx, y=ly, yaxis=yref, mode="lines", showlegend=False, hoverinfo="skip",
                legendgroup=f"news:{lv}", meta="news", visible=vis,
                line=dict(width=1, dash="dot", color=colour), opacity=0.35))
            out["lines"][lv] = len(fig.data) - 1
        # the ACTUALS, on their own invisible markers -- rewind clips these and only these
        done = [(t, [e for e in r if e.get("actual")]) for t, r in groups]
        done = [(t, r) for t, r in done if r]
        if done:
            fig.add_trace(go.Scatter(
                x=xs([t for t, _ in done]), y=[Y_RESULT] * len(done), yaxis=yref,
                mode="markers", showlegend=False, legendgroup=f"news:{lv}",
                meta="news_result", visible=vis,
                customdata=["<br>".join(["released:"] + [_line(e, True) for e in r])
                            for _, r in done],
                hovertemplate="%{customdata}<extra></extra>",
                marker=dict(size=size, color="rgba(0,0,0,0)")))
            out["results"][lv] = len(fig.data) - 1

    # ── the dropdown: addresses ONLY these traces, by index ───────────────────────────────
    idx, lvl_of = [], []
    for kind in ("markers", "lines", "results"):
        for lv, i in out[kind].items():
            idx.append(i)
            lvl_of.append(lv)
    buttons = [dict(label=lab, method="restyle",
                    args=[{"visible": [True if lv in SHOW[key] else "legendonly"
                                       for lv in lvl_of]}, idx])
               for key, lab in MENU_LABELS]
    active = [k for k, _ in MENU_LABELS].index(show)
    menu = dict(type="dropdown", direction="up", showactive=True, active=active,
                x=1.0, xanchor="right", y=-0.035, yanchor="top",
                bgcolor="rgba(28,32,38,0.96)", bordercolor="#4a5058", borderwidth=1,
                font=dict(size=11, color="#dfe3e8"), pad=dict(l=2, r=2, t=2, b=2),
                buttons=buttons)
    fig.update_layout(updatemenus=list(fig.layout.updatemenus or ()) + [menu])
    b = (fig.layout.margin.b if fig.layout.margin and fig.layout.margin.b is not None else 40)
    fig.update_layout(margin=dict(b=max(b, 68)))
    out["menu"] = menu

    if rewind:
        meta = dict(fig.layout.meta or {}) if isinstance(fig.layout.meta, dict) else {}
        rw = list(meta.get("rewind", []))
        meta["rewind"] = rw + [i for i in out["results"].values()]
        fig.update_layout(meta=meta)
    return out
