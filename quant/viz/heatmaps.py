"""
Analytical charts for reports: monthly-returns heatmap, hour x weekday performance heatmap,
and parameter-sweep heatmaps. Static plotly figures (safe for HTML export and non-Jupyter use).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DIVERGE = "RdYlGn"


def monthly_returns_heatmap(monthly_pivot: pd.DataFrame, *, title: str = "Monthly Returns %") -> go.Figure:
    """`monthly_pivot`: from analytics.monthly_returns (index=year, columns=1..12)."""
    if monthly_pivot.empty:
        return go.Figure()
    cols = list(monthly_pivot.columns)
    z = monthly_pivot.values
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=[_MONTHS[c - 1] for c in cols], y=[str(y) for y in monthly_pivot.index],
        colorscale=_DIVERGE, zmid=0, zmin=-lim, zmax=lim,
        text=np.round(z, 2), texttemplate="%{text}", textfont={"size": 10},
        colorbar=dict(title="%")))
    fig.update_layout(title=title, template="plotly_white", height=90 + 34 * len(monthly_pivot),
                      margin=dict(l=55, r=30, t=55, b=40))
    return fig


def hour_weekday_heatmap(trades: pd.DataFrame, *, metric: str = "total_pnl",
                         tz: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """Grid of a per-(weekday, hour) trade metric ('total_pnl' | 'win_rate_pct' | 'n_trades')."""
    if trades.empty:
        return go.Figure()
    t = pd.to_datetime(trades["entry_time"])
    if tz is not None and t.dt.tz is not None:
        t = t.dt.tz_convert(tz)
    d = pd.DataFrame({"wd": t.dt.weekday, "hr": t.dt.hour, "pnl": trades["pnl"].to_numpy()})
    if metric == "n_trades":
        grid = d.groupby(["wd", "hr"]).size().rename("v").reset_index()
    elif metric == "win_rate_pct":
        grid = d.assign(w=(d["pnl"] > 0)).groupby(["wd", "hr"])["w"].mean().mul(100).rename("v").reset_index()
    else:
        grid = d.groupby(["wd", "hr"])["pnl"].sum().rename("v").reset_index()
    piv = grid.pivot(index="wd", columns="hr", values="v").reindex(range(7))
    z = piv.values.astype(float)
    diverging = metric in ("total_pnl",)
    kw = dict(colorscale=_DIVERGE if diverging else "Blues")
    if diverging and np.isfinite(z).any():
        lim = np.nanmax(np.abs(z)) or 1.0
        kw.update(zmid=0, zmin=-lim, zmax=lim)
    fig = go.Figure(go.Heatmap(z=z, x=[f"{h:02d}" for h in piv.columns],
                               y=[_DAYS[i] for i in piv.index], **kw, colorbar=dict(title=metric)))
    fig.update_layout(title=title or f"{metric} by hour x weekday", template="plotly_white",
                      height=340, margin=dict(l=55, r=30, t=55, b=40),
                      xaxis_title="hour", yaxis_title="")
    return fig


def sweep_heatmap(results: pd.DataFrame, x: str, y: str, z: str = "sharpe", *,
                  agg: str = "max", title: Optional[str] = None) -> go.Figure:
    """Heatmap of a sweep metric over two parameters (e.g. x='fast', y='slow', z='sharpe')."""
    piv = results.pivot_table(index=y, columns=x, values=z, aggfunc=agg)
    zz = piv.values.astype(float)
    fig = go.Figure(go.Heatmap(z=zz, x=[str(c) for c in piv.columns], y=[str(i) for i in piv.index],
                               colorscale="Viridis", colorbar=dict(title=z),
                               text=np.round(zz, 2), texttemplate="%{text}", textfont={"size": 9}))
    fig.update_layout(title=title or f"{z} ({agg}) over {x} x {y}", template="plotly_white",
                      height=380, margin=dict(l=60, r=30, t=55, b=45),
                      xaxis_title=x, yaxis_title=y)
    return fig
