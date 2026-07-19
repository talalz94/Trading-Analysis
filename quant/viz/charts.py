"""
Plotly charts for backtest results (clean, downsample-aware).

price_and_trades  — price line + entry/exit markers (handles 1m spans by downsampling
                    the price trace; trade markers are always kept).
equity_and_drawdown — equity curve + drawdown subplot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _downsample(df: pd.DataFrame, x: str, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step]


def price_and_trades(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    price_col: str = "close",
    time_col: str = "t",
    max_points: int = 6000,
    title: str = "Price & Trades",
) -> go.Figure:
    p = _downsample(df[[time_col, price_col]].dropna(), time_col, max_points)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p[time_col], y=p[price_col], mode="lines",
                             name=price_col, line=dict(width=1, color="#4b6cb7")))

    if trades is not None and not trades.empty:
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]
        for side_df, ecolor, name in [(longs, "#16a34a", "long entry"),
                                      (shorts, "#dc2626", "short entry")]:
            if not side_df.empty:
                fig.add_trace(go.Scatter(
                    x=side_df["entry_time"], y=side_df["entry_price"], mode="markers",
                    name=name, marker=dict(symbol="triangle-up", size=8, color=ecolor)))
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        for ex_df, color, name in [(wins, "#16a34a", "exit (win)"),
                                   (losses, "#dc2626", "exit (loss)")]:
            if not ex_df.empty:
                fig.add_trace(go.Scatter(
                    x=ex_df["exit_time"], y=ex_df["exit_price"], mode="markers",
                    name=name, marker=dict(symbol="x", size=7, color=color)))

    fig.update_layout(title=title, template="plotly_white", height=460,
                      margin=dict(l=50, r=30, t=55, b=40),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


def equity_and_drawdown(equity_curve: pd.DataFrame, *, time_col: str = "t",
                        title: str = "Equity & Drawdown", max_points: int = 6000) -> go.Figure:
    eq = _downsample(equity_curve, time_col, max_points)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.05,
                        subplot_titles=("Equity", "Drawdown %"))
    fig.add_trace(go.Scatter(x=eq[time_col], y=eq["equity"], mode="lines",
                             name="equity", line=dict(color="#4b6cb7", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq[time_col], y=-eq["drawdown"] * 100.0, mode="lines",
                             name="drawdown", fill="tozeroy",
                             line=dict(color="#dc2626", width=1)), row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", height=520, showlegend=False,
                      margin=dict(l=50, r=30, t=60, b=40))
    return fig
