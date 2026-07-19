from __future__ import annotations
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go

from simulation.simulator import Trade


def _interval_to_minutes(interval: str) -> Optional[float]:
    interval = interval.strip()
    try:
        if interval.endswith("m"):
            return float(interval[:-1])
        if interval.endswith("h"):
            return float(interval[:-1]) * 60.0
        if interval.endswith("d"):
            return float(interval[:-1]) * 60.0 * 24.0
        if interval.endswith("w"):
            return float(interval[:-1]) * 60.0 * 24.0 * 7.0
    except Exception:
        return None
    return None


def trades_to_frame(
    trades: List[Trade],
    initial_cash: float,
    bar_minutes: Optional[float] = None,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-trade dataframe for closed trades.

    - Adds initial point (trade_no=0, balance=initial_cash) so equity plot starts at initial balance.
    - duration_min is minutes (not months).
    - candle_count computed if bar_minutes or interval provided.
    """
    if bar_minutes is None and interval is not None:
        bar_minutes = _interval_to_minutes(interval)

    rows = []
    bal = float(initial_cash)

    # initial point for plotting
    rows.append({
        "trade_no": 0,
        "trade_id": None,
        "side": None,
        "entry_time": None,
        "exit_time": None,
        "entry_price": None,
        "exit_price": None,
        "qty": None,
        "pnl": 0.0,
        "return_pct": 0.0,
        "duration_min": 0.0,
        "candle_count": 0.0,
        "balance": bal,
        "open_reason": None,
        "close_reason": None,
    })

    closed = [t for t in trades if t.exit_time is not None and t.pnl is not None]
    closed = sorted(closed, key=lambda x: x.exit_time)

    for k, t in enumerate(closed, start=1):
        bal += float(t.pnl)

        dur_min = (t.exit_time - t.entry_time).total_seconds() / 60.0 if t.exit_time is not None else None

        candle_count = None
        if dur_min is not None and bar_minutes and bar_minutes > 0:
            candle_count = dur_min / bar_minutes

        ret_pct = None
        if t.entry_price and t.exit_price:
            if t.side == "long":
                ret_pct = (t.exit_price / t.entry_price - 1.0) * 100.0
            else:
                ret_pct = (t.entry_price / t.exit_price - 1.0) * 100.0

        rows.append({
            "trade_no": k,
            "trade_id": t.trade_id,
            "side": t.side,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "qty": t.qty,
            "pnl": float(t.pnl),
            "return_pct": float(ret_pct) if ret_pct is not None else None,
            "duration_min": float(dur_min) if dur_min is not None else None,
            "candle_count": float(candle_count) if candle_count is not None else None,
            "balance": bal,
            "open_reason": t.open_reason,
            "close_reason": t.close_reason,
        })

    return pd.DataFrame(rows)


def plot_balance_by_trade(
    trades: List[Trade],
    initial_cash: float,
    interval: Optional[str] = None,
    bar_minutes: Optional[float] = None,
    title: str = "Balance by Trade",
) -> go.Figure:
    df = trades_to_frame(trades, initial_cash, bar_minutes=bar_minutes, interval=interval)
    if len(df) < 2:
        raise ValueError("Not enough closed trades to plot.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["trade_no"],
        y=df["balance"],
        mode="lines+markers",
        name="Balance",
        hovertemplate="Trade #%{x}<br>Balance: %{y:,.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Trade # (closed)",
        yaxis_title="Balance",
        height=450,
        template="plotly_white",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    # ensure axis includes initial cash clearly
    ymin = min(float(df["balance"].min()), float(initial_cash))
    ymax = max(float(df["balance"].max()), float(initial_cash))
    pad = (ymax - ymin) * 0.08 if ymax > ymin else 100.0
    fig.update_yaxes(range=[ymin - pad, ymax + pad])

    # clean trade ticks
    fig.update_xaxes(dtick=1)

    return fig


def plot_trade_pnl_bars(
    trades: List[Trade],
    initial_cash: float,
    interval: Optional[str] = None,
    bar_minutes: Optional[float] = None,
    title: str = "PnL per Trade",
) -> go.Figure:
    df = trades_to_frame(trades, initial_cash, bar_minutes=bar_minutes, interval=interval)
    df = df[df["trade_no"] > 0].copy()
    if df.empty:
        raise ValueError("No closed trades to plot.")

    colors = ["rgba(0,160,0,0.85)" if p >= 0 else "rgba(200,0,0,0.85)" for p in df["pnl"]]
    custom = df[["return_pct", "duration_min", "candle_count", "side"]].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["trade_no"],
        y=df["pnl"],
        marker_color=colors,
        customdata=custom,
        hovertemplate=(
            "Trade #%{x}<br>"
            "PnL: %{y:,.2f}<br>"
            "Return: %{customdata[0]:.2f}%<br>"
            "Duration: %{customdata[1]:.0f} minutes<br>"
            "Candles: %{customdata[2]:.0f}<br>"
            "Side: %{customdata[3]}<br>"
            "<extra></extra>"
        ),
        name="PnL"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Trade # (closed)",
        yaxis_title="PnL",
        height=450,
        template="plotly_white",
        margin=dict(l=50, r=30, t=60, b=50),
    )
    fig.update_xaxes(dtick=1)

    return fig


def trade_summary_table(
    trades: List[Trade],
    initial_cash: float,
    interval: Optional[str] = None,
    bar_minutes: Optional[float] = None,
) -> pd.DataFrame:
    df = trades_to_frame(trades, initial_cash, bar_minutes=bar_minutes, interval=interval)
    df = df[df["trade_no"] > 0].copy()
    keep = [
        "trade_no","trade_id","side","entry_time","exit_time",
        "entry_price","exit_price","qty","pnl","return_pct",
        "duration_min","candle_count","balance","open_reason","close_reason"
    ]
    return df[keep].copy()