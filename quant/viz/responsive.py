"""
Responsive research charting for Jupyter — stays smooth at millions of 1-minute candles.

Built on plotly-resampler (FigureWidgetResampler): a live relayout callback re-fetches only
the visible window on every zoom/pan and redraws ~a few thousand points per trace, using
MinMaxLTTB decimation that PRESERVES local highs/lows (wicks/spikes survive). Line overlays and
trade markers are decimated automatically; candlesticks are re-aggregated to the viewport (the
OHLC equivalent of decimation, which is what TradingView does under the hood).

Usage (in a notebook cell):
    from quant.viz import ResearchChart
    ch = ResearchChart(df, candles=True)
    ch.add_ema(50); ch.add_ema(200)
    ch.add_trades(res.trades)
    ch.show()                     # inline, responsive; pan/zoom reloads higher resolution

Toggle any layer via the legend (click). If plotly-resampler is unavailable, everything falls
back to the static (pre-downsampled) charts in quant.viz.charts.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..indicators import ema as _ema

try:
    from plotly_resampler import FigureWidgetResampler
    from plotly_resampler.aggregation import MinMaxLTTB
    _HAVE_RESAMPLER = True
except Exception:  # pragma: no cover
    _HAVE_RESAMPLER = False


_UP = "#16a34a"
_DOWN = "#dc2626"
_PRICE = "#3b82f6"


def _target_freq_minutes(span_minutes: float, target: int) -> int:
    return max(1, int(span_minutes / max(target, 1)))


def _aggregate_candles(df: pd.DataFrame, x0, x1, *, time_col: str, target: int) -> pd.DataFrame:
    view = df[(df[time_col] >= x0) & (df[time_col] <= x1)]
    if len(view) <= target:
        return view[[time_col, "open", "high", "low", "close"]]
    span_min = (pd.Timestamp(x1) - pd.Timestamp(x0)).total_seconds() / 60.0
    freq = _target_freq_minutes(span_min, target)
    agg = (view.set_index(time_col)
           .resample(f"{freq}min", label="left", closed="left")
           .agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last")).dropna().reset_index())
    return agg


class ResearchChart:
    """Interactive, viewport-resampling price chart for Jupyter."""

    def __init__(self, df: pd.DataFrame, *, time_col: str = "t", candles: bool = True,
                 n_shown: int = 3000, candle_target: int = 400, height: int = 680,
                 initial_bars: int = 400, title: str = "Price"):
        if time_col not in df.columns:
            raise ValueError(f"df must contain '{time_col}'")
        self.df = df.reset_index(drop=True)
        self.time_col = time_col
        self.candles = candles
        self.candle_target = candle_target
        # Open zoomed to the most recent `initial_bars` bars so candles render at their NATIVE
        # resolution (not the whole history binned into multi-hour blobs). Pan/zoom for the rest.
        self.initial_bars = min(int(initial_bars), len(self.df))
        self._overlays: List[str] = []
        self._built = False

        if _HAVE_RESAMPLER:
            self.fig = FigureWidgetResampler(
                go.Figure(), default_downsampler=MinMaxLTTB(), default_n_shown_samples=n_shown,
                verbose=False)
        else:
            self.fig = go.Figure()
        self.fig.update_layout(title=title, template="plotly_white", height=height,
                               margin=dict(l=50, r=30, t=55, b=40), dragmode="pan",
                               legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                               xaxis_rangeslider_visible=False)

    # ---- layers ----
    def add_line(self, col: str, *, name: Optional[str] = None, width: float = 1.2,
                 color: Optional[str] = None) -> "ResearchChart":
        name = name or col
        tr = go.Scattergl(name=name, mode="lines", line=dict(width=width, color=color))
        self._add_hf(tr, self.df[self.time_col], self.df[col])
        self._overlays.append(col)
        return self

    def add_ema(self, period: int, *, source: str = "close", color: Optional[str] = None) -> "ResearchChart":
        y = _ema(self.df[source], period)
        tr = go.Scattergl(name=f"EMA{period}", mode="lines", line=dict(width=1.1, color=color))
        self._add_hf(tr, self.df[self.time_col], y)
        return self

    def add_trades(self, trades: pd.DataFrame, *, markers: bool = True) -> "ResearchChart":
        # Trade markers are added as PLAIN traces (not resampled) so their per-trade hover data
        # stays intact — there are only a few thousand, which renders fine.
        if trades is None or trades.empty or not markers:
            return self
        t = trades.copy()
        for c in ("entry_time", "exit_time"):
            t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d %H:%M")
        if "close_reason" not in t:
            t["close_reason"] = ""
        if "return_pct" not in t:
            t["return_pct"] = float("nan")

        def _cd(sub):
            return np.column_stack([
                sub["side"].astype(str),
                sub["entry_time"], sub["entry_price"].round(3).astype(str),
                sub["exit_time"], sub["exit_price"].round(3).astype(str),
                sub["pnl"].round(2).astype(str),
                sub["return_pct"].round(2).astype(str),
                sub["close_reason"].astype(str),
            ])

        HOVER = ("<b>%{customdata[0]} trade</b><br>"
                 "entry: %{customdata[1]} @ %{customdata[2]}<br>"
                 "exit:  %{customdata[3]} @ %{customdata[4]}<br>"
                 "pnl:   %{customdata[5]}  (%{customdata[6]}%)<br>"
                 "reason: %{customdata[7]}<extra></extra>")

        def _markers(sub, xcol, ycol, name, symbol, color, size=10):
            if sub.empty:
                return
            self.fig.add_trace(go.Scattergl(
                x=sub[xcol], y=sub[ycol], mode="markers", name=name,
                marker=dict(symbol=symbol, size=size, color=color, line=dict(width=0.6, color="white")),
                customdata=_cd(sub), hovertemplate=HOVER))

        _markers(t[t["side"] == "long"], "entry_time", "entry_price", "long entry", "triangle-up", _UP)
        _markers(t[t["side"] == "short"], "entry_time", "entry_price", "short entry", "triangle-down", _DOWN)
        _markers(t[t["pnl"] > 0], "exit_time", "exit_price", "exit (win)", "x", _UP, size=8)
        _markers(t[t["pnl"] <= 0], "exit_time", "exit_price", "exit (loss)", "x", _DOWN, size=8)
        return self

    # ---- build / show ----
    def _add_hf(self, trace, x, y):
        if _HAVE_RESAMPLER:
            self.fig.add_trace(trace, hf_x=np.asarray(x), hf_y=np.asarray(y))
        else:
            trace.x = np.asarray(x)
            trace.y = np.asarray(y)
            self.fig.add_trace(trace)

    def _initial_range(self):
        i0 = max(0, len(self.df) - self.initial_bars)
        return self.df[self.time_col].iloc[i0], self.df[self.time_col].iloc[-1]

    def _build_candles(self):
        if not self.candles:
            return
        x0, x1 = self._initial_range()   # candles at native resolution for the opening window
        agg = _aggregate_candles(self.df, x0, x1, time_col=self.time_col, target=self.candle_target)
        self.fig.add_trace(go.Candlestick(
            x=agg[self.time_col], open=agg["open"], high=agg["high"], low=agg["low"],
            close=agg["close"], name="candles",
            increasing_line_color=_UP, decreasing_line_color=_DOWN))
        self._candle_idx = len(self.fig.data) - 1
        if _HAVE_RESAMPLER and hasattr(self.fig, "layout"):
            try:
                self.fig.layout.on_change(self._on_zoom, "xaxis.range")
            except Exception:
                pass

    def _on_zoom(self, layout, xrange):
        if not xrange:
            return
        x0, x1 = xrange
        agg = _aggregate_candles(self.df, x0, x1, time_col=self.time_col, target=self.candle_target)
        with self.fig.batch_update():
            c = self.fig.data[self._candle_idx]
            c.x, c.open, c.high, c.low, c.close = (
                agg[self.time_col], agg["open"], agg["high"], agg["low"], agg["close"])

    def show(self):
        if not self._built:
            # resampled close line = zoomed-out backbone (thin, so candles read as primary)
            self._add_hf(go.Scattergl(name="close", mode="lines",
                         line=dict(width=0.8, color=_PRICE), opacity=0.5),
                         self.df[self.time_col], self.df["close"])
            self._build_candles()
            x0, x1 = self._initial_range()
            self.fig.update_xaxes(range=[x0, x1])   # open zoomed to the recent window
            self._built = True
        return self.fig


def price_chart(df, *, overlays: Optional[Sequence[str]] = None, trades=None,
                candles: bool = True, emas: Optional[Sequence[int]] = None, **kw) -> "go.Figure":
    """One-call responsive price chart. Returns a FigureWidget (inline in Jupyter)."""
    ch = ResearchChart(df, candles=candles, **kw)
    for p in (emas or []):
        ch.add_ema(p)
    for c in (overlays or []):
        ch.add_line(c)
    if trades is not None:
        ch.add_trades(trades)
    return ch.show()


def equity_chart(equity_curve: pd.DataFrame, *, time_col: str = "t", height: int = 320,
                 title: str = "Equity") -> "go.Figure":
    """Responsive equity curve (drawdown available via the drawdown column)."""
    if _HAVE_RESAMPLER:
        fig = FigureWidgetResampler(go.Figure(), default_downsampler=MinMaxLTTB(),
                                    default_n_shown_samples=2500, verbose=False)
        fig.add_trace(go.Scattergl(name="equity", line=dict(color=_PRICE, width=1.3)),
                      hf_x=equity_curve[time_col].to_numpy(), hf_y=equity_curve["equity"].to_numpy())
    else:
        from .charts import equity_and_drawdown
        return equity_and_drawdown(equity_curve, time_col=time_col, title=title)
    fig.update_layout(title=title, template="plotly_white", height=height,
                      margin=dict(l=50, r=30, t=50, b=35), dragmode="pan")
    return fig
