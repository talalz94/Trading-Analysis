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
    fig = ch.show()
    fig.show()   # IMPORTANT: call fig.show() (plotly's own renderer), not bare `fig` / `ch.show()`.
                 # A bare FigureWidget renders through Jupyter's ipywidgets widget-view mimetype
                 # (a separate JS bundle in the notebook frontend), which in some environments
                 # fails to draw Candlestick traces even though line/marker traces render fine.
                 # fig.show() instead goes through plotly.io's renderer, which draws correctly.

Toggle any layer via the legend (click). If plotly-resampler is unavailable, the class still
works (falls back to a plain go.Figure with all data un-decimated), just without live
re-aggregation on browser-side mouse pan/zoom — explicit `fig.update_xaxes(...)` calls still
refresh candles correctly either way.
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
                 initial_bars: int = 400, title: str = "Price", candle_name: str = "candles"):
        if time_col not in df.columns:
            raise ValueError(f"df must contain '{time_col}'")
        self.df = df.reset_index(drop=True)
        self.time_col = time_col
        self.candles = candles
        self.candle_name = candle_name
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
                               autosize=True, width=None,
                               margin=dict(l=50, r=30, t=55, b=40), dragmode="pan",
                               legend=dict(orientation="h", y=1.02, yanchor="bottom"),
                               xaxis_rangeslider_visible=False)

    # ---- layers ----
    # IMPORTANT: every price-chart trace uses go.Scatter (SVG), NOT go.Scattergl (WebGL).
    # Candlestick is an SVG-only trace (there is no Candlestickgl). Mixing a WebGL scattergl
    # layer with the SVG candlestick makes the candles disappear in many notebook front-ends
    # (the WebGL canvas composites over the SVG layer). Keeping everything on the SVG layer is
    # the reliable cross-renderer fix. plotly-resampler decimates each trace to ~n_shown points
    # (default 3000) before it reaches the browser, so SVG stays fast even on millions of bars.
    def add_line(self, col: str, *, name: Optional[str] = None, width: float = 1.2,
                 color: Optional[str] = None) -> "ResearchChart":
        name = name or col
        tr = go.Scatter(name=name, mode="lines", line=dict(width=width, color=color))
        self._add_hf(tr, self.df[self.time_col], self.df[col])
        self._overlays.append(col)
        return self

    def add_ema(self, period: int, *, source: str = "close", color: Optional[str] = None) -> "ResearchChart":
        y = _ema(self.df[source], period)
        tr = go.Scatter(name=f"EMA{period}", mode="lines", line=dict(width=1.1, color=color))
        self._add_hf(tr, self.df[self.time_col], y)
        return self

    def add_indicator(self, *handles, width: float = 1.2, colors: Optional[Sequence[str]] = None
                      ) -> "ResearchChart":
        """Plot tagged indicator handles by reading THEIR columns off the charted frame.

        This is the safe way to draw indicators: the line plotted is literally the column the
        rules evaluated, so the chart can never disagree with the strategy (unlike `add_ema`,
        which recomputes its own EMA). Requires charting the PREPARED frame.
        """
        for i, h in enumerate(handles):
            col = getattr(h, "column", h)
            if col not in self.df.columns:
                raise KeyError(
                    f"Column {col!r} not in the charted frame — chart the prepared frame "
                    f"(e.g. ResearchChart(strat.prepare(df), ...)) so indicator columns exist.")
            name = getattr(h, "label", str(col))
            color = colors[i] if colors and i < len(colors) else None
            self.add_line(col, name=name, width=width, color=color)
        return self

    def add_sr_zones(self, zones: pd.DataFrame, *, tiers=((1, 2), (3, 5), (6, 10 ** 9)),
                     color: str = "#7c3aed", time_col: Optional[str] = None) -> "ResearchChart":
        """Draw causal support/resistance zones from `sr_zone_table(df)`.

        Each row is a segment: the level a zone sat at over a bar range. Segments start at the bar
        the zone's first pivot was CONFIRMED, so what you see is what was actually known then — no
        end-state snapshot. Zones are grouped into strength tiers by touch count (thicker/more
        opaque = more touches) and each tier is one legend-toggleable trace.
        """
        if zones is None or zones.empty:
            return self
        tcol = time_col or ("start_t" if "start_t" in zones.columns else None)
        if tcol is None:
            raise ValueError("zone table needs start_t/end_t (pass a df with a time column)")
        for lo_t, hi_t in tiers:
            sub = zones[(zones["touches"] >= lo_t) & (zones["touches"] <= hi_t)]
            if sub.empty:
                continue
            xs, ys, txt = [], [], []
            for _, z in sub.iterrows():
                lbl = f"S/R {z['price']:.2f} · {int(z['touches'])} touches"
                xs += [z["start_t"], z["end_t"], None]
                ys += [z["price"], z["price"], None]
                txt += [lbl, lbl, None]
            width = 1.0 + 1.2 * tiers.index((lo_t, hi_t))
            opacity = 0.35 + 0.25 * tiers.index((lo_t, hi_t))
            name = f"S/R {lo_t}+ touches" if hi_t > 10 ** 8 else f"S/R {lo_t}-{hi_t} touches"
            self._add_static(go.Scatter(
                x=xs, y=ys, mode="lines", name=name, opacity=opacity,
                line=dict(color=color, width=width), text=txt, hoverinfo="text"))
        return self

    def add_trades(self, trades: pd.DataFrame, *, markers: bool = True,
                   stops: bool = True) -> "ResearchChart":
        # Defer rendering to show() so ALL resampled hf traces (close, EMAs) are added first and
        # the static overlays (markers, stops) come last — mixing the two desyncs the resampler.
        self._pending_trades = (trades, markers, stops)
        return self

    def _render_trades(self):
        trades, markers, stops = getattr(self, "_pending_trades", (None, True, True))
        if trades is None or trades.empty or not markers:
            return
        # Preserve the ORIGINAL trades index as the trade id, so it matches the row label in the
        # trade table (e.g. res.trades[cols]) for cross-referencing entry/exit/stop of one trade.
        trade_id = trades.index.to_numpy()
        t = trades.reset_index(drop=True)
        et = pd.to_datetime(t["entry_time"]).to_numpy()
        xt = pd.to_datetime(t["exit_time"]).to_numpy()
        ets = pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m-%d %H:%M").to_numpy()
        xts = pd.to_datetime(t["exit_time"]).dt.strftime("%Y-%m-%d %H:%M").to_numpy()
        side = t["side"].astype(str).to_numpy()
        entry_px = t["entry_price"].to_numpy(); exit_px = t["exit_price"].to_numpy()
        pnl = t["pnl"].to_numpy()
        rp = (t["return_pct"] if "return_pct" in t else t["pnl"]).round(2).astype(str).to_numpy()
        cr = (t["close_reason"].astype(str) if "close_reason" in t else pd.Series([""] * len(t))).to_numpy()
        stop = t["stop_price"].to_numpy() if "stop_price" in t else np.full(len(t), np.nan)

        def _cd(m):
            return np.column_stack([trade_id[m].astype(str), side[m], ets[m],
                                    np.round(entry_px[m], 3).astype(str),
                                    xts[m], np.round(exit_px[m], 3).astype(str),
                                    np.round(pnl[m], 2).astype(str), rp[m], cr[m]])

        HOVER = ("<b>trade #%{customdata[0]} — %{customdata[1]}</b><br>"
                 "entry: %{customdata[2]} @ %{customdata[3]}<br>"
                 "exit:  %{customdata[4]} @ %{customdata[5]}<br>pnl: %{customdata[6]} "
                 "(%{customdata[7]}%)<br>reason: %{customdata[8]}<extra></extra>")

        def _markers(m, x, y, name, symbol, color, size=10):
            if not m.any():
                return
            self._add_static(go.Scatter(
                x=x[m], y=y[m], mode="markers", name=name,
                marker=dict(symbol=symbol, size=size, color=color, line=dict(width=0.6, color="white")),
                customdata=_cd(m), hovertemplate=HOVER))

        _markers(side == "long", et, entry_px, "long entry", "triangle-up", _UP)
        _markers(side == "short", et, entry_px, "short entry", "triangle-down", _DOWN)
        _markers(pnl > 0, xt, exit_px, "exit (win)", "x", _UP, 8)
        _markers(pnl <= 0, xt, exit_px, "exit (loss)", "x", _DOWN, 8)

        # stop-loss: a short dotted horizontal segment at each trade's stop, spanning entry->exit.
        # Each segment carries its trade id in hovertext so it can be matched to the entry/exit
        # markers and the trade table above.
        if stops and np.isfinite(stop).any():
            xs, ys, txt = [], [], []
            for tid, e, x, s in zip(trade_id, et, xt, stop):
                if np.isfinite(s):
                    label = f"trade #{tid} stop-loss @ {s:.3f}"
                    xs += [e, x, None]; ys += [s, s, None]; txt += [label, label, None]
            self._add_static(go.Scatter(x=xs, y=ys, mode="lines", name="stop-loss",
                             line=dict(color="#f59e0b", width=1, dash="dot"),
                             text=txt, hoverinfo="text"))

    # ---- build / show ----
    def _add_hf(self, trace, x, y):
        if _HAVE_RESAMPLER:
            self.fig.add_trace(trace, hf_x=np.asarray(x), hf_y=np.asarray(y))
        else:
            trace.x = np.asarray(x)
            trace.y = np.asarray(y)
            self.fig.add_trace(trace)

    def _add_static(self, trace):
        # Overlay traces (markers, stop-loss lines) must NOT be resampled — marker/segmented data
        # (with None gaps) breaks the aggregator's monotonic check. Passing a huge max_n_samples
        # tells plotly-resampler to keep the trace as-is (no aggregation).
        if _HAVE_RESAMPLER:
            try:
                self.fig.add_trace(trace, max_n_samples=10_000_000)
                return
            except Exception:
                pass
        self.fig.add_trace(trace)

    def _initial_range(self):
        i0 = max(0, len(self.df) - self.initial_bars)
        return self.df[self.time_col].iloc[i0], self.df[self.time_col].iloc[-1]

    def add_candle_alt(self, alt_df: pd.DataFrame, *, name: str = "Heikin Ashi") -> "ResearchChart":
        """Add a second candlestick series (e.g. Heikin Ashi) alongside the primary one, plus a
        plot-level button to switch between them. `alt_df` needs the same time_col + OHLC columns
        as the primary df. Rendered hidden (visible=False) until toggled; the button is native
        Plotly `updatemenus` (client-side restyle), so it works in an exported static HTML too —
        no Python callback or live kernel required to flip it.
        """
        self._pending_candle_alt = (alt_df, name)
        return self

    def _build_candles(self):
        if not self.candles:
            return
        self._candle_sources: List[dict] = []
        self._add_candle_source(self.df, name=self.candle_name, visible=True)
        # Registering on_change works on a plain go.Figure too (it's a plotly.py trait callback,
        # not an ipywidgets-only feature) — so candles re-aggregate on ANY xaxis.range change,
        # including a plain `fig.update_xaxes(range=...)` call from Python. This is what keeps
        # candles visible when navigating away from the initial window even WITHOUT
        # plotly-resampler installed (browser-side mouse pan/zoom still needs a live ipywidgets
        # kernel + plotly-resampler to refresh, but scripted zooms always work).
        if hasattr(self.fig, "layout"):
            try:
                self.fig.layout.on_change(self._on_zoom, "xaxis.range")
            except Exception:
                pass

    def _add_candle_source(self, source_df: pd.DataFrame, *, name: str, visible: bool) -> None:
        x0, x1 = self._initial_range()   # candles at native resolution for the opening window
        agg = _aggregate_candles(source_df, x0, x1, time_col=self.time_col, target=self.candle_target)
        self.fig.add_trace(go.Candlestick(
            x=agg[self.time_col], open=agg["open"], high=agg["high"], low=agg["low"],
            close=agg["close"], name=name, visible=visible,
            increasing_line_color=_UP, decreasing_line_color=_DOWN))
        self._candle_sources.append({"df": source_df, "idx": len(self.fig.data) - 1, "name": name})
        if len(self._candle_sources) == 1:
            self._candle_idx = self._candle_sources[0]["idx"]   # back-compat for existing readers

    def _render_candle_alt(self):
        pending = getattr(self, "_pending_candle_alt", None)
        if pending is None or not self.candles:
            return
        alt_df, name = pending
        self._add_candle_source(alt_df, name=name, visible=False)
        primary_idx = self._candle_sources[0]["idx"]
        alt_idx = self._candle_sources[1]["idx"]
        n_traces = len(self.fig.data)

        def _visibility(shown_idx: int) -> List[bool]:
            v = [True] * n_traces
            v[primary_idx] = shown_idx == primary_idx
            v[alt_idx] = shown_idx == alt_idx
            return v

        self.fig.update_layout(updatemenus=[dict(
            type="buttons", direction="left", showactive=True,
            x=1.0, xanchor="right", y=1.16, yanchor="top", pad=dict(r=4, t=2),
            buttons=[
                dict(label=self._candle_sources[0]["name"], method="restyle",
                     args=[{"visible": _visibility(primary_idx)}]),
                dict(label=name, method="restyle",
                     args=[{"visible": _visibility(alt_idx)}]),
            ],
        )])

    def _on_zoom(self, layout, xrange):
        if not xrange:
            return
        x0, x1 = xrange
        with self.fig.batch_update():
            for src in getattr(self, "_candle_sources", []):
                agg = _aggregate_candles(src["df"], x0, x1, time_col=self.time_col,
                                         target=self.candle_target)
                c = self.fig.data[src["idx"]]
                c.x, c.open, c.high, c.low, c.close = (
                    agg[self.time_col], agg["open"], agg["high"], agg["low"], agg["close"])

    def show(self):
        if not self._built:
            # resampled close line = zoomed-out backbone (thin, so candles read as primary).
            # go.Scatter (SVG), not Scattergl — see the note on add_line/add_ema above.
            self._add_hf(go.Scatter(name="close", mode="lines",
                         line=dict(width=0.8, color=_PRICE), opacity=0.5),
                         self.df[self.time_col], self.df["close"])
            self._build_candles()
            self._render_candle_alt()
            self._render_trades()   # static overlays LAST, after all resampled hf traces
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
