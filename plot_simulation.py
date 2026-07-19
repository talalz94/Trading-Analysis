from __future__ import annotations
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go

from plotter import plot_interactive, PlotConfig
from pipeline import IndicatorSpec
from simulation.simulator import Trade

def _parse_dt_in_tz(dt_str: str, tz: str) -> pd.Timestamp:
    ts = pd.to_datetime(dt_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts


def _compute_plot_window(df_raw: pd.DataFrame, plot_cfg: PlotConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    tmp = df_raw.copy()
    tmp["t"] = tmp["open_time"].dt.tz_convert(plot_cfg.tz)

    if getattr(plot_cfg, "date_from", None) or getattr(plot_cfg, "date_to", None):
        t0 = _parse_dt_in_tz(plot_cfg.date_from, plot_cfg.tz) if plot_cfg.date_from else tmp["t"].min()
        t1 = _parse_dt_in_tz(plot_cfg.date_to, plot_cfg.tz) if plot_cfg.date_to else tmp["t"].max()
        return t0, t1

    if plot_cfg.show_last:
        from charting.data import parse_show_last
        window = parse_show_last(plot_cfg.show_last)
        tmax = tmp["t"].max()
        return (tmax - window), tmax

    return tmp["t"].min(), tmp["t"].max()


def plot_simulation(
    df_raw: pd.DataFrame,
    symbol: str,
    interval: str,
    market: str,
    trades: List[Trade],
    ma_windows: Optional[List[int]] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    plot_cfg: Optional[PlotConfig] = None,
):
    ma_windows = ma_windows or []
    indicators = indicators or []
    plot_cfg = plot_cfg or PlotConfig()

    # Build base figure (this already slices candles/indicators correctly)
    fig, _, _ = plot_interactive(
        df=df_raw,
        symbol=symbol,
        interval=interval,
        market=market,
        ma_windows=ma_windows,
        indicators=indicators,
        plot_cfg=plot_cfg,
        show=False,
        return_fig=True,
        return_feature_df=False,
    )

    # Determine visible window
    t_start, t_end = _compute_plot_window(df_raw, plot_cfg)

    # Force x-axis range so markers can't stretch it
    fig.update_xaxes(range=[t_start, t_end])

    # Prepare a lookup for highs/lows
    tmp = df_raw.copy()
    tmp["t"] = tmp["open_time"].dt.tz_convert(plot_cfg.tz)
    tmp = tmp.set_index("t")

    # Filter trades to those overlapping the window
    def overlaps(tr: Trade) -> bool:
        et = tr.entry_time
        xt = tr.exit_time if tr.exit_time is not None else tr.entry_time
        return (et <= t_end) and (xt >= t_start)

    trades_in_view = [tr for tr in trades if overlaps(tr)]

    # Markers
    entries_x, entries_y, entries_txt = [], [], []
    exits_x, exits_y, exits_txt = [], [], []

    for tr in trades_in_view:
        if tr.entry_time in tmp.index and (t_start <= tr.entry_time <= t_end):
            px = float(tmp.loc[tr.entry_time, "low"] if tr.side == "long" else tmp.loc[tr.entry_time, "high"])
            entries_x.append(tr.entry_time)
            entries_y.append(px)
            entries_txt.append(f"OPEN {tr.side} #{tr.trade_id}<br>{tr.open_reason}")

        if tr.exit_time is not None and tr.exit_time in tmp.index and (t_start <= tr.exit_time <= t_end):
            px = float(tmp.loc[tr.exit_time, "high"] if tr.side == "long" else tmp.loc[tr.exit_time, "low"])
            exits_x.append(tr.exit_time)
            exits_y.append(px)
            exits_txt.append(
                f"CLOSE {tr.side} #{tr.trade_id}<br>{tr.close_reason}<br>PNL: {tr.pnl:.2f}"
                if tr.pnl is not None else f"CLOSE #{tr.trade_id}"
            )

    fig.add_trace(
        go.Scatter(
            x=entries_x, y=entries_y,
            mode="markers",
            name="Entries",
            marker=dict(symbol="triangle-up", size=12, color="green", line=dict(width=2)),
            text=entries_txt,
            hovertemplate="%{text}<extra></extra>",
            cliponaxis=True,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=exits_x, y=exits_y,
            mode="markers",
            name="Exits",
            marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=2)),
            text=exits_txt,
            hovertemplate="%{text}<extra></extra>",
            cliponaxis=True,
        ),
        row=1, col=1
    )

    fig.show()