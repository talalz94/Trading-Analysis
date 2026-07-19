from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import parse_show_last
from pipeline import IndicatorSpec, build_feature_df
from indicators_load import INDICATOR_REGISTRY


@dataclass(frozen=True)
class PlotConfig:
    tz: str = "Asia/Karachi"
    show_last: Optional[str] = None

    # Notebook-friendly sizing
    height: int = 1300
    width: int = 1850

    template: str = "plotly_white"
    vertical_spacing: float = 0.03
    price_row_weight: float = 0.62  # if panels exist
    date_from: Optional[str] = None   # e.g. "2026-02-10 09:00"
    date_to: Optional[str] = None     # e.g. "2026-02-12 18:00"

def _parse_dt_in_tz(dt_str: str, tz: str) -> pd.Timestamp:
    ts = pd.to_datetime(dt_str)
    # if naive, assume it's in tz
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts
    
def plot_interactive(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    market: str = "spot",
    ma_windows: Optional[List[int]] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    plot_cfg: Optional[PlotConfig] = None,
    show: bool = True,
    return_fig: bool = False,
    return_feature_df: bool = True,
) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame], Optional[List[IndicatorSpec]]]:
    """
    If return_feature_df=True, returns df_feat (full) and indicator specs with .cols populated.
    Plot slicing (show_last) affects only plotting, not df_feat.
    """
    ma_windows = ma_windows or []
    indicators = indicators or []
    plot_cfg = plot_cfg or PlotConfig()

    df_feat, indicators, ma_cols = build_feature_df(
        raw_df=df,
        tz=plot_cfg.tz,
        ma_windows=ma_windows,
        indicators=indicators,
    )

    # Slice for display only
    plot_df = df_feat
    
    # ✅ Date range takes priority
    if plot_cfg.date_from or plot_cfg.date_to:
        if plot_cfg.date_from:
            d0 = _parse_dt_in_tz(plot_cfg.date_from, plot_cfg.tz)
            plot_df = plot_df[plot_df["t"] >= d0]
        if plot_cfg.date_to:
            d1 = _parse_dt_in_tz(plot_cfg.date_to, plot_cfg.tz)
            plot_df = plot_df[plot_df["t"] <= d1]
    
    # fallback to show_last
    elif plot_cfg.show_last:
        window = parse_show_last(plot_cfg.show_last)
        tmax = plot_df["t"].max()
        plot_df = plot_df[plot_df["t"] >= (tmax - window)]
    
    plot_df = plot_df.copy()

    # Separate overlay vs panel indicators
    overlay_specs: List[IndicatorSpec] = []
    panel_specs: List[IndicatorSpec] = []
    for spec in indicators:
        ind = INDICATOR_REGISTRY[spec.name]
        (overlay_specs if ind.is_overlay else panel_specs).append(spec)

    rows = 1 + len(panel_specs)
    if rows == 1:
        row_heights = [1.0]
    else:
        remaining = 1.0 - float(plot_cfg.price_row_weight)
        remaining = remaining if remaining > 0 else 0.38
        weights = [INDICATOR_REGISTRY[s.name].row_weight for s in panel_specs]
        wsum = sum(weights) if sum(weights) > 0 else len(weights)
        panel_heights = [remaining * (w / wsum) for w in weights]
        row_heights = [float(plot_cfg.price_row_weight)] + panel_heights

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=plot_cfg.vertical_spacing,
        specs=[[{"secondary_y": False}] for _ in range(rows)],
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=plot_df["t"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="OHLC",
            customdata=plot_df[["volume","quote_volume","num_trades"]].to_numpy(),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Open: %{open}<br>"
                "High: %{high}<br>"
                "Low: %{low}<br>"
                "Close: %{close}<br>"
                "Volume: %{customdata[0]:,.4f}<br>"
                "Quote Vol: %{customdata[1]:,.2f}<br>"
                "Trades: %{customdata[2]:,.0f}<br>"
                "<extra></extra>"
            ),
        ),
        row=1, col=1
    )

    # MA overlays
    for c in ma_cols:
        fig.add_trace(
            go.Scatter(
                x=plot_df["t"], y=plot_df[c],
                mode="lines",
                name=c,
                hovertemplate=f"<b>%{{x}}</b><br>{c}: %{{y:.6f}}<extra></extra>",
            ),
            row=1, col=1
        )

    # Overlay indicators
    for spec in overlay_specs:
        ind = INDICATOR_REGISTRY[spec.name]
        ind.add_traces(fig, plot_df, spec.config, spec.tag, row=1, price_row=1)

    # Panel indicators
    r = 2
    for spec in panel_specs:
        ind = INDICATOR_REGISTRY[spec.name]
        ind.add_traces(fig, plot_df, spec.config, spec.tag, row=r, price_row=1)
        title = ind.yaxis_title(spec.config, spec.tag)
        if title:
            fig.update_yaxes(title_text=title, row=r, col=1)
        r += 1

    fig.update_layout(
        title=f"{symbol.upper()} ({market}) — {interval}",
        template=plot_cfg.template,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=plot_cfg.height,
        width=plot_cfg.width,
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)

    if show:
        fig.show()

    out_fig = fig if return_fig else None
    out_df = df_feat if return_feature_df else None
    out_specs = indicators if return_feature_df else None
    return out_fig, out_df, out_specs