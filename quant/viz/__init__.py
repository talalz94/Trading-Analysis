"""
Visualization.

Primary (Jupyter, responsive at millions of points): `ResearchChart`, `price_chart`,
`equity_chart` from `responsive` — viewport-resampling via plotly-resampler.

Fallback (static, pre-downsampled, works anywhere incl. HTML export): `price_and_trades`,
`equity_and_drawdown` from `charts`.
"""
from __future__ import annotations

from .charts import equity_and_drawdown, price_and_trades
from .responsive import ResearchChart, equity_chart, price_chart, _HAVE_RESAMPLER
from .heatmaps import hour_weekday_heatmap, monthly_returns_heatmap, sweep_heatmap
from .export import heikin_ashi_candles, open_in_browser
from .studies import add_pattern_overlays, pattern_figure, scenario_figure, squeeze_momentum_figure, studies_figure, zoom_to_trade, CROSSHAIR_JS

__all__ = [
    "ResearchChart", "price_chart", "equity_chart",
    "price_and_trades", "equity_and_drawdown", "_HAVE_RESAMPLER",
    "monthly_returns_heatmap", "hour_weekday_heatmap", "sweep_heatmap",
    "heikin_ashi_candles", "open_in_browser", "scenario_figure", "add_pattern_overlays", "pattern_figure", "zoom_to_trade", "squeeze_momentum_figure", "studies_figure", "CROSSHAIR_JS",
]
