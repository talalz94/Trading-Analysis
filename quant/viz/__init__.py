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

__all__ = [
    "ResearchChart", "price_chart", "equity_chart",
    "price_and_trades", "equity_and_drawdown", "_HAVE_RESAMPLER",
]
