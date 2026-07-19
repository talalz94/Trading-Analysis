from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol
import pandas as pd
import plotly.graph_objects as go


class Indicator(Protocol):
    name: str
    is_overlay: bool
    row_weight: float

    def compute(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame: ...
    def add_traces(self, fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], row: int, price_row: int) -> None: ...
    def yaxis_title(self, cfg: Dict[str, Any]) -> str: ...


# --- import indicator implementations
from indicators.rsi_divergence import RSI_Divergence
from indicators.macd import MACD
from indicators.stochastic import Stochastic
from indicators.volume_ma import VolumeMA
from indicators.supertrend import Supertrend
from indicators.bollinger_bands import BollingerBands
from indicators.momentum import Momentum
from indicators.moving_average import MovingAverage
from indicators.precomputed import Precomputed
from indicators.market_structure import MarketStructure

INDICATOR_REGISTRY = {
    RSI_Divergence.name: RSI_Divergence,
    MACD.name: MACD,
    Stochastic.name: Stochastic,
    VolumeMA.name: VolumeMA,
    Supertrend.name: Supertrend,
    BollingerBands.name: BollingerBands,
    Momentum.name: Momentum,
    MovingAverage.name: MovingAverage,
    Precomputed.name: Precomputed,
    MarketStructure.name: MarketStructure,
}