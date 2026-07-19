"""
Plug-and-play strategy interface.

A strategy is a small dataclass carrying its parameters, plus two methods:
  - prepare(df)        -> add the indicator columns it needs (vectorized, reusable)
  - build_signals(df)  -> Signals (entry/exit boolean arrays)

Adding a new strategy = one dataclass. Parameter optimization = sweeping the dataclass
fields; no strategy code changes. See ema_ribbon.py for a reference implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd

from ..engine import BacktestConfig, Signals, SimResult, run_backtest


@dataclass
class Strategy(ABC):
    name: str = "strategy"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with any indicator columns this strategy needs. Default: no-op."""
        return df

    @abstractmethod
    def build_signals(self, df: pd.DataFrame) -> Signals:
        """Return entry/exit signal arrays from prepared columns."""

    def params(self) -> dict:
        d = asdict(self)
        d.pop("name", None)
        return d

    def backtest(self, df: pd.DataFrame, cfg: BacktestConfig,
                 *, time_col: str = "t", price_col: str = "close") -> SimResult:
        prepared = self.prepare(df)
        signals = self.build_signals(prepared)
        return run_backtest(prepared, signals, cfg, time_col=time_col, price_col=price_col)
