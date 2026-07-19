"""
Key-level strategy: buy bounces off swing-low support, take profit into swing-high resistance.

Pair with a ref_col structure stop for a full "buy support, stop below the swing low" setup:
    BacktestConfig(exit_enabled=True, sl_mode="ref_col", sl_ref_long_col="swing_last_low",
                   sl_buffer_pct=0.05, tp_mode="rr", tp_value=2.0, sizing_mode="risk_pct_equity")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..engine import Signals
from ..indicators import add_swings
from .. import signals as S
from .base import Strategy


@dataclass
class KeyLevelBounce(Strategy):
    name: str = "key_level"
    left: int = 10
    right: int = 10
    near_pct: float = 0.15        # "near" a level = within this % of price

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_swings(df, self.left, self.right)

    def build_signals(self, df: pd.DataFrame) -> Signals:
        close = df["close"].to_numpy(np.float64)
        sup = df["swing_last_low"].to_numpy(np.float64)
        res = df["swing_last_high"].to_numpy(np.float64)
        thr = self.near_pct / 100.0

        with np.errstate(invalid="ignore"):
            dist_sup = (close - sup) / close
            dist_res = (res - close) / close
        near_sup = (dist_sup >= 0) & (dist_sup <= thr)
        near_res = (dist_res >= 0) & (dist_res <= thr)

        entry_long = S.all_of(near_sup, S.is_green(df))     # bounce candle at support
        exit_long = near_res                                 # into resistance
        return Signals(entry_long=entry_long, exit_long=exit_long)
