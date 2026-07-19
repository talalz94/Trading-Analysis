"""
EMA-ribbon strategy (reference implementation).

Implements the pattern from the brief:
  BUY when price crosses above EMA(fast) AND price has stayed above EMA(slow) for
  `confirm_n` candles. Exit when price crosses back below EMA(fast).

All logic is vectorized signal primitives; the numba engine handles execution/exits.
Every field is a tunable parameter — sweep them without touching this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..engine import Signals
from ..indicators import add_emas
from .. import signals as S
from .base import Strategy


@dataclass
class EmaRibbon(Strategy):
    name: str = "ema_ribbon"
    fast: int = 50
    slow: int = 200
    confirm_n: int = 3
    source: str = "close"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        periods = sorted({int(self.fast), int(self.slow)})
        return add_emas(df, periods, source=self.source)

    def build_signals(self, df: pd.DataFrame) -> Signals:
        fast_col = f"ema_{int(self.fast)}"
        slow_col = f"ema_{int(self.slow)}"
        entry = S.all_of(
            S.cross_up(df, self.source, fast_col),
            S.last_all_above(df, self.source, slow_col, int(self.confirm_n)),
        )
        exit_ = S.cross_down(df, self.source, fast_col)
        return Signals(entry_long=entry, exit_long=exit_)
