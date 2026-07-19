"""MACD trend strategy: enter on MACD/signal cross (optionally only above/below the zero line)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engine import Signals
from ..indicators import add_macd
from .. import signals as S
from .base import Strategy


@dataclass
class MacdTrend(Strategy):
    name: str = "macd_trend"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    require_zero_side: bool = True   # long crosses only when MACD > 0, short only when < 0
    allow_short_signals: bool = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_macd(df, self.fast, self.slow, self.signal)

    def build_signals(self, df: pd.DataFrame) -> Signals:
        up = S.cross_up(df, "macd", "macd_signal")
        dn = S.cross_down(df, "macd", "macd_signal")
        if self.require_zero_side:
            entry_long = S.all_of(up, S.above(df, "macd", 0.0))
        else:
            entry_long = up
        exit_long = dn
        entry_short = exit_short = None
        if self.allow_short_signals:
            entry_short = S.all_of(dn, S.below(df, "macd", 0.0)) if self.require_zero_side else dn
            exit_short = up
        return Signals(entry_long=entry_long, exit_long=exit_long,
                      entry_short=entry_short, exit_short=exit_short)
