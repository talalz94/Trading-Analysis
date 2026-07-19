"""RSI reversal strategy: enter after N consecutive oversold/overbought bars, exit past mid."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engine import Signals
from ..indicators import add_rsi
from .. import signals as S
from .base import Strategy


@dataclass
class RsiReversal(Strategy):
    name: str = "rsi_reversal"
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    long_consec: int = 3          # RSI < oversold for this many consecutive bars -> long
    short_consec: int = 5         # RSI > overbought for this many consecutive bars -> short
    exit_mid: float = 50.0
    allow_short_signals: bool = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_rsi(df, self.period)

    def build_signals(self, df: pd.DataFrame) -> Signals:
        r = f"rsi_{int(self.period)}"
        entry_long = S.last_all_below(df, r, self.oversold, int(self.long_consec))
        exit_long = S.above(df, r, self.exit_mid)
        entry_short = exit_short = None
        if self.allow_short_signals:
            entry_short = S.last_all_above(df, r, self.overbought, int(self.short_consec))
            exit_short = S.below(df, r, self.exit_mid)
        return Signals(entry_long=entry_long, exit_long=exit_long,
                      entry_short=entry_short, exit_short=exit_short)
