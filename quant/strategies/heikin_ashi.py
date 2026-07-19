"""Heikin Ashi trend strategy: enter after N consecutive bullish HA candles, exit on colour flip."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engine import Signals
from ..indicators import add_heikin_ashi
from .. import signals as S
from .base import Strategy


@dataclass
class HeikinAshiTrend(Strategy):
    name: str = "heikin_ashi"
    n_consec: int = 3             # consecutive bullish HA candles to enter long
    allow_short_signals: bool = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_heikin_ashi(df)

    def build_signals(self, df: pd.DataFrame) -> Signals:
        # HA candle is bullish when ha_close > ha_open.
        entry_long = S.last_all_above(df, "ha_close", "ha_open", int(self.n_consec))
        exit_long = S.below(df, "ha_close", "ha_open")           # first bearish HA (colour reversal)
        entry_short = exit_short = None
        if self.allow_short_signals:
            entry_short = S.last_all_below(df, "ha_close", "ha_open", int(self.n_consec))
            exit_short = S.above(df, "ha_close", "ha_open")
        return Signals(entry_long=entry_long, exit_long=exit_long,
                      entry_short=entry_short, exit_short=exit_short)
