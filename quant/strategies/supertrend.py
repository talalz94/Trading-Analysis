"""Supertrend flip strategy: enter when Supertrend direction turns up, exit when it turns down."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engine import Signals
from ..indicators import add_supertrend
from .. import signals as S
from .base import Strategy


@dataclass
class SupertrendFlip(Strategy):
    name: str = "supertrend"
    period: int = 10
    multiplier: float = 3.0
    allow_short_signals: bool = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_supertrend(df, self.period, self.multiplier)

    def build_signals(self, df: pd.DataFrame) -> Signals:
        entry_long = S.cross_up(df, "st_dir", 0.0)     # dir flips -1 -> +1
        exit_long = S.cross_down(df, "st_dir", 0.0)
        entry_short = exit_short = None
        if self.allow_short_signals:
            entry_short = S.cross_down(df, "st_dir", 0.0)
            exit_short = S.cross_up(df, "st_dir", 0.0)
        return Signals(entry_long=entry_long, exit_long=exit_long,
                      entry_short=entry_short, exit_short=exit_short)
