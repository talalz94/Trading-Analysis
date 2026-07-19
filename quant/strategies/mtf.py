"""
Multi-timeframe trend strategy: 1m entry trigger + 5m trend filter + 15m momentum filter.

Demonstrates the MTF utilities: higher-timeframe features are resampled, computed, and aligned
back onto the 1m grid lookahead-safe (see quant.data.timeframe).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import build_mtf
from ..engine import Signals
from ..indicators import add_emas
from .. import signals as S
from .base import Strategy


@dataclass
class MtfTrend(Strategy):
    name: str = "mtf_trend"
    fast_1m: int = 50             # 1m entry EMA
    trend_5m_fast: int = 50       # 5m trend EMAs
    trend_5m_slow: int = 200
    mom_15m: int = 50             # 15m momentum EMA (must be rising)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        base = add_emas(df, [int(self.fast_1m)])
        return build_mtf(base, {
            "5min": lambda d: add_emas(d, [int(self.trend_5m_fast), int(self.trend_5m_slow)]),
            "15min": lambda d: add_emas(d, [int(self.mom_15m)]),
        })

    def build_signals(self, df: pd.DataFrame) -> Signals:
        f1 = f"ema_{int(self.fast_1m)}"
        t5f = f"5min__ema_{int(self.trend_5m_fast)}"
        t5s = f"5min__ema_{int(self.trend_5m_slow)}"
        m15 = f"15min__ema_{int(self.mom_15m)}"
        entry_long = S.all_of(
            S.cross_up(df, "close", f1),        # 1m trigger
            S.above(df, t5f, t5s),              # 5m uptrend
            S.rising(df, m15, 1),               # 15m momentum up
        )
        exit_long = S.cross_down(df, "close", f1)
        return Signals(entry_long=entry_long, exit_long=exit_long)
