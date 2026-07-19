"""
Plug-and-play strategy interface.

A strategy is a small dataclass carrying its parameters, plus two methods:
  - prepare(df)        -> add the indicator columns it needs (vectorized, reusable)
  - build_signals(df)  -> Signals (raw entry/exit boolean arrays)

The base class also gives EVERY strategy free, consistent time-filtering (session / hours /
weekday / weekend) applied to entries via `signals(df)`, which both single runs and sweeps use.

Adding a new strategy = one dataclass. Parameter optimization = sweeping the dataclass fields;
no strategy code changes. See ema_ribbon.py / rsi.py / heikin_ashi.py for references.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .. import signals as S
from ..engine import BacktestConfig, Signals, SimResult, run_backtest


@dataclass
class Strategy(ABC):
    name: str = "strategy"

    # Universal optional time filters (applied to ENTRIES only).
    session: Optional[str] = None                 # 'london' | 'newyork' | 'tokyo' | 'sydney'
    hours: Optional[Tuple[int, int]] = None       # (start_hour, end_hour) local to `time_col` tz
    weekdays: Optional[Tuple[int, ...]] = None    # 0=Mon .. 6=Sun
    avoid_weekends: bool = False
    filter_tz: Optional[str] = None               # evaluate filters in this tz (else the t-col tz)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with any indicator columns this strategy needs. Default: no-op."""
        return df

    @abstractmethod
    def build_signals(self, df: pd.DataFrame) -> Signals:
        """Return raw entry/exit signal arrays from prepared columns."""

    # ---- time filter ----
    def _time_mask(self, df: pd.DataFrame, time_col: str = "t") -> Optional[np.ndarray]:
        masks = []
        if self.session:
            masks.append(S.in_session(df, self.session, time_col=time_col, tz=self.filter_tz))
        if self.hours:
            masks.append(S.hour_between(df, self.hours[0], self.hours[1],
                                        time_col=time_col, tz=self.filter_tz))
        if self.weekdays:
            masks.append(S.weekday_in(df, self.weekdays, time_col=time_col, tz=self.filter_tz))
        if self.avoid_weekends:
            masks.append(S.not_weekend(df, time_col=time_col, tz=self.filter_tz))
        if not masks:
            return None
        return S.all_of(*masks)

    def signals(self, df: pd.DataFrame, time_col: str = "t") -> Signals:
        """build_signals + time filter on entries (the method the framework calls)."""
        sig = self.build_signals(df)
        mask = self._time_mask(df, time_col=time_col)
        if mask is not None:
            el = np.asarray(sig.entry_long, bool) & mask
            es = None if sig.entry_short is None else (np.asarray(sig.entry_short, bool) & mask)
            sig = Signals(entry_long=el, exit_long=sig.exit_long,
                          entry_short=es, exit_short=sig.exit_short)
        return sig

    def params(self) -> dict:
        d = asdict(self)
        d.pop("name", None)
        return d

    def backtest(self, df: pd.DataFrame, cfg: BacktestConfig,
                 *, time_col: str = "t", price_col: str = "close") -> SimResult:
        prepared = self.prepare(df)
        signals = self.signals(prepared, time_col=time_col)
        return run_backtest(prepared, signals, cfg, time_col=time_col, price_col=price_col)
