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
    # ALLOW-lists — say what to KEEP:
    session: Optional[str] = None                 # 'london' | 'newyork' | 'tokyo' | 'sydney'
    hours: Optional[Tuple[int, int]] = None       # (start_hour, end_hour), one contiguous window
    weekdays: Optional[Tuple[int, ...]] = None    # 0=Mon .. 6=Sun
    avoid_weekends: bool = False
    # BLOCK-lists — say what to EXCLUDE. Arbitrary/non-contiguous sets, which `hours` cannot express.
    block_hours: Optional[Tuple[int, ...]] = None       # e.g. (12, 16, 18) -> no entries in those hours
    block_weekdays: Optional[Tuple[int, ...]] = None    # e.g. (5, 6) -> no entries Sat/Sun
    block_times: Optional[Tuple[Tuple[str, str], ...]] = None  # (("13:30","14:15"), ("20:00","20:30"))
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
        # block-lists: keep everything EXCEPT these
        if self.block_hours:
            masks.append(~S.hour_in(df, self.block_hours, time_col=time_col, tz=self.filter_tz))
        if self.block_weekdays:
            masks.append(~S.weekday_in(df, self.block_weekdays, time_col=time_col, tz=self.filter_tz))
        if self.block_times:
            masks.append(~S.in_any_times(df, self.block_times, time_col=time_col, tz=self.filter_tz))
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
            # Preserve any per-rule leg arrays (unaffected by the entry mask: a masked-out entry
            # simply never opens, so its leg index is moot).
            sig = Signals(entry_long=el, exit_long=sig.exit_long,
                          entry_short=es, exit_short=sig.exit_short,
                          entry_leg_long=sig.entry_leg_long, entry_leg_short=sig.entry_leg_short,
                          exit_legs_long=sig.exit_legs_long, exit_legs_short=sig.exit_legs_short,
                          stop_specs=sig.stop_specs)
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
