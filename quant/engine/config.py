"""Backtest configuration + signal container (Python-side; translated to kernel scalars)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_SL_MODES = {"none": 0, "entry_pct": 1, "price_abs": 2}
_TP_MODES = {"none": 0, "entry_pct": 1, "price_abs": 2, "rr": 3}
_SIZING_MODES = {"cash": 0, "risk_pct_equity": 1}


@dataclass
class Signals:
    """Precomputed boolean signal arrays (one per bar). Missing sides default to all-False."""
    entry_long: np.ndarray
    exit_long: Optional[np.ndarray] = None
    entry_short: Optional[np.ndarray] = None
    exit_short: Optional[np.ndarray] = None

    def as_u8(self, n: int):
        def u8(a):
            if a is None:
                return np.zeros(n, np.uint8)
            arr = np.asarray(a)
            if arr.shape[0] != n:
                raise ValueError(f"signal length {arr.shape[0]} != n_bars {n}")
            return arr.astype(np.uint8)
        return u8(self.entry_long), u8(self.exit_long), u8(self.entry_short), u8(self.exit_short)


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    cash_per_trade: Optional[float] = None
    max_open_trades: int = 1
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    allow_short: bool = False

    # Exit / risk module
    exit_enabled: bool = False
    sl_mode: str = "none"          # none | entry_pct | price_abs
    sl_value: float = 0.0
    tp_mode: str = "none"          # none | entry_pct | price_abs | rr
    tp_value: float = 0.0
    sizing_mode: str = "cash"      # cash | risk_pct_equity
    sizing_value: float = 1.0
    max_notional_pct: float = 100.0
    allow_leverage: bool = False
    allow_rule_close: bool = True
    intrabar_priority: str = "stop_first"   # stop_first | take_profit_first

    def resolved_cash_per_trade(self) -> float:
        if self.cash_per_trade is not None:
            return float(self.cash_per_trade)
        return float(self.initial_cash) / max(int(self.max_open_trades), 1)

    def kernel_args(self) -> dict:
        return dict(
            initial_cash=float(self.initial_cash),
            cash_per_trade=self.resolved_cash_per_trade(),
            fee_bps=float(self.fee_bps),
            slippage_bps=float(self.slippage_bps),
            max_open_trades=int(self.max_open_trades),
            allow_short=1 if self.allow_short else 0,
            exit_enabled=1 if self.exit_enabled else 0,
            sl_mode=_SL_MODES[self.sl_mode],
            sl_value=float(self.sl_value),
            tp_mode=_TP_MODES[self.tp_mode],
            tp_value=float(self.tp_value),
            sizing_mode=_SIZING_MODES[self.sizing_mode],
            sizing_value=float(self.sizing_value),
            max_notional_pct=float(self.max_notional_pct),
            allow_leverage=1 if self.allow_leverage else 0,
            allow_rule_close=1 if self.allow_rule_close else 0,
            intrabar_stop_first=1 if self.intrabar_priority == "stop_first" else 0,
        )
