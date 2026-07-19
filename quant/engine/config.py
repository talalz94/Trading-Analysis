"""
Backtest configuration + signal container (Python-side; translated to kernel scalars/arrays).

Supports the full exit model: stop-loss (entry_pct / price_abs / ref_col structure stops with
buffer + max-risk cap + fallback), trailing stops, and multiple laddered take-profits each with
partial close and optional post-TP stop movement (breakeven / entry_pct / price_abs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SL_MODES = {"none": 0, "entry_pct": 1, "price_abs": 2, "ref_col": 3}
_TP_MODES = {"entry_pct": 1, "price_abs": 2, "rr": 3}
_SIZING_MODES = {"cash": 0, "risk_pct_equity": 1, "risk_amount": 2, "lots": 3}
_MOVE_MODES = {"none": 0, "breakeven": 1, "entry_pct": 2, "price_abs": 3}
_TRAIL_MODES = {"none": 0, "pct": 1, "price_abs": 2}
_FALLBACK_MODES = {"entry_pct": 1, "price_abs": 2}

MAX_TP = 6  # maximum laddered take-profit levels per trade


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
class TakeProfit:
    """One take-profit level.

    mode: 'entry_pct' | 'price_abs' | 'rr'
    close_pct: percent of the CURRENT remaining position to close at this level.
    move_stop_mode: after this TP fills, move the stop ('none'|'breakeven'|'entry_pct'|'price_abs').

    Note: TP price levels are computed ONCE at entry (rr uses the original stop distance). This is
    intentional and differs from the legacy simulator, which recomputed rr TPs each bar from the
    live stop — so a breakeven stop-move there collapsed later rr TPs onto entry. Fixing levels at
    entry is the intended, more predictable behavior.
    """
    mode: str
    value: float
    close_pct: float = 100.0
    move_stop_mode: str = "none"
    move_stop_value: float = 0.0


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

    # --- stop loss ---
    sl_mode: str = "none"           # none | entry_pct | price_abs | ref_col
    sl_value: float = 0.0
    sl_buffer_pct: float = 0.0      # ref_col: buffer beyond the structure level
    sl_max_ref_risk_pct: float = 0.0  # ref_col: cap; 0 = no cap (use fallback if exceeded)
    sl_fallback_mode: str = "entry_pct"  # used when ref_col unusable
    sl_fallback_value: float = 0.75
    sl_ref_long_col: Optional[str] = None   # column of long stop levels (e.g. swing low)
    sl_ref_short_col: Optional[str] = None  # column of short stop levels (e.g. swing high)

    # --- trailing stop ---
    trail_mode: str = "none"        # none | pct | price_abs
    trail_value: float = 0.0

    # --- take profits (laddered / partial) ---
    take_profits: Tuple[TakeProfit, ...] = ()
    # Back-compat convenience single TP (used only if take_profits is empty):
    tp_mode: str = "none"           # none | entry_pct | price_abs | rr
    tp_value: float = 0.0

    # --- sizing ---
    sizing_mode: str = "cash"       # cash | risk_pct_equity | risk_amount | lots
    sizing_value: float = 1.0
    max_notional_pct: float = 100.0
    allow_leverage: bool = False

    # --- margin / leverage (Exness-style; opt-in) ---
    margin_enabled: bool = False    # when True: margin accounting + stop-out liquidation
    leverage: float = 1.0           # e.g. 100 for 1:100, 500 for 1:500
    contract_size: float = 1.0      # units per lot (gold XAUUSD = 100 oz/lot; crypto spot = 1)
    stop_out_level: float = 0.0     # margin level %% at which open positions are liquidated
    margin_call_level: float = 0.0  # informational; margin level %% flagged in equity stats

    allow_rule_close: bool = True
    intrabar_priority: str = "stop_first"   # stop_first | take_profit_first

    def resolved_cash_per_trade(self) -> float:
        if self.cash_per_trade is not None:
            return float(self.cash_per_trade)
        return float(self.initial_cash) / max(int(self.max_open_trades), 1)

    def _tp_list(self) -> List[TakeProfit]:
        if self.take_profits:
            return list(self.take_profits)
        if self.tp_mode and self.tp_mode != "none":
            return [TakeProfit(mode=self.tp_mode, value=self.tp_value, close_pct=100.0)]
        return []

    def tp_arrays(self):
        """Fixed-size arrays describing take-profit levels for the kernel."""
        tps = self._tp_list()
        n_tp = min(len(tps), MAX_TP)
        modes = np.zeros(MAX_TP, np.int64)
        values = np.zeros(MAX_TP, np.float64)
        close_pcts = np.zeros(MAX_TP, np.float64)
        mv_modes = np.zeros(MAX_TP, np.int64)
        mv_values = np.zeros(MAX_TP, np.float64)
        for k in range(n_tp):
            tp = tps[k]
            modes[k] = _TP_MODES[tp.mode]
            values[k] = float(tp.value)
            close_pcts[k] = float(tp.close_pct)
            mv_modes[k] = _MOVE_MODES[tp.move_stop_mode]
            mv_values[k] = float(tp.move_stop_value)
        return n_tp, modes, values, close_pcts, mv_modes, mv_values

    def scalar_args(self) -> dict:
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
            sl_buffer_pct=float(self.sl_buffer_pct),
            sl_max_ref_risk_pct=float(self.sl_max_ref_risk_pct),
            sl_fallback_mode=_FALLBACK_MODES[self.sl_fallback_mode],
            sl_fallback_value=float(self.sl_fallback_value),
            trail_mode=_TRAIL_MODES[self.trail_mode],
            trail_value=float(self.trail_value),
            sizing_mode=_SIZING_MODES[self.sizing_mode],
            sizing_value=float(self.sizing_value),
            max_notional_pct=float(self.max_notional_pct),
            allow_leverage=1 if self.allow_leverage else 0,
            margin_enabled=1 if self.margin_enabled else 0,
            leverage=float(self.leverage) if self.leverage and self.leverage > 0 else 1.0,
            contract_size=float(self.contract_size) if self.contract_size and self.contract_size > 0 else 1.0,
            stop_out_level=float(self.stop_out_level),
            allow_rule_close=1 if self.allow_rule_close else 0,
            intrabar_stop_first=1 if self.intrabar_priority == "stop_first" else 0,
        )
