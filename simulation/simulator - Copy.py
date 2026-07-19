from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any
import logging
import time
import math
import re

import numpy as np
import pandas as pd

from simulation.rules import RuleGroup
from simulation.context_mixins import RuleContextMixin


Side = Literal["long", "short"]
StopMode = Literal["entry_pct", "price_abs"]
SizingMode = Literal["cash", "risk_pct_equity", "risk_amount"]
TPMode = Literal["entry_pct", "price_abs", "rr"]
MoveStopMode = Literal["none", "breakeven", "entry_pct", "price_abs"]


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def _make_logger(name: str = "simulation.engine") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def _maybe_tqdm(enabled: bool, total: int, desc: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit="bar", leave=True)
    except Exception:
        return None


def _duration_minutes(t0: Any, t1: Any) -> Optional[float]:
    if t0 is None or t1 is None:
        return None
    try:
        return (t1 - t0).total_seconds() / 60.0
    except Exception:
        try:
            return pd.Timedelta(t1 - t0).total_seconds() / 60.0
        except Exception:
            return None


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    if len(values) == 0:
        return float(default)
    return float(np.mean(values))


def _safe_median(values: np.ndarray, default: float = 0.0) -> float:
    if len(values) == 0:
        return float(default)
    return float(np.median(values))


def _max_consecutive(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for x in mask:
        if bool(x):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _calc_stop_price(entry_price: float, side: str, stop_cfg: Optional[StopLossConfig]) -> Optional[float]:
    if stop_cfg is None:
        return None

    if stop_cfg.mode == "entry_pct":
        pct = stop_cfg.value / 100.0
        if side == "long":
            return entry_price * (1.0 - pct)
        return entry_price * (1.0 + pct)

    if stop_cfg.mode == "price_abs":
        if side == "long":
            return entry_price - stop_cfg.value
        return entry_price + stop_cfg.value

    raise ValueError(f"Unknown stop-loss mode: {stop_cfg.mode}")


def _calc_tp_price(
    entry_price: float,
    stop_price: Optional[float],
    side: str,
    tp_cfg: TakeProfitConfig,
) -> float:
    if tp_cfg.mode == "entry_pct":
        pct = tp_cfg.value / 100.0
        if side == "long":
            return entry_price * (1.0 + pct)
        return entry_price * (1.0 - pct)

    if tp_cfg.mode == "price_abs":
        if side == "long":
            return entry_price + tp_cfg.value
        return entry_price - tp_cfg.value

    if tp_cfg.mode == "rr":
        if stop_price is None:
            raise ValueError("TP mode='rr' requires stop_loss to be configured.")

        risk_per_unit = abs(entry_price - stop_price)

        if side == "long":
            return entry_price + (risk_per_unit * tp_cfg.value)
        return entry_price - (risk_per_unit * tp_cfg.value)

    raise ValueError(f"Unknown take-profit mode: {tp_cfg.mode}")


def _calc_moved_stop_price(
    entry_price: float,
    side: str,
    tp_cfg: TakeProfitConfig,
) -> Optional[float]:
    mode = tp_cfg.move_stop_mode

    if mode == "none":
        return None

    if mode == "breakeven":
        return entry_price

    if mode == "entry_pct":
        pct = tp_cfg.move_stop_value / 100.0

        if side == "long":
            return entry_price * (1.0 + pct)
        return entry_price * (1.0 - pct)

    if mode == "price_abs":
        if side == "long":
            return entry_price + tp_cfg.move_stop_value
        return entry_price - tp_cfg.move_stop_value

    raise ValueError(f"Unknown move_stop_mode: {mode}")


def _stop_hit(side: str, low: float, high: float, stop_price: Optional[float]) -> bool:
    if stop_price is None:
        return False

    if side == "long":
        return low <= stop_price

    return high >= stop_price


def _tp_hit(side: str, low: float, high: float, tp_price: float) -> bool:
    if side == "long":
        return high >= tp_price

    return low <= tp_price


def _apply_slippage(price: float, side: str, action: str, slippage_bps: float) -> float:
    """
    action:
      "entry"
      "exit"

    Conservative slippage:
      long entry worse = higher
      long exit worse = lower
      short entry worse = lower
      short exit worse = higher
    """
    slip = slippage_bps / 10_000.0

    if side == "long":
        if action == "entry":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    if side == "short":
        if action == "entry":
            return price * (1.0 - slip)
        return price * (1.0 + slip)

    raise ValueError(f"Unknown side: {side}")


def _pnl_for_qty(side: str, entry_price: float, exit_price: float, qty: float) -> float:
    if side == "long":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def _calc_qty_for_entry(
    cash: float,
    equity: float,
    entry_price: float,
    stop_price: Optional[float],
    cfg: SimConfig,
) -> float:
    """
    Returns quantity.

    If sizing.mode='cash', use existing cash_per_trade style.
    If sizing.mode='risk_pct_equity', size by account risk and stop distance.
    """
    sizing = cfg.exit.sizing

    max_notional = equity * (sizing.max_notional_pct_of_equity / 100.0)

    if not sizing.allow_leverage:
        max_notional = min(max_notional, cash)

    if sizing.mode == "cash":
        notional = min(max_notional, cash)
        return max(notional / entry_price, 0.0)

    if stop_price is None:
        raise ValueError("Risk-based sizing requires stop_loss to be configured.")

    risk_per_unit = abs(entry_price - stop_price)

    if risk_per_unit <= 0:
        return 0.0

    if sizing.mode == "risk_pct_equity":
        risk_amount = equity * (sizing.value / 100.0)
    elif sizing.mode == "risk_amount":
        risk_amount = sizing.value
    else:
        raise ValueError(f"Unknown sizing mode: {sizing.mode}")

    qty_by_risk = risk_amount / risk_per_unit
    qty_by_notional_cap = max_notional / entry_price

    return max(min(qty_by_risk, qty_by_notional_cap), 0.0)

# -----------------------------------------------------------------------------
# Fast simulation context
# -----------------------------------------------------------------------------

class _FastCtx(RuleContextMixin):
    """
    Drop-in runtime context for Rule lambdas.

    Compatible with:
      lambda c: c.v("rsi14__RSI")
      lambda c: c.prev_all_below(...)
      lambda c: c.cross_up_pair("close", "MA50")
      lambda c: c["close"] > c["MA50"]

    It caches columns as NumPy arrays to avoid repeated pandas .iloc calls.
    """

    __slots__ = ("df", "i", "_arrays")

    def __init__(self, df: pd.DataFrame, arrays: Dict[str, np.ndarray]):
        self.df = df
        self.i = 0
        self._arrays = arrays

    @property
    def row(self) -> pd.Series:
        return self.df.iloc[self.i]

    def _arr(self, col: str) -> np.ndarray:
        arr = self._arrays.get(col)
        if arr is None:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' not found in simulation df.")
            arr = self.df[col].to_numpy()
            self._arrays[col] = arr
        return arr

    def __getitem__(self, col: str):
        return self.v(col)

    @staticmethod
    def _finite(x) -> bool:
        try:
            return bool(np.isfinite(x))
        except Exception:
            return False

    def v(self, col: str, shift: int = 0):
        j = self.i + shift
        if j < 0 or j >= len(self.df):
            return np.nan
        return self._arr(col)[j]

    def is_finite(self, col: str, shift: int = 0) -> bool:
        return self._finite(self.v(col, shift=shift))


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class Trade:
    trade_id: int
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    open_reason: str

    entry_i: Optional[int] = None
    exit_i: Optional[int] = None

    entry_fee: float = 0.0
    exit_fee: float = 0.0
    gross_pnl: Optional[float] = None
    return_pct: Optional[float] = None
    duration_min: Optional[float] = None
    bars_held: Optional[int] = None

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    close_reason: Optional[str] = None
    pnl: Optional[float] = None

    # Add to your Trade dataclass / object
    qty_initial: float = 0.0
    qty_remaining: float = 0.0
    stop_price: Optional[float] = None
    tp_done: set = field(default_factory=set)
    partial_pnl: float = 0.0
    partial_fees: float = 0.0

@dataclass
class SimResult:
    trades: List[Trade]
    events: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: Dict[str, float]

def align_timeframes(
    base_df: pd.DataFrame,
    other_dfs: Dict[str, pd.DataFrame],
    base_label: str = "1m",
    shift_other_features: bool = True,
    htf_shift_bars: int = 1,
) -> pd.DataFrame:
    """
    Merge other timeframes into base_df using merge_asof on 't'.

    Important:
      By default, features from other_dfs are shifted by 1 candle before alignment.
      This prevents lookahead bias for higher-timeframe indicators.

    Example:
      Base = 1m
      Other = 5m

      At 10:03, merge_asof matches 5m row t=10:00.
      But the 10:00 5m candle is not closed until 10:05.
      Therefore the 5m row t=10:00 must contain shifted feature values
      from the prior closed 5m candle.

    Output:
      - Other timeframe feature columns get prefix '<tf>__'
      - Base timeframe feature columns get prefix '<base_label>__'
      - Base OHLCV columns remain unprefixed: open/high/low/close/volume
    """
    from simulation.timeframe_utils import (
        CORE_OHLCV_COLS,
        feature_columns,
        shift_htf_features_to_closed_candle,
    )

    if "t" not in base_df.columns:
        raise ValueError("base_df must contain 't' column")

    merged = base_df.sort_values("t").copy()

    for tf, d in other_dfs.items():
        d2 = d.sort_values("t").copy()

        # Shift only indicator/feature columns to avoid HTF lookahead.
        # This is automatic and replaces the old manual call.
        if shift_other_features:
            d2 = shift_htf_features_to_closed_candle(
                d2,
                shift_bars=htf_shift_bars,
            )

        # Keep only feature columns + t from other timeframe.
        # Do not merge HTF raw OHLCV columns by default to avoid confusion
        # with base timeframe OHLCV.
        other_feature_cols = feature_columns(d2)
        keep_cols = ["t"] + other_feature_cols
        d2 = d2[keep_cols]

        # Prefix other timeframe feature columns.
        rename = {c: f"{tf}__{c}" for c in d2.columns if c != "t"}
        d2 = d2.rename(columns=rename)

        merged = pd.merge_asof(
            merged.sort_values("t"),
            d2.sort_values("t"),
            on="t",
            direction="backward",
            allow_exact_matches=True,
        )

    # Prefix base timeframe feature columns, but keep base OHLCV unprefixed.
    base_rename = {
        c: f"{base_label}__{c}"
        for c in merged.columns
        if (
            c not in CORE_OHLCV_COLS
            and not c.startswith(f"{base_label}__")
            and not any(c.startswith(f"{tf}__") for tf in other_dfs.keys())
        )
    }

    merged = merged.rename(columns=base_rename)

    return merged

@dataclass
class Strategy:
    open_rules_long: RuleGroup
    close_rules_long: RuleGroup

    allow_short: bool = False
    open_rules_short: Optional[RuleGroup] = None
    close_rules_short: Optional[RuleGroup] = None

@dataclass
class SimConfig:
    initial_cash: float = 10_000.0
    max_open_trades: int = 1
    cash_per_trade: Optional[float] = None
    fee_bps: float = 0.0
    slippage_bps: float = 0.0

    # Simulation window
    sim_start: Optional[str] = None   # e.g. "2026-05-03"
    sim_end: Optional[str] = None     # e.g. "2026-05-06"
    sim_tz: Optional[str] = None      # e.g. "Asia/Karachi"

    # Logging/progress
    log_level: str = "INFO"
    progress: bool = True
    progress_bar: bool = True
    progress_every: Optional[int] = None

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple


Side = Literal["long", "short"]
StopMode = Literal["entry_pct", "price_abs"]
SizingMode = Literal["cash", "risk_pct_equity", "risk_amount"]
TPMode = Literal["entry_pct", "price_abs", "rr"]
MoveStopMode = Literal["none", "breakeven", "entry_pct", "price_abs"]


@dataclass(frozen=True)
class StopLossConfig:
    """
    Stop-loss definition.

    mode="entry_pct":
      value=0.5 means 0.5% away from entry.

    mode="price_abs":
      value=500 means $500 away from entry price.
    """
    mode: StopMode = "entry_pct"
    value: float = 0.5


@dataclass(frozen=True)
class PositionSizingConfig:
    """
    Position sizing.

    mode="cash":
      Uses existing cash_per_trade behavior.

    mode="risk_pct_equity":
      Risk X% of current equity/cash based on stop-loss distance.
      Example: value=1.0 means risk 1% of account.

    mode="risk_amount":
      Risk a fixed amount in account currency.
      Example: value=100 means risk $100.
    """
    mode: SizingMode = "cash"
    value: float = 1.0
    max_notional_pct_of_equity: float = 100.0
    allow_leverage: bool = False


@dataclass(frozen=True)
class TakeProfitConfig:
    """
    Take-profit level.

    mode="entry_pct":
      value=0.5 means TP is 0.5% from entry.

    mode="price_abs":
      value=500 means TP is $500 from entry.

    mode="rr":
      value=2.0 means take profit at 2R.

    close_pct:
      Percentage of current remaining position to close.
      Example: 50 means close 50% of remaining quantity.

    move_stop_mode:
      "breakeven" -> move SL to entry price
      "entry_pct" -> move SL to entry +/- move_stop_value %
      "price_abs" -> move SL to entry +/- move_stop_value dollars
    """
    label: str
    mode: TPMode
    value: float
    close_pct: float = 100.0

    move_stop_mode: MoveStopMode = "none"
    move_stop_value: float = 0.0


@dataclass(frozen=True)
class TradeExitConfig:
    enabled: bool = False

    stop_loss: Optional[StopLossConfig] = None
    sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)

    take_profits: Tuple[TakeProfitConfig, ...] = ()

    # If a candle touches both SL and TP, this controls the assumed fill order.
    # "stop_first" is conservative.
    intrabar_priority: Literal["stop_first", "take_profit_first"] = "stop_first"

    # If True, normal close_rules can still close the remaining position.
    allow_rule_close: bool = True
# -----------------------------------------------------------------------------
# Core simulation helpers
# -----------------------------------------------------------------------------

def _apply_slippage(price: float, side: Side, bps: float, is_entry: bool) -> float:
    m = bps / 10_000.0
    if side == "long":
        return price * (1 + m) if is_entry else price * (1 - m)
    else:
        return price * (1 - m) if is_entry else price * (1 + m)


def _finalize_trade(
    tr: Trade,
    exit_time,
    exit_i: int,
    exit_px: float,
    exit_fee: float,
    gross_pnl: float,
    close_reason: str,
) -> Trade:
    tr.exit_time = exit_time
    tr.exit_i = exit_i
    tr.exit_price = float(exit_px)
    tr.exit_fee = float(exit_fee)
    tr.gross_pnl = float(gross_pnl)

    # Net PnL includes both entry and exit fees.
    tr.pnl = float(gross_pnl - tr.entry_fee - exit_fee)
    tr.close_reason = close_reason

    tr.bars_held = int(exit_i - tr.entry_i) if tr.entry_i is not None else None
    tr.duration_min = _duration_minutes(tr.entry_time, tr.exit_time)

    notional = tr.entry_price * tr.qty
    tr.return_pct = float((tr.pnl / notional) * 100.0) if notional > 0 else None

    return tr


def _build_stats(
    trades: List[Trade],
    equity_curve: pd.DataFrame,
    initial_cash: float,
    final_cash: float,
    total_fees: float,
    bars_with_position: int,
    open_count_sum: int,
    total_bars: int,
) -> Dict[str, float]:
    closed = [tr for tr in trades if tr.pnl is not None]
    n = len(closed)

    if n == 0:
        max_dd = float(equity_curve["drawdown"].max() * 100.0) if not equity_curve.empty else 0.0
        return {
            "initial_cash": float(initial_cash),
            "final_cash": float(final_cash),
            "total_pnl": 0.0,
            "total_return_pct": float((final_cash / initial_cash - 1.0) * 100.0) if initial_cash > 0 else 0.0,
            "num_trades": 0.0,
            "num_winners": 0.0,
            "num_losers": 0.0,
            "num_breakeven": 0.0,
            "win_rate_pct": 0.0,
            "loss_rate_pct": 0.0,
            "max_drawdown_pct": max_dd,
            "profit_factor": 0.0,
            "total_fees": float(total_fees),
            "exposure_bars_pct": float((bars_with_position / total_bars) * 100.0) if total_bars else 0.0,
            "avg_open_trades": float(open_count_sum / total_bars) if total_bars else 0.0,
        }

    pnls = np.array([float(tr.pnl) for tr in closed], dtype=np.float64)
    gross_pnls = np.array([float(tr.gross_pnl or 0.0) for tr in closed], dtype=np.float64)
    returns = np.array([float(tr.return_pct) for tr in closed if tr.return_pct is not None], dtype=np.float64)
    durations = np.array([float(tr.duration_min) for tr in closed if tr.duration_min is not None], dtype=np.float64)
    bars_held = np.array([float(tr.bars_held) for tr in closed if tr.bars_held is not None], dtype=np.float64)

    win_mask = pnls > 0
    loss_mask = pnls < 0
    breakeven_mask = pnls == 0

    winners = pnls[win_mask]
    losers = pnls[loss_mask]

    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(-losers.sum()) if len(losers) else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_win = _safe_mean(winners)
    avg_loss = _safe_mean(losers)
    avg_loss_abs = abs(avg_loss)

    payoff_ratio = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else float("inf") if avg_win > 0 else 0.0
    expectancy = float(pnls.mean())
    expectancy_pct = float(expectancy / initial_cash * 100.0) if initial_cash > 0 else 0.0

    max_dd = float(equity_curve["drawdown"].max() * 100.0) if not equity_curve.empty else 0.0
    total_pnl = float(pnls.sum())

    recovery_factor = (total_pnl / abs(max_dd / 100.0 * initial_cash)) if max_dd > 0 else float("inf") if total_pnl > 0 else 0.0

    sides = np.array([tr.side for tr in closed], dtype=object)
    long_pnls = pnls[sides == "long"]
    short_pnls = pnls[sides == "short"]

    def _side_stats(side_pnls: np.ndarray, prefix: str) -> Dict[str, float]:
        if len(side_pnls) == 0:
            return {
                f"{prefix}_trades": 0.0,
                f"{prefix}_pnl": 0.0,
                f"{prefix}_win_rate_pct": 0.0,
                f"{prefix}_avg_pnl": 0.0,
            }
        return {
            f"{prefix}_trades": float(len(side_pnls)),
            f"{prefix}_pnl": float(side_pnls.sum()),
            f"{prefix}_win_rate_pct": float((side_pnls > 0).mean() * 100.0),
            f"{prefix}_avg_pnl": float(side_pnls.mean()),
        }

    stats = {
        "initial_cash": float(initial_cash),
        "final_cash": float(final_cash),
        "total_pnl": total_pnl,
        "total_return_pct": float((final_cash / initial_cash - 1.0) * 100.0) if initial_cash > 0 else 0.0,

        "num_trades": float(n),
        "num_winners": float(win_mask.sum()),
        "num_losers": float(loss_mask.sum()),
        "num_breakeven": float(breakeven_mask.sum()),
        "win_rate_pct": float(win_mask.mean() * 100.0),
        "loss_rate_pct": float(loss_mask.mean() * 100.0),

        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),

        "avg_pnl": float(pnls.mean()),
        "median_pnl": _safe_median(pnls),
        "avg_winner": float(avg_win),
        "avg_loser": float(avg_loss),             # negative number
        "avg_loser_abs": float(avg_loss_abs),     # positive number
        "largest_winner": float(winners.max()) if len(winners) else 0.0,
        "largest_loser": float(losers.min()) if len(losers) else 0.0,

        "payoff_ratio": float(payoff_ratio),
        "expectancy_per_trade": float(expectancy),
        "expectancy_pct_initial_cash": float(expectancy_pct),

        "avg_return_pct": _safe_mean(returns),
        "median_return_pct": _safe_median(returns),
        "best_return_pct": float(returns.max()) if len(returns) else 0.0,
        "worst_return_pct": float(returns.min()) if len(returns) else 0.0,

        "avg_duration_min": _safe_mean(durations),
        "median_duration_min": _safe_median(durations),
        "avg_bars_held": _safe_mean(bars_held),
        "median_bars_held": _safe_median(bars_held),

        "max_consecutive_wins": float(_max_consecutive(win_mask)),
        "max_consecutive_losses": float(_max_consecutive(loss_mask)),

        "max_drawdown_pct": max_dd,
        "recovery_factor": float(recovery_factor),

        "total_fees": float(total_fees),
        "avg_fee_per_trade": float(total_fees / n) if n else 0.0,

        "exposure_bars_pct": float((bars_with_position / total_bars) * 100.0) if total_bars else 0.0,
        "avg_open_trades": float(open_count_sum / total_bars) if total_bars else 0.0,
    }

    stats.update(_side_stats(long_pnls, "long"))
    stats.update(_side_stats(short_pnls, "short"))

    return stats


# -----------------------------------------------------------------------------
# Main simulation function
# -----------------------------------------------------------------------------

def run_simulation(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: SimConfig,
    time_col: str = "t",
    price_col: str = "close",
) -> SimResult:
    """
    Same external API as before:
        run_simulation(df, strategy, cfg, time_col="t", price_col="close")

    Internally faster:
      - uses cached NumPy arrays for prices and rule columns
      - reuses one context object
      - avoids repeated df.iloc and df[col].iloc inside helper methods

    Adds:
      - optional progress bar/logging via SimConfig
      - more detailed stats
    """
    if df.empty:
        raise ValueError("Empty df")
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in df.")
    if price_col not in df.columns:
        raise KeyError(f"price_col '{price_col}' not found in df.")

    logger = _make_logger()
    logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    t0_perf = time.perf_counter()

    df = df.sort_values(time_col).reset_index(drop=True)
    df = _slice_simulation_window(df, time_col=time_col, cfg=cfg, logger=logger)
    
    n_bars = len(df)
    times = df[time_col].tolist()
    prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=np.float64)

    arrays: Dict[str, np.ndarray] = {
        price_col: prices,
    }
    ctx = _FastCtx(df=df, arrays=arrays)

    cash = float(cfg.initial_cash)
    cash_per_trade = float(cfg.cash_per_trade) if cfg.cash_per_trade is not None else (
        cfg.initial_cash / max(cfg.max_open_trades, 1)
    )

    trades: List[Trade] = []
    open_trades: List[Trade] = []
    events: List[Dict] = []
    equity_rows: List[Dict] = []

    peak_equity = float(cfg.initial_cash)
    max_dd = 0.0

    total_fees = 0.0
    trade_id = 0

    opened_count = 0
    closed_count = 0

    bars_with_position = 0
    open_count_sum = 0

    progress_every = cfg.progress_every or max(1, n_bars // 10)
    pbar = _maybe_tqdm(cfg.progress and cfg.progress_bar, n_bars, "Simulation")

    logger.info(
        f"Simulation started | bars={n_bars:,} | initial_cash={cfg.initial_cash:,.2f} | "
        f"cash_per_trade={cash_per_trade:,.2f} | max_open_trades={cfg.max_open_trades} | "
        f"fee_bps={cfg.fee_bps} | slippage_bps={cfg.slippage_bps}"
    )

    try:
        last_pbar_i = 0

        for i in range(n_bars):
            ctx.i = i

            t = times[i]
            px = float(prices[i])

            if not np.isfinite(px):
                # Skip invalid price bars; still record equity using last cash only.
                equity_rows.append({
                    "t": t,
                    "bar_index": i,
                    "cash": cash,
                    "open_pnl": 0.0,
                    "equity": cash,
                    "drawdown": 0.0,
                    "open_trades": len(open_trades),
                })
                continue

            # -----------------------
            # Mark-to-market equity
            # -----------------------
            open_pnl = 0.0
            for tr in open_trades:
                if tr.side == "long":
                    open_pnl += (px - tr.entry_price) * tr.qty
                else:
                    open_pnl += (tr.entry_price - px) * tr.qty

            equity = cash + open_pnl
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_dd = max(max_dd, dd)

            equity_rows.append({
                "t": t,
                "bar_index": i,
                "cash": cash,
                "open_pnl": open_pnl,
                "equity": equity,
                "drawdown": dd,
                "open_trades": len(open_trades),
            })

            # -----------------------
            # CLOSE logic
            # -----------------------
            still_open: List[Trade] = []
            for tr in open_trades:
                if tr.side == "long":
                    ok, reasons = strategy.close_rules_long.eval(ctx)
                else:
                    if not (strategy.allow_short and strategy.close_rules_short):
                        ok, reasons = False, []
                    else:
                        ok, reasons = strategy.close_rules_short.eval(ctx)

                if ok:
                    exit_px = _apply_slippage(px, tr.side, cfg.slippage_bps, is_entry=False)
                    exit_notional = exit_px * tr.qty
                    exit_fee = exit_notional * (cfg.fee_bps / 10_000.0)
                    total_fees += exit_fee

                    if tr.side == "long":
                        gross_pnl = (exit_px - tr.entry_price) * tr.qty
                    else:
                        gross_pnl = (tr.entry_price - exit_px) * tr.qty

                    # Cash update excludes entry fee because it was already paid at entry.
                    cash += gross_pnl - exit_fee

                    close_reason = ", ".join(reasons) if reasons else "close_rules"
                    tr = _finalize_trade(
                        tr=tr,
                        exit_time=t,
                        exit_i=i,
                        exit_px=exit_px,
                        exit_fee=exit_fee,
                        gross_pnl=gross_pnl,
                        close_reason=close_reason,
                    )

                    closed_count += 1
                    trades.append(tr)

                    events.append({
                        "t": t,
                        "bar_index": i,
                        "event": "CLOSE",
                        "trade_id": tr.trade_id,
                        "side": tr.side,
                        "price": exit_px,
                        "qty": tr.qty,
                        "gross_pnl": tr.gross_pnl,
                        "pnl": tr.pnl,
                        "entry_fee": tr.entry_fee,
                        "exit_fee": tr.exit_fee,
                        "fees": tr.entry_fee + tr.exit_fee,
                        "return_pct": tr.return_pct,
                        "duration_min": tr.duration_min,
                        "bars_held": tr.bars_held,
                        "cash_after": cash,
                        "reason": tr.close_reason,
                    })
                else:
                    still_open.append(tr)

            open_trades = still_open

            # -----------------------
            # OPEN logic
            # -----------------------
            if len(open_trades) < cfg.max_open_trades:
                ok_long, reasons_long = strategy.open_rules_long.eval(ctx)
                opened = False

                if ok_long and cash_per_trade > 0 and cash > 0:
                    trade_id += 1

                    entry_px = _apply_slippage(px, "long", cfg.slippage_bps, is_entry=True)
                    qty = cash_per_trade / entry_px
                    entry_fee = (entry_px * qty) * (cfg.fee_bps / 10_000.0)

                    cash -= entry_fee
                    total_fees += entry_fee

                    tr = Trade(
                        trade_id=trade_id,
                        side="long",
                        entry_time=t,
                        entry_i=i,
                        entry_price=float(entry_px),
                        qty=float(qty),
                        entry_fee=float(entry_fee),
                        open_reason=", ".join(reasons_long) if reasons_long else "open_rules_long",
                    )

                    open_trades.append(tr)
                    opened_count += 1

                    events.append({
                        "t": t,
                        "bar_index": i,
                        "event": "OPEN",
                        "trade_id": tr.trade_id,
                        "side": tr.side,
                        "price": entry_px,
                        "qty": qty,
                        "pnl": None,
                        "entry_fee": entry_fee,
                        "fees": entry_fee,
                        "cash_after": cash,
                        "reason": tr.open_reason,
                    })
                    opened = True

                if (
                    (not opened)
                    and strategy.allow_short
                    and strategy.open_rules_short
                    and len(open_trades) < cfg.max_open_trades
                ):
                    ok_short, reasons_short = strategy.open_rules_short.eval(ctx)

                    if ok_short and cash_per_trade > 0 and cash > 0:
                        trade_id += 1

                        entry_px = _apply_slippage(px, "short", cfg.slippage_bps, is_entry=True)
                        qty = cash_per_trade / entry_px
                        entry_fee = (entry_px * qty) * (cfg.fee_bps / 10_000.0)

                        cash -= entry_fee
                        total_fees += entry_fee

                        tr = Trade(
                            trade_id=trade_id,
                            side="short",
                            entry_time=t,
                            entry_i=i,
                            entry_price=float(entry_px),
                            qty=float(qty),
                            entry_fee=float(entry_fee),
                            open_reason=", ".join(reasons_short) if reasons_short else "open_rules_short",
                        )

                        open_trades.append(tr)
                        opened_count += 1

                        events.append({
                            "t": t,
                            "bar_index": i,
                            "event": "OPEN",
                            "trade_id": tr.trade_id,
                            "side": tr.side,
                            "price": entry_px,
                            "qty": qty,
                            "pnl": None,
                            "entry_fee": entry_fee,
                            "fees": entry_fee,
                            "cash_after": cash,
                            "reason": tr.open_reason,
                        })

            # exposure stats after open/close actions for this bar
            if open_trades:
                bars_with_position += 1
            open_count_sum += len(open_trades)

            # -----------------------
            # Progress
            # -----------------------
            if pbar is not None:
                # Updating every bar can be expensive in notebooks; update in chunks.
                if (i + 1) - last_pbar_i >= progress_every or i == n_bars - 1:
                    pbar.update((i + 1) - last_pbar_i)
                    last_pbar_i = i + 1
                    pbar.set_postfix(
                        opened=opened_count,
                        closed=closed_count,
                        open=len(open_trades),
                        cash=f"{cash:,.0f}",
                    )
            elif cfg.progress and ((i + 1) % progress_every == 0 or i == n_bars - 1):
                elapsed = time.perf_counter() - t0_perf
                speed = (i + 1) / elapsed if elapsed > 0 else 0.0
                pct = (i + 1) / n_bars * 100.0
                logger.info(
                    f"Progress {pct:5.1f}% | bars={i+1:,}/{n_bars:,} | "
                    f"opened={opened_count:,} | closed={closed_count:,} | "
                    f"currently_open={len(open_trades)} | cash={cash:,.2f} | "
                    f"speed={speed:,.0f} bars/s | elapsed={elapsed:.1f}s"
                )

    finally:
        if pbar is not None:
            pbar.close()

    # -----------------------
    # Force-close remaining positions on last valid bar
    # -----------------------
    last_i = n_bars - 1
    last_t = times[last_i]
    last_px = float(prices[last_i])

    if open_trades:
        logger.info(f"Force-closing {len(open_trades)} open trade(s) at final bar.")

    for tr in open_trades:
        exit_px = _apply_slippage(last_px, tr.side, cfg.slippage_bps, is_entry=False)
        exit_notional = exit_px * tr.qty
        exit_fee = exit_notional * (cfg.fee_bps / 10_000.0)
        total_fees += exit_fee

        if tr.side == "long":
            gross_pnl = (exit_px - tr.entry_price) * tr.qty
        else:
            gross_pnl = (tr.entry_price - exit_px) * tr.qty

        cash += gross_pnl - exit_fee

        tr = _finalize_trade(
            tr=tr,
            exit_time=last_t,
            exit_i=last_i,
            exit_px=exit_px,
            exit_fee=exit_fee,
            gross_pnl=gross_pnl,
            close_reason="forced_close_end",
        )

        closed_count += 1
        trades.append(tr)

        events.append({
            "t": last_t,
            "bar_index": last_i,
            "event": "CLOSE",
            "trade_id": tr.trade_id,
            "side": tr.side,
            "price": exit_px,
            "qty": tr.qty,
            "gross_pnl": tr.gross_pnl,
            "pnl": tr.pnl,
            "entry_fee": tr.entry_fee,
            "exit_fee": tr.exit_fee,
            "fees": tr.entry_fee + tr.exit_fee,
            "return_pct": tr.return_pct,
            "duration_min": tr.duration_min,
            "bars_held": tr.bars_held,
            "cash_after": cash,
            "reason": tr.close_reason,
        })

    ev = pd.DataFrame(events)
    eq = pd.DataFrame(equity_rows)

    stats = _build_stats(
        trades=trades,
        equity_curve=eq,
        initial_cash=float(cfg.initial_cash),
        final_cash=float(cash),
        total_fees=float(total_fees),
        bars_with_position=bars_with_position,
        open_count_sum=open_count_sum,
        total_bars=n_bars,
    )

    elapsed = time.perf_counter() - t0_perf
    logger.info(
        f"Simulation finished | elapsed={elapsed:.2f}s | bars={n_bars:,} | "
        f"opened={opened_count:,} | closed={closed_count:,} | "
        f"final_cash={cash:,.2f} | total_return={stats.get('total_return_pct', 0.0):.2f}% | "
        f"max_dd={stats.get('max_drawdown_pct', 0.0):.2f}%"
    )

    return SimResult(trades=trades, events=ev, equity_curve=eq, stats=stats)

def _is_date_only(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value.strip()) is not None


def _coerce_sim_bound(value, series_tz, sim_tz: Optional[str]):
    """
    Converts sim_start/sim_end to a timestamp comparable with df[time_col].
    """
    if value is None:
        return None

    ts = pd.to_datetime(value)

    target_tz = sim_tz or series_tz

    # If bound is naive and target timezone exists, localize it.
    if ts.tzinfo is None and target_tz is not None:
        ts = ts.tz_localize(target_tz)

    # If bound is timezone-aware and target timezone exists, convert it.
    elif ts.tzinfo is not None and target_tz is not None:
        ts = ts.tz_convert(target_tz)

    # If dataframe time column is naive, make bound naive too.
    if series_tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts


def _slice_simulation_window(
    df: pd.DataFrame,
    time_col: str,
    cfg: "SimConfig",
    logger,
) -> pd.DataFrame:
    """
    Applies optional simulation date window.

    Behavior:
      - sim_start is inclusive.
      - sim_end is inclusive.
      - If sim_end is date-only, e.g. "2026-05-06", it includes the full day.
    """
    if cfg.sim_start is None and cfg.sim_end is None:
        return df

    out = df.copy()

    times = pd.to_datetime(out[time_col])
    series_tz = times.dt.tz

    start_ts = _coerce_sim_bound(cfg.sim_start, series_tz, cfg.sim_tz)
    end_ts = _coerce_sim_bound(cfg.sim_end, series_tz, cfg.sim_tz)

    before = len(out)

    if start_ts is not None:
        out = out[times >= start_ts]

    if end_ts is not None:
        # If user passes "2026-05-06", include all of May 6.
        if _is_date_only(cfg.sim_end):
            end_exclusive = end_ts + pd.Timedelta(days=1)
            out = out[pd.to_datetime(out[time_col]) < end_exclusive]
        else:
            out = out[pd.to_datetime(out[time_col]) <= end_ts]

    out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError(
            f"No rows left after simulation date filtering. "
            f"sim_start={cfg.sim_start}, sim_end={cfg.sim_end}, sim_tz={cfg.sim_tz}"
        )

    logger.info(
        f"Simulation window applied | rows={len(out):,}/{before:,} | "
        f"range={out[time_col].min()} -> {out[time_col].max()}"
    )

    return out