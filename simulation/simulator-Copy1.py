from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import pandas as pd

from simulation.rules import RuleGroup


Side = Literal["long", "short"]


@dataclass
class Trade:
    trade_id: int
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    open_reason: str

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    close_reason: Optional[str] = None
    pnl: Optional[float] = None


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
) -> pd.DataFrame:
    """
    Merge other timeframes into base_df using merge_asof on 't'.
    Columns of other dfs should already be feature-rich.
    This function prefixes other timeframe columns with '<tf>__'.
    """
    if "t" not in base_df.columns:
        raise ValueError("base_df must contain 't' column")

    merged = base_df.sort_values("t").copy()

    for tf, d in other_dfs.items():
        d2 = d.sort_values("t").copy()

        # drop raw OHLCV duplicates to avoid confusion (keep only feature columns + t)
        keep_cols = ["t"] + [c for c in d2.columns if c not in {"open_time","open","high","low","close","volume","quote_volume","num_trades","taker_buy_base","taker_buy_quote"} and c != "t"]
        d2 = d2[keep_cols]

        # prefix
        rename = {c: f"{tf}__{c}" for c in d2.columns if c != "t"}
        d2 = d2.rename(columns=rename)

        merged = pd.merge_asof(
            merged.sort_values("t"),
            d2.sort_values("t"),
            on="t",
            direction="backward",
            allow_exact_matches=True,
        )

    # prefix base timeframe feature columns too (except core OHLCV + t)
    core = {"open_time","open","high","low","close","volume","quote_volume","num_trades","taker_buy_base","taker_buy_quote","t"}
    base_rename = {c: f"{base_label}__{c}" for c in merged.columns if c not in core and not c.startswith(f"{base_label}__")}
    merged = merged.rename(columns=base_rename)

    return merged


@dataclass
class Strategy:
    """
    open_rules / close_rules are RuleGroups evaluated on each bar.

    If allow_short=True, open_rules_short/close_rules_short can be provided.
    """
    open_rules_long: RuleGroup
    close_rules_long: RuleGroup

    allow_short: bool = False
    open_rules_short: Optional[RuleGroup] = None
    close_rules_short: Optional[RuleGroup] = None


@dataclass
class SimConfig:
    initial_cash: float = 10_000.0
    max_open_trades: int = 1
    cash_per_trade: Optional[float] = None  # if None, auto = initial_cash / max_open_trades
    fee_bps: float = 0.0
    slippage_bps: float = 0.0


def _apply_slippage(price: float, side: Side, bps: float, is_entry: bool) -> float:
    """
    long entry: pay up; long exit: receive less
    short entry: receive less (sell lower); short exit: pay up (buy higher)
    """
    m = bps / 10_000.0
    if side == "long":
        return price * (1 + m) if is_entry else price * (1 - m)
    else:
        return price * (1 - m) if is_entry else price * (1 + m)


def run_simulation(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: SimConfig,
    time_col: str = "t",
    price_col: str = "close",
) -> SimResult:
    if df.empty:
        raise ValueError("Empty df")

    df = df.sort_values(time_col).reset_index(drop=True)

    cash = float(cfg.initial_cash)
    cash_per_trade = float(cfg.cash_per_trade) if cfg.cash_per_trade is not None else (cfg.initial_cash / max(cfg.max_open_trades, 1))

    trades: List[Trade] = []
    open_trades: List[Trade] = []
    events: List[Dict] = []

    equity_rows: List[Dict] = []
    peak_equity = cfg.initial_cash
    max_dd = 0.0

    trade_id = 0

    for i in range(len(df)):
        row = df.iloc[i]
        t = row[time_col]
        px = float(row[price_col])

        # Mark-to-market equity
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

        equity_rows.append({"t": t, "cash": cash, "open_pnl": open_pnl, "equity": equity, "drawdown": dd})

        # ---- CLOSE logic (evaluate per open trade)
        still_open: List[Trade] = []
        for tr in open_trades:
            if tr.side == "long":
                ok, reasons = strategy.close_rules_long.eval(row)
            else:
                if not (strategy.allow_short and strategy.close_rules_short):
                    ok, reasons = False, []
                else:
                    ok, reasons = strategy.close_rules_short.eval(row)

            if ok:
                exit_px = _apply_slippage(px, tr.side, cfg.slippage_bps, is_entry=False)
                notional = exit_px * tr.qty
                fee = notional * (cfg.fee_bps / 10_000.0)

                if tr.side == "long":
                    pnl = (exit_px - tr.entry_price) * tr.qty - fee
                else:
                    pnl = (tr.entry_price - exit_px) * tr.qty - fee

                cash += pnl

                tr.exit_time = t
                tr.exit_price = exit_px
                tr.pnl = pnl
                tr.close_reason = ", ".join(reasons) if reasons else "close_rules"

                events.append({
                    "t": t, "event": "CLOSE", "trade_id": tr.trade_id, "side": tr.side,
                    "price": exit_px, "qty": tr.qty, "pnl": pnl, "reason": tr.close_reason
                })
                trades.append(tr)
            else:
                still_open.append(tr)

        open_trades = still_open

        # ---- OPEN logic (only if slots available)
        if len(open_trades) < cfg.max_open_trades:
            ok_long, reasons_long = strategy.open_rules_long.eval(row)
            opened = False

            if ok_long and cash_per_trade > 0 and cash > 0:
                trade_id += 1
                entry_px = _apply_slippage(px, "long", cfg.slippage_bps, is_entry=True)
                qty = cash_per_trade / entry_px
                fee = (entry_px * qty) * (cfg.fee_bps / 10_000.0)
                cash -= fee  # pay fee on entry

                tr = Trade(
                    trade_id=trade_id,
                    side="long",
                    entry_time=t,
                    entry_price=entry_px,
                    qty=qty,
                    open_reason=", ".join(reasons_long) if reasons_long else "open_rules_long",
                )
                open_trades.append(tr)
                events.append({
                    "t": t, "event": "OPEN", "trade_id": tr.trade_id, "side": tr.side,
                    "price": entry_px, "qty": qty, "pnl": None, "reason": tr.open_reason
                })
                opened = True

            # Optional shorts
            if (not opened) and strategy.allow_short and strategy.open_rules_short and len(open_trades) < cfg.max_open_trades:
                ok_short, reasons_short = strategy.open_rules_short.eval(row)
                if ok_short and cash_per_trade > 0 and cash > 0:
                    trade_id += 1
                    entry_px = _apply_slippage(px, "short", cfg.slippage_bps, is_entry=True)
                    qty = cash_per_trade / entry_px
                    fee = (entry_px * qty) * (cfg.fee_bps / 10_000.0)
                    cash -= fee

                    tr = Trade(
                        trade_id=trade_id,
                        side="short",
                        entry_time=t,
                        entry_price=entry_px,
                        qty=qty,
                        open_reason=", ".join(reasons_short) if reasons_short else "open_rules_short",
                    )
                    open_trades.append(tr)
                    events.append({
                        "t": t, "event": "OPEN", "trade_id": tr.trade_id, "side": tr.side,
                        "price": entry_px, "qty": qty, "pnl": None, "reason": tr.open_reason
                    })

    # Force-close remaining positions on last bar
    last = df.iloc[-1]
    t = last[time_col]
    px = float(last[price_col])

    for tr in open_trades:
        exit_px = _apply_slippage(px, tr.side, cfg.slippage_bps, is_entry=False)
        notional = exit_px * tr.qty
        fee = notional * (cfg.fee_bps / 10_000.0)

        pnl = (exit_px - tr.entry_price) * tr.qty - fee if tr.side == "long" else (tr.entry_price - exit_px) * tr.qty - fee
        cash += pnl

        tr.exit_time = t
        tr.exit_price = exit_px
        tr.pnl = pnl
        tr.close_reason = "forced_close_end"

        events.append({
            "t": t, "event": "CLOSE", "trade_id": tr.trade_id, "side": tr.side,
            "price": exit_px, "qty": tr.qty, "pnl": pnl, "reason": tr.close_reason
        })
        trades.append(tr)

    ev = pd.DataFrame(events)
    eq = pd.DataFrame(equity_rows)

    # Stats
    closed = [tr for tr in trades if tr.pnl is not None]
    n = len(closed)
    wins = sum(1 for tr in closed if tr.pnl and tr.pnl > 0)
    losses = n - wins
    total_pnl = sum(tr.pnl for tr in closed if tr.pnl is not None)
    final_equity = float(cash)

    gross_profit = sum(tr.pnl for tr in closed if tr.pnl and tr.pnl > 0)
    gross_loss = -sum(tr.pnl for tr in closed if tr.pnl and tr.pnl < 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    stats = {
        "initial_cash": float(cfg.initial_cash),
        "final_cash": float(cash),
        "total_pnl": float(total_pnl),
        "total_return_pct": float((final_equity / cfg.initial_cash - 1) * 100.0) if cfg.initial_cash > 0 else 0.0,
        "num_trades": float(n),
        "win_rate_pct": float((wins / n) * 100.0) if n > 0 else 0.0,
        "max_drawdown_pct": float(max_dd * 100.0),
        "profit_factor": float(profit_factor),
    }

    return SimResult(trades=trades, events=ev, equity_curve=eq, stats=stats)