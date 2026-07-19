"""
Numba-JIT position/PnL kernel — the hot core of the backtester.

Consumes numpy arrays (OHLC + precomputed entry/exit signal bools) and walks the bars
once in compiled machine code, producing trade records + an equity curve. Designed to
reproduce the legacy `simulation.simulator.run_simulation` semantics exactly for the
supported feature subset, so results can be validated numerically (see tests/).

Per-bar order of operations (matches legacy):
  1. mark-to-market equity is recorded (pre-exit),
  2. risk exits: stop-loss / take-profit (intrabar priority configurable),
  3. rule-based close (exit signals),
  4. open a new position (long first, then short) if a slot is free.

PnL model (matches legacy): margin/CFD-style. Cash reflects fees + realized PnL only
(entry notional is NOT deducted); equity = cash + unrealized PnL.

Supported subset (v1): SL entry_pct/price_abs; single TP entry_pct/price_abs/rr closing
100%; sizing cash / risk_pct_equity; fees, slippage, multi-position, force-close at end.
Partial/laddered TPs, trailing/stop-movement, and ref_col structure stops are planned
follow-ups (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba should be installed
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn
        return _wrap(args[0]) if args and callable(args[0]) else _wrap


# Close-reason codes (kept in sync with run.py REASON_NAMES).
R_SIGNAL = 0
R_STOP = 1
R_TAKE_PROFIT = 2
R_FORCED = 3


@njit(cache=True)
def equity_stats(equity):
    """Single compiled pass over the equity curve -> drawdown + return moments.

    Returns (max_dd, sum_r, sum_r2, sum_dr, sum_dr2, n_r, n_dr) where r are per-bar
    simple returns and dr are the negative (downside) subset. Lets callers compute
    Sharpe/Sortino/max-drawdown without multiple numpy passes over ~500k bars.
    """
    n = equity.shape[0]
    peak = -np.inf
    max_dd = 0.0
    sum_r = 0.0
    sum_r2 = 0.0
    sum_dr = 0.0
    sum_dr2 = 0.0
    n_r = 0
    n_dr = 0
    prev = np.nan
    for i in range(n):
        e = equity[i]
        if e != e:  # NaN
            continue
        if e > peak:
            peak = e
        if peak > 0.0:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        if prev == prev and prev != 0.0:
            r = (e - prev) / prev
            sum_r += r
            sum_r2 += r * r
            n_r += 1
            if r < 0.0:
                sum_dr += r
                sum_dr2 += r * r
                n_dr += 1
        prev = e
    return max_dd, sum_r, sum_r2, sum_dr, sum_dr2, n_r, n_dr


@njit(cache=True, fastmath=False)
def run_kernel(
    open_, high, low, close,          # float64[:]
    entry_long, exit_long,            # uint8[:]
    entry_short, exit_short,          # uint8[:]
    initial_cash, cash_per_trade,
    fee_bps, slippage_bps,
    max_open_trades, allow_short,
    exit_enabled,
    sl_mode, sl_value,                # 0 none, 1 entry_pct, 2 price_abs
    tp_mode, tp_value,                # 0 none, 1 entry_pct, 2 price_abs, 3 rr
    sizing_mode, sizing_value,        # 0 cash, 1 risk_pct_equity
    max_notional_pct, allow_leverage,
    allow_rule_close, intrabar_stop_first,
):
    n = close.shape[0]
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

    # --- trade records (preallocated; at most 1 open per bar => <= n trades) ---
    t_side = np.zeros(n, np.int8)
    t_entry_i = np.zeros(n, np.int64)
    t_exit_i = np.full(n, -1, np.int64)
    t_entry_px = np.zeros(n, np.float64)
    t_exit_px = np.zeros(n, np.float64)
    t_qty = np.zeros(n, np.float64)
    t_gross = np.zeros(n, np.float64)
    t_entry_fee = np.zeros(n, np.float64)
    t_exit_fee = np.zeros(n, np.float64)
    t_pnl = np.zeros(n, np.float64)
    t_reason = np.zeros(n, np.int8)
    n_tr = 0

    # --- open positions (order-preserving compaction) ---
    o_tr = np.zeros(max_open_trades, np.int64)
    o_side = np.zeros(max_open_trades, np.int8)
    o_entry_px = np.zeros(max_open_trades, np.float64)
    o_qty = np.zeros(max_open_trades, np.float64)
    o_stop = np.zeros(max_open_trades, np.float64)
    o_tp = np.zeros(max_open_trades, np.float64)
    o_cnt = 0

    equity_curve = np.empty(n, np.float64)
    pos_count = np.zeros(n, np.int64)

    cash = initial_cash

    for i in range(n):
        px = close[i]
        hi = high[i]
        lo = low[i]

        if np.isnan(px):
            equity_curve[i] = cash
            pos_count[i] = o_cnt
            continue
        if np.isnan(hi):
            hi = px
        if np.isnan(lo):
            lo = px

        # 1) mark-to-market equity (pre-exit)
        open_pnl = 0.0
        for j in range(o_cnt):
            open_pnl += o_side[j] * (px - o_entry_px[j]) * o_qty[j]
        equity = cash + open_pnl
        equity_curve[i] = equity

        # 2) risk exits (SL / TP), order-preserving compaction
        w = 0
        for j in range(o_cnt):
            side = o_side[j]
            stop = o_stop[j]
            tp = o_tp[j]

            sl_hit = False
            if not np.isnan(stop):
                sl_hit = (lo <= stop) if side == 1 else (hi >= stop)
            tp_hit = False
            if not np.isnan(tp):
                tp_hit = (hi >= tp) if side == 1 else (lo <= tp)

            do_close = False
            level = 0.0
            reason = R_SIGNAL
            if sl_hit and tp_hit:
                if intrabar_stop_first == 1:
                    do_close = True; level = stop; reason = R_STOP
                else:
                    do_close = True; level = tp; reason = R_TAKE_PROFIT
            elif sl_hit:
                do_close = True; level = stop; reason = R_STOP
            elif tp_hit:
                do_close = True; level = tp; reason = R_TAKE_PROFIT

            if do_close:
                exit_px = level * (1.0 - side * slip)
                qty = o_qty[j]
                gross = side * (exit_px - o_entry_px[j]) * qty
                exit_fee = exit_px * qty * fee
                cash += gross - exit_fee
                k = o_tr[j]
                t_exit_i[k] = i
                t_exit_px[k] = exit_px
                t_gross[k] = gross
                t_exit_fee[k] = exit_fee
                t_pnl[k] = gross - t_entry_fee[k] - exit_fee
                t_reason[k] = reason
            else:
                o_tr[w] = o_tr[j]; o_side[w] = o_side[j]
                o_entry_px[w] = o_entry_px[j]; o_qty[w] = o_qty[j]
                o_stop[w] = o_stop[j]; o_tp[w] = o_tp[j]
                w += 1
        o_cnt = w

        # 3) rule-based close (exit signals)
        if (exit_enabled == 0) or (allow_rule_close == 1):
            w = 0
            for j in range(o_cnt):
                side = o_side[j]
                sig = exit_long[i] if side == 1 else exit_short[i]
                if sig == 1:
                    exit_px = px * (1.0 - side * slip)
                    qty = o_qty[j]
                    gross = side * (exit_px - o_entry_px[j]) * qty
                    exit_fee = exit_px * qty * fee
                    cash += gross - exit_fee
                    k = o_tr[j]
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_gross[k] = gross
                    t_exit_fee[k] = exit_fee
                    t_pnl[k] = gross - t_entry_fee[k] - exit_fee
                    t_reason[k] = R_SIGNAL
                else:
                    o_tr[w] = o_tr[j]; o_side[w] = o_side[j]
                    o_entry_px[w] = o_entry_px[j]; o_qty[w] = o_qty[j]
                    o_stop[w] = o_stop[j]; o_tp[w] = o_tp[j]
                    w += 1
            o_cnt = w

        # 4) open a new position (long first, then short)
        if o_cnt < max_open_trades and cash > 0.0 and cash_per_trade > 0.0:
            want_long = entry_long[i] == 1
            want_short = (allow_short == 1) and (entry_short[i] == 1)
            side = 0
            if want_long:
                side = 1
            elif want_short:
                side = -1

            if side != 0:
                entry_px = px * (1.0 + side * slip)

                # stop
                stop = np.nan
                if exit_enabled == 1 and sl_mode != 0:
                    if sl_mode == 1:
                        stop = entry_px * (1.0 - side * (sl_value / 100.0))
                    elif sl_mode == 2:
                        stop = entry_px - side * sl_value

                # size
                qty = 0.0
                if (exit_enabled == 0) or (sizing_mode == 0):
                    notional = cash_per_trade
                    if exit_enabled == 1:
                        max_notional = equity * (max_notional_pct / 100.0)
                        if allow_leverage == 0 and max_notional > cash:
                            max_notional = cash
                        if notional > max_notional:
                            notional = max_notional
                    qty = notional / entry_px
                elif sizing_mode == 1:
                    if not np.isnan(stop):
                        rpu = abs(entry_px - stop)
                        if rpu > 0.0:
                            risk_amount = equity * (sizing_value / 100.0)
                            qty = risk_amount / rpu
                            max_notional = equity * (max_notional_pct / 100.0)
                            if allow_leverage == 0 and max_notional > cash:
                                max_notional = cash
                            cap = max_notional / entry_px
                            if qty > cap:
                                qty = cap

                if qty > 0.0:
                    entry_fee = entry_px * qty * fee
                    if cash > 0.0 and entry_fee <= cash:
                        # tp (needs stop for rr)
                        tp = np.nan
                        if exit_enabled == 1 and tp_mode != 0:
                            if tp_mode == 1:
                                tp = entry_px * (1.0 + side * (tp_value / 100.0))
                            elif tp_mode == 2:
                                tp = entry_px + side * tp_value
                            elif tp_mode == 3 and not np.isnan(stop):
                                risk = abs(entry_px - stop)
                                tp = entry_px + side * risk * tp_value

                        cash -= entry_fee
                        k = n_tr
                        t_side[k] = side
                        t_entry_i[k] = i
                        t_entry_px[k] = entry_px
                        t_qty[k] = qty
                        t_entry_fee[k] = entry_fee
                        n_tr += 1

                        o_tr[o_cnt] = k; o_side[o_cnt] = side
                        o_entry_px[o_cnt] = entry_px; o_qty[o_cnt] = qty
                        o_stop[o_cnt] = stop; o_tp[o_cnt] = tp
                        o_cnt += 1

        pos_count[i] = o_cnt

    # force-close remaining at last valid bar
    last_i = n - 1
    last_px = close[last_i]
    if np.isnan(last_px):
        for j in range(last_i, -1, -1):
            if not np.isnan(close[j]):
                last_i = j
                last_px = close[j]
                break

    for j in range(o_cnt):
        side = o_side[j]
        exit_px = last_px * (1.0 - side * slip)
        qty = o_qty[j]
        gross = side * (exit_px - o_entry_px[j]) * qty
        exit_fee = exit_px * qty * fee
        cash += gross - exit_fee
        k = o_tr[j]
        t_exit_i[k] = last_i
        t_exit_px[k] = exit_px
        t_gross[k] = gross
        t_exit_fee[k] = exit_fee
        t_pnl[k] = gross - t_entry_fee[k] - exit_fee
        t_reason[k] = R_FORCED

    return (
        t_side[:n_tr], t_entry_i[:n_tr], t_exit_i[:n_tr],
        t_entry_px[:n_tr], t_exit_px[:n_tr], t_qty[:n_tr],
        t_gross[:n_tr], t_entry_fee[:n_tr], t_exit_fee[:n_tr],
        t_pnl[:n_tr], t_reason[:n_tr],
        equity_curve, pos_count, cash,
    )
