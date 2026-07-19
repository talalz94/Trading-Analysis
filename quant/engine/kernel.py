"""
Numba-JIT position/PnL kernel — the hot core of the backtester.

Consumes numpy arrays (OHLC + precomputed entry/exit signals + exit config) and walks the
bars once in compiled machine code, producing trade records + an equity curve. Reproduces
the legacy `simulation.simulator` semantics for the supported feature set.

Per-bar order of operations (matches legacy):
  1. mark-to-market equity is recorded (pre-exit),
  2. risk exits: trailing update -> stop-loss / take-profits (intrabar priority configurable),
  3. rule-based close (exit signals),
  4. open a new position (long first, then short) if a slot is free.

Exit model:
  - Stop loss: entry_pct | price_abs | ref_col (structure level w/ buffer, max-risk cap, fallback).
  - Trailing stop: pct | price_abs (ratchets on the high-water mark).
  - Take profits: up to MAX_TP laddered levels, each closing `close_pct` of the remaining
    position, with optional post-TP stop movement (breakeven | entry_pct | price_abs).
  - Sizing: cash | risk_pct_equity | risk_amount. Fees (bps), slippage (bps), multi-position.

PnL model (matches legacy): margin/CFD-style. Cash reflects fees + realized PnL only
(entry notional is NOT deducted); equity = cash + unrealized PnL.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
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
R_MARGIN = 4

_TINY = 1e-12


@njit(cache=True)
def equity_stats(equity):
    """Single compiled pass over the equity curve -> drawdown + return moments.

    Returns (max_dd, sum_r, sum_r2, sum_dr, sum_dr2, n_r, n_dr) where r are per-bar simple
    returns and dr are the negative (downside) subset. Lets callers compute Sharpe/Sortino/
    max-drawdown without multiple numpy passes over ~500k bars.
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
        if e != e:
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


@njit(cache=True)
def _fixed_stop(entry_px, side, mode, value):
    # mode: 1 entry_pct, 2 price_abs
    if mode == 1:
        pct = value / 100.0
        return entry_px * (1.0 - side * pct)
    return entry_px - side * value


@njit(cache=True)
def _calc_stop(entry_px, side, sl_mode, sl_value, ref, buffer_pct, max_ref_risk_pct,
               fb_mode, fb_value):
    if sl_mode == 0:
        return np.nan
    if sl_mode == 1 or sl_mode == 2:
        return _fixed_stop(entry_px, side, sl_mode, sl_value)
    # ref_col
    if not (ref == ref) or ref <= 0.0:
        return _fixed_stop(entry_px, side, fb_mode, fb_value)
    buf = buffer_pct / 100.0
    if side == 1:
        stop = ref * (1.0 - buf)
        if stop >= entry_px:
            return _fixed_stop(entry_px, side, fb_mode, fb_value)
        risk = (entry_px - stop) / entry_px * 100.0
    else:
        stop = ref * (1.0 + buf)
        if stop <= entry_px:
            return _fixed_stop(entry_px, side, fb_mode, fb_value)
        risk = (stop - entry_px) / entry_px * 100.0
    if max_ref_risk_pct > 0.0 and risk > max_ref_risk_pct:
        return _fixed_stop(entry_px, side, fb_mode, fb_value)
    return stop


@njit(cache=True)
def _calc_tp(entry_px, stop, side, mode, value):
    # mode: 1 entry_pct, 2 price_abs, 3 rr
    if mode == 1:
        return entry_px * (1.0 + side * value / 100.0)
    if mode == 2:
        return entry_px + side * value
    if mode == 3:
        if not (stop == stop):  # rr needs a stop
            return np.nan
        risk = abs(entry_px - stop)
        return entry_px + side * risk * value
    return np.nan


@njit(cache=True)
def run_kernel(
    open_, high, low, close,
    entry_long, exit_long, entry_short, exit_short,
    sl_ref_long, sl_ref_short,
    tp_modes, tp_values, tp_close_pcts, tp_move_modes, tp_move_values, n_tp,
    initial_cash, cash_per_trade, fee_bps, slippage_bps, max_open_trades, allow_short,
    exit_enabled,
    sl_mode, sl_value, sl_buffer_pct, sl_max_ref_risk_pct, sl_fallback_mode, sl_fallback_value,
    trail_mode, trail_value,
    sizing_mode, sizing_value, max_notional_pct, allow_leverage,
    margin_enabled, leverage, contract_size, stop_out_level,
    allow_rule_close, intrabar_stop_first,
):
    n = close.shape[0]
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

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

    mo = max_open_trades
    o_tr = np.zeros(mo, np.int64)
    o_side = np.zeros(mo, np.int8)
    o_entry_px = np.zeros(mo, np.float64)
    o_qty0 = np.zeros(mo, np.float64)
    o_qty = np.zeros(mo, np.float64)
    o_stop = np.zeros(mo, np.float64)
    o_extreme = np.zeros(mo, np.float64)
    o_tp_done = np.zeros(mo, np.int64)
    o_tp_px = np.zeros((mo, tp_modes.shape[0]), np.float64)
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

        # 1) mark-to-market (pre-exit) + used margin
        open_pnl = 0.0
        used_margin = 0.0
        for j in range(o_cnt):
            open_pnl += o_side[j] * (px - o_entry_px[j]) * o_qty[j]
            if margin_enabled == 1:
                used_margin += (o_entry_px[j] * o_qty[j]) / leverage
        equity = cash + open_pnl
        equity_curve[i] = equity

        # 1.5) margin stop-out: broker liquidates open positions when equity falls to the
        # stop-out fraction of used margin (Exness-style). Conservative: liquidate all at market.
        if margin_enabled == 1 and o_cnt > 0 and used_margin > 0.0 and \
                equity <= (stop_out_level / 100.0) * used_margin:
            for j in range(o_cnt):
                side = o_side[j]
                exit_px = px * (1.0 - side * slip)
                q = o_qty[j]
                gross = side * (exit_px - o_entry_px[j]) * q
                ef = exit_px * q * fee
                cash += gross - ef
                k = o_tr[j]
                t_gross[k] += gross
                t_exit_fee[k] += ef
                t_exit_i[k] = i
                t_exit_px[k] = exit_px
                t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                t_reason[k] = R_MARGIN
            o_cnt = 0
            pos_count[i] = 0
            continue

        # 2) risk exits
        if exit_enabled == 1:
            w = 0
            for j in range(o_cnt):
                side = o_side[j]
                k = o_tr[j]

                # trailing update (ratchet)
                if trail_mode != 0:
                    if side == 1:
                        if hi > o_extreme[j]:
                            o_extreme[j] = hi
                        cand = o_extreme[j] * (1.0 - trail_value / 100.0) if trail_mode == 1 else o_extreme[j] - trail_value
                        if np.isnan(o_stop[j]) or cand > o_stop[j]:
                            o_stop[j] = cand
                    else:
                        if lo < o_extreme[j]:
                            o_extreme[j] = lo
                        cand = o_extreme[j] * (1.0 + trail_value / 100.0) if trail_mode == 1 else o_extreme[j] + trail_value
                        if np.isnan(o_stop[j]) or cand < o_stop[j]:
                            o_stop[j] = cand

                stop = o_stop[j]
                sl_hit = (not np.isnan(stop)) and ((lo <= stop) if side == 1 else (hi >= stop))

                tp_any = False
                for kk in range(n_tp):
                    if (o_tp_done[j] >> kk) & 1:
                        continue
                    tpx = o_tp_px[j, kk]
                    if np.isnan(tpx):
                        continue
                    if (hi >= tpx) if side == 1 else (lo <= tpx):
                        tp_any = True
                        break

                closed = False

                if sl_hit and ((not tp_any) or intrabar_stop_first == 1):
                    exit_px = stop * (1.0 - side * slip)
                    q = o_qty[j]
                    gross = side * (exit_px - o_entry_px[j]) * q
                    ef = exit_px * q * fee
                    cash += gross - ef
                    t_gross[k] += gross
                    t_exit_fee[k] += ef
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                    t_reason[k] = R_STOP
                    closed = True
                else:
                    for kk in range(n_tp):
                        if o_qty[j] <= _TINY:
                            break
                        if (o_tp_done[j] >> kk) & 1:
                            continue
                        tpx = o_tp_px[j, kk]
                        if np.isnan(tpx):
                            continue
                        hit = (hi >= tpx) if side == 1 else (lo <= tpx)
                        if not hit:
                            continue
                        exit_px = tpx * (1.0 - side * slip)
                        cf = tp_close_pcts[kk] / 100.0
                        if cf > 1.0:
                            cf = 1.0
                        if cf < 0.0:
                            cf = 0.0
                        qc = o_qty[j] * cf
                        o_tp_done[j] |= (1 << kk)
                        if qc <= _TINY:
                            continue
                        gross = side * (exit_px - o_entry_px[j]) * qc
                        ef = exit_px * qc * fee
                        cash += gross - ef
                        t_gross[k] += gross
                        t_exit_fee[k] += ef
                        o_qty[j] -= qc

                        mvmode = tp_move_modes[kk]
                        if mvmode != 0:
                            if mvmode == 1:
                                newstop = o_entry_px[j]
                            elif mvmode == 2:
                                newstop = o_entry_px[j] * (1.0 + side * tp_move_values[kk] / 100.0)
                            else:
                                newstop = o_entry_px[j] + side * tp_move_values[kk]
                            if np.isnan(o_stop[j]):
                                o_stop[j] = newstop
                            elif side == 1:
                                if newstop > o_stop[j]:
                                    o_stop[j] = newstop
                            else:
                                if newstop < o_stop[j]:
                                    o_stop[j] = newstop

                        if o_qty[j] <= _TINY:
                            t_exit_i[k] = i
                            t_exit_px[k] = exit_px
                            t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                            t_reason[k] = R_TAKE_PROFIT
                            closed = True
                            break

                    # take_profit_first: re-check stop after TP/stop-move
                    if (not closed) and sl_hit and intrabar_stop_first == 0:
                        stop2 = o_stop[j]
                        still = (not np.isnan(stop2)) and ((lo <= stop2) if side == 1 else (hi >= stop2))
                        if still:
                            exit_px = stop2 * (1.0 - side * slip)
                            q = o_qty[j]
                            gross = side * (exit_px - o_entry_px[j]) * q
                            ef = exit_px * q * fee
                            cash += gross - ef
                            t_gross[k] += gross
                            t_exit_fee[k] += ef
                            t_exit_i[k] = i
                            t_exit_px[k] = exit_px
                            t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                            t_reason[k] = R_STOP
                            closed = True

                if not closed:
                    o_tr[w] = o_tr[j]
                    o_side[w] = o_side[j]
                    o_entry_px[w] = o_entry_px[j]
                    o_qty0[w] = o_qty0[j]
                    o_qty[w] = o_qty[j]
                    o_stop[w] = o_stop[j]
                    o_extreme[w] = o_extreme[j]
                    o_tp_done[w] = o_tp_done[j]
                    o_tp_px[w, :] = o_tp_px[j, :]
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
                    q = o_qty[j]
                    gross = side * (exit_px - o_entry_px[j]) * q
                    ef = exit_px * q * fee
                    cash += gross - ef
                    k = o_tr[j]
                    t_gross[k] += gross
                    t_exit_fee[k] += ef
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                    t_reason[k] = R_SIGNAL
                else:
                    o_tr[w] = o_tr[j]
                    o_side[w] = o_side[j]
                    o_entry_px[w] = o_entry_px[j]
                    o_qty0[w] = o_qty0[j]
                    o_qty[w] = o_qty[j]
                    o_stop[w] = o_stop[j]
                    o_extreme[w] = o_extreme[j]
                    o_tp_done[w] = o_tp_done[j]
                    o_tp_px[w, :] = o_tp_px[j, :]
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

                stop = np.nan
                if exit_enabled == 1 and sl_mode != 0:
                    ref = sl_ref_long[i] if side == 1 else sl_ref_short[i]
                    stop = _calc_stop(entry_px, side, sl_mode, sl_value, ref,
                                      sl_buffer_pct, sl_max_ref_risk_pct,
                                      sl_fallback_mode, sl_fallback_value)

                qty = 0.0
                if sizing_mode == 3:
                    # lots: qty = lots * contract_size (e.g. gold 1 lot = 100 oz)
                    qty = sizing_value * contract_size
                elif (exit_enabled == 0) or (sizing_mode == 0):
                    notional = cash_per_trade
                    if exit_enabled == 1 and margin_enabled == 0:
                        max_notional = equity * (max_notional_pct / 100.0)
                        if allow_leverage == 0 and max_notional > cash:
                            max_notional = cash
                        if notional > max_notional:
                            notional = max_notional
                    qty = notional / entry_px
                else:
                    if not np.isnan(stop):
                        rpu = abs(entry_px - stop)
                        if rpu > 0.0:
                            if sizing_mode == 1:
                                risk_amount = equity * (sizing_value / 100.0)
                            else:
                                risk_amount = sizing_value
                            qty = risk_amount / rpu
                            if margin_enabled == 0:
                                max_notional = equity * (max_notional_pct / 100.0)
                                if allow_leverage == 0 and max_notional > cash:
                                    max_notional = cash
                                cap = max_notional / entry_px
                                if qty > cap:
                                    qty = cap

                if qty > 0.0:
                    entry_fee = entry_px * qty * fee
                    can_open = cash > 0.0 and entry_fee <= cash
                    if can_open and margin_enabled == 1:
                        # affordability check: free margin must cover this position's margin
                        cur_open_pnl = 0.0
                        cur_used = 0.0
                        for jj in range(o_cnt):
                            cur_open_pnl += o_side[jj] * (px - o_entry_px[jj]) * o_qty[jj]
                            cur_used += (o_entry_px[jj] * o_qty[jj]) / leverage
                        free_margin = (cash + cur_open_pnl) - cur_used
                        if (entry_px * qty) / leverage > free_margin:
                            can_open = False
                    if can_open:
                        cash -= entry_fee
                        k = n_tr
                        t_side[k] = side
                        t_entry_i[k] = i
                        t_entry_px[k] = entry_px
                        t_qty[k] = qty
                        t_entry_fee[k] = entry_fee
                        n_tr += 1

                        o_tr[o_cnt] = k
                        o_side[o_cnt] = side
                        o_entry_px[o_cnt] = entry_px
                        o_qty0[o_cnt] = qty
                        o_qty[o_cnt] = qty
                        o_stop[o_cnt] = stop
                        o_extreme[o_cnt] = entry_px
                        o_tp_done[o_cnt] = 0
                        if exit_enabled == 1:
                            for kk in range(n_tp):
                                o_tp_px[o_cnt, kk] = _calc_tp(entry_px, stop, side,
                                                             tp_modes[kk], tp_values[kk])
                        else:
                            for kk in range(o_tp_px.shape[1]):
                                o_tp_px[o_cnt, kk] = np.nan
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
        q = o_qty[j]
        gross = side * (exit_px - o_entry_px[j]) * q
        ef = exit_px * q * fee
        cash += gross - ef
        k = o_tr[j]
        t_gross[k] += gross
        t_exit_fee[k] += ef
        t_exit_i[k] = last_i
        t_exit_px[k] = exit_px
        t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
        t_reason[k] = R_FORCED

    return (
        t_side[:n_tr], t_entry_i[:n_tr], t_exit_i[:n_tr],
        t_entry_px[:n_tr], t_exit_px[:n_tr], t_qty[:n_tr],
        t_gross[:n_tr], t_entry_fee[:n_tr], t_exit_fee[:n_tr],
        t_pnl[:n_tr], t_reason[:n_tr],
        equity_curve, pos_count, cash,
    )
