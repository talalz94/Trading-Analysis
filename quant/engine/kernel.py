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
R_ALT_EXIT = 5
R_TIME = 6
R_TP_MOVED = 7     # closed on a target that had been MOVED CLOSER (tp_move_*)
R_REF_TARGET = 8   # closed on the per-trade level read from a column at entry (tp_ref_*)

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
def _fill_px(base, side, slip, spread, is_entry):
    # buy fills at ask (+half-spread), sell fills at bid (-half-spread), plus slippage.
    # `spread` is the fixed bid/ask width in price units; the monetary cost scales with qty
    # (= lots x contract_size) automatically, since the half-spread is charged per unit.
    if is_entry == 1:
        return base * (1.0 + side * slip) + side * spread * 0.5
    return base * (1.0 - side * slip) - side * spread * 0.5


@njit(cache=True)
def _costs(fill_px, q, fee, comm_per_lot, contract_size):
    # commission = %% of notional (fee) + fixed per-lot (per side).
    return fill_px * q * fee + comm_per_lot * (q / contract_size)


@njit(cache=True)
def run_kernel(
    open_, high, low, close,
    entry_long, exit_long, entry_short, exit_short,
    entry_leg_long, entry_leg_short, exit_legs_long, exit_legs_short, n_legs,
    per_leg_stops, leg_sl_mode, leg_sl_value, leg_sl_buffer, leg_sl_maxrisk,
    leg_sl_fb_mode, leg_sl_fb_value, leg_sl_ref_long, leg_sl_ref_short,
    sl_ref_long, sl_ref_short,
    tp_ref_long, tp_ref_short,
    exit_arm_long, exit_arm_short, exit_alt_long, exit_alt_short, alt_exit_enabled, alt_replace_tp,
    alt_arm_on_tp, arm_touch_n, arm_wrong_m, alt_min_pct, session_id,
    tp_modes, tp_values, tp_close_pcts, tp_move_modes, tp_move_values, n_tp,
    initial_cash, cash_per_trade, fee_bps, slippage_bps, max_open_trades, allow_short,
    exit_enabled,
    sl_mode, sl_value, sl_buffer_pct, sl_max_ref_risk_pct, sl_fallback_mode, sl_fallback_value,
    trail_mode, trail_value,
    sizing_mode, sizing_value, max_notional_pct, allow_leverage,
    margin_enabled, leverage, contract_size, stop_out_level,
    spread, commission_per_lot,
    allow_rule_close, intrabar_stop_first, daily_loss_limit, max_bars_in_trade,
    tp_ref_on=0, tp_move_to=0.0, tp_move_near=0.0, tp_move_back=0.0, tp_move_after=0,
):
    # Per-rule ("leg") exits: when n_legs > 1 each open trade remembers which rule opened it
    # (o_leg) and closes on THAT rule's exit column (exit_legs_*[leg, i]). When n_legs == 1 the
    # code takes the original exit_long/exit_short path below — byte-identical to before.
    #
    # Secondary ("alt") exit, alt_exit_enabled == 1 only: each open trade also remembers whether it
    # has been ARMED (o_armed) by `exit_arm_*` firing at some bar SINCE its entry. Once armed, the
    # trade closes on `exit_alt_*` — but only while it is IN PROFIT (measured at the exit fill, so
    # spread/slippage count). This is per-trade, path-dependent state that a global signal array
    # cannot express. With alt_exit_enabled == 0 the branch is skipped entirely — byte-identical.
    # `alt_replace_tp` additionally drops the take-profit ladder once a trade is armed (the secondary
    # rule then owns the exit); the stop-loss and trailing stop are never affected.
    # `alt_min_pct` is how far the price must have moved your way, as a percent of entry, before the
    # secondary may close a trade. 0 keeps the original "any favourable move" gate, which on a 5m
    # chart routinely exits for less than the round trip costs.
    # `arm_touch_n` / `arm_wrong_m` arm a trade from its OWN behaviour rather than from a signal:
    # how many bars since entry have traded back through its entry price, and how many candles have
    # formed WHOLLY on the losing side of it. Neither can be a global array — the level is the
    # trade's own entry — which is why they are counted here. Both 0 skips the block entirely.
    # `alt_arm_on_tp` arms a trade when it REACHES a take-profit level instead of closing there, so a
    # winner can keep running and is handed to the secondary rule (it implies the TP suppression).
    #
    # `max_bars_in_trade` > 0 is a TIME STOP: any trade still open that many bars after its entry is
    # closed at that bar's close (reason `time_stop`). It is checked after the stop/TP and signal
    # exits, so a trade that would have closed for a real reason on the same bar keeps that reason.
    # At 0 the whole block is skipped - byte-identical to before.
    #
    # `tp_ref_on` == 1 gives each trade a SECOND TARGET read from `tp_ref_long/short` AT ITS ENTRY
    # BAR -- a level that differs per trade and so cannot be a ladder rung (those are R multiples).
    # The first of the two that price reaches closes the whole position (reason `ref_target`). A
    # level on the wrong side of the entry, or NaN, means no second target for that trade.
    #
    # `tp_move_to` > 0 MOVES THE TARGET CLOSER, once per trade, to that fraction of its original
    # distance. Two triggers, either enough: a NEAR MISS (the best price since entry reached
    # `tp_move_near` of the way to the target AND the close has since given back `tp_move_back` of
    # that distance) or a LONG WAIT (`tp_move_after` bars held). Evaluated on the bar's CLOSE, after
    # every exit, and applied from the NEXT bar -- intrabar order is unknown, so a bar that both
    # reached and gave back cannot also fill the new target. If price already sits beyond the new
    # target, the trade closes at that close. Either way the reason is `tp_moved`, and a target that
    # later fills after being moved is `tp_moved` too, so the rule's trades can be told apart.
    # Rungs that close nothing (breakeven moves) are left where they are. 0 skips it all.
    #
    # `daily_loss_limit` > 0 stops NEW entries for the rest of a session once that session's realised
    # loss reaches the limit (a fraction of the equity the session opened with). `session_id` labels
    # each bar's session, so this works for calendar days or any custom grouping. Open trades are
    # never force-closed - their stops still run - because a broker-style daily stop-out and a
    # "stop opening new risk" rule are different things and only the latter is modelled here.
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
    t_stop = np.full(n, np.nan)      # initial stop price per trade (for charts / risk display)
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
    o_leg = np.zeros(mo, np.int64)      # which rule (leg) opened each open trade
    o_armed = np.zeros(mo, np.int8)     # secondary exit armed for this trade (alt_exit_enabled)
    o_touch = np.zeros(mo, np.int64)    # bars since entry that traded back through the entry price
    o_wrong = np.zeros(mo, np.int64)    # candles formed wholly on the losing side of the entry
    o_best = np.zeros(mo, np.float64)   # best price since entry (tp_move_*)
    o_moved = np.zeros(mo, np.int8)     # target already moved closer (tp_move_*)
    o_ref = np.full(mo, np.nan)         # this trade's second target (tp_ref_*)
    o_cnt = 0
    t_move_i = np.full(n, -1, np.int64)   # bar the target was moved on, -1 = never
    t_move_px = np.full(n, np.nan)        # where it was moved to
    t_ref = np.full(n, np.nan)            # the second target the trade carried

    equity_curve = np.empty(n, np.float64)
    pos_count = np.zeros(n, np.int64)
    cash = initial_cash

    # session loss cap state
    cur_session = session_id[0] if n > 0 else 0
    session_start_equity = initial_cash
    session_realised = 0.0
    session_blocked = False

    for i in range(n):
        px = close[i]
        hi = high[i]
        lo = low[i]

        if daily_loss_limit > 0.0:
            sid = session_id[i]
            if sid != cur_session:                 # new session -> reset the budget
                cur_session = sid
                session_start_equity = cash
                session_realised = 0.0
                session_blocked = False
            cash_at_bar_start = cash
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
                q = o_qty[j]
                exit_px = _fill_px(px, side, slip, spread, 0)
                gross = side * (exit_px - o_entry_px[j]) * q
                ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
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

        # 1.75) secondary-exit ARMING. Done before the risk exits so that the take-profit gate
        # (alt_replace_tp) and the alt exit in step 3 both see the same armed state on the same bar.
        if alt_exit_enabled == 1:
            # SELF-ARMING from the trade's own path. Counted before the signal check so a trade can
            # arm on the same bar either way. This runs before entries open, so a trade never counts
            # its own entry bar.
            if arm_touch_n > 0 or arm_wrong_m > 0:
                for j in range(o_cnt):
                    if o_armed[j] == 1:
                        continue
                    epx = o_entry_px[j]
                    if lo <= epx <= hi:
                        o_touch[j] += 1
                    elif (hi < epx) if o_side[j] == 1 else (lo > epx):
                        o_wrong[j] += 1
                    if arm_touch_n > 0 and o_touch[j] >= arm_touch_n:
                        o_armed[j] = 1
                    elif arm_wrong_m > 0 and o_wrong[j] >= arm_wrong_m:
                        o_armed[j] = 1
            for j in range(o_cnt):
                if o_armed[j] == 0:
                    if (exit_arm_long[i] if o_side[j] == 1 else exit_arm_short[i]) == 1:
                        o_armed[j] = 1
                    elif alt_arm_on_tp == 1 and exit_enabled == 1:
                        # "let the winner run": reaching a take-profit level ARMS the trade instead
                        # of closing it, so the position is handed to the secondary rule (e.g. exit
                        # on N opposite candles) and keeps running while candles stay favourable.
                        sd = o_side[j]
                        for kk in range(n_tp):
                            if (o_tp_done[j] >> kk) & 1:
                                continue
                            tpx = o_tp_px[j, kk]
                            if np.isnan(tpx):
                                continue
                            if (hi >= tpx) if sd == 1 else (lo <= tpx):
                                o_armed[j] = 1
                                break

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

                # once ARMED, `sec_replace_tp` drops the original take-profit ladder for this trade —
                # the secondary rule owns the exit from here. The stop/trailing are untouched.
                n_tp_eff = n_tp
                if alt_exit_enabled == 1 and o_armed[j] == 1 and                         (alt_replace_tp == 1 or alt_arm_on_tp == 1):
                    n_tp_eff = 0

                tp_any = False
                near_tp = np.inf         # distance to the nearest ladder level this bar reaches
                for kk in range(n_tp_eff):
                    if (o_tp_done[j] >> kk) & 1:
                        continue
                    tpx = o_tp_px[j, kk]
                    if np.isnan(tpx):
                        continue
                    if (hi >= tpx) if side == 1 else (lo <= tpx):
                        tp_any = True
                        if tp_ref_on == 0:
                            break
                        if abs(tpx - o_entry_px[j]) < near_tp:
                            near_tp = abs(tpx - o_entry_px[j])

                # the SECOND TARGET. When a bar reaches it and a ladder level together, whichever
                # is NEARER the entry was reached first -- price cannot get to the far one without
                # passing the near one -- so the ladder only goes first if it is closer.
                ref_hit = False
                ref_first = False
                if tp_ref_on == 1:
                    rx = o_ref[j]
                    if rx == rx and ((hi >= rx) if side == 1 else (lo <= rx)):
                        ref_hit = True
                        tp_any = True
                        ref_first = abs(rx - o_entry_px[j]) <= near_tp

                closed = False

                if ref_first and not (sl_hit and intrabar_stop_first == 1):
                    q = o_qty[j]
                    exit_px = _fill_px(o_ref[j], side, slip, spread, 0)
                    gross = side * (exit_px - o_entry_px[j]) * q
                    ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
                    cash += gross - ef
                    t_gross[k] += gross
                    t_exit_fee[k] += ef
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                    t_reason[k] = R_REF_TARGET
                    closed = True
                elif sl_hit and ((not tp_any) or intrabar_stop_first == 1):
                    q = o_qty[j]
                    exit_px = _fill_px(stop, side, slip, spread, 0)
                    gross = side * (exit_px - o_entry_px[j]) * q
                    ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
                    cash += gross - ef
                    t_gross[k] += gross
                    t_exit_fee[k] += ef
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                    t_reason[k] = R_STOP
                    closed = True
                else:
                    for kk in range(n_tp_eff):
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
                        cf = tp_close_pcts[kk] / 100.0
                        if cf > 1.0:
                            cf = 1.0
                        if cf < 0.0:
                            cf = 0.0
                        qc = o_qty[j] * cf
                        o_tp_done[j] |= (1 << kk)
                        exit_px = _fill_px(tpx, side, slip, spread, 0)
                        if qc > _TINY:
                            gross = side * (exit_px - o_entry_px[j]) * qc
                            ef = _costs(exit_px, qc, fee, commission_per_lot, contract_size)
                            cash += gross - ef
                            t_gross[k] += gross
                            t_exit_fee[k] += ef
                            o_qty[j] -= qc

                        # A level that closes NOTHING (close_pct = 0) may still MOVE THE STOP. That
                        # is how a plain "go to breakeven once 1R is reached" rule is expressed, with
                        # no partial taken. Previously `continue` skipped this block, so such a level
                        # was silently inert and the config ran byte-identically to having no level
                        # at all. Levels that do close something are unaffected.
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

                        if qc > _TINY and o_qty[j] <= _TINY:
                            t_exit_i[k] = i
                            t_exit_px[k] = exit_px
                            t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                            t_reason[k] = R_TP_MOVED if o_moved[j] == 1 else R_TAKE_PROFIT
                            closed = True
                            break

                    # the second target, when the ladder was nearer but did not close everything
                    if (not closed) and ref_hit and o_qty[j] > _TINY:
                        q = o_qty[j]
                        exit_px = _fill_px(o_ref[j], side, slip, spread, 0)
                        gross = side * (exit_px - o_entry_px[j]) * q
                        ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
                        cash += gross - ef
                        t_gross[k] += gross
                        t_exit_fee[k] += ef
                        t_exit_i[k] = i
                        t_exit_px[k] = exit_px
                        t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                        t_reason[k] = R_REF_TARGET
                        closed = True

                    # take_profit_first: re-check stop after TP/stop-move
                    if (not closed) and sl_hit and intrabar_stop_first == 0:
                        stop2 = o_stop[j]
                        still = (not np.isnan(stop2)) and ((lo <= stop2) if side == 1 else (hi >= stop2))
                        if still:
                            q = o_qty[j]
                            exit_px = _fill_px(stop2, side, slip, spread, 0)
                            gross = side * (exit_px - o_entry_px[j]) * q
                            ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
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
                    o_leg[w] = o_leg[j]
                    o_armed[w] = o_armed[j]
                    o_touch[w] = o_touch[j]
                    o_wrong[w] = o_wrong[j]
                    o_best[w] = o_best[j]
                    o_moved[w] = o_moved[j]
                    o_ref[w] = o_ref[j]
                    w += 1
            o_cnt = w

        # 3) rule-based close (exit signals)
        if (exit_enabled == 0) or (allow_rule_close == 1):
            w = 0
            for j in range(o_cnt):
                side = o_side[j]
                if n_legs == 1:
                    sig = exit_long[i] if side == 1 else exit_short[i]
                else:
                    sig = exit_legs_long[o_leg[j], i] if side == 1 else exit_legs_short[o_leg[j], i]
                # secondary exit: arm on `exit_arm_*` (since THIS trade opened), then close on
                # `exit_alt_*` while in profit. Skipped wholly when the feature is off.
                alt_fired = 0
                if alt_exit_enabled == 1 and sig == 0 and o_armed[j] == 1:
                    alt = exit_alt_long[i] if side == 1 else exit_alt_short[i]
                    if alt == 1:
                        alt_px = _fill_px(px, side, slip, spread, 0)
                        # "in profit" has to mean AFTER COSTS, or it does not mean what it says.
                        # Measured on BTC 5m: three of ten bail-outs had a favourable price move and
                        # still lost money -- one moved 0.92 in price against 15.39 of fees.
                        # `alt_min_pct` is the move, as a percent of entry, the exit must clear.
                        need = o_entry_px[j] * alt_min_pct * 0.01
                        if side * (alt_px - o_entry_px[j]) > need:
                            alt_fired = 1
                if sig == 1 or alt_fired == 1:
                    q = o_qty[j]
                    exit_px = _fill_px(px, side, slip, spread, 0)
                    gross = side * (exit_px - o_entry_px[j]) * q
                    ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
                    cash += gross - ef
                    k = o_tr[j]
                    t_gross[k] += gross
                    t_exit_fee[k] += ef
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                    if alt_fired == 1:
                        t_reason[k] = R_ALT_EXIT
                    else:
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
                    o_leg[w] = o_leg[j]
                    o_armed[w] = o_armed[j]
                    o_touch[w] = o_touch[j]
                    o_wrong[w] = o_wrong[j]
                    o_best[w] = o_best[j]
                    o_moved[w] = o_moved[j]
                    o_ref[w] = o_ref[j]
                    w += 1
            o_cnt = w

        # 3b) time stop: close anything held longer than max_bars_in_trade
        if max_bars_in_trade > 0:
            w = 0
            for j in range(o_cnt):
                k = o_tr[j]
                if i - t_entry_i[k] >= max_bars_in_trade:
                    side = o_side[j]
                    q = o_qty[j]
                    exit_px = _fill_px(px, side, slip, spread, 0)
                    gross = side * (exit_px - o_entry_px[j]) * q
                    ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
                    cash += gross - ef
                    t_gross[k] += gross
                    t_exit_fee[k] += ef
                    t_exit_i[k] = i
                    t_exit_px[k] = exit_px
                    t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                    t_reason[k] = R_TIME
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
                    o_leg[w] = o_leg[j]
                    o_armed[w] = o_armed[j]
                    o_touch[w] = o_touch[j]
                    o_wrong[w] = o_wrong[j]
                    o_best[w] = o_best[j]
                    o_moved[w] = o_moved[j]
                    o_ref[w] = o_ref[j]
                    w += 1
            o_cnt = w

        # 3c) MOVE THE TARGET CLOSER -- on the close, for trades opened on an earlier bar
        if tp_move_to > 0.0:
            w = 0
            for j in range(o_cnt):
                side = o_side[j]
                k = o_tr[j]
                epx = o_entry_px[j]
                if (hi > o_best[j]) if side == 1 else (lo < o_best[j]):
                    o_best[j] = hi if side == 1 else lo
                shut = False
                if o_moved[j] == 0:
                    dist = 0.0                  # the FARTHEST level that still closes something
                    for kk in range(n_tp):
                        if ((o_tp_done[j] >> kk) & 1) or tp_close_pcts[kk] <= 0.0:
                            continue
                        tpx = o_tp_px[j, kk]
                        if tpx == tpx and abs(tpx - epx) > dist:
                            dist = abs(tpx - epx)
                    if dist > 0.0:
                        best = side * (o_best[j] - epx) / dist
                        now = side * (px - epx) / dist
                        go = False
                        if tp_move_near > 0.0 and best >= tp_move_near and \
                                best - now >= tp_move_back:
                            go = True
                        if tp_move_after > 0 and i - t_entry_i[k] >= tp_move_after:
                            go = True
                        if go:
                            o_moved[j] = 1
                            t_move_i[k] = i
                            for kk in range(n_tp):
                                if ((o_tp_done[j] >> kk) & 1) or tp_close_pcts[kk] <= 0.0:
                                    continue
                                tpx = o_tp_px[j, kk]
                                if tpx == tpx:
                                    o_tp_px[j, kk] = epx + side * tp_move_to * abs(tpx - epx)
                                    t_move_px[k] = o_tp_px[j, kk]
                            if now >= tp_move_to:   # already past the new target: take it now
                                q = o_qty[j]
                                exit_px = _fill_px(px, side, slip, spread, 0)
                                gross = side * (exit_px - epx) * q
                                ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
                                cash += gross - ef
                                t_gross[k] += gross
                                t_exit_fee[k] += ef
                                t_exit_i[k] = i
                                t_exit_px[k] = exit_px
                                t_pnl[k] = t_gross[k] - t_entry_fee[k] - t_exit_fee[k]
                                t_reason[k] = R_TP_MOVED
                                shut = True
                if not shut:
                    o_tr[w] = o_tr[j]
                    o_side[w] = o_side[j]
                    o_entry_px[w] = o_entry_px[j]
                    o_qty0[w] = o_qty0[j]
                    o_qty[w] = o_qty[j]
                    o_stop[w] = o_stop[j]
                    o_extreme[w] = o_extreme[j]
                    o_tp_done[w] = o_tp_done[j]
                    o_tp_px[w, :] = o_tp_px[j, :]
                    o_leg[w] = o_leg[j]
                    o_armed[w] = o_armed[j]
                    o_touch[w] = o_touch[j]
                    o_wrong[w] = o_wrong[j]
                    o_best[w] = o_best[j]
                    o_moved[w] = o_moved[j]
                    o_ref[w] = o_ref[j]
                    w += 1
            o_cnt = w

        # 4) open a new position (long first, then short)
        if daily_loss_limit > 0.0:
            session_realised += cash - cash_at_bar_start        # realised P&L booked this bar
            if session_start_equity > 0.0 and \
                    session_realised <= -(daily_loss_limit / 100.0) * session_start_equity:
                session_blocked = True

        if o_cnt < max_open_trades and cash > 0.0 and cash_per_trade > 0.0 and \
                not (daily_loss_limit > 0.0 and session_blocked):
            want_long = entry_long[i] == 1
            want_short = (allow_short == 1) and (entry_short[i] == 1)
            side = 0
            if want_long:
                side = 1
            elif want_short:
                side = -1

            if side != 0:
                entry_px = _fill_px(px, side, slip, spread, 1)   # buy at ask (+half-spread)
                lg = entry_leg_long[i] if side == 1 else entry_leg_short[i]

                stop = np.nan
                if exit_enabled == 1:
                    if per_leg_stops == 1:
                        # this rule's own stop method (per-leg arrays indexed by the opening leg)
                        smode = leg_sl_mode[lg]
                        if smode != 0:
                            ref = leg_sl_ref_long[lg, i] if side == 1 else leg_sl_ref_short[lg, i]
                            stop = _calc_stop(entry_px, side, smode, leg_sl_value[lg], ref,
                                              leg_sl_buffer[lg], leg_sl_maxrisk[lg],
                                              leg_sl_fb_mode[lg], leg_sl_fb_value[lg])
                    elif sl_mode != 0:
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
                    entry_fee = _costs(entry_px, qty, fee, commission_per_lot, contract_size)
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
                        t_stop[k] = stop
                        n_tr += 1

                        o_tr[o_cnt] = k
                        o_side[o_cnt] = side
                        o_entry_px[o_cnt] = entry_px
                        o_qty0[o_cnt] = qty
                        o_qty[o_cnt] = qty
                        o_stop[o_cnt] = stop
                        o_extreme[o_cnt] = entry_px
                        o_tp_done[o_cnt] = 0
                        o_leg[o_cnt] = lg
                        o_armed[o_cnt] = 0
                        o_touch[o_cnt] = 0
                        o_wrong[o_cnt] = 0
                        o_best[o_cnt] = entry_px
                        o_moved[o_cnt] = 0
                        o_ref[o_cnt] = np.nan
                        if tp_ref_on == 1:
                            rx = tp_ref_long[i] if side == 1 else tp_ref_short[i]
                            # only a level on the PROFIT side is a target
                            if rx == rx and side * (rx - entry_px) > 0.0:
                                o_ref[o_cnt] = rx
                                t_ref[k] = rx
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
        q = o_qty[j]
        exit_px = _fill_px(last_px, side, slip, spread, 0)
        gross = side * (exit_px - o_entry_px[j]) * q
        ef = _costs(exit_px, q, fee, commission_per_lot, contract_size)
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
        equity_curve, pos_count, cash, t_stop[:n_tr],
        t_move_i[:n_tr], t_move_px[:n_tr], t_ref[:n_tr],
    )
