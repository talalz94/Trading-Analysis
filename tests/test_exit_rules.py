"""
The two optional exit rules -- MOVE THE TARGET after a near miss or a long wait, and a SECOND TARGET
per trade read at entry (the FVG exit) -- plus the plumbing that serves them.

Hand-built bars throughout, so every assertion names the exact bar and price it expects. Each rule
is tested for the thing that would make it quietly wrong: acting on the bar that triggered it
(intrabar order is unknown), re-reading a level after entry, beating the stop on a bar that hits
both, moving a breakeven rung, and -- above all -- changing anything when it is OFF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.engine import BacktestConfig, Signals, TakeProfit
from quant.engine.run import run_backtest


def _df(rows, ref_short=None, ref_long=None):
    """rows = [(open, high, low, close), ...] on a 5m grid."""
    o, h, lo, c = (np.array(x, float) for x in zip(*rows))
    t = pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC")
    d = pd.DataFrame({"open_time": t, "t": t, "open": o, "high": h, "low": lo, "close": c,
                      "volume": 1.0})
    if ref_short is not None:
        d["ref_s"] = ref_short
    if ref_long is not None:
        d["ref_l"] = ref_long
    return d


def _short_at(n, i=0):
    e = np.zeros(n, bool)
    e[i] = True
    return Signals(entry_long=np.zeros(n, bool), entry_short=e)


def _long_at(n, i=0):
    e = np.zeros(n, bool)
    e[i] = True
    return Signals(entry_long=e, entry_short=np.zeros(n, bool))


# a short at 100, stop 101 (1R), target 3R = 97
BASE = dict(initial_cash=10_000, exit_enabled=True, allow_short=True, sl_mode="price_abs",
            sl_value=1.0, tp_mode="rr", tp_value=3.0)

NEAR_MISS = [(100, 100, 100, 100),        # 0 entry
             (100, 100.2, 97.2, 97.5),    # 1 best 97.2 = 0.93 of the way; gave back 0.1 -- not yet
             (97.5, 98.8, 97.4, 98.7),    # 2 close 98.7: gave back 0.5 of the way -> MOVE (to 98.2)
                                          #   this bar's low 97.4 is already past 98.2 -- it must
                                          #   NOT fill here, the order inside the bar is unknown
             (98.7, 98.9, 98.1, 98.3),    # 3 low 98.1 reaches the moved target 98.2
             (98.3, 101.5, 98.2, 101.2)]  # 4 without the rule, the stop at 101 is hit here


# ---------------------------------------------------------------- move the target
def test_a_near_miss_moves_the_target_and_it_fills_on_a_LATER_bar():
    d = _df(NEAR_MISS)
    off = run_backtest(d, _short_at(len(d)), BacktestConfig(**BASE)).trades
    on = run_backtest(d, _short_at(len(d)), BacktestConfig(
        **BASE, tp_move_to=0.6, tp_move_near=0.9, tp_move_back=0.4)).trades
    assert off.close_reason.iloc[0] == "stop_loss" and off.exit_i.iloc[0] == 4
    r = on.iloc[0]
    assert r.close_reason == "tp_moved"
    assert r.exit_i == 3, "filled on the bar that triggered the move -- intrabar order is unknown"
    assert r.exit_price == pytest.approx(98.2)
    assert r.target_moved_i == 2 and r.target_moved_to == pytest.approx(98.2)


def test_not_enough_giveback_means_no_move():
    d = _df(NEAR_MISS)
    on = run_backtest(d, _short_at(len(d)), BacktestConfig(
        **BASE, tp_move_to=0.6, tp_move_near=0.9, tp_move_back=0.6)).trades
    assert on.close_reason.iloc[0] == "stop_loss" and on.target_moved_i.iloc[0] == -1


def test_already_past_the_new_target_closes_at_that_close():
    d = _df([(100, 100, 100, 100), (100, 100.1, 99.0, 99.0), (99.0, 99.2, 98.0, 98.2),
             (98.2, 101.5, 98.1, 101.2)])
    on = run_backtest(d, _short_at(len(d)), BacktestConfig(
        **BASE, tp_move_to=0.5, tp_move_after=2)).trades.iloc[0]
    # held 2 bars at bar 2's close; the new target 98.5 is already behind a close of 98.2
    assert on.close_reason == "tp_moved" and on.exit_i == 2
    assert on.exit_price == pytest.approx(98.2)


def test_a_breakeven_rung_is_left_where_it_is():
    """A rung that closes nothing is a stop rule, not a target. Moved to 0.5R it would put the stop
    at entry on bar 2 and bar 3 would stop the trade out; left alone it does neither."""
    d = _df([(100, 100, 100, 100), (100, 100.1, 99.7, 99.8), (99.8, 99.9, 99.4, 99.6),
             (99.6, 100.5, 99.5, 100.2), (100.2, 100.3, 98.4, 98.6)])
    cfg = BacktestConfig(**{**BASE, "tp_mode": "none"}, tp_move_to=0.5, tp_move_after=1,
                         take_profits=(TakeProfit("rr", 1.0, close_pct=0,
                                                  move_stop_mode="breakeven"),
                                       TakeProfit("rr", 3.0, close_pct=100)))
    r = run_backtest(d, _short_at(len(d)), cfg).trades.iloc[0]
    assert r.exit_i == 4 and r.close_reason == "tp_moved", r
    assert r.exit_price == pytest.approx(98.5)


def test_a_long_moves_its_target_the_mirror_way():
    rows = [(100, 100, 100, 100), (100, 102.8, 99.8, 102.5), (102.5, 102.6, 101.2, 101.3),
            (101.3, 101.9, 101.1, 101.7), (101.7, 101.8, 98.5, 98.8)]
    d = _df(rows)
    cfg = BacktestConfig(**BASE, tp_move_to=0.6, tp_move_near=0.9, tp_move_back=0.4)
    r = run_backtest(d, _long_at(len(d)), cfg).trades.iloc[0]
    assert r.close_reason == "tp_moved" and r.exit_i == 3
    assert r.exit_price == pytest.approx(101.8)


# ---------------------------------------------------------------- the second target
def test_the_second_target_closes_the_trade_before_the_ladder():
    d = _df([(100, 100, 100, 100), (100, 100.1, 97.9, 98.3), (98.3, 98.4, 96.5, 96.8)],
            ref_short=[98.0, np.nan, np.nan])
    r = run_backtest(d, _short_at(len(d)), BacktestConfig(
        **BASE, tp_ref_short_col="ref_s")).trades.iloc[0]
    assert r.close_reason == "ref_target" and r.exit_i == 1
    assert r.exit_price == pytest.approx(98.0) and r.ref_target == pytest.approx(98.0)


def test_the_level_is_read_AT_ENTRY_and_never_again():
    d = _df([(100, 100, 100, 100), (100, 100.1, 99.4, 99.6), (99.6, 99.7, 97.95, 98.1)],
            ref_short=[98.0, 99.5, 99.5])
    r = run_backtest(d, _short_at(len(d)), BacktestConfig(
        **BASE, tp_ref_short_col="ref_s")).trades.iloc[0]
    assert r.exit_i == 2 and r.exit_price == pytest.approx(98.0), "re-read the column after entry"


def test_a_level_on_the_losing_side_is_no_target():
    d = _df(NEAR_MISS, ref_short=[100.5] * 5)
    a = run_backtest(d, _short_at(len(d)), BacktestConfig(**BASE)).trades
    b = run_backtest(d, _short_at(len(d)), BacktestConfig(**BASE, tp_ref_short_col="ref_s")).trades
    assert b.ref_target.isna().all()
    pd.testing.assert_frame_equal(a, b.drop(columns="ref_target"))


def test_whichever_level_is_nearer_the_entry_fills_first():
    """One bar through both: price had to pass the nearer one to reach the farther."""
    rows = [(100, 100, 100, 100), (100, 100.1, 96.4, 96.6)]
    far = run_backtest(_df(rows, ref_short=[96.5, np.nan]), _short_at(2),
                       BacktestConfig(**BASE, tp_ref_short_col="ref_s")).trades.iloc[0]
    near = run_backtest(_df(rows, ref_short=[98.0, np.nan]), _short_at(2),
                        BacktestConfig(**BASE, tp_ref_short_col="ref_s")).trades.iloc[0]
    assert far.close_reason == "take_profit" and far.exit_price == pytest.approx(97.0)
    assert near.close_reason == "ref_target" and near.exit_price == pytest.approx(98.0)


def test_the_stop_still_wins_a_bar_that_hits_both():
    d = _df([(100, 100, 100, 100), (100, 101.2, 97.9, 99.0)], ref_short=[98.0, np.nan])
    r = run_backtest(d, _short_at(2), BacktestConfig(**BASE, tp_ref_short_col="ref_s")).trades
    assert r.close_reason.iloc[0] == "stop_loss"


# ---------------------------------------------------------------- off means OFF
def _walk(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    rows = [(a, max(a, b) + abs(e), min(a, b) - abs(f), b) for a, b, e, f in
            zip(np.r_[c[0], c[:-1]], c, rng.normal(0, 0.05, n), rng.normal(0, 0.05, n))]
    d = _df(rows)
    el = np.zeros(n, bool)
    es = np.zeros(n, bool)
    el[::97] = True
    es[40::89] = True
    return d, Signals(entry_long=el, entry_short=es)


def test_both_rules_armed_but_never_firing_change_nothing():
    """The new branches run (a NaN level on every bar, a near-miss nobody can reach) and must leave
    every trade and the equity curve exactly as the original path computes them."""
    d, sig = _walk()
    d["nan_ref"] = np.nan
    base = dict(initial_cash=10_000, exit_enabled=True, sl_mode="entry_pct", sl_value=0.5,
                tp_mode="rr", tp_value=2.0, fee_bps=5, slippage_bps=1)
    a = run_backtest(d, sig, BacktestConfig(**base))
    b = run_backtest(d, sig, BacktestConfig(**base, tp_ref_long_col="nan_ref",
                                            tp_ref_short_col="nan_ref", tp_move_to=0.5,
                                            tp_move_near=5.0, tp_move_back=0.1))
    assert len(a.trades) > 20
    pd.testing.assert_frame_equal(
        a.trades, b.trades.drop(columns=["target_moved_i", "target_moved_time",
                                          "target_moved_to", "ref_target"]))
    np.testing.assert_array_equal(a.equity_curve.equity.to_numpy(),
                                  b.equity_curve.equity.to_numpy())


def test_off_adds_no_columns():
    d, sig = _walk(800)
    t = run_backtest(d, sig, BacktestConfig(initial_cash=10_000, exit_enabled=True,
                                            sl_mode="entry_pct", sl_value=0.5)).trades
    assert not {"target_moved_i", "ref_target"} & set(t.columns)


def test_the_sweep_runs_again():
    """run_grid built the kernel's argument list by hand and fell behind it (e61d7ab): every sweep
    died with 'expected 70 arguments'. It now goes through invoke_kernel, like run_backtest."""
    from quant.optimize import run_grid
    from quant.strategies import EmaRibbon
    d, _ = _walk(1500)
    cfg = BacktestConfig(initial_cash=10_000, exit_enabled=True, sl_mode="entry_pct",
                         sl_value=0.5, tp_mode="rr", tp_value=2.0)
    g = run_grid(d, EmaRibbon, {"fast": [10, 20], "slow": [50]}, cfg, n_jobs=1, progress=False)
    assert len(g) == 2
    one = EmaRibbon(fast=10, slow=50).backtest(d, cfg)
    row = g[g.fast == 10].iloc[0]
    assert row["num_trades"] == len(one.trades)


# ---------------------------------------------------------------- the FVG exit's gaps
def _gap_frames():
    """30 hours of 5m bars with ONE bullish 1h gap: the 20:00 candle tops at 100, the 22:00 candle
    bottoms at 102.2, and at 24:30 a 5m bar dips to 101.0 -- through the gap's midpoint (101.1)."""
    from quant.data import resample_ohlcv
    rows = []
    for hr in range(30):
        for k in range(12):
            if hr <= 20:
                r = (99.5, 100.0, 99.0, 99.5)
            elif hr == 21:
                p = 100 + 4 * k / 11
                r = (p, p + 0.2, p, p + 0.1)
            elif hr == 22:
                r = (103.5, 105.0, 102.2, 103.5)
            elif hr == 24 and k == 6:
                r = (103.2, 103.4, 101.0, 103.0)
            else:
                r = (103.5, 104.0, 103.0, 103.5)
            rows.append(r)
    base = _df(rows)
    return base, {"1h": resample_ohlcv(base, "60min").reset_index(drop=True)}


def test_a_higher_timeframe_gap_is_live_only_from_its_third_candles_close():
    from quant.indicators import fvg_on_grid
    base, frames = _gap_frames()
    g = fvg_on_grid(base, frames, mitigation="ce", min_size_atr=0.0)
    assert len(g) == 1
    r = g.iloc[0]
    assert (r.kind, r.bot, r.top) == (1, 100.0, 102.2)
    assert r.from_i == 23 * 12, "visible before the 22:00 candle closed -- a one-hour lookahead"
    assert r.to_i == 24 * 12 + 6, "mitigation must be found on the BASE bars, not the 1h candle"


def test_the_fvg_exit_picks_the_gap_near_the_target_on_the_profit_side():
    from quant.indicators import fvg_on_grid
    from quant.strategies import fvg_target
    base, frames = _gap_frames()
    g = fvg_on_grid(base, frames, mitigation="ce", min_size_atr=0.0)
    n = len(base)
    stop = np.full(n, 104.5)                       # a short at 103.5 risks 1.0; 3R = 100.5
    i = 23 * 12
    kw = dict(side="short", stop_ref=stop, stop_buffer_pct=0.0, tp_rr=3.0, near_r=1.0)
    assert fvg_target(base, g, at="ce", **kw)[i] == pytest.approx(101.1)
    assert fvg_target(base, g, at="touch", **kw)[i] == pytest.approx(102.2)
    assert fvg_target(base, g, at="full", **kw)[i] == pytest.approx(100.0)
    lv = fvg_target(base, g, at="ce", **kw)
    assert np.isnan(lv[i - 1]), "used before it was knowable"
    assert np.isnan(lv[24 * 12 + 6]), "used after it was filled"
    far = fvg_target(base, g, at="ce", **{**kw, "tp_rr": 6.0})       # target 97.5: gap too far
    assert np.isnan(far[i])
    # THE KIND, separately from the side of price: a gap ABOVE a long's entry, near its target,
    # that is BULLISH. A live bullish gap above price barely exists in practice (price would have
    # had to fall through it), which is exactly why only a hand-built one can test the filter.
    one = pd.DataFrame({"from_i": [0], "to_i": [n], "top": [107.0], "bot": [106.0],
                        "kind": [1]})
    kl = dict(side="long", stop_ref=np.full(n, 102.5), stop_buffer_pct=0.0, tp_rr=3.0,
              near_r=1.0, at="ce")                  # a long at 103.5: 3R = 106.5
    assert np.isnan(fvg_target(base, one, **kl)[i]), "a long aims at a BEARISH gap, never bullish"
    assert fvg_target(base, one.assign(kind=-1), **kl)[i] == pytest.approx(106.5)


# ---------------------------------------------------------------- the chart
def test_the_chart_draws_a_moved_target_as_a_step_inside_the_same_trace():
    """Every trade picker addresses a trade's traces by a fixed count, so the moved target and the
    second target must live INSIDE the target trace, never beside it."""
    from quant.viz.studies import pattern_figure
    d = _df(NEAR_MISS, ref_short=[98.5] * 5)
    d = pd.concat([_df([(100, 100.2, 99.8, 100)] * 60), d], ignore_index=True)
    d["t"] = d["open_time"] = pd.date_range("2026-01-01", periods=len(d), freq="5min", tz="UTC")
    sig = _short_at(len(d), 60)
    plain = run_backtest(d, sig, BacktestConfig(**BASE)).trades
    moved = run_backtest(d, sig, BacktestConfig(**BASE, tp_move_to=0.6, tp_move_near=0.9,
                                                tp_move_back=0.4)).trades
    f0 = pattern_figure(d, patterns=[], sr_zones=False, emas=(), trendlines=False,
                        trades=plain, trade_tp_rr=3.0)
    f1 = pattern_figure(d, patterns=[], sr_zones=False, emas=(), trendlines=False,
                        trades=moved, trade_tp_rr=3.0)
    assert len(f0.data) == len(f1.data), "a moved target added a trace"
    tgt = [t for t in f1.data if t.name == "trade target"][0]
    assert pytest.approx(98.2) in [y for y in tgt.y if y is not None]


# ---------------------------------------------------------------- the notebook's A/B
NB = Path(__file__).resolve().parents[1] / "notebooks"
CACHED = (Path(__file__).resolve().parents[1] / "data" / "binance_spot_BTCUSDT_5m.parquet").is_file()


@pytest.mark.skipif(not CACHED, reason="needs the cached BTCUSDT data")
def test_retrade_is_exactly_a_full_rebuild_and_refuses_anything_else():
    import sys
    sys.path.insert(0, str(NB))
    import nb21_config as CFG
    from quant.research import fvg_lab, variant
    lab = fvg_lab(CFG)
    fast = lab.retrade(USE_TP_MOVE=True, USE_FVG_EXIT=True).trades
    slow = fvg_lab(variant(CFG, USE_TP_MOVE=True, USE_FVG_EXIT=True)).trades
    pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow.reset_index(drop=True))
    with pytest.raises(KeyError):
        lab.retrade(TP_RR=2.0)                  # changes the entries' geometry: needs a rebuild


def test_notebook_21s_arms_only_touch_exit_settings():
    """So run_arms can load once and re-trade -- and so the table compares EXITS, nothing else."""
    import runpy
    from quant.research.fvg_lab import EXIT_KEYS
    cfg = runpy.run_path(str(NB / "nb21_config.py"))
    assert cfg["COMPARE"] and all(set(v) <= EXIT_KEYS for v in cfg["COMPARE"].values())
