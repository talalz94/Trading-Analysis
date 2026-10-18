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
    # exits, and then the wider trading settings: a confirmation switched off and a new target
    for over in (dict(USE_TP_MOVE=True, USE_FVG_EXIT=True),
                 dict(TL_CONFIRM_N=0, TL_CONFIRM_CANCEL=False, TP_RR=2.0)):
        fast = lab.retrade(**over).trades
        slow = fvg_lab(variant(CFG, **over)).trades
        pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow.reset_index(drop=True))
    with pytest.raises(KeyError):
        lab.retrade(TRENDLINE={})               # the loader fits lines with it: needs a rebuild


def test_notebook_21s_arms_only_touch_exit_settings():
    """So run_arms can load once and re-trade -- and so the table compares EXITS, nothing else."""
    import runpy
    from quant.research.fvg_lab import EXIT_KEYS
    cfg = runpy.run_path(str(NB / "nb21_config.py"))
    assert cfg["COMPARE"] and all(set(v) <= EXIT_KEYS for v in cfg["COMPARE"].values())


# ---------------------------------------------------------------- the exit tags on the chart
def _labelled(rows, move=None, **cfg_kw):
    """Run, then label with the SAME rule settings the run used -- as fvg_lab does."""
    from types import SimpleNamespace
    from quant.research.exit_labels import label_exits
    d = _df(rows)
    t = run_backtest(d, _short_at(len(d)), BacktestConfig(**BASE, **cfg_kw)).trades
    cfg = SimpleNamespace(TP_RR=3.0, TP_MOVE=dict(tag="MOVED", **(move or {})),
                          FVG_EXIT=dict(tag="FVG"))
    return label_exits(t, d, cfg).iloc[0]


def test_a_moved_target_says_which_trigger_moved_it_and_where_to():
    r = _labelled(NEAR_MISS, dict(near=0.9, back=0.4, after_bars=0, to=0.6),
                  tp_move_to=0.6, tp_move_near=0.9, tp_move_back=0.4)
    assert r.exit_tag == "TP·MOVED"
    # best 97.2 = 93% of the 3-point target; the bar-2 close of 98.7 gave back 50%; 3R -> 1.8R
    assert "near miss, got 93% of the way, then a close gave back 50%" in r.exit_why
    assert "3R -> 1.8R" in r.exit_why and "held" not in r.exit_why


def test_a_time_triggered_move_says_so_and_a_plain_target_says_TP():
    rows = [(100, 100, 100, 100), (100, 100.1, 99.0, 99.0), (99.0, 99.2, 98.0, 98.2),
            (98.2, 101.5, 98.1, 101.2)]
    r = _labelled(rows, dict(near=0, back=0, after_bars=2, to=0.5),
                  tp_move_to=0.5, tp_move_after=2)
    assert r.exit_tag == "TP·MOVED" and "held 2 bars" in r.exit_why
    assert "already past it" in r.exit_why            # closed on the move bar itself
    plain = _labelled([(100, 100, 100, 100), (100, 100.1, 96.9, 97.0)])
    assert plain.exit_tag == "TP" and plain.exit_why == "TP: reached the 3R target"


def test_labels_are_drawn_only_when_asked_and_read_from_customdata():
    """Off, the chart is exactly what it was (notebooks 18-20). On, the tag comes from customdata:
    rewind filters customdata with the markers it drops, and would leave a `text` array behind --
    sliding every later label onto the wrong trade, since exits are NOT in time order."""
    from quant.research.exit_labels import label_exits
    from quant.viz.studies import pattern_figure
    from types import SimpleNamespace
    d = pd.concat([_df([(100, 100.2, 99.8, 100)] * 60), _df(NEAR_MISS)], ignore_index=True)
    d["t"] = d["open_time"] = pd.date_range("2026-01-01", periods=len(d), freq="5min", tz="UTC")
    t = run_backtest(d, _short_at(len(d), 60), BacktestConfig(**BASE)).trades
    t = label_exits(t, d, SimpleNamespace(TP_RR=3.0))
    kw = dict(patterns=[], sr_zones=False, emas=(), trendlines=False, trades=t, trade_tp_rr=3.0)
    ex_off = [x for x in pattern_figure(d, **kw).data if str(x.name).startswith("exits")][0]
    ex_on = [x for x in pattern_figure(d, exit_labels=True, **kw).data
             if str(x.name).startswith("exits")][0]
    assert ex_off.mode == "markers" and ex_off.texttemplate is None
    assert ex_on.mode == "markers+text" and ex_on.texttemplate == "%{customdata[3]}"
    assert ex_on.text is None, "labels in `text` would detach from their trades on rewind"
    assert list(ex_on.customdata[:, 3]) == ["SL"]


@pytest.mark.skipif(not CACHED, reason="needs the cached BTCUSDT data")
def test_trade_tfs_switches_a_timeframe_off_without_undrawing_it():
    """TRADE_TFS picks which fitted timeframes TRADE; the rest are still fitted and drawn. A trade
    whose entry both a 5m and a 15m line justified survives with 5m off, relabelled 15m."""
    import sys
    sys.path.insert(0, str(NB))
    import nb21_config as CFG
    from quant.research import fvg_lab, variant
    # the timeframes are PINNED here: the user's own LINE_TFS / TRADE_TFS move day to day
    lab = fvg_lab(variant(CFG, LINE_TFS=("15m", "1h"), TRADE_TFS=("5m", "15m", "1h")))
    off = lab.retrade(TRADE_TFS=("15m", "1h"))
    # by PART -- "15m" contains the text "5m", so a substring test passes nothing and fails all
    assert len(off.trades) and not any("5m" in t.split("+") for t in off.trades.trigger)
    assert "tl_res" in off.D.columns, "the 5m lines must still be fitted, for the chart"
    both = lab.trades[lab.trades.trigger.str.contains(r"5m\+15m")]
    for r in both.itertuples():         # a 5m+15m trade keeps its entry, justified by 15m alone
        kept = off.trades[off.trades.entry_time == r.entry_time]
        assert len(kept) and kept.trigger.iloc[0] == "15m"
    with pytest.raises(ValueError, match="not fitted"):
        lab.retrade(TRADE_TFS=("4h",))


# ---------------------------------------------------------------- the per-trade time rules
def _rules(rows, side="short", **cols):
    """A backtest with per-trade rule columns: every kwarg ending _col names a constant column."""
    d = _df(rows)
    kw = {}
    for name, val in cols.items():
        if name.endswith("_col"):
            d[name] = val
            kw[name] = name
        else:
            kw[name] = val
    sig = _short_at(len(d)) if side == "short" else _long_at(len(d))
    return run_backtest(d, sig, BacktestConfig(**BASE, **kw)).trades


def test_the_target_decays_linearly_and_fills_on_a_later_bar():
    """The user's own example, in bars: 2% of the target's distance every period, so after five
    periods a 3R target is 2.7R. The decay is judged on the close and used from the next bar."""
    rows = [(100, 100, 100, 100)] + [(99.5, 99.9, 99.4, 99.5)] * 5 + \
           [(99.5, 99.6, 97.25, 97.8)]   # bar 6: 97.3 = 2.7R is inside this bar, 97.0 (3R) is not
    t = _rules(rows, tp_decay_every_col=1.0, tp_decay_cut_col=0.02)
    r = t.iloc[0]
    assert r.close_reason == "tp_decayed" and r.exit_i == 6
    assert r.exit_price == pytest.approx(97.3)
    off = run_backtest(_df(rows), _short_at(len(rows)), BacktestConfig(**BASE)).trades.iloc[0]
    assert off.close_reason == "forced_close_end"


def test_a_decayed_target_already_behind_the_close_closes_there():
    """Cut 40% a bar: after bar 1 the 3R target (97) is 1.8R (98.2). Bar 1 CLOSED at 98.1, already
    past it, so the trade is out at that close -- not at 98.2, a price it never had to wait for."""
    rows = [(100, 100, 100, 100), (100, 100.1, 98.0, 98.1), (98.1, 98.2, 96.5, 96.8)]
    r = _rules(rows, tp_decay_every_col=1.0, tp_decay_cut_col=0.4).iloc[0]
    assert r.close_reason == "tp_decayed" and r.exit_i == 1
    assert r.exit_price == pytest.approx(98.1)


def test_the_hard_close_closes_after_n_bars_whatever_the_trade_is_doing():
    rows = [(100, 100, 100, 100)] + [(99.8, 100.2, 99.6, 99.9)] * 6
    r = _rules(rows, close_after_col=3.0).iloc[0]
    assert r.close_reason == "time_stop" and r.exit_i == 3 and r.exit_price == pytest.approx(99.9)


def test_the_time_rule_moves_the_stop_only_once_a_close_is_in_profit():
    rows = [(100, 100, 100, 100),
            (100, 100.4, 99.9, 100.3),     # bar 1: 2 bars not yet held
            (100.3, 100.5, 100.1, 100.4),  # bar 2: held 2, but closing in LOSS -> wait
            (100.4, 100.4, 99.2, 99.3),    # bar 3: first close in profit -> stop to breakeven (100)
            (99.3, 100.2, 99.2, 100.1)]    # bar 4: high 100.2 takes the moved stop
    t = _rules(rows, stop_time_after_col=2.0, stop_time_move_col=1.0, stop_time_if_profit=True)
    r = t.iloc[0]
    assert r.close_reason == "stop_loss" and r.exit_i == 4
    assert r.exit_price == pytest.approx(100.0) and r.stop_moved_by == "time"
    # without the profit check it moves at bar 2, where the close (100.4) is already through a
    # breakeven stop -- so it closes at that close, never at a price the bar did not trade
    u = _rules(rows, stop_time_after_col=2.0, stop_time_move_col=1.0,
               stop_time_if_profit=False).iloc[0]
    assert u.exit_i == 2 and u.exit_price == pytest.approx(100.4) and u.close_reason == "stop_loss"


def test_a_stop_step_moves_the_stop_a_share_of_the_risk_when_price_gets_there():
    """At 50% of the way to a 3R target (1.5R, 98.5), move the stop 50% of the risk: 101 -> 100.5."""
    rows = [(100, 100, 100, 100), (100, 100.1, 98.4, 98.6), (98.6, 100.6, 98.5, 100.4)]
    cfg = BacktestConfig(**{**BASE, "tp_mode": "none"},
                         take_profits=(TakeProfit("rr", 1.5, close_pct=0, move_stop_mode="risk_pct",
                                                  move_stop_value=50),
                                       TakeProfit("rr", 3.0, close_pct=100)))
    r = run_backtest(_df(rows), _short_at(3), cfg).trades.iloc[0]
    assert r.close_reason == "stop_loss" and r.exit_i == 2
    assert r.exit_price == pytest.approx(100.5) and r.stop_moved_by == "step"


def test_the_time_rules_off_change_nothing():
    d, sig = _walk()
    for c in ("a", "b"):
        d[c] = np.nan
    base = dict(initial_cash=10_000, exit_enabled=True, sl_mode="entry_pct", sl_value=0.5,
                tp_mode="rr", tp_value=2.0, fee_bps=5, slippage_bps=1)
    x = run_backtest(d, sig, BacktestConfig(**base))
    y = run_backtest(d, sig, BacktestConfig(**base, tp_decay_every_col="a", tp_decay_cut_col="b",
                                            close_after_col="a", stop_time_after_col="a",
                                            stop_time_move_col="b"))
    pd.testing.assert_frame_equal(x.trades, y.trades.drop(columns=["stop_moved_by"]))
    np.testing.assert_array_equal(x.equity_curve.equity.to_numpy(), y.equity_curve.equity.to_numpy())


@pytest.mark.skipif(not CACHED, reason="needs the cached BTCUSDT data")
def test_each_trade_takes_the_rule_settings_of_the_line_that_fired_it():
    """HARD_CLOSE per line timeframe: 5m-line trades out after 1h, 15m ones left alone."""
    import sys
    sys.path.insert(0, str(NB))
    import nb21_config as CFG
    from quant.research import fvg_lab, variant
    # its OWN window: nb21_config.py is the user's file and its dates move (2026-10-18 they moved to
    # Sep-Oct, a window with no "5m+15m" trade), which must not decide whether this rule holds
    lab = fvg_lab(variant(CFG, LINE_TFS=("15m", "1h"), TRADE_TFS=("5m", "15m", "1h"),
                          USE_FVG_EXIT=False, TL_CONFIRM_N={"1h": 3, "15m": 3, "5m": 2, "*": 0},
                          VIEW_START="2026-06-01", END="2026-07-15", WARMUP_START="2026-05-01"))
    t = lab.retrade(USE_HARD_CLOSE=True,
                    HARD_CLOSE=dict(tag="HARD", after_h={"5m": 1, "*": 0})).trades
    five = t[t.trigger == "5m"]
    assert len(five) and (five.bars_held <= 12).all(), "a 5m-line trade outlived its 1h close"
    others = t[t.trigger.str.split("+").map(lambda p: "5m" not in p)]
    assert (others.close_reason != "time_stop").all() and (others.bars_held > 12).any()
    assert set(t.loc[t.close_reason == "time_stop", "exit_tag"]) <= {"HARD"}
    # a trade BOTH a 5m and a 15m line fired uses the COARSER line's settings -- so no 1h close
    mixed = t[t.trigger == "5m+15m"]
    assert len(mixed) and (mixed.close_reason != "time_stop").all()
