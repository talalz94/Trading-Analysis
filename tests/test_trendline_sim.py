"""Multi-timeframe trendline entries, and the trade overlay that draws them.

The rule is "a fitted line from ANY timeframe gives way". Two things have to hold for that to mean
anything: a higher timeframe's level must not be visible before its bar closed, and the same
confirmation rule must behave the same way on every timeframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _frame(n=3600, seed=7, freq="1min"):
    """A trend that swings, then reverses hard — geometry a trendline fitter can actually find.

    A random walk is the wrong fixture here. At this length it grew a line on two of six
    (timeframe x side) combinations and never once put a whole candle past one, so the properties
    under test were mostly untested. This is a rising channel with a dozen clean swings, broken by
    a sharp decline, which gives every timeframe both a support and a resistance line and produces
    breaks of both kinds.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    turn = int(n * 0.7)
    trend = np.where(i < turn, 30000 + 2.2 * i, 30000 + 2.2 * turn - 6.0 * (i - turn))
    swing = 260 * np.sin(2 * np.pi * i / 240)
    close = trend + swing + rng.normal(0, 12, n)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 18, n))
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    t = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": 1.0})


@pytest.fixture(scope="module")
def mtf():
    from quant.data import resample_ohlcv
    from quant.indicators.trendlines import mtf_trendlines

    base = _frame()
    frames = {"5min": resample_ohlcv(base, "5min"), "15min": resample_ohlcv(base, "15min")}
    return base, frames, mtf_trendlines(base, frames, min_span_bars=20, max_age_bars=400)


def test_mtf_trendlines_puts_every_timeframes_level_on_the_base_grid(mtf):
    base, frames, D = mtf
    assert len(D) == len(base)
    for col in ("tl_res", "tl_sup", "5min__tl_res", "15min__tl_sup"):
        assert col in D.columns
    assert np.isfinite(D["tl_res"].to_numpy(float)).any(), "no base-timeframe line at all"


def test_a_higher_timeframe_level_is_invisible_until_its_bar_closed(mtf):
    """The whole point of the merge. A 15m line read off the bar stamped 12:00 is knowable at 12:15.

    Checked by truncation, the same test the detectors get: rebuild everything from the first k bars
    only, and every value that was already published must be unchanged. If any higher-timeframe
    level leaked backwards, the truncated run would disagree.
    """
    from quant.data import resample_ohlcv
    from quant.indicators.trendlines import mtf_trendlines

    base, frames, D = mtf
    k = 1800
    part = mtf_trendlines(base.iloc[:k].reset_index(drop=True),
                          {tf: f[pd.to_datetime(f["t"]) < base["t"].iloc[k]].reset_index(drop=True)
                           for tf, f in frames.items()},
                          min_span_bars=20, max_age_bars=400)
    for col in ("5min__tl_res", "5min__tl_sup", "15min__tl_res", "15min__tl_sup"):
        a = D[col].to_numpy(float)[:k]
        b = part[col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both]), f"{col} changed when the future was removed"


def test_trendline_level_cols_lists_the_base_frame_first_then_coarsest(mtf):
    from quant.indicators.trendlines import trendline_level_cols

    _, _, D = mtf
    assert trendline_level_cols(D, "long") == ["tl_res", "15min__tl_res", "5min__tl_res"]
    assert trendline_level_cols(D, "short") == ["tl_sup", "15min__tl_sup", "5min__tl_sup"]


def test_the_break_rule_behaves_the_same_on_every_timeframe(mtf):
    """A line is RETIRED the moment it breaks, so a second confirmation bar would be comparing
    itself against a level that no longer exists. Measured before the fix: the base timeframe fired
    ZERO times at n=2 while the merged higher ones, whose values persist across their own bar,
    fired hundreds. The entry holds the last published value for the length of the confirmation
    window, which is a past value and so still causal.
    """
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    tested = 0
    for pref in ("", "5min__", "15min__"):
        for side in ("long", "short"):
            base = dict(tl_prefixes=(pref,), tl_full_m=0, tl_touch_k=0, tl_combine="all")
            one = _tl_entry(dict(base, tl_close_n=1), side)(D).sum()
            two = _tl_entry(dict(base, tl_close_n=2), side)(D).sum()
            if not one:
                continue          # this synthetic frame grew no line of that kind — nothing to test
            tested += 1
            assert two > 0, f"{pref or 'base'} {side} fires at n=1 but not n=2 — the level is not held"
            assert two <= one, "asking for more confirmation cannot produce MORE entries"
    assert tested >= 2, "too few live lines to have tested anything"


def test_the_level_is_held_across_the_confirmation_window():
    """The exact bug, on a hand-built frame so no fitter is involved.

    `add_trendlines` RETIRES a line the moment price closes through it, so the level column goes NaN
    at the break bar. Asking for two closes beyond the line then asks bar 2 to compare itself
    against nothing, and `NaN` comparisons are False — so the rule fired ZERO times on the base
    timeframe while the merged higher ones, whose values persist across their own bar, fired
    hundreds. Holding the last published value for the length of the window fixes it and stays
    causal, because the held value is one the frame had already published.
    """
    from quant.scenarios.entries import _tl_entry

    n = 12
    close = np.array([100.0] * 6 + [105.0] * 6)      # breaks up at bar 6 and stays there
    lvl = np.array([101.0] * 7 + [np.nan] * 5)       # ...and the line retires the moment it does
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
                      "open": close, "high": close, "low": close, "close": close,
                      "volume": 1.0, "tl_res": lvl})
    p = dict(tl_prefixes=("",), tl_full_m=0, tl_touch_k=0, tl_combine="all")

    one = _tl_entry(dict(p, tl_close_n=1), "long")(d)
    assert one.sum() == 1 and one[6], "the break bar itself must fire at n=1"

    two = _tl_entry(dict(p, tl_close_n=2), "long")(d)
    assert two[7], "the second close beyond a line that just retired must still confirm"

    # the hold is exactly as long as the window asks for, never longer
    assert not _tl_entry(dict(p, tl_close_n=2), "long")(
        d.assign(close=np.r_[close[:7], [100.0] * 5])).any(), "price came back — nothing should fire"


def test_long_takes_resistance_and_short_takes_support(mtf):
    """A long breaks UP through resistance; a short breaks DOWN through support. Never the reverse."""
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=1, tl_full_m=0, tl_touch_k=0, tl_combine="all")
    for side, w, cmp in (("long", "res", np.greater), ("short", "sup", np.less)):
        fired = _tl_entry(p, side)(D)
        assert fired.any()
        c = D["close"].to_numpy(float)
        beyond = np.zeros(len(D), bool)
        for pref in ("", "5min__"):
            lvl = D[f"{pref}tl_{w}"].to_numpy(float)
            beyond |= np.isfinite(lvl) & cmp(c, lvl)
        assert (fired & ~beyond).sum() == 0, f"{side} fired without price being through a {w} line"


def test_a_full_candle_beyond_is_stricter_than_a_close_beyond(mtf):
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    seen = 0
    for side in ("long", "short"):
        base = dict(tl_prefixes=("", "5min__", "15min__"), tl_touch_k=0, tl_combine="all")
        closes = _tl_entry(dict(base, tl_close_n=1, tl_full_m=0), side)(D).sum()
        fulls = _tl_entry(dict(base, tl_close_n=0, tl_full_m=1), side)(D).sum()
        assert fulls <= closes, "a whole candle past the line cannot be commoner than a close past it"
        seen += int(fulls > 0)
    assert seen, "no side ever put a whole candle past a line, so nothing was compared"


def test_the_scenarios_run_both_ways_with_the_stop_on_the_right_side(mtf):
    from quant.scenarios import run_pieces

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__", "15min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    cfg = dict(initial_cash=10_000, exit_enabled=False, sizing_mode="risk_pct_equity",
               sizing_value=1.0, max_open_trades=1)
    for side in ("long", "short"):
        run = run_pieces(D, f"mtf_trendline_{side}@v1", "high_of_n@v1", "be_ladder@v1",
                         params=p, cfg=cfg)
        tr = run.trades
        assert len(tr), f"{side} produced no trades"
        assert (tr["side"] == side).all()
        ok = (tr.stop_price < tr.entry_price) if side == "long" else (tr.stop_price > tr.entry_price)
        assert ok.all(), f"a {side}'s stop must sit on the losing side of entry"


def test_each_trade_is_labelled_with_the_timeframe_that_fired(mtf):
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__", "15min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    out = trendline_trigger(run.df, run.trades, p, prefix_names=names)
    assert "trigger" in out.columns and len(out) == len(run.trades)
    assert (out["trigger"] != "?").all(), "a trade fired without any timeframe claiming it"
    assert set("+".join(out["trigger"]).split("+")) <= set(names.values())


def test_the_chart_draws_entries_stops_targets_and_exits(mtf):
    from quant.scenarios import run_pieces
    from quant.viz.studies import pattern_figure

    _, frames, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    tr = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                    params=p, cfg=dict(initial_cash=10_000, exit_enabled=False)).trades
    assert len(tr)

    bare = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=False)
    fig = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=False,
                         trades=tr, trade_tp_rr=3.0)
    names = [t.name for t in fig.data if t.name]
    assert "trade stop" in names and "trade target" in names
    assert any(n.startswith("short entries") for n in names)
    assert any(n.startswith("exits") for n in names)
    # the overlay is additive: the chart underneath is untouched
    assert len(fig.data) > len(bare.data)

    tgt = [t for t in fig.data if t.name == "trade target"][0]
    stp = [t for t in fig.data if t.name == "trade stop"][0]
    ys, yt = np.array(stp.y, float), np.array(tgt.y, float)
    m = np.isfinite(ys) & np.isfinite(yt)
    assert (ys[m] > yt[m]).all(), "a short's stop must be drawn above its target"
