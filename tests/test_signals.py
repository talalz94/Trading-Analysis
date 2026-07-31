"""Unit tests for vectorized signal primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import signals as S


def _df(close, open_=None):
    n = len(close)
    return pd.DataFrame({
        "open": open_ if open_ is not None else close,
        "close": close,
        "high": close,
        "low": close,
    }, index=range(n))


def test_cross_up_and_down():
    df = _df([1.0, 2.0, 3.0, 2.0, 1.0])
    up = S.cross_up(df, "close", 2.5)      # crosses above constant 2.5 at index 2
    dn = S.cross_down(df, "close", 2.5)    # crosses below at index 3
    assert up.tolist() == [False, False, True, False, False]
    assert dn.tolist() == [False, False, False, True, False]


def test_last_all_above_include_current():
    df = _df([1.0, 3.0, 3.0, 3.0, 0.0])
    # close > 2 for the last 3 bars (indices 1,2,3) -> True at index 3 only
    m = S.last_all_above(df, "close", 2.0, 3)
    assert m.tolist() == [False, False, False, True, False]


def test_prev_all_below_excludes_current():
    df = _df([1.0, 1.0, 1.0, 5.0])
    # previous 2 completed bars below 2 -> True at index 2 (prev 0,1) and index 3 (prev 1,2)
    m = S.prev_all_below(df, "close", 2.0, 2)
    assert m.tolist() == [False, False, True, True]


def test_consecutive_green():
    # green when close>open
    close = [2, 3, 4, 1]
    open_ = [1, 2, 3, 2]
    df = _df(close, open_)
    m = S.consecutive_green(df, 2)   # last 2 candles green
    assert m.tolist() == [False, True, True, False]


def test_refs_ordered_ribbon():
    df = pd.DataFrame({"a": [3, 1], "b": [2, 2], "c": [1, 3]})
    m = S.refs_ordered(df, ["a", "b", "c"], descending=True)
    assert m.tolist() == [True, False]


def test_combinators():
    a = np.array([True, True, False])
    b = np.array([True, False, False])
    assert S.all_of(a, b).tolist() == [True, False, False]
    assert S.any_of(a, b).tolist() == [True, True, False]
    assert S.none_of(a, b).tolist() == [False, False, True]


def test_becomes_true_rising_edge():
    # fires only on the bar the mask FIRST turns true; re-arms after it goes false again
    m = np.array([False, True, True, False, True, False])
    assert S.becomes_true(m).tolist() == [False, True, False, False, True, False]
    # a mask that is true on bar 0 fires on bar 0 (pre-start counts as False)
    assert S.becomes_true(np.array([True, True, False])).tolist() == [True, False, False]


def test_latch_set_reset():
    # on from the set bar until the reset bar; reset wins on ties
    s = np.array([False, True, False, False, True, False])
    r = np.array([False, False, False, True, False, False])
    assert S.latch(s, r).tolist() == [False, True, True, False, True, True]
    # set and reset on the same bar -> stays OFF that bar
    assert S.latch(np.array([True, False]), np.array([True, False])).tolist() == [False, False]


def test_arm_reset_trigger_independent_reset_condition():
    # fire at 1 and 4; a fire at 2 is suppressed (still disarmed, no reset happened yet);
    # a fire at 6 is suppressed too (disarmed since bar 4, no reset since)
    fire  = np.array([False, True, True, False, True, False, True])
    reset = np.array([False, False, False, True, False, False, False])
    assert S.arm_reset_trigger(fire, reset).tolist() == \
        [False, True, False, False, True, False, False]
    # reset on the SAME bar as a would-be fire does not fire that bar (re-arms for a later bar)
    fire2  = np.array([True, False, True])
    reset2 = np.array([False, True, False])
    out = S.arm_reset_trigger(fire2, reset2)
    assert out.tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# time-filter block-lists: accept both a single range and a sequence of ranges
# ---------------------------------------------------------------------------
def _tdf(n=24 * 60):
    t = pd.date_range("2026-01-05", periods=n, freq="1min", tz="UTC")   # a Monday
    return pd.DataFrame({"t": t})


def test_normalize_time_ranges_accepts_single_pair():
    from quant.signals.time_filters import normalize_time_ranges
    assert normalize_time_ranges(("13:30", "14:15")) == [("13:30", "14:15")]
    assert normalize_time_ranges((("13:30", "14:15"),)) == [("13:30", "14:15")]
    assert normalize_time_ranges((("13:30", "14:15"), ("20:00", "20:30"))) == [
        ("13:30", "14:15"), ("20:00", "20:30")]
    assert normalize_time_ranges(None) == []


def test_normalize_time_ranges_rejects_bad_input():
    from quant.signals.time_filters import normalize_time_ranges
    with pytest.raises(ValueError):
        normalize_time_ranges(("13:30", "14:15", "20:00"))


def test_in_any_times_single_and_multi():
    df = _tdf()
    one = S.in_any_times(df, ("13:30", "14:15"))
    assert one.sum() == 45                                   # 45 one-minute bars
    two = S.in_any_times(df, (("13:30", "14:15"), ("20:00", "20:30")))
    assert two.sum() == 75
    assert (one & ~two).sum() == 0                           # the single range is a subset


def test_hour_in_and_weekday_in_accept_bare_int():
    df = _tdf()
    assert S.hour_in(df, 12).sum() == 60
    assert np.array_equal(S.hour_in(df, 12), S.hour_in(df, [12]))
    assert S.weekday_in(df, 0).all()                         # the whole day is a Monday
    assert not S.weekday_in(df, 6).any()


def test_block_filters_exclude_entries():
    """Strategy-level: blocked hours/days must never appear in entry signals."""
    from quant.strategies.builder import Rule, RuleStrategy
    from quant.strategies.conditions import Cond
    df = _tdf()
    df["open"] = df["high"] = df["low"] = df["close"] = 100.0
    always = Cond(lambda d: np.ones(len(d), bool), "always")
    strat = RuleStrategy(rules=(Rule("r", entry=always),), block_hours=(12, 16, 18))
    el = strat.signals(df).entry_long
    assert not set(df.t.dt.hour[el]) & {12, 16, 18}
    assert el.sum() == len(df) - 3 * 60
