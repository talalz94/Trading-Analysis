"""Unit tests for vectorized signal primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
