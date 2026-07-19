"""
Vectorized signal primitives (numpy-native hot path).

Each primitive consumes whole columns and returns a full boolean numpy array (one value
per bar) — the vectorized counterpart of the legacy per-bar `RuleContextMixin` predicates
(`cross_up_pair`, `close_above_all`, `consecutive_green`, `refs_ordered`, `prev_all_below`…).

A "ref" is a column name (str), a scalar, a pandas Series, or a numpy array. Resolution goes
through `_arr` (fast: avoids re-coercing already-numeric columns). Anti-lookahead: primitives
only look at the current and PAST bars, never the future.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd

Ref = Union[str, float, int, pd.Series, np.ndarray]


def _arr(df: pd.DataFrame, ref: Ref) -> np.ndarray:
    """Resolve a ref to a float64 numpy array aligned to df rows (no needless copies)."""
    if isinstance(ref, np.ndarray):
        return ref if ref.dtype == np.float64 else ref.astype(np.float64)
    if isinstance(ref, pd.Series):
        a = ref.to_numpy()
        return a if a.dtype == np.float64 else pd.to_numeric(ref, errors="coerce").to_numpy(np.float64)
    if isinstance(ref, str):
        if ref not in df.columns:
            raise KeyError(f"Column '{ref}' not in df.")
        s = df[ref]
        a = s.to_numpy()
        if a.dtype == np.float64:
            return a
        if a.dtype.kind in "iuf":
            return a.astype(np.float64)
        return pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
    return np.full(len(df), float(ref), dtype=np.float64)


def col(df: pd.DataFrame, ref: Ref) -> pd.Series:
    """Resolve a ref to a float Series (public/manual convenience)."""
    return pd.Series(_arr(df, ref), index=df.index)


def _shift1(a: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[0] = np.nan
    out[1:] = a[:-1]
    return out


def _roll_all(cond: np.ndarray, n: int, include_current: bool = True) -> np.ndarray:
    """True where `cond` held for n consecutive bars (ending at i). O(n) via cumsum.

    include_current=False => the n bars are the PREVIOUS completed bars (excludes i).
    """
    n = int(n)
    c = cond.astype(np.int32)
    if not include_current:
        shifted = np.zeros_like(c)
        shifted[1:] = c[:-1]
        c = shifted
    m = c.shape[0]
    out = np.zeros(m, dtype=bool)
    if n <= 0 or n > m:
        return out
    csum = np.empty(m + 1, dtype=np.int64)
    csum[0] = 0
    np.cumsum(c, out=csum[1:])
    # window sum ending at i (inclusive of n bars) = csum[i+1] - csum[i+1-n].
    # Use slicing (views), not fancy indexing, so this stays a couple of passes.
    out[n - 1:] = (csum[n:] - csum[:-n]) == n
    return out


# --- comparisons -------------------------------------------------------------

def above(df, a: Ref, b: Ref) -> np.ndarray:
    return _arr(df, a) > _arr(df, b)


def below(df, a: Ref, b: Ref) -> np.ndarray:
    return _arr(df, a) < _arr(df, b)


def above_all(df, a: Ref, refs: Sequence[Ref]) -> np.ndarray:
    a_v = _arr(df, a)
    out = np.ones(a_v.shape[0], dtype=bool)
    for r in refs:
        out &= a_v > _arr(df, r)
    return out


def below_all(df, a: Ref, refs: Sequence[Ref]) -> np.ndarray:
    a_v = _arr(df, a)
    out = np.ones(a_v.shape[0], dtype=bool)
    for r in refs:
        out &= a_v < _arr(df, r)
    return out


# --- crosses -----------------------------------------------------------------

def cross_up(df, a: Ref, b: Ref) -> np.ndarray:
    """a crosses above b on this bar: prev a<=b and now a>b."""
    d = _arr(df, a) - _arr(df, b)
    prev = _shift1(d)
    return (prev <= 0) & (d > 0)


def cross_down(df, a: Ref, b: Ref) -> np.ndarray:
    d = _arr(df, a) - _arr(df, b)
    prev = _shift1(d)
    return (prev >= 0) & (d < 0)


def crossed_up_within(df, a: Ref, b: Ref, lookback: int = 1) -> np.ndarray:
    """A crossed above B at some point within the last `lookback` bars (inclusive)."""
    up = cross_up(df, a, b).astype(np.int32)
    m = up.shape[0]
    csum = np.empty(m + 1, np.int64)
    csum[0] = 0
    np.cumsum(up, out=csum[1:])
    lb = int(lookback)
    idx = np.arange(m)
    start = np.maximum(idx + 1 - lb, 0)
    return (csum[idx + 1] - csum[start]) > 0


# --- persistence over a window ----------------------------------------------

def last_all_above(df, x: Ref, level: Ref, n: int, include_current: bool = True) -> np.ndarray:
    return _roll_all(_arr(df, x) > _arr(df, level), n, include_current)


def last_all_below(df, x: Ref, level: Ref, n: int, include_current: bool = True) -> np.ndarray:
    return _roll_all(_arr(df, x) < _arr(df, level), n, include_current)


def prev_all_above(df, x: Ref, level: Ref, n: int) -> np.ndarray:
    return _roll_all(_arr(df, x) > _arr(df, level), n, include_current=False)


def prev_all_below(df, x: Ref, level: Ref, n: int) -> np.ndarray:
    return _roll_all(_arr(df, x) < _arr(df, level), n, include_current=False)


# --- candles -----------------------------------------------------------------

def is_green(df, open_col: str = "open", close_col: str = "close") -> np.ndarray:
    return _arr(df, close_col) > _arr(df, open_col)


def is_red(df, open_col: str = "open", close_col: str = "close") -> np.ndarray:
    return _arr(df, close_col) < _arr(df, open_col)


def consecutive_green(df, n: int, open_col="open", close_col="close") -> np.ndarray:
    return _roll_all(is_green(df, open_col, close_col), n, include_current=True)


def consecutive_red(df, n: int, open_col="open", close_col="close") -> np.ndarray:
    return _roll_all(is_red(df, open_col, close_col), n, include_current=True)


# --- monotonic / ribbon ------------------------------------------------------

def _diff1(a: np.ndarray) -> np.ndarray:
    d = np.empty_like(a)
    d[0] = np.nan
    d[1:] = a[1:] - a[:-1]
    return d


def rising(df, x: Ref, n: int = 1) -> np.ndarray:
    return _roll_all(_diff1(_arr(df, x)) > 0, n, include_current=True)


def falling(df, x: Ref, n: int = 1) -> np.ndarray:
    return _roll_all(_diff1(_arr(df, x)) < 0, n, include_current=True)


def refs_ordered(df, refs: Sequence[Ref], *, descending: bool = True) -> np.ndarray:
    arrs = [_arr(df, r) for r in refs]
    out = np.ones(arrs[0].shape[0], dtype=bool)
    for i in range(len(arrs) - 1):
        out &= (arrs[i] >= arrs[i + 1]) if descending else (arrs[i] <= arrs[i + 1])
    return out


# --- boolean combinators -----------------------------------------------------

def all_of(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("all_of requires at least one mask")
    out = np.asarray(masks[0], bool).copy()
    for m in masks[1:]:
        out &= np.asarray(m, bool)
    return out


def any_of(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("any_of requires at least one mask")
    out = np.asarray(masks[0], bool).copy()
    for m in masks[1:]:
        out |= np.asarray(m, bool)
    return out


def none_of(*masks: np.ndarray) -> np.ndarray:
    return ~any_of(*masks)
