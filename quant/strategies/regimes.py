"""
Regime conditioners — `Cond`s that say *when* a setup is allowed to fire, not *what* it fires on.

These are meant to be ANDed onto an entry (`run_pieces(..., entry_filter=...)`) so a trigger can be
tested unconditionally first and then conditioned, which is the only way to tell a real conditioner
from one that just happened to remove the losing trades in-sample.

Everything here is CAUSAL and self-scaling. Thresholds are expressed as **rolling quantiles of the
instrument's own history**, shifted one bar, rather than absolute numbers — an ATR of 300 means
something different on BTC at 30k and at 120k, and an absolute cut would silently become a date
filter. The rolling window is deliberately long (default ~1.5 trading days of 1m bars) so the
threshold tracks the regime without chasing the last hour.

    from quant.strategies.regimes import vol_regime, trendiness, vol_shock
    run = run_pieces(df, "wt_cross_long@v1", "atr@v1", "rr@v1",
                     entry_filter=vol_regime("high") & trendiness("trending"))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr as _atr
from .conditions import Cond, raw

DEFAULT_WINDOW = 2000


def _roll_q(s: pd.Series, q: float, window: int) -> np.ndarray:
    """Rolling quantile, shifted one bar so a threshold never uses the bar it is judging."""
    return s.rolling(int(window), min_periods=int(window) // 4).quantile(q).shift(1).to_numpy()


def vol_regime(mode: str = "high", *, period: int = 14, window: int = DEFAULT_WINDOW,
               lo_q: float = 0.33, hi_q: float = 0.67) -> Cond:
    """Volatility tercile by ATR, measured against the instrument's own recent distribution.

    `mode`: 'high' | 'low' | 'mid'. Costs are a fixed price width, so a rule that only survives in
    high volatility is a rule whose edge is smaller than the spread the rest of the time — worth
    knowing either way.
    """
    def fn(df):
        a = _atr(df, period)
        a = a if isinstance(a, pd.Series) else pd.Series(np.asarray(a), index=df.index)
        v = a.to_numpy(float)
        lo, hi = _roll_q(a, lo_q, window), _roll_q(a, hi_q, window)
        with np.errstate(invalid="ignore"):
            if mode == "high":
                return v > hi
            if mode == "low":
                return v < lo
            return (v >= lo) & (v <= hi)

    return raw(fn, f"vol_{mode}")


def efficiency_ratio(period: int = 30) -> pd.Series:
    """Kaufman efficiency ratio over `period` bars: |net move| / total path travelled.

    1.0 = a straight line, 0.0 = pure chop. Uses only closed bars.
    """
    def _er(c: pd.Series) -> pd.Series:
        net = (c - c.shift(period)).abs()
        path = c.diff().abs().rolling(period, min_periods=period).sum()
        return net / path.replace(0, np.nan)

    return _er


def trendiness(mode: str = "trending", *, period: int = 30, window: int = DEFAULT_WINDOW,
               q: float = 0.67) -> Cond:
    """Is the market currently travelling in a line or churning? (efficiency-ratio terciles)

    `mode`: 'trending' (ER above its own upper tercile) | 'choppy' (below the lower one).
    Directly relevant here: the earlier hour-of-day work found BTC 1m choppy at *every* hour, so a
    conditioner on realised structure is the untried alternative to a clock filter.
    """
    def fn(df):
        c = df["close"]
        net = (c - c.shift(period)).abs()
        path = c.diff().abs().rolling(period, min_periods=period).sum()
        er = net / path.replace(0, np.nan)
        v = er.to_numpy(float)
        hi, lo = _roll_q(er, q, window), _roll_q(er, 1.0 - q, window)
        with np.errstate(invalid="ignore"):
            return v > hi if mode == "trending" else v < lo

    return raw(fn, f"er_{mode}")


def vol_shock(*, lookback: int = 30, window: int = DEFAULT_WINDOW, q: float = 0.99,
              within: int = 0) -> Cond:
    """ANNOUNCEMENT PROXY. No economic calendar is available here, so a scheduled release is stood in
    for by what it actually does to the tape: a burst of realised volatility far outside the recent
    distribution.

    `within=0`  -> the shock bar itself (trading INTO/DURING the event)
    `within=N`  -> the N bars AFTER a shock (trading the aftermath)
    A rule that only works with `within=0` is trading the spike itself, where real slippage is
    nothing like the modelled spread — treat any such result as unrealisable.
    """
    def fn(df):
        r = np.log(df["close"].astype(float)).diff()
        rv = r.rolling(int(lookback), min_periods=int(lookback) // 2).std()
        thr = _roll_q(rv, q, window)
        with np.errstate(invalid="ignore"):
            shock = rv.to_numpy(float) > thr
        shock = np.nan_to_num(shock, nan=0.0).astype(bool)
        if within <= 0:
            return shock
        # any shock in the preceding `within` bars (excluding this one)
        s = pd.Series(shock.astype(float))
        return (s.shift(1).rolling(int(within), min_periods=1).max().fillna(0) > 0).to_numpy()

    return raw(fn, f"vol_shock(within={within})")


def not_(cond: Cond) -> Cond:
    """Readable negation, for building 'avoid the announcement window' filters."""
    return ~cond


def range_position(mode: str = "upper", *, lookback: int = 240, frac: float = 0.25) -> Cond:
    """Where price sits inside its own recent range: 'upper' / 'lower' `frac` of the last `lookback`
    bars. A cheap, causal stand-in for "extended" that needs no level at all."""
    def fn(df):
        c = df["close"].to_numpy(float)
        hi = df["high"].rolling(lookback, min_periods=lookback // 2).max().to_numpy(float)
        lo = df["low"].rolling(lookback, min_periods=lookback // 2).min().to_numpy(float)
        with np.errstate(invalid="ignore"):
            pos = (c - lo) / (hi - lo)
        return pos > (1 - frac) if mode == "upper" else pos < frac

    return raw(fn, f"range_{mode}")


CONDITIONERS = {
    "vol_high": lambda: vol_regime("high"),
    "vol_low": lambda: vol_regime("low"),
    "vol_mid": lambda: vol_regime("mid"),
    "trending": lambda: trendiness("trending"),
    "choppy": lambda: trendiness("choppy"),
    "shock_now": lambda: vol_shock(within=0),
    "post_shock": lambda: vol_shock(within=60),
    "no_shock": lambda: ~vol_shock(within=60),
    "range_upper": lambda: range_position("upper"),
    "range_lower": lambda: range_position("lower"),
}
