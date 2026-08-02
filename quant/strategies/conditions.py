"""
Composable, vectorized entry/exit *conditions* for the declarative rule-builder.

A `Cond` wraps a function `df -> bool np.ndarray` and supports boolean algebra with the natural
operators so complex rules read like their description:

    entry = full_above("ema_200", n=2) & above("close", "ema_50")
    exit  = full_below("ema_50") | ha_red(3)

Everything is fully vectorized (numpy / the existing `quant.signals` primitives) — building a Cond
does no work; calling it on a prepared DataFrame returns the whole boolean series in one pass. This
is the same speed class as every other signal in the engine; there are no per-bar Python loops.

Candle source
-------------
`full_above` / `full_below` / green/red helpers read OHLC. Pass `src="ha"` (or use the module-level
`ha` helpers) to evaluate on Heikin Ashi columns (`ha_open/ha_high/ha_low/ha_close`) instead of real
OHLC — the strategy just needs the `heikin_ashi()` indicator declared so those columns exist.

Indicator specs
---------------
`ema(200)`, `sma(50)`, `heikin_ashi()`, `swings(5, 5)`, `atr(14)` are tiny declarations of the
columns a strategy needs; `RuleStrategy` applies them once (de-duplicated) in `prepare()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .. import signals as S
from ..data import build_mtf
from ..indicators import (add_atr, add_candle_patterns, add_chart_patterns, add_emas,
                          add_heikin_ashi, add_smas, add_sr_zones, add_swings)
from ..indicators.candle_patterns import PATTERN_DIRECTION as _CDL_DIR
from ..indicators.chart_patterns import PATTERN_DIRECTION as _PAT_DIR
from ..indicators.overlays import ema as _ema_series
from .indicators import HeikinAshi, Ind, SRZones, col_of


# ---------------------------------------------------------------------------
# Cond: a named, composable boolean predicate over a prepared DataFrame
# ---------------------------------------------------------------------------
class Cond:
    """A boolean predicate `df -> np.ndarray[bool]`, composable with & | ~."""

    def __init__(self, fn: Callable[[pd.DataFrame], np.ndarray], name: str = ""):
        self._fn = fn
        self.name = name

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._fn(df), dtype=bool)

    def __and__(self, other: "Cond") -> "Cond":
        return Cond(lambda df: self(df) & other(df), f"({self.name} & {other.name})")

    def __or__(self, other: "Cond") -> "Cond":
        return Cond(lambda df: self(df) | other(df), f"({self.name} | {other.name})")

    def __invert__(self) -> "Cond":
        return Cond(lambda df: ~self(df), f"~{self.name}")

    def __repr__(self) -> str:
        return f"Cond({self.name})"


def _moving_average(s: pd.Series, length: int, kind: str) -> pd.Series:
    """A moving average of `s` matching TradingView's MA types."""
    n = int(length)
    k = kind.lower()
    if k == "sma":
        return s.rolling(n, min_periods=n).mean()
    if k == "ema":
        return _ema_series(s, n)
    if k in ("rma", "smma", "wilder"):                 # Wilder / SMMA: alpha = 1/n
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    if k == "wma":                                      # linearly weighted (weights 1..n)
        w = np.arange(1, n + 1, dtype=float)
        return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    raise ValueError(f"unknown MA type {kind!r}; use sma/ema/rma/wma")


def _ohlc(src):
    """Return (open, high, low, close) column names for the requested candle source.

    `src` may be 'real' (default OHLC), 'ha' (default `ha_*` columns), or a tagged
    `HeikinAshi` handle (uses that handle's own columns, e.g. HeikinAshi('hax') -> 'hax_open').
    """
    if isinstance(src, HeikinAshi):
        return src.open, src.high, src.low, src.close
    if src == "ha":
        return "ha_open", "ha_high", "ha_low", "ha_close"
    if src in ("", "real", "px"):
        return "open", "high", "low", "close"
    raise ValueError(f"bad candle src {src!r}; use 'real', 'ha', or a HeikinAshi handle")


def _src_tag(src) -> str:
    return "ha" if isinstance(src, HeikinAshi) or src == "ha" else ""


def close_of(src="real") -> str:
    """Close-price column for a candle source: 'real' -> 'close', a HeikinAshi handle -> its close.
    Lets a rule switch candle type in one place: `above(close_of(SRC), trend)`."""
    return _ohlc(src)[3]


def open_of(src="real") -> str:
    """Open-price column for a candle source."""
    return _ohlc(src)[0]


# ---- comparison / state conditions ----------------------------------------
# Every builder accepts a tagged indicator handle OR a raw column name; `col_of` resolves it once
# here (at condition-construction time), so there is zero per-bar cost.
def above(a, b) -> Cond:
    """`a` above `b` (indicator handles, column names or a scalar), per bar."""
    a, b = col_of(a), col_of(b)
    return Cond(lambda df: S.above(df, a, b), f"{a}>{b}")


def below(a, b) -> Cond:
    a, b = col_of(a), col_of(b)
    return Cond(lambda df: S.below(df, a, b), f"{a}<{b}")


def cross_up(a, b) -> Cond:
    a, b = col_of(a), col_of(b)
    return Cond(lambda df: S.cross_up(df, a, b), f"{a}x^{b}")


def cross_down(a, b) -> Cond:
    a, b = col_of(a), col_of(b)
    return Cond(lambda df: S.cross_down(df, a, b), f"{a}xv{b}")


def full_above(level, n: int = 1, *, src="real") -> Cond:
    """The WHOLE candle is above `level` (low > level) for the last `n` bars."""
    level = col_of(level)
    _o, _h, lo, _c = _ohlc(src)
    tag = _src_tag(src)
    return Cond(lambda df: S.last_all_above(df, lo, level, n), f"{tag}full_above({level},{n})")


def full_below(level, n: int = 1, *, src="real") -> Cond:
    """The WHOLE candle is below `level` (high < level) for the last `n` bars."""
    level = col_of(level)
    _o, hi, _l, _c = _ohlc(src)
    tag = _src_tag(src)
    return Cond(lambda df: S.last_all_below(df, hi, level, n), f"{tag}full_below({level},{n})")


def green(n: int = 1, *, src="real") -> Cond:
    """`n` consecutive bullish candles (close > open)."""
    o, _h, _l, c = _ohlc(src)
    tag = _src_tag(src)
    return Cond(lambda df: S.consecutive_green(df, n, o, c), f"{tag}green({n})")


def red(n: int = 1, *, src="real") -> Cond:
    """`n` consecutive bearish candles (close < open)."""
    o, _h, _l, c = _ohlc(src)
    tag = _src_tag(src)
    return Cond(lambda df: S.consecutive_red(df, n, o, c), f"{tag}red({n})")


def ha_green(n: int = 1, *, src="ha") -> Cond:
    return green(n, src=src)


def ha_red(n: int = 1, *, src="ha") -> Cond:
    return red(n, src=src)


def raw(fn: Callable[[pd.DataFrame], np.ndarray], name: str = "raw") -> Cond:
    """Escape hatch: wrap any custom `df -> bool array` as a Cond."""
    return Cond(fn, name)


# ---- volatility / large-move conditions (#3) ------------------------------
def big_candle(mult: float = 2.0, lookback: int = 20, *, use: str = "range",
               src="real") -> Cond:
    """The current candle is 'large' — its size exceeds `mult` × the average size of the prior
    `lookback` candles. `use='range'` measures high-low; `use='body'` measures |close-open|.
    Anti-lookahead: the average uses the PRIOR `lookback` bars (shifted), not the current one.
    """
    o, h, lo, c = _ohlc(src)

    def fn(df):
        size = (df[h] - df[lo]) if use == "range" else (df[c] - df[o]).abs()
        avg = size.rolling(int(lookback), min_periods=int(lookback)).mean().shift(1)
        return (size > mult * avg).to_numpy()

    return Cond(fn, f"big_{use}(>{mult}x{lookback})")


def big_move_up(mult: float = 2.0, lookback: int = 20, *, src="real") -> Cond:
    """A large BULLISH candle: green and its body exceeds `mult` × the prior average body."""
    o, _h, _l, c = _ohlc(src)
    body = big_candle(mult, lookback, use="body", src=src)
    return Cond(lambda df: body(df) & (df[c] > df[o]).to_numpy(), f"big_up(>{mult}x{lookback})")


def big_move_down(mult: float = 2.0, lookback: int = 20, *, src="real") -> Cond:
    """A large BEARISH candle: red and its body exceeds `mult` × the prior average body."""
    o, _h, _l, c = _ohlc(src)
    body = big_candle(mult, lookback, use="body", src=src)
    return Cond(lambda df: body(df) & (df[c] < df[o]).to_numpy(), f"big_down(>{mult}x{lookback})")


# ---- higher-timeframe bias (#5) -------------------------------------------
def htf_ema_col(tf: str, period: int) -> str:
    """Column name produced by `mtf_ema(tf, period)` — e.g. mtf_ema('15min', 50) -> '15min__ema_50'."""
    return f"{tf}__ema_{int(period)}"


# ---- per-rule stop specifications (used by RuleStrategy `Rule(stop=...)`) --
@dataclass(frozen=True)
class StopSpec:
    """A stop-loss method for one rule. Maps onto the engine's per-trade stop model. `mode` mirrors
    BacktestConfig.sl_mode ('entry_pct'|'price_abs'|'ref_col'); `ref_col_long`/`ref_col_short` name
    the structure columns for ref_col stops; a fallback (pct) applies when the ref is unusable."""
    mode: str = "entry_pct"
    value: float = 1.0
    ref_col_long: Optional[str] = None
    ref_col_short: Optional[str] = None
    buffer_pct: float = 0.0
    max_ref_risk_pct: float = 0.0
    fallback_mode: str = "entry_pct"
    fallback_value: float = 1.0


def pct_stop(value: float) -> StopSpec:
    """Stop a fixed % away from entry (e.g. pct_stop(0.6) = 0.6% stop)."""
    return StopSpec(mode="entry_pct", value=float(value))


def ref_col_stop(long_col="swing_last_low", short_col="swing_last_high", *,
                 buffer_pct: float = 0.0, max_ref_risk_pct: float = 0.0,
                 fallback_pct: float = 1.0) -> StopSpec:
    """Structure stop at a level column (e.g. the previous swing low), with a % fallback if the
    level is missing/too far. This is the 'previous low' stop."""
    long_col, short_col = col_of(long_col), col_of(short_col)
    return StopSpec(mode="ref_col", value=0.0, ref_col_long=long_col, ref_col_short=short_col,
                    buffer_pct=buffer_pct, max_ref_risk_pct=max_ref_risk_pct,
                    fallback_mode="entry_pct", fallback_value=float(fallback_pct))


def price_abs_stop(value: float) -> StopSpec:
    """Stop at an absolute price."""
    return StopSpec(mode="price_abs", value=float(value))


# ---------------------------------------------------------------------------
# Indicator specs — declare the columns a strategy needs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicator:
    """A declared indicator; `apply(df)` adds its column(s). `key` de-dups across rules."""
    kind: str
    args: tuple = ()

    @property
    def key(self):
        return (self.kind, self.args)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "ema":
            period, source, smooth = self.args
            out = add_emas(df, [period], source=source)
            if smooth:                       # TradingView-style extra smoothing: EMA of the EMA
                col = f"ema_{int(period)}"
                out[col] = _ema_series(out[col], int(smooth))
            return out
        if self.kind == "ma":
            length, source, ma_type, smooth, smooth_type, name = self.args
            out = df.copy()
            base = _moving_average(out[source], int(length), ma_type)
            if smooth:                       # apply the "Smoothing Line" over "Smoothing Length"
                base = _moving_average(base, int(smooth), smooth_type)
            out[name] = base
            return out
        if self.kind == "sma":
            period, source = self.args
            return add_smas(df, [period], source=source)
        if self.kind == "heikin_ashi":
            return add_heikin_ashi(df)
        if self.kind == "swings":
            left, right = self.args
            return add_swings(df, left, right)
        if self.kind == "atr":
            (period,) = self.args
            return add_atr(df, period)
        if self.kind == "candle_patterns":
            names, gap_mode, engulf_ratio, body_window, trend_window = self.args
            return add_candle_patterns(df, list(names) if names else None, gap_mode=gap_mode,
                                       engulf_ratio=engulf_ratio, body_window=body_window,
                                       trend_window=trend_window)
        if self.kind == "chart_patterns":
            names, atr_mult, tol, lookback_pivots, max_wait = self.args
            return add_chart_patterns(df, patterns=list(names) if names else None,
                                      atr_mult=atr_mult, tol=tol,
                                      lookback_pivots=lookback_pivots, max_wait=max_wait)
        if self.kind == "mtf_ema":
            tf, period, source = self.args
            p = int(period)
            return build_mtf(df, {tf: lambda d: add_emas(d, [p], source=source)})
        if self.kind == "sr_zones":
            tf, left, right, cluster_pct, max_zones, prefix = self.args
            if tf is None:
                return add_sr_zones(df, left, right, cluster_pct=cluster_pct,
                                    max_zones=max_zones, prefix=prefix)
            # higher-timeframe zones merged onto the base grid (anti-lookahead via build_mtf)
            return build_mtf(df, {tf: lambda d: add_sr_zones(
                d, left, right, cluster_pct=cluster_pct, max_zones=max_zones, prefix=prefix)})
        raise ValueError(f"unknown indicator {self.kind}")


def ema(period: int, *, source: str = "close", smooth: int = 0) -> Indicator:
    """EMA column `ema_{period}` (from `close`, or `ha_close` for a Heikin-Ashi EMA).

    `smooth`>0 applies a second EMA of that length ON TOP of the base EMA (EMA-of-EMA), matching
    TradingView's "Smoothing Length" when its Smoothing Line is set to EMA. Leave 0 for a plain EMA.
    """
    return Indicator("ema", (int(period), source, int(smooth)))


def ma(length: int, *, type: str = "sma", source: str = "close",
       smooth: int = 0, smooth_type: str = "ema", name: Optional[str] = None) -> Indicator:
    """A fully-configurable moving average matching TradingView's "MA" dialog, one field per input:

        Length          -> length            Source          -> source
        (base MA type)  -> type (sma/ema/rma/wma)
        Smoothing Line  -> smooth_type        Smoothing Length -> smooth (0 disables smoothing)

    Column name defaults to `ma_{length}` (override with `name=`). Example matching the screenshot
    (Length 20, Smoothing Line EMA, Smoothing Length 9) with an SMA base:
        ma(20, type='sma', smooth=9, smooth_type='ema')   # -> column 'ma_20'
    """
    return Indicator("ma", (int(length), source, type, int(smooth), smooth_type,
                            name or f"ma_{int(length)}"))


def mtf_ema(tf: str, period: int, *, source: str = "close") -> Indicator:
    """Higher-timeframe EMA merged onto the base grid (anti-lookahead), column `{tf}__ema_{period}`.
    e.g. mtf_ema('15min', 50) -> use as a bias filter via `above('close', '15min__ema_50')`."""
    return Indicator("mtf_ema", (tf, int(period), source))


def sr_zones(*, tf: Optional[str] = None, left: int = 10, right: int = 10, cluster_pct: float = 0.15,
             max_zones: int = 128, prefix: str = "sr") -> Indicator:
    """Swing-cluster support/resistance zones (anti-lookahead). Adds `{prefix}_support`,
    `{prefix}_resistance` and their `_touches` strength columns. Pass `tf='1h'` (etc.) for MAJOR
    higher-timeframe levels (columns are then prefixed `{tf}__`). Use with `near_support`,
    `bounce_off_support`, and the `min_touches` filter for major-vs-minor."""
    return Indicator("sr_zones", (tf, int(left), int(right), float(cluster_pct),
                                  int(max_zones), prefix))


def _sr_col(name: str, tf: Optional[str], prefix: str, zones=None) -> str:
    """Resolve an S/R column. A tagged `SRZones` handle supplies its own columns directly."""
    if zones is not None:
        return getattr(zones, name)
    return f"{tf}__{prefix}_{name}" if tf else f"{prefix}_{name}"


def near_support(within_pct: float = 0.1, *, zones=None, min_touches: int = 1, tf: Optional[str] = None,
                 prefix: str = "sr") -> Cond:
    """Price is within `within_pct`% ABOVE the nearest support zone (of at least `min_touches`
    strength) — i.e. price has come down into the zone."""
    scol = _sr_col("support", tf, prefix, zones)
    tcol = _sr_col("support_touches", tf, prefix, zones)

    def fn(df):
        s = df[scol].to_numpy(); c = df["close"].to_numpy(); t = df[tcol].to_numpy()
        with np.errstate(invalid="ignore"):
            close_enough = (c >= s) & (c <= s * (1.0 + within_pct / 100.0))
        return close_enough & (t >= min_touches) & ~np.isnan(s)

    return Cond(fn, f"near_support(<{within_pct}%,>= {min_touches}t)")


def near_resistance(within_pct: float = 0.1, *, zones=None, min_touches: int = 1, tf: Optional[str] = None,
                    prefix: str = "sr") -> Cond:
    """Price is within `within_pct`% BELOW the nearest resistance zone."""
    rcol = _sr_col("resistance", tf, prefix, zones)
    tcol = _sr_col("resistance_touches", tf, prefix, zones)

    def fn(df):
        r = df[rcol].to_numpy(); c = df["close"].to_numpy(); t = df[tcol].to_numpy()
        with np.errstate(invalid="ignore"):
            close_enough = (c <= r) & (c >= r * (1.0 - within_pct / 100.0))
        return close_enough & (t >= min_touches) & ~np.isnan(r)

    return Cond(fn, f"near_resistance(<{within_pct}%,>= {min_touches}t)")


def bounce_off_support(within_pct: float = 0.1, *, zones=None, min_touches: int = 1, tf: Optional[str] = None,
                       prefix: str = "sr", src: str = "real") -> Cond:
    """A bullish rejection at support: this candle's LOW dipped into the support zone but it CLOSED
    back above the zone (green-ish rejection). A tradeable 'bounce off support'."""
    _o, _h, lo, c = _ohlc(src)
    scol = _sr_col("support", tf, prefix, zones)
    tcol = _sr_col("support_touches", tf, prefix, zones)

    def fn(df):
        s = df[scol].to_numpy(); low = df[lo].to_numpy(); close = df[c].to_numpy(); t = df[tcol].to_numpy()
        with np.errstate(invalid="ignore"):
            touched = low <= s * (1.0 + within_pct / 100.0)
            closed_above = close > s
        return touched & closed_above & (t >= min_touches) & ~np.isnan(s)

    return Cond(fn, f"bounce_off_support(>= {min_touches}t)")


def bounce_off_resistance(within_pct: float = 0.1, *, zones=None, min_touches: int = 1, tf: Optional[str] = None,
                          prefix: str = "sr", src: str = "real") -> Cond:
    """A bearish rejection at resistance: this candle's HIGH poked into the resistance zone but it
    CLOSED back below it."""
    _o, hi, _l, c = _ohlc(src)
    rcol = _sr_col("resistance", tf, prefix, zones)
    tcol = _sr_col("resistance_touches", tf, prefix, zones)

    def fn(df):
        r = df[rcol].to_numpy(); high = df[hi].to_numpy(); close = df[c].to_numpy(); t = df[tcol].to_numpy()
        with np.errstate(invalid="ignore"):
            touched = high >= r * (1.0 - within_pct / 100.0)
            closed_below = close < r
        return touched & closed_below & (t >= min_touches) & ~np.isnan(r)

    return Cond(fn, f"bounce_off_resistance(>= {min_touches}t)")


# ---- breakdown / breakout conditions (for scenarios) ----------------------
def broke_below_support(*, zones=None, min_touches: int = 1, tf: Optional[str] = None,
                        prefix: str = "sr", src: str = "real") -> Cond:
    """Price BREAKS DOWN through a support zone: the candle closes below the support level that was
    holding on the PRIOR bar. (Compares to prior-bar support because the zone table steps to the
    next-lower zone the moment price closes below it.)"""
    _o, _h, _l, c = _ohlc(src)
    scol = _sr_col("support", tf, prefix, zones)
    tcol = _sr_col("support_touches", tf, prefix, zones)

    def fn(df):
        s_prev = df[scol].shift(1).to_numpy()
        t_prev = df[tcol].shift(1).to_numpy()
        close = df[c].to_numpy()
        with np.errstate(invalid="ignore"):
            broke = (close < s_prev) & (t_prev >= min_touches) & ~np.isnan(s_prev)
        return broke

    return Cond(fn, f"broke_below_support(>= {min_touches}t)")


def broke_above_resistance(*, zones=None, min_touches: int = 1, tf: Optional[str] = None,
                           prefix: str = "sr", src: str = "real") -> Cond:
    """Mirror of broke_below_support: close breaks above the prior-bar resistance level."""
    _o, _h, _l, c = _ohlc(src)
    rcol = _sr_col("resistance", tf, prefix, zones)
    tcol = _sr_col("resistance_touches", tf, prefix, zones)

    def fn(df):
        r_prev = df[rcol].shift(1).to_numpy()
        t_prev = df[tcol].shift(1).to_numpy()
        close = df[c].to_numpy()
        with np.errstate(invalid="ignore"):
            return (close > r_prev) & (t_prev >= min_touches) & ~np.isnan(r_prev)

    return Cond(fn, f"broke_above_resistance(>= {min_touches}t)")


def broke_below_prev_low(n: int = 20, *, src: str = "real") -> Cond:
    """Close breaks below the lowest low of the prior `n` bars (a fresh `n`-bar low). Structure-free
    breakdown trigger for the no-S/R scenario variant."""
    _o, _h, lo, c = _ohlc(src)

    def fn(df):
        prior_min = df[lo].rolling(int(n), min_periods=int(n)).min().shift(1).to_numpy()
        return (df[c].to_numpy() < prior_min) & ~np.isnan(prior_min)

    return Cond(fn, f"broke_below_prev_low({n})")


def n_red_below(n: int, level: str, *, src: str = "real") -> Cond:
    """The last `n` candles are ALL red (close < open) AND fully below `level` (high < level).
    Used to confirm a real breakdown below the EMA before entering."""
    level = col_of(level)
    o, hi, _l, c = _ohlc(src)

    def fn(df):
        red_below = (df[c] < df[o]) & (df[hi] < df[level])
        return (red_below.rolling(int(n), min_periods=int(n)).sum() == int(n)).to_numpy()

    return Cond(fn, f"n_red_below({n},{level})")
    return Indicator("sma", (int(period), source))


def heikin_ashi() -> Indicator:
    """Adds ha_open/ha_high/ha_low/ha_close/ha_green."""
    return Indicator("heikin_ashi")


def swings(left: int = 12, right: int = 12) -> Indicator:
    """Adds swing_last_low / swing_last_high (ref_col structure-stop sources)."""
    return Indicator("swings", (int(left), int(right)))


def atr(period: int = 14) -> Indicator:
    return Indicator("atr", (int(period),))


# ---------------------------------------------------------------------------
# Candlestick / chart patterns
# ---------------------------------------------------------------------------
def candle_patterns(*names: str, gap_mode: str = "none", engulf_ratio: float = 1.0,
                    body_window: int = 20, trend_window: int = 10) -> Indicator:
    """Declare the candlestick columns a rule needs. No names = the whole library.

    `gap_mode='none'` is the default because on a 24/7 market 0.000% of bars gap, so the textbook
    gap tests can never fire (see `quant.indicators.candle_patterns`).
    """
    return Indicator("candle_patterns", (tuple(names), gap_mode, engulf_ratio,
                                         body_window, trend_window))


def chart_patterns(*names: str, atr_mult: float = 2.0, tol: float = 0.12,
                   lookback_pivots: int = 6, max_wait: int = 150) -> Indicator:
    """Declare the chart-pattern columns a rule needs. No names = every shape."""
    return Indicator("chart_patterns", (tuple(names), atr_mult, tol, lookback_pivots, max_wait))


def candle(name: str, *, prefix: str = "cdl") -> Cond:
    """A specific candlestick pattern completed on this bar, e.g. `candle('engulfing_bull')`."""
    if name not in _CDL_DIR:
        raise KeyError(f"unknown candle pattern {name!r}; see quant.indicators.CANDLE_PATTERNS")
    col = f"{prefix}_{name}"
    return Cond(lambda df: df[col].to_numpy(bool), f"cdl:{name}")


def candle_any(*names: str, prefix: str = "cdl") -> Cond:
    """Any of the named candlestick patterns completed on this bar."""
    for n in names:
        if n not in _CDL_DIR:
            raise KeyError(f"unknown candle pattern {n!r}")
    cols = [f"{prefix}_{n}" for n in names]

    def fn(df):
        return np.logical_or.reduce([df[c].to_numpy(bool) for c in cols])

    return Cond(fn, "cdl_any(" + ",".join(names) + ")")


def candle_bias(direction: str = "bull", *, min_n: int = 1, net: bool = False,
                prefix: str = "cdl") -> Cond:
    """Use the whole library at once: at least `min_n` bullish (or bearish) patterns fired.

    With `net=True` the requirement is on the NET score (bullish minus bearish), which suppresses
    bars where conflicting patterns all fire together.
    """
    col = f"{prefix}_score" if net else f"{prefix}_{'bull' if direction == 'bull' else 'bear'}_n"
    sign = 1 if direction == "bull" else -1

    def fn(df):
        v = df[col].to_numpy(float)
        return (sign * v) >= min_n if net else v >= min_n

    return Cond(fn, f"cdl_{direction}{'_net' if net else ''}>={min_n}")


def chart_pattern(name: str, *, prefix: str = "pat") -> Cond:
    """A specific chart pattern CONFIRMED on this bar (its neckline broke)."""
    if name not in _PAT_DIR:
        raise KeyError(f"unknown chart pattern {name!r}; see quant.indicators.CHART_PATTERNS")
    col = f"{prefix}_{name}"
    return Cond(lambda df: df[col].to_numpy(bool), f"pat:{name}")


def chart_pattern_any(direction: Optional[str] = None, *, prefix: str = "pat") -> Cond:
    """Any chart pattern confirmed on this bar, optionally filtered to 'bull' or 'bear'."""
    if direction is None:
        return Cond(lambda df: df[f"{prefix}_signal"].to_numpy(bool), "pat:any")
    want = 1 if direction == "bull" else -1

    def fn(df):
        return df[f"{prefix}_signal"].to_numpy(bool) & (df[f"{prefix}_dir"].to_numpy(int) == want)

    return Cond(fn, f"pat:any_{direction}")


def apply_indicators(df: pd.DataFrame, indicators: Sequence[Indicator]) -> pd.DataFrame:
    """Apply each indicator once (de-duplicated by key), in declared order."""
    out = df
    seen = set()
    for ind in indicators:
        if ind.key in seen:
            continue
        seen.add(ind.key)
        out = ind.apply(out)
    return out


# ---- higher-timeframe conditions (use the anti-lookahead MTF columns) ------
def mtf_col(tf: str, period: int, kind: str = "ema") -> str:
    """Column name for a higher-timeframe indicator: mtf_col('5min', 200) -> '5min__ema_200'."""
    return f"{tf}__{kind}_{int(period)}"


def below_mtf(tf: str, period: int, n: int = 1, *, mode: str = "close", src="real",
              kind: str = "ema") -> Cond:
    """Price is BELOW a higher-timeframe MA for the last `n` bars.

    mode='close' — the last `n` CLOSES are below it.
    mode='full'  — the last `n` candles are ENTIRELY below it (high < level).

    The `{tf}__{kind}_{period}` column must exist; scenarios add it via the `mtf_emas` param, which
    routes through `build_mtf` (each HTF feature is shifted one HTF bar before the merge, so the
    value on a base bar was already closed). Combine freely:

        below_mtf('5min', 200, n=3) & red(2)          # 3 closes below the 5m EMA AND 2 red candles
        (below_mtf('5min', 200) & red(3)) | below_mtf('15min', 200, n=5, mode='full')
    """
    col = mtf_col(tf, period, kind)
    return full_below(col, n=n, src=src) if mode == "full" else below_n(col, n=n, src=src)


def above_mtf(tf: str, period: int, n: int = 1, *, mode: str = "close", src="real",
              kind: str = "ema") -> Cond:
    """Mirror of `below_mtf` — price above the higher-timeframe MA for `n` bars."""
    col = mtf_col(tf, period, kind)
    return full_above(col, n=n, src=src) if mode == "full" else above_n(col, n=n, src=src)


def below_n(level, n: int = 1, *, src="real") -> Cond:
    """The last `n` CLOSES are below `level` (vs `full_below`, which needs the whole candle)."""
    level = col_of(level)
    _o, _h, _l, c = _ohlc(src)

    def fn(df):
        m = (df[c] < df[level])
        return (m.rolling(int(n), min_periods=int(n)).sum() == int(n)).to_numpy()

    return Cond(fn, f"below_n({level},{n})")


def above_n(level, n: int = 1, *, src="real") -> Cond:
    """The last `n` CLOSES are above `level`."""
    level = col_of(level)
    _o, _h, _l, c = _ohlc(src)

    def fn(df):
        m = (df[c] > df[level])
        return (m.rolling(int(n), min_periods=int(n)).sum() == int(n)).to_numpy()

    return Cond(fn, f"above_n({level},{n})")
