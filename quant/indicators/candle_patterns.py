"""
Candlestick pattern recognition — fully vectorized, strictly causal.

Every pattern is a boolean column `cdl_<name>` that is True on the bar where the pattern COMPLETES.
A three-candle pattern is flagged on the third candle, never on the first: the pattern is not known
until its last bar closes, and flagging it earlier would be lookahead.

Design notes that matter for correctness:

* **Scale-free thresholds.** Every comparison is a ratio (body vs range, shadow vs body) or is
  measured against a ROLLING AVERAGE body. An absolute price threshold would mean something
  different on BTC at 60,000 and gold at 2,000, and would silently become a date filter as price
  drifts. `avg_body` is the `body_window` mean of the real body, the same convention TA-Lib uses.
* **Trend context.** Several patterns are the SAME SHAPE and differ only by what preceded them — a
  hammer in a downtrend is a hanging man in an uptrend, an inverted hammer is a shooting star. The
  context is `close` versus its `trend_window` SMA, computed on bars strictly before the pattern.
* **Gaps, and why 24/7 markets break half the classical library.** Measured on BTC 15m over 2024:
  **0.000%** of bars gap by range in either direction, and **52%** open at EXACTLY the prior close.
  Two consequences, both handled explicitly rather than silently:
    - textbook gap patterns (kicker, abandoned baby, tasuki, piercing line, dark cloud cover) can
      NEVER fire on a continuous market. `gap_mode='none'` drops the gap requirement and keeps the
      body relationship, which is the form crypto traders actually use;
    - patterns whose test is "opens at the prior close" (identical three crows) are trivially
      satisfied and degenerate into weaker patterns, so they carry extra body constraints here.
  `gap_mode`: 'none' (crypto-usable, the default) | 'body' (bodies must not overlap) | 'range'
  (true high/low gap — the textbook definition, structurally impossible on BTC).

Use them through `quant.strategies.conditions.candle` in rules, or read the columns directly.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# direction of each pattern: +1 bullish, -1 bearish, 0 neutral/context-dependent
PATTERN_DIRECTION: Dict[str, int] = {}
_BUILDERS: Dict[str, callable] = {}


def _register(name: str, direction: int):
    def deco(fn):
        _BUILDERS[name] = fn
        PATTERN_DIRECTION[name] = direction
        return fn
    return deco


class _Anatomy:
    """Per-bar candle geometry plus shifted views, computed once and shared by every pattern."""

    def __init__(self, df: pd.DataFrame, *, body_window: int, trend_window: int,
                 doji_body: float, long_body: float, small_body: float,
                 shadow_ratio: float, equal_tol: float, near_edge: float):
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        self.o, self.h, self.l, self.c = o, h, l, c
        self.n = len(o)
        self.rng = np.maximum(h - l, 1e-12)
        self.body = np.abs(c - o)
        self.top = np.maximum(o, c)
        self.bot = np.minimum(o, c)
        self.up_sh = h - self.top
        self.dn_sh = self.bot - l
        self.green = c > o
        self.red = c < o
        self.mid = (o + c) / 2.0
        self.hl_mid = (h + l) / 2.0
        # rolling average body — the scale reference for "long" and "short" bodies
        self.avg_body = (pd.Series(self.body).rolling(body_window, min_periods=body_window // 2)
                         .mean().to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            self.body_r = self.body / self.rng                     # body as a share of the range
            self.body_vs_avg = self.body / np.where(self.avg_body > 0, self.avg_body, np.nan)
        # classification
        self.is_doji = self.body_r <= doji_body
        self.is_long = self.body_vs_avg >= long_body
        self.is_small = self.body_vs_avg <= small_body
        # trend context from bars STRICTLY before this one
        sma = pd.Series(c).rolling(trend_window, min_periods=trend_window // 2).mean().to_numpy()
        prev_c = np.r_[np.nan, c[:-1]]
        prev_sma = np.r_[np.nan, sma[:-1]]
        with np.errstate(invalid="ignore"):
            self.downtrend = prev_c < prev_sma
            self.uptrend = prev_c > prev_sma
        self.shadow_ratio = shadow_ratio
        self.equal_tol = equal_tol
        self.near_edge = near_edge

    # ---- shifted accessors: p(x, k) = the value k bars ago -------------------
    def p(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        if k <= 0:
            return arr
        k = min(k, len(arr))                         # fewer bars than k: all padding, same length
        pad = np.full(k, np.nan if arr.dtype.kind == "f" else False)
        return np.concatenate([pad, arr[:len(arr) - k]])

    def equal(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Two prices are 'the same level' within `equal_tol` of the average true range."""
        ref = np.maximum(self.avg_body, 1e-12)
        with np.errstate(invalid="ignore"):
            return np.abs(a - b) <= self.equal_tol * ref

    def gap_up(self, mode: str, k: int = 1) -> np.ndarray:
        """This bar sits above bar k back — 'range' = a true gap, 'body' = bodies do not overlap,
        'none' = no gap required (the only workable setting on a continuous 24/7 market)."""
        if mode == "range":
            return self.l > self.p(self.h, k)
        if mode == "body":
            return self.bot > self.p(self.top, k)
        return np.ones(self.n, bool)

    def gap_dn(self, mode: str, k: int = 1) -> np.ndarray:
        if mode == "range":
            return self.h < self.p(self.l, k)
        if mode == "body":
            return self.top < self.p(self.bot, k)
        return np.ones(self.n, bool)

    def opens_below(self, mode: str) -> np.ndarray:
        """The classical "opened below yesterday's low", relaxed to "at or below the prior close"
        when gaps are structurally unavailable."""
        return self.o < self.p(self.l) if mode in ("range", "body") else self.o <= self.p(self.c)

    def opens_above(self, mode: str) -> np.ndarray:
        return self.o > self.p(self.h) if mode in ("range", "body") else self.o >= self.p(self.c)


# ---------------------------------------------------------------------------
# ONE-CANDLE
# ---------------------------------------------------------------------------
@_register("doji", 0)
def _doji(a: _Anatomy, **_):
    return a.is_doji


@_register("long_legged_doji", 0)
def _long_legged_doji(a: _Anatomy, **_):
    return a.is_doji & (a.up_sh > a.body * 2) & (a.dn_sh > a.body * 2)


@_register("dragonfly_doji", +1)
def _dragonfly(a: _Anatomy, **_):
    return a.is_doji & (a.dn_sh >= a.shadow_ratio * a.rng) & (a.up_sh <= 0.1 * a.rng)


@_register("gravestone_doji", -1)
def _gravestone(a: _Anatomy, **_):
    return a.is_doji & (a.up_sh >= a.shadow_ratio * a.rng) & (a.dn_sh <= 0.1 * a.rng)


def _hammer_shape(a: _Anatomy) -> np.ndarray:
    """Small body near the TOP of the range with a long lower shadow."""
    return ((a.dn_sh >= 2.0 * a.body) & (a.up_sh <= a.body) & (~a.is_doji)
            & (a.body_r <= 0.5) & (a.dn_sh >= a.shadow_ratio * a.rng))


def _inv_hammer_shape(a: _Anatomy) -> np.ndarray:
    """Small body near the BOTTOM of the range with a long upper shadow."""
    return ((a.up_sh >= 2.0 * a.body) & (a.dn_sh <= a.body) & (~a.is_doji)
            & (a.body_r <= 0.5) & (a.up_sh >= a.shadow_ratio * a.rng))


@_register("hammer", +1)
def _hammer(a: _Anatomy, **_):
    return _hammer_shape(a) & a.downtrend


@_register("hanging_man", -1)
def _hanging_man(a: _Anatomy, **_):
    return _hammer_shape(a) & a.uptrend          # identical shape, opposite context


@_register("inverted_hammer", +1)
def _inv_hammer(a: _Anatomy, **_):
    return _inv_hammer_shape(a) & a.downtrend


@_register("shooting_star", -1)
def _shooting_star(a: _Anatomy, **_):
    return _inv_hammer_shape(a) & a.uptrend


@_register("marubozu_bull", +1)
def _marubozu_bull(a: _Anatomy, **_):
    return a.green & a.is_long & (a.up_sh <= 0.05 * a.rng) & (a.dn_sh <= 0.05 * a.rng)


@_register("marubozu_bear", -1)
def _marubozu_bear(a: _Anatomy, **_):
    return a.red & a.is_long & (a.up_sh <= 0.05 * a.rng) & (a.dn_sh <= 0.05 * a.rng)


@_register("spinning_top_bull", +1)
def _spin_bull(a: _Anatomy, **_):
    return (a.green & (a.body_r <= 0.35) & (~a.is_doji)
            & (a.up_sh > a.body) & (a.dn_sh > a.body))


@_register("spinning_top_bear", -1)
def _spin_bear(a: _Anatomy, **_):
    return (a.red & (a.body_r <= 0.35) & (~a.is_doji)
            & (a.up_sh > a.body) & (a.dn_sh > a.body))


@_register("belt_hold_bull", +1)
def _belt_bull(a: _Anatomy, **_):
    return a.green & a.is_long & (a.dn_sh <= 0.05 * a.rng) & a.downtrend


@_register("belt_hold_bear", -1)
def _belt_bear(a: _Anatomy, **_):
    return a.red & a.is_long & (a.up_sh <= 0.05 * a.rng) & a.uptrend


# ---------------------------------------------------------------------------
# TWO-CANDLE
# ---------------------------------------------------------------------------
@_register("engulfing_bull", +1)
def _engulf_bull(a: _Anatomy, engulf_ratio: float = 1.0, **_):
    """On a gapless market `open <= prior close` is automatic, so engulfing fires far more often
    than in stocks; `engulf_ratio` (body vs the body it engulfs) restores selectivity."""
    return (a.p(a.red) & a.green
            & (a.o <= a.p(a.c)) & (a.c >= a.p(a.o))
            & (a.body >= engulf_ratio * a.p(a.body)))


@_register("engulfing_bear", -1)
def _engulf_bear(a: _Anatomy, engulf_ratio: float = 1.0, **_):
    return (a.p(a.green) & a.red
            & (a.o >= a.p(a.c)) & (a.c <= a.p(a.o))
            & (a.body >= engulf_ratio * a.p(a.body)))


@_register("harami_bull", +1)
def _harami_bull(a: _Anatomy, **_):
    return (a.p(a.red) & a.p(a.is_long) & a.green
            & (a.top <= a.p(a.o)) & (a.bot >= a.p(a.c)))


@_register("harami_bear", -1)
def _harami_bear(a: _Anatomy, **_):
    return (a.p(a.green) & a.p(a.is_long) & a.red
            & (a.top <= a.p(a.c)) & (a.bot >= a.p(a.o)))


@_register("harami_cross_bull", +1)
def _harami_cross_bull(a: _Anatomy, **_):
    return _harami_bull(a) & a.is_doji


@_register("harami_cross_bear", -1)
def _harami_cross_bear(a: _Anatomy, **_):
    return _harami_bear(a) & a.is_doji


@_register("kicker_bull", +1)
def _kicker_bull(a: _Anatomy, gap_mode: str = "body", **_):
    """A bearish candle, then a bullish one that opens ABOVE it with no body overlap."""
    return a.p(a.red) & a.green & (a.o >= a.p(a.o)) & a.gap_up(gap_mode)


@_register("kicker_bear", -1)
def _kicker_bear(a: _Anatomy, gap_mode: str = "body", **_):
    return a.p(a.green) & a.red & (a.o <= a.p(a.o)) & a.gap_dn(gap_mode)


@_register("piercing_line", +1)
def _piercing(a: _Anatomy, gap_mode: str = "none", **_):
    prev_mid = a.p(a.mid)
    return (a.p(a.red) & a.p(a.is_long) & a.green & a.opens_below(gap_mode)
            & (a.c > prev_mid) & (a.c < a.p(a.o)))


@_register("dark_cloud_cover", -1)
def _dark_cloud(a: _Anatomy, gap_mode: str = "none", **_):
    prev_mid = a.p(a.mid)
    return (a.p(a.green) & a.p(a.is_long) & a.red & a.opens_above(gap_mode)
            & (a.c < prev_mid) & (a.c > a.p(a.o)))


@_register("tweezer_bottom", +1)
def _tweezer_bottom(a: _Anatomy, **_):
    return a.p(a.red) & a.green & a.equal(a.l, a.p(a.l)) & a.downtrend


@_register("tweezer_top", -1)
def _tweezer_top(a: _Anatomy, **_):
    return a.p(a.green) & a.red & a.equal(a.h, a.p(a.h)) & a.uptrend


@_register("matching_low", +1)
def _matching_low(a: _Anatomy, **_):
    return a.p(a.red) & a.red & a.equal(a.c, a.p(a.c)) & a.downtrend


@_register("counterattack_bull", +1)
def _counter_bull(a: _Anatomy, **_):
    return a.p(a.red) & a.green & a.equal(a.c, a.p(a.c)) & (a.o < a.p(a.c))


@_register("counterattack_bear", -1)
def _counter_bear(a: _Anatomy, **_):
    return a.p(a.green) & a.red & a.equal(a.c, a.p(a.c)) & (a.o > a.p(a.c))


@_register("on_neck", -1)
def _on_neck(a: _Anatomy, gap_mode: str = "none", **_):
    return (a.p(a.red) & a.p(a.is_long) & a.green & a.opens_below(gap_mode)
            & a.equal(a.c, a.p(a.l)))


@_register("thrusting", -1)
def _thrusting(a: _Anatomy, gap_mode: str = "none", **_):
    prev_mid = a.p(a.mid)
    return (a.p(a.red) & a.p(a.is_long) & a.green & a.opens_below(gap_mode)
            & (a.c > a.p(a.c)) & (a.c < prev_mid))


@_register("separating_lines_bull", +1)
def _sep_bull(a: _Anatomy, **_):
    return a.p(a.red) & a.green & a.equal(a.o, a.p(a.o)) & a.is_long


@_register("separating_lines_bear", -1)
def _sep_bear(a: _Anatomy, **_):
    return a.p(a.green) & a.red & a.equal(a.o, a.p(a.o)) & a.is_long


@_register("inside_bar", 0)
def _inside_bar(a: _Anatomy, **_):
    return (a.h <= a.p(a.h)) & (a.l >= a.p(a.l))


@_register("outside_bar", 0)
def _outside_bar(a: _Anatomy, **_):
    return (a.h > a.p(a.h)) & (a.l < a.p(a.l))


# ---------------------------------------------------------------------------
# THREE-CANDLE AND LONGER
# ---------------------------------------------------------------------------
@_register("morning_star", +1)
def _morning_star(a: _Anatomy, gap_mode: str = "body", **_):
    first_mid = a.p(a.mid, 2)
    return (a.p(a.red, 2) & a.p(a.is_long, 2)
            & a.p(a.is_small, 1) & a.p(a.gap_dn(gap_mode), 1)
            & a.green & (a.c > first_mid))


@_register("evening_star", -1)
def _evening_star(a: _Anatomy, gap_mode: str = "body", **_):
    first_mid = a.p(a.mid, 2)
    return (a.p(a.green, 2) & a.p(a.is_long, 2)
            & a.p(a.is_small, 1) & a.p(a.gap_up(gap_mode), 1)
            & a.red & (a.c < first_mid))


@_register("morning_doji_star", +1)
def _morning_doji_star(a: _Anatomy, **kw):
    return _morning_star(a, **kw) & a.p(a.is_doji, 1)


@_register("evening_doji_star", -1)
def _evening_doji_star(a: _Anatomy, **kw):
    return _evening_star(a, **kw) & a.p(a.is_doji, 1)


@_register("abandoned_baby_bull", +1)
def _ab_baby_bull(a: _Anatomy, **_):
    """A doji fully gapped away on BOTH sides — the strictest star, by full range."""
    gap_dn_into = a.p(a.h, 1) < a.p(a.l, 2)
    gap_up_out = a.l > a.p(a.h, 1)
    return (a.p(a.red, 2) & a.p(a.is_doji, 1) & gap_dn_into & gap_up_out & a.green)


@_register("abandoned_baby_bear", -1)
def _ab_baby_bear(a: _Anatomy, **_):
    gap_up_into = a.p(a.l, 1) > a.p(a.h, 2)
    gap_dn_out = a.h < a.p(a.l, 1)
    return (a.p(a.green, 2) & a.p(a.is_doji, 1) & gap_up_into & gap_dn_out & a.red)


@_register("three_white_soldiers", +1)
def _three_soldiers(a: _Anatomy, **_):
    near_high = a.up_sh <= a.near_edge * a.rng
    return (a.p(a.green, 2) & a.p(a.green, 1) & a.green
            & a.p(a.is_long, 1) & a.is_long
            & (a.c > a.p(a.c, 1)) & (a.p(a.c, 1) > a.p(a.c, 2))
            & (a.o < a.p(a.c, 1)) & (a.o > a.p(a.o, 1))
            & (a.p(a.o, 1) < a.p(a.c, 2)) & (a.p(a.o, 1) > a.p(a.o, 2))
            & near_high & a.p(near_high, 1))


@_register("three_black_crows", -1)
def _three_crows(a: _Anatomy, **_):
    near_low = a.dn_sh <= a.near_edge * a.rng
    return (a.p(a.red, 2) & a.p(a.red, 1) & a.red
            & a.p(a.is_long, 1) & a.is_long
            & (a.c < a.p(a.c, 1)) & (a.p(a.c, 1) < a.p(a.c, 2))
            & (a.o > a.p(a.c, 1)) & (a.o < a.p(a.o, 1))
            & (a.p(a.o, 1) > a.p(a.c, 2)) & (a.p(a.o, 1) < a.p(a.o, 2))
            & near_low & a.p(near_low, 1))


@_register("identical_three_crows", -1)
def _identical_crows(a: _Anatomy, **_):
    """"Opens at the prior close" is trivially true on a continuous market, which would collapse
    this into "three red bars". The long-body requirement is what keeps it selective."""
    return (a.p(a.red, 2) & a.p(a.red, 1) & a.red
            & a.p(a.is_long, 2) & a.p(a.is_long, 1) & a.is_long
            & a.equal(a.o, a.p(a.c, 1)) & a.equal(a.p(a.o, 1), a.p(a.c, 2))
            & (a.c < a.p(a.c, 1)) & (a.p(a.c, 1) < a.p(a.c, 2)))


@_register("three_inside_up", +1)
def _three_inside_up(a: _Anatomy, **_):
    return a.p(_harami_bull(a), 1) & a.green & (a.c > a.p(a.o, 2))


@_register("three_inside_down", -1)
def _three_inside_down(a: _Anatomy, **_):
    return a.p(_harami_bear(a), 1) & a.red & (a.c < a.p(a.o, 2))


@_register("three_outside_up", +1)
def _three_outside_up(a: _Anatomy, **kw):
    return (a.p(_engulf_bull(a, engulf_ratio=kw.get("engulf_ratio", 1.0)), 1)
            & a.green & (a.c > a.p(a.c, 1)))


@_register("three_outside_down", -1)
def _three_outside_down(a: _Anatomy, **kw):
    return (a.p(_engulf_bear(a, engulf_ratio=kw.get("engulf_ratio", 1.0)), 1)
            & a.red & (a.c < a.p(a.c, 1)))


@_register("three_line_strike_bull", +1)
def _tls_bull(a: _Anatomy, **_):
    """Three rising green candles, then one red that engulfs all three. Bullish CONTINUATION."""
    rising = (a.p(a.green, 3) & a.p(a.green, 2) & a.p(a.green, 1)
              & (a.p(a.c, 1) > a.p(a.c, 2)) & (a.p(a.c, 2) > a.p(a.c, 3)))
    return rising & a.red & (a.o > a.p(a.c, 1)) & (a.c < a.p(a.o, 3))


@_register("three_line_strike_bear", -1)
def _tls_bear(a: _Anatomy, **_):
    falling = (a.p(a.red, 3) & a.p(a.red, 2) & a.p(a.red, 1)
               & (a.p(a.c, 1) < a.p(a.c, 2)) & (a.p(a.c, 2) < a.p(a.c, 3)))
    return falling & a.green & (a.o < a.p(a.c, 1)) & (a.c > a.p(a.o, 3))


@_register("rising_three_methods", +1)
def _rising_three(a: _Anatomy, **_):
    """Long green, three small pullback candles held inside its range, then a new high close."""
    inside = ((a.p(a.h, 1) <= a.p(a.h, 4)) & (a.p(a.l, 1) >= a.p(a.l, 4))
              & (a.p(a.h, 2) <= a.p(a.h, 4)) & (a.p(a.l, 2) >= a.p(a.l, 4))
              & (a.p(a.h, 3) <= a.p(a.h, 4)) & (a.p(a.l, 3) >= a.p(a.l, 4)))
    return (a.p(a.green, 4) & a.p(a.is_long, 4) & inside
            & a.p(a.red, 3) & a.p(a.red, 2) & a.p(a.red, 1)
            & a.green & a.is_long & (a.c > a.p(a.c, 4)))


@_register("falling_three_methods", -1)
def _falling_three(a: _Anatomy, **_):
    inside = ((a.p(a.h, 1) <= a.p(a.h, 4)) & (a.p(a.l, 1) >= a.p(a.l, 4))
              & (a.p(a.h, 2) <= a.p(a.h, 4)) & (a.p(a.l, 2) >= a.p(a.l, 4))
              & (a.p(a.h, 3) <= a.p(a.h, 4)) & (a.p(a.l, 3) >= a.p(a.l, 4)))
    return (a.p(a.red, 4) & a.p(a.is_long, 4) & inside
            & a.p(a.green, 3) & a.p(a.green, 2) & a.p(a.green, 1)
            & a.red & a.is_long & (a.c < a.p(a.c, 4)))


@_register("upside_tasuki_gap", +1)
def _tasuki_up(a: _Anatomy, gap_mode: str = "body", **_):
    return (a.p(a.green, 2) & a.p(a.green, 1) & a.p(a.gap_up(gap_mode), 1)
            & a.red & (a.o < a.p(a.c, 1)) & (a.o > a.p(a.o, 1)) & (a.c < a.p(a.o, 1))
            & (a.c > a.p(a.c, 2)))


@_register("downside_tasuki_gap", -1)
def _tasuki_dn(a: _Anatomy, gap_mode: str = "body", **_):
    return (a.p(a.red, 2) & a.p(a.red, 1) & a.p(a.gap_dn(gap_mode), 1)
            & a.green & (a.o > a.p(a.c, 1)) & (a.o < a.p(a.o, 1)) & (a.c > a.p(a.o, 1))
            & (a.c < a.p(a.c, 2)))


@_register("stick_sandwich", +1)
def _stick_sandwich(a: _Anatomy, **_):
    return (a.p(a.red, 2) & a.p(a.green, 1) & a.red & a.equal(a.c, a.p(a.c, 2)))


@_register("advance_block", -1)
def _advance_block(a: _Anatomy, **_):
    """Three green candles with SHRINKING bodies and growing upper shadows — a stalling rally."""
    return (a.p(a.green, 2) & a.p(a.green, 1) & a.green
            & (a.p(a.body, 1) < a.p(a.body, 2)) & (a.body < a.p(a.body, 1))
            & (a.up_sh > a.body) & (a.c > a.p(a.c, 1)) & (a.p(a.c, 1) > a.p(a.c, 2)))


@_register("deliberation", -1)
def _deliberation(a: _Anatomy, **_):
    return (a.p(a.green, 2) & a.p(a.green, 1) & a.green
            & a.p(a.is_long, 2) & a.p(a.is_long, 1) & a.is_small
            & (a.c > a.p(a.c, 1)) & (a.p(a.c, 1) > a.p(a.c, 2)))


@_register("homing_pigeon", +1)
def _homing_pigeon(a: _Anatomy, **_):
    return (a.p(a.red, 1) & a.p(a.is_long, 1) & a.red
            & (a.top <= a.p(a.o, 1)) & (a.bot >= a.p(a.c, 1)) & a.downtrend)


@_register("three_stars_in_the_south", +1)
def _three_stars_south(a: _Anatomy, **_):
    return (a.p(a.red, 2) & a.p(a.red, 1) & a.red & a.p(a.is_long, 2)
            & (a.p(a.dn_sh, 2) > 0) & (a.p(a.l, 1) > a.p(a.l, 2)) & (a.l > a.p(a.l, 1))
            & (a.p(a.body, 1) < a.p(a.body, 2)) & (a.body < a.p(a.body, 1))
            & (a.dn_sh <= a.near_edge * a.rng))


@_register("ladder_bottom", +1)
def _ladder_bottom(a: _Anatomy, **_):
    return (a.p(a.red, 4) & a.p(a.red, 3) & a.p(a.red, 2) & a.p(a.red, 1)
            & (a.p(a.c, 3) < a.p(a.c, 4)) & (a.p(a.c, 2) < a.p(a.c, 3))
            & (a.p(a.up_sh, 1) > a.p(a.body, 1)) & a.green & (a.o > a.p(a.o, 1)))


@_register("hikkake_bull", +1)
def _hikkake_bull(a: _Anatomy, **_):
    """Inside bar, then a FALSE downside break, then price reclaims the inside bar's high."""
    inside = a.p(_inside_bar(a), 2)
    false_break = a.p(a.l, 1) < a.p(a.l, 2)
    return inside & false_break & (a.h > a.p(a.h, 2))


@_register("hikkake_bear", -1)
def _hikkake_bear(a: _Anatomy, **_):
    inside = a.p(_inside_bar(a), 2)
    false_break = a.p(a.h, 1) > a.p(a.h, 2)
    return inside & false_break & (a.l < a.p(a.l, 2))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
ALL_PATTERNS: List[str] = list(_BUILDERS)
BULLISH: List[str] = [k for k, v in PATTERN_DIRECTION.items() if v > 0]
BEARISH: List[str] = [k for k, v in PATTERN_DIRECTION.items() if v < 0]


def add_candle_patterns(df: pd.DataFrame, patterns: Optional[Sequence[str]] = None, *,
                        prefix: str = "cdl", body_window: int = 20, trend_window: int = 10,
                        doji_body: float = 0.05, long_body: float = 1.3, small_body: float = 0.6,
                        shadow_ratio: float = 0.6, equal_tol: float = 0.1,
                        near_edge: float = 0.25, gap_mode: str = "none",
                        engulf_ratio: float = 1.0,
                        add_score: bool = True) -> pd.DataFrame:
    """Add one boolean column per candlestick pattern, flagged on the bar the pattern COMPLETES.

    `patterns` selects a subset (default: all ~50). Thresholds:
      `body_window`   bars in the rolling average body that defines long/short
      `trend_window`  SMA length for the up/down context that separates hammer from hanging man
      `doji_body`     body <= this share of the range counts as a doji
      `long_body`     body >= this multiple of the average body is 'long'
      `small_body`    body <= this multiple of the average body is 'small'
      `shadow_ratio`  a 'long' shadow is at least this share of the total range
      `equal_tol`     two prices are 'equal' within this multiple of the average body
      `near_edge`     a close 'near the high/low' leaves at most this share of range as shadow
      `gap_mode`      'none' (no gap required — the ONLY workable setting on 24/7 crypto, where
                      0.000% of bars gap) | 'body' (bodies must not overlap) | 'range' (true gap)
      `engulf_ratio`  an engulfing body must be at least this multiple of the body it engulfs

    With `add_score`, also adds `{prefix}_bull_n` / `{prefix}_bear_n` (how many patterns fired on
    this bar) and `{prefix}_score` (bull minus bear) — a cheap way to use the whole library at once.
    """
    names = list(patterns) if patterns is not None else ALL_PATTERNS
    unknown = [p for p in names if p not in _BUILDERS]
    if unknown:
        raise KeyError(f"unknown candle pattern(s) {unknown}; available: {sorted(_BUILDERS)}")
    a = _Anatomy(df, body_window=body_window, trend_window=trend_window, doji_body=doji_body,
                 long_body=long_body, small_body=small_body, shadow_ratio=shadow_ratio,
                 equal_tol=equal_tol, near_edge=near_edge)
    out = df.copy()
    cols = {}
    for name in names:
        v = _BUILDERS[name](a, gap_mode=gap_mode, engulf_ratio=engulf_ratio)
        cols[f"{prefix}_{name}"] = np.nan_to_num(np.asarray(v, dtype=float), nan=0.0).astype(bool)
    # Drop any columns we are about to write. Without this a second call (e.g. a loader adds every
    # pattern, then a scenario prepares a subset) leaves DUPLICATE names, and `df[col]` then returns
    # a 2-column DataFrame instead of a Series -- which silently breaks every downstream boolean
    # test rather than raising.
    dupes = [c for c in cols if c in out.columns]
    if dupes:
        out = out.drop(columns=dupes)
    out = pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)
    if add_score:
        bull = [f"{prefix}_{p}" for p in names if PATTERN_DIRECTION[p] > 0]
        bear = [f"{prefix}_{p}" for p in names if PATTERN_DIRECTION[p] < 0]
        out[f"{prefix}_bull_n"] = out[bull].sum(axis=1).to_numpy() if bull else 0
        out[f"{prefix}_bear_n"] = out[bear].sum(axis=1).to_numpy() if bear else 0
        out[f"{prefix}_score"] = out[f"{prefix}_bull_n"] - out[f"{prefix}_bear_n"]
    return out


def candle_pattern_table(df: pd.DataFrame, *, prefix: str = "cdl") -> pd.DataFrame:
    """Long-form (time, pattern, direction) table of every hit — for charting and inspection."""
    cols = [c for c in df.columns if c.startswith(f"{prefix}_")
            and c.split(f"{prefix}_", 1)[1] in PATTERN_DIRECTION]
    rows = []
    tcol = "t" if "t" in df.columns else None
    for c in cols:
        name = c.split(f"{prefix}_", 1)[1]
        idx = np.flatnonzero(df[c].to_numpy(bool))
        if not len(idx):
            continue
        rows.append(pd.DataFrame({
            "i": idx,
            "t": (df[tcol].to_numpy()[idx] if tcol else idx),
            "pattern": name,
            "direction": PATTERN_DIRECTION[name],
            "high": df["high"].to_numpy()[idx],
            "low": df["low"].to_numpy()[idx],
        }))
    if not rows:
        return pd.DataFrame(columns=["i", "t", "pattern", "direction", "high", "low"])
    return pd.concat(rows, ignore_index=True).sort_values("i").reset_index(drop=True)
