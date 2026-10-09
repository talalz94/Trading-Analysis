"""
What can fire an alert -- one Detector per KIND. Each reads ONE timeframe's FINAL candles and says
on which of them it fired. Every kind wraps an indicator from quant/ that is causal (tested by
truncation), so a detector run on candles ending at bar i says exactly what was knowable at bar i's
close: nothing repaints. tests/test_alerts.py checks that again for every kind registered here.

A kind declares its SETTINGS with their defaults (`params`); a setting it does not know is refused
by name, so a typo cannot silently fall back to a default. `python -m quant.alerts list` prints the
whole menu from these declarations, which is why the catalogue cannot drift from the code.

To add a kind: subclass Detector, fill in kind / category / summary / params / example, write
`check` (optional), `warmup_bars` and `evaluate`, and add it to REGISTRY. The engine never changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CATEGORY_ORDER = ("Candle patterns", "Chart patterns", "Levels & lines", "Indicators", "Setups")
DIRECTIONS = ("both", "up", "down")


class Detector(ABC):
    kind: str = ""
    category: str = "Alerts"          # the email section, unless the alert names its own
    summary: str = ""                 # one line for the catalogue
    params: Dict[str, Tuple[object, str]] = {}   # setting -> (default, what it means)
    example: Dict[str, object] = {}   # the kind's own settings in a ready-to-paste alert

    def settings(self, params: dict) -> dict:
        return {k: params.get(k, d) for k, (d, _) in self.params.items()}

    def validate(self, params: dict) -> None:
        unknown = sorted(set(params) - set(self.params))
        if unknown:
            raise KeyError(f"'{self.kind}' alerts have no setting {unknown}; "
                           f"they take {sorted(self.params) or 'none'}")
        p = self.settings(params)
        if "direction" in p and p["direction"] not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {p['direction']!r}")
        self.check(p)

    def check(self, p: dict) -> None:
        """Kind-specific checks on the merged settings; raise KeyError / ValueError."""

    @abstractmethod
    def warmup_bars(self, params: dict) -> int:
        """Closed bars of history needed before the bar it judges."""

    @abstractmethod
    def evaluate(self, frame: pd.DataFrame, params: dict, tf: str) -> List[Dict[str, object]]:
        """Every row of `frame` it fires on: dicts with `i` (row), `what` (a short phrase),
        `side` ('bull' / 'bear' / '') and optionally `note` (a level, a target)."""


# ── helpers ─────────────────────────────────────────────────────────────────────────────────
def _want(p, up: bool) -> bool:
    return p.get("direction", "both") in ("both", "up" if up else "down")


def _hits(mask, what, side, notes=None):
    idx = np.flatnonzero(np.asarray(mask, bool))
    return [dict(i=int(i), what=what, side=side,
                 **({"note": notes(i)} if notes is not None else {})) for i in idx]


def _cross_up(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def _cross_dn(a, b):
    return _cross_up(-np.asarray(a, float), -np.asarray(b, float))


def _px(v) -> str:
    v = float(v)
    dp = 5 if abs(v) < 10 else 3 if abs(v) < 1000 else 2 if abs(v) < 10_000 else 1
    return f"{v:,.{dp}f}"


def _pretty(name: str) -> str:
    """engulfing_bull -> bullish engulfing"""
    words = name.split("_")
    side = {"bull": "bullish", "bear": "bearish"}.get(words[-1])
    return f"{side} {' '.join(words[:-1])}" if side else " ".join(words)


def _pattern_list(value, names, direction_of, label):
    if value in ("all", "bullish", "bearish"):
        want = {"all": (-1, 0, 1), "bullish": (1,), "bearish": (-1,)}[value]
        return [n for n in names if direction_of[n] in want]
    value = (value,) if isinstance(value, str) else tuple(value or ())
    bad = [v for v in value if v not in names]
    if bad or not value:
        raise KeyError(f"unknown {label} {bad or value}; see `python -m quant.alerts list`")
    return list(value)


# ══ PATTERNS ════════════════════════════════════════════════════════════════════════════════
class CandleDetector(Detector):
    kind, category = "candle", "Candle patterns"
    summary = "a candlestick pattern completes (62 of them; flagged on the pattern's LAST candle)"
    params = {"patterns": (None, "names, or 'all' / 'bullish' / 'bearish'")}
    example = dict(patterns=("engulfing_bull", "engulfing_bear"))

    def _names(self, p):
        from quant.indicators import CANDLE_PATTERNS, CANDLE_PATTERN_DIRECTION
        return _pattern_list(p["patterns"], CANDLE_PATTERNS, CANDLE_PATTERN_DIRECTION,
                             "candle pattern")

    def check(self, p):
        if p["patterns"] is None:
            raise KeyError("candle alerts need patterns=(...), or 'all' / 'bullish' / 'bearish'")
        self._names(p)

    def warmup_bars(self, params):
        return 60          # a 20-bar average body and a 10-bar trend, plus three-candle patterns

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_candle_patterns, CANDLE_PATTERN_DIRECTION
        pats = self._names(self.settings(params))
        out = add_candle_patterns(frame[["open", "high", "low", "close"]], patterns=pats,
                                  add_score=False)
        side = {1: "bull", -1: "bear", 0: ""}
        hits = []
        for p in pats:
            hits += _hits(out[f"cdl_{p}"], _pretty(p), side[CANDLE_PATTERN_DIRECTION[p]])
        return hits


class ChartDetector(Detector):
    kind, category = "chart", "Chart patterns"
    summary = ("a chart pattern BREAKS OUT: price closes through its neckline / boundary "
               "(22 of them, with the measured-move target)")
    params = {"patterns": ("all", "names, or 'all' / 'bullish' / 'bearish'"),
              "sensitivity": ("normal", "major | normal | minor -- bigger or smaller swings")}
    example = dict(patterns=("double_top", "double_bottom", "head_shoulders",
                             "inverse_head_shoulders"))

    def _names(self, p):
        from quant.indicators import CHART_PATTERNS, CHART_PATTERN_DIRECTION
        return _pattern_list(p["patterns"], CHART_PATTERNS, CHART_PATTERN_DIRECTION,
                             "chart pattern")

    def check(self, p):
        self._names(p)
        if p["sensitivity"] not in ("major", "normal", "minor"):
            raise ValueError("sensitivity must be major | normal | minor")

    def warmup_bars(self, params):
        return 400         # 24 zigzag pivots, and a shape may wait 150 bars for its breakout

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_chart_patterns, CHART_PATTERN_DIRECTION
        p = self.settings(params)
        pats = self._names(p)
        out = add_chart_patterns(frame[["open", "high", "low", "close", "volume"]],
                                 sensitivity=p["sensitivity"], patterns=pats)
        side = {1: "bull", -1: "bear", 0: ""}
        name, tgt, stop = out["pat_name"].to_numpy(), out["pat_target"], out["pat_stop"]

        def note(pat):
            return lambda i: (f"target {_px(tgt.iat[i])} · stop {_px(stop.iat[i])}"
                              if name[i] == pat and np.isfinite(tgt.iat[i]) else "")
        hits = []
        for pat in pats:
            col = out[f"pat_{pat}"].to_numpy(bool)
            d = side[CHART_PATTERN_DIRECTION[pat]]
            if d == "":                               # channels / symmetrical: side of the break
                for i in np.flatnonzero(col):
                    s = {1: "bull", -1: "bear"}.get(int(np.sign(out["pat_dir"].iat[i])), "")
                    hits.append(dict(i=int(i), what=_pretty(pat), side=s, note=note(pat)(i)))
            else:
                hits += _hits(col, _pretty(pat), d, note(pat))
        return hits


# ══ LEVELS & LINES ══════════════════════════════════════════════════════════════════════════
class TrendlineDetector(Detector):
    kind, category = "trendline", "Levels & lines"
    summary = "price closes through a FITTED trendline (support or resistance with N touches)"
    params = {"direction": ("both", "up = through resistance · down = through support"),
              "min_touches": (3, "pivots the line must touch"),
              "swing_size": (3.5, "zigzag pivot size in ATRs: bigger = longer, fewer lines")}
    example = dict(direction="both", min_touches=3)

    def check(self, p):
        if int(p["min_touches"]) < 2:
            raise ValueError("min_touches must be at least 2")

    def warmup_bars(self, params):
        return 500         # a line may live 500 bars

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_trendlines
        p = self.settings(params)
        out = add_trendlines(frame[["open", "high", "low", "close"]].copy(),
                             min_touches=int(p["min_touches"]), atr_mult=float(p["swing_size"]))
        res, sup = out["tl_res"].shift(1), out["tl_sup"].shift(1)
        rt, st = out["tl_res_touches"].shift(1), out["tl_sup_touches"].shift(1)
        hits = []
        if _want(p, True):
            hits += _hits(out["tl_break_up"], "broke a resistance trendline", "bull",
                          lambda i: f"line {_px(res.iat[i])} · {int(rt.iat[i] or 0)} touches"
                          if np.isfinite(res.iat[i]) else "")
        if _want(p, False):
            hits += _hits(out["tl_break_dn"], "broke a support trendline", "bear",
                          lambda i: f"line {_px(sup.iat[i])} · {int(st.iat[i] or 0)} touches"
                          if np.isfinite(sup.iat[i]) else "")
        return hits


class SRZoneDetector(Detector):
    kind, category = "sr_zone", "Levels & lines"
    summary = ("a support / resistance zone (clustered swing highs and lows) is BROKEN, or price "
               "BOUNCES off it")
    params = {"event": ("break", "break | bounce | both"),
              "direction": ("both", "up = resistance broken / support held · down = the mirror"),
              "sensitivity": ("normal", "tight | normal | loose -- fewer, stronger zones to more"),
              "min_touches": (None, "swings a zone needs; None = the tuned value for the tf")}
    example = dict(event="break", sensitivity="normal")

    def check(self, p):
        if p["event"] not in ("break", "bounce", "both"):
            raise ValueError("event must be break | bounce | both")
        if p["sensitivity"] not in ("tight", "normal", "loose"):
            raise ValueError("sensitivity must be tight | normal | loose")

    def warmup_bars(self, params):
        return 500

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_sr_zones, sr_params
        p = self.settings(params)
        sp = sr_params(tf, p["sensitivity"])
        need = int(p["min_touches"] if p["min_touches"] is not None else sp.pop("min_touches", 1))
        sp.pop("min_touches", None)
        out = add_sr_zones(frame[["open", "high", "low", "close"]].copy(), **sp)
        c, h, l = (frame[k].to_numpy(float) for k in ("close", "high", "low"))
        sup = out["sr_support"].shift(1).to_numpy(float)
        res = out["sr_resistance"].shift(1).to_numpy(float)
        sup[out["sr_support_touches"].shift(1).fillna(0).to_numpy() < need] = np.nan
        res[out["sr_resistance_touches"].shift(1).fillna(0).to_numpy() < need] = np.nan
        prev = np.r_[np.nan, c[:-1]]
        with np.errstate(invalid="ignore"):
            events = {
                ("break", True): (c > res) & (prev <= res), ("break", False): (c < sup) & (prev >= sup),
                ("bounce", True): (l <= sup) & (c > sup), ("bounce", False): (h >= res) & (c < res)}
        words = {("break", True): ("broke resistance", res), ("break", False): ("broke support", sup),
                 ("bounce", True): ("bounced off support", sup),
                 ("bounce", False): ("rejected at resistance", res)}
        hits = []
        for (ev, up), mask in events.items():
            if p["event"] in (ev, "both") and _want(p, up):
                what, lvl = words[(ev, up)]
                hits += _hits(mask, what, "bull" if up else "bear",
                              lambda i, lvl=lvl: f"zone {_px(lvl[i])}")
        return hits


class FvgDetector(Detector):
    kind, category = "fvg", "Levels & lines"
    summary = "a fair value gap (three-candle imbalance) FORMS, or a live one gets FILLED"
    params = {"event": ("formed", "formed | filled | both"),
              "direction": ("both", "up = bullish gaps · down = bearish gaps"),
              "min_size_atr": (0.25, "smallest gap, in ATRs"),
              "mitigation": ("ce", "what counts as filled: touch | ce (midpoint) | full")}
    example = dict(event="formed", min_size_atr=0.5)

    def check(self, p):
        if p["event"] not in ("formed", "filled", "both"):
            raise ValueError("event must be formed | filled | both")
        if p["mitigation"] not in ("touch", "ce", "full"):
            raise ValueError("mitigation must be touch | ce | full")

    def warmup_bars(self, params):
        return 500         # a gap lives up to 500 bars; it must be born in the window to fill

    def evaluate(self, frame, params, tf):
        from quant.indicators import fvg_table
        p = self.settings(params)
        g = fvg_table(frame[["open", "high", "low", "close", "t"]].copy(),
                      mitigation=p["mitigation"], min_size_atr=float(p["min_size_atr"]))
        hits = []
        for r in g.itertuples():
            up = r.kind == 1
            if not _want(p, up):
                continue
            side, word = ("bull", "bullish") if up else ("bear", "bearish")
            band = f"gap {_px(r.bot)}–{_px(r.top)}"
            if p["event"] in ("formed", "both"):
                hits.append(dict(i=int(r.born_i), what=f"{word} FVG formed", side=side, note=band))
            if p["event"] in ("filled", "both") and r.end_reason == "filled" and r.dead_i >= 0:
                hits.append(dict(i=int(r.dead_i), what=f"{word} FVG filled", side=side,
                                 note=band))
        return hits


# ══ INDICATORS ══════════════════════════════════════════════════════════════════════════════
class MaCrossDetector(Detector):
    kind, category = "ma_cross", "Indicators"
    summary = ("price closes across a moving average (period=), or a fast MA crosses a slow one "
               "(fast= + slow=: golden / death cross)")
    params = {"ma": ("ema", "ema | sma"),
              "period": (200, "price vs this MA (ignored when fast/slow are set)"),
              "fast": (None, "fast MA period, for an MA-vs-MA cross"),
              "slow": (None, "slow MA period"),
              "direction": ("both", "up = crosses above · down = crosses below")}
    example = dict(ma="ema", period=200)

    def check(self, p):
        if p["ma"] not in ("ema", "sma"):
            raise ValueError("ma must be ema | sma")
        if (p["fast"] is None) != (p["slow"] is None):
            raise ValueError("set both fast and slow, or neither")
        if p["fast"] is not None and int(p["fast"]) >= int(p["slow"]):
            raise ValueError("fast must be shorter than slow")

    def warmup_bars(self, params):
        p = self.settings(params)
        n = int(p["slow"] if p["slow"] is not None else p["period"])
        return 3 * n + 10 if p["ma"] == "ema" else n + 5   # an EMA needs ~3x its period to settle

    def evaluate(self, frame, params, tf):
        from quant.indicators import ema, sma
        p = self.settings(params)
        f = ema if p["ma"] == "ema" else sma
        name = p["ma"].upper()
        c = frame["close"]
        if p["fast"] is not None:
            a, b = f(c, int(p["fast"])), f(c, int(p["slow"]))
            lab = f"{name} {p['fast']} crossed {{}} {name} {p['slow']}"
            tags = (" (golden cross)", " (death cross)")
        else:
            a, b = c, f(c, int(p["period"]))
            lab = f"closed {{}} {name} {p['period']}"
            tags = ("", "")
        bv = b.to_numpy(float)
        note = (lambda i: f"{name} at {_px(bv[i])}")
        hits = []
        if _want(p, True):
            hits += _hits(_cross_up(a, b), lab.format("above") + tags[0], "bull", note)
        if _want(p, False):
            hits += _hits(_cross_dn(a, b), lab.format("below") + tags[1], "bear", note)
        return hits


class RsiDetector(Detector):
    kind, category = "rsi", "Indicators"
    summary = "RSI crosses into overbought / oversold (when='enter'), or back out of it ('exit')"
    params = {"period": (14, "RSI length"), "above": (70, "overbought line"),
              "below": (30, "oversold line"), "when": ("enter", "enter | exit"),
              "direction": ("both", "up = crossing upward · down = crossing downward")}
    example = dict(period=14, above=70, below=30, when="exit")

    def check(self, p):
        if p["when"] not in ("enter", "exit"):
            raise ValueError("when must be enter | exit")
        if not 0 < float(p["below"]) < float(p["above"]) < 100:
            raise ValueError("need 0 < below < above < 100")

    def warmup_bars(self, params):
        return max(100, 10 * int(self.settings(params)["period"]))

    def evaluate(self, frame, params, tf):
        from quant.indicators import rsi
        p = self.settings(params)
        r = rsi(frame["close"], int(p["period"])).to_numpy(float)
        hi, lo = float(p["above"]), float(p["below"])
        n = len(r)
        lvl_hi, lvl_lo = np.full(n, hi), np.full(n, lo)
        note = (lambda i: f"RSI {r[i]:.1f}")
        hits = []
        if p["when"] == "enter":
            if _want(p, True):
                hits += _hits(_cross_up(r, lvl_hi), f"RSI {p['period']} into overbought (>{hi:g})",
                              "bull", note)
            if _want(p, False):
                hits += _hits(_cross_dn(r, lvl_lo), f"RSI {p['period']} into oversold (<{lo:g})",
                              "bear", note)
        else:
            if _want(p, True):
                hits += _hits(_cross_up(r, lvl_lo), f"RSI {p['period']} back out of oversold",
                              "bull", note)
            if _want(p, False):
                hits += _hits(_cross_dn(r, lvl_hi), f"RSI {p['period']} back out of overbought",
                              "bear", note)
        return hits


class RsiDivergenceDetector(Detector):
    kind, category = "rsi_divergence", "Indicators"
    summary = ("price makes a new swing high / low that RSI does not confirm; known `right` "
               "bars after the swing, so it always arrives that late")
    params = {"period": (14, "RSI length"), "left": (5, "bars left of a swing"),
              "right": (5, "bars right of a swing = how late it is known"),
              "direction": ("both", "up = bullish divergence · down = bearish")}
    example = dict(period=14, left=5, right=5)

    def warmup_bars(self, params):
        return 300

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_rsi_divergence
        p = self.settings(params)
        out = add_rsi_divergence(frame[["open", "high", "low", "close"]].copy(),
                                 period=int(p["period"]), left=int(p["left"]),
                                 right=int(p["right"]))
        late = f"confirmed {p['right']} bars after the swing"
        hits = []
        if _want(p, True):
            hits += _hits(out["rsi_div_bull"], "bullish RSI divergence", "bull", lambda i: late)
        if _want(p, False):
            hits += _hits(out["rsi_div_bear"], "bearish RSI divergence", "bear", lambda i: late)
        return hits


class MacdDetector(Detector):
    kind, category = "macd", "Indicators"
    summary = "the MACD line crosses its signal line (cross='signal') or zero (cross='zero')"
    params = {"fast": (12, "fast EMA length"), "slow": (26, "slow EMA length"),
              "signal": (9, "signal line length"),
              "cross": ("signal", "signal | zero"),
              "direction": ("both", "up = bullish cross · down = bearish")}
    example = dict(fast=12, slow=26, signal=9, cross="signal")

    def check(self, p):
        if p["cross"] not in ("signal", "zero"):
            raise ValueError("cross must be signal | zero")

    def warmup_bars(self, params):
        p = self.settings(params)
        return 3 * int(p["slow"]) + 3 * int(p["signal"])

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_macd
        p = self.settings(params)
        out = add_macd(frame[["close"]].copy(), int(p["fast"]), int(p["slow"]), int(p["signal"]))
        m = out["macd"].to_numpy(float)
        ref = out["macd_signal"].to_numpy(float) if p["cross"] == "signal" else np.zeros(len(m))
        word = "its signal line" if p["cross"] == "signal" else "zero"
        hits = []
        if _want(p, True):
            hits += _hits(_cross_up(m, ref), f"MACD crossed above {word}", "bull")
        if _want(p, False):
            hits += _hits(_cross_dn(m, ref), f"MACD crossed below {word}", "bear")
        return hits


class StochasticDetector(Detector):
    kind, category = "stochastic", "Indicators"
    summary = ("%K crosses %D -- upward only while oversold (below `below`), downward only "
               "while overbought (above `above`)")
    params = {"k": (14, "%K lookback"), "d": (3, "%D smoothing"), "smooth": (3, "%K smoothing"),
              "above": (80, "overbought line (100 = any downward cross)"),
              "below": (20, "oversold line (0 = any upward cross)"),
              "direction": ("both", "up = bullish cross · down = bearish")}
    example = dict(k=14, d=3, smooth=3, above=80, below=20)

    def warmup_bars(self, params):
        p = self.settings(params)
        return int(p["k"]) + int(p["d"]) + int(p["smooth"]) + 20

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_stochastic
        p = self.settings(params)
        out = add_stochastic(frame[["high", "low", "close"]].copy(), int(p["k"]), int(p["d"]),
                             int(p["smooth"]))
        k, d = out["stoch_k"].to_numpy(float), out["stoch_d"].to_numpy(float)
        prev_d = np.r_[np.nan, d[:-1]]
        lo, hi = float(p["below"]), float(p["above"])
        any_up, any_dn = lo <= 0 or lo >= 100, hi >= 100 or hi <= 0   # 0 / 100 = no zone test
        note = (lambda i: f"%K {k[i]:.1f}")
        hits = []
        with np.errstate(invalid="ignore"):
            if _want(p, True):
                zone = np.ones(len(k), bool) if any_up else prev_d <= lo
                hits += _hits(_cross_up(k, d) & zone, "stochastic crossed up"
                              + ("" if any_up else " in oversold"), "bull", note)
            if _want(p, False):
                zone = np.ones(len(k), bool) if any_dn else prev_d >= hi
                hits += _hits(_cross_dn(k, d) & zone, "stochastic crossed down"
                              + ("" if any_dn else " in overbought"), "bear", note)
        return hits


class SupertrendDetector(Detector):
    kind, category = "supertrend", "Indicators"
    summary = "Supertrend flips direction"
    params = {"period": (10, "ATR length"), "multiplier": (3.0, "band width in ATRs"),
              "direction": ("both", "up = flips bullish · down = flips bearish")}
    example = dict(period=10, multiplier=3.0)

    def warmup_bars(self, params):
        return 150

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_supertrend
        p = self.settings(params)
        out = add_supertrend(frame[["high", "low", "close"]].copy(), int(p["period"]),
                             float(p["multiplier"]))
        d = out["st_dir"].to_numpy(int)
        prev = np.r_[0, d[:-1]]
        st = out["st"].to_numpy(float)
        note = (lambda i: f"line {_px(st[i])}")
        hits = []
        if _want(p, True):
            hits += _hits((d == 1) & (prev == -1), "Supertrend flipped up", "bull", note)
        if _want(p, False):
            hits += _hits((d == -1) & (prev == 1), "Supertrend flipped down", "bear", note)
        return hits


class HeikinAshiDetector(Detector):
    kind, category = "heikin_ashi", "Indicators"
    summary = "Heikin-Ashi candles change colour after at least `after` candles of the other"
    params = {"after": (3, "candles of the other colour before the flip"),
              "direction": ("both", "up = turns green · down = turns red")}
    example = dict(after=3)

    def warmup_bars(self, params):
        return 50 + int(self.settings(params)["after"])

    def evaluate(self, frame, params, tf):
        from quant.indicators import add_heikin_ashi
        p = self.settings(params)
        out = add_heikin_ashi(frame[["open", "high", "low", "close"]].copy())
        g = out["ha_green"].to_numpy(bool)
        n = int(p["after"])
        run = np.zeros(len(g), int)                 # length of the colour run ending at i-1
        for i in range(1, len(g)):
            run[i] = run[i - 1] + 1 if i > 1 and g[i - 1] == g[i - 2] else 1
        flip = np.r_[False, g[1:] != g[:-1]] & (run >= n)
        hits = []
        if _want(p, True):
            hits += _hits(flip & g, f"Heikin-Ashi turned green after {n}+ red", "bull")
        if _want(p, False):
            hits += _hits(flip & ~g, f"Heikin-Ashi turned red after {n}+ green", "bear")
        return hits


class CandleRunDetector(Detector):
    kind, category = "candle_run", "Indicators"
    summary = "N candles in a row of one colour (fires once, on the Nth)"
    params = {"n": (4, "candles in a row"),
              "direction": ("both", "up = green run · down = red run")}
    example = dict(n=4)

    def warmup_bars(self, params):
        return int(self.settings(params)["n"]) + 5

    def evaluate(self, frame, params, tf):
        from quant.indicators.candles import add_candle_runs
        p = self.settings(params)
        out = add_candle_runs(frame[["open", "high", "low", "close"]].copy())
        n = int(p["n"])
        hits = []
        if _want(p, True):
            hits += _hits(out["up_run"].to_numpy() == n, f"{n} green candles in a row", "bull")
        if _want(p, False):
            hits += _hits(out["dn_run"].to_numpy() == n, f"{n} red candles in a row", "bear")
        return hits


class BigCandleDetector(Detector):
    kind, category = "big_candle", "Indicators"
    summary = "a candle whose high-low range is at least `atr_mult` x the ATR before it"
    params = {"atr_mult": (2.5, "range / ATR"), "atr_period": (14, "ATR length"),
              "direction": ("both", "up = green candle · down = red candle")}
    example = dict(atr_mult=2.5)

    def warmup_bars(self, params):
        return 5 * int(self.settings(params)["atr_period"])

    def evaluate(self, frame, params, tf):
        from quant.indicators import atr
        p = self.settings(params)
        a = atr(frame, int(p["atr_period"])).shift(1).to_numpy(float)   # the ATR BEFORE it
        rng = (frame["high"] - frame["low"]).to_numpy(float)
        green = (frame["close"] >= frame["open"]).to_numpy(bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = rng / a
        big = x >= float(p["atr_mult"])
        note = (lambda i: f"{x[i]:.1f}x ATR")
        hits = []
        if _want(p, True):
            hits += _hits(big & green, "big green candle", "bull", note)
        if _want(p, False):
            hits += _hits(big & ~green, "big red candle", "bear", note)
        return hits


# ══ SETUPS — the research entries in quant/scenarios ═══════════════════════════════════════
_NO_SETUP = ("time_entry", "mtf_trendline")   # a metronome; needs other timeframes' lines


def setup_keys() -> List[str]:
    from quant.scenarios.base import SCENARIOS
    return sorted(k for k in SCENARIOS if k.split("@")[0].rsplit("_", 1)[0] not in _NO_SETUP)


class SetupDetector(Detector):
    kind, category = "setup", "Setups"
    summary = ("a research ENTRY SETUP from quant/scenarios would open a trade here (one per "
               "setup episode, exactly as the backtest takes it)")
    params = {"setup": (None, "a key from the Setups list, e.g. 'bos_long@v1'"),
              "options": ({}, "that setup's own settings to change, as a dict")}
    example = dict(setup="bos_long@v1")

    def _scenario(self, p):
        from quant.scenarios.base import SCENARIOS
        if p["setup"] not in setup_keys():
            raise KeyError(f"unknown setup {p['setup']!r}; see `python -m quant.alerts list`")
        return SCENARIOS[p["setup"]]

    def check(self, p):
        s = self._scenario(p)
        bad = sorted(set(p["options"] or {}) - set(s.default_params))
        if bad:
            raise KeyError(f"{p['setup']} has no option {bad}; it takes {sorted(s.default_params)}")

    def _p(self, params):
        p = self.settings(params)
        s = self._scenario(p)
        q = dict(s.default_params)
        q.update(p["options"] or {})
        q["_side"] = s.side
        return s, q

    def warmup_bars(self, params):
        _, q = self._p(params)
        need = [500]
        for k, v in q.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k.endswith("_window") or k.endswith("_long") or k.endswith("_lookback"):
                    need.append(int(v) + 50)
                elif k in ("grad_period", "ge_period", "cb_ema", "div_rsi"):
                    need.append(3 * int(v) + 10)
        return max(need)

    def evaluate(self, frame, params, tf):
        from quant.strategies.builder import Rule, RuleStrategy
        s, q = self._p(params)
        d = s.prepare(frame.copy(), q)
        rule = Rule("setup", entry=s.entry(q), side=s.side, once=True)
        sig = RuleStrategy(indicators=(), rules=(rule,), reset=s.reset(q)).signals(d)
        e = sig.entry_long if s.side == "long" else sig.entry_short
        name = s.name.replace("_long", "").replace("_short", "").replace("_", " ")
        return _hits(e, f"{name} setup ({s.side})", "bull" if s.side == "long" else "bear")


REGISTRY: Dict[str, Detector] = {d.kind: d for d in (
    CandleDetector(), ChartDetector(),
    TrendlineDetector(), SRZoneDetector(), FvgDetector(),
    MaCrossDetector(), RsiDetector(), RsiDivergenceDetector(), MacdDetector(),
    StochasticDetector(), SupertrendDetector(), HeikinAshiDetector(), CandleRunDetector(),
    BigCandleDetector(),
    SetupDetector(),
)}
