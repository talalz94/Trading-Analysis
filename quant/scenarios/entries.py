"""
Entry families as PRIMARY triggers — the axis this project never varied.

Every search here until now used one entry (break of an S/R support under a falling EMA) and varied
~150 combinations of exits around it. That answers "how should I manage this trade", never "is there
anything worth trading". These scenarios put each candidate trigger in the driver's seat so they can
be compared head to head, long and short, on the same stops, targets, costs and windows.

Ten families, each registered long AND short (a one-sided result is a directional bet, not an edge):

    level_event   interaction with a reference level — break / fade / sweep / retest, any of the
                  levels in `quant.indicators.levels` (prev-day, session, weekly, opening range,
                  round numbers, S/R zones, VWAP bands). Covers "level fade", "liquidity sweep"
                  and "opening-range break/fade" through one parameterised trigger.
    bos           break of the last ADAPTIVE zigzag swing — with the trend (BOS) or against it (CHoCH)
    vol_thrust    a volume z-score spike with a directional candle (volume was never used before)
    squeeze       squeeze fires off with momentum in the traded direction
    wt_cross      WaveTrend cross inside the overbought/oversold band
    rsi_div       confirmed RSI divergence (or the zero-lag provisional variant)
    vwap_rev      price stretched N sigma from session VWAP, betting on reversion
    grad_extreme  EMA gradient in its own top/bottom quantile — trend continuation

Episode handling: unless a family says otherwise, the episode boundary is simply "the trigger is not
firing", so a run of consecutive trigger bars produces ONE entry and separated events each get their
own. That is the sane default for event-shaped signals.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..indicators import (add_atr, add_candle_patterns, add_chart_patterns,
                          add_ema_gradient, add_emas, add_prev_day_levels, add_round_levels,
                          add_rsi_divergence, add_rsi_divergence_realtime, add_session_levels,
                          add_sr_zones, add_squeeze_momentum, add_volume_features, add_vwap,
                          add_wavetrend, add_week_levels, add_zigzag_structure, level_events)
from ..strategies.conditions import Cond, raw
from .base import Scenario, register_scenario


class FnScenario(Scenario):
    """A scenario built from plain functions, so a family is a few lines instead of a class.

    `prep(df, p) -> df` adds columns; `ent(p) -> Cond` is the trigger. `rst` defaults to "the trigger
    is not firing", which turns a run of consecutive trigger bars into a single entry.
    """

    def __init__(self, name: str, side: str, prep: Callable, ent: Callable, *,
                 version: str = "v1", description: str = "", defaults: Optional[dict] = None,
                 rst: Optional[Callable] = None, prepare_keys: Optional[tuple] = None):
        self.name, self.version, self.side = name, version, side
        self.description = description
        self.default_params = dict(defaults or {})
        self._prep, self._ent, self._rst = prep, ent, rst
        # Which params change prepare()'s OUTPUT. A sweep can then share one prepared frame across
        # every variant that only differs in trigger/threshold params, which for the compression
        # family is a ~30x saving (the rolling percentile rank over 300k bars dominates everything).
        self.prepare_keys = prepare_keys

    def prepare(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        return self._prep(df, p)

    def entry(self, p: dict) -> Cond:
        return self._ent(p, self.side)

    def reset(self, p: dict) -> Optional[Cond]:
        return self._rst(p, self.side) if self._rst else ~self._ent(p, self.side)


def _both(name: str, prep, ent, *, description: str = "", defaults: Optional[dict] = None,
          rst=None, version: str = "v1", prepare_keys: Optional[tuple] = None):
    """Register the long and the short version of a family under `<name>_long@v1` / `<name>_short@v1`."""
    return tuple(register_scenario(FnScenario(f"{name}_{s}", s, prep, ent, version=version,
                                              description=description, defaults=defaults, rst=rst,
                                              prepare_keys=prepare_keys))
                 for s in ("long", "short"))


# ---------------------------------------------------------------------------
# 1. level_event — any reference level x any interaction
# ---------------------------------------------------------------------------
_LEVEL_DEFAULTS = {"lv_col": "pd_high", "lv_side": "above", "lv_event": "fade",
                   "lv_tol_pct": 0.05, "lv_sweep_bars": 2, "lv_or_bars": 30,
                   "lv_round_step": 1000.0, "lv_sr_left": 10, "lv_sr_right": 10,
                   "lv_sr_cluster_pct": 0.15, "lv_vwap_anchor": "session"}


def _prep_levels(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Add only the level family the requested column belongs to — the screen touches 20 columns but
    a single backtest needs one, and these are the expensive groupbys."""
    col = p["lv_col"]
    out = df
    if col.startswith("pd_"):
        out = add_prev_day_levels(out)
    elif col.startswith("wk_"):
        out = add_week_levels(out)
    elif col.startswith("sess_"):
        out = add_session_levels(out, or_bars=int(p["lv_or_bars"]))
    elif col.startswith("rnd_"):
        out = add_round_levels(out, float(p["lv_round_step"]))
    elif col.startswith("sr_"):
        out = add_sr_zones(out, left=int(p["lv_sr_left"]), right=int(p["lv_sr_right"]),
                           cluster_pct=float(p["lv_sr_cluster_pct"]))
    elif col.startswith("vwap"):
        out = add_vwap(out, anchor=p["lv_vwap_anchor"])
    if col not in out.columns:
        raise KeyError(f"level column {col!r} was not produced; check lv_col")
    # the level in play is worth drawing, and `zone_top@v1` reads it for a stop
    out = out.copy()
    out["break_level"] = out[col].to_numpy(float)
    return out


def _ent_level(p: dict, side: str) -> Cond:
    col, lv_side, ev = p["lv_col"], p["lv_side"], p["lv_event"]
    tol, sb = float(p["lv_tol_pct"]), int(p["lv_sweep_bars"])
    return raw(lambda df: level_events(df, col, tol_pct=tol, side=lv_side,
                                       sweep_bars=sb)[ev],
               f"{ev}@{col}[{lv_side}]")


LEVEL_LONG, LEVEL_SHORT = _both(
    "level_event", _prep_levels, _ent_level, defaults=_LEVEL_DEFAULTS,
    description="interaction with a reference level: break / fade / sweep / retest")


# ---------------------------------------------------------------------------
# 2. bos / choch — break of the last adaptive zigzag swing
# ---------------------------------------------------------------------------
def _prep_bos(df, p):
    return add_zigzag_structure(df, atr_mult=float(p["zz_atr_mult"]),
                                atr_period=int(p["zz_atr_period"]))


def _ent_bos(p, side):
    mode = p["bos_mode"]

    def fn(df):
        c = df["close"].to_numpy(float)
        prev = np.r_[np.nan, c[:-1]]
        trend = df["zz_trend"].to_numpy(float)
        if side == "long":
            lv = df["zz_last_high"].to_numpy(float)
            with np.errstate(invalid="ignore"):
                brk = (c > lv) & (prev <= lv)
            gate = trend > 0 if mode == "bos" else trend < 0     # choch = break AGAINST the trend
        else:
            lv = df["zz_last_low"].to_numpy(float)
            with np.errstate(invalid="ignore"):
                brk = (c < lv) & (prev >= lv)
            gate = trend < 0 if mode == "bos" else trend > 0
        return brk & gate & np.isfinite(lv)

    return raw(fn, f"{mode}_{side}")


BOS_LONG, BOS_SHORT = _both(
    "bos", _prep_bos, _ent_bos,
    defaults={"zz_atr_mult": 2.5, "zz_atr_period": 14, "bos_mode": "bos"},
    description="break of the last confirmed zigzag swing, with the trend (bos) or against it (choch)")


# ---------------------------------------------------------------------------
# 3. vol_thrust — a volume spike with a directional candle
# ---------------------------------------------------------------------------
def _prep_vol(df, p):
    return add_volume_features(df, lookback=int(p["vt_lookback"]))


def _ent_vol(p, side):
    z, use_tod = float(p["vt_z"]), bool(p["vt_use_tod"])

    def fn(df):
        v = df["vol_rel_tod" if use_tod else "vol_z"].to_numpy(float)
        thr = (float(p["vt_tod_ratio"]) if use_tod else z)
        c, o = df["close"].to_numpy(float), df["open"].to_numpy(float)
        dirn = (c > o) if side == "long" else (c < o)
        with np.errstate(invalid="ignore"):
            return (v > thr) & dirn

    return raw(fn, f"vol_thrust_{side}")


VOL_LONG, VOL_SHORT = _both(
    "vol_thrust", _prep_vol, _ent_vol,
    defaults={"vt_lookback": 50, "vt_z": 3.0, "vt_use_tod": False, "vt_tod_ratio": 3.0},
    description="volume z-score spike (or relative-to-time-of-day) with a candle in the traded direction")


# ---------------------------------------------------------------------------
# 4. squeeze — squeeze releases with momentum in the traded direction
# ---------------------------------------------------------------------------
def _prep_sqz(df, p):
    return add_squeeze_momentum(df, bb_length=int(p["sqz_bb_length"]),
                                kc_length=int(p["sqz_kc_length"]),
                                kc_mult=float(p["sqz_kc_mult"]))


def _ent_sqz(p, side):
    def fn(df):
        on = df["sqz_on"].to_numpy(bool)
        off = df["sqz_off"].to_numpy(bool)
        mom = df["sqz_mom"].to_numpy(float)
        fired = off & np.r_[False, on[:-1]]                 # the bar the squeeze releases
        with np.errstate(invalid="ignore"):
            return fired & ((mom > 0) if side == "long" else (mom < 0))

    return raw(fn, f"squeeze_release_{side}")


SQZ_LONG, SQZ_SHORT = _both(
    "squeeze", _prep_sqz, _ent_sqz,
    defaults={"sqz_bb_length": 20, "sqz_kc_length": 20, "sqz_kc_mult": 1.5},
    description="squeeze fires off, traded in the direction of the momentum histogram")


# ---------------------------------------------------------------------------
# 5. wt_cross — WaveTrend cross inside the OB/OS band
# ---------------------------------------------------------------------------
def _prep_wt(df, p):
    return add_wavetrend(df, channel_length=int(p["wt_channel"]),
                         average_length=int(p["wt_average"]))


def _ent_wt(p, side):
    lim = float(p["wt_level"])

    def fn(df):
        wt2 = df["wt2"].to_numpy(float)
        up = df["wt_cross_up"].to_numpy(bool)
        dn = df["wt_cross_down"].to_numpy(bool)
        with np.errstate(invalid="ignore"):
            return (up & (wt2 < -lim)) if side == "long" else (dn & (wt2 > lim))

    return raw(fn, f"wt_cross_{side}")


WT_LONG, WT_SHORT = _both(
    "wt_cross", _prep_wt, _ent_wt,
    defaults={"wt_channel": 10, "wt_average": 21, "wt_level": 53.0},
    description="WaveTrend fast/slow cross while the slow line is beyond the OB/OS level")


# ---------------------------------------------------------------------------
# 6. rsi_div — confirmed (lagging) or provisional zero-lag divergence
# ---------------------------------------------------------------------------
def _prep_div(df, p):
    if p["div_mode"] == "realtime":
        return add_rsi_divergence_realtime(df, period=int(p["div_rsi"]))
    return add_rsi_divergence(df, period=int(p["div_rsi"]), left=int(p["div_left"]),
                              right=int(p["div_right"]))


def _ent_div(p, side):
    pre = "rsi_rtdiv" if p["div_mode"] == "realtime" else "rsi_div"
    col = f"{pre}_bull" if side == "long" else f"{pre}_bear"
    return raw(lambda df: df[col].to_numpy(bool), f"{col}")


DIV_LONG, DIV_SHORT = _both(
    "rsi_div", _prep_div, _ent_div,
    defaults={"div_rsi": 14, "div_left": 5, "div_right": 5, "div_mode": "confirmed"},
    description="RSI divergence; div_mode='confirmed' lags by div_right bars, 'realtime' is zero-lag "
                "but only ~53% ratified")


# ---------------------------------------------------------------------------
# 7. vwap_rev — stretched from session VWAP, betting on reversion
# ---------------------------------------------------------------------------
def _prep_vwap(df, p):
    return add_vwap(df, anchor=p["vr_anchor"], bands=(1.0, 2.0))


def _ent_vwap(p, side):
    k = float(p["vr_sigma"])

    def fn(df):
        z = df["vwap_z"].to_numpy(float)
        prev = np.r_[np.nan, z[:-1]]
        with np.errstate(invalid="ignore"):
            # long when stretched BELOW vwap and turning back up (and mirror for short)
            return ((z < -k) & (z > prev)) if side == "long" else ((z > k) & (z < prev))

    return raw(fn, f"vwap_rev_{side}")


VWAP_LONG, VWAP_SHORT = _both(
    "vwap_rev", _prep_vwap, _ent_vwap,
    defaults={"vr_anchor": "session", "vr_sigma": 2.0},
    description="price beyond N sigma of session VWAP and starting back toward it")


# ---------------------------------------------------------------------------
# 8. grad_extreme — EMA gradient in its own top/bottom quantile (trend continuation)
# ---------------------------------------------------------------------------
def _prep_grad(df, p):
    out = add_ema_gradient(df, period=int(p["ge_period"]), smooth=int(p["ge_smooth"]),
                           lookback=int(p["ge_lookback"]))
    q = float(p["ge_quantile"])
    w = int(p["ge_window"])
    g = pd.Series(out["grad"].to_numpy(float))
    # EXPANDING quantiles over a trailing window, shifted one bar: the threshold a bar is judged
    # against may only use bars that already closed.
    out["grad_q_hi"] = g.rolling(w, min_periods=w // 4).quantile(q).shift(1).to_numpy()
    out["grad_q_lo"] = g.rolling(w, min_periods=w // 4).quantile(1.0 - q).shift(1).to_numpy()
    return out


def _ent_grad(p, side):
    def fn(df):
        g = df["grad"].to_numpy(float)
        with np.errstate(invalid="ignore"):
            if side == "long":
                return g > df["grad_q_hi"].to_numpy(float)
            return g < df["grad_q_lo"].to_numpy(float)

    return raw(fn, f"grad_extreme_{side}")


GRAD_LONG, GRAD_SHORT = _both(
    "grad_extreme", _prep_grad, _ent_grad,
    defaults={"ge_period": 200, "ge_smooth": 9, "ge_lookback": 5, "ge_quantile": 0.9,
              "ge_window": 2000},
    description="EMA gradient beyond its own trailing quantile — trend continuation")


# ---------------------------------------------------------------------------
# 9. time_entry — the null benchmark: enter every N bars, no signal at all
# ---------------------------------------------------------------------------
def _prep_null(df, p):
    return add_atr(df, 14)


def _ent_null(p, side):
    every = int(p["te_every"])
    return raw(lambda df: (np.arange(len(df)) % every) == 0, f"every_{every}_bars")


NULL_LONG, NULL_SHORT = _both(
    "time_entry", _prep_null, _ent_null, defaults={"te_every": 200},
    description="NULL BENCHMARK: enter on a fixed bar schedule. Any real entry must beat this on the "
                "same stops, targets and costs, or the result belongs to the exit rules.")


ENTRY_FAMILIES = ["level_event", "bos", "vol_thrust", "squeeze", "wt_cross", "rsi_div",
                  "vwap_rev", "grad_extreme", "time_entry"]


# ---------------------------------------------------------------------------
# 10. compression_break — the "quiet range then a big move" setup
# ---------------------------------------------------------------------------
_CMP_DEFAULTS = {"cb_short": 60, "cb_long": 480, "cb_rank_window": 2000,
                 "cb_quiet_q": 0.2, "cb_trigger": "range_break", "cb_ema": 200,
                 "cb_fade": False, "cb_buffer_bps": 0.0}


def _prep_cmp(df, p):
    from ..indicators import add_compression
    out = add_compression(df, short=int(p["cb_short"]), long=int(p["cb_long"]),
                          rank_window=int(p["cb_rank_window"]))
    out = add_emas(out, [int(p["cb_ema"])])
    s = int(p["cb_short"])
    # the range being broken, from bars STRICTLY before this one
    out["cb_hi"] = out["high"].rolling(s, min_periods=s).max().shift(1).to_numpy()
    out["cb_lo"] = out["low"].rolling(s, min_periods=s).min().shift(1).to_numpy()
    out["break_level"] = np.where(np.asarray(p["_side"] == "long"),
                                  out["cb_hi"], out["cb_lo"])
    return out


def _ent_cmp(p, side):
    q = float(p["cb_quiet_q"])
    trig = p["cb_trigger"]
    ema_col = f"ema_{int(p['cb_ema'])}"
    fade = bool(p["cb_fade"])
    buf = float(p["cb_buffer_bps"]) / 10000.0
    want_up = (side == "long") != fade          # a fade takes the opposite side of the break

    def fn(df):
        c = df["close"].to_numpy(float)
        prev = np.r_[np.nan, c[:-1]]
        quiet = df["cmp_pct"].to_numpy(float) <= q
        with np.errstate(invalid="ignore"):
            if trig == "ema_cross":
                e = df[ema_col].to_numpy(float)
                pe = np.r_[np.nan, e[:-1]]
                fired = ((c > e * (1 + buf)) & (prev <= pe) if want_up
                         else (c < e * (1 - buf)) & (prev >= pe))
            else:                                # range_break
                lv = (df["cb_hi"] if want_up else df["cb_lo"]).to_numpy(float)
                fired = ((c > lv * (1 + buf)) & (prev <= lv) if want_up
                         else (c < lv * (1 - buf)) & (prev >= lv))
            return fired & quiet & np.isfinite(c)

    return raw(fn, f"cmp_{trig}_{'fade' if fade else 'go'}_{side}")


CMP_LONG, CMP_SHORT = _both(
    "compression_break", _prep_cmp, _ent_cmp, defaults=_CMP_DEFAULTS,
    prepare_keys=("cb_short", "cb_long", "cb_rank_window", "cb_ema"),
    description="break out of (or fade) a VOLATILITY-COMPRESSED range. `cb_quiet_q` is the trailing "
                "percentile of ATR(short)/ATR(long) below which the market counts as quiet; set it "
                "to 1.0 to disable the compression filter entirely, which is the control that "
                "isolates what the compression itself contributes.")

ENTRY_FAMILIES.append("compression_break")


# ---------------------------------------------------------------------------
# 11/12. candlestick and chart patterns as primary triggers
# ---------------------------------------------------------------------------
_CDL_DEFAULTS = {"cdl_names": None, "cdl_min_n": 1, "cdl_net": False, "cdl_gap_mode": "none",
                 "cdl_engulf_ratio": 1.0, "cdl_body_window": 20, "cdl_trend_window": 10}


def _prep_cdl(df, p):
    names = p.get("cdl_names")
    return add_candle_patterns(df, list(names) if names else None,
                               gap_mode=p["cdl_gap_mode"],
                               engulf_ratio=float(p["cdl_engulf_ratio"]),
                               body_window=int(p["cdl_body_window"]),
                               trend_window=int(p["cdl_trend_window"]))


def _ent_cdl(p, side):
    """Either a named subset of patterns, or the whole library via its bullish/bearish score."""
    names = p.get("cdl_names")
    want = "bull" if side == "long" else "bear"
    if names:
        cols = [f"cdl_{n}" for n in names]
        return raw(lambda df: np.logical_or.reduce([df[c].to_numpy(bool) for c in cols]),
                   f"cdl_any({len(cols)})_{side}")
    col = "cdl_score" if p["cdl_net"] else f"cdl_{want}_n"
    sign = 1 if want == "bull" else -1
    m = int(p["cdl_min_n"])
    net = bool(p["cdl_net"])

    def fn(df):
        v = df[col].to_numpy(float)
        return (sign * v >= m) if net else (v >= m)

    return raw(fn, f"cdl_{want}>={m}")


CDL_LONG, CDL_SHORT = _both(
    "candle_pattern", _prep_cdl, _ent_cdl, defaults=_CDL_DEFAULTS,
    prepare_keys=("cdl_names", "cdl_gap_mode", "cdl_engulf_ratio", "cdl_body_window",
                  "cdl_trend_window"),
    description="candlestick patterns as the trigger: a named subset, or the whole library through "
                "its bull/bear score")


_PAT_DEFAULTS = {"pat_names": None, "pat_sensitivity": "normal", "pat_settings": None,
                 "pat_atr_mult": 3.5, "pat_lookback_pivots": 24, "pat_max_wait": 150}


def _prep_pat(df, p):
    names = p.get("pat_names")
    out = add_chart_patterns(df, patterns=list(names) if names else None,
                             sensitivity=p["pat_sensitivity"],
                             settings=p.get("pat_settings"),
                             atr_mult=float(p["pat_atr_mult"]),
                             lookback_pivots=int(p["pat_lookback_pivots"]),
                             max_wait=int(p["pat_max_wait"]))
    # the shape's own protective level, carried per bar so `pattern_stop@v1` can read it
    out["break_level"] = out["pat_entry_ff"].to_numpy(float)
    return out


def _ent_pat(p, side):
    names = p.get("pat_names")
    want = 1 if side == "long" else -1
    if names:
        cols = [f"pat_{n}" for n in names]

        def fn(df):
            hit = np.logical_or.reduce([df[c].to_numpy(bool) for c in cols])
            return hit & (df["pat_dir"].to_numpy(int) == want)

        return raw(fn, f"pat_any({len(cols)})_{side}")

    def fn2(df):
        return df["pat_signal"].to_numpy(bool) & (df["pat_dir"].to_numpy(int) == want)

    return raw(fn2, f"pat_any_{side}")


PAT_LONG, PAT_SHORT = _both(
    "chart_pattern", _prep_pat, _ent_pat, defaults=_PAT_DEFAULTS,
    prepare_keys=("pat_names", "pat_sensitivity", "pat_settings", "pat_atr_mult",
                  "pat_lookback_pivots", "pat_max_wait"),
    description="classical chart patterns (double top/bottom, H&S, wedges, triangles, flags) fired "
                "on the bar their neckline BREAKS, never when the shape first looks complete")

ENTRY_FAMILIES += ["candle_pattern", "chart_pattern"]


# ---------------------------------------------------------------------------
# 11. mtf_trendline — a fitted support/resistance line, from any timeframe, giving way
# ---------------------------------------------------------------------------
# The line levels have to be on the frame ALREADY (`quant.indicators.mtf_trendlines` puts them
# there, higher timeframes shifted so a 1h line is only visible once its bar closed). This family
# only reads them, so one run can trade 1m, 5m, 15m and 1h lines together and the merge stays in
# one tested place.
#
# LONG breaks a RESISTANCE line upward; SHORT breaks a SUPPORT line downward. Three conditions,
# combined by `tl_combine`:
#
#   tl_close_n   n consecutive CLOSES beyond the line          (the standard break rule)
#   tl_full_m    m consecutive candles ENTIRELY beyond it      (wick included — much stricter)
#   tl_touch_k   the line had been touched k times before it broke (a quality gate on the line,
#                not a separate trigger: a line nobody respected is not a level)
#   tl_dir_full_n  n consecutive candles that are BOTH the trade's colour AND entirely clear of the
#                line, wicks included. The line is EXTENDED along its own slope to be measured
#                against, because it is retired the moment it breaks (`tl_extend`, on by default).
#                Always the execution frame's candles: it is a geometric test against the line, and
#                the chart frame is what is drawn against it.
#   tl_dir_n     n consecutive candles the trade's OWN colour -- red for a short, green for a long.
#                Not the same test as tl_close_n: a candle can close through a support and still be
#                green. `tl_dir_src` picks whose candles: "chart" (the execution frame, the default)
#                or "line" (the timeframe the line was fitted on).
#
# Any set to 0 is off. `tl_combine="all"` requires every enabled one, "any" requires one.
# `tl_close_n`, `tl_full_m` and `tl_touch_k` may each be ONE number for every timeframe, or a DICT
# keyed by timeframe -- {"5m": 2, "15m": 3} -- because how many candles are worth waiting for is a
# property of the timeframe, not of the strategy: two 15m closes is half an hour and two 1m closes
# is two minutes. `tl_tf_names` maps a column prefix to the name you use in those dicts.
_MTF_TL_DEFAULTS = {
    "tl_prefixes": ("",),          # "" is the base frame; "5m__", "1h__" are merged timeframes
    "tl_tf_names": None,           # {"": "1m", "5m__": "5m", ...}; prefixes work as keys too
    "tl_close_n": 2, "tl_full_m": 0, "tl_touch_k": 0, "tl_dir_n": 0, "tl_dir_full_n": 0,
    "tl_dir_src": "chart",         # "chart" = the frame you execute on; "line" = the line's own
    "tl_extend": True,             # carry a held line along its OWN SLOPE, not flat
    "tl_once_per_line": False,     # True: a given line may trigger only its FIRST entry
    "tl_combine": "all", "tl_tol_atr": 0.0, "tl_atr_period": 14,
}


def tl_per_tf(p: dict, key: str, pref: str) -> int:
    """One knob's value for one timeframe. A plain number applies everywhere; a dict is looked up
    by the timeframe's name, then by its raw column prefix, then by `"*"` as the default."""
    v = p.get(key, 0)
    if not isinstance(v, dict):
        return int(v or 0)
    names = p.get("tl_tf_names") or {}
    for k in (names.get(pref), pref, pref.rstrip("_"), "*"):
        if k is not None and k in v:
            return int(v[k] or 0)
    return 0


def _held_level(df: pd.DataFrame, col: str, hold: int, *, slope_col=None) -> np.ndarray:
    """The line's price, carried `hold` bars past the point it stopped being published.

    A line is retired the moment price closes through it, so any rule that wants more than one bar
    of confirmation has to be told where the line WOULD have been. Holding the last value flat is
    wrong for a sloping line -- a rising support keeps rising, and over the three or four bars a
    confirmation rule spans, a flat hold quietly makes the test easier on one side and harder on the
    other. With `slope_col` the held value is EXTRAPOLATED along the line's own slope, which is what
    "extend the line" means. It reads only values the frame had already published, so it stays causal.
    """
    v = df[col]
    if hold <= 0:
        return v.to_numpy(float)
    if slope_col is None or slope_col not in df.columns:
        return v.ffill(limit=hold).to_numpy(float)

    idx = np.arange(len(df), dtype=float)
    live = np.isfinite(v.to_numpy(float))
    last_v = v.ffill(limit=hold).to_numpy(float)
    last_i = pd.Series(np.where(live, idx, np.nan)).ffill(limit=hold).to_numpy(float)
    slope = df[slope_col].where(live).ffill(limit=hold).to_numpy(float)
    out = last_v + np.where(np.isfinite(slope), slope, 0.0) * (idx - last_i)
    return np.where(live, v.to_numpy(float), out)


def _tl_beyond(df: pd.DataFrame, p: dict, side: str, pref: str):
    """(close beyond, whole candle beyond, the level itself) for one timeframe's line.

    The level is HELD for as many bars as the confirmation rule needs. `add_trendlines` RETIRES a
    line the moment price closes through it -- correct for the indicator, fatal here: asking for two
    closes beyond a line meant asking the second bar to compare itself against a level that no longer
    existed, so on the base timeframe the rule fired ZERO times while the merged higher timeframes,
    whose values persist across their own bar, fired hundreds. Holding the last published value for
    the length of the confirmation window is what makes "n closes beyond it" mean the same thing on
    every timeframe. It reads only past values, so it stays causal.
    """
    which = "res" if side == "long" else "sup"
    col = f"{pref}tl_{which}"
    if col not in df.columns:
        return None
    hold = max(tl_per_tf(p, "tl_close_n", pref), tl_per_tf(p, "tl_full_m", pref),
               tl_per_tf(p, "tl_dir_full_n", pref)) - 1
    lvl = _held_level(df, col, hold,
                      slope_col=f"{pref}tl_{which}_slope" if p.get("tl_extend", True) else None)
    tol = float(p.get("tl_tol_atr") or 0.0)
    if tol:
        a = add_atr(df, int(p.get("tl_atr_period", 14)))[f"atr_{int(p.get('tl_atr_period', 14))}"]
        pad = tol * a.to_numpy(float)
    else:
        pad = 0.0
    c = df["close"].to_numpy(float)
    lo, hi = df["low"].to_numpy(float), df["high"].to_numpy(float)
    if side == "long":
        return c > lvl + pad, lo > lvl + pad, lvl
    return c < lvl - pad, hi < lvl - pad, lvl


def _tl_entry(p: dict, side: str) -> Cond:
    prefs = tuple(p.get("tl_prefixes") or ("",))
    combine_all = str(p.get("tl_combine", "all")).lower() == "all"
    which = "res" if side == "long" else "sup"

    def fn(df: pd.DataFrame) -> np.ndarray:
        fired = np.zeros(len(df), bool)
        for pref in prefs:
            got = _tl_beyond(df, p, side, pref)
            if got is None:
                continue
            # resolved PER TIMEFRAME, so "wait 2 candles on 5m and 3 on 15m" is one configuration
            n = tl_per_tf(p, "tl_close_n", pref)
            m = tl_per_tf(p, "tl_full_m", pref)
            k = tl_per_tf(p, "tl_touch_k", pref)
            dn = tl_per_tf(p, "tl_dir_n", pref)
            dfn = tl_per_tf(p, "tl_dir_full_n", pref)
            beyond_c, beyond_full, lvl = got
            live = np.isfinite(lvl)
            parts = []
            if n > 0:
                parts.append(_run_of(beyond_c, n))
            if m > 0:
                parts.append(_run_of(beyond_full, m))
            if k > 0:
                tcol = f"{pref}tl_{which}_touches"
                parts.append(df[tcol].to_numpy(float) >= k if tcol in df.columns
                             else np.zeros(len(df), bool))
            if dn > 0:
                # CANDLE COLOUR, which is not the same question as "closed beyond the line": a
                # candle can close under a support and still be green. Counted on the timeframe
                # named by `tl_dir_src` -- "line" asks for three red 15m candles, "chart" for three
                # red bars of the frame you actually execute on.
                rp = pref if str(p.get("tl_dir_src", "chart")).lower() == "line" else ""
                rcol = f"{rp}{'up_run' if side == 'long' else 'dn_run'}"
                parts.append(df[rcol].to_numpy(float) >= dn if rcol in df.columns
                             else np.zeros(len(df), bool))
            if dfn > 0:
                # BOTH AT ONCE, on the same candles: the right colour AND the whole candle clear of
                # the line. Asking for `tl_dir_n` and `tl_full_m` separately is the same conjunction
                # only while `tl_combine` is "all"; as one condition it survives "any" and can be
                # OR-ed against something else.
                o = df["open"].to_numpy(float)
                c = df["close"].to_numpy(float)
                right_colour = (c > o) if side == "long" else (c < o)
                parts.append(_run_of(beyond_full & right_colour, dfn))
            if not parts:
                continue
            ok = parts[0].copy()
            for q in parts[1:]:
                ok = (ok & q) if combine_all else (ok | q)
            hit = ok & live
            if p.get("tl_once_per_line", False):
                # ONE TRADE PER LINE. A steeply falling resistance keeps dropping, so a chart-
                # timeframe close can clear it again hours later at a much lower level while the
                # line itself is still live -- its own timeframe never closed through it. Two
                # trades, same line, same name on the chart, which reads as a duplicate. Off by
                # default because it IS two separate breaks; on when you want a line spent once.
                idc = f"{pref}tl_{which}_n"
                if idc in df.columns:
                    ids = df[idc].to_numpy(np.int64)
                    seen, keep = set(), np.zeros(len(hit), bool)
                    for i in np.flatnonzero(hit):
                        k = int(ids[i])
                        if k >= 0 and k in seen:
                            continue          # -1 means no identifiable line: never suppressed
                        seen.add(k)
                        keep[i] = True
                    hit = keep
            fired |= hit
        return fired

    return raw(fn, f"mtf_trendline_{side}")


def _run_of(flags: np.ndarray, n: int) -> np.ndarray:
    """True where `flags` has been true for the last `n` bars INCLUDING this one."""
    if n <= 1:
        return flags.copy()
    out = flags.copy()
    for j in range(1, n):
        out[j:] &= flags[:-j]
        out[:j] = False
    return out


MTF_TRENDLINE = _both(
    "mtf_trendline", lambda df, p: df, _tl_entry,
    description=("break of a FITTED trendline from any timeframe -- long through resistance, short "
                 "through support. Levels must already be on the frame "
                 "(quant.indicators.mtf_trendlines). tl_close_n / tl_full_m / tl_touch_k accept a "
                 "dict keyed by timeframe, so each one can wait a different number of candles."),
    defaults=_MTF_TL_DEFAULTS, prepare_keys=())


def trendline_trigger(df: pd.DataFrame, trades: pd.DataFrame, params: dict,
                      *, prefix_names: Optional[dict] = None) -> pd.DataFrame:
    """Label each trade with WHICH timeframe's line fired on its entry bar.

    The entry is an OR across timeframes, so the engine only knows that something fired. Re-running
    each timeframe's condition on its own and reading it at the entry bars is the cheapest way to
    say which -- and a trade can honestly have more than one.
    """
    if not len(trades):
        return trades.assign(trigger="")
    names = prefix_names or {}
    hits = {}
    for pref in tuple(params.get("tl_prefixes") or ("",)):
        for side in ("long", "short"):
            one = dict(params, tl_prefixes=(pref,))
            hits[(pref, side)] = _tl_entry(one, side)(df)
    held = {}
    for p_ in tuple(params.get("tl_prefixes") or ("",)):
        hold = max(tl_per_tf(params, "tl_close_n", p_), tl_per_tf(params, "tl_full_m", p_),
                   tl_per_tf(params, "tl_dir_full_n", p_)) - 1
        for w in ("res", "sup"):
            col = f"{p_}tl_{w}_n"
            if col not in df.columns:
                continue
            v = df[col].where(df[col] >= 0)
            held[(p_, w)] = (v.ffill(limit=hold) if hold > 0 else v).fillna(-1).to_numpy(int)
    lab, cause = [], []
    for r in trades.itertuples():
        i, side = int(r.entry_i), r.side
        w = "res" if side == "long" else "sup"
        who, lines = [], []
        for p in tuple(params.get("tl_prefixes") or ("",)):
            if not hits[(p, side)][i]:
                continue
            tf = names.get(p, p.rstrip("_") or "chart")
            who.append(tf)
            # WHICH line, not just which timeframe -- `tl_{res,sup}_n` carries the line number that
            # `trendline_table` names R<n> / S<n>, so the trade and the chart agree by construction
            col = f"{p}tl_{w}_n"
            # held for the same window as the LEVEL is, and for the same reason: the line is
            # retired on the bar it breaks, so at the entry bar of an n>1 rule its id is already
            # gone. Without this every base-timeframe trade named its line "?" while the merged
            # higher timeframes, whose columns persist across their own bar, named theirs correctly.
            n = int(held[(p, w)][i]) if col in df.columns else -1
            lines.append(f"{tf}·{'R' if w == 'res' else 'S'}{n}" if n >= 0 else f"{tf}·?")
        lab.append("+".join(who) if who else "?")
        cause.append("+".join(lines) if lines else "?")
    return trades.assign(trigger=lab, cause=cause)
