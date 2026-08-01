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

from ..indicators import (add_atr, add_ema_gradient, add_prev_day_levels, add_round_levels,
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
                 rst: Optional[Callable] = None):
        self.name, self.version, self.side = name, version, side
        self.description = description
        self.default_params = dict(defaults or {})
        self._prep, self._ent, self._rst = prep, ent, rst

    def prepare(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        return self._prep(df, p)

    def entry(self, p: dict) -> Cond:
        return self._ent(p, self.side)

    def reset(self, p: dict) -> Optional[Cond]:
        return self._rst(p, self.side) if self._rst else ~self._ent(p, self.side)


def _both(name: str, prep, ent, *, description: str = "", defaults: Optional[dict] = None,
          rst=None, version: str = "v1"):
    """Register the long and the short version of a family under `<name>_long@v1` / `<name>_short@v1`."""
    return tuple(register_scenario(FnScenario(f"{name}_{s}", s, prep, ent, version=version,
                                              description=description, defaults=defaults, rst=rst))
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
