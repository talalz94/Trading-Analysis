"""
The LONG mirror of the support-break family.

Every result up to now has been short-only, which means the strategy inherits the market's direction:
it printed money in a -46% year and lost ~43% in a +26% one. That is beta, not edge. Mirroring the
setup lets the same *shape* trade both ways, so the setup can finally be judged independently of which
way the market happened to go.

Mirrored piece by piece:
  short: below a falling EMA  + break DOWN through support   -> sell
  long : above a rising  EMA  + break UP   through resistance -> buy

`grad_min` is the mirror of `grad_max` (entry needs grad > grad_min). LEVEL B is the resistance that
broke, latched per episode exactly as in the short version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import signals as S
from ..indicators import add_ema_gradient, add_heikin_ashi, add_sr_zones, add_swings
from ..strategies.conditions import Cond, above, below, full_above, green, red
from .base import Scenario, register_scenario
from .support_break import _candle_cols, _grad_cols, _GRAD, _SR, _TIMING

_BREAK_L = {"break_mode": "close_above", "break_min_ema_dist_pct": None,
            "break_max_ema_dist_pct": None}


def _n_green_above(n: int, level: str, *, src: str = "real") -> Cond:
    """`n` consecutive GREEN candles all fully ABOVE `level` — mirror of n_red_below."""
    o, _hi, lo, c = ("ha_open", "ha_high", "ha_low", "ha_close") if src == "ha" \
        else ("open", "high", "low", "close")

    def fn(df):
        m = (df[c] > df[o]) & (df[lo] > df[level])
        return (m.rolling(int(n), min_periods=int(n)).sum() == int(n)).to_numpy()

    return Cond(fn, f"n_green_above({n},{level})")


def _latch_break_level_long(out: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Freeze the resistance that broke, held until price closes back below the EMA line."""
    _pre, line = _grad_cols(p)
    raw = out["break_level_raw"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    ln = out[line].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        broke = (close > raw) & ~np.isnan(raw)
        reset = close <= ln
    first = S.arm_reset_trigger(broke, reset)
    ep = np.cumsum(reset)
    out["break_level"] = pd.Series(np.where(first, raw, np.nan)).groupby(ep).ffill().to_numpy()
    return out


def _break_cond_long(p) -> Cond:
    mode = p.get("break_mode", "close_above")
    hi_c, cl_c = _candle_cols(p.get("candle_src", "real"))
    lo_c = "ha_low" if p.get("candle_src") == "ha" else "low"
    mt = int(p.get("sr_min_touches", 1))
    line = f"{p['grad_prefix']}_line"
    lo_d, hi_d = p.get("break_min_ema_dist_pct"), p.get("break_max_ema_dist_pct")

    def fn(df):
        if mode == "full_above":
            lvl = df["break_level"].to_numpy(dtype=float)
            probe = df[lo_c].to_numpy()
            ok = probe > lvl
        else:
            lvl = df["break_level_raw"].to_numpy(dtype=float)
            probe = df[cl_c].to_numpy()
            ok = probe > lvl
        with np.errstate(invalid="ignore"):
            ok = ok & ~np.isnan(lvl)
            if "break_touches" in df.columns:
                t = df["break_touches"].to_numpy(dtype=float)
                ok = ok & (np.nan_to_num(t, nan=0.0) >= mt)
            if lo_d is not None or hi_d is not None:
                lnv = df[line].to_numpy(dtype=float)
                dist = np.abs(lvl - lnv) / lnv * 100.0
                if lo_d is not None:
                    ok = ok & (dist >= float(lo_d))
                if hi_d is not None:
                    ok = ok & (dist <= float(hi_d))
        return ok

    return Cond(fn, f"break_{mode}(B)")


class ResistanceBreakLongV10(Scenario):
    name, version = "resistance_break_long", "v1.0"
    description = ("LONG mirror: close above the smoothed EMA + a break of a resistance zone. "
                   "One entry per episode (re-arms when price closes back below the EMA line).")
    side = "long"
    default_params = {**_GRAD, **_SR, **_TIMING, **_BREAK_L, "grad_min": None}

    def prepare(self, df, p):
        out = add_ema_gradient(df, period=int(p["grad_period"]), smooth=int(p["grad_smooth"]),
                               lookback=int(p["grad_lookback"]), atr_period=int(p["grad_atr"]),
                               prefix=p["grad_prefix"])
        out = add_sr_zones(out, left=int(p["sr_left"]), right=int(p["sr_right"]),
                           cluster_pct=float(p["sr_cluster_pct"]), prefix="sr")
        if p.get("candle_src") == "ha":
            out = add_heikin_ashi(out)
        out["break_level_raw"] = out["sr_resistance"].shift(1)
        out["break_touches"] = out["sr_resistance_touches"].shift(1)
        return _latch_break_level_long(out, p)

    def _context(self, p) -> Cond:
        pre, line = _grad_cols(p)
        ctx = above("close", line)
        if p.get("grad_min") is not None:
            ctx = ctx & above(pre, float(p["grad_min"]))
        return ctx

    def _break(self, p) -> Cond:
        return _break_cond_long(p)

    def entry(self, p) -> Cond:
        return self._context(p) & self._break(p)

    def reset(self, p) -> Cond:
        _pre, line = _grad_cols(p)
        return below("close", line)


class ResistanceBreakLongV12(ResistanceBreakLongV10):
    name, version = "resistance_break_long", "v1.2"
    description = ("LONG mirror, no S/R zones: N green candles above the EMA + a break of the highest "
                   "high of the prior N bars.")
    default_params = {**_GRAD, **_TIMING, **_BREAK_L, "grad_min": None,
                      "prev_high_n": 20, "confirm_green": 2}

    def prepare(self, df, p):
        out = add_ema_gradient(df, period=int(p["grad_period"]), smooth=int(p["grad_smooth"]),
                               lookback=int(p["grad_lookback"]), atr_period=int(p["grad_atr"]),
                               prefix=p["grad_prefix"])
        if p.get("candle_src") == "ha":
            out = add_heikin_ashi(out)
        n = int(p["prev_high_n"])
        out["break_level_raw"] = out["high"].rolling(n, min_periods=n).max().shift(1)
        out = out.drop(columns=["break_touches"], errors="ignore")
        return _latch_break_level_long(out, p)

    def _context(self, p) -> Cond:
        pre, line = _grad_cols(p)
        src = p.get("candle_src", "real")
        ctx = full_above(line, n=1, src=src) & _n_green_above(int(p["confirm_green"]), line, src=src)
        if p.get("grad_min") is not None:
            ctx = ctx & above(pre, float(p["grad_min"]))
        return ctx


register_scenario(ResistanceBreakLongV10())
register_scenario(ResistanceBreakLongV12())
