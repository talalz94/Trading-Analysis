"""
Versioned stop-loss methods for scenarios (short setups).

Each is a small, named, versioned recipe that (a) adds any per-bar reference column it needs and
(b) sets the `BacktestConfig` stop fields. Together they answer the user's Q1 — *where does the stop
go* — as a swept knob: structural (above the swing high / the highest of the last N bars / the broken
support zone) vs tight (a fixed % or an ATR multiple).

Params (flat, read from the merged scenario params):
  stop_swing_lr    swings look-around for `swing_high` (bars each side; default 12)
  stop_n           bars for `high_of_n` (default 20)
  stop_buffer_pct  buffer beyond the structural level, % (default 0.05)
  stop_pct         fixed stop distance for `pct`, % from entry (default 0.6)
  stop_atr_period  ATR period for `atr` (default 14)
  stop_atr_mult    ATR multiple for `atr` (default 1.5)
  stop_max_risk_pct  cap: if a structural stop risks more than this %, fall back to `stop_pct` (0=off)
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..engine import BacktestConfig
from ..indicators import add_atr, add_swings
from .base import StopMethod, register_stop


class _PctStop(StopMethod):
    name, version = "pct", "v1"
    description = "Tight fixed % stop from entry."
    default_params = {"stop_pct": 0.6}

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(cfg, sl_mode="entry_pct", sl_value=float(p["stop_pct"]))


class _SwingHighStop(StopMethod):
    name, version = "swing_high", "v1"
    description = "Structural stop above the last confirmed swing high."
    default_params = {"stop_swing_lr": 12, "stop_buffer_pct": 0.05, "stop_max_risk_pct": 0.0,
                      "stop_pct": 0.6}

    def prepare(self, df, p):
        lr = int(p["stop_swing_lr"])
        return add_swings(df, left=lr, right=lr)

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(
            cfg, sl_mode="ref_col", sl_ref_short_col="swing_last_high",
            sl_buffer_pct=float(p["stop_buffer_pct"]),
            sl_max_ref_risk_pct=float(p.get("stop_max_risk_pct", 0.0)),
            sl_fallback_mode="entry_pct", sl_fallback_value=float(p.get("stop_pct", 0.6)))


class _HighOfNStop(StopMethod):
    name, version = "high_of_n", "v1"
    description = "Structural stop above the highest high of the last N bars."
    default_params = {"stop_n": 20, "stop_buffer_pct": 0.05, "stop_max_risk_pct": 0.0,
                      "stop_pct": 0.6}

    def prepare(self, df, p):
        out = df.copy()
        n = int(p["stop_n"])
        out["stop_high_n"] = out["high"].rolling(n, min_periods=1).max()
        return out

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(
            cfg, sl_mode="ref_col", sl_ref_short_col="stop_high_n",
            sl_buffer_pct=float(p["stop_buffer_pct"]),
            sl_max_ref_risk_pct=float(p.get("stop_max_risk_pct", 0.0)),
            sl_fallback_mode="entry_pct", sl_fallback_value=float(p.get("stop_pct", 0.6)))


class _ZoneTopStop(StopMethod):
    name, version = "zone_top", "v1"
    description = "Stop just above the support zone that broke (now resistance)."
    default_params = {"stop_buffer_pct": 0.05, "stop_max_risk_pct": 0.0, "stop_pct": 0.6}

    def prepare(self, df, p):
        out = df.copy()
        # `break_level` IS the level that broke (latched per episode by the scenario), which is exactly
        # what this stop wants — and every scenario version provides it, including the ones with no S/R
        # zones. Fall back to the prior-bar support only if a scenario didn't supply one.
        if "break_level" in out.columns:
            out["stop_zone_top"] = out["break_level"]
        else:
            out["stop_zone_top"] = out["sr_support"].shift(1)
        return out

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(
            cfg, sl_mode="ref_col", sl_ref_short_col="stop_zone_top",
            sl_buffer_pct=float(p["stop_buffer_pct"]),
            sl_max_ref_risk_pct=float(p.get("stop_max_risk_pct", 0.0)),
            sl_fallback_mode="entry_pct", sl_fallback_value=float(p.get("stop_pct", 0.6)))


class _LegHighStop(StopMethod):
    name, version = "leg_high", "v1"
    description = ("Stop above the high of the CURRENT down-leg — the running max since price last "
                   "closed below the EMA line. No confirmation lag, so it tracks the high a human "
                   "would draw; set stop_above_ema_only to use only highs that poked above the EMA.")
    default_params = {"stop_above_ema_only": False, "stop_buffer_pct": 0.05,
                      "stop_max_risk_pct": 0.0, "stop_pct": 0.6, "grad_prefix": "grad"}

    def prepare(self, df, p):
        out = df.copy()
        line = out[f"{p.get('grad_prefix', 'grad')}_line"].to_numpy(dtype=float)
        close = out["close"].to_numpy(dtype=float)
        high = out["high"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            below = close < line
            vals = np.where(high > line, high, np.nan) if p.get("stop_above_ema_only") else high
        # episode = a contiguous run of "price below the line"; the running max restarts on each one.
        # Unlike a pivot high this needs NO right-hand confirmation, so it is usable the same bar.
        ep = np.cumsum(below & ~np.r_[False, below[:-1]])
        s = pd.Series(np.where(below, vals, np.nan))
        out["leg_high"] = s.groupby(ep).cummax().groupby(ep).ffill().to_numpy()
        return out

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(
            cfg, sl_mode="ref_col", sl_ref_short_col="leg_high",
            sl_buffer_pct=float(p["stop_buffer_pct"]),
            sl_max_ref_risk_pct=float(p.get("stop_max_risk_pct", 0.0)),
            sl_fallback_mode="entry_pct", sl_fallback_value=float(p.get("stop_pct", 0.6)))


class _PrevHighStop(StopMethod):
    name, version = "prev_high", "v1"
    description = ("Stop above the most recent CONFIRMED pivot high (small look-around) — optionally "
                   "only counting highs above the EMA line. NOTE: any pivot needs `stop_lr` bars of "
                   "confirmation to its right, so the level is always >= stop_lr bars old; use "
                   "leg_high@v1 if you want a lag-free 'previous high'.")
    default_params = {"stop_lr": 3, "stop_require_above_ema": False, "stop_buffer_pct": 0.05,
                      "stop_max_risk_pct": 0.0, "stop_pct": 0.6, "grad_prefix": "grad"}

    def prepare(self, df, p):
        out = df.copy()
        lr = int(p["stop_lr"])
        w = 2 * lr + 1
        high = out["high"]
        # pivot high = centre bar is the max of its window; only KNOWN `lr` bars later (anti-lookahead,
        # same convention as add_swings). Small `lr` -> a genuinely *recent* high.
        ph = high.where(high == high.rolling(w, center=True, min_periods=w).max())
        conf = ph.shift(lr)
        if p.get("stop_require_above_ema"):
            # keep only pivots that sat ABOVE the EMA line at the pivot bar (for a short: the highs
            # that actually poked through the trend line, not minor ones underneath it)
            line = out[f"{p.get('grad_prefix', 'grad')}_line"].shift(lr)
            conf = conf.where(conf > line)
        out["prev_high"] = conf.ffill()
        return out

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(
            cfg, sl_mode="ref_col", sl_ref_short_col="prev_high",
            sl_buffer_pct=float(p["stop_buffer_pct"]),
            sl_max_ref_risk_pct=float(p.get("stop_max_risk_pct", 0.0)),
            sl_fallback_mode="entry_pct", sl_fallback_value=float(p.get("stop_pct", 0.6)))


class _AtrStop(StopMethod):
    name, version = "atr", "v1"
    description = "Tight volatility stop: entry + ATR × multiple (above, for a short)."
    default_params = {"stop_atr_period": 14, "stop_atr_mult": 1.5, "stop_pct": 0.6}

    def prepare(self, df, p):
        out = df.copy()
        a = add_atr(out, int(p["stop_atr_period"]))[f"atr_{int(p['stop_atr_period'])}"]
        out["stop_atr"] = out["close"] + float(p["stop_atr_mult"]) * a
        return out

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(
            cfg, sl_mode="ref_col", sl_ref_short_col="stop_atr", sl_buffer_pct=0.0,
            sl_fallback_mode="entry_pct", sl_fallback_value=float(p.get("stop_pct", 0.6)))


register_stop(_PctStop())
register_stop(_SwingHighStop())
register_stop(_LegHighStop())
register_stop(_PrevHighStop())
register_stop(_HighOfNStop())
register_stop(_ZoneTopStop())
register_stop(_AtrStop())
