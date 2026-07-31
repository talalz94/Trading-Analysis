"""
Versioned take-profit / exit methods for scenarios (short setups).

Each realises the *target* either as `BacktestConfig` fields (rr / fixed % / trailing / laddered
partials) or as an exit **signal** (N green candles / price back above the EMA line / price into the
next support zone). Together they answer the user's Q3 — *how do we take profit* — as a swept knob.

Params (flat, read from the merged scenario params):
  tp_rr           R-multiple for `rr` and the final leg of `rr_partials` (default 2.0)
  tp_pct          fixed % target for `fixed_pct` (default 0.6)
  trail_pct       trailing-stop distance %, for `trailing` (default 1.0)
  tp_n_green      green candles to exit on, for `n_green_exit` (default 2)
  tp_zone_pct     within-% of the next support zone to exit on, for `next_zone` (default 0.1)
  tp_partial_rr1  first-leg R for `rr_partials` (default 1.0)
  tp_partial_pct1 % of position closed at the first leg, for `rr_partials` (default 50.0)
"""
from __future__ import annotations

import dataclasses

from ..engine import BacktestConfig, TakeProfit
from ..indicators import add_heikin_ashi, add_sr_zones
from ..strategies.conditions import (above, below, green, near_resistance, near_support, red)
from .base import TPMethod, register_tp


class _NoTarget(TPMethod):
    name, version = "none", "v1"
    description = ("No take-profit at all — the trade can only be closed by the stop-loss or by a "
                   "SECONDARY rule. Use this to make a secondary rule the sole exit.")
    default_params = {}

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        # leave tp_mode='none' and take_profits=() exactly as they came in
        return cfg


class _RRTarget(TPMethod):
    name, version = "rr", "v1"
    description = "Fixed reward:risk target (R multiple of the stop distance)."
    default_params = {"tp_rr": 2.0}

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(cfg, tp_mode="rr", tp_value=float(p["tp_rr"]))


class _FixedPctTarget(TPMethod):
    name, version = "fixed_pct", "v1"
    description = "Fixed % profit target from entry (a small locked range)."
    default_params = {"tp_pct": 0.6}

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(cfg, tp_mode="entry_pct", tp_value=float(p["tp_pct"]))


class _Trailing(TPMethod):
    name, version = "trailing", "v1"
    description = "Let it run: trailing % stop that ratchets on the move."
    default_params = {"trail_pct": 1.0}

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return dataclasses.replace(cfg, trail_mode="pct", trail_value=float(p["trail_pct"]))


class _NGreenExit(TPMethod):
    name, version = "n_green_exit", "v1"
    description = "Exit on N consecutive green candles (momentum reversed)."
    default_params = {"tp_n_green": 2}

    def exit(self, p: dict):
        n = int(p["tp_n_green"])
        return red(n) if p.get("_side") == "long" else green(n)


class _BackAboveEma(TPMethod):
    name, version = "back_above_ema", "v1"
    description = "Exit when price closes back above the smoothed EMA line (setup invalidated)."
    default_params = {"grad_prefix": "grad"}

    def exit(self, p: dict):
        line = f"{p.get('grad_prefix', 'grad')}_line"
        return below("close", line) if p.get("_side") == "long" else above("close", line)


class _NextZoneSignal(TPMethod):
    name, version = "next_zone", "v1"
    description = "Exit when price reaches the next support zone below (signal-based target)."
    default_params = {"tp_zone_pct": 0.1, "tp_zone_min_touches": 1}

    def prepare(self, df, p):
        # This TP needs S/R zones; scenarios that don't use zones (e.g. v1.2) won't have added them,
        # so add them here rather than making the combination unusable.
        if "sr_support" not in df.columns:
            return add_sr_zones(df, left=int(p.get("sr_left", 10)), right=int(p.get("sr_right", 10)),
                                cluster_pct=float(p.get("sr_cluster_pct", 0.15)), prefix="sr")
        return df

    def exit(self, p: dict):
        pct, mt = float(p["tp_zone_pct"]), int(p.get("tp_zone_min_touches", 1))
        return (near_resistance(pct, min_touches=mt) if p.get("_side") == "long"
                else near_support(pct, min_touches=mt))


class _OppositeCandle(TPMethod):
    """"TP L" — ride the move for as long as the candles keep going your way."""
    name, version = "opposite_candle", "v1"
    description = ("Hold while candles stay in your favour (red for a short); close as soon as "
                   "`tp_opp_n` opposite-colour candle(s) CLOSE. Works on real or Heikin-Ashi candles.")
    default_params = {"tp_opp_n": 1, "candle_src": "real"}

    def prepare(self, df, p):
        if p.get("candle_src") == "ha" and "ha_close" not in df.columns:
            return add_heikin_ashi(df)
        return df

    def exit(self, p: dict):
        # the opposite-colour candle: GREEN ends a short, RED ends a long
        n, src = int(p["tp_opp_n"]), p.get("candle_src", "real")
        return red(n, src=src) if p.get("_side") == "long" else green(n, src=src)


class _RRPartials(TPMethod):
    name, version = "rr_partials", "v1"
    description = "Scale out: close part at 1R (stop to breakeven), the rest at the final R."
    default_params = {"tp_partial_rr1": 1.0, "tp_partial_pct1": 50.0, "tp_rr": 2.0}

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        tps = (
            TakeProfit(mode="rr", value=float(p["tp_partial_rr1"]),
                       close_pct=float(p["tp_partial_pct1"]), move_stop_mode="breakeven"),
            TakeProfit(mode="rr", value=float(p["tp_rr"]), close_pct=100.0),
        )
        return dataclasses.replace(cfg, take_profits=tps)


register_tp(_NoTarget())
register_tp(_RRTarget())
register_tp(_FixedPctTarget())
register_tp(_Trailing())
register_tp(_NGreenExit())
register_tp(_BackAboveEma())
register_tp(_NextZoneSignal())
register_tp(_OppositeCandle())
register_tp(_RRPartials())
