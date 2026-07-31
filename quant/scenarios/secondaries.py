"""
Versioned SECONDARY rules — logic that only plays out *after the main scenario has opened a trade*.

A secondary has two conditions, both evaluated per bar while a trade is open:
  `arm`   — once true at/after THIS trade's entry, the trade is armed (permanently, per trade);
  `exit`  — once armed, closes the trade, but only while it is **in profit**.

The per-trade arming lives in the kernel (`o_armed` / `alt_exit_enabled`); trades closed this way get
the `alt_exit` close reason. Not setting a secondary leaves the original engine path untouched.

Params (flat, merged with the scenario params):
  sec_arm_mode   how close price must get to the EMA line to ARM the trade:
                 'near' | 'touch' (default) | 'close_above' | 'full_above'  — see `arm_cond`
  sec_arm_tol_pct  tolerance for 'near', in % below the line (default 0.05)
  sec_touch      LEGACY: used only when sec_arm_mode is None ('touch' if True else 'close_above')
  sec_opp_n      opposite-colour candles that trigger the bail-out (default 1)
  candle_src     'real' | 'ha' — which candles the colour test reads
  grad_prefix    which EMA line to watch (default 'grad' -> the `grad_line` column)
  sec_replace_tp once armed, DROP the original take-profit so only this rule (or the stop) can close
                 the trade. Default False = keep both, whichever triggers first wins.
"""
from __future__ import annotations

import numpy as np

from ..indicators import add_heikin_ashi
from ..strategies.conditions import Cond, above, green
from .base import SecondaryMethod, register_secondary


def arm_cond(p: dict) -> Cond:
    """How close price must get to the EMA line before a trade counts as ARMED.

    `sec_arm_mode`, loosest to strictest:
      'near'        — the high came within `sec_arm_tol_pct`% BELOW the line (never quite touched).
                      Use when a bar misses by a hair: a 0.001% miss is not a real distinction.
      'touch'       — the high reached the line (high >= line).            [default]
      'close_above' — the candle CLOSED above the line.
      'full_above'  — the WHOLE candle formed above the line (low > line).

    Left unset, the legacy `sec_touch` flag still decides ('touch' if True else 'close_above'), so
    existing configs behave exactly as before.
    """
    line = f"{p.get('grad_prefix', 'grad')}_line"
    mode = p.get("sec_arm_mode")
    if mode is None:
        mode = "touch" if p.get("sec_touch", True) else "close_above"
    tol = float(p.get("sec_arm_tol_pct", 0.0) or 0.0)

    if mode == "close_above":
        return above("close", line)

    if mode == "full_above":
        def full(df):
            return df["low"].to_numpy() > df[line].to_numpy()
        return Cond(full, f"full_above({line})")

    if mode == "touch":
        def touched(df):
            return df["high"].to_numpy() >= df[line].to_numpy()
        # a close above the line also counts, even on a bar whose high is odd/missing
        return above("close", line) | Cond(touched, f"touch({line})")

    if mode == "near":
        def near(df):
            ln = df[line].to_numpy()
            return df["high"].to_numpy() >= ln * (1.0 - tol / 100.0)
        return above("close", line) | Cond(near, f"near({line},{tol}%)")

    raise ValueError(f"unknown sec_arm_mode {mode!r}: "
                     f"use near|touch|close_above|full_above")


class _ShakyTrendBail(SecondaryMethod):
    name, version = "shaky_trend_bail", "v1"
    description = ("After the trade is open: once price gets to the EMA line (mode configurable via "
                   "sec_arm_mode = near|touch|close_above|full_above) the trend is shaky — from then "
                   "on, bail on the first opposite-colour candle, provided we are in profit. The "
                   "stop-loss is unchanged.")
    default_params = {"sec_arm_mode": None, "sec_arm_tol_pct": 0.05, "sec_touch": True,
                      "sec_opp_n": 1, "candle_src": "real",
                      "grad_prefix": "grad", "sec_replace_tp": False}

    def prepare(self, df, p):
        if p.get("candle_src") == "ha" and "ha_close" not in df.columns:
            return add_heikin_ashi(df)
        return df

    def arm(self, p: dict) -> Cond:
        return arm_cond(p)

    def exit(self, p: dict) -> Cond:
        # short setups: bail on the opposite (green) candle
        return green(int(p.get("sec_opp_n", 1)), src=p.get("candle_src", "real"))


class _ProfitCandleBail(SecondaryMethod):
    """The same bail-out, with the arming condition removed — for using it as the SOLE exit rule.

    `shaky_trend_bail@v1` waits for the trend to look shaky (price back at the EMA) before it will
    bail; that gate is what makes it *secondary*. Promoted to be the only exit, the gate usually is
    not wanted: you simply want "ride the move while candles keep going my way, and take what is
    there the moment one closes against me — provided I'm green".

    The in-profit gate still applies (it lives in the engine), so this can never realise a loss by
    itself; a losing trade is still handed to the stop-loss.
    """
    name, version = "profit_candle_bail", "v1"
    description = ("Always armed: close on the first opposite-colour candle while IN PROFIT. No EMA "
                   "touch required. Designed to be the sole exit (pair with tp='none@v1').")
    default_params = {"sec_opp_n": 1, "candle_src": "real", "sec_replace_tp": False}

    def prepare(self, df, p):
        if p.get("candle_src") == "ha" and "ha_close" not in df.columns:
            return add_heikin_ashi(df)
        return df

    def arm(self, p: dict) -> Cond:
        # armed on every bar; the engine still only arms a trade from the bar AFTER its own entry
        return Cond(lambda df: np.ones(len(df), dtype=bool), "always_armed")

    def exit(self, p: dict) -> Cond:
        return green(int(p.get("sec_opp_n", 1)), src=p.get("candle_src", "real"))


class _RunPastTp(SecondaryMethod):
    """Let a winner run THROUGH its take-profit while the candles keep going your way.

    Normally hitting the TP closes the trade. With this secondary the TP level instead **arms** the
    trade (engine flag `alt_arm_on_tp`) and the position stays open; it then closes on the first
    `sec_opp_n` opposite-colour candle(s) — and only while in profit, which past the TP it is. So:

        reach TP -> keep riding red candles -> exit when N green candles close.

    The stop-loss and trailing stop are untouched, and a trade that never reaches TP behaves exactly
    as before. Pair with a real target (e.g. tp='rr@v1') — the target defines where riding begins.
    """
    name, version = "run_past_tp", "v1"
    description = ("Reaching the take-profit ARMS the trade instead of closing it; it then rides "
                   "while candles stay favourable and exits on N opposite-colour candles.")
    default_params = {"sec_opp_n": 1, "candle_src": "real", "sec_arm_on_tp": True}

    def prepare(self, df, p):
        if p.get("candle_src") == "ha" and "ha_close" not in df.columns:
            return add_heikin_ashi(df)
        return df

    def arm(self, p: dict) -> Cond:
        # never arms from a signal — the kernel arms it when the TP level is reached
        return Cond(lambda df: np.zeros(len(df), dtype=bool), "arm_on_tp_only")

    def exit(self, p: dict) -> Cond:
        return green(int(p.get("sec_opp_n", 1)), src=p.get("candle_src", "real"))


register_secondary(_ShakyTrendBail())
register_secondary(_RunPastTp())
register_secondary(_ProfitCandleBail())
