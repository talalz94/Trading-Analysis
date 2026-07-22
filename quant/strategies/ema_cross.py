"""
EmaCross — a highly configurable price-vs-EMA strategy for systematic variant exploration.

Base idea: long when price is above the Nth EMA, short when below. Every variant the research
question asks about is a parameter, so a single sweep can answer "which variant works best?":

  entry_mode   : how price relates to the EMA to trigger
                 'cross'       -> close crosses the EMA (event)
                 'close'       -> close is above/below the EMA (state, held confirm_n bars)
                 'full_candle' -> the WHOLE candle is above/below (low>EMA / high<EMA), confirm_n bars
  confirm_n    : how many candles the condition must hold before entering
  confirm_color: also require those candles to be bullish (long) / bearish (short)
  use_heikin_ashi: evaluate candles + EMA on Heikin-Ashi values instead of regular OHLC
  htf / htf_ema  : higher-timeframe bias filter — only long above the HTF EMA, only short below
  allow_long / allow_short : trade one or both directions
  exit_mode    : 'opposite' (flip when price crosses back), 'none' (rely on SL/TP from cfg),
                 'ha_flip' (exit on first opposite Heikin-Ashi candle),
                 'below_ema' (exit when a full candle forms beyond a shorter `exit_ema`)

`prepare()` also adds swing levels (`swing_last_low/high`) so ref_col structure stops work out of
the box, and ATR (`atr_14`) for ATR-based experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..data import build_mtf
from ..engine import Signals
from ..indicators import add_atr, add_emas, add_heikin_ashi, add_swings, ema
from .. import signals as S
from .base import Strategy


@dataclass
class EmaCross(Strategy):
    name: str = "ema_cross"
    ema_period: int = 50
    entry_mode: str = "cross"          # cross | close | full_candle
    confirm_n: int = 1
    confirm_color: bool = False
    use_heikin_ashi: bool = False
    allow_long: bool = True
    allow_short: bool = True
    htf: Optional[str] = None          # e.g. "1h" — higher-timeframe bias
    htf_ema: Optional[int] = None
    exit_mode: str = "opposite"        # opposite | none | ha_flip | below_ema
    exit_ema: Optional[int] = None     # shorter EMA for exit_mode="below_ema"
    swings_left: int = 12
    swings_right: int = 12

    @property
    def _src_tag(self) -> str:
        return "ha" if self.use_heikin_ashi else "px"

    @property
    def _entry_col(self) -> str:
        return f"ema_entry_{self._src_tag}_{int(self.ema_period)}"

    @property
    def _exit_col(self) -> str:
        return f"ema_exit_{self._src_tag}_{int(self.exit_ema or 0)}"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        need_ha = self.use_heikin_ashi or self.exit_mode == "ha_flip"
        out = add_heikin_ashi(df) if need_ha else df.copy()
        src = "ha_close" if self.use_heikin_ashi else "close"
        # Period-encoded column names so sweeps over ema_period produce DISTINCT columns
        # (run_grid caches indicator columns by name; a fixed name would silently reuse one EMA).
        out[self._entry_col] = ema(out[src], int(self.ema_period))
        if self.exit_mode == "below_ema" and self.exit_ema:
            out[self._exit_col] = ema(out[src], int(self.exit_ema))
        out = add_swings(out, self.swings_left, self.swings_right)   # for ref_col stops
        out = add_atr(out, 14)                                       # for ATR-based experiments
        if self.htf and self.htf_ema:
            p = int(self.htf_ema)
            out = build_mtf(out, {self.htf: lambda d: add_emas(d, [p])})
        return out

    def build_signals(self, df: pd.DataFrame) -> Signals:
        n = len(df)
        ha = self.use_heikin_ashi
        c = "ha_close" if ha else "close"
        lo = "ha_low" if ha else "low"
        hi = "ha_high" if ha else "high"
        o_col, c_col = ("ha_open", "ha_close") if ha else ("open", "close")
        ema_col = self._entry_col
        zeros = np.zeros(n, dtype=bool)

        def entry(side_long: bool):
            if self.entry_mode == "cross":
                trig = S.cross_up(df, c, ema_col) if side_long else S.cross_down(df, c, ema_col)
            elif self.entry_mode == "close":
                trig = (S.last_all_above(df, c, ema_col, self.confirm_n) if side_long
                        else S.last_all_below(df, c, ema_col, self.confirm_n))
            elif self.entry_mode == "full_candle":
                trig = (S.last_all_above(df, lo, ema_col, self.confirm_n) if side_long
                        else S.last_all_below(df, hi, ema_col, self.confirm_n))
            else:
                raise ValueError(f"bad entry_mode {self.entry_mode}")
            if self.confirm_color:
                col = (S.consecutive_green(df, self.confirm_n, o_col, c_col) if side_long
                       else S.consecutive_red(df, self.confirm_n, o_col, c_col))
                trig = S.all_of(trig, col)
            if self.htf and self.htf_ema:
                hcol = f"{self.htf}__ema_{int(self.htf_ema)}"
                bias = S.above(df, c, hcol) if side_long else S.below(df, c, hcol)
                trig = S.all_of(trig, bias)
            return trig

        def exit_(side_long: bool):
            if self.exit_mode == "none":
                return zeros
            if self.exit_mode == "opposite":
                return S.below(df, c, ema_col) if side_long else S.above(df, c, ema_col)
            if self.exit_mode == "ha_flip":
                return (S.below(df, "ha_close", "ha_open") if side_long
                        else S.above(df, "ha_close", "ha_open"))
            if self.exit_mode == "below_ema":
                ec = self._exit_col
                return S.below(df, hi, ec) if side_long else S.above(df, lo, ec)
            raise ValueError(f"bad exit_mode {self.exit_mode}")

        el = entry(True) if self.allow_long else zeros
        xl = exit_(True) if self.allow_long else zeros
        es = entry(False) if self.allow_short else None
        xs = exit_(False) if self.allow_short else None
        return Signals(entry_long=el, exit_long=xl, entry_short=es, exit_short=xs)
