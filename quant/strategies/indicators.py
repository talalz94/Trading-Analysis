"""
Tagged indicator handles — declare an indicator once with a **tag**, then reference the handle in
rules, stops and plots instead of a raw column-name string.

The tag IS the column name, so the same indicator with different settings never collides:

    trend  = MA('trend',  200, type='sma', smooth=9, smooth_type='ema')   # -> column 'trend'
    fast   = MA('fast',    20, type='sma', smooth=9, smooth_type='ema')   # -> column 'fast'
    h1bias = MA('h1bias',  50, type='ema', tf='1h')      # higher TF, merged anti-lookahead
    rsi_f  = RSI('rsi_f',   7)
    rsi_s  = RSI('rsi_s',  21)                            # same kind, different settings: no clash
    ha     = HeikinAshi('ha')                             # ha.open/.high/.low/.close/.green
    sw     = Swings('sw', 5, 5)                           # sw.last_low / sw.last_high
    sr     = SRZones('sr', tf='1h')                       # sr.support / sr.resistance / *_touches

    entry = full_above(trend, n=2) & above('close', h1bias) & below(rsi_f, 30)
    stop  = ref_col_stop(sw.last_low)

Multi-column indicators expose their sub-columns as plain column-name strings (`sw.last_low`), so
they drop straight into any condition. Single-value handles resolve via `.column`.

Cost: all of this is *declaration-time*. Tags resolve to column names once when conditions are
built and indicators are computed once in `prepare()` (de-duplicated by tag). Nothing here runs
per bar — the signal arrays and the numba kernel are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..data import build_mtf
from ..indicators import (add_atr, add_heikin_ashi, add_rsi, add_sr_zones, add_swings)
from ..indicators.overlays import ema as _ema_series, sma as _sma_series
from ..indicators.volatility import atr as _atr_series
from ..indicators.oscillators import rsi as _rsi_series


def _moving_average(s: pd.Series, length: int, kind: str) -> pd.Series:
    """Moving average matching TradingView's MA types (sma / ema / rma|smma / wma)."""
    import numpy as np
    n = int(length)
    k = kind.lower()
    if k == "sma":
        return _sma_series(s, n)
    if k == "ema":
        return _ema_series(s, n)
    if k in ("rma", "smma", "wilder"):
        return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    if k == "wma":
        w = np.arange(1, n + 1, dtype=float)
        return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    raise ValueError(f"unknown MA type {kind!r}; use sma/ema/rma/wma")


class Ind:
    """Base class for a tagged indicator. Subclasses implement `_build` and `columns`."""

    kind: str = "ind"

    def __init__(self, tag: str, *, tf: Optional[str] = None, **params: Any):
        if not tag or not isinstance(tag, str):
            raise ValueError(f"{type(self).__name__}: tag must be a non-empty string")
        self.tag = tag
        self.tf = tf
        self.params: Dict[str, Any] = params

    # ---- identity -------------------------------------------------------
    @property
    def columns(self) -> List[str]:
        """All column names this indicator produces."""
        return [self.tag]

    @property
    def column(self) -> str:
        """The single value column (only for single-valued indicators)."""
        cols = self.columns
        if len(cols) != 1:
            raise TypeError(
                f"{self.tag!r} produces {len(cols)} columns ({', '.join(cols)}); reference a "
                f"specific one (e.g. {self.tag}.{cols[0].split('_', 1)[-1]}) instead of the handle.")
        return cols[0]

    @property
    def label(self) -> str:
        """Human-readable legend label."""
        return self.tag

    def signature(self) -> tuple:
        """Identity for de-dup / collision detection: same tag + same signature = same indicator."""
        return (self.kind, self.tf, tuple(sorted(self.params.items())))

    # ---- computation ----------------------------------------------------
    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add this indicator's column(s). With `tf`, computes on the resampled frame and merges
        back anti-lookahead (shifted one HTF bar), renaming to the plain tag columns."""
        if self.tf is None:
            return self._build(df)
        out = build_mtf(df, {self.tf: self._build})
        ren = {f"{self.tf}__{c}": c for c in self.columns}
        missing = [k for k in ren if k not in out.columns]
        if missing:
            raise RuntimeError(f"{self.tag!r}: MTF merge did not produce {missing}")
        return out.rename(columns=ren)

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        tf = f", tf={self.tf!r}" if self.tf else ""
        return f"{type(self).__name__}({self.tag!r}, {p}{tf})"


# ---------------------------------------------------------------------------
# Single-value indicators
# ---------------------------------------------------------------------------
class MA(Ind):
    """Configurable moving average — maps 1:1 onto TradingView's MA dialog.

        MA('trend', 200, type='sma', smooth=9, smooth_type='ema')
             tag  Length  base type   Smoothing Length / Smoothing Line
    """
    kind = "ma"

    def __init__(self, tag: str, length: int, *, type: str = "ema", source: str = "close",
                 smooth: int = 0, smooth_type: str = "ema", tf: Optional[str] = None):
        super().__init__(tag, tf=tf, length=int(length), type=type, source=source,
                         smooth=int(smooth), smooth_type=smooth_type)

    @property
    def label(self) -> str:
        p = self.params
        base = f"{p['type'].upper()}{p['length']}"
        return f"{self.tag} ({base}+{p['smooth_type']}{p['smooth']})" if p["smooth"] else \
               f"{self.tag} ({base})"

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()
        s = _moving_average(out[p["source"]], p["length"], p["type"])
        if p["smooth"]:
            s = _moving_average(s, p["smooth"], p["smooth_type"])
        out[self.tag] = s
        return out


class EMA(MA):
    """Plain exponential moving average: EMA('fast', 20)."""

    def __init__(self, tag: str, length: int, *, source: str = "close",
                 smooth: int = 0, smooth_type: str = "ema", tf: Optional[str] = None):
        super().__init__(tag, length, type="ema", source=source, smooth=smooth,
                         smooth_type=smooth_type, tf=tf)


class SMA(MA):
    """Plain simple moving average: SMA('base', 50)."""

    def __init__(self, tag: str, length: int, *, source: str = "close",
                 smooth: int = 0, smooth_type: str = "ema", tf: Optional[str] = None):
        super().__init__(tag, length, type="sma", source=source, smooth=smooth,
                         smooth_type=smooth_type, tf=tf)


class RSI(Ind):
    """Wilder RSI: RSI('rsi_f', 7)."""
    kind = "rsi"

    def __init__(self, tag: str, period: int = 14, *, source: str = "close",
                 tf: Optional[str] = None):
        super().__init__(tag, tf=tf, period=int(period), source=source)

    @property
    def label(self) -> str:
        return f"{self.tag} (RSI{self.params['period']})"

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.tag] = _rsi_series(out[self.params["source"]], self.params["period"])
        return out


class ATR(Ind):
    """Average True Range: ATR('atr', 14)."""
    kind = "atr"

    def __init__(self, tag: str, period: int = 14, *, tf: Optional[str] = None):
        super().__init__(tag, tf=tf, period=int(period))

    @property
    def label(self) -> str:
        return f"{self.tag} (ATR{self.params['period']})"

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.tag] = _atr_series(out, self.params["period"])
        return out


# ---------------------------------------------------------------------------
# Multi-column indicators (sub-columns are plain column-name strings)
# ---------------------------------------------------------------------------
class HeikinAshi(Ind):
    """Heikin Ashi candles: ha = HeikinAshi('ha') -> ha.open/.high/.low/.close/.green."""
    kind = "heikin_ashi"

    def __init__(self, tag: str = "ha"):
        super().__init__(tag)

    @property
    def columns(self) -> List[str]:
        return [f"{self.tag}_{s}" for s in ("open", "high", "low", "close", "green")]

    open = property(lambda self: f"{self.tag}_open")
    high = property(lambda self: f"{self.tag}_high")
    low = property(lambda self: f"{self.tag}_low")
    close = property(lambda self: f"{self.tag}_close")
    green = property(lambda self: f"{self.tag}_green")

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_heikin_ashi(df, prefix=self.tag)


class Swings(Ind):
    """Swing pivots: sw = Swings('sw', 5, 5) -> sw.last_low / sw.last_high (ref_col stop sources)."""
    kind = "swings"

    def __init__(self, tag: str, left: int = 10, right: int = 10):
        super().__init__(tag, left=int(left), right=int(right))

    @property
    def columns(self) -> List[str]:
        return [f"{self.tag}_{s}" for s in ("high", "low", "last_high", "last_low")]

    last_low = property(lambda self: f"{self.tag}_last_low")
    last_high = property(lambda self: f"{self.tag}_last_high")

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_swings(df, self.params["left"], self.params["right"], prefix=self.tag)


class SRZones(Ind):
    """Swing-cluster support/resistance zones: sr = SRZones('sr', tf='1h')
    -> sr.support / sr.resistance / sr.support_touches / sr.resistance_touches."""
    kind = "sr_zones"

    def __init__(self, tag: str = "sr", left: int = 10, right: int = 10, *,
                 cluster_pct: float = 0.15, max_zones: int = 128, tf: Optional[str] = None):
        super().__init__(tag, tf=tf, left=int(left), right=int(right),
                         cluster_pct=float(cluster_pct), max_zones=int(max_zones))

    @property
    def columns(self) -> List[str]:
        return [f"{self.tag}_{s}" for s in
                ("support", "resistance", "support_touches", "resistance_touches")]

    support = property(lambda self: f"{self.tag}_support")
    resistance = property(lambda self: f"{self.tag}_resistance")
    support_touches = property(lambda self: f"{self.tag}_support_touches")
    resistance_touches = property(lambda self: f"{self.tag}_resistance_touches")

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return add_sr_zones(df, p["left"], p["right"], cluster_pct=p["cluster_pct"],
                            max_zones=p["max_zones"], prefix=self.tag)


# ---------------------------------------------------------------------------
# Application / collision detection
# ---------------------------------------------------------------------------
def apply_tagged(df: pd.DataFrame, inds: Sequence[Ind]) -> pd.DataFrame:
    """Apply tagged indicators once each, in declared order.

    De-dups identical (tag, settings) declarations and raises on a tag reused with DIFFERENT
    settings — the collision that silently corrupted auto-named columns before.
    """
    out = df
    seen: Dict[str, tuple] = {}
    for ind in inds:
        sig = ind.signature()
        prev = seen.get(ind.tag)
        if prev is not None:
            if prev != sig:
                raise ValueError(
                    f"Indicator tag {ind.tag!r} declared twice with different settings. "
                    f"Give them distinct tags (e.g. {ind.tag!r} and '{ind.tag}_2').")
            continue                      # identical re-declaration: compute once
        seen[ind.tag] = sig
        out = ind.apply(out)
    return out


def col_of(x: Any) -> Any:
    """Resolve an indicator handle to its column name; pass through strings/numbers unchanged.
    Called once when a condition is built — never per bar."""
    if isinstance(x, Ind):
        return x.column
    return x
