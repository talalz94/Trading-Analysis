"""
What can fire an alert. Each detector reads ONE timeframe's closed candles and says on which of
them it fired. The library's indicators are causal (tested by truncation), so a detector run on a
window ending at bar i says exactly what was knowable at bar i's close -- no repainting.

To add a kind: subclass Detector, give it `kind`, `validate`, `warmup_bars` and `evaluate`, and
register it. The engine never needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd


class Detector(ABC):
    kind: str = ""

    @abstractmethod
    def validate(self, params: dict) -> None:
        """Raise KeyError / ValueError naming the problem."""

    @abstractmethod
    def warmup_bars(self, params: dict) -> int:
        """Closed bars of history the detector needs before the bar it judges."""

    @abstractmethod
    def evaluate(self, frame: pd.DataFrame, params: dict) -> List[Dict[str, object]]:
        """Every bar of `frame` it fires on: dicts with `i` (row), `what` (a short phrase),
        `side` ('bull' / 'bear' / ''), and anything else worth putting in the email."""


def _pretty(name: str) -> str:
    """engulfing_bull -> bullish engulfing"""
    words = name.split("_")
    side = {"bull": "bullish", "bear": "bearish"}.get(words[-1])
    return f"{side} {' '.join(words[:-1])}" if side else " ".join(words)


class CandleDetector(Detector):
    """Candlestick patterns, flagged on the candle that COMPLETES them.

        dict(kind="candle", patterns=("engulfing_bull", "engulfing_bear"), ...)

    Names are quant.indicators.candle_patterns' (62 of them, e.g. hammer, morning_star,
    engulfing_bull); `patterns="all"` watches every one.
    """
    kind = "candle"

    def validate(self, params):
        from quant.indicators.candle_patterns import ALL_PATTERNS
        pats = params.get("patterns")
        if pats is None:
            raise KeyError("candle alerts need patterns=(...) or patterns='all'")
        if pats != "all":
            pats = (pats,) if isinstance(pats, str) else tuple(pats)
            bad = [p for p in pats if p not in ALL_PATTERNS]
            if bad:
                raise KeyError(f"unknown candle pattern(s) {bad}; e.g. {ALL_PATTERNS[:6]}")

    def warmup_bars(self, params):
        return 60          # a 20-bar average body and a 10-bar trend, plus three-candle patterns

    def evaluate(self, frame, params):
        from quant.indicators import add_candle_patterns
        from quant.indicators.candle_patterns import ALL_PATTERNS
        pats = params["patterns"]
        pats = list(ALL_PATTERNS) if pats == "all" else ([pats] if isinstance(pats, str)
                                                         else list(pats))
        out = add_candle_patterns(frame[["open", "high", "low", "close"]], patterns=pats,
                                  add_score=False)
        hits = []
        for p in pats:
            for i in out.index[out[f"cdl_{p}"].to_numpy(bool)]:
                side = "bull" if p.endswith("_bull") else "bear" if p.endswith("_bear") else ""
                hits.append(dict(i=int(i), what=_pretty(p), side=side))
        return hits


REGISTRY: Dict[str, Detector] = {d.kind: d for d in (CandleDetector(),)}
