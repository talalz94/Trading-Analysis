"""Plug-and-play strategies. Every field is a tunable parameter; sweep them without code changes."""
from __future__ import annotations

from .base import Strategy
from .ema_ribbon import EmaRibbon
from .ema_cross import EmaCross
from .rsi import RsiReversal
from .macd import MacdTrend
from .heikin_ashi import HeikinAshiTrend
from .supertrend import SupertrendFlip
from .mtf import MtfTrend
from .key_level import KeyLevelBounce
from .builder import RuleStrategy, Rule
from .fvg_veto import (build_fvg_rules, describe_fvg_rules, fvg_blocks, fvg_clear,
                       fvg_layers, fvg_rule_cond, parse_fvg_rule, VETO_MODES)
from .indicators import (Ind, MA, EMA, SMA, RSI, ATR, HeikinAshi, Swings, SRZones)
from . import conditions, indicators as ind

# Registry so strategies can be selected by name (e.g. from a config/CLI).
REGISTRY: dict[str, type[Strategy]] = {
    EmaRibbon.name: EmaRibbon,
    EmaCross.name: EmaCross,
    RsiReversal.name: RsiReversal,
    MacdTrend.name: MacdTrend,
    HeikinAshiTrend.name: HeikinAshiTrend,
    SupertrendFlip.name: SupertrendFlip,
    MtfTrend.name: MtfTrend,
    KeyLevelBounce.name: KeyLevelBounce,
}

__all__ = [
    "fvg_blocks", "fvg_clear", "fvg_layers", "VETO_MODES",
    "parse_fvg_rule", "fvg_rule_cond", "build_fvg_rules", "describe_fvg_rules",
    "Strategy", "EmaRibbon", "EmaCross", "RsiReversal", "MacdTrend", "HeikinAshiTrend",
    "SupertrendFlip", "MtfTrend", "KeyLevelBounce", "REGISTRY",
    "RuleStrategy", "Rule", "conditions", "ind",
    "Ind", "MA", "EMA", "SMA", "RSI", "ATR", "HeikinAshi", "Swings", "SRZones",
]
