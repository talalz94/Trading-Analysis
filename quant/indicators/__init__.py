"""Vectorized indicator library (compute only; rendering lives in quant.viz)."""
from __future__ import annotations

from .overlays import add_emas, add_smas, ema, sma
from .oscillators import (add_macd, add_rsi, add_rsi_divergence, add_rsi_divergence_realtime,
                          add_stochastic, rsi,
                          rsi_divergence_table)
from .candles import add_heikin_ashi
from .candle_patterns import (add_candle_patterns, candle_pattern_table,
                              ALL_PATTERNS as CANDLE_PATTERNS,
                              PATTERN_DIRECTION as CANDLE_PATTERN_DIRECTION)
from .chart_patterns import (add_chart_patterns, chart_pattern_shapes, chart_pattern_table,
                             chart_pattern_shapes_tuned, tuned_specs,
                             TUNED_SPECS, TUNED_COMMON,
                             ALL_PATTERNS as CHART_PATTERNS,
                             PATTERN_DIRECTION as CHART_PATTERN_DIRECTION)
from .volatility import add_atr, add_compression, add_supertrend, atr
from .structure import (add_pivot_points, add_swings, add_sr_zones, add_trend_structure, add_zigzag_structure,
                        sr_zone_table)
from .levels import (add_prev_day_levels, add_round_levels, add_session_levels,
                     add_volume_features, add_vwap, add_week_levels, level_events)
from .momentum import add_squeeze_momentum, add_wavetrend, add_ema_gradient
from .trendlines import add_trendlines, trendline_table

__all__ = [
    "ema", "sma", "add_emas", "add_smas",
    "rsi", "add_rsi", "add_rsi_divergence", "add_rsi_divergence_realtime", "rsi_divergence_table", "add_macd", "add_stochastic",
    "add_heikin_ashi",
    "add_candle_patterns", "candle_pattern_table", "CANDLE_PATTERNS",
    "CANDLE_PATTERN_DIRECTION",
    "add_chart_patterns", "chart_pattern_table", "chart_pattern_shapes", "CHART_PATTERNS",
    "chart_pattern_shapes_tuned", "tuned_specs", "TUNED_SPECS", "TUNED_COMMON",
    "CHART_PATTERN_DIRECTION",
    "atr", "add_atr", "add_compression", "add_supertrend",
    "add_swings", "add_pivot_points", "add_sr_zones", "sr_zone_table", "add_trend_structure", "add_zigzag_structure",
    "add_squeeze_momentum", "add_wavetrend", "add_ema_gradient",
    "add_trendlines", "trendline_table",
    "add_prev_day_levels", "add_week_levels", "add_session_levels", "add_vwap",
    "add_volume_features", "add_round_levels", "level_events",
]
