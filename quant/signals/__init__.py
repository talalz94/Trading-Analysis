"""Vectorized signal primitives — the single signal representation for manual & swept runs."""
from __future__ import annotations

from . import primitives, time_filters
from .primitives import (
    above, above_all, all_of, any_of, below, below_all, col,
    consecutive_green, consecutive_red, cross_down, cross_up, crossed_up_within,
    falling, is_green, is_red, last_all_above, last_all_below, none_of,
    prev_all_above, prev_all_below, refs_ordered, rising,
)
from .time_filters import (
    between_times, hour_between, in_session, not_weekend, weekday_in,
)

__all__ = [
    "primitives", "time_filters", "col", "above", "below", "above_all", "below_all",
    "cross_up", "cross_down", "crossed_up_within",
    "last_all_above", "last_all_below", "prev_all_above", "prev_all_below",
    "is_green", "is_red", "consecutive_green", "consecutive_red",
    "rising", "falling", "refs_ordered", "all_of", "any_of", "none_of",
    "hour_between", "in_session", "weekday_in", "not_weekend", "between_times",
]
