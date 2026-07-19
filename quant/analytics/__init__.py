"""Analytics: performance metrics + time/regime attribution."""
from __future__ import annotations

from .metrics import compute_stats
from .attribution import (
    by_hour, by_month, by_regime, by_session, by_weekday, bucket_stats, monthly_returns,
)

__all__ = [
    "compute_stats", "bucket_stats",
    "by_hour", "by_weekday", "by_month", "by_session", "by_regime", "monthly_returns",
]
