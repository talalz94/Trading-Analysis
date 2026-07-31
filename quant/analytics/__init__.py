"""Analytics: performance metrics + time/regime attribution."""
from __future__ import annotations

from .metrics import compute_stats
from .attribution import (
    by_hour, by_month, by_regime, by_session, by_weekday, bucket_stats, monthly_returns,
)
from .timing import (
    SESSIONS, bucket_by_period, efficiency_by_block, ema_followthrough, persistence,
    trade_buckets, walkforward_block,
)

__all__ = [
    "compute_stats", "bucket_stats",
    "by_hour", "by_weekday", "by_month", "by_session", "by_regime", "monthly_returns",
    "efficiency_by_block", "ema_followthrough", "bucket_by_period", "persistence",
    "walkforward_block", "trade_buckets", "SESSIONS",
]
