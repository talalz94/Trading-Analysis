"""
quant.news -- the economic calendar: what is released when, how much it matters, what it said.

    from quant.news import load_events, news_features, news_blackout
    ev = load_events("2024-01-01", "2024-12-31", coverage={"USD": "high"})
    ev[["t", "currency", "title", "level", "actual", "forecast", "surprise_z", "sentiment"]]

Sources, their faults and how the merge trusts them: quant/news/sources.py and calendar.py.
Keep it current with `python -m quant.news update`; see docs/NEWS.md.
"""
from .catalog import (CURRENCIES, DEFAULT_COVERAGE, LEVELS, RANK, category, criticality,
                      direction, parse_coverage)
from .calendar import EVENT_COLS, NewsStore, build_events, load_events, select
from .features import event_times, news_blackout, news_features
from .values import parse_value, parse_values, sentiment, surprise_z

__all__ = ["CURRENCIES", "DEFAULT_COVERAGE", "LEVELS", "RANK", "category", "criticality",
           "direction", "parse_coverage", "EVENT_COLS", "NewsStore", "build_events",
           "load_events", "select", "event_times", "news_blackout", "news_features",
           "parse_value", "parse_values", "sentiment", "surprise_z"]
