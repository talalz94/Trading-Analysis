"""
quant.alerts -- watch markets for patterns and rules, and email when one forms.

    alerts/alerts_config.py     WHAT to watch (markets, timeframes, alerts, delivery) -- yours
    python -m quant.alerts ...  check | test-email | tick | run | status
    docs/ALERTS.md              how it works, how to set up email, how to add an alert kind

Layout: config (read + validate the config file) -> feeds (closed 5m bars) -> detectors (what can
fire) -> engine (one resumable tick) -> compose (the email) -> notify (SMTP) -> runner (the loop).
State lives in alerts/state/ (git-ignored): SQLite for cursors/events/outbox, parquet for bars.
"""
from .config import AlertsConfig, ConfigError, load_config
from .detectors import REGISTRY, Detector
from .engine import TickReport, preview, tick

__all__ = ["AlertsConfig", "ConfigError", "load_config", "REGISTRY", "Detector", "TickReport",
           "preview", "tick"]
