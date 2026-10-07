"""
ALERTS — WHAT TO WATCH, AND WHEN TO EMAIL YOU.  THIS FILE IS YOURS.

The monitor re-reads this file every minute: save, and the next tick uses it, no restart.
Try a change first with   python -m quant.alerts check   -- it shows what WOULD have fired in the
last 48 hours as the email it would have sent, and sends nothing. Full guide: docs/ALERTS.md.

  §1 markets   §2 alerts   §3 delivery
"""

# ══ §1  MARKETS — every one is fetched at 5m; each alert's timeframes are built from that ══════
MARKETS = {
    "Gold":   dict(symbol="XAUUSD",  source="dukascopy"),
    "USDJPY": dict(symbol="USDJPY",  source="dukascopy"),
    "BTC":    dict(symbol="BTCUSDT", source="binance"),
}
GROUPS = {                           # a name an alert's `markets=` can use; "all" always exists
    "metals & FX": ("Gold", "USDJPY"),
}

# ══ §2  ALERTS — categories appear in an email in this order ════════════════════════════════════
CATEGORIES = ("Candle patterns",)

# kind="candle": any of the 62 patterns in quant/indicators/candle_patterns.py, or patterns="all".
# priority  urgent = emailed the minute it is found · normal = waits for the next digest
# cooldown_bars = after firing, stay quiet on that market + timeframe for this many of its bars
# How often each fired on these three markets, Mar-Jul 2026 (a day, all three together):
#   engulfing   15m ~38   1h ~10   4h ~2.5      hammer + shooting star   15m ~10   1h ~2.4   4h ~0.6
ALERTS = [
    dict(name="engulfing 4h", category="Candle patterns", kind="candle",
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs=("4h",), priority="urgent"),
    dict(name="engulfing 1h", category="Candle patterns", kind="candle",
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs=("1h",), priority="normal"),
    dict(name="hammer / shooting star", category="Candle patterns", kind="candle",
         patterns=("hammer", "shooting_star"),
         markets="all", tfs=("15m", "1h", "4h"), priority="normal", cooldown_bars=4),
]

# ══ §3  DELIVERY ════════════════════════════════════════════════════════════════════════════════
DELIVERY = dict(
    digest_minutes=15,               # normal alerts: at most one email this often
    catchup_max_age_minutes=30,      # found later than this after its bar closed (PC was off)
)                                    #   -> listed under "While offline", never sent as urgent
