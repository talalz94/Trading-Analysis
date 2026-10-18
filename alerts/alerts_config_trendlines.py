"""
ALERTS — TRENDLINES AND A FEW CHOSEN PATTERNS.  A READY-TO-USE COPY OF alerts_config.py.

What it watches, on the same markets as alerts_config.py (Gold, USDJPY, BTC):
  TRENDLINES  fitted STRICT: each line runs exactly through two real swing highs or lows, and no
              candle between its first touch and the moment it is drawn closes through it. (The
              default fit only checks the swings, so a wick could cross a "support" by two ATR --
              docs/CHART_GUIDE.md, "Trendlines".)
  PATTERNS    engulfing, hammer / shooting star and morning / evening star on 4h; double tops /
              bottoms and head & shoulders on 1h and 4h
  NEWS        exactly as in alerts_config.py
  About 4.7 alerts a day with the switches as set; most go in the digest, not one email each.

HOW TO USE IT -- three steps
  1. LOOK FIRST. This sends nothing; it shows what WOULD have fired in the last 48 hours:
         python -m quant.alerts check --config alerts/alerts_config_trendlines.py
     then open alerts/state/preview.html in your browser.

  2. SWITCH TO IT. The monitor only ever reads alerts/alerts_config.py, so this file goes there
     (keep a backup of the old one). An alert it has not seen before starts from NOW, so nothing
     from the past is mailed.

     The trendline alerts use settings added on 18 Oct 2026 (fit= / touch_tol= / wick_tol=). A
     monitor running OLDER code refuses them by name and keeps its old config, so the code goes
     first, then the file.

     On the Google Cloud server, in the browser SSH window (once this code has been pushed):
         cd ~/quant && git stash && git pull && git stash pop && bash scripts/install_alerts_linux.sh
     That updates the code and restarts the monitor; this file arrives with it. Then:
         cp alerts/alerts_config.py alerts/alerts_config_before_trendlines.py
         cp alerts/alerts_config_trendlines.py alerts/alerts_config.py
         .venv/bin/python -c "from quant.alerts.config import load_config; load_config(); print('config OK')"
     (a mistake is named instead of "config OK"). The monitor picks it up within a minute.

     On this PC, in PowerShell, from the project folder:
         Copy-Item alerts/alerts_config.py alerts/alerts_config_before_trendlines.py
         Copy-Item alerts/alerts_config_trendlines.py alerts/alerts_config.py
     and, if the monitor's scheduled task is running here, restart it once so it loads the code:
         Stop-ScheduledTask quant-alerts; Start-ScheduledTask quant-alerts

  3. GOING BACK is the same copy the other way round:
         Copy-Item alerts/alerts_config_before_trendlines.py alerts/alerts_config.py
     (on the server: cp alerts/alerts_config_before_trendlines.py alerts/alerts_config.py)

  A typo cannot stop the monitor: a config that does not load is refused BY NAME in the log
  (python -m quant.alerts status) and the last good one stays in use.

CHANGE IT like the original: flip on=True / False, add a timeframe to tfs, move an alert between
"urgent" (emailed at once) and "normal" (the next digest). Every setting, pattern name and kind is
in docs/ALERTS_CATALOGUE.md (or: python -m quant.alerts list).

  §1 MARKETS     §2 ALERTS: trendlines, then patterns     §3 DELIVERY     §4 NEWS
"""

# ══ §1  MARKETS ═════════════════════════════════════════════════════════════════════════════════
#   "any name you like": dict(symbol=..., source=...)
#   source  "binance"   = crypto (any Binance spot pair: BTCUSDT, ETHUSDT ...)
#           "dukascopy" = gold, silver, FX, oil, indices (XAUUSD, EURUSD, WTI, US500, GER40 ...)

MARKETS = {
    "Gold":     dict(symbol="XAUUSD",  source="dukascopy"),
    "USDJPY":   dict(symbol="USDJPY",  source="dukascopy"),
    "BTC":      dict(symbol="BTCUSDT", source="binance"),
    # "ETH":      dict(symbol="ETHUSDT", source="binance"),
    # "S&P 500":  dict(symbol="US500",   source="dukascopy"),
}

GROUPS = {                           # a name markets= can use; "all" = every market above
    "metals & FX": ("Gold", "USDJPY"),
}

# ══ §2  ALERTS ══════════════════════════════════════════════════════════════════════════════════
#   on        True = watching · False = kept here, switched off
#   name      any name, unique (renaming an alert starts it afresh)
#   markets   "all", a group, or names from §1     tfs   "5m" "15m" "30m" "1h" "2h" "4h"
#   priority  "urgent" = email at once · "normal" = next digest
#   cooldown_bars  (optional) after firing, quiet for this many of that timeframe's bars
#
#   ~N/day = how often it fired on Gold + USDJPY + BTC together, Mar-Aug 2026, as set here.

STRICT = dict(fit="strict", touch_tol=0.3, wick_tol=0.15)
#   fit="strict"   through two real swing extremes; no candle closes through the line
#   touch_tol=0.3  a third swing within 0.3 ATR of the line counts as a touch (default fit: 0.6)
#   wick_tol=0.15  a wick may poke through by at most 0.15 ATR before the line is drawn
#   Delete **STRICT from an alert to get the default fit back (about twice as many lines).

ALERTS = [

    # ── TRENDLINES ── event: "formed" (a new line is drawn) · "break" (price breaks it by a RULE)
    #      the rule is counted from the first close through:
    #      closes=N  N closes in a row beyond · full=N  N whole candles beyond (wicks too)
    #      colour=N  N candles of the break's colour · confirm=N  then, within N candles, one of
    #      the break's colour must close beyond the WHOLE candle that broke it (notebook 21's rule)
    #      direction="up" = resistance lines only · "down" = support lines only · default both
    dict(on=True,  name="trendline formed, 1h/4h", kind="trendline",               # ~0.3/day
         event="formed", **STRICT, markets="all", tfs=("1h", "4h"), priority="normal"),

    dict(on=True,  name="trendline broken, 2 closes, 1h/4h", kind="trendline",    # ~0.1/day
         event="break", closes=2, **STRICT, markets="all", tfs=("1h", "4h"), priority="urgent"),

    dict(on=True,  name="15m trendline break, confirmed", kind="trendline",        # ~0.3/day
         event="break", closes=1, confirm=3, **STRICT, markets="all", tfs="15m",
         priority="urgent"),

    dict(on=False, name="15m trendline formed", kind="trendline",                   # ~1.2/day
         event="formed", **STRICT, markets="all", tfs="15m", priority="normal"),

    dict(on=False, name="5m trendline, 2 closes beyond", kind="trendline",         # ~1.4/day
         event="break", closes=2, **STRICT, markets="all", tfs="5m", priority="urgent"),

    # ── CANDLE PATTERNS ── on the bar the pattern COMPLETES (docs/CHART_GUIDE.md lists all 62)
    dict(on=True,  name="engulfing 4h", kind="candle",                              # ~2.5/day
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs="4h", priority="urgent"),

    dict(on=True,  name="hammer / shooting star 4h", kind="candle",                 # ~0.6/day
         patterns=("hammer", "shooting_star"),
         markets="all", tfs="4h", priority="normal"),

    dict(on=True,  name="morning / evening star 4h", kind="candle",                 # ~0.3/day
         patterns=("morning_star", "evening_star"),
         markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="engulfing 1h", kind="candle",                              # ~9.6/day
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs="1h", priority="normal"),

    # ── CHART PATTERNS ── event: "formed" (shape in place) · "breakout" (close through) · "both"
    dict(on=True,  name="tops & bottoms, 1h/4h", kind="chart", event="both",       # ~0.6/day
         patterns=("double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders"),
         markets="all", tfs=("1h", "4h"), priority="urgent"),
]

# ══ §3  DELIVERY ════════════════════════════════════════════════════════════════════════════════
DELIVERY = dict(
    digest_minutes=15,               # normal alerts: at most one email this often
    catchup_max_age_minutes=30,      # found later than this after its candle closed (PC was off)
)                                    #   -> listed under "While offline", never sent as urgent

# ══ §4  NEWS ════════════════════════════════════════════════════════════════════════════════════
#   Driven by the economic calendar, not by candles. See what it covers right now:
#       python -m quant.alerts news
#   ~24 emails a week as set (measured on 2024's calendar): ~19 reminders + results, ~4 evening
#   previews, 1 weekly outlook. Results need JBLANKED_API_KEY in .env; without it they are skipped.

NEWS = dict(
    on=True,
    currencies={"USD": "medium", "EUR": "high", "GBP": "high", "JPY": "high"},
    #   more:   "AUD" "CAD" "CHF" "NZD" "CNY", or "*" for every currency not named
    #   levels: "low" "medium" "high" "critical" -- the LOWEST kept for that currency
    #           (critical = CPI, payrolls, rate decisions ...: `python -m quant.alerts list`)
    timezone="Asia/Karachi",                                # the times in your news emails
    week_ahead="Sun 20:00",                                 # the week's high + critical news
    day_ahead="20:00",                                      # tomorrow's covered news, each evening
    remind_before={"critical": (60, 15), "high": (15,)},    # minutes before the release
    results="high",                    # actual vs forecast, and what it means for the currency
    warn_alerts=30,                    # a candle alert within 30 min of high news says so
)                                      #   (None turns week_ahead / day_ahead / results off)
