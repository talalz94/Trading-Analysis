"""
ALERTS — TRENDLINES, WEDGES & TRIANGLES, NEWS. ONE EMAIL PER QUARTER HOUR AT MOST.

What it watches, on Gold, USDJPY and BTC:
  TRENDLINES  a line FORMS or price BREAKS it, on 1m 5m 15m 1h 4h, fitted STRICT: each line runs
              exactly through two real swing highs or lows and no candle closes through it before
              it is drawn (docs/CHART_GUIDE.md, "Trendlines")
  PATTERNS    rising / falling wedge, ascending / descending / symmetrical triangle FORMED and
              BROKEN OUT of, on 15m 1h 4h (1m and 5m are written below, switched off: ~40 a day)
  NEWS        high and critical releases: reminders 1 day, 12 h, 6 h, 2 h, 1 h, 30 and 15 min
              before; the evening preview, the weekly outlook and the results as before
  No candlestick alerts.

HOW THE EMAILS COME (§3 DELIVERY)
  Every 15 minutes at most: at :02 :17 :32 :47 UTC, once the quarter hour's candles have closed,
  ONE email with everything found in that quarter hour -- sections Chart patterns, Trendlines,
  News, in that order. A 1m or 5m signal waits for the quarter hour it falls in, wherever in it
  the candle closed. The same signal repeating -- five 1m closes through a line between :00 and
  :15 -- is ONE row: "5 times, 10:01-10:13 UTC". Nothing found, no email.
  A news reminder comes in the email of the quarter hour at or before its time: a 15-minute
  reminder for a release at 08:30 arrives at 08:17. Never after the release.

THE SUBJECT SAYS HOW IMPORTANT THE EMAIL IS -- it leads with the most important row in it:
  🔴 P1  act now    critical news (CPI, payrolls, rate decisions), or ANY high US news, within the
                    hour, and its result; a trendline break or pattern breakout on 1h / 4h
  🟠 P2  look soon  other high news within the hour; critical / US news 2-6 h ahead; a moved
                    release; a break on 15m; a line or pattern forming on 1h / 4h
  ⚪ P3  record     1m / 5m signals, 15m formations, reminders a day or 12 h ahead
  e.g.  [Alerts] 🔴 P1 · USD CPI m/m in 15 min · BTC 1h broke support line · +4 more · 13:17 UTC
  In Gmail, a filter on  subject:"P1"  can star it and mark it important (and send it to your
  phone); one on  subject:"P3"  can skip the inbox. Back to the plain subject: subject="classic".

CHARTS IN THE EMAIL (charts=3): the 3 most important signals are drawn at the top of the email --
the candles, the line or shape that fired, a star on the candle it fired on. ~40 KB each. On the
server they need matplotlib (the install script adds it); without it the email simply goes
without pictures and the log says why.

HOW TO USE IT
  1. LOOK FIRST. This sends nothing; it shows what WOULD have fired in the last 48 hours:
         python -m quant.alerts check --config alerts/alerts_config_trendlines.py
     then open alerts/state/preview.html in your browser.

  2. SWITCH TO IT. The monitor only ever reads alerts/alerts_config.py, so this file goes there
     (keep a backup of the old one). An alert it has not seen before starts from NOW, so nothing
     from the past is mailed. It needs the code of 20 Oct 2026 or later (1m bars, the 15-minute
     clock, the strict trendlines): older code refuses its settings BY NAME and keeps the old file.

     On the Google Cloud server, in the browser SSH window:
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
         cp alerts/alerts_config_before_trendlines.py alerts/alerts_config.py
     (on this PC: Copy-Item alerts/alerts_config_before_trendlines.py alerts/alerts_config.py)

  A typo cannot stop the monitor: a config that does not load is refused BY NAME in the log
  (python -m quant.alerts status) and the last good one stays in use.

CHANGE IT: flip on=True / False, add or drop a timeframe in tfs. Every setting, pattern name and
kind is in docs/ALERTS_CATALOGUE.md (or: python -m quant.alerts list).

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

# The email's sections, in this order (News last).
CATEGORIES = ("Chart patterns", "Trendlines", "News")

# ══ §2  ALERTS ══════════════════════════════════════════════════════════════════════════════════
#   on        True = watching · False = kept here, switched off
#   name      any name, unique (renaming an alert starts it afresh)
#   markets   "all", a group, or names from §1     tfs   "1m" "5m" "15m" "30m" "1h" "2h" "4h"
#   priority  "urgent" / "normal": marks the row; on the 15-minute clock both wait for it
#
#   ~N/day = how often it fired, as set: 5m-4h on Gold + USDJPY + BTC, Mar-Aug 2026; 1m on BTC
#   alone, Jun-Aug 2026 (the only 1m history kept) -- the three markets together are ~2-3x that.

STRICT = dict(fit="strict", touch_tol=0.3, wick_tol=0.15)
#   fit="strict"   through two real swing extremes; no candle closes through the line
#   touch_tol=0.3  a third swing within 0.3 ATR of the line counts as a touch
#   wick_tol=0.15  a wick may poke through by at most 0.15 ATR before the line is drawn

ALERTS = [

    # ── TRENDLINES ── "formed": a new line is drawn · "break": a candle CLOSES through a line
    #      (closes=2 would want two in a row). direction="up" = resistance lines only, "down" =
    #      support lines only; both by default.
    #      ~N/day   1m (BTC)  5m    15m   1h    4h
    #      formed    9.8      3.9   1.2   0.2   0.1
    #      broken    4.4      1.9   0.6   0.1   0.0
    dict(on=True,  name="trendline formed", kind="trendline", category="Trendlines",
         event="formed", **STRICT,
         markets="all", tfs=("1m", "5m", "15m", "1h", "4h"), priority="normal"),

    dict(on=True,  name="trendline broken", kind="trendline", category="Trendlines",
         event="break", closes=1, **STRICT,
         markets="all", tfs=("1m", "5m", "15m", "1h", "4h"), priority="urgent"),

    # ── WEDGES & TRIANGLES ── "formed": the shape is in place (its pivots confirmed); the note
    #      says where it would break and the measured target. "breakout": price CLOSED through
    #      it; the note gives the target and the stop. "both" = the two.
    #      fit="strict": each boundary through two real swing extremes, no candle closing through
    #      it before the shape forms; touch_tol=0.5: a third swing within 0.5 ATR of it counts.
    #      ~N/day formed  15m 0.5 · 1h 0.2 · 4h 0.0   (the default fit: 15m 3.0 · 1h 0.6 · 4h 0.1,
    #      including the Gold 15m "falling wedge" of 20 Oct whose top line price had closed above)
    # (named "formed" though it now fires on breakouts too: a renamed alert gets new email refs,
    #  and the refs in emails already sent would no longer open with `show`)
    dict(on=True,  name="wedge / triangle formed", kind="chart", event="both",
         fit="strict", touch_tol=0.5,
         patterns=("rising_wedge", "falling_wedge", "ascending_triangle",
                   "descending_triangle", "symmetrical_triangle"),
         markets="all", tfs=("15m", "1h", "4h"), priority="normal"),

    dict(on=False, name="wedge / triangle formed, 1m 5m", kind="chart", event="formed",
         fit="strict", touch_tol=0.5,
         patterns=("rising_wedge", "falling_wedge", "ascending_triangle",
                   "descending_triangle", "symmetrical_triangle"),
         markets="all", tfs=("1m", "5m"), priority="normal"),
]

# ══ §3  DELIVERY ════════════════════════════════════════════════════════════════════════════════
DELIVERY = dict(
    every_minutes=15,                # ONE email per quarter hour at most, at :02 :17 :32 :47 UTC,
                                     #   with everything found in it -- urgent and news included
    collapse_repeats=True,           # a signal repeating in that quarter hour is one row, with
                                     #   how many times and from when to when
    subject="tiers",                 # 🔴 P1 / 🟠 P2 / ⚪ P3 first (see the top of this file);
                                     #   "classic" = "N signals · markets · time"
    charts=3,                        # a picture of the top 3 signal rows, in the email: the line
                                     #   or shape that fired on its candles. 3 covered every row
                                     #   of every email in a 48 h test; 0 = no pictures
    catchup_max_age_minutes=30,      # found later than this after its candle closed (PC was off)
)                                    #   -> listed under "While offline", never sent as urgent

# ══ §4  NEWS ════════════════════════════════════════════════════════════════════════════════════
#   Driven by the economic calendar, not by candles. See what it covers right now:
#       python -m quant.alerts news
#   Results need JBLANKED_API_KEY in .env for non-US releases; US ones come from the agencies.

NEWS = dict(
    on=True,
    currencies={"USD": "medium", "EUR": "high", "GBP": "high", "JPY": "high"},
    #   more:   "AUD" "CAD" "CHF" "NZD" "CNY", or "*" for every currency not named
    #   levels: "low" "medium" "high" "critical" -- the LOWEST kept for that currency
    #           (critical = CPI, payrolls, rate decisions ...: `python -m quant.alerts list`)
    timezone="Asia/Karachi",                                # the times in your news emails
    week_ahead="Sun 20:00",                                 # the week's high + critical news
    day_ahead="20:00",                                      # tomorrow's covered news, each evening
    remind_before={"high": (1440, 720, 360, 120, 60, 30, 15)},   # minutes before; "high" covers
                                                            #   critical too, never medium
    results="high",                    # actual vs forecast, and what it means for the currency
    warn_alerts=30,                    # an alert within 30 min of high news says so
)                                      #   (None turns week_ahead / day_ahead / results off)
