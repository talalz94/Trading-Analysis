"""
ALERTS — TRENDLINES, WEDGES & TRIANGLES, NEWS. BREAKS AT ONCE, THE REST EVERY 5 MINUTES.

What it watches, on Gold, USDJPY and BTC:
  TRENDLINES  a line FORMS or price BREAKS it, on 1m 5m 15m 1h 4h, fitted STRICT: each line runs
              exactly through two real swing highs or lows and no candle closes through it before
              it is drawn (docs/CHART_GUIDE.md, "Trendlines"); 1m a little looser (§2, STRICT_1M)
  EARLY       a 15m / 1h / 4h line closed through by a 5m candle, a 5m line by a 1m candle -- up
              to an hour before the line's own candle confirms it (§2, "EARLY BREAKS")
  PATTERNS    rising / falling wedge, ascending / descending / symmetrical triangle FORMED and
              BROKEN OUT of, on 15m 1h 4h (1m and 5m are written below, switched off: ~40 a day)
  NEWS        high and critical releases: reminders 1 day, 12 h, 6 h, 2 h, 1 h, 30 and 15 min
              before; MEDIUM US releases (flash PMIs, jobless claims ...) 15 min before; the
              evening preview, the weekly outlook and the results as before
  No candlestick alerts.

HOW THE EMAILS COME (§3 DELIVERY)
  BREAKS AT ONCE: a trendline broken on 1m-4h, or an early break, is emailed as soon as its
  candle has closed -- the monitor looks at :20 past every minute, so ~20-30 seconds after the
  close -- with whatever else was already waiting (priority="instant" in §2).
  EVERYTHING ELSE every 5 minutes at most: at :02 :07 :12 ... UTC, once those 5 minutes' candles
  have closed, ONE email with everything found in them -- sections Chart patterns, Trendlines,
  News, in that order. A line forming goes in the next of them: a 1m one 20 s to ~5 minutes after
  its candle (its 1m feed is read every minute for the breaks), a 5m, 15m, 1h or 4h one ~2 minutes
  after its close. The same signal repeating in those 5 minutes is ONE row: "3 times,
  10:01-10:04 UTC". Nothing found, no email.
  ~52 emails a day either way (20-23 Oct 2026, last 3 days of candles): ~22 of them breaks sent
  at once, which take along what was waiting, so the 5-minute emails have less to carry.
  A news reminder comes in the email at or before its time: a 15-minute reminder for a release
  at 08:30 arrives at 08:17. Never after the release.

THE SUBJECT SAYS HOW IMPORTANT THE EMAIL IS -- it leads with the most important row in it:
  🔴 P1  act now    critical news (CPI, payrolls, rate decisions), or ANY high US news, within the
                    hour, and its result; a trendline break or pattern breakout on 1h / 4h
  🟠 P2  look soon  other high news within the hour; critical / US news 2-6 h ahead; a moved
                    release; a break on 15m; a line or pattern forming on 1h / 4h
  ⚪ P3  record     1m / 5m signals, 15m formations, reminders a day or 12 h ahead
  e.g.  [Alerts] 🔴 P1 · USD CPI m/m in 15 min · BTC 1h broke support line · +4 more · 13:17 UTC
  In Gmail, a filter on  subject:"P1"  can star it and mark it important (and send it to your
  phone); one on  subject:"P3"  can skip the inbox. Back to the plain subject: subject="classic".

PUSH NOTIFICATIONS (push="ntfy"): every email also pops up on your phone (the ntfy app) and PC
(ntfy.sh/app in Chrome or Edge) the moment it is composed -- the subject as the title, a line per
signal, P1 urgent / P2 high / P3 normal, a tap opens the chart. Needs NTFY_TOPIC in the server's
.env; `python -m quant.alerts test-push` checks it. Without it the emails go as before.

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
     from the past is mailed. It needs the code of 23 Oct 2026 or later (priority="instant"):
     older code refuses its settings BY NAME and keeps the old file.

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
#   priority  "instant" = emailed as soon as its candle closes (~20-30 s), not at the 5-minute
#             mark · "urgent" / "normal" = how the row is marked; both wait for the 5-minute mark
#
#   ~N/day = how often it fired, as set: 5m-4h on Gold + USDJPY + BTC, Mar-Aug 2026; 1m on BTC
#   alone, Jun-Aug 2026 (the only 1m history kept) -- the three markets together are ~2-3x that.

STRICT = dict(fit="strict", touch_tol=0.3, wick_tol=0.15)
#   fit="strict"   through two real swing extremes; no candle closes through the line
#   touch_tol=0.3  a third swing within 0.3 ATR of the line counts as a touch
#   wick_tol=0.15  a wick may poke through by at most 0.15 ATR before the line is drawn

STRICT_1M = dict(fit="strict", touch_tol=0.4, wick_tol=0.3, max_distance=20,
                 prefer_newer=True, hold=1, span=1/3)
#   1m is looser, because its ATR is tiny (Gold: ~1 point) and shrinks further in quiet hours:
#   touch_tol=0.4 / wick_tol=0.3  0.3 / 0.15 ATR asked for a hand-drawn Gold line's touches to sit
#                                 within 0.3 points, so the line of 21 Oct 17:17 never existed
#   max_distance=20               a line is dropped once price closes this many ATR away; at 8 the
#                                 same line was dropped at 18:03, in a 6.5-point rally that a 0.8-
#                                 point ATR made 7.8 ATR -- 45 minutes before the 18:48 break
#   prefer_newer=True             a new line with as many touches as the live one takes its place
#                                 if its last touch is more recent (one support and one resistance
#                                 line are live at a time). 22 Oct: a falling support held the slot
#                                 while a rising one formed under price; this lets the rising one in
#   hold=1                        a break needs TWO closes in a row; a close back after the first
#                                 keeps the line alive. On 1m ~70% of single-close breaks were
#                                 closed back across within 10 minutes; with hold=1, 42% of breaks
#                                 are still beyond the line 10 minutes later (31% before). Both
#                                 Gold lines still alert: 21 Oct at 18:51, 22 Oct at 03:15
#   span=1/3                      a line's touches must cover a third (not half) of the stretch the
#                                 fitter remembers -- its last 16 swing points, which after big
#                                 swings reach back hours. BTC 24 Oct: 01:38 / 02:20 / 02:53 = 75 min
#                                 against 111 needed, never drawn; now formed 02:59, broke 03:10.
#                                 1m lines a day, three markets: 29 -> 38 formed, 17 -> 21 broken;
#                                 the Gold lines above unchanged. A smaller swing size caught BTC
#                                 too but doubled the alerts and lost the 22 Oct Gold line

ALERTS = [

    # ── TRENDLINES ── "formed": a new line is drawn · "break": a candle CLOSES through a line
    #      (closes=2 would want two in a row). direction="up" = resistance lines only, "down" =
    #      support lines only; both by default.
    #      ~N/day   5m    15m   1h    4h       (Gold + USDJPY + BTC, Mar-Aug 2026)
    #      formed   3.9   1.2   0.2   0.1
    #      broken   1.9   0.6   0.1   0.0
    dict(on=True,  name="trendline formed", kind="trendline", category="Trendlines",
         event="formed", **STRICT,
         markets="all", tfs=("5m", "15m", "1h", "4h"), priority="normal"),

    dict(on=True,  name="trendline broken", kind="trendline", category="Trendlines",
         event="break", closes=1, **STRICT,
         markets="all", tfs=("5m", "15m", "1h", "4h"), priority="instant"),

    #      1m, PER MARKET per day (12-24 Oct 2026):  Gold 11.6 formed 6.0 broken · USDJPY 13.1 / 7.4
    #      · BTC 13.5 / 7.4   (before span=1/3: 8.6 / 4.8 · 10.2 / 5.9 · 10.4 / 6.0)
    dict(on=True,  name="trendline formed, 1m", kind="trendline", category="Trendlines",
         event="formed", **STRICT_1M,
         markets="all", tfs=("1m",), priority="normal"),

    dict(on=True,  name="trendline broken, 1m", kind="trendline", category="Trendlines",
         event="break", closes=1, **STRICT_1M,
         markets="all", tfs=("1m",), priority="instant"),

    # ── EARLY BREAKS ── a 15m / 1h / 4h line closed through by a 5m candle, and a 5m line by a
    #      1m candle: up to an hour before the line's own candle closes. The same lines as the
    #      alerts above (same STRICT fit); once per line; the line's own "broken" alert still
    #      comes when its candle closes -- the confirmation, ranked a step higher.
    #      22 Oct: USDJPY's 1h support (formed 21 Oct 20:00), first 5m close below at 03:05;
    #      the 1h alert came at 04:02, 10 pips lower. Early: 1h/4h lines P2, 15m and 5m lines P3
    dict(on=True,  name="early break, 15m-4h lines", kind="early_break", category="Trendlines",
         line_tfs=("15m", "1h", "4h"), **STRICT,
         markets="all", tfs=("5m",), priority="instant"),

    dict(on=True,  name="early break, 5m lines", kind="early_break", category="Trendlines",
         line_tfs=("5m",), **STRICT,
         markets="all", tfs=("1m",), priority="instant"),

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
    every_minutes=5,                 # ONE email every 5 minutes at most, at :02 :07 :12 ... UTC,
                                     #   with everything found in it -- urgent and news included;
                                     #   priority="instant" alerts do not wait for it (§2)
                                     #   15 = one a quarter hour: fewer emails, but a 1m break can
                                     #   arrive 17 min late (21 Oct: at the low, not 4 min after)
    collapse_repeats=True,           # a signal repeating in those 5 minutes is one row, with
                                     #   how many times and from when to when
    subject="tiers",                 # 🔴 P1 / 🟠 P2 / ⚪ P3 first (see the top of this file);
                                     #   "classic" = "N signals · markets · time"
    charts=3,                        # a picture of the top 3 signal rows, in the email: the line
                                     #   or shape that fired on its candles. 3 covered every row
                                     #   of every email in a 48 h test; 0 = no pictures
    catchup_max_age_minutes=30,      # found later than this after its candle closed (PC was off)
                                     #   -> listed under "While offline", never sent as urgent
    push="ntfy",                     # every email ALSO pops up on your phone and PC, at the same
)                                    #   moment: NTFY_TOPIC in .env, the ntfy app / ntfy.sh/app
                                     #   subscribed to it (docs/ALERTS.md, "Push notifications")

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
    remind_before={"high": (1440, 720, 360, 120, 60, 30, 15),   # minutes before; "high" covers
                   "medium": (15,)},                        #   critical too, never medium.
                                                            #   medium: only USD is covered at
                                                            #   medium (currencies above): 23 Oct's
                                                            #   13:45 flash PMIs broke Gold's line
    results="high",                    # actual vs forecast, and what it means for the currency
    warn_alerts=30,                    # an alert within 30 min of high news says so
)                                      #   (None turns week_ahead / day_ahead / results off)
