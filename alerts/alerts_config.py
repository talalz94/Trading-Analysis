"""
ALERTS — WHAT TO WATCH, AND WHEN TO EMAIL YOU.  THIS FILE IS YOURS.

  Change it   edit and save; the running monitor picks it up within a minute
  Test it     python -m quant.alerts check    what WOULD have fired in the last 48h (sends nothing)
  Options     python -m quant.alerts list     every instrument, pattern, kind and setup there is
                                              (also in docs/ALERTS_CATALOGUE.md)

  §1 MARKETS     which instruments
  §2 ALERTS      what to look for -- each one has an on/off switch
  §3 DELIVERY    when to email
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
#   kind      what to look for -- its own settings follow it (python -m quant.alerts list):
#               candle · chart · trendline · sr_zone · fvg · ma_cross · rsi · rsi_divergence ·
#               macd · stochastic · supertrend · heikin_ashi · candle_run · big_candle · setup
#   markets   "all", a group, or names from §1     tfs   "5m" "15m" "30m" "1h" "2h" "4h"
#   priority  "urgent" = email at once · "normal" = next digest
#   cooldown_bars  (optional) after firing, quiet for this many of that timeframe's bars
#
#   ~N/day = how often it fired on Gold + USDJPY + BTC together, Mar-Aug 2026, as set here.

ALERTS = [

    # ── CANDLE PATTERNS ────────────────────────────────────────────────────────────────────────
    dict(on=True,  name="engulfing 4h", kind="candle",                              # ~2.5/day
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs="4h", priority="urgent"),

    dict(on=True,  name="engulfing 1h", kind="candle",                              # ~9.6/day
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs="1h", priority="normal"),

    dict(on=True,  name="hammer / shooting star", kind="candle",                    # ~11/day
         patterns=("hammer", "shooting_star"),
         markets="all", tfs=("15m", "1h", "4h"), priority="normal", cooldown_bars=4),

    dict(on=False, name="morning / evening star", kind="candle",                    # ~1.4/day
         patterns=("morning_star", "evening_star"),
         markets="all", tfs=("1h", "4h"), priority="normal"),

    # ── CHART PATTERNS ── event: "formed" (shape in place) · "breakout" (close through) · "both"
    dict(on=False, name="tops & bottoms", kind="chart", event="both",               # ~0.6/day
         patterns=("double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders"),
         markets="all", tfs=("1h", "4h"), priority="urgent"),

    # ── LEVELS & LINES ── a break can need a RULE, counted from the first close through:
    #      closes=N  N closes in a row beyond · full=N  N whole candles beyond (wicks too)
    #      colour=N  N candles of the break's colour (red down, green up) · all that are set
    dict(on=False, name="5m trendline: 2 closes beyond", kind="trendline",          # ~2.4/day
         event="break", closes=2, markets="all", tfs="5m", priority="urgent"),

    dict(on=False, name="trendline formed", kind="trendline", event="formed",       # ~0.6/day
         markets="all", tfs=("1h", "4h"), priority="normal"),

    dict(on=False, name="trendline break", kind="trendline", min_touches=3,         # ~0.3/day
         markets="all", tfs=("1h", "4h"), priority="normal"),

    dict(on=False, name="S/R zone: 2 closes beyond", kind="sr_zone", event="break", # ~0.8/day
         closes=2, markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="big FVG formed", kind="fvg", event="formed",               # ~1.3/day
         min_size_atr=1.0, markets="all", tfs=("1h", "4h"), priority="normal"),

    # ── INDICATORS ─────────────────────────────────────────────────────────────────────────────
    dict(on=False, name="price crosses EMA 200", kind="ma_cross", ma="ema",         # ~0.4/day
         period=200, markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="golden / death cross", kind="ma_cross", ma="sma",          # ~0.1/day
         fast=50, slow=200, markets="all", tfs="4h", priority="urgent"),

    dict(on=False, name="RSI leaves overbought / oversold", kind="rsi",             # ~0.4/day
         when="exit", above=70, below=30, markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="RSI divergence", kind="rsi_divergence",                    # ~0.2/day
         markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="MACD signal cross", kind="macd", cross="signal",           # ~1.1/day
         markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="stochastic cross in extreme", kind="stochastic",           # ~1.1/day
         above=80, below=20, markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="Supertrend flip", kind="supertrend",                       # ~0.3/day
         markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="Heikin-Ashi flip", kind="heikin_ashi", after=4,            # ~1.6/day
         markets="all", tfs="4h", priority="normal"),

    dict(on=False, name="5 candles in a row", kind="candle_run", n=5,               # ~1.5/day
         markets="all", tfs="1h", priority="normal"),

    dict(on=False, name="big candle", kind="big_candle", atr_mult=3.0,              # ~1.1/day
         markets="all", tfs="1h", priority="urgent"),

    # ── SETUPS ── research entries from quant/scenarios (28 of them: see `list`) ──────────────
    dict(on=False, name="break of structure", kind="setup", setup="bos_long@v1",    # ~0.6/day
         markets="all", tfs="1h", priority="normal"),

    dict(on=False, name="squeeze fires", kind="setup", setup="squeeze_long@v1",     # ~1.0/day
         markets="all", tfs="1h", priority="normal"),
]

# ══ §3  DELIVERY ════════════════════════════════════════════════════════════════════════════════
DELIVERY = dict(
    digest_minutes=15,               # normal alerts: at most one email this often
    catchup_max_age_minutes=30,      # found later than this after its candle closed (PC was off)
)                                    #   -> listed under "While offline", never sent as urgent
