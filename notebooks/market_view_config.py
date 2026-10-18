"""
MARKET VIEW — one instrument's latest days on a 1-minute chart, with trendlines, support/resistance
zones and fair value gaps from five timeframes, and the economic calendar along the bottom.
No strategy, no trades.  THIS FILE IS YOURS.

    §1  WHICH INSTRUMENT      pick a name; add your own
    §2  WHICH WINDOW          the chart's timeframe and how many days back from the latest candle
    §3  WHAT IS DRAWN         which timeframes give trendlines, gaps and zones
    §4  TRENDLINES            how a line is fitted
    §5  FAIR VALUE GAPS       what counts as a gap, and when it dies
    §6  SUPPORT / RESISTANCE  how many levels
    §7  NEWS                  the release strip
    §8  THE CHART             size, menus, rewind

Run notebooks/market_view.ipynb after changing anything here. If Jupyter offers "File Changed",
pick RELOAD.
"""

# ── §1  WHICH INSTRUMENT ─────────────────────────────────────────────────────────────────────
INSTRUMENT = "BTC"                   # one of the names below

# "any name you like": dict(symbol=..., source=...) -- the same shape as alerts/alerts_config.py
#   source "binance"   = crypto, any Binance spot pair (BTCUSDT, ETHUSDT, SOLUSDT ...)
#   source "dukascopy" = gold, silver, FX, oil, indices and their futures; any of Dukascopy's
#                        1,380 codes works as given (python -m quant.alerts list shows them)
INSTRUMENTS = {
    "BTC":    dict(symbol="BTCUSDT", source="binance"),
    "ETH":    dict(symbol="ETHUSDT", source="binance"),
    "NASDAQ": dict(symbol="NAS100",  source="dukascopy"),   # Nasdaq 100 futures (E_NQ-100)
    "SP500":  dict(symbol="US500",   source="dukascopy"),   # S&P 500 futures (E_SandP-500)
    "DOW":    dict(symbol="US30",    source="dukascopy"),   # Dow Jones futures
    "GOLD":   dict(symbol="XAUUSD",  source="dukascopy"),   # spot gold
    "SILVER": dict(symbol="XAGUSD",  source="dukascopy"),
    "OIL":    dict(symbol="WTI",     source="dukascopy"),   # WTI crude (E_Light)
    "USDJPY": dict(symbol="USDJPY",  source="dukascopy"),
    "EURUSD": dict(symbol="EURUSD",  source="dukascopy"),
    "GBPUSD": dict(symbol="GBPUSD",  source="dukascopy"),
}

# ── §2  WHICH WINDOW ─────────────────────────────────────────────────────────────────────────
TF   = "1m"                          # the chart's candles
DAYS = 2                             # the last N days of TRADING, ending at the latest closed
                                     # candle: counted in candles, so a Monday morning on gold
                                     # reaches back past the weekend instead of showing it empty

TF_WARMUP_BARS = 800                 # history before the window, in each timeframe's OWN bars:
                                     # lines and zones need pivots from before the first candle
                                     # you see. 800 is what the lab uses (a 4h line can span
                                     # 100-200 bars); it is computed, never drawn

# ── §3  WHAT IS DRAWN — the chart's own timeframe must be in each list whose feature is on ─────
SHOW_TRENDLINES = True
LINE_TFS        = ("1m", "5m", "15m", "1h", "4h")

FVG_DRAW = True
FVG_TFS  = ("1m", "5m", "15m", "1h", "4h")

SHOW_SR_ZONES = True
SR_TFS        = ("1m", "5m", "15m", "1h", "4h")

# ── §4  TRENDLINES — the strategy file's values, but fitted STRICT (docs/CHART_GUIDE.md) ───────
TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.3,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0,
                 break_sustain_bars=1,   # a break must hold one more close before the line dies
                 fit="strict",           # no candle closes through a line; "ransac" = the old fit
                 wick_tol_atr=0.15)      # how far a wick may poke through it, in ATR

# ── §5  FAIR VALUE GAPS — the strategy file's values ───────────────────────────────────────────
FVG_MITIGATION   = "ce"              # "touch" | "ce" (midpoint) | "full" — when a gap dies
FVG_MIN_SIZE_ATR = 0.50              # keeps the larger half; the median gap is 0.49-0.55 ATR
FVG_MIN_DISP_ATR = 0.0               # the middle candle's body, in ATR. 0 = off
FVG_MAX_AGE_BARS = 1000              # in each gap's own bars
FVG_ATR_PERIOD   = 14
FVG_EXTEND       = True              # an unmitigated box runs to the right edge
FVG_SHOW         = ("bull", "bear")

# ── §6  SUPPORT / RESISTANCE ─────────────────────────────────────────────────────────────────
SR_SENSITIVITY = "normal"            # "tight" (2-4 lines) | "normal" (4-8) | "loose" (8-16)
ZONE_LOOKBACK  = 2000                # bars of history per layer, IN ITS OWN BARS
SR_MAX_ZONES   = 40                  # per layer

# ── §7  NEWS — economic releases along the bottom of the chart (docs/NEWS.md) ──────────────────
NEWS        = {"USD": "medium", "EUR": "high", "GBP": "high", "JPY": "high"}   # None = no strip
NEWS_SHOW   = "high"                 # opens on: "critical" | "high" (+critical) | "medium" | "all"
NEWS_TZ     = "Asia/Karachi"         # the zone the hover and the summary read in
NEWS_UPDATE = True                   # refresh the calendar before drawing (a few seconds; the
                                     # daily scheduled task keeps it current anyway)

# ── §8  THE CHART ────────────────────────────────────────────────────────────────────────────
HEIGHT = 900

# which layers are switched ON when the chart opens. Everything in §3 is drawn either way; the rest
# start in the legend, one click away, as do the `FVGs:` and `S/R:` button rows. The 1m layer alone
# holds ~600 gaps in two days of BTC, so opening on it hides the price
OPEN_WITH = dict(trendlines=("15m", "1h", "4h"),
                 gaps=("15m", "1h", "4h"),
                 zones=("1h", "4h"))

EMAS = ()                            # e.g. (20, 50, 200) — none for now
REWIND        = True                 # click a candle in the browser; everything clips to it

# dropdown caps — everything is DRAWN regardless; a plotly menu clips rather than scrolls
MAX_FVG_PICKER = 6                   # per timeframe per direction
MAX_SR_PICKER  = 6                   # per timeframe
MAX_PICKER     = 24                  # trendline instances, all timeframes
MAX_SHAPES_TF  = 12                  # per extra timeframe
