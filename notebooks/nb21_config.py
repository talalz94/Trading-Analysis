"""
NOTEBOOK 21 — 1H BREAK CONFIRMATION.  THIS FILE IS YOURS.

Notebook 20's settings with one strategy turned on: a 1h trendline break must be CONFIRMED by a
later candle before the trade is taken, and the setup is cancelled if price closes back over the
candle that fired it. Everything else is identical to 20, so the confirmation is the only variable.

  §1 what and when   §2 the trade   §3 gaps   §4 S/R   §5 chart   §6 rewind

The rules themselves are documented in quant/scenarios/entries.py (_confirm_beyond); this file only
says which values are in force. If Jupyter offers "File Changed", pick RELOAD.
"""
import numpy as np, pandas as pd     # noqa: F401

# ── §1  WHAT AND WHEN ────────────────────────────────────────────────────────────────────────
SYMBOL, SOURCE = "BTCUSDT", "binance"
TF             = "5m"                # chart + execution timeframe
VIEW_START     = "2026-06-01"        # "2026-06-01"
END            = "2026-06-20"        # "2026-06-10"
WARMUP_START   = "2026-05-01"        # history for lines and gaps
TF_WARMUP_BARS = 800                 # per extra timeframe, in its own bars

# ── §2  THE TRADE ────────────────────────────────────────────────────────────────────────────
# 2.1  which lines
LINE_TFS  = ("15m", "1h")            # fitted AND traded (gap timeframes are §3)
TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0,
                 # a break must STICK for this many bars before the line is retired. 0 = the first
                 # close through kills it, which throws the line away on a poke nothing confirmed.
                 # A close back on the original side resets the count. max_distance_atr still
                 # retires a line price has simply walked away from.
                 break_sustain_bars=1)

# 2.2  the break — each is a run of n chart candles ending at the entry, measured against the LINE
TL_CLOSE_N    = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # closes beyond it
TL_FULL_M     = 0                    # whole candles beyond it
TL_TOUCH_K    = 0                    # prior touches the line must have
TL_DIR_N      = 0                    # candles of the trade's own colour
TL_DIR_SRC    = "chart"              # "chart" | "line" — whose candles TL_DIR_N counts
TL_DIR_FULL_N = 0                    # right colour AND wholly beyond
TL_COMBINE    = "all"                # "all" = every enabled one | "any"
TL_TOL_ATR    = 0.0                  # how far past counts as past, in ATR
TL_EXTEND     = True                 # a held line keeps its slope (only acts when hold > 0)
TL_ONCE_PER_LINE = False             # True = only the FIRST break of each line may trade

# 2.3  THE CONFIRMATION — measured against rs, the candle that fired, NOT against the line.
#      THIS IS WHAT NOTEBOOK 21 IS FOR.
TL_CONFIRM_FROM   = "first"          # which bar is rs: "first" = the bar the break STARTED on |
                                     #   "every" = every bar the line reads broken. A 1h line reads
                                     #   broken for a RUN of 5m bars; "every" lets the rule confirm
                                     #   against a small chop candle instead of the real break
TL_CONFIRM_N      = {"1h": 3, "15m": 3, "*": 0}   # candles of grace after rs, per LINE timeframe
TL_CONFIRM_ALL    = False            # False = ONE of the n confirms | True = all n, consecutively
TL_CONFIRM_MODE   = "close"          # "close" beyond rs | "full" = the whole candle beyond rs
TL_CONFIRM_COLOUR = True             # the confirming candle must be the trade's own colour
TL_CONFIRM_CANCEL = True             # rs dies if a candle CLOSES BACK OVER it — a bounce voids it

# 2.4  extra rules, ANDed onto the break
TRADE_LONG, TRADE_SHORT = True, True
ENTRY_RULES = [
    "long:  1 candles EMA_15m_100 close above",
    "short: 1 candles EMA_15m_100 close below",
]
RULES_COMBINE = "all"                # "all" = every rule for that side | "any"

# 2.5  the exit
STOP_N          = 20                 # stop at the extreme of the last N bars
STOP_BUFFER_PCT = 0.05               # pushed this % beyond it
BE_AT_RR        = 0                  # breakeven at this R; 0 = leave the stop alone
TP_RR           = 3.0                # target, in R
USE_RETEST_BAIL = False              # notebook 18's post-entry bail-out
SEC_TOUCH_N, SEC_WRONG_M, SEC_OPP_N = 2, 0, 1
SEC_MIN_PCT, SEC_REPLACE_TP         = 0.20, True

# 2.6  the money
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

# ── §3  FAIR VALUE GAPS ──────────────────────────────────────────────────────────────────────
FVG_TFS          = ("15m", "1h", "4h")   # drawn, and allowed to veto. May be finer than TF
FVG_MITIGATION   = "ce"              # "touch" | "ce" (midpoint) | "full" — when a gap dies
FVG_MIN_SIZE_ATR = 0.50              # keeps the larger half; the median gap is 0.49-0.55 ATR
FVG_MIN_DISP_ATR = 0.0               # the middle candle's body, in ATR. 0 = off
FVG_MAX_AGE_BARS = 500
FVG_ATR_PERIOD   = 14

FVG_DRAW   = True
FVG_EXTEND = True                    # an unmitigated box runs to the right edge
FVG_SHOW   = ("bull", "bear")

# the veto — measured and it does not work (docs/WHAT_WORKS.md). Here to experiment with.
USE_FVG_VETO      = False
FVG_VETO_MODE     = "retrace"        # "retrace" (gap behind) | "barrier" (gap ahead)
FVG_VETO_TFS      = ("1h", "4h")     # not the chart's own: 5m has ~96 gaps live at once
FVG_VETO_MAX_AGE  = 60               # in the gap's OWN timeframe's bars
FVG_VETO_MIN_SIZE = 0.0              # leave at 0 — these gates test only the NEAREST gap
FVG_VETO_MAX_DIST = 3.0              # in ATR
FVG_RULES = [                        # <opposite|same|bull|bear> [tf] [within N] [min X] [near X]
    # "short: opposite FVG 15m+1h within 20 candles min 0.8 ATR near 3 ATR",
]

# ── §4  SUPPORT / RESISTANCE ─────────────────────────────────────────────────────────────────
SHOW_SR_ZONES  = True                # layers start legendonly; the `S/R:` row switches them on
SR_TFS         = ("15m", "1h", "4h") # clustered per timeframe, in its own bar space
ZONE_LOOKBACK  = 2000                # bars of history per layer, IN ITS OWN BARS
SR_MAX_ZONES   = 40                  # per layer

# THE KNOB — tuned per timeframe by experiments/sr_zone_search.py; each layer gets its own TF's
# settings. "tight" = fewer, sharper levels | "normal" = the best | "loose" = more context.
# None = the pre-2026-09-27 zones exactly (backup: notebooks/backups/2026-09-27_pre_sr_tuning/).
# To pin one value, state it and it wins over the table:  SR_CLUSTER_PCT = 0.15, ZONES_LEFT = 10,
# ZONES_RIGHT = 10, SR_MIN_TOUCHES = 2, SR_BREAK_BARS = 4, SR_ANCHOR = True
SR_SENSITIVITY = "normal"

# ── §5  THE CHART ────────────────────────────────────────────────────────────────────────────
EMAS     = (20, 50, 100, 200)        # of TF
MTF_EMAS = [("1m", 20, 50, 100, 200),
            ("15m", 20, 50, 100, 200),
            ("1h", 20, 50, 100, 200)]
EMAS_ON  = False                     # drawn, but off at open
SHOW_TRENDLINES = True
START_VISIBLE   = ("trades",)        # + "fvg_bull" / "fvg_bear" / "sr_zones"
SHOW_REJECTED   = True               # draw the breaks that did NOT trade, one legend entry
                                     #   per reason. Always starts OFF the chart; costs a
                                     #   re-run of the entry conditions, so it is opt-in
HEIGHT          = 900

# dropdown caps — everything is DRAWN regardless; a plotly menu clips rather than scrolls
MAX_FVG_PICKER   = 6                 # per timeframe per direction
MAX_SR_PICKER    = 6                 # per timeframe
MAX_PICKER       = 24                # pattern/trendline instances, all timeframes
MAX_SHAPES_TF    = 12                # shapes per extra timeframe
MAX_TRADE_PICKER = 40

# ── §6  REWIND ───────────────────────────────────────────────────────────────────────────────
REWIND = True                        # click a candle in the browser; everything clips to it
AS_OF  = None                        # or "2026-06-05 20:00" to rebuild on truncated data
