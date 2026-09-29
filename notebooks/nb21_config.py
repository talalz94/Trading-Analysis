"""
NOTEBOOK 21 — TRENDLINE BREAKS, CONFIRMED.  THIS FILE IS YOURS.

Short a break of a fitted support line (long: resistance) from 5m, 15m or 1h. A 15m or 1h break
must be CONFIRMED by a later candle before it trades. Two optional EXIT STRATEGIES can end a trade
before its target; each has a TAG that the chart writes on the exit marker.

  §1 market & dates   §2 entry   §3 exit, and the two exit strategies   §4 money   §5 A/B
  §6 gaps   §7 support / resistance   §8 chart

The rules are documented beside their code; this file only says which values are in force.
If Jupyter offers "File Changed", pick RELOAD.
"""
import numpy as np, pandas as pd     # noqa: F401

# ══ §1  MARKET & DATES ═══════════════════════════════════════════════════════════════════════
SYMBOL, SOURCE = "BTCUSDT", "binance"
TF             = "5m"                # chart and execution timeframe
VIEW_START     = "2026-06-01"        # the chart starts here
END            = "2026-06-20"        # ...and ends here
WARMUP_START   = "2026-05-01"        # history for lines and gaps; never traded
TF_WARMUP_BARS = 800                 # history per extra timeframe, in its own bars

# ══ §2  ENTRY — a line breaks, the break is confirmed, the extra rules agree ═══════════════════
# 2.1  which lines
LINE_TFS  = ("15m", "1h")            # fitted and drawn, as well as TF itself
TRADE_TFS = ("5m", "15m", "1h")      # whose line breaks may TRADE; the rest are only drawn
TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0,
                 break_sustain_bars=1)   # closes beyond before a line retires; 0 = the first one

# 2.2  the break — counted over the candles ending at the entry, against the LINE
TL_CLOSE_N       = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # closes beyond it
TL_FULL_M        = 0                 # whole candles beyond it
TL_TOUCH_K       = 0                 # prior touches the line must have
TL_DIR_N         = 0                 # candles of the trade's own colour
TL_DIR_SRC       = "chart"           # "chart" | "line" — whose candles TL_DIR_N counts
TL_DIR_FULL_N    = 0                 # right colour AND wholly beyond
TL_COMBINE       = "all"             # "all" = every test above | "any"
TL_TOL_ATR       = 0.0               # how far past counts as past, in ATR
TL_EXTEND        = True              # a held line keeps its slope
TL_ONCE_PER_LINE = False             # True = only a line's FIRST break may trade

# 2.3  the confirmation — against rs, the candle that FIRED, not against the line
TL_CONFIRM_FROM   = "first"          # rs = the bar the break started on | "every" bar it reads broken
TL_CONFIRM_N      = {"1h": 3, "15m": 3, "5m": 2, "*": 0}   # candles of grace after rs, per line timeframe
TL_CONFIRM_ALL    = False            # False = ONE of them confirms | True = all of them, in a row
TL_CONFIRM_MODE   = "close"          # "close" beyond rs | "full" = the whole candle beyond it
TL_CONFIRM_COLOUR = True             # the confirming candle must be the trade's own colour
TL_CONFIRM_CANCEL = True             # a close back over rs cancels the setup

# 2.4  extra rules, ANDed onto the break
TRADE_LONG, TRADE_SHORT = True, True
ENTRY_RULES = [
    "long:  1 candles EMA_15m_100 close above",
    "short: 1 candles EMA_15m_100 close below",
]
RULES_COMBINE = "all"                # "all" = every rule for that side | "any"

# ══ §3  EXIT — stop, target, and two optional exit strategies ═════════════════════════════════
# 3.1  stop and target — chart tags: TP = target reached, SL = stopped, BE = stopped at breakeven
STOP_N          = 20                 # stop at the extreme of the last N bars...
STOP_BUFFER_PCT = 0.05               # ...pushed this % beyond it
TP_RR           = 3.0                # the target, in R
BE_AT_RR        = 0                  # move the stop to entry at this R; 0 = never

# 3.2  EXIT STRATEGY 1 — a near miss, or a long wait, moves the target closer
USE_TP_MOVE = False
TP_MOVE     = dict(tag="MOVED",      # the chart says TP·MOVED
                   near=0.8,         # price got 80% of the way to the target...
                   back=0.3,         # ...then a close gave back 30% of that distance
                   after_bars=0,     # or: the trade has been open this many bars (0 = off)
                   to=0.6)           # -> the target moves to 60% of its distance (3R -> 1.8R)

# 3.3  EXIT STRATEGY 2 — close when price fills a gap sitting near the target
USE_FVG_EXIT = True
FVG_EXIT     = dict(tag="FVG",       # the chart says TP·FVG
                    tfs=("15m", "1h", "4h"),   # whose gaps count, detected as §6 says
                    near_r=1.0,      # the gap lies within this many R of the target
                    at="ce")         # exit at its "touch" | "ce" (middle) | "full" (far edge)

# 3.4  notebook 18's post-entry bail-out (off)
USE_RETEST_BAIL = False
SEC_TOUCH_N, SEC_WRONG_M, SEC_OPP_N = 2, 0, 1
SEC_MIN_PCT, SEC_REPLACE_TP         = 0.20, True

# ══ §4  MONEY ════════════════════════════════════════════════════════════════════════════════
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

# ══ §5  A/B — the notebook's comparison table; each arm changes only what it names ═════════════
COMPARE = {
    "neither":     dict(USE_TP_MOVE=False, USE_FVG_EXIT=False),
    "move target": dict(USE_TP_MOVE=True,  USE_FVG_EXIT=False),
    "fvg exit":    dict(USE_TP_MOVE=False, USE_FVG_EXIT=True),
    "both":        dict(USE_TP_MOVE=True,  USE_FVG_EXIT=True),
}

# ══ §6  FAIR VALUE GAPS — drawn, and read by the veto and the FVG exit ═════════════════════════
FVG_TFS          = ("15m", "1h", "4h")   # may be finer than TF
FVG_MITIGATION   = "ce"              # "touch" | "ce" (midpoint) | "full" — when a gap dies
FVG_MIN_SIZE_ATR = 0.50              # keeps the larger half of gaps
FVG_MIN_DISP_ATR = 0.0               # the middle candle's body, in ATR; 0 = off
FVG_MAX_AGE_BARS = 1000              # in each gap's own bars
FVG_ATR_PERIOD   = 14
FVG_DRAW         = True
FVG_EXTEND       = True              # an unfilled box runs to the right edge
FVG_SHOW         = ("bull", "bear")

# the veto — measured, and it does not work (docs/WHAT_WORKS.md)
USE_FVG_VETO      = False
FVG_VETO_MODE     = "retrace"        # "retrace" (gap behind) | "barrier" (gap ahead)
FVG_VETO_TFS      = ("1h", "4h")     # not the chart's own: 5m has ~96 gaps live at once
FVG_VETO_MAX_AGE  = 60               # in the gap's own bars
FVG_VETO_MIN_SIZE = 0.0
FVG_VETO_MAX_DIST = 3.0              # in ATR
FVG_RULES = []                       # e.g. "short: opposite FVG 15m+1h within 20 candles"

# ══ §7  SUPPORT / RESISTANCE — drawn, never traded ═══════════════════════════════════════════
SHOW_SR_ZONES  = True                # start hidden; the `S/R:` buttons show them
SR_TFS         = ("15m", "1h", "4h")
ZONE_LOOKBACK  = 2000                # bars of history per layer, in its own bars
SR_MAX_ZONES   = 40                  # per layer
SR_SENSITIVITY = "normal"            # "tight" | "normal" | "loose" | None = the pre-2026-09-27 zones

# ══ §8  CHART ════════════════════════════════════════════════════════════════════════════════
EXIT_LABELS     = True               # write TP / SL / BE / TP·MOVED / TP·FVG on every exit marker
EMAS            = (20, 50, 100, 200) # of TF
MTF_EMAS        = [("1m", 20, 50, 100, 200),
                   ("15m", 20, 50, 100, 200),
                   ("1h", 20, 50, 100, 200)]
EMAS_ON         = False              # drawn, but hidden at open
SHOW_TRENDLINES = True
SHOW_REJECTED   = True               # the breaks that did NOT trade; start hidden
START_VISIBLE   = ("trades",)        # + "fvg_bull" / "fvg_bear" / "sr_zones"
HEIGHT          = 900
REWIND          = True               # click a candle; everything clips to what was known there
AS_OF           = None               # or "2026-06-05 20:00" to rebuild on data up to then

# dropdown caps — everything is still drawn; a plotly menu clips rather than scrolls
MAX_FVG_PICKER   = 6                 # per timeframe per direction
MAX_SR_PICKER    = 6                 # per timeframe
MAX_PICKER       = 24                # line instances, all timeframes
MAX_SHAPES_TF    = 12                # shapes per extra timeframe
MAX_TRADE_PICKER = 40
