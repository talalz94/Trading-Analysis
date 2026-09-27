"""
STRATEGY — TRENDLINE BREAK, CONFIRMED.  THIS FILE IS YOURS.

Short a break of a fitted support line (long a break of resistance) from 5m, 15m or 1h, with the
15m 100-EMA on the trade's side. A 15m or 1h break must be CONFIRMED by a later candle closing
beyond the one that fired it, and is cancelled if price closes back over that candle. Stop at the
20-bar extreme, target 3R.

Only WHAT is traded lives here. Which window you look at and how it is drawn is lab_config.py;
which quarters judge it is evaluate_config.py. A new idea is a toggle here plus an arm in COMPARE,
or a copy of this file under a new name -- never a new notebook.

  §1 market   §2 the trade   §3 gaps   §4 A/B

Started 2026-09-27 as notebook 21's settings, value for value. If Jupyter offers "File Changed",
pick RELOAD.
"""

# ── §1  MARKET ───────────────────────────────────────────────────────────────────────────────
SYMBOL, SOURCE = "BTCUSDT", "binance"
TF             = "5m"                # execution timeframe
TF_WARMUP_BARS = 800                 # history per extra timeframe, in its own bars

# ── §2  THE TRADE ────────────────────────────────────────────────────────────────────────────
# 2.1  which lines
LINE_TFS  = ("15m", "1h")            # fitted AND traded (gap timeframes are §3)
TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0,
                 # a break must STICK for this many bars before the line is retired. 0 = the first
                 # close through kills it, which throws the line away on a poke nothing confirmed.
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

# 2.3  the confirmation — measured against rs, the candle that fired, NOT against the line
TL_CONFIRM_FROM   = "first"          # rs = the bar the break STARTED on | "every" = any bar it reads broken
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

# 2.6  optional exits — both off; neither beat noise over 19 quarters (docs/WHAT_WORKS.md)
USE_TP_MOVE  = False                 # a near miss, or a long wait, moves the target closer
TP_MOVE      = dict(near=0.8, back=0.3,    # got 80% of the way, then a close gave back 30%...
                    after_bars=0,          # ...or held this many bars (0 = off)
                    to=0.6)                # -> target moves to 60% of its distance (3R -> 1.8R)
USE_FVG_EXIT = False                 # close when price fills a gap sitting near the target
FVG_EXIT     = dict(tfs=("15m", "1h", "4h"),   # whose gaps, detected as §3 says
                    near_r=1.0,            # the gap lies within this many R of the target
                    at="ce")               # exit at its "touch" | "ce" (middle) | "full" (far edge)

# 2.7  the money
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

# ── §3  FAIR VALUE GAPS — what the rules know about gaps (the lab draws these) ──────────────────
FVG_TFS          = ("15m", "1h", "4h")   # detected (and drawn). May be finer than TF
FVG_MITIGATION   = "ce"              # "touch" | "ce" (midpoint) | "full" — when a gap dies
FVG_MIN_SIZE_ATR = 0.50              # keeps the larger half; the median gap is 0.49-0.55 ATR
FVG_MIN_DISP_ATR = 0.0               # the middle candle's body, in ATR. 0 = off
FVG_MAX_AGE_BARS = 1000              # in each gap's own bars
FVG_ATR_PERIOD   = 14

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

# ── §4  A/B — the questions this strategy is asking ─────────────────────────────────────────────
# One run per arm, each changing ONLY what it names. The lab shows them on its window; evaluate
# judges them over the quarters. An arm that changes nothing IS the strategy as set.
COMPARE = {
    "neither":         dict(USE_TP_MOVE=False, USE_FVG_EXIT=False),
    "move target":     dict(USE_TP_MOVE=True,  USE_FVG_EXIT=False),
    "fvg exit":        dict(USE_TP_MOVE=False, USE_FVG_EXIT=True),
    "both":            dict(USE_TP_MOVE=True,  USE_FVG_EXIT=True),
    "no confirmation": dict(TL_CONFIRM_N=0, TL_CONFIRM_CANCEL=False),
}
