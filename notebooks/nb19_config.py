"""Notebook 19's settings — FAIR VALUE GAPS. THIS FILE IS YOURS.

Same deal as `nb18_config.py`: the notebook is regenerated from a script, so nothing that matters
may live in a notebook cell. Nothing regenerates THIS file. A later update may APPEND a new setting
at the bottom; it will never rewrite a line you have edited.

If you have this file open in an editor while a new setting is appended, SAVING that stale buffer
writes the old tail back and the new line disappears. That already happened once with EMAS_ON. The
notebook defends against it — cell 2 carries a default for every setting added after the fact and
prints the line to paste in — but if Jupyter offers you "File Changed", pick RELOAD, not overwrite.

The trading settings below are a COPY of notebook 18's as of this file's creation, on purpose: with
them identical, the only difference between the two notebooks is the fair-value-gap veto, so
comparing them measures the veto and nothing else.
"""
import numpy as np, pandas as pd     # noqa: F401  — available to expressions below

SYMBOL, SOURCE = "BTCUSDT", "binance"
TF          = "5m"                       # chart + execution timeframe
LINE_TFS    = ("15m", "1h")              # ALSO trade lines fitted on these
VIEW_START, END = "2026-06-01", "2026-06-10"
WARMUP_START    = "2026-05-01"           # data starts here so the lines have pivots
TF_WARMUP_BARS  = 800                    # history per extra timeframe, in ITS OWN bars

# ===========================================================================================
# FAIR VALUE GAPS
# ===========================================================================================
# Which timeframes' gaps to DRAW and to feed the veto. These are loaded on top of LINE_TFS and are
# deliberately NOT added to it: LINE_TFS is what trendlines are fitted on and what gets TRADED, so
# putting 4h there just to see its gaps would quietly start trading 4h lines.
FVG_TFS = ("15m", "1h", "4h")

# HOW A GAP DIES. Price coming back into the band mitigates it, and there are three defensible
# thresholds. All three are recorded for every gap either way; this picks which one the live counts,
# the veto and the drawn box use.
#   "touch"  price trades into the band at all      — earliest; one wick kills a gap
#   "ce"     price reaches the band's MIDPOINT      — "consequent encroachment", the usual default
#   "full"   price trades through the far edge      — latest; a half-eaten gap stays live
FVG_MITIGATION = "ce"

# WHICH GAPS ARE WORTH ANYTHING. Both gates are in ATR so they mean the same thing at 60,000 and at
# 600 — an absolute number would quietly become a filter on WHEN rather than on WHAT.
FVG_MIN_SIZE_ATR = 0.50   # the band must be at least this tall. 0 keeps every three-candle gap.
                          #   0.50 is just above the MEDIAN gap on this data (0.49-0.55 ATR on every
                          #   timeframe measured), so it keeps the larger half. Chosen from that
                          #   census, not from what it did to any backtest
FVG_MIN_DISP_ATR = 0.0    # the MIDDLE candle's body must be this large (0 = off). A real gap comes
                          #   from displacement, not from two neighbours happening not to overlap
FVG_MAX_AGE_BARS = 500    # a gap older than this stops being live. Also bounds the mitigation scan
FVG_ATR_PERIOD   = 14

FVG_DRAW      = True      # draw the boxes at all
FVG_EXTEND    = True      # an unmitigated gap's box runs to the right edge; False stops it short
FVG_SHOW      = ("bull", "bear")   # ("bull",) to draw only bullish gaps
MAX_FVG_PICKER = 6        # isolatable gaps in the dropdown, PER TIMEFRAME PER DIRECTION. The boxes
                          #   are all drawn regardless — this only caps how many get their own entry

# ===========================================================================================
# THE VETO — an unfilled gap stops the trade being taken
# ===========================================================================================
# TWO READINGS, both built, because they are exact mirrors and only measurement separates them:
#
#   "retrace"  UNFINISHED BUSINESS BEHIND YOU. A bullish gap sits BELOW price and is filled by a
#              move DOWN, so an unmitigated bullish gap is a downward magnet — and it is a LONG
#              breakout that dies when price turns round to take it. Longs blocked by bullish gaps,
#              shorts by bearish ones. This is the "price will return to sweep it" idea.
#
#   "barrier"  AN OPPOSING IMPULSE AHEAD OF YOU. A bearish gap is where sellers displaced through;
#              buying into it fights them. Longs blocked by bearish gaps, shorts by bullish ones.
#
# A result that looks good under one and bad under the other says something real about which story
# holds. One that looks the same under both is noise.
#
# WHAT THE FIRST LOOK SHOWED, and why it is not a reason to pick one: over 2026-06-01..10 the whole
# sim takes 8 trades, 4 winners and 4 losers. Across 96 gate combinations "barrier" tended to block
# losers and "retrace" to block winners -- but a 3-trade block that happens to be all losers occurs
# by chance with probability C(4,3)/C(8,3) = 7.1%, so 96 combinations should produce about 7 of them
# for free, and it produced about that many. The 96 cells are also not 96 observations: they all
# score the SAME eight trades. This is one window and it settles nothing.
#
# `experiments/fvg_veto_search.py` is the version that could settle it -- many windows, scored by
# how many come out positive, against the chance baseline. Run that before believing either mode.
USE_FVG_VETO = True
FVG_VETO_MODE = "retrace"

# Which layers may veto. None = every timeframe loaded. ("15m", "1h") to let only those block a
# trade; "" or "chart" means the chart's own timeframe.
#
# NOT the chart's own timeframe by default, and this is the one default worth explaining. Over this
# window the 5m frame produced 1,974 gaps with about 96 unmitigated at any moment; a gap of the
# threatening direction is therefore live on almost every bar, so including 5m turns the veto into a
# constant rather than a filter -- it blocked all 8 trades. That is a fact about the gap census, not
# about profitability. The coarser layers are sparse enough to discriminate.
FVG_VETO_TFS = ("1h", "4h")

# HOW FAR BACK, HOW BIG, HOW NEAR. All three are "0 = off", and all three are the escape hatches for
# "we may still take the trade regardless of its formation".
FVG_VETO_MAX_AGE  = 60    # the gap must have formed within this many bars OF ITS OWN TIMEFRAME —
                          #   60 on the 1h layer is 60 hours, not 60 chart bars
FVG_VETO_MIN_SIZE = 0.0   # ignore gaps smaller than this in ATR. 0 relies on FVG_MIN_SIZE_ATR
                          #   above, which is the better place to do it — see the note below
FVG_VETO_MAX_DIST = 3.0   # ignore gaps price has already walked this far away from, in ATR

# NOTE on FVG_VETO_MIN_SIZE: the per-bar columns describe the NEAREST live gap of each direction,
# not all of them, so a size gate here is tested against that nearest one — if the nearest is tiny
# and a big one sits just behind it, this will not see the big one. Dropping insignificant gaps at
# DETECTION (FVG_MIN_SIZE_ATR) removes them from the count entirely, which makes "nearest" mean
# "nearest that mattered". Filter there; leave this at 0 unless you want a second, stricter pass.

# ===========================================================================================
# THE TRADING SIM — a copy of notebook 18's, so the only difference is the veto
# ===========================================================================================
TL_CLOSE_N  = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # or just an int for every timeframe
TL_FULL_M   = 0          # consecutive candles ENTIRELY beyond the line (0 = off; also per-tf)
TL_TOUCH_K  = 0          # the line must have had this many touches (0 = off; also per-tf)
TL_DIR_N    = 0          # n candles of the TRADE'S OWN COLOUR. 0 = off
TL_DIR_SRC  = "chart"    # "chart" = candles of TF (what you execute on) | "line" = the line's own
TL_DIR_FULL_N = 0        # n candles BOTH the right colour AND wholly clear of the line. 0 = off
TL_EXTEND     = True     # False holds a broken line FLAT instead of following its slope
TL_COMBINE  = "all"      # "all" = every enabled condition, "any" = one of them
TL_TOL_ATR  = 0.0        # how far past the line counts as past it, in ATR

TRADE_LONG, TRADE_SHORT = True, True

STOP_N, STOP_BUFFER_PCT = 20, 0.05       # stop at the last N bars' extreme, pushed out this far
BE_AT_RR, TP_RR         = 0, 3.0         # 0 leaves the stop where it was placed; target 3R

USE_RETEST_BAIL = False   # the notebook-18 post-entry rule; off here to isolate the FVG veto
SEC_TOUCH_N, SEC_WRONG_M, SEC_OPP_N = 2, 0, 1
SEC_MIN_PCT, SEC_REPLACE_TP = 0.20, True

# EXTRA ENTRY RULES, one sentence each — the same language as notebook 18. ANDed onto the trendline
# break, so they can only remove entries. Cell 4 prints what each sentence parsed to.
ENTRY_RULES = [
    "long:  1 candles EMA_15m_100 close above",
    "short: 1 candles EMA_15m_100 close below",
]
RULES_COMBINE = "all"     # "all" = every rule for that side, "any" = one of them

EMAS     = (20, 50, 100, 200)                  # of TF
MTF_EMAS = [("15m", 50, 200), ("1h", 50, 200)]  # of other timeframes, stepped at their own close
EMAS_ON  = False          # draw them but start them off the chart — the gaps are the subject here

COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0)

HEIGHT, MAX_PICKER, MAX_SHAPES_TF, MAX_TRADE_PICKER = 900, 24, 12, 40
SHOW_SR_ZONES, ZONE_LOOKBACK = False, 2000
SHOW_TRENDLINES = True
