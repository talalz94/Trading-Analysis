"""
NOTEBOOK 18 — MULTI-TIMEFRAME TRENDLINE TRADING.  THIS FILE IS YOURS.  Edit anything below.

    §1  WHAT AND WHEN       symbol, timeframe, date range
    §2  ENTRY               how much confirmation a trendline break needs
    §3  EXIT                stop, breakeven, target
    §4  POST-ENTRY BAIL     give up on the target when the move stops convincing
    §5  EXTRA ENTRY RULES   one-line sentences ANDed onto the break
    §6  CHART FURNITURE     EMAs, S/R zones, size, dropdown caps
    §7  COSTS AND FITTING   fees, sizing, the trendline detector

WHY THIS IS A FILE AND NOT A NOTEBOOK CELL. The notebook is regenerated from a script, and that
script used to hardcode every value — so adding one setting silently re-imposed all the others.
Commit 8a8b8de did exactly that: TF "1m" -> "5m", LINE_TFS ("5m","15m","1h") -> ("15m","1h"),
WARMUP_START "2026-05-25" -> "2026-05-01", none of it asked for. Nothing regenerates this file.

Those three were left where they ended up rather than silently changed back — swapping them again
without asking would be the same mistake. To go back to the original 1m setup:
    TF = "1m"  ·  LINE_TFS = ("5m", "15m", "1h")  ·  WARMUP_START = "2026-05-25"

CAREFUL: if this file is open in an editor while a setting is appended, SAVING that stale buffer
writes the old tail back and the new line disappears. That happened once, to EMAS_ON, and the
notebook died on a NameError. If Jupyter offers "File Changed", pick RELOAD.

See docs/NOTEBOOK_GUIDE.md for the rules this file and the notebook follow.
"""
import numpy as np, pandas as pd     # noqa: F401  — available to expressions below

# =============================================================================================
# §1  WHAT AND WHEN
# =============================================================================================
SYMBOL, SOURCE  = "BTCUSDT", "binance"
TF              = "5m"                   # chart + execution timeframe
LINE_TFS        = ("15m", "1h")          # lines are ALSO fitted on these, and these get traded
VIEW_START, END = "2026-06-01", "2026-06-10"
WARMUP_START    = "2026-05-01"           # data starts here so the lines have pivots
TF_WARMUP_BARS  = 800                    # history per extra timeframe, in ITS OWN bars

TRADE_LONG, TRADE_SHORT = True, True

# =============================================================================================
# §2  ENTRY — how much confirmation a trendline break needs
# =============================================================================================
# Each of these takes a DICT keyed by timeframe (or a plain int for all of them), because two 15m
# closes is half an hour and two 1m closes is two minutes. 0 = off.
#
# HOW MANY CANDLES TO WAIT before taking a break. From experiments/trendline_wait_search.py over
# 5.5 months in 6 windows — read the appendix for how much of it to believe (short answer: the 5m
# number; nothing cleared all six windows).
TL_CLOSE_N  = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # closes BEYOND the line
TL_FULL_M   = 0          # candles formed ENTIRELY beyond it, wicks included
TL_TOUCH_K  = 0          # the line must have had this many touches before it broke

# n consecutive candles of the TRADE'S OWN COLOUR — red for a short, green for a long. A different
# question from TL_CLOSE_N: a candle can close through a support and still be green, and on this
# data 30 of 72 short entry bars were not red.
TL_DIR_N    = 0
TL_DIR_SRC  = "chart"    # "chart" = candles of TF (what you execute on) | "line" = the line's own

# n candles that are BOTH the right colour AND wholly clear of the line, wicks included — a red
# candle whose HIGH is still under the support. The line is EXTENDED along its own slope to be
# measured against, since it is retired the moment it breaks.
TL_DIR_FULL_N = 0        # 74 -> 39 -> 15 -> 5 short entries at 0/1/2/3
TL_EXTEND     = True     # False holds a broken line FLAT instead of following its slope

TL_COMBINE  = "all"      # "all" = every enabled condition, "any" = one of them
TL_TOL_ATR  = 0.0        # how far past the line counts as past it, in ATR

# =============================================================================================
# §3  EXIT — stop, breakeven, target
# =============================================================================================
STOP_N, STOP_BUFFER_PCT = 20, 0.05       # stop at the last N bars' extreme, pushed out this far

# BE_AT_RR MOVES THE STOP. Once the trade is +N R up, the stop walks to the entry price, so a
# reversal from there exits at breakeven (a small loss: the fees) instead of at the red line on the
# chart. That is why some trades show `stop_loss` with an exit sitting at entry — the chart tags
# those `sl@be`. From docs/WHAT_WORKS.md, worth +0.0095R per trade over a large sample: real but
# tiny, and it gives up winners that would have run.
BE_AT_RR, TP_RR = 0, 3.0                 # 0 leaves the stop where it was placed; target 3R

# =============================================================================================
# §4  POST-ENTRY BAIL — stop aiming at the target when the move stops convincing
# =============================================================================================
# A trade that goes your way and then keeps coming back to where you got in is oscillating around
# your entry, not trending. Either trigger arms it; it then exits on the first opposite candle
# WHILE IN PROFIT. Both counts are per trade — the level is this trade's own entry price.
USE_RETEST_BAIL = False   # True turns the whole rule on
SEC_TOUCH_N     = 2       # bars since entry that traded back THROUGH the entry price
SEC_WRONG_M     = 0       # candles formed WHOLLY on the losing side of it (0 = off)
SEC_OPP_N       = 1       # opposite candles that then close it

# "in profit" must mean AFTER COSTS. The engine's own gate is "any favourable move", and on this
# data three of ten bail-outs moved the right way and still lost money — one by 0.92 in price
# against 15.39 in fees. A round trip here is (8 + 1.5) x 2 = 19 bps, so 0.2% clears it.
SEC_MIN_PCT    = 0.20
SEC_REPLACE_TP = True     # drop the 3R target once armed; False races it instead

# =============================================================================================
# §5  EXTRA ENTRY RULES — one sentence each
# =============================================================================================
# ANDed onto the trendline break, so they can only remove entries, never add them. Order inside a
# sentence does not matter and filler words are ignored.
#
#     <n>  <reference>  <close | formed fully | touched>  [above | below]  [red | green | NA]
#
#   reference     EMA_<tf>_<period>     e.g. EMA_5m_200, EMA_15m_50
#                 RES_<tf> / SUP_<tf>   the fitted trendline of that timeframe
#                 or any column already on D
#   close         the candle's CLOSE is above/below it
#   formed fully  the WHOLE candle is, wicks included — much stricter than `close`
#   touched       the candle's range CONTAINS it (neither above nor below, so no side)
#   `long:` / `short:` at the front restricts a rule to that side; without it, both.
#
# The Run cell prints what each sentence parsed to — read it before trusting a rule.
ENTRY_RULES = [
    #"long:  1 candles EMA_15m_100 formed fully above",
    #"short: 1 candles EMA_15m_100 formed fully below",
    # "3 candles EMA_15m_50 formed fully below red",
]
RULES_COMBINE = "all"     # "all" = every rule for that side, "any" = one of them

# =============================================================================================
# §6  CHART FURNITURE
# =============================================================================================
# EMAs — these stay visible when you isolate a single trade, which is the point of having them
EMAS     = (20, 50, 100, 200)            # of TF
MTF_EMAS = [("1m", 20, 50, 100, 200),    # of other timeframes, stepped at their own bar close.
            ("15m", 20, 50, 100, 200),   #   A timeframe FINER than the chart needs its own real
            ("1h", 20, 50, 100, 200)]    #   frame — the notebook loads it automatically

# ALL THE EMAs AT ONCE. False draws every EMA but starts it OFF the chart — each one is still in the
# legend, so you click on the two you want instead of clicking off the six you don't. The chart also
# gets an "EMAs on / off" button pair for doing this without re-running.
# (Plotly's own way to do it was double-clicking a legend entry, which ISOLATES that entry — so
# double-clicking an EMA while you had a trade picked took the trade and its trendline away with it.
# A double-click now does the same as a single click, and this is the control that replaces it.)
EMAS_ON = True

SHOW_SR_ZONES, ZONE_LOOKBACK = True, 2000

HEIGHT           = 900
MAX_PICKER       = 24     # isolatable pattern/trendline instances in the dropdown, all timeframes
MAX_SHAPES_TF    = 12     # shapes drawn per extra timeframe
MAX_TRADE_PICKER = 40     # rows in the trade dropdown

# =============================================================================================
# §7  COSTS AND FITTING
# =============================================================================================
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0)
