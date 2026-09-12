"""Notebook 18's settings. THIS FILE IS YOURS.

It was split out of the notebook because the notebook is regenerated from a script, and that script
hardcoded every value — so adding one setting silently re-imposed all the others. Commit 8a8b8de did
exactly that: it changed TF from "1m" to "5m", LINE_TFS from ("5m","15m","1h") to ("15m","1h"), and
WARMUP_START from "2026-05-25" to "2026-05-01", none of which had been asked for.

Nothing regenerates this file. A later update may APPEND a new setting at the bottom; it will never
rewrite a line you have edited.

The three values above were left where they ended up rather than silently changed back — swapping
them again without asking would be the same mistake. To go back to the original 1m setup:
    TF = "1m"  ·  LINE_TFS = ("5m", "15m", "1h")  ·  WARMUP_START = "2026-05-25"
"""
import numpy as np, pandas as pd     # noqa: F401  — available to expressions below

SYMBOL, SOURCE = "BTCUSDT", "binance"
TF          = "5m"                       # chart + execution timeframe
LINE_TFS    = ("15m", "1h")              # ALSO trade lines fitted on these
VIEW_START, END = "2026-06-01", "2026-06-10"
WARMUP_START    = "2026-05-01"           # data starts here so the lines have pivots
TF_WARMUP_BARS  = 800                    # history per extra timeframe, in ITS OWN bars

# HOW MANY CANDLES TO WAIT before taking a break. Per timeframe, because two 15m closes is half an
# hour and two 1m closes is two minutes. From experiments/trendline_wait_search.py over 5.5 months
# in 6 windows — read the appendix for how much of it to believe (short answer: the 5m number).
TL_CLOSE_N  = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # or just an int for every timeframe
TL_FULL_M   = 0          # consecutive candles ENTIRELY beyond the line (0 = off; also per-tf)
TL_TOUCH_K  = 0          # the line must have had this many touches (0 = off; also per-tf)

# n consecutive candles of the TRADE'S OWN COLOUR before entering — red for a short, green for a
# long. A different question from TL_CLOSE_N: a candle can close through a support and still be
# green, and on this data 30 of 72 short entry bars were not red. Per timeframe like the rest.
TL_DIR_N    = 0          # 0 = off. e.g. 2, or {"5m": 2, "15m": 3, "*": 1}
TL_DIR_SRC  = "chart"    # "chart" = candles of TF (what you execute on) | "line" = the line's own

# n candles that are BOTH the right colour AND wholly clear of the line, wicks included — a red
# candle whose HIGH is still under the support. The line is EXTENDED along its own slope to be
# measured against, since it is retired the moment it breaks.
TL_DIR_FULL_N = 0        # 0 = off. Also per-tf.  74 -> 39 -> 15 -> 5 short entries at 0/1/2/3
TL_EXTEND     = True     # False holds a broken line FLAT instead of following its slope
TL_COMBINE  = "all"      # "all" = every enabled condition, "any" = one of them
TL_TOL_ATR  = 0.0        # how far past the line counts as past it, in ATR

TRADE_LONG, TRADE_SHORT = True, True

STOP_N, STOP_BUFFER_PCT = 20, 0.05       # stop at the last N bars' extreme, pushed out this far
# BE_AT_RR MOVES THE STOP. Once the trade is +1R up, the stop walks to the entry price, so a
# reversal from there exits at breakeven (a small loss: the fees) instead of at the red line on the
# chart. That is why some trades show `stop_loss` with an exit sitting at entry. From
# docs/WHAT_WORKS.md, worth +0.0095R per trade over a large sample -- real but tiny, and it gives up
# winners that would have run. Set 0 to leave the stop where it was placed.
BE_AT_RR, TP_RR         = 0, 3.0       # move stop to breakeven at 1R; target 3R

# IF THE MOVE STOPS CONVINCING YOU, stop aiming at the target. A trade that goes your way and then
# keeps coming back to where you got in is oscillating around your entry, not trending. Either
# trigger arms it; it then exits on the first opposite candle WHILE IN PROFIT (a red candle for a
# long). Both counts are per trade — the level is this trade's own entry price.
USE_RETEST_BAIL = False   # True turns the whole rule on
SEC_TOUCH_N     = 2       # bars since entry that traded back THROUGH the entry price
SEC_WRONG_M     = 0       # candles formed WHOLLY on the losing side of it (0 = off)
SEC_OPP_N       = 1       # opposite candles that then close it
# "in profit" must mean AFTER COSTS. The engine's own gate is "any favourable move", and on this
# data three of ten bail-outs moved the right way and still lost money — one by 0.92 in price
# against 15.39 in fees. A round trip here is (8 + 1.5) x 2 = 19 bps, so 0.2% clears it.
SEC_MIN_PCT     = 0.20
SEC_REPLACE_TP  = True    # drop the 3R target once armed; False races it instead

# EXTRA ENTRY RULES, one sentence each. ANDed onto the trendline break, so they can only remove
# entries, never add them. Order inside a sentence does not matter and filler words are ignored.
#
#     <n>  <reference>  <close | formed fully | touched>  [above | below]  [red | green | NA]
#
#   reference   EMA_<tf>_<period>   e.g. EMA_5m_200, EMA_15m_50
#               RES_<tf> / SUP_<tf> the fitted trendline of that timeframe
#               or any column already on D
#   close       the candle's CLOSE is above/below it
#   formed fully  the WHOLE candle is, wicks included — much stricter than `close`
#   touched     the candle's range CONTAINS it (neither above nor below, so no side)
#   `long:` / `short:` at the front restricts a rule to that side; without it, both.
#
# Cell 3 prints what each sentence parsed to — read it before trusting a rule.
ENTRY_RULES = [
    #"long:  1 candles EMA_5m_200 close above",
    #"short: 1 candles EMA_5m_200 close below",
    # "3 candles EMA_15m_50 formed fully below red",
]
RULES_COMBINE = "all"     # "all" = every rule for that side, "any" = one of them

# EMAs — these stay visible when you isolate a single trade, which is the point of having them
EMAS     = (20, 50, 200)                  # of TF
MTF_EMAS = [("15m", 50), ("1h", 200)]     # of other timeframes, stepped at their own bar close

COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0)

HEIGHT, MAX_PICKER, MAX_SHAPES_TF, MAX_TRADE_PICKER = 900, 24, 12, 40
SHOW_SR_ZONES, ZONE_LOOKBACK = True, 2000
