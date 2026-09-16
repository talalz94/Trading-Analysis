"""
NOTEBOOK 20 — REWIND.  THIS FILE IS YOURS.  Edit anything below.

A copy of notebook 19's settings, plus §7. Same chart, same gaps, same levels, same sim — with one
extra thing: you can put the chart back to what was knowable at any single candle.

    §1  WHAT AND WHEN      symbol, timeframe, date range
    §2  GAP DETECTION      what counts as a gap and when it dies
    §3  GAP DRAWING        what the chart shows
    §3b SUPPORT/RESISTANCE zones, per timeframe
    §4  THE VETO           an unfilled gap stopping a trade
    §5  THE TRADING SIM    the same sim notebooks 18 and 19 run
    §6  CHART FURNITURE    EMAs, size, dropdown caps
    §7  REWIND             the point-in-time tool

Notebook 19 is untouched and keeps working exactly as it did; this is a separate notebook with a
separate config so neither can disturb the other.

CAREFUL: if this file is open in an editor while a setting is appended, SAVING that stale buffer
writes the old tail back and the new line disappears. If Jupyter offers "File Changed", pick RELOAD.
"""
import numpy as np, pandas as pd     # noqa: F401  — available to expressions below

# =============================================================================================
# §1  WHAT AND WHEN
# =============================================================================================
SYMBOL, SOURCE  = "BTCUSDT", "binance"
TF              = "5m"                   # chart + execution timeframe
VIEW_START, END = "2026-06-01", "2026-06-10"
WARMUP_START    = "2026-05-01"           # data starts here so lines and gaps have history
TF_WARMUP_BARS  = 800                    # history per extra timeframe, in ITS OWN bars

# =============================================================================================
# §2  GAP DETECTION — what counts as a gap, and when it stops counting
# =============================================================================================
# WHICH TIMEFRAMES' GAPS. Drawn, and allowed to veto. May include timeframes FINER than the chart
# ("1m" on a 5m chart is fine — each is detected on its own real bars, never resampled).
#
# These are loaded ON TOP OF §5's LINE_TFS and are deliberately NOT the same setting: LINE_TFS is
# what trendlines are fitted on and therefore what gets TRADED, so putting "4h" there just to see
# its gaps would quietly start trading 4h lines.
FVG_TFS = ("15m", "1h", "4h")

# HOW A GAP DIES. Price coming back into the band mitigates it. All three thresholds are recorded
# for every gap regardless; this picks which one retires it — for the live counts, the veto, and
# where the drawn box STOPS.
#   "touch"  price trades into the band at all     — earliest; one wick kills a gap
#   "ce"     price reaches the band's MIDPOINT     — "consequent encroachment"; the usual default
#   "full"   price trades through the far edge     — latest; a half-eaten gap stays live
#
# This is what makes a box end while the band still looks unfilled on screen: under "ce" the box
# stops the moment price reaches the midpoint, not when the gap is closed. Set "full" to draw each
# box until price has actually traded through it.
FVG_MITIGATION = "ce"

# WHICH GAPS ARE WORTH ANYTHING. Both gates are in ATR so they mean the same thing at 60,000 and at
# 600 — an absolute number would quietly become a filter on WHEN rather than on WHAT.
FVG_MIN_SIZE_ATR = 0.50   # the band must be at least this tall. 0 keeps every three-candle gap.
                          #   0.50 is just above the MEDIAN gap on this data (0.49-0.55 ATR on
                          #   every timeframe measured), so it keeps the larger half. Chosen from
                          #   that census, not from what it did to any backtest
FVG_MIN_DISP_ATR = 0.0    # the MIDDLE candle's body must be this large (0 = off). A real gap comes
                          #   from displacement, not from two neighbours happening not to overlap
FVG_MAX_AGE_BARS = 500    # a gap older than this stops being live. Also bounds the mitigation scan
FVG_ATR_PERIOD   = 14

# =============================================================================================
# §3  GAP DRAWING
# =============================================================================================
FVG_DRAW       = True     # draw the boxes at all
FVG_EXTEND     = True     # an unmitigated box runs to the right edge; False stops it short
FVG_SHOW       = ("bull", "bear")   # ("bull",) to draw only bullish gaps
MAX_FVG_PICKER = 6        # isolatable gaps in the dropdown, PER TIMEFRAME PER DIRECTION. Every box
                          #   is drawn regardless — this only caps how many get their own entry
START_VISIBLE  = ("trades",)   # what is ON when the chart opens. Gaps and levels are DRAWN but
                          #   start off it -- the chart opens on price and trades, and you switch
                          #   layers on from the `FVGs:` / `S/R:` button rows. Add "fvg_bull",
                          #   "fvg_bear" or "sr_zones" here to have those on from the start.

# =============================================================================================
# §3b  SUPPORT / RESISTANCE ZONES — per timeframe, same controls as the gaps
# =============================================================================================
# A level clustered from 1h pivots is a DIFFERENT level from one clustered from 5m pivots, so each
# timeframe is fitted in its own bar space. On the chart they get their own legend row, their own
# `S/R:` button, and an entry in the timeframe -> family -> occurrence cascade — and WEIGHT carries
# the timeframe, exactly as it does for trendlines: the 4h levels are the thickest and brightest,
# because those are the ones that matter.
SHOW_SR_ZONES = True      # the master switch; False draws none of them. The layers start
                          #   LEGENDONLY -- drawn and one button press away, but not on screen
                          #   until you ask, so they cannot clutter the chart by default. Put
                          #   "sr_zones" in START_VISIBLE to have the chart's own layer on.
SR_TFS        = ("15m", "1h", "4h")   # extra timeframes to cluster levels on, beside TF itself

ZONE_LOOKBACK = 2000      # bars of history each layer clusters over, IN ITS OWN BARS — so 2000 is
                          #   ~7 days on 5m and ~11 months on 4h. Bounded on purpose: a long
                          #   warm-up in a trend fills the chart with levels 20% below price
ZONES_LEFT, ZONES_RIGHT = 10, 10   # bars either side a pivot needs to be confirmed
SR_CLUSTER_PCT = 0.15     # two pivots within this % of each other are the same zone
SR_MIN_TOUCHES = 2        # a zone needs this many pivots before it is drawn
SR_MAX_ZONES   = 40       # per layer
MAX_SR_PICKER  = 6        # isolatable zones in the dropdown, PER TIMEFRAME. Every zone is drawn
                          #   regardless — this only caps how many get their own entry

# =============================================================================================
# §4  THE VETO — an unfilled gap stops a trade being taken
# =============================================================================================
# MEASURED, AND IT DOES NOT WORK. experiments/fvg_veto_search.py: 24 settings x 6 windows, scored by
# R per trade with the top winner trimmed — 0 settings clear all six against a chance baseline of
# 0.38, and 4 clear >=4 where chance predicts 8.2. Below chance. Left here to experiment with, not
# because it earns its place. See docs/WHAT_WORKS.md.
USE_FVG_VETO = False

# TWO READINGS, both built, because they are exact mirrors and only measurement separates them:
#   "retrace"  UNFINISHED BUSINESS BEHIND YOU. A bullish gap sits BELOW price and is filled by a
#              move DOWN, so it is a downward magnet — and a LONG breakout dies when price turns
#              round to take it. Longs blocked by bullish gaps, shorts by bearish ones.
#   "barrier"  AN OPPOSING IMPULSE AHEAD OF YOU. A bearish gap is where sellers displaced through;
#              buying into it fights them. Longs blocked by bearish gaps, shorts by bullish ones.
FVG_VETO_MODE = "retrace"

# WHICH LAYERS MAY VETO. None = every timeframe loaded; "" or "chart" means the chart's own.
# NOT the chart's own by default, and this is the one default worth explaining: over this window the
# 5m frame produced ~2,000 gaps with ~96 unmitigated at any moment, so a gap of the threatening
# direction is live on almost every bar and the veto becomes a constant rather than a filter — it
# blocked all 8 trades. A fact about the gap census, not about profitability.
FVG_VETO_TFS = ("1h", "4h")

# HOW FAR BACK, HOW BIG, HOW NEAR. All three are "0 = off", and all three are the escape hatches for
# "take the trade anyway despite the gap".
FVG_VETO_MAX_AGE  = 60    # formed within this many bars OF THE GAP'S OWN TIMEFRAME — 60 on the 1h
                          #   layer is 60 hours, not 60 chart bars
FVG_VETO_MIN_SIZE = 0.0   # ignore gaps smaller than this in ATR. Leave at 0: these gates test the
                          #   NEAREST live gap, so a size cut here misses a big gap sitting behind
                          #   a tiny one. §2's FVG_MIN_SIZE_ATR removes them from the count
                          #   entirely, which makes "nearest" mean "nearest that mattered"
FVG_VETO_MAX_DIST = 3.0   # ignore gaps price has already walked this far away from, in ATR

# =============================================================================================
# §5  THE TRADING SIM — identical to notebooks 18 and 19
# =============================================================================================
LINE_TFS = ("15m", "1h")   # trendlines are fitted on TF plus these, and these get TRADED

# entry: how much confirmation a trendline break needs. Per timeframe, because two 15m closes is
# half an hour and two 1m closes is two minutes. 0 = off for everything except TL_CLOSE_N.
TL_CLOSE_N    = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # closes beyond the line
TL_FULL_M     = 0          # candles formed ENTIRELY beyond it
TL_TOUCH_K    = 0          # the line must have had this many touches
TL_DIR_N      = 0          # candles of the TRADE'S OWN COLOUR (red for a short, green for a long)
TL_DIR_SRC    = "chart"    # "chart" = candles of TF | "line" = the line's own timeframe
TL_DIR_FULL_N = 0          # candles BOTH the right colour AND wholly clear of the line
TL_EXTEND     = True       # a broken line keeps its slope; False holds it flat
TL_ONCE_PER_LINE = False   # True: a given trendline may trigger only its FIRST entry.
                           #   A steeply falling resistance keeps dropping, so a 5m close can clear
                           #   it again hours later at a much lower level while the 15m line is
                           #   still live -- its own timeframe never closed through it. That is two
                           #   real breaks, but both are labelled `15m.R8` on the chart and read as
                           #   a duplicate. Off by default because they ARE separate breaks.
TL_COMBINE    = "all"      # "all" = every enabled condition, "any" = one of them
TL_TOL_ATR    = 0.0        # how far past the line counts as past it, in ATR

TRADE_LONG, TRADE_SHORT = True, True

# exit: stop at the last N bars' extreme, then a 3R target
STOP_N, STOP_BUFFER_PCT = 20, 0.05
BE_AT_RR, TP_RR         = 0, 3.0       # 0 leaves the stop where it was placed

# post-entry bail-out (notebook 18's rule). Off here, to isolate the FVG veto.
USE_RETEST_BAIL = False
SEC_TOUCH_N, SEC_WRONG_M, SEC_OPP_N = 2, 0, 1
SEC_MIN_PCT, SEC_REPLACE_TP = 0.20, True

# EXTRA ENTRY RULES, one sentence each. ANDed onto the trendline break, so they can only remove
# entries.  <n> <reference> <close|formed fully|touched> [above|below] [red|green|NA]
#   reference:  EMA_<tf>_<period>  ·  RES_<tf> / SUP_<tf>  ·  any column on the frame
#   `long:` / `short:` at the front restricts a rule to that side
ENTRY_RULES = [
    "long:  1 candles EMA_15m_100 close above",
    "short: 1 candles EMA_15m_100 close below",
]
RULES_COMBINE = "all"     # "all" = every rule for that side, "any" = one of them

COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)

TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0)

# =============================================================================================
# §6  CHART FURNITURE
# =============================================================================================
EMAS     = (20, 50, 100, 200)            # of TF
MTF_EMAS = [("1m", 20, 50, 100, 200),    # of other timeframes, stepped at their own bar close.
            ("15m", 20, 50, 100, 200),   #   A timeframe FINER than the chart needs its own real
            ("1h", 20, 50, 100, 200)]    #   frame — the notebook loads it automatically
EMAS_ON  = False          # drawn but starting OFF the chart — the gaps are the subject here

SHOW_TRENDLINES = True          # S/R zones are in §3b

HEIGHT      = 900
MAX_PICKER  = 24          # isolatable pattern/trendline instances in the dropdown, all timeframes
MAX_SHAPES_TF = 12        # shapes drawn per extra timeframe
MAX_TRADE_PICKER = 40     # rows in the trade dropdown


# =============================================================================================
# §7  REWIND — put the chart back to one candle
# =============================================================================================
# TWO WAYS TO DO IT, and they answer slightly different questions.
#
# REWIND (browser tool). Arm "⟲ Rewind" on the chart and click a candle. Every gap, level,
# trendline and trade is clipped to that bar, instantly, with no kernel — a gap filled LATER is
# shown still open, a trendline that broke later is shown unbroken, a trade taken later disappears.
# Candles and EMAs are left alone on purpose: you want to see what happened NEXT.
# Works only in the browser tab (the script is injected by open_in_browser), not in a notebook cell.
REWIND = True
#
# AS_OF (rebuild). Set a timestamp and the whole pipeline is re-run with the data stopping there.
# Slower and not clickable, but EXACT and reproducible — nothing is filtered, the detectors simply
# never see the future — and it works anywhere, including inline. Use it to pin a study.
# None = off. e.g. "2026-06-05 20:00"
AS_OF = None

# =============================================================================================
# §8  TWO MORE ENTRY CONDITIONS
# =============================================================================================
# WAIT FOR THE MOVE TO REPEAT ITSELF. A break candle that is huge next to its neighbours and closes
# a whole new body beyond the line looks decisive — and is exactly the shape that gets swept back,
# so entering on its close is entering at the worst price of the move. Non-zero means: within this
# many CHART candles, one candle must be the TRADE'S OWN COLOUR and close beyond the WHOLE trigger
# candle (below its low for a short, above its high for a long). The entry then happens on that
# candle's close. No confirmation inside the window = no trade, which was the question being asked.
#
# This is also the ONLY setting that lets an entry happen later than the break: TL_CLOSE_N asks for
# n CONSECUTIVE closes beyond the line, which is a different condition.
TL_CONFIRM_N = 0          # 0 = off; 3 means "within three 5m candles"

# GAP VETOES, one sentence each. ANDed onto everything else, so they can only remove entries.
#
#     <opposite|same|bull|bear>  [timeframe]  [within N]  [min X]  [near X]
#
#   opposite / same   relative to the TRADE — for a short, "opposite" is a BULLISH gap, the one
#                     price would come back up through. bull / bear say it absolutely.
#   timeframe         15m · 1h+4h · chart · any          (default: every layer loaded)
#   within N          the gap formed within N bars OF ITS OWN TIMEFRAME   (default: no age limit)
#   min X             the gap is at least X ATR tall                      (default: no limit)
#   near X            price is within X ATR of it                         (default: no limit)
#
# Order does not matter and filler words are ignored. `long:` / `short:` scopes a rule to one side.
# Cell 2 prints what each sentence parsed to — read it before trusting a rule.
FVG_RULES = [
    # "short: opposite FVG 15m+1h within 20 candles min 0.8 ATR near 3 ATR",
    # "long:  opposite FVG 15m+1h within 20 candles min 0.8 ATR near 3 ATR",
]
