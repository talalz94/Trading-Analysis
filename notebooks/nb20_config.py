"""
NOTEBOOK 20 — REWIND.  THIS FILE IS YOURS.  Edit anything below.

Notebook 19's chart and sim, plus one thing: you can put the chart back to what was knowable at any
single candle. Notebook 19 is untouched and keeps working exactly as it did — separate notebook,
separate config, neither can disturb the other.

    §1  WHAT AND WHEN        symbol, timeframe, date range, warm-up
    §2  THE TRADE            what is traded, and what has to be true first
        2.1  which lines     the timeframes fitted and traded, and how a line is found
        2.2  the break       how much confirmation a break needs
        2.3  extra rules     conditions ANDed onto the break
        2.4  the exit        stop, target, post-entry bail-out
        2.5  the money       costs and sizing
    §3  FAIR VALUE GAPS      detection, drawing, and gaps that stop a trade
        3.1  detection       what counts as a gap and when it dies
        3.2  drawing         what the chart shows
        3.3  the veto        an unfilled gap stopping a trade
    §4  SUPPORT / RESISTANCE zones, per timeframe
    §5  THE CHART            EMAs, what is on at open, size, dropdown caps
    §6  REWIND               the point-in-time tool

    WANT TO...                                        GO TO
    trade a different period                          §1
    change what counts as a break                     §2.2
    wait for a confirmation candle before entering    §2.2   TL_CONFIRM_N
    add a condition to the entry                      §2.3   ENTRY_RULES
    move the stop or the target                       §2.4
    change which gaps are drawn                       §3.1 / §3.2
    skip a trade when a gap is in the way             §3.3   FVG_RULES
    change the S/R levels                             §4
    turn the EMAs on, resize the chart                §5
    put the chart back to one candle                  §6

CAREFUL: if this file is open in an editor while a setting is added, SAVING that stale buffer writes
the old text back and the new line disappears. If Jupyter offers "File Changed", pick RELOAD.
"""
import numpy as np, pandas as pd     # noqa: F401  — available to expressions below

# =============================================================================================
# §1  WHAT AND WHEN
# =============================================================================================
SYMBOL = "BTCUSDT"
SOURCE = "binance"
TF     = "5m"                   # chart + execution timeframe

VIEW_START = "2026-06-01"       # the window you actually look at
END        = "2026-06-10"

WARMUP_START   = "2026-05-01"   # data starts here so lines and gaps have history before VIEW_START
TF_WARMUP_BARS = 800            # history per extra timeframe, in ITS OWN bars


# =============================================================================================
# §2  THE TRADE
# =============================================================================================
# A trendline from some timeframe gives way, and everything else here decides whether that break
# is worth taking: how convincingly it broke (§2.2), what else must be true (§2.3), where you get
# out (§2.4). Gap vetoes are the one entry condition that lives elsewhere — §3.3, with the gaps.

# ---------------------------------------------------------------------------------------------
# 2.1  WHICH LINES
# ---------------------------------------------------------------------------------------------
LINE_TFS = ("15m", "1h")   # trendlines are fitted on TF plus these, and these get TRADED.
                           #   NOT the same setting as §3.1's FVG_TFS: adding "4h" here starts
                           #   trading 4h lines, adding it there only draws 4h gaps

# how a line is found in the first place. A line through 2 points is not a trendline — two points
# always fit — so the fitter wants touches and an envelope nothing pokes through.
TRENDLINE = dict(atr_mult=3.5,        # zigzag sensitivity: bigger = fewer, larger swings
                 min_touches=3,       # pivots the line must pass through
                 lookback_pivots=16,  # how far back it may reach for them
                 line_tol_atr=0.6,    # how far off the line a pivot still counts as a touch
                 min_span_bars=40,    # a line must cover at least this many bars
                 max_age_bars=500,    # retire it this long after birth even unbroken
                 break_buffer_atr=0.0)  # how far past it a close must be to count as a break

# ---------------------------------------------------------------------------------------------
# 2.2  THE BREAK — how much confirmation it needs
# ---------------------------------------------------------------------------------------------
# Each condition takes a DICT keyed by timeframe, because two 15m closes is half an hour and two
# 1m closes is two minutes. 0 = off for everything except TL_CLOSE_N.
TL_CLOSE_N    = {"5m": 1, "15m": 1, "1h": 1, "*": 2}   # closes beyond the line
TL_FULL_M     = 0          # candles formed ENTIRELY beyond it
TL_TOUCH_K    = 0          # the line must have had this many touches
TL_DIR_N      = 0          # candles of the TRADE'S OWN COLOUR (red for a short, green for a long)
TL_DIR_SRC    = "chart"    # "chart" = candles of TF | "line" = the line's own timeframe
TL_DIR_FULL_N = 0#{"5m": 1, "15m": 1, "1h": 1, "*": 1}           # candles BOTH the right colour AND wholly clear of the line

TL_COMBINE = "all"         # "all" = every enabled condition, "any" = one of them
TL_TOL_ATR = 0.0           # how far past the line counts as past it, in ATR
TL_EXTEND  = True          # a broken line keeps its slope; False holds it flat

TL_ONCE_PER_LINE = False   # True: a given trendline may trigger only its FIRST entry.
                           #   A steeply falling resistance keeps dropping, so a 5m close can clear
                           #   it again hours later at a much lower level while the 15m line is
                           #   still live -- its own timeframe never closed through it. That is two
                           #   real breaks, but both are labelled `15m.R8` on the chart and read as
                           #   a duplicate. Off by default because they ARE separate breaks.

# WAIT FOR THE MOVE TO REPEAT ITSELF. A break candle that is huge next to its neighbours and closes
# a whole new body beyond the line looks decisive — and is exactly the shape that gets swept back,
# so entering on its close is entering at the worst price of the move. Non-zero means: within this
# many CHART candles, one candle must be the TRADE'S OWN COLOUR and close beyond the WHOLE trigger
# candle (below its low for a short, above its high for a long). The entry then happens on that
# candle's close. No confirmation inside the window = no trade, which was the question being asked.
#
# This is also the ONLY setting that lets an entry happen later than the break: TL_CLOSE_N asks for
# n CONSECUTIVE closes beyond the line, which is a different condition.
# THE REFERENCE IS THE TRIGGER CANDLE -- call it `rs`. Every other setting in §2.2 measures
# against the LINE; this one measures against rs, which is the only way to ask "did the move hold
# below the candle that fired it". Takes a plain number, or a DICT per LINE timeframe, so a noisy
# 1h break can be made to prove itself while a 5m break is taken as it comes:
#     TL_CONFIRM_N = {"1h": 2, "*": 0}       two candles after rs, on 1h lines only
TL_CONFIRM_N = 0          # 0 = off; 3 means "within three 5m candles"

TL_CONFIRM_ALL = False    # False: ONE of the next n candles must qualify (entry on the first).
                          #   True:  ALL n must, consecutively from rs+1 (entry on the nth).
                          #   True is the stricter reading of "n candles must form below rs"
TL_CONFIRM_MODE = "close"  # "close": the candle CLOSES beyond rs   (below rs's LOW, for a short)
                          #   "full":  the WHOLE candle is beyond rs (its HIGH below rs's LOW)
                          #   Both measure against rs's far edge, so "beyond rs" means clear of the
                          #   entire candle -- closing back inside its range is the sweep
TL_CONFIRM_COLOUR = True  # the confirming candles must also be the trade's own colour
                          #   (red for a short, green for a long). False drops the colour test

# ONE rs PER LINE? Careful: a 1h line typically reads BROKEN on several consecutive 5m bars, so
# every one of them is an rs and each gets its own confirmation window -- fail at 18:50 and a later
# bar can still confirm and trade. Set TL_ONCE_PER_LINE = True above if you mean "the FIRST break
# is the only rs, and if it is not confirmed the line is spent".

# ---------------------------------------------------------------------------------------------
# 2.3  EXTRA RULES — ANDed onto the break, so they can only remove entries
# ---------------------------------------------------------------------------------------------
TRADE_LONG  = True
TRADE_SHORT = True

# One sentence each.  <n> <reference> <close|formed fully|touched> [above|below] [red|green|NA]
#   reference:  EMA_<tf>_<period>  ·  RES_<tf> / SUP_<tf>  ·  any column on the frame
#   `long:` / `short:` at the front restricts a rule to that side
ENTRY_RULES = [
    "long:  1 candles EMA_15m_100 close above",
    "short: 1 candles EMA_15m_100 close below",
]
RULES_COMBINE = "all"     # "all" = every rule for that side, "any" = one of them

# Gap vetoes are the same idea in the same style, but they belong with the gaps — see §3.3.

# ---------------------------------------------------------------------------------------------
# 2.4  THE EXIT
# ---------------------------------------------------------------------------------------------
STOP_N          = 20      # stop at the extreme of the last N bars
STOP_BUFFER_PCT = 0.05    # pushed this % beyond it, so the wick itself does not take you out

BE_AT_RR = 0              # move the stop to breakeven at this R. 0 leaves it where it was placed
TP_RR    = 3.0            # target, in multiples of the risk taken

# POST-ENTRY BAIL-OUT (notebook 18's rule). Off here, to isolate the FVG veto.
USE_RETEST_BAIL = False
SEC_TOUCH_N     = 2       # bars that traded back THROUGH the entry price before it arms
SEC_WRONG_M     = 0       # candles formed WHOLLY on the losing side before it arms
SEC_OPP_N       = 1       # opposite-colour candles that then close it
SEC_MIN_PCT     = 0.20    # the move the exit must clear, so a bail-out cannot exit below cost
SEC_REPLACE_TP  = True    # an armed trade drops the target and lives on the bail-out

# ---------------------------------------------------------------------------------------------
# 2.5  THE MONEY
# ---------------------------------------------------------------------------------------------
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)


# =============================================================================================
# §3  FAIR VALUE GAPS
# =============================================================================================
# ---------------------------------------------------------------------------------------------
# 3.1  DETECTION — what counts as a gap, and when it stops counting
# ---------------------------------------------------------------------------------------------
# WHICH TIMEFRAMES' GAPS. Drawn, and allowed to veto. May include timeframes FINER than the chart
# ("1m" on a 5m chart is fine — each is detected on its own real bars, never resampled).
#
# These are loaded ON TOP OF §2.1's LINE_TFS and are deliberately NOT the same setting: LINE_TFS is
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

# ---------------------------------------------------------------------------------------------
# 3.2  DRAWING
# ---------------------------------------------------------------------------------------------
FVG_DRAW   = True         # draw the boxes at all
FVG_EXTEND = True         # an unmitigated box runs to the right edge; False stops it short
FVG_SHOW   = ("bull", "bear")   # ("bull",) to draw only bullish gaps

# ---------------------------------------------------------------------------------------------
# 3.3  THE VETO — an unfilled gap stops a trade being taken
# ---------------------------------------------------------------------------------------------
# MEASURED, AND IT DOES NOT WORK. experiments/fvg_veto_search.py: 24 settings x 6 windows, scored by
# R per trade with the top winner trimmed — 0 settings clear all six against a chance baseline of
# 0.38, and 4 clear >=4 where chance predicts 8.2. Below chance. Left here to experiment with, not
# because it earns its place. See docs/WHAT_WORKS.md.
#
# TWO WAYS TO SAY IT. The settings just below are the blunt one — they test the NEAREST live gap.
# FVG_RULES at the end of this section is the precise one, a sentence per condition, and it is the
# one to reach for. They are independent: either, both, or neither.
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
                          #   a tiny one. §3.1's FVG_MIN_SIZE_ATR removes them from the count
                          #   entirely, which makes "nearest" mean "nearest that mattered"
FVG_VETO_MAX_DIST = 3.0   # ignore gaps price has already walked this far away from, in ATR

# GAP VETOES AS SENTENCES, one per rule. ANDed onto everything else, so they can only remove
# entries.
#
#     <opposite|same|bull|bear>  [timeframe]  [within N]  [min X]  [near X]
#
#   opposite / same   relative to the TRADE, meaning the gap whose FILL moves price AGAINST it.
#                     For a SHORT that is a BEARISH gap (it sits above and is filled by price
#                     RISING — the sweep that kills the short); for a LONG, a BULLISH gap below.
#                     bull / bear say it absolutely, if you would rather not think about it.
#   timeframe         15m · 1h+4h · chart · any          (default: every layer loaded)
#                     "15m+1h" means EITHER layer may object, not both at once
#   within N          the gap formed within N bars OF ITS OWN TIMEFRAME   (default: no age limit)
#   min X             the gap is at least X ATR tall                      (default: no limit)
#   near X            price is within X ATR of it                         (default: no limit)
#
# X ATR is a multiple of the 14-period ATR OF THAT LAYER, at that bar — so it means the same thing
# at 60,000 and at 600. Order does not matter and filler words are ignored. `long:` / `short:`
# scopes a rule to one side. Cell 2 prints what each sentence parsed to — read it before trusting
# a rule.
FVG_RULES = [
    # "short: opposite FVG 15m+1h within 20 candles min 0.8 ATR near 3 ATR",
    # "long:  opposite FVG 15m+1h within 20 candles min 0.8 ATR near 3 ATR",
]


# =============================================================================================
# §4  SUPPORT / RESISTANCE ZONES — per timeframe, same controls as the gaps
# =============================================================================================
# A level clustered from 1h pivots is a DIFFERENT level from one clustered from 5m pivots, so each
# timeframe is fitted in its own bar space. On the chart they get their own legend row, their own
# `S/R:` button, and an entry in the timeframe -> family -> occurrence cascade — and WEIGHT carries
# the timeframe, exactly as it does for trendlines: the 4h levels are the thickest and brightest,
# because those are the ones that matter.
SHOW_SR_ZONES = True      # the master switch; False draws none of them. The layers start
                          #   LEGENDONLY -- drawn and one button press away, but not on screen
                          #   until you ask, so they cannot clutter the chart by default. Put
                          #   "sr_zones" in START_VISIBLE (§5) to have the chart's own layer on.
SR_TFS = ("15m", "1h", "4h")   # extra timeframes to cluster levels on, beside TF itself

ZONE_LOOKBACK = 2000      # bars of history each layer clusters over, IN ITS OWN BARS — so 2000 is
                          #   ~7 days on 5m and ~11 months on 4h. Bounded on purpose: a long
                          #   warm-up in a trend fills the chart with levels 20% below price
ZONES_LEFT  = 10          # bars either side a pivot needs before it is confirmed
ZONES_RIGHT = 10

SR_CLUSTER_PCT = 0.15     # two pivots within this % of each other are the same zone
SR_MIN_TOUCHES = 2        # a zone needs this many pivots before it is drawn
SR_MAX_ZONES   = 40       # per layer


# =============================================================================================
# §5  THE CHART
# =============================================================================================
EMAS     = (20, 50, 100, 200)            # of TF
MTF_EMAS = [("1m", 20, 50, 100, 200),    # of other timeframes, stepped at their own bar close.
            ("15m", 20, 50, 100, 200),   #   A timeframe FINER than the chart needs its own real
            ("1h", 20, 50, 100, 200)]    #   frame — the notebook loads it automatically
EMAS_ON  = False          # drawn but starting OFF the chart — the gaps are the subject here

SHOW_TRENDLINES = True

START_VISIBLE = ("trades",)   # what is ON when the chart opens. Gaps and levels are DRAWN but
                          #   start off it -- the chart opens on price and trades, and you switch
                          #   layers on from the `FVGs:` / `S/R:` button rows. Add "fvg_bull",
                          #   "fvg_bear" or "sr_zones" here to have those on from the start.

HEIGHT = 900

# DROPDOWN CAPS. Everything is DRAWN regardless — these only cap how many things get their own
# entry in a picker, because a plotly menu does not scroll, it clips.
MAX_FVG_PICKER   = 6      # isolatable gaps, PER TIMEFRAME PER DIRECTION
MAX_SR_PICKER    = 6      # isolatable S/R zones, PER TIMEFRAME
MAX_PICKER       = 24     # isolatable pattern/trendline instances, all timeframes
MAX_SHAPES_TF    = 12     # shapes drawn per extra timeframe
MAX_TRADE_PICKER = 40     # rows in the trade dropdown


# =============================================================================================
# §6  REWIND — put the chart back to one candle
# =============================================================================================
# TWO WAYS TO DO IT, and they answer slightly different questions.
#
# REWIND (browser tool). Arm "⟲ Rewind" on the chart and click a candle. Every gap, level,
# trendline and trade is clipped to that bar, instantly, with no kernel — a gap filled LATER is
# shown still open, a trendline that broke later is shown unbroken, a trade taken later disappears.
# Candles and EMAs are left alone on purpose: you want to see what happened NEXT.
# Works only in the browser tab (the script is injected by open_in_browser), not in a notebook cell.
# Everything it quotes is UTC, which is what the chart's axis is.
REWIND = True

# AS_OF (rebuild). Set a timestamp and the whole pipeline is re-run with the data stopping there.
# Slower and not clickable, but EXACT and reproducible — nothing is filtered, the detectors simply
# never see the future — and it works anywhere, including inline. Use it to pin a study.
# None = off. e.g. "2026-06-05 20:00"
AS_OF = None
