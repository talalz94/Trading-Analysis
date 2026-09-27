"""
THE LAB — which strategy, which window, and how it is drawn.  THIS FILE IS YOURS.

The strategy's rules live in notebooks/strategies/<STRATEGY>.py. This file never states a trading
setting, and the lab refuses a setting found in both, so each value has exactly one home.

  §1 which and when   §2 gaps   §3 S/R   §4 chart   §5 rewind

If Jupyter offers "File Changed", pick RELOAD.
"""

# ── §1  WHICH AND WHEN ───────────────────────────────────────────────────────────────────────
STRATEGY     = "tl_break_confirm"    # a file in notebooks/strategies/
VIEW_START   = "2026-06-01"
END          = "2026-06-20"
WARMUP_START = "2026-05-01"          # history for lines and gaps; not traded

# ── §2  GAPS — which the strategy detects is its own; this is only how they are drawn ──────────
FVG_DRAW   = True
FVG_EXTEND = True                    # an unmitigated box runs to the right edge
FVG_SHOW   = ("bull", "bear")

# ── §3  SUPPORT / RESISTANCE — drawn, never traded ──────────────────────────────────────────────
SHOW_SR_ZONES  = True                # layers start legendonly; the `S/R:` row switches them on
SR_TFS         = ("15m", "1h", "4h") # clustered per timeframe, in its own bar space
ZONE_LOOKBACK  = 2000                # bars of history per layer, IN ITS OWN BARS
SR_MAX_ZONES   = 40                  # per layer
SR_SENSITIVITY = "normal"            # "tight" | "normal" | "loose" | None = the pre-2026-09-27 zones

# ── §4  THE CHART ────────────────────────────────────────────────────────────────────────────
EMAS     = (20, 50, 100, 200)        # of the strategy's TF
MTF_EMAS = [("1m", 20, 50, 100, 200),
            ("15m", 20, 50, 100, 200),
            ("1h", 20, 50, 100, 200)]
EMAS_ON  = False                     # drawn, but off at open
SHOW_TRENDLINES = True
START_VISIBLE   = ("trades",)        # + "fvg_bull" / "fvg_bear" / "sr_zones"
SHOW_REJECTED   = True               # the breaks that did NOT trade; always start off the chart
HEIGHT          = 900

# dropdown caps — everything is DRAWN regardless; a plotly menu clips rather than scrolls
MAX_FVG_PICKER   = 6                 # per timeframe per direction
MAX_SR_PICKER    = 6                 # per timeframe
MAX_PICKER       = 24                # pattern/trendline instances, all timeframes
MAX_SHAPES_TF    = 12                # shapes per extra timeframe
MAX_TRADE_PICKER = 40

# ── §5  REWIND ───────────────────────────────────────────────────────────────────────────────
REWIND = True                        # click a candle in the browser; everything clips to it
AS_OF  = None                        # or "2026-06-05 20:00" to rebuild on truncated data
