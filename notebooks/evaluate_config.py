"""
THE EVALUATION — which strategy, and which quarters judge it.  THIS FILE IS YOURS.

evaluate.ipynb runs the strategy as set, and every arm of its COMPARE, over each quarter below.
Choose between arms on TRAIN; read the verdict on FORWARD, which the choice never saw.

If Jupyter offers "File Changed", pick RELOAD.
"""

STRATEGY = "tl_break_confirm"        # a file in notebooks/strategies/

TRAIN    = ("2022Q1", "2024Q2")      # first and last quarter, inclusive -- choose on these
FORWARD  = ("2024Q3", "2026Q3")      # ...and judge on these, never the other way round
DATA_END = "2026-08-31"              # nothing after this is used; the cache ends 2026-09-01 and
                                     #   later bars would be fetched from the exchange

ZERO_COST = True                     # also run it at zero cost: is the result the rule or the costs?
