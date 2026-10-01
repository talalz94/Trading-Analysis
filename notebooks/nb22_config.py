"""
NOTEBOOK 22 — NOTEBOOK 21'S STRATEGY ON OTHER MARKETS.  THIS FILE IS YOURS.

Every TRADING setting is read from nb21_config.py each time this runs, so notebook 22 always tests
exactly what notebook 21 trades -- change a rule there and this follows. Only what has to differ per
market lives here: its symbol, where its data comes from, and what a trade costs on it.

  §1 the strategy   §2 markets   §3 quarters   §4 chart

If Jupyter offers "File Changed", pick RELOAD.
"""

# ══ §1  THE STRATEGY ═════════════════════════════════════════════════════════════════════════
BASE = "nb21_config.py"              # notebook 21's settings; its SYMBOL and costs are replaced

# ══ §2  MARKETS — costs REPLACE notebook 21's fee_bps / slippage_bps / spread (unset = 0) ═══════
MARKETS = {
    "BTC":     dict(symbol="BTCUSDT", source="binance",   fee_bps=8, slippage_bps=1.5),  # 21's own
    "Gold":    dict(symbol="XAUUSD",  source="dukascopy", spread=0.25),      # $0.25 an ounce
    "WTI oil": dict(symbol="WTI",     source="dukascopy", spread=0.03),      # 3 cents a barrel
    "EURUSD":  dict(symbol="EURUSD",  source="dukascopy", spread=0.00008),   # 0.8 pip, all-in
    "GBPUSD":  dict(symbol="GBPUSD",  source="dukascopy", spread=0.00012),   # 1.2 pips
    "USDJPY":  dict(symbol="USDJPY",  source="dukascopy", spread=0.010),     # 1.0 pip
    # the yen crosses: added to test whether USDJPY's result REPLICATES on pairs never looked at
    "EURJPY":  dict(symbol="EURJPY",  source="dukascopy", spread=0.015),     # 1.5 pips
    "GBPJPY":  dict(symbol="GBPJPY",  source="dukascopy", spread=0.025),     # 2.5 pips
}

# ══ §3  QUARTERS — choose on TRAIN, judge on FORWARD ═════════════════════════════════════════
TRAIN      = ("2022Q1", "2024Q2")
FORWARD    = ("2024Q3", "2026Q3")
DATA_END   = "2026-08-31"            # data is downloaded up to here
NULL_EVERY = 12                      # the metronome enters every this many bars (12 x 5m = hourly)
RERUN      = False                   # False = read the saved scan; True = run it again (~20 min)

# ══ §4  THE CHART — one market, drawn with notebook 21's chart settings ══════════════════════
CHART_MARKET = "USDJPY"              # a name from MARKETS
CHART_START  = "2026-06-01"
CHART_END    = "2026-06-20"
