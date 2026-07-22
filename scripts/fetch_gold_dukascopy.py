"""
Fetch true spot XAU/USD (Dukascopy) for 1m/5m/15m/1h/4h from 2026-01-01 to latest.

Resumable + auto-saving: data is pulled in MONTHLY chunks and each chunk is merged into the
incremental Parquet cache as it completes. If the run is interrupted (network, Ctrl-C), just
re-run this script — it reads the cache, works out what's missing, and continues from there.

    python scripts/fetch_gold_dukascopy.py            # all timeframes
    python scripts/fetch_gold_dukascopy.py 5m 15m     # only these

Requires: pip install "quant[data]"  (dukascopy-python).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.data import get_ohlcv, store          # noqa: E402
from quant.logging_utils import get_logger         # noqa: E402

log = get_logger("fetch_gold")

SYMBOL, SOURCE, MARKET = "XAUUSD", "dukascopy", "cfd"
START = pd.Timestamp("2026-01-01", tz="UTC")
DEFAULT_TFS = ["1m", "5m", "15m", "1h", "4h"]


def fetch_tf(tf: str) -> None:
    now = pd.Timestamp.now(tz="UTC")
    # monthly checkpoints from START to now; request a growing end each step so each month is saved
    bounds = list(pd.date_range(START, now, freq="MS")[1:]) + [now]
    log.info("[%s] fetching %s -> %s in %d monthly chunks", tf, START.date(), now.date(), len(bounds))
    for m in bounds:
        try:
            df = get_ohlcv(SYMBOL, tf, start=str(START), end=str(m), source=SOURCE,
                           market=MARKET, tz="UTC", progress=False)
            cmin, cmax = store.cache_range(SYMBOL, tf, source=SOURCE, market=MARKET)
            log.info("[%s] cached %s rows through %s (target %s)", tf, f"{len(df):,}", cmax, m.date())
        except KeyboardInterrupt:
            log.warning("[%s] interrupted at %s — re-run to resume", tf, m.date())
            raise
        except Exception as e:  # noqa: BLE001
            log.error("[%s] chunk to %s failed (%s: %s) — re-run to resume", tf, m.date(),
                      type(e).__name__, e)
            return


def main(argv=None) -> int:
    tfs = argv or DEFAULT_TFS
    for tf in tfs:
        fetch_tf(tf)
    log.info("done. Load with: get_ohlcv('XAUUSD', '<tf>', start='2026-01-01', end=..., source='dukascopy')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
