"""
End-to-end demo: an EMA-ribbon strategy on gold (PAXGUSDT) through the new `quant` stack.

Run from the repo root:
    python examples/gold_ema_demo.py

Shows: data load (pushdown) -> vectorized signals -> numba engine -> metrics -> charts,
plus a bounded-parallel parameter sweep. Writes interactive charts to reports/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from quant.data import get_ohlcv
from quant.engine import BacktestConfig
from quant.strategies import EmaRibbon
from quant.optimize import run_grid
from quant.viz import price_and_trades, equity_and_drawdown

SYMBOL, TF = "PAXGUSDT", "1m"          # PAX Gold — the gold proxy currently cached
START, END = "2025-06-01", "2026-05-31"
TZ = "UTC"

REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    print(f"\n=== Loading {SYMBOL} {TF}  {START}..{END} ===")
    t0 = time.perf_counter()
    df = get_ohlcv(SYMBOL, TF, start=START, end=END, tz=TZ, progress=False)
    print(f"loaded {len(df):,} bars in {(time.perf_counter()-t0):.2f}s")

    cfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
        exit_enabled=True,
        sl_mode="entry_pct", sl_value=0.6,
        tp_mode="rr", tp_value=2.0,
        sizing_mode="risk_pct_equity", sizing_value=1.0,
        allow_rule_close=True, intrabar_priority="stop_first",
    )

    # ---- single run ----
    print("\n=== Single backtest: EmaRibbon(fast=50, slow=200, confirm_n=5) ===")
    strat = EmaRibbon(fast=50, slow=200, confirm_n=5)
    res = strat.backtest(df, cfg)
    show = ["num_trades", "win_rate_pct", "total_return_pct", "profit_factor",
            "sharpe", "sortino", "max_drawdown_pct", "expectancy_per_trade",
            "recovery_factor", "engine_elapsed_s"]
    for k in show:
        print(f"  {k:22s} {res.stats.get(k)}")

    price_and_trades(df, res.trades, title=f"{SYMBOL} — EmaRibbon(50/200/5)").write_html(
        REPORTS / "gold_ema_price_trades.html")
    equity_and_drawdown(res.equity_curve, title=f"{SYMBOL} — Equity & Drawdown").write_html(
        REPORTS / "gold_ema_equity.html")
    print(f"  charts -> {REPORTS/'gold_ema_price_trades.html'} , {REPORTS/'gold_ema_equity.html'}")

    # ---- parameter sweep ----
    print("\n=== Parameter sweep ===")
    grid = {
        "fast": [20, 30, 50, 75, 100],
        "slow": [150, 200, 300],
        "confirm_n": [1, 3, 5, 10],
    }
    results = run_grid(
        df, EmaRibbon, grid, cfg,
        valid_fn=lambda p: p["fast"] < p["slow"],
        keep_stats=["num_trades", "total_return_pct", "win_rate_pct", "profit_factor",
                    "sharpe", "max_drawdown_pct"],
        progress=False,
    )
    top = results.sort_values("total_return_pct", ascending=False).head(8)
    print(top.to_string(index=False))
    out = REPORTS / "gold_ema_sweep.csv"
    results.to_csv(out, index=False)
    print(f"  full results -> {out}")


if __name__ == "__main__":
    main()
