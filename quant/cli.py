"""
Command-line interface: `quant <command>`.

  quant fetch    PAXGUSDT 1m --start 2025-06-01 --end 2026-05-31 [--source binance --market spot]
  quant backtest --symbol PAXGUSDT --tf 1m --start 2025-06-01 --strategy ema_ribbon \
                 --params '{"fast":50,"slow":200,"confirm_n":5}' --sl-value 0.6 --tp rr:2.0
  quant report   ...same as backtest... --out reports/run.html

Run `quant <command> -h` for the full flag list.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .data import get_ohlcv
from .engine import BacktestConfig
from .strategies import REGISTRY


def _add_data_args(p):
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", default="1m")
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--source", default="binance")
    p.add_argument("--market", default="spot")
    p.add_argument("--tz", default="UTC")


def _add_cfg_args(p):
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    p.add_argument("--params", default="{}", help="JSON dict of strategy params")
    p.add_argument("--cash", type=float, default=10_000)
    p.add_argument("--fee-bps", type=float, default=8.0)
    p.add_argument("--slippage-bps", type=float, default=1.5)
    p.add_argument("--sl-mode", default="entry_pct")
    p.add_argument("--sl-value", type=float, default=0.6)
    p.add_argument("--tp", default="rr:2.0", help="MODE:VALUE, e.g. rr:2.0 or entry_pct:1.5 or none")
    p.add_argument("--sizing-mode", default="risk_pct_equity")
    p.add_argument("--sizing-value", type=float, default=1.0)
    p.add_argument("--no-exit", action="store_true", help="disable SL/TP (signal-only)")


def _build_cfg(a) -> BacktestConfig:
    tp_mode, tp_value = "none", 0.0
    if a.tp and a.tp.lower() != "none":
        tp_mode, tp_value = a.tp.split(":")
        tp_value = float(tp_value)
    return BacktestConfig(
        initial_cash=a.cash, fee_bps=a.fee_bps, slippage_bps=a.slippage_bps,
        exit_enabled=not a.no_exit, sl_mode=a.sl_mode, sl_value=a.sl_value,
        tp_mode=tp_mode, tp_value=tp_value,
        sizing_mode=a.sizing_mode, sizing_value=a.sizing_value,
    )


def _run(a):
    df = get_ohlcv(a.symbol, a.tf, start=a.start, end=a.end, source=a.source,
                   market=a.market, tz=a.tz)
    strat = REGISTRY[a.strategy](**json.loads(a.params))
    return df, strat.backtest(df, _build_cfg(a), time_col="t", price_col="close")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="quant", description="Quant backtesting CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="download/update cached data")
    pf.add_argument("symbol")
    pf.add_argument("tf")
    pf.add_argument("--start", required=True)
    pf.add_argument("--end", default=None)
    pf.add_argument("--source", default="binance")
    pf.add_argument("--market", default="spot")

    pb = sub.add_parser("backtest", help="run a strategy and print a summary")
    _add_data_args(pb)
    _add_cfg_args(pb)

    pr = sub.add_parser("report", help="run a strategy and write an HTML report")
    _add_data_args(pr)
    _add_cfg_args(pr)
    pr.add_argument("--out", default="reports/report.html")

    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        df = get_ohlcv(a.symbol, a.tf, start=a.start, end=a.end, source=a.source,
                       market=a.market, tz="UTC", refresh=True)
        print(f"cached {len(df):,} bars {df['open_time'].min()} -> {df['open_time'].max()}")
        return 0

    df, res = _run(a)
    from .reporting import print_summary
    print_summary(res, df=df)

    if a.cmd == "report":
        from .reporting import to_html
        out = to_html(res, a.out, df=df, price_df=df, title=f"{a.symbol} {a.strategy}")
        print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
