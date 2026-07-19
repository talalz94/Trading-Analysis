"""
Reporting: a compact performance summary and a self-contained HTML report.

`summary(res)` returns a dict of headline stats + attribution tables (great in a notebook).
`print_summary(res)` prints it. `to_html(res, df, path)` assembles a standalone HTML report
(equity/drawdown, price+trades, monthly-returns heatmap, hour x weekday heatmap) — static
figures so the file works offline and outside Jupyter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .. import analytics as A
from ..engine.run import SimResult

HEADLINE = [
    ("total_return_pct", "Total return %", "{:.2f}"),
    ("final_cash", "Final cash", "{:,.2f}"),
    ("num_trades", "Trades", "{:.0f}"),
    ("win_rate_pct", "Win rate %", "{:.2f}"),
    ("profit_factor", "Profit factor", "{:.2f}"),
    ("expectancy_per_trade", "Expectancy / trade", "{:.4f}"),
    ("sharpe", "Sharpe", "{:.2f}"),
    ("sortino", "Sortino", "{:.2f}"),
    ("calmar", "Calmar", "{:.2f}"),
    ("max_drawdown_pct", "Max drawdown %", "{:.2f}"),
    ("recovery_factor", "Recovery factor", "{:.2f}"),
    ("avg_bars_held", "Avg bars held", "{:.0f}"),
    ("total_fees", "Total fees", "{:,.2f}"),
]


def summary(res: SimResult, *, df: Optional[pd.DataFrame] = None, tz: Optional[str] = None) -> Dict:
    """Structured summary: headline stats + attribution tables."""
    s = res.stats
    out = {
        "headline": {label: s.get(key) for key, label, _ in HEADLINE},
        "by_hour": A.by_hour(res.trades, tz=tz),
        "by_weekday": A.by_weekday(res.trades, tz=tz),
        "by_session": A.by_session(res.trades, tz=tz),
        "monthly_returns": A.monthly_returns(res.equity_curve),
    }
    if df is not None:
        out["by_regime"] = A.by_regime(res.trades, df)
    return out


def print_summary(res: SimResult, *, df: Optional[pd.DataFrame] = None, tz: Optional[str] = None) -> None:
    s = res.stats
    print("=" * 52)
    print(" PERFORMANCE SUMMARY")
    print("=" * 52)
    for key, label, fmt in HEADLINE:
        v = s.get(key)
        vs = fmt.format(v) if isinstance(v, (int, float)) else str(v)
        print(f"  {label:22s} {vs:>18}")
    best = A.by_weekday(res.trades, tz=tz)
    if not best.empty:
        top = best.sort_values("total_pnl", ascending=False).iloc[0]
        print(f"\n  Best weekday: {top.get('day', top['weekday'])} "
              f"(pnl={top['total_pnl']:.2f}, win={top['win_rate_pct']:.1f}%)")
    sess = A.by_session(res.trades, tz=tz)
    if not sess.empty:
        top = sess.sort_values("total_pnl", ascending=False).iloc[0]
        print(f"  Best session: {top['session']} "
              f"(pnl={top['total_pnl']:.2f}, n={top['n_trades']})")
    print("=" * 52)


def to_html(res: SimResult, path: str, *, df: Optional[pd.DataFrame] = None,
            price_df: Optional[pd.DataFrame] = None, title: str = "Backtest Report",
            tz: Optional[str] = None) -> str:
    """Write a standalone HTML report and return its path."""
    from ..viz.charts import equity_and_drawdown, price_and_trades
    from ..viz.heatmaps import hour_weekday_heatmap, monthly_returns_heatmap

    figs = []
    if price_df is not None:
        figs.append(price_and_trades(price_df, res.trades, title="Price & Trades"))
    figs.append(equity_and_drawdown(res.equity_curve))
    mr = A.monthly_returns(res.equity_curve)
    if not mr.empty:
        figs.append(monthly_returns_heatmap(mr))
    if not res.trades.empty:
        figs.append(hour_weekday_heatmap(res.trades, metric="total_pnl", tz=tz))

    rows = "".join(
        f"<tr><td>{label}</td><td style='text-align:right'>"
        f"{(fmt.format(res.stats.get(key)) if isinstance(res.stats.get(key), (int, float)) else res.stats.get(key))}"
        f"</td></tr>"
        for key, label, fmt in HEADLINE)
    stats_html = f"<table class='stats'>{rows}</table>"

    chart_html = ""
    for i, fig in enumerate(figs):
        chart_html += fig.to_html(full_html=False, include_plotlyjs=(i == 0), default_height="460px")

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1f2937;background:#fff}}
 h1{{font-size:22px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}}
 table.stats{{border-collapse:collapse;min-width:340px}} table.stats td{{padding:4px 14px;border-bottom:1px solid #f1f5f9}}
 table.stats td:first-child{{color:#6b7280}}
</style></head><body>
<h1>{title}</h1><h2>Headline</h2>{stats_html}<h2>Charts</h2>{chart_html}
</body></html>"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)
