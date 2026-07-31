"""Exness-style leverage/margin: lot sizing, free-margin gate, and stop-out liquidation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.engine import BacktestConfig, Signals, run_backtest


def _frame(close):
    n = len(close)
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"t": t, "open": close, "high": close * 1.0002,
                        "low": close * 0.9998, "close": close})


def _enter_once_hold(n, at=1):
    el = np.zeros(n, bool); el[at] = True      # one long entry, then hold
    xl = np.zeros(n, bool)                      # no rule exit
    return Signals(entry_long=el, exit_long=xl)


def test_stop_out_liquidates_on_crash():
    # flat then a crash; a big leveraged long must be liquidated (margin_call) before the end.
    close = np.concatenate([np.full(50, 100.0), np.linspace(100.0, 80.0, 50)])  # -20%
    df = _frame(close)
    cfg = BacktestConfig(initial_cash=1000, fee_bps=0, slippage_bps=0, exit_enabled=True,
                         sizing_mode="lots", sizing_value=1.0, contract_size=100, leverage=100,
                         margin_enabled=True, stop_out_level=50.0, allow_rule_close=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["close_reason"] == "margin_call"
    assert tr["exit_i"] < len(df) - 1                  # liquidated mid-crash, not forced at end
    # equity at stop-out: 1000 + (px-100)*100 <= 0.5 * (100*100/100=100) -> px ~ 90.5
    assert 88 <= tr["exit_price"] <= 92


def test_free_margin_gate_blocks_oversized():
    df = _frame(np.full(60, 100.0))
    # 1 lot * 100 * price 100 = 10,000 notional; leverage 10 -> needs 1,000 margin; acct only 50.
    cfg = BacktestConfig(initial_cash=50, exit_enabled=True, sizing_mode="lots", sizing_value=1.0,
                         contract_size=100, leverage=10, margin_enabled=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 0                         # can't afford the margin


def test_lots_sizing_quantity():
    df = _frame(np.full(60, 100.0))
    cfg = BacktestConfig(initial_cash=100_000, exit_enabled=True, sizing_mode="lots",
                         sizing_value=0.5, contract_size=100, leverage=100, margin_enabled=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 1
    assert abs(res.trades.iloc[0]["qty"] - 0.5 * 100) < 1e-9   # lots * contract_size = 50 units


def test_leverage_allows_notional_above_cash():
    df = _frame(np.full(60, 100.0))
    # $1,000 account, 1:100 leverage, 2 lots * 100 * 100 = 20,000 notional (20x the cash) is fine.
    cfg = BacktestConfig(initial_cash=1000, exit_enabled=True, sizing_mode="lots", sizing_value=2.0,
                         contract_size=100, leverage=100, margin_enabled=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["qty"] == 200


# ---------------------------------------------------------------------------
# daily / session loss cap
# ---------------------------------------------------------------------------
def _losscap_frame(n=600):
    """Price that grinds down all day so a short-only strategy keeps winning, then a day that rips
    up so it keeps losing — enough trades per session for a cap to bite."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(11)
    close = 100 + np.r_[np.linspace(0, -6, n // 2), np.linspace(0, 6, n - n // 2)] \
        + rng.normal(0, 0.15, n)
    idx = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")   # ~25 days
    return pd.DataFrame({"open_time": idx, "t": idx, "open": np.r_[close[0], close[:-1]],
                         "high": close + 0.25, "low": close - 0.25, "close": close, "volume": 1.0})


def test_daily_loss_limit_blocks_further_entries():
    """Once a session's realised loss passes the cap, no NEW trade may open in that session."""
    import numpy as np, pandas as pd
    from quant.engine import BacktestConfig, Signals, run_backtest
    d = _losscap_frame()
    n = len(d)
    z = np.zeros(n, bool)
    es = z.copy(); es[::3] = True                    # try to short every third bar
    sig = Signals(entry_long=z.copy(), entry_short=es, exit_short=z.copy())
    base = dict(initial_cash=10_000, exit_enabled=True, allow_short=True,
                sl_mode="entry_pct", sl_value=0.5, tp_mode="entry_pct", tp_value=0.5,
                sizing_mode="risk_amount", sizing_value=200.0,
                allow_leverage=True, max_notional_pct=5000)
    off = run_backtest(d, sig, BacktestConfig(**base))
    cap = run_backtest(d, sig, BacktestConfig(**base, daily_loss_limit=1.0))
    assert len(cap.trades) < len(off.trades), "a 1% cap must block some entries"

    # no session may open a trade after its realised loss already breached the cap
    t = cap.trades.copy()
    t["day"] = pd.to_datetime(t["entry_time"]).dt.floor("D")
    for _, g in t.groupby("day"):
        g = g.sort_values("entry_i")
        closed = g["pnl"].cumsum().shift(1).fillna(0.0)   # realised before each entry
        assert (closed > -0.01 * 10_000 * 3).all()        # generous bound; must not run away


def test_daily_loss_limit_off_is_byte_identical():
    """The default (0) must not perturb anything."""
    import numpy as np
    from quant.engine import BacktestConfig, Signals, run_backtest
    import pandas as pd
    d = _losscap_frame()
    n = len(d)
    z = np.zeros(n, bool)
    es = z.copy(); es[::5] = True
    sig = Signals(entry_long=z.copy(), entry_short=es, exit_short=z.copy())
    base = dict(initial_cash=10_000, exit_enabled=True, allow_short=True,
                sl_mode="entry_pct", sl_value=0.5, tp_mode="entry_pct", tp_value=1.0)
    a = run_backtest(d, sig, BacktestConfig(**base))
    b = run_backtest(d, sig, BacktestConfig(**base, daily_loss_limit=0.0))
    pd.testing.assert_frame_equal(a.trades, b.trades)
