"""Tests for the declarative rule-builder (conditions + RuleStrategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.engine import BacktestConfig
from quant.strategies import RuleStrategy, Rule, EmaCross
from quant.strategies.conditions import (
    ema, mtf_ema, heikin_ashi, swings, above, below, full_above, full_below, green, ha_red,
    big_candle, big_move_up, htf_ema_col, apply_indicators, ref_col_stop, pct_stop,
    sr_zones, ma, near_support, near_resistance, bounce_off_support,
)
from quant.indicators import add_sr_zones
from quant.indicators.overlays import ema as ema_series


@pytest.fixture(scope="module")
def gold_df():
    rng = np.random.default_rng(1)
    n = 4000
    ret = rng.normal(0, 0.001, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ot = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": ot, "t": ot, "open": open_, "high": high,
                        "low": low, "close": close, "volume": rng.random(n)})


def _cfg():
    return BacktestConfig(initial_cash=10_000, spread=0.1, slippage_bps=1.0, exit_enabled=True,
                          sl_mode="ref_col", sl_ref_long_col="swing_last_low",
                          sl_fallback_mode="entry_pct", sl_fallback_value=1.0, tp_mode="none",
                          sizing_mode="risk_pct_equity", sizing_value=1.0, allow_rule_close=True)


# ---- condition algebra ----
def test_cond_operators_match_columns(gold_df):
    prep = apply_indicators(gold_df.copy(), [ema(50), heikin_ashi()])
    # full_above(ema_50) == low > ema_50 ; & and | compose elementwise
    fa = full_above("ema_50")(prep)
    assert np.array_equal(fa, (prep["low"] > prep["ema_50"]).to_numpy())
    both = (above("close", "ema_50") & below("close", "high"))(prep)
    assert np.array_equal(both, ((prep["close"] > prep["ema_50"]) &
                                 (prep["close"] < prep["high"])).to_numpy())
    inv = (~above("close", "ema_50"))(prep)
    assert np.array_equal(inv, ~(prep["close"] > prep["ema_50"]).to_numpy())


def test_ha_red_and_green(gold_df):
    prep = apply_indicators(gold_df.copy(), [heikin_ashi()])
    har = ha_red(3)(prep)
    manual = ((prep["ha_close"] < prep["ha_open"]).rolling(3).sum() == 3).to_numpy()
    manual[:2] = False
    assert np.array_equal(har, manual)
    hag = green(2, src="ha")(prep)
    manual_g = ((prep["ha_close"] > prep["ha_open"]).rolling(2).sum() == 2).to_numpy()
    manual_g[:1] = False
    assert np.array_equal(hag, manual_g)


# ---- builder ↔ EmaCross parity for the primary rule ----
def test_breakout_rule_matches_emacross(gold_df):
    rs = RuleStrategy(indicators=[ema(200), swings(5, 5)], reset=full_below("ema_200"),
                      rules=[Rule("breakout", entry=full_above("ema_200", n=2),
                                  exit=full_below("ema_200"), once=True)])
    ec = EmaCross(ema_period=200, entry_mode="full_candle", confirm_n=2, allow_long=True,
                  allow_short=False, exit_mode="opposite", one_entry_per_cross=True,
                  swings_left=5, swings_right=5)
    sr = rs.build_signals(rs.prepare(gold_df.copy()))
    se = ec.build_signals(ec.prepare(gold_df.copy()))
    assert np.array_equal(np.asarray(sr.entry_long, bool), np.asarray(se.entry_long, bool))


# ---- sequencing: `after` gates a rule behind an earlier one, per episode ----
def test_after_gating_is_strict_and_per_episode(gold_df):
    strat = RuleStrategy(
        indicators=[ema(200), ema(20), heikin_ashi(), swings(5, 5)], reset=full_below("ema_200"),
        rules=[Rule("breakout", entry=full_above("ema_200", n=2), exit=full_below("ema_20"), once=True),
               Rule("pullback", entry=above("close", "ema_200") & full_above("ema_20", n=2),
                    exit=full_below("ema_20"), after="breakout", once=True)])
    prep = strat.prepare(gold_df.copy())
    c = strat._compile(prep)
    _, bo_fired, _, _ = c.long_legs[0]
    _, pb_fired, _, _ = c.long_legs[1]
    reset = full_below("ema_200")(prep)
    bo_idx = np.where(bo_fired)[0]
    reset_idx = np.where(reset)[0]
    assert pb_fired.sum() > 0 and bo_fired.sum() > 0, "test setup should produce both fires"
    for p in np.where(pb_fired)[0]:
        last_reset = reset_idx[reset_idx < p].max() if (reset_idx < p).any() else -1
        # a breakout must fire STRICTLY before this pullback and after the last episode reset
        assert any(last_reset < b < p for b in bo_idx), f"pullback {p} not gated behind a breakout"


def test_after_must_reference_earlier_rule(gold_df):
    strat = RuleStrategy(indicators=[ema(200)],
                         rules=[Rule("a", entry=full_above("ema_200"), after="b"),
                                Rule("b", entry=full_above("ema_200"))])
    with pytest.raises(ValueError):
        strat.build_signals(strat.prepare(gold_df.copy()))


# ---- volatility, MTF, smoothing, AND/OR ----
def test_big_candle_range_and_body(gold_df):
    prep = gold_df.copy()
    bc = big_candle(mult=2.0, lookback=20)(prep)
    rng = prep["high"] - prep["low"]
    avg = rng.rolling(20, min_periods=20).mean().shift(1)
    assert np.array_equal(bc, (rng > 2.0 * avg).to_numpy())
    # big_move_up requires the candle to also be green
    bu = big_move_up(mult=1.5, lookback=20)(prep)
    assert np.all(prep["close"].to_numpy()[bu] > prep["open"].to_numpy()[bu])


def test_ema_smoothing_is_ema_of_ema(gold_df):
    prep = apply_indicators(gold_df.copy(), [ema(100, smooth=9)])
    expected = ema_series(ema_series(gold_df["close"], 100), 9).to_numpy()
    assert np.allclose(prep["ema_100"].to_numpy(), expected, equal_nan=True)
    raw = apply_indicators(gold_df.copy(), [ema(100)])
    both = prep["ema_100"].notna() & raw["ema_100"].notna()
    assert not np.allclose(prep["ema_100"][both], raw["ema_100"][both])   # smoothing changes the line


def test_mtf_ema_bias_filter_anti_lookahead(gold_df):
    prep = apply_indicators(gold_df.copy(), [mtf_ema("15min", 50)])
    col = htf_ema_col("15min", 50)
    assert col in prep.columns and prep[col].notna().sum() > 0
    # the 15m EMA is piecewise-constant on the 1m grid (updates only ~once per 15m bar)
    changes = int(prep[col].dropna().diff().ne(0).sum())
    assert changes < len(prep) / 10


def test_conditions_compose_and_or_in_one_rule(gold_df):
    # #1 AND/OR/NOT: a single rule can require several conditions at once (AND), alternatives (OR),
    # and negation (NOT) — composed with & | ~ on exact elementwise conditions.
    prep = apply_indicators(gold_df.copy(), [ema(200), ema(50), mtf_ema("15min", 50)])
    expr = (above("close", "15min__ema_50") & above("close", "ema_200")) | (~above("close", "ema_50"))
    m = expr(prep)
    a = (prep["close"] > prep["15min__ema_50"]).to_numpy()
    b = (prep["close"] > prep["ema_200"]).to_numpy()
    cc = ~(prep["close"] > prep["ema_50"]).to_numpy()
    assert np.array_equal(m, (a & b) | cc)


def test_mtf_bias_reduces_entries(gold_df):
    """A HTF bias filter (AND-ed) never increases entries vs the unfiltered rule."""
    base = RuleStrategy(indicators=[ema(200)], reset=full_below("ema_200"),
                        rules=[Rule("b", entry=full_above("ema_200", n=2),
                                    exit=full_below("ema_200"), once=True)])
    filt = RuleStrategy(indicators=[ema(200), mtf_ema("15min", 50)], reset=full_below("ema_200"),
                        rules=[Rule("b", entry=full_above("ema_200", n=2) & above("close", "15min__ema_50"),
                                    exit=full_below("ema_200"), once=True)])
    sb = base.build_signals(base.prepare(gold_df.copy()))
    sf = filt.build_signals(filt.prepare(gold_df.copy()))
    assert np.asarray(sf.entry_long, bool).sum() <= np.asarray(sb.entry_long, bool).sum()


# ---- per-rule exits (leg-aware kernel) ----
def test_shared_exit_uses_fast_path(gold_df):
    """Identical exits per side compile to a plain Signals (no leg arrays -> standard kernel)."""
    strat = RuleStrategy(indicators=[ema(200), ema(20), swings(5, 5)], reset=full_below("ema_200"),
        rules=[Rule("a", entry=full_above("ema_200", n=2), exit=full_below("ema_20"), once=True),
               Rule("b", entry=above("close", "ema_200") & full_above("ema_20", n=2),
                    exit=full_below("ema_20"), after="a", once=True)])
    sig = strat.signals(strat.prepare(gold_df.copy()))
    assert sig.exit_legs_long is None, "shared exits should not engage the leg path"


def test_distinct_exits_close_on_own_rule(gold_df):
    """With different exits, every signal-closed trade closes on the exit of the rule that opened it."""
    strat = RuleStrategy(indicators=[ema(200), ema(50), ema(20), swings(5, 5)],
        reset=full_below("ema_200"),
        rules=[Rule("breakout", entry=full_above("ema_200", n=2), exit=full_below("ema_50"), once=True),
               Rule("pullback", entry=above("close", "ema_200") & full_above("ema_20", n=2),
                    exit=full_below("ema_20"), after="breakout", once=True)])
    prep = strat.prepare(gold_df.copy())
    sig = strat.signals(prep)
    assert sig.exit_legs_long is not None and sig.exit_legs_long.shape[0] == 2
    res = strat.backtest(gold_df, _cfg())
    fb50 = full_below("ema_50")(prep.reset_index(drop=True))
    fb20 = full_below("ema_20")(prep.reset_index(drop=True))
    ell = sig.entry_leg_long
    checked = 0
    for _, tr in res.trades.iterrows():
        if tr["close_reason"] != "signal":
            continue
        checked += 1
        leg = ell[int(tr["entry_i"])]
        assert (fb50 if leg == 0 else fb20)[int(tr["exit_i"])], "trade did not close on its own rule's exit"
    assert checked > 0


def test_leg_kernel_matches_single_leg_when_exits_identical(gold_df):
    """Golden-style: forcing the leg path with two IDENTICAL exit columns reproduces the standard
    single-exit backtest exactly (trade-for-trade)."""
    from quant.engine import Signals
    from quant.engine.run import run_backtest
    strat = EmaCross(ema_period=50, entry_mode="full_candle", confirm_n=2, allow_long=True,
                     allow_short=False, exit_mode="below_ema", exit_ema=20, swings_left=5, swings_right=5)
    prep = strat.prepare(gold_df.copy())
    base = strat.signals(prep)
    n = len(prep)
    el, xl, es, xs = base.as_u8(n)
    legged = Signals(
        entry_long=base.entry_long, exit_long=base.exit_long,
        entry_leg_long=np.zeros(n, np.int64), entry_leg_short=np.zeros(n, np.int64),
        exit_legs_long=np.stack([xl, xl]), exit_legs_short=np.stack([xs, xs]))  # leg 0 == the exit
    r_base = run_backtest(prep, base, _cfg())
    r_leg = run_backtest(prep, legged, _cfg())
    pd.testing.assert_frame_equal(r_base.trades, r_leg.trades)


# ---- support/resistance zones ----
def test_sr_zones_bracket_price_and_are_causal(gold_df):
    p = add_sr_zones(gold_df.copy(), left=8, right=8, cluster_pct=0.15)
    for c in ["sr_support", "sr_resistance", "sr_support_touches", "sr_resistance_touches"]:
        assert c in p.columns
    m_s = p["sr_support"].notna()
    m_r = p["sr_resistance"].notna()
    assert (p.loc[m_s, "sr_support"] <= p.loc[m_s, "close"]).all()      # support below price
    assert (p.loc[m_r, "sr_resistance"] >= p.loc[m_r, "close"]).all()   # resistance above price
    # anti-lookahead: recomputing on a truncated frame yields the SAME level at the boundary
    k = min(1500, len(gold_df))
    full = add_sr_zones(gold_df.copy(), 8, 8, cluster_pct=0.15)["sr_support"].to_numpy()[k - 1]
    trunc = add_sr_zones(gold_df.iloc[:k].copy(), 8, 8, cluster_pct=0.15)["sr_support"].to_numpy()[k - 1]
    assert np.isclose(full, trunc, equal_nan=True)


def test_sr_min_touches_filters_strength(gold_df):
    p = add_sr_zones(gold_df.copy(), left=8, right=8, cluster_pct=0.15)
    weak = near_support(within_pct=0.2, min_touches=1)(p).sum()
    strong = near_support(within_pct=0.2, min_touches=3)(p).sum()
    assert strong <= weak            # requiring more touches never fires more often


def test_sr_bounce_strategy_runs(gold_df):
    strat = RuleStrategy(
        indicators=[ema(200), swings(5, 5), sr_zones(left=8, right=8, cluster_pct=0.12)],
        rules=[Rule("bounce",
                    entry=bounce_off_support(within_pct=0.1, min_touches=2) & above("close", "ema_200"),
                    exit=near_resistance(within_pct=0.1, min_touches=1),
                    stop=ref_col_stop("sr_support", fallback_pct=0.5))])
    res = strat.backtest(gold_df, _cfg())
    assert res.equity_curve.shape[0] == len(gold_df)
    if not res.trades.empty:
        assert res.trades["exit_time"].notna().all()


def test_ma_configurable_matches_tradingview_fields(gold_df):
    from quant.indicators.overlays import sma as sma_series
    p = apply_indicators(gold_df.copy(), [ma(20, type="sma", smooth=9, smooth_type="ema")])
    expected = ema_series(sma_series(gold_df["close"], 20), 9).to_numpy()
    assert np.allclose(p["ma_20"].to_numpy(), expected, equal_nan=True)


# ---- per-rule stops (leg-aware kernel) ----
def test_per_rule_stops_apply_own_method(gold_df):
    """Each trade's stop is set by the stop method of the rule that opened it."""
    strat = RuleStrategy(indicators=[ema(200), ema(20), swings(5, 5)], reset=full_below("ema_200"),
        rules=[Rule("breakout", entry=full_above("ema_200", n=2), exit=full_below("ema_20"),
                    stop=ref_col_stop("swing_last_low", fallback_pct=1.0), once=True),
               Rule("pullback", entry=above("close", "ema_200") & full_above("ema_20", n=2),
                    exit=full_below("ema_20"), stop=pct_stop(0.3), after="breakout", once=True)])
    cfg = BacktestConfig(initial_cash=10_000, spread=0.1, slippage_bps=1.0, exit_enabled=True,
                         sl_mode="none", tp_mode="none", sizing_mode="risk_pct_equity",
                         sizing_value=1.0, allow_rule_close=True)
    prep = strat.prepare(gold_df.copy())
    sig = strat.signals(prep)
    assert sig.stop_specs is not None and len(sig.stop_specs) == 2
    res = strat.backtest(gold_df, cfg)
    ell = sig.entry_leg_long
    swing = prep.reset_index(drop=True)["swing_last_low"].to_numpy()
    checked = 0
    for _, tr in res.trades.iterrows():
        ei = int(tr["entry_i"]); leg = ell[ei]; stop = tr["stop_price"]; entry = tr["entry_price"]
        checked += 1
        if leg == 1:                                   # pullback -> fixed 0.3% stop
            assert abs(stop - entry * (1 - 0.003)) < 0.05
        else:                                          # breakout -> swing-low OR the 1% fallback
            assert abs(stop - swing[ei]) < 0.5 or abs(stop - entry * (1 - 0.01)) < 0.05
    assert checked > 0


def test_no_stop_uses_cfg_stop(gold_df):
    """Rules without a stop= don't engage leg-stop mode; the cfg stop applies (fast path)."""
    strat = RuleStrategy(indicators=[ema(200), ema(20), swings(5, 5)], reset=full_below("ema_200"),
        rules=[Rule("b", entry=full_above("ema_200", n=2), exit=full_below("ema_20"), once=True)])
    sig = strat.signals(strat.prepare(gold_df.copy()))
    assert sig.stop_specs is None


# ---- full backtest runs and respects the sequence ----
def test_rule_strategy_backtests(gold_df):
    strat = RuleStrategy(
        indicators=[ema(200), ema(20), heikin_ashi(), swings(5, 5)], reset=full_below("ema_200"),
        rules=[Rule("breakout", entry=full_above("ema_200", n=2),
                    exit=full_below("ema_20") | ha_red(3), once=True),
               Rule("pullback", entry=above("close", "ema_200") & full_above("ema_20", n=2),
                    exit=full_below("ema_20") | ha_red(3), after="breakout", once=True)])
    res = strat.backtest(gold_df, _cfg())
    assert res.equity_curve.shape[0] == len(gold_df)
    assert {"total_return_pct", "sharpe", "max_drawdown_pct"}.issubset(res.stats)
    if not res.trades.empty:
        assert res.trades["exit_time"].notna().all()


# ---- tagged indicator handles ----
def test_tags_are_columns_and_dont_collide(gold_df):
    from quant.strategies import MA, RSI, HeikinAshi, Swings
    rsi_f, rsi_s = RSI("rsi_f", 7), RSI("rsi_s", 21)
    trend = MA("trend", 100, type="sma", smooth=9, smooth_type="ema")
    ha, sw = HeikinAshi("ha"), Swings("sw", 5, 5)
    strat = RuleStrategy(indicators=[trend, rsi_f, rsi_s, ha, sw],
                         rules=[Rule("r", entry=above("close", trend))])
    prep = strat.prepare(gold_df.copy())
    for c in ["trend", "rsi_f", "rsi_s", "ha_close", "sw_last_low"]:
        assert c in prep.columns
    # same indicator kind, different settings -> genuinely different series
    both = prep["rsi_f"].notna() & prep["rsi_s"].notna()
    assert not np.allclose(prep.loc[both, "rsi_f"], prep.loc[both, "rsi_s"])
    # sub-column handles resolve to real column names
    assert ha.close == "ha_close" and sw.last_low == "sw_last_low" and trend.column == "trend"


def test_duplicate_tag_different_settings_raises(gold_df):
    from quant.strategies import MA
    strat = RuleStrategy(indicators=[MA("x", 10), MA("x", 20)],
                         rules=[Rule("r", entry=above("close", "x"))])
    with pytest.raises(ValueError, match="declared twice"):
        strat.prepare(gold_df.copy())


def test_identical_tag_redeclaration_is_deduped(gold_df):
    from quant.strategies import MA
    a = MA("x", 10, type="sma")
    strat = RuleStrategy(indicators=[a, MA("x", 10, type="sma")],
                         rules=[Rule("r", entry=above("close", a))])
    prep = strat.prepare(gold_df.copy())          # must not raise
    assert "x" in prep.columns


def test_handles_work_in_conditions_and_stops(gold_df):
    from quant.strategies import MA, HeikinAshi, Swings
    trend, fast = MA("trend", 200, type="sma", smooth=9), MA("fast", 20, type="sma", smooth=9)
    ha, sw = HeikinAshi("ha"), Swings("sw", 5, 5)
    strat = RuleStrategy(
        indicators=[trend, fast, ha, sw], reset=full_below(trend),
        rules=[Rule("b", entry=full_above(trend, n=2), exit=full_below(fast) | ha_red(3, src=ha),
                    stop=ref_col_stop(sw.last_low, fallback_pct=1.0), once=True)])
    res = strat.backtest(gold_df, _cfg())
    assert res.equity_curve.shape[0] == len(gold_df)


def test_mtf_tag_is_piecewise_constant(gold_df):
    from quant.strategies import MA
    h = MA("h1bias", 5, type="ema", tf="1h")      # short period so it warms up on the fixture
    strat = RuleStrategy(indicators=[h], rules=[Rule("r", entry=above("close", h))])
    prep = strat.prepare(gold_df.copy())
    assert "h1bias" in prep.columns and prep["h1bias"].notna().any()
    assert int(prep["h1bias"].dropna().diff().ne(0).sum()) < len(prep) / 10


# ---- S/R zone table (for plotting) ----
def test_sr_zone_table_matches_nearest_support_column(gold_df):
    """The zone timeline, reduced to the nearest zone below price, reproduces add_sr_zones'
    sr_support column — so the drawn zones match what a strategy would react to."""
    from quant.indicators import sr_zone_table, add_sr_zones
    zt = sr_zone_table(gold_df.copy(), 10, 10, cluster_pct=0.15)
    p = add_sr_zones(gold_df.copy(), 10, 10, cluster_pct=0.15)
    assert not zt.empty and {"zone", "start_i", "end_i", "price", "touches"}.issubset(zt.columns)
    close = p["close"].to_numpy(); sup = p["sr_support"].to_numpy()
    for i in (300, 700, 1200, 1800, len(gold_df) - 1):
        active = zt[(zt["start_i"] <= i) & (zt["end_i"] >= i)]
        below = active[active["price"] <= close[i]]
        exp = below["price"].max() if len(below) else np.nan
        assert (np.isnan(exp) and np.isnan(sup[i])) or np.isclose(exp, sup[i], atol=1e-6)


def test_sr_zone_segments_are_causal(gold_df):
    """A zone segment never starts before its pivot could be confirmed (start_i >= right)."""
    from quant.indicators import sr_zone_table
    zt = sr_zone_table(gold_df.copy(), left=8, right=8, cluster_pct=0.15)
    assert (zt["start_i"] >= 8).all()
    assert (zt["end_i"] >= zt["start_i"]).all()
