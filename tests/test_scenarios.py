"""Tests for the versioned, composable scenario framework (quant.scenarios)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import scenarios as sc
from quant.engine import BacktestConfig, Signals, run_backtest
from quant.scenarios import Preset
from quant.strategies import conditions as C


# ---------------------------------------------------------------------------
# synthetic OHLCV with a clear support-break-short shape
# ---------------------------------------------------------------------------
def _synthetic(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # up-trend, a top, then a rollover + breakdown, repeated a few times
    legs = []
    price = 100.0
    for _ in range(4):
        legs += list(price + np.linspace(0, 12, 90) + rng.normal(0, 0.3, 90)); price += 12   # rally
        legs += list(price + rng.normal(0, 0.4, 40))                                          # top/chop
        legs += list(price - np.linspace(0, 14, 95) + rng.normal(0, 0.3, 95)); price -= 14    # breakdown
    close = np.array(legs[:n], dtype=float)
    if len(close) < n:
        close = np.pad(close, (0, n - len(close)), mode="edge")
    high = close + np.abs(rng.normal(0.4, 0.2, n))
    low = close - np.abs(rng.normal(0.4, 0.2, n))
    open_ = np.r_[close[0], close[:-1]]
    t = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_, "high": high, "low": low,
                         "close": close, "volume": 1.0})


@pytest.fixture(scope="module")
def df():
    return _synthetic()


# ---------------------------------------------------------------------------
# new conditions
# ---------------------------------------------------------------------------
def test_broke_below_support_hand():
    d = pd.DataFrame({"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1],
                      "close": [105, 101, 99, 98],
                      "sr_support": [100, 100, 97, 97], "sr_support_touches": [2, 2, 2, 2]})
    got = C.broke_below_support()(d)
    assert got.tolist() == [False, False, True, False]   # bar 2 closes below prior support 100


def test_broke_below_prev_low_hand():
    d = pd.DataFrame({"open": [10] * 5, "high": [11, 11, 11, 11, 7], "low": [9, 8, 9, 7, 6],
                      "close": [10, 10, 10, 10, 6.5]})
    got = C.broke_below_prev_low(3, src="real")(d)
    assert got.tolist() == [False, False, False, False, True]   # 6.5 < min(8,9,7)=7


def test_n_red_below_hand():
    d = pd.DataFrame({"open": [10, 10, 10, 10], "high": [9, 9, 9, 9], "low": [1, 1, 1, 1],
                      "close": [9.5, 8.0, 7.0, 6.0], "ema": [12, 12, 12, 12]})
    # all candles red (close<open) and fully below ema(12); n=2 true from bar 1 on
    got = C.n_red_below(2, "ema", src="real")(d)
    assert got.tolist() == [False, True, True, True]


# ---------------------------------------------------------------------------
# registries + every version prepares/signals/backtests
# ---------------------------------------------------------------------------
def test_registries_populated():
    assert "support_break_short@v1.0" in sc.SCENARIOS
    assert {"pct@v1", "swing_high@v1", "high_of_n@v1", "zone_top@v1", "atr@v1"} <= set(sc.STOP_METHODS)
    assert {"rr@v1", "trailing@v1", "n_green_exit@v1", "next_zone@v1", "rr_partials@v1"} <= set(sc.TP_METHODS)


@pytest.mark.parametrize("scenario", ["support_break_short@v1.0", "support_break_short@v1.1",
                                      "support_break_short@v1.2", "support_break_short@v1.3"])
def test_every_scenario_runs_short_only(df, scenario):
    r = sc.run_pieces(df, scenario, "swing_high@v1", "rr@v1",
                      params={"grad_max": 0.0, "tp_rr": 2.0}, cfg={"spread": 0.0})
    assert r.result.elapsed_s >= 0.0
    if len(r.trades):
        assert set(r.trades["side"]) == {"short"}          # single source of truth: shorts only
        # every short stop sits at/above its entry
        assert bool((r.trades["stop_price"] >= r.trades["entry_price"]).all())


@pytest.mark.parametrize("stop", ["pct@v1", "swing_high@v1", "prev_high@v1", "leg_high@v1",
                                  "high_of_n@v1", "zone_top@v1", "atr@v1"])
@pytest.mark.parametrize("tp", ["rr@v1", "fixed_pct@v1", "trailing@v1", "n_green_exit@v1",
                                "back_above_ema@v1", "next_zone@v1", "opposite_candle@v1",
                                "rr_partials@v1"])
def test_stop_tp_matrix_runs(df, stop, tp):
    r = sc.run_pieces(df, "support_break_short@v1.0", stop, tp,
                      params={"grad_max": 0.0}, cfg={"spread": 0.0})
    assert "total_return_pct" in r.stats
    assert r.cfg.exit_enabled is True


@pytest.mark.parametrize("mode", ["break_close", "wait_n", "n_red", "pullback"])
def test_entry_modes_run(df, mode):
    r = sc.run_pieces(df, "support_break_short@v1.0", "swing_high@v1", "rr@v1",
                      params={"grad_max": 0.0, "entry_mode": mode}, cfg={"spread": 0.0})
    assert "num_trades" in r.stats


# ---------------------------------------------------------------------------
# Preset round-trip reproduces identical trades
# ---------------------------------------------------------------------------
def test_preset_roundtrip_reproduces(df, tmp_path):
    p = Preset(name="rt", scenario="support_break_short@v1.0", stop="atr@v1", tp="trailing@v1",
               params={"grad_max": 0.0, "stop_atr_mult": 1.5, "trail_pct": 1.0},
               cfg={"spread": 0.0, "initial_cash": 10_000})
    path = p.to_json(tmp_path / "preset.json")
    p2 = Preset.from_json(path)
    assert p2.scenario == p.scenario and p2.stop == p.stop and p2.tp == p.tp

    r1 = sc.run(p, df=df)
    r2 = sc.run(p2, df=df)
    assert len(r1.trades) == len(r2.trades)
    if len(r1.trades):
        pd.testing.assert_series_equal(r1.trades["entry_price"], r2.trades["entry_price"])
        pd.testing.assert_series_equal(r1.trades["exit_price"], r2.trades["exit_price"])
    assert r1.stats["total_return_pct"] == r2.stats["total_return_pct"]


# ---------------------------------------------------------------------------
# Level B — the broken level is exposed, correct, and drives the break rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["support_break_short@v1.0", "support_break_short@v1.1",
                                      "support_break_short@v1.2", "support_break_short@v1.3"])
def test_break_level_exposed_and_rule_holds(df, scenario):
    r = sc.run_pieces(df, scenario, "leg_high@v1", "rr@v1",
                      params={"grad_max": 0.0}, cfg={"spread": 0.0})
    assert "break_level" in r.df.columns and "break_level_raw" in r.df.columns
    for i in r.trades["entry_i"]:
        i = int(i)
        # the default rule really is "the candle CLOSED below level B"
        assert r.df["close"].iloc[i] < r.df["break_level_raw"].iloc[i]
        # and the latched level (what the chart draws) is defined for every trade
        assert np.isfinite(r.df["break_level"].iloc[i])


@pytest.mark.parametrize("scenario", ["support_break_short@v1.0", "support_break_short@v1.2"])
def test_break_mode_full_below(df, scenario):
    """'full_below' means the WHOLE candle formed below the latched level B."""
    r = sc.run_pieces(df, scenario, "leg_high@v1", "rr@v1",
                      params={"grad_max": 0.0, "break_mode": "full_below"}, cfg={"spread": 0.0})
    for i in r.trades["entry_i"]:
        i = int(i)
        assert r.df["high"].iloc[i] < r.df["break_level"].iloc[i]


def test_break_level_latched_not_live_zone(df):
    """Level B must NOT follow the S/R support down after the break — it names the level that broke."""
    r = sc.run_pieces(df, "support_break_short@v1.0", "leg_high@v1", "rr@v1",
                      params={"grad_max": 0.0}, cfg={"spread": 0.0})
    d = r.df
    lat, raw = d["break_level"].to_numpy(), d["break_level_raw"].to_numpy()
    both = np.isfinite(lat) & np.isfinite(raw)
    # they coincide at the break bar but must diverge somewhere afterwards (the zone steps down)
    assert both.any()
    assert not np.allclose(lat[both], raw[both])


def test_ema_distance_filter_only_removes_entries(df):
    base = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                         params={"grad_max": 0.0}, cfg={"spread": 0.0})
    filt = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                         params={"grad_max": 0.0, "break_max_ema_dist_pct": 0.05},
                         cfg={"spread": 0.0})
    assert len(filt.trades) <= len(base.trades)


# ---------------------------------------------------------------------------
# Stops: the lag property that motivated leg_high
# ---------------------------------------------------------------------------
def test_leg_high_has_no_confirmation_lag(df):
    """leg_high tracks the current leg's high; a pivot stop cannot be fresher than its look-around,
    so leg_high must never sit ABOVE the pivot-based swing high at an entry."""
    lg = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                       params={"grad_max": 0.0}, cfg={"spread": 0.0})
    sw = sc.run_pieces(df, "support_break_short@v1.2", "swing_high@v1", "rr@v1",
                       params={"grad_max": 0.0}, cfg={"spread": 0.0})
    common = set(lg.trades["entry_i"]) & set(sw.trades["entry_i"])
    assert common, "expected shared entries (the stop must not change the entry logic)"
    lg_i = lg.trades.set_index("entry_i")["stop_price"]
    sw_i = sw.trades.set_index("entry_i")["stop_price"]
    for i in common:
        assert lg_i.loc[i] <= sw_i.loc[i] + 1e-9


def test_stops_are_above_entry_for_shorts(df):
    for stop in ("leg_high@v1", "prev_high@v1", "swing_high@v1", "high_of_n@v1", "zone_top@v1"):
        r = sc.run_pieces(df, "support_break_short@v1.2", stop, "rr@v1",
                          params={"grad_max": 0.0}, cfg={"spread": 0.0})
        if len(r.trades):
            assert bool((r.trades["stop_price"] >= r.trades["entry_price"]).all()), stop


# ---------------------------------------------------------------------------
# Secondary rule (per-trade arm + in-profit gate) — the engine feature
# ---------------------------------------------------------------------------
def test_secondary_absent_is_identical(df):
    """Not declaring a secondary must leave results untouched (byte-identical engine path)."""
    a = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                      params={"grad_max": 0.0}, cfg={"spread": 0.0})
    b = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1", secondary=None,
                      params={"grad_max": 0.0}, cfg={"spread": 0.0})
    pd.testing.assert_frame_equal(a.trades, b.trades)


def _alt_case(closes, arm_bar, alt_bar, *, entry_bar=1, replace_tp=False, tp_mode="none",
              tp_value=0.0):
    """Hand-built one-short-trade engine case for the secondary-exit semantics.

    Deterministic: high == low == close, zero costs, so 'in profit' is purely the price.
    """
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
                      "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    z = np.zeros(n, bool)
    es = z.copy(); es[entry_bar] = True
    arm = z.copy()
    if arm_bar is not None:
        arm[arm_bar] = True
    alt = z.copy()
    if alt_bar is not None:
        alt[alt_bar] = True
    sig = Signals(entry_long=z.copy(), entry_short=es, exit_short=z.copy(),
                  exit_arm_short=arm, exit_alt_short=alt, alt_replace_tp=replace_tp)
    cfg = BacktestConfig(initial_cash=10_000, exit_enabled=True, allow_short=True,
                         sl_mode="entry_pct", sl_value=50.0,       # stop far away, never hit
                         tp_mode=tp_mode, tp_value=tp_value,
                         sizing_mode="cash", cash_per_trade=1_000)
    return run_backtest(d, sig, cfg)


def test_secondary_closes_when_in_profit():
    # short at 100, price falls -> alt at bar 3 (98 < 100) must close in profit
    res = _alt_case([100, 100, 99, 98, 97, 97], arm_bar=2, alt_bar=3)
    t = res.trades.iloc[0]
    assert t["close_reason"] == "alt_exit"
    assert t["exit_i"] == 3 and t["exit_price"] < t["entry_price"] and t["pnl"] > 0


def test_secondary_does_not_close_at_a_loss():
    # same wiring but price RISES -> the alt signal must be ignored (gate is 'only if in profit')
    res = _alt_case([100, 100, 101, 102, 103, 103], arm_bar=2, alt_bar=3)
    t = res.trades.iloc[0]
    assert t["close_reason"] == "forced_close_end"      # rode to the end, never alt-closed


def test_secondary_needs_arming_first():
    # profitable and the alt signal fires, but nothing ever arms -> no secondary close
    res = _alt_case([100, 100, 99, 98, 97, 97], arm_bar=None, alt_bar=3)
    assert res.trades.iloc[0]["close_reason"] == "forced_close_end"


def test_secondary_arm_before_entry_does_not_carry():
    # arm fires only at bar 0, BEFORE the trade opens at bar 1 -> that trade is never armed
    res = _alt_case([100, 100, 99, 98, 97, 97], arm_bar=0, alt_bar=3)
    assert res.trades.iloc[0]["close_reason"] == "forced_close_end"


def test_secondary_replace_tp_drops_the_target():
    """TP at 95 is reached at bar 4. Keeping it -> take_profit; replacing it -> the TP is ignored and
    the armed secondary closes instead. The stop is untouched either way."""
    closes = [100, 100, 99, 98, 94, 94, 94]
    keep = _alt_case(closes, arm_bar=2, alt_bar=5, replace_tp=False,
                     tp_mode="entry_pct", tp_value=5.0)
    repl = _alt_case(closes, arm_bar=2, alt_bar=5, replace_tp=True,
                     tp_mode="entry_pct", tp_value=5.0)
    assert keep.trades.iloc[0]["close_reason"] == "take_profit"
    assert keep.trades.iloc[0]["exit_i"] == 4
    assert repl.trades.iloc[0]["close_reason"] == "alt_exit"
    assert repl.trades.iloc[0]["exit_i"] == 5


def test_secondary_scenario_level_exits_are_profitable(df):
    """Whatever the scenario produces, every alt_exit it yields must be a profitable close."""
    r = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                      secondary="shaky_trend_bail@v1",
                      params={"grad_max": 0.0}, cfg={"spread": 0.0})
    alt = r.trades[r.trades["close_reason"] == "alt_exit"]
    if len(alt):
        assert bool((alt["exit_price"] < alt["entry_price"]).all())
        assert bool((alt["pnl"] > 0).all())
        assert bool((alt["exit_i"] > alt["entry_i"]).all())


def test_preset_roundtrip_with_secondary(df, tmp_path):
    p = Preset(name="sec", scenario="support_break_short@v1.2", stop="leg_high@v1", tp="rr@v1",
               secondary="shaky_trend_bail@v1",
               params={"grad_max": 0.0, "sec_opp_n": 1}, cfg={"spread": 0.0})
    p2 = Preset.from_json(p.to_json(tmp_path / "sec.json"))
    assert p2.secondary == "shaky_trend_bail@v1"
    r1, r2 = sc.run(p, df=df), sc.run(p2, df=df)
    pd.testing.assert_frame_equal(r1.trades, r2.trades)


# ---------------------------------------------------------------------------
# Heikin-Ashi candle source
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["support_break_short@v1.0", "support_break_short@v1.2"])
def test_heikin_ashi_candle_source(df, scenario):
    r = sc.run_pieces(df, scenario, "leg_high@v1", "opposite_candle@v1",
                      params={"grad_max": 0.0, "candle_src": "ha"}, cfg={"spread": 0.0})
    assert "ha_close" in r.df.columns
    assert "total_return_pct" in r.stats


def test_shipped_presets_load_and_run(df):
    import glob
    import os
    files = glob.glob(os.path.join("scenarios", "presets", "*.json"))
    assert files, "no shipped presets found"
    for f in files:
        p = Preset.from_json(f)
        assert p.scenario in sc.SCENARIOS and p.stop in sc.STOP_METHODS and p.tp in sc.TP_METHODS
        r = sc.run(p, df=df)                       # run on the synthetic frame (no network)
        assert "total_return_pct" in r.stats


# ---------------------------------------------------------------------------
# multiple secondaries combine (they don't conflict); new arm modes; run_past_tp
# ---------------------------------------------------------------------------
def test_secondaries_combine(df):
    """A list of secondaries must run together: each adds another way to ARM, and their exits OR."""
    one = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                        secondary="shaky_trend_bail@v1",
                        params={"grad_max": 0.0}, cfg={"spread": 0.0})
    both = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                         secondary=["shaky_trend_bail@v1", "run_past_tp@v1"],
                         params={"grad_max": 0.0}, cfg={"spread": 0.0})
    assert len(both.secondary) == 2
    assert {s.key for s in both.secondary} == {"shaky_trend_bail@v1", "run_past_tp@v1"}
    # a single key still resolves to a one-element list (back-compat)
    assert len(one.secondary) == 1
    # run_past_tp suppresses plain take-profit closes (the TP now arms instead of closing)
    assert int((both.trades["close_reason"] == "take_profit").sum()) <= \
        int((one.trades["close_reason"] == "take_profit").sum())


@pytest.mark.parametrize("mode", ["near", "touch", "close_above", "full_above"])
def test_sec_arm_modes_run(df, mode):
    r = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                      secondary="shaky_trend_bail@v1",
                      params={"grad_max": 0.0, "sec_arm_mode": mode, "sec_arm_tol_pct": 0.05},
                      cfg={"spread": 0.0})
    assert "total_return_pct" in r.stats


def test_sec_arm_mode_strictness_ordering(df):
    """Looser arming can only arm MORE trades, so alt_exit count must be monotone in strictness."""
    def n_alt(mode):
        r = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "none@v1",
                          secondary="shaky_trend_bail@v1",
                          params={"grad_max": 0.0, "sec_arm_mode": mode, "sec_arm_tol_pct": 0.5},
                          cfg={"spread": 0.0})
        return int((r.trades["close_reason"] == "alt_exit").sum())
    assert n_alt("near") >= n_alt("touch") >= n_alt("full_above")


def test_bad_arm_mode_raises(df):
    with pytest.raises(ValueError):
        sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                      secondary="shaky_trend_bail@v1",
                      params={"grad_max": 0.0, "sec_arm_mode": "nonsense"}, cfg={"spread": 0.0})


def test_run_past_tp_holds_through_target(df):
    """With run_past_tp no trade may close on 'take_profit' — the TP arms it instead."""
    r = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                      secondary="run_past_tp@v1",
                      params={"grad_max": 0.0, "tp_rr": 1.0}, cfg={"spread": 0.0})
    assert int((r.trades["close_reason"] == "take_profit").sum()) == 0
    base = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                         params={"grad_max": 0.0, "tp_rr": 1.0}, cfg={"spread": 0.0})
    assert int((base.trades["close_reason"] == "take_profit").sum()) > 0


def test_trend_structure_and_rsi_divergence(df):
    from quant.indicators import add_trend_structure, rsi_divergence_table
    s = add_trend_structure(df, 10, 10)
    assert set(np.unique(s["struct_trend"])) <= {-1, 0, 1}
    labs = {v for v in s["struct_label"].dropna().unique()}
    assert labs <= {"HH", "HL", "LH", "LL"} and labs
    tab = rsi_divergence_table(df, 14, 5, 5)
    for _, z in tab.iterrows():
        assert z.i1 > z.i0
        if z.kind == "bear":
            assert z.px1 > z.px0 and z.rsi1 < z.rsi0      # higher high, lower RSI
        else:
            assert z.px1 < z.px0 and z.rsi1 > z.rsi0      # lower low, higher RSI


# ---------------------------------------------------------------------------
# extra entry rules: MTF conditions + the (A & B) | C algebra
# ---------------------------------------------------------------------------
def test_mtf_ema_columns_added_by_param(df):
    r = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                      params={"grad_max": 0.0, "mtf_emas": ("5min", [100, 200])},
                      cfg={"spread": 0.0})
    assert "5min__ema_100" in r.df.columns and "5min__ema_200" in r.df.columns


def test_entry_filter_only_removes_entries(df):
    """An extra rule is ANDed onto the entry, so it can only ever REMOVE trades."""
    from quant.strategies.conditions import below_mtf
    P = {"grad_max": 0.0, "mtf_emas": ("5min", 200)}
    base = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                         params=P, cfg={"spread": 0.0})
    filt = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                         params=P, cfg={"spread": 0.0},
                         entry_filter=below_mtf("5min", 200, n=3))
    assert len(filt.trades) <= len(base.trades)
    for i in filt.trades["entry_i"]:
        i = int(i)
        assert filt.df["close"].iloc[i] < filt.df["5min__ema_200"].iloc[i]


def test_below_mtf_modes_and_n(df):
    """`mode='full'` is stricter than 'close'; a bigger `n` is stricter than a smaller one."""
    from quant.strategies.conditions import below_mtf
    P = {"grad_max": 0.0, "mtf_emas": ("5min", 200)}
    prepared = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                             params=P, cfg={"spread": 0.0}).df
    c1 = below_mtf("5min", 200, n=1)(prepared)
    c3 = below_mtf("5min", 200, n=3)(prepared)
    full1 = below_mtf("5min", 200, n=1, mode="full")(prepared)
    assert (c3 & ~c1).sum() == 0        # n=3 implies n=1
    assert (full1 & ~c1).sum() == 0     # whole candle below implies the close is below


def test_rule_algebra_and_or(df):
    """(A & B) | C must equal the plain boolean combination of the three arrays."""
    from quant.strategies.conditions import below_mtf, red
    P = {"grad_max": 0.0, "mtf_emas": ("5min", [100, 200])}
    prepared = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                             params=P, cfg={"spread": 0.0}).df
    A = below_mtf("5min", 200, n=3)
    B = red(2)
    C = below_mtf("5min", 100, n=5)
    combo = (A & B) | C
    expected = (A(prepared) & B(prepared)) | C(prepared)
    assert np.array_equal(combo(prepared), expected)
    assert "&" in combo.name and "|" in combo.name        # readable description
