"""Reference levels, the Phase-2 entry families, and the engine time stop.

The level tests are mostly CAUSALITY tests. Every one of these columns is trivial to compute with a
peek at the future (a session's high is not known until the session ends), and a lookahead here would
silently invalidate every screen and backtest built on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import scenarios as sc
from quant.engine import BacktestConfig, Signals, run_backtest
from quant.indicators import (add_prev_day_levels, add_round_levels, add_session_levels,
                              add_volume_features, add_vwap, add_week_levels,
                              add_zigzag_structure, level_events)


def _bars(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    high = close + np.abs(rng.normal(0.08, 0.04, n))
    low = close - np.abs(rng.normal(0.08, 0.04, n))
    open_ = np.r_[close[0], close[:-1]]
    t = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.lognormal(0, 0.5, n)})


@pytest.fixture(scope="module")
def df():
    return _bars()


# ---------------------------------------------------------------------------
# levels — causality
# ---------------------------------------------------------------------------
def test_prev_day_levels_are_the_previous_day(df):
    out = add_prev_day_levels(df)
    day = out["t"].dt.floor("D")
    days = sorted(day.unique())
    assert out.loc[day == days[0], "pd_high"].isna().all()          # nothing before the first day
    for prev, cur in zip(days[:-1], days[1:]):
        want_hi = df.loc[day == prev, "high"].max()
        want_lo = df.loc[day == prev, "low"].min()
        got = out.loc[day == cur]
        assert np.allclose(got["pd_high"], want_hi)
        assert np.allclose(got["pd_low"], want_lo)
        assert np.allclose(got["pd_mid"], (want_hi + want_lo) / 2)


def test_session_running_extremes_are_expanding_not_final(df):
    out = add_session_levels(df, or_bars=30)
    g = out.groupby("sess_id", sort=False)
    # non-decreasing within a session, and never above the bar's own high seen so far
    assert (g["sess_run_high"].diff().fillna(0) >= -1e-12).all()
    assert (g["sess_run_low"].diff().fillna(0) <= 1e-12).all()
    assert (out["sess_run_high"] >= out["high"] - 1e-12).all()
    assert (out["sess_run_low"] <= out["low"] + 1e-12).all()
    # a running extreme must be STRICTLY less informative than the session's final one somewhere
    final_hi = g["high"].transform("max")
    assert (out["sess_run_high"] < final_hi - 1e-9).any()


def test_opening_range_hidden_until_it_completes(df):
    out = add_session_levels(df, or_bars=30)
    early = out["sess_bar"] < 30
    assert out.loc[early, "sess_or_high"].isna().all()
    assert out.loc[early, "sess_or_low"].isna().all()
    late = out[out["sess_bar"] >= 30]
    assert late["sess_or_high"].notna().any()
    # the OR must equal the extremes of exactly the first 30 bars of that session
    sid = late["sess_id"].iloc[0]
    first30 = out[(out.sess_id == sid) & (out.sess_bar < 30)]
    row = late[late.sess_id == sid].iloc[0]
    assert row["sess_or_high"] == pytest.approx(first30["high"].max())
    assert row["sess_or_low"] == pytest.approx(first30["low"].min())


def test_prev_session_levels_come_from_the_previous_session(df):
    out = add_session_levels(df)
    done = out.groupby("sess_id", sort=False).agg(h=("high", "max"), l=("low", "min"))
    ids = list(dict.fromkeys(out["sess_id"]))
    assert out.loc[out.sess_id == ids[0], "sess_prev_high"].isna().all()
    for prev, cur in zip(ids[:-1], ids[1:]):
        got = out.loc[out.sess_id == cur]
        assert np.allclose(got["sess_prev_high"], done.loc[prev, "h"])
        assert np.allclose(got["sess_prev_low"], done.loc[prev, "l"])


def test_vwap_stays_inside_the_session_range_and_is_expanding(df):
    out = add_vwap(df, anchor="session")
    g = out.groupby(out["sess_id"] if "sess_id" in out else
                    add_session_levels(df)["sess_id"], sort=False)
    lo = g["low"].transform("min")
    hi = g["high"].transform("max")
    assert (out["vwap"] >= lo - 1e-9).all() and (out["vwap"] <= hi + 1e-9).all()
    assert (out["vwap_sd"] >= -1e-12).all()
    assert (out["vwap_u1"] >= out["vwap"]).all() and (out["vwap_l1"] <= out["vwap"]).all()
    # first bar of a session: vwap == that bar's typical price, sd == 0 (no future leakage)
    first = out.groupby(add_session_levels(df)["sess_id"], sort=False).head(1)
    tp = (first["high"] + first["low"] + first["close"]) / 3
    assert np.allclose(first["vwap"], tp)


def test_week_levels_shift_by_one_week(df):
    out = add_week_levels(df)
    wk = out["t"].dt.to_period("W").astype(str)
    weeks = list(dict.fromkeys(wk))
    assert out.loc[wk == weeks[0], "wk_high"].isna().all()
    if len(weeks) > 1:
        want = df.loc[wk == weeks[0], "high"].max()
        assert np.allclose(out.loc[wk == weeks[1], "wk_high"], want)


def test_volume_features_use_only_past_bars(df):
    out = add_volume_features(df, lookback=50)
    # z-score of a constant-volume prefix is undefined -> NaN, never a peeked value
    assert out["vol_z"].iloc[:24].isna().all()
    v = df["volume"].to_numpy()
    i = 500
    m, s = v[i - 49:i + 1].mean(), v[i - 49:i + 1].std(ddof=0)
    assert out["vol_z"].iloc[i] == pytest.approx((v[i] - m) / s)


def test_round_levels_bracket_the_close(df):
    out = add_round_levels(df, 5.0)
    assert (out["rnd_below"] <= out["close"] + 1e-12).all()
    assert (out["rnd_above"] >= out["close"] - 1e-12).all()
    assert (out["rnd_above"] - out["rnd_below"] <= 5.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# level_events — the classification itself
# ---------------------------------------------------------------------------
def test_level_events_classification_is_exact():
    d = pd.DataFrame({
        "high":  [10.0, 12.0, 11.0, 12.5, 10.5],
        "low":   [ 9.0,  9.5, 10.0, 11.0,  9.5],
        "close": [ 9.5, 10.5, 10.2, 12.2, 10.0],
        "lv":    [11.0, 11.0, 11.0, 11.0, 11.0],
    })
    ev = level_events(d, "lv", tol_pct=0.0, side="above", sweep_bars=1)
    # bar 1: high 12 > 11 but close 10.5 < 11  -> sweep + fade, no break
    assert ev["sweep"][1] and ev["fade"][1] and not ev["break"][1]
    # bar 3: close 12.2 > 11 -> break, and the FIRST one
    assert ev["break"][3] and ev["first_break"][3]
    # bar 2 never reaches the level (high 11.0 >= 11.0 exactly -> reached, not pierced)
    assert not ev["sweep"][2]
    # a break is never also a fade
    assert not (ev["break"] & ev["fade"]).any()


def test_level_events_below_side_mirrors_above():
    d = pd.DataFrame({"high": [10.0, 10.0], "low": [9.0, 8.0], "close": [9.5, 8.5],
                      "lv": [9.0, 9.0]})
    ev = level_events(d, "lv", tol_pct=0.0, side="below", sweep_bars=1)
    assert ev["break"][1] and ev["first_break"][1]
    assert not ev["break"][0]


def test_level_events_ignores_nan_levels():
    d = pd.DataFrame({"high": [10.0, 10.0], "low": [9.0, 9.0], "close": [9.5, 9.5],
                      "lv": [np.nan, 9.0]})
    ev = level_events(d, "lv", side="below")
    assert not ev["break"][0] and not ev["fade"][0] and not ev["sweep"][0]


# ---------------------------------------------------------------------------
# zigzag: running last confirmed pivots
# ---------------------------------------------------------------------------
def test_zigzag_last_pivots_forward_fill_confirmed_prices(df):
    out = add_zigzag_structure(df, atr_mult=2.0)
    lab, px = out["zz_label"], out["zz_price"]
    hi_idx = np.where(lab.isin(("HH", "LH")).to_numpy())[0]
    assert len(hi_idx) > 0
    i = hi_idx[0]
    assert out["zz_last_high"].iloc[:i].isna().all()          # nothing before the first confirmation
    assert out["zz_last_high"].iloc[i] == pytest.approx(px.iloc[i])
    if len(hi_idx) > 1:
        j = hi_idx[1]
        # holds the previous pivot right up to the next confirmation
        assert out["zz_last_high"].iloc[j - 1] == pytest.approx(px.iloc[i])


# ---------------------------------------------------------------------------
# entry families
# ---------------------------------------------------------------------------
ENTRY_KEYS = [k for k in sc.SCENARIOS
              if any(k.startswith(f) for f in
                     ("level_event_", "bos_", "vol_thrust_", "squeeze_", "wt_cross_",
                      "rsi_div_", "vwap_rev_", "grad_extreme_", "time_entry_"))]


def test_every_entry_family_is_registered_long_and_short():
    for fam in ("level_event", "bos", "vol_thrust", "squeeze", "wt_cross", "rsi_div",
                "vwap_rev", "grad_extreme", "time_entry"):
        assert f"{fam}_long@v1" in sc.SCENARIOS
        assert f"{fam}_short@v1" in sc.SCENARIOS
        assert sc.SCENARIOS[f"{fam}_long@v1"].side == "long"
        assert sc.SCENARIOS[f"{fam}_short@v1"].side == "short"


@pytest.mark.parametrize("key", sorted(ENTRY_KEYS))
def test_entry_family_runs_and_stops_sit_on_the_right_side(df, key):
    run = sc.run_pieces(df, key, "atr@v1", "rr@v1",
                        params={"stop_atr_mult": 1.5, "tp_rr": 2.0},
                        cfg={"initial_cash": 10_000, "spread": 0.02})
    t = run.trades
    assert len(t) > 0, f"{key} produced no trades"
    side = sc.SCENARIOS[key].side
    stop, entry = t["stop_price"], t["entry_price"]
    ok = stop.notna()
    if side == "short":
        assert (stop[ok] > entry[ok]).all()
    else:
        assert (stop[ok] < entry[ok]).all()


def test_level_event_rejects_an_unknown_level_column(df):
    with pytest.raises(KeyError):
        sc.run_pieces(df, "level_event_short@v1", "atr@v1", "rr@v1",
                      params={"lv_col": "not_a_level"})


def test_time_entry_null_fires_on_its_schedule(df):
    run = sc.run_pieces(df, "time_entry_long@v1", "atr@v1", "rr@v1",
                        params={"te_every": 250, "stop_atr_mult": 1.5, "tp_rr": 2.0},
                        cfg={"initial_cash": 10_000})
    assert len(run.trades) > 0
    assert (run.trades["entry_i"] % 250 == 0).all()


# ---------------------------------------------------------------------------
# time stop
# ---------------------------------------------------------------------------
def _sig(n, every=100):
    e = np.zeros(n, bool)
    e[::every] = True
    return Signals(entry_long=e, exit_long=np.zeros(n, bool),
                   entry_short=np.zeros(n, bool), exit_short=np.zeros(n, bool))


def test_time_stop_caps_holding_period(df):
    n = len(df)
    cfg = BacktestConfig(initial_cash=10_000, exit_enabled=True, max_bars_in_trade=20)
    res = run_backtest(df, _sig(n), cfg)
    t = res.trades
    assert len(t) > 0
    assert (t["bars_held"] <= 20).all()
    assert (t["close_reason"] == "time_stop").any()


def test_time_stop_off_is_byte_identical(df):
    n = len(df)
    base = BacktestConfig(initial_cash=10_000, exit_enabled=True, sl_mode="entry_pct", sl_value=0.5,
                          tp_mode="rr", tp_value=2.0)
    a = run_backtest(df, _sig(n), base)
    b = run_backtest(df, _sig(n), BacktestConfig(**{**base.__dict__, "max_bars_in_trade": 0}))
    pd.testing.assert_frame_equal(a.trades, b.trades)
    np.testing.assert_array_equal(a.equity_curve["equity"].to_numpy(),
                                  b.equity_curve["equity"].to_numpy())


def test_time_stop_does_not_override_a_real_close_reason(df):
    """A stop that would fire on the same bar keeps its own reason — the time stop is checked last."""
    n = len(df)
    cfg = BacktestConfig(initial_cash=10_000, exit_enabled=True, sl_mode="entry_pct", sl_value=0.05,
                         max_bars_in_trade=30)
    res = run_backtest(df, _sig(n), cfg)
    t = res.trades
    assert (t["close_reason"] == "stop_loss").any()
    assert (t.loc[t["close_reason"] == "time_stop", "bars_held"] == 30).all()
