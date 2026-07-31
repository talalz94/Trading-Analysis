"""Market-structure + divergence indicators: causality, labelling, and the lag contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.indicators import (add_rsi_divergence_realtime, add_trend_structure,
                              add_zigzag_structure, rsi_divergence_table)


def _wave(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """Trending price with clear swings + noise, so both methods find real structure."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = (100 + 0.02 * t + 6 * np.sin(t / 40.0) + 2 * np.sin(t / 7.0)
             + rng.normal(0, 0.35, n)).astype(float)
    high = close + np.abs(rng.normal(0.35, 0.15, n))
    low = close - np.abs(rng.normal(0.35, 0.15, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    # `open_time` is part of the OHLCV contract (resample_ohlcv keys off it); `t` is the display copy
    return pd.DataFrame({"open_time": idx, "t": idx, "open": np.r_[close[0], close[:-1]],
                         "high": high, "low": low, "close": close, "volume": 1.0})


@pytest.fixture(scope="module")
def df():
    return _wave()


# ---------------------------------------------------------------------------
# ZigZag (adaptive) structure
# ---------------------------------------------------------------------------
def test_zigzag_is_causal(df):
    """A pivot may only be emitted at/after the bar it happened — never before."""
    z = add_zigzag_structure(df, atr_mult=1.5)
    m = z["zz_label"].notna().to_numpy()
    assert m.any(), "expected some pivots"
    bars = np.flatnonzero(m)
    piv = z["zz_pivot_i"].to_numpy()[m]
    lag = z["zz_lag"].to_numpy()[m]
    assert (piv <= bars).all()          # the extreme is in the past
    assert (lag >= 0).all()
    assert np.allclose(lag, bars - piv)


def test_zigzag_labels_and_trend(df):
    z = add_zigzag_structure(df, atr_mult=1.5)
    labs = {v for v in z["zz_label"].dropna().unique()}
    assert labs <= {"HH", "HL", "LH", "LL"} and labs
    assert set(np.unique(z["zz_trend"])) <= {-1, 0, 1}


def test_zigzag_alternates_high_and_low(df):
    """Swings must alternate: a high is always followed by a low and vice versa."""
    z = add_zigzag_structure(df, atr_mult=1.5)
    kinds = ["H" if v in ("HH", "LH") else "L"
             for v in z["zz_label"].dropna().tolist()]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "consecutive pivots of the same kind"


def test_zigzag_threshold_monotonicity(df):
    """A bigger ATR threshold -> fewer, more significant swings, and a longer wait to confirm."""
    prev_n, prev_lag = None, None
    for m in (1.0, 2.0, 4.0):
        z = add_zigzag_structure(df, atr_mult=m)
        n = int(z["zz_label"].notna().sum())
        lag = float(z["zz_lag"].dropna().mean())
        if prev_n is not None:
            assert n <= prev_n
            assert lag >= prev_lag
        prev_n, prev_lag = n, lag


def test_zigzag_beats_fractal_on_lag(df):
    """The point of the adaptive method: comparable pivot count at materially less lag."""
    frac_lr = 10
    f = add_trend_structure(df, frac_lr, frac_lr)
    n_f = int(f["struct_label"].notna().sum())
    z = add_zigzag_structure(df, atr_mult=2.5)
    assert float(z["zz_lag"].dropna().mean()) < frac_lr
    assert int(z["zz_label"].notna().sum()) > 0 and n_f > 0


# ---------------------------------------------------------------------------
# RSI divergence — the lag contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("right", [1, 2, 5])
def test_confirmed_divergence_lag_equals_right(df, right):
    """Lag is exactly `right` bars — the whole point of allowing an asymmetric left/right."""
    tab = rsi_divergence_table(df, 14, left=10, right=right)
    assert len(tab)
    assert (tab["confirm_i"] - tab["i1"] == right).all()


def test_confirmed_divergence_direction(df):
    tab = rsi_divergence_table(df, 14, left=10, right=2)
    for _, z in tab.iterrows():
        assert z.i1 > z.i0
        if z.kind == "bear":
            assert z.px1 > z.px0 and z.rsi1 < z.rsi0     # higher high, lower RSI
        else:
            assert z.px1 < z.px0 and z.rsi1 > z.rsi0     # lower low, higher RSI


def test_left_is_free_smaller_right_never_delays(df):
    """Increasing `left` must not change the lag; only `right` does."""
    a = rsi_divergence_table(df, 14, left=5, right=2)
    b = rsi_divergence_table(df, 14, left=20, right=2)
    assert (a["confirm_i"] - a["i1"] == 2).all()
    assert (b["confirm_i"] - b["i1"] == 2).all()


def test_realtime_divergence_is_causal(df):
    """A zero-lag signal at bar i must be reproducible from data up to i only."""
    rt = add_rsi_divergence_realtime(df, 14, left=10)
    flags = (rt["rsi_rtdiv_bear"] | rt["rsi_rtdiv_bull"]).to_numpy()
    bars = [b for b in np.flatnonzero(flags) if b > 400][:5]
    assert bars, "expected some provisional signals"
    for b in bars:
        trunc = add_rsi_divergence_realtime(df.iloc[:b + 1].copy(), 14, left=10)
        assert bool(trunc["rsi_rtdiv_bear"].iloc[-1] or trunc["rsi_rtdiv_bull"].iloc[-1]), \
            f"signal at bar {b} not reproducible from truncated data -> lookahead"


def test_realtime_fires_more_often_than_confirmed(df):
    """Removing the lag buys earlier signals at the cost of more of them (provisional)."""
    rt = add_rsi_divergence_realtime(df, 14, left=10)
    n_rt = int((rt["rsi_rtdiv_bear"] | rt["rsi_rtdiv_bull"]).sum())
    n_conf = len(rsi_divergence_table(df, 14, left=10, right=2))
    assert n_rt > n_conf


# ---------------------------------------------------------------------------
# higher-timeframe EMA drawn on a lower-timeframe chart
# ---------------------------------------------------------------------------
def test_mtf_ema_updates_only_on_closed_htf_bars(df):
    """A 5m EMA shown on a 1m grid must change once per CLOSED 5m bar — not every minute.
    That is the anti-lookahead contract: no leakage from the unfinished higher-TF candle."""
    from quant.data import build_mtf
    from quant.indicators.overlays import add_emas
    m = build_mtf(df, {"5min": lambda z: add_emas(z, [20])})
    col = "5min__ema_20"
    assert col in m.columns
    v = m[col].dropna()
    steps = int(v.diff().ne(0).sum())
    bars_per_step = len(v) / max(steps, 1)
    assert 4.0 <= bars_per_step <= 6.0, f"expected ~5 1m bars per 5m step, got {bars_per_step:.1f}"


def test_mtf_ema_is_not_ahead_of_the_base_chart(df):
    """The HTF value at bar i must be reproducible from data up to i (no future 5m candle)."""
    from quant.data import build_mtf
    from quant.indicators.overlays import add_emas
    col = "5min__ema_20"
    full = build_mtf(df, {"5min": lambda z: add_emas(z, [20])})[col]
    for i in (600, 900, 1200):
        trunc = build_mtf(df.iloc[:i + 1].copy(), {"5min": lambda z: add_emas(z, [20])})[col]
        a, b = full.iloc[i], trunc.iloc[-1]
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, \
            f"HTF EMA at bar {i} changes when future bars are removed -> lookahead"
