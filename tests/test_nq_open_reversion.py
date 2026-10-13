"""
experiments/nq_open_reversion.py -- the rules of JJ Simon's New York open reversion, checked on
candles built by hand, so the experiment's verdict rests on definitions that are known to be right.
"""
import numpy as np
import pandas as pd
import pytest

from experiments.nq_open_reversion import (OPEN_MIN, entry_signals, gates, run_arm, sessions,
                                           TARGET_PTS, ROOM)


def frame(rows, start="2026-07-14 13:30"):
    t = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "open_time", t)
    return df


@pytest.mark.parametrize("day, utc", [("2026-07-14", "13:30"), ("2026-01-13", "14:30")])
def test_fair_price_is_the_930_eastern_open_in_summer_and_winter(day, utc):
    t = pd.date_range(f"{day} 12:00", f"{day} 21:00", freq="1min", tz="UTC", inclusive="left")
    df = pd.DataFrame({"open_time": t, "open": np.arange(len(t), dtype=float)})
    df["high"], df["low"], df["close"] = df.open + 1, df.open - 1, df.open
    s = sessions(df)
    assert t[int(s.fair.iloc[0])].strftime("%H:%M") == utc
    assert s.et.iloc[0].strftime("%H:%M") == "09:00" and s.et.iloc[-1].strftime("%H:%M") == "15:59"
    assert s.loc[s.window, "minute"].min() == OPEN_MIN and s.loc[s.window, "minute"].max() == 659


CANDLES = [  # open, high, low, close
    (100, 101, 99, 100.5),
    (100.5, 102, 100, 101.5),        # green
    (101.5, 101.8, 99.0, 99.2),      # red, bigger body, closes below the green one's low: DISPLACE
    (99.2, 100, 98.0, 99.5),         # a swing low at 98 ...
    (99.5, 100.5, 99.1, 100.2),      # ... confirmed by this higher low
    (100.2, 100.4, 98.5, 98.6),      # displaces the green candle, but does not break 98
    (98.6, 98.7, 97.0, 97.5),        # first close below 98: BREAK OF STRUCTURE
    (97.5, 97.6, 96.5, 96.8),        # still below: not a second break
]


def test_his_two_entries_fire_exactly_where_his_definitions_say():
    sig = entry_signals(sessions(frame(CANDLES)))
    assert np.flatnonzero(sig["disp_s"]).tolist() == [2, 5]
    assert np.flatnonzero(sig["bos_s"]).tolist() == [6]
    assert not sig["disp_l"].any() and not sig["bos_l"].any()


def test_a_displacement_needs_the_opposite_colour_before_it():
    rows = [list(r) for r in CANDLES]
    rows[1] = [101.5, 102, 100, 100.5]   # candle 1 now red: candle 2 displaces nothing
    sig = entry_signals(sessions(frame(rows)))
    assert 2 not in np.flatnonzero(sig["disp_s"]).tolist()


def test_no_entry_signal_ever_uses_a_later_candle():
    rng = np.random.default_rng(4)
    n = 400
    c = 20_000 + rng.normal(0, 4, n).cumsum()
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 1, n)
    rows = list(zip(o, np.maximum(o, c) + rng.exponential(2, n),
                    np.minimum(o, c) - rng.exponential(2, n), c))
    full = entry_signals(sessions(frame(rows)))
    # EVERY cut, not a few: a swing counted one candle early differs only when a break lands on
    # exactly that candle -- four fixed cuts let it through
    for k in range(3, n + 1):
        cut = entry_signals(sessions(frame(rows[:k])))
        for key in full:
            assert (cut[key] == full[key][:k]).all(), (key, k)


def test_the_side_is_toward_fair_price_and_needs_room():
    rows = [(100, 101, 99, 100)] + [(100, 140, 99, 100 + x) for x in (10, 30, 31)]
    s = sessions(frame(rows))
    room = ROOM * TARGET_PTS * s["scale"]
    shorts, longs = gates(s, toward=True)
    assert shorts.tolist() == (s["close"] - s["fair"] >= room).tolist() and not longs.any()
    away_s, away_l = gates(s, toward=False)
    assert (away_l == shorts).all() and not away_s.any()


def test_three_losses_in_a_row_ends_the_session(monkeypatch):
    from experiments import nq_open_reversion as nq
    days = ["2026-07-14"] * 5 + ["2026-07-15"] * 2
    fake = pd.DataFrame(dict(side=["short"] * 7, entry_i=range(7), exit_price=[101, 101, 101, 99,
                                                                             101, 101, 99],
                             entry_price=[100] * 7))

    class R:
        trades = fake

    monkeypatch.setattr("quant.engine.run_backtest", lambda *a, **k: R())
    s = pd.DataFrame(dict(day=pd.to_datetime(days).date, scale=1.0, close=100.0, fair=100.0,
                          last=False, window=True, minute=OPEN_MIN))
    sig = {k: np.zeros(7, bool) for k in ("disp_s", "disp_l", "bos_s", "bos_l")}
    kept = run_arm(s, "his_rules", sig)
    # day 1: loss, loss, loss -> the win on trade 4 and trade 5 never happen; day 2 starts fresh
    assert kept["entry_i"].tolist() == [0, 1, 2, 5, 6]
    every = run_arm(s, "no_3loss", sig)
    assert len(every) == 7
