"""
experiments/nq_news_reversion.py -- which days are news days, what "the price before the news" is,
and which bars may trade, all on hand-built rows so the verdict rests on definitions known right.
"""
import numpy as np
import pandas as pd
import pytest

from experiments import nq_news_reversion as nr


def ev(t_ny, title="CPI m/m", level="critical", cur="USD", z=0.5, q="official"):
    t = pd.Timestamp(t_ny).tz_localize("America/New_York").tz_convert("UTC")
    return dict(t=t, title=title, level=level, currency=cur, surprise_z=z, time_q=q)


def test_a_news_day_is_us_high_impact_news_at_830_new_york():
    days = nr.news_days(pd.DataFrame([
        ev("2024-07-11 08:30", z=1.48), ev("2024-07-11 08:30", "Unemployment Claims", "high", z=-0.2),
        ev("2024-07-01 10:00", "ISM Manufacturing PMI", "high"),          # red, but not 8:30
        ev("2024-07-02 08:30", "German Thing", "high", cur="EUR"),        # not US
        ev("2024-07-03 08:30", "Philly Fed", "medium"),                   # not red
        ev("2024-07-05 08:30", "Non-Farm Employment Change", z=2.1),
        ev("2024-01-11 08:30", z=np.nan),                                 # winter time too
    ]))
    d = {str(k): r for k, r in days.iterrows()}
    assert set(d) == {"2024-07-11", "2024-07-01", "2024-07-05", "2024-01-11"}
    assert d["2024-07-11"].red830 and d["2024-07-11"].aplus
    assert np.isclose(d["2024-07-11"].max_z, 1.48)
    assert d["2024-07-01"].any_red and not d["2024-07-01"].red830
    assert d["2024-07-05"].aplus and d["2024-01-11"].red830
    assert np.isnan(d["2024-01-11"].max_z)


@pytest.mark.parametrize("day, utc_829", [("2024-07-11", "12:29"), ("2024-01-11", "13:29")])
def test_fair_price_is_the_829_close_and_the_window_starts_after_the_news_candle(day, utc_829):
    t = pd.date_range(f"{day} 11:00", f"{day} 22:00", freq="1min", tz="UTC", inclusive="left")
    df = pd.DataFrame({"open_time": t, "open": np.arange(len(t), dtype=float)})
    df["high"], df["low"], df["close"] = df.open + 1, df.open - 1, df.open + 0.5
    s = nr.sessions(df)
    i829 = int(np.flatnonzero(t.strftime("%H:%M") == utc_829)[0])
    assert (s["fair"] == df.close[i829]).all()
    w = s.loc[s.window, "et"].dt.strftime("%H:%M")
    assert w.iloc[0] == "08:31" and w.iloc[-1] == "10:59"
    assert s.loc[s.window_open, "et"].dt.strftime("%H:%M").iloc[0] == "09:30"


def test_arms_trade_only_their_own_days():
    days = pd.DataFrame({"any_red": [True, True, True, True], "red830": [True, True, False, True],
                         "aplus": [True, False, False, False], "max_z": [0.4, 2.0, np.nan, np.nan],
                         "titles": ""},
                        index=pd.to_datetime(["2024-07-11", "2024-07-12", "2024-07-15",
                                              "2024-07-16"]).date)
    s = pd.DataFrame({"day": pd.to_datetime(["2024-07-11", "2024-07-12", "2024-07-15",
                                             "2024-07-16", "2024-07-17"]).date})
    on = {a: s["day"][nr.day_mask(s, days, a)].astype(str).tolist() for a in nr.ARMS}
    assert on["news_all"] == ["2024-07-11", "2024-07-12", "2024-07-16"]
    assert on["news_aplus"] == ["2024-07-11"]
    assert on["priced_in"] == ["2024-07-11"] and on["outsiders"] == ["2024-07-12"]
    # a day with no z is neither: an unknown surprise is not a small one
    assert "2024-07-16" not in on["priced_in"] + on["outsiders"]
    # quiet = no red news at all, not merely none at 8:30
    assert on["quiet_days"] == ["2024-07-17"]


def test_entries_are_toward_the_pre_news_price_inside_the_window():
    n = 6
    s = pd.DataFrame({"day": [pd.Timestamp("2024-07-11").date()] * n,
                      "close": [100, 140, 140, 60, 140, 140.0], "fair": 100.0, "scale": 1.0,
                      "window": [True, True, True, True, False, True],
                      "window_open": False, "minute": [511, 511, 516, 521, 600, 526]})
    days = pd.DataFrame({"any_red": [True], "red830": [True], "aplus": [True], "max_z": [0.1],
                         "titles": ""}, index=[pd.Timestamp("2024-07-11").date()])
    sig = {k: np.ones(n, bool) for k in ("disp_s", "disp_l", "bos_s", "bos_l")}
    es, el = nr.arm_entries(s, days, "news_all", sig)
    assert es.tolist() == [False, True, True, False, False, True]   # above fair by >= 0.8 x 38
    assert el.tolist() == [False, False, False, True, False, False]
    ms, ml = nr.arm_entries(s, days, "metro_toward", sig)
    assert ms.tolist() == [False, True, True, False, False, True]   # every 5 min from 8:31
    away_s, away_l = nr.arm_entries(s, days, "metro_away", sig)
    assert away_l.tolist() == [False, True, True, False, False, True] and not away_s[:3].any()
