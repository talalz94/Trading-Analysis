"""
experiments/breaking_news_check.py and the strict BREAKING list (quant/news/headlines.py), on
hand-built candles and the real 21 Oct lines. The promises: the list flags war, evacuation,
emergency-policy and tariff news and none of the 4 Nov false alarms or a printed data result; a
sharp move right after a scheduled release is its result and is left out; each way's alerts are
counted per day and timed as they would have gone (a move only once it is known); the Gulf story
is found by every way that catches it.
"""
import numpy as np
import pandas as pd
import pytest

from experiments import breaking_news_check as bn
from quant.news.headlines import breaking_rule


def at(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize("text, rule", [
    ("US Embassy posts alert over Houthis attacks against Saudi Arabia", "war"),
    ("US Officials: US State Dept is preparing to issue an ordered departure of some personnel",
     "security"),
    ("Israel launched missiles at targets in Iran - Reuters", "war"),
    ("Iran closes the Strait of Hormuz to tankers", "security"),
    ("Japan MOF: intervened in the FX market", "central bank"),
    ("Fed announces emergency rate cut", "central bank"),
    ("Trump imposes 25% tariffs on Mexico", "policy"),
    ("Binance halts withdrawals after exploit", "crypto"),
    # the three lines a looser list flagged on 4 Nov 2026, none of them critical
    ("North Korea test-fired a newly designed advanced air-defence missile system - KCNA", None),
    ("US Treasury: Imposing sanctions against 4 India-based individuals and 9 companies", None),
    ("The UAE mulls a third Crude oil pipeline to avoid the Strait of Hormuz.", None),
    ("Iran could close the Strait of Hormuz, analysts say", None),        # talk, not news
    ("Iran could close the Strait of Hormuz", None),
    ("US CPI MoM Actual 0.5% (Forecast 0.3%)", None),                    # a scheduled result
])
def test_the_breaking_list_flags_critical_news_and_not_the_false_alarms(text, rule):
    assert breaking_rule(text) == rule


START, END = at("2026-10-20"), at("2026-10-23")
RELEASE = at("2026-10-22 12:30")
LATER = at("2026-10-22 14:00")              # its move comes 7 min later: still the result


def _bars():
    t = pd.date_range(START - pd.Timedelta(days=1), END, freq="1min", inclusive="left", tz="UTC")
    r = np.random.default_rng(0).normal(0, 0.0001, len(t))
    r[t.get_indexer([at("2026-10-21 18:44")])] = -0.008      # the Gulf drop, in one minute
    r[t.get_indexer([RELEASE + pd.Timedelta(minutes=1)])] = 0.005   # the CPI jump: a result
    r[t.get_indexer([LATER + pd.Timedelta(minutes=7)])] = -0.005    # a slower result
    return {"Gold": pd.DataFrame({"open_time": t, "close": 4000 * np.exp(np.cumsum(r))})}


LINES = pd.DataFrame([
    (at("2026-10-20 10:00"), "North Korea test-fired a newly designed advanced air-defence "
                             "missile system - KCNA"),
    (at("2026-10-21 18:30"), "Gold steady ahead of US data"),          # earlier, not critical
    (at("2026-10-21 18:34:15"), "US Embassy posts alert over Houthis attacks against Saudi Arabia"),
    (at("2026-10-21 18:47:50"), "US Officials: US State Dept is preparing to issue an ordered "
                                "departure of some personnel and family members"),
    (RELEASE + pd.Timedelta(seconds=5), "US CPI MoM Actual 0.5% (Forecast 0.3%)"),
    (at("2026-10-22 15:00"), "Trump imposes 25% tariffs on Mexico"),
], columns=["t", "text"]).assign(rule=lambda d: d["text"].map(breaking_rule))


def test_a_move_right_after_a_scheduled_release_is_its_result_and_left_out():
    mv = bn.find_all_moves(_bars(), [RELEASE, LATER])
    assert len(mv) == 3
    gulf, cpi, later = mv.iloc[0], mv.iloc[1], mv.iloc[2]
    assert at("2026-10-21 18:39") <= gulf["start"] <= at("2026-10-21 18:45") and not gulf["release"]
    assert cpi["release"] and later["release"] and later["start"] > LATER
    al = bn.alerts(LINES, mv)
    assert (al["way"] == "move").sum() == 1                  # the CPI jump sends nothing


def test_each_way_is_counted_per_day_and_timed_as_it_would_have_gone():
    mv = bn.find_all_moves(_bars(), [RELEASE, LATER])
    al = bn.alerts(LINES, mv)
    days = bn.per_day(al, START, END)
    assert list(days["words"][:3]) == [0, 2, 1]               # embassy + departure; tariffs
    assert list(days["move"][:3]) == [0, 1, 0] and list(days["words + move"][:3]) == [0, 1, 0]
    move = al[al["way"] == "move"].iloc[0]
    assert move["at"] == mv.iloc[0]["start"] + bn.KNOWN      # sent once the move is known
    assert "Embassy" in move["text"]                          # the BREAKING line near it, first
    both = al[al["way"] == "words + move"].iloc[0]
    assert both["at"] == move["at"]                           # the line came before the move


def test_the_gulf_story_is_found_by_every_way_that_catches_it():
    al = bn.alerts(LINES, bn.find_all_moves(_bars(), [RELEASE, LATER]))
    got = bn.first_alert(al)
    assert got["words"]["at"] == at("2026-10-21 18:34:15")    # 10 min before the drop
    assert got["move"] is not None and got["words + move"] is not None
    none = bn.first_alert(al[al["way"] != "move"].assign(at=lambda d: d["at"] + pd.Timedelta(days=5)))
    assert all(v is None for v in none.values())


def test_the_report_prints_the_counts_the_story_and_every_flagged_line(tmp_path):
    p = tmp_path / "FinancialJuice.csv"
    pd.DataFrame({"id": range(len(LINES)), "t": LINES["t"].astype(str),
                  "title": ["FinancialJuice: " + s + "|FJ" for s in LINES["text"]]}).to_csv(p)
    lines = bn.load_lines(p)
    assert list(lines["text"]) == list(LINES["text"])        # the tags are trimmed
    text = bn.report(lines, _bars(), [RELEASE, LATER], START, END)
    assert "ALERTS A DAY" in text and "a day (mean)" in text
    assert "words         18:34:15  US Embassy posts alert" in text
    assert text.count("[war]") == 1 and "[policy]  Trump imposes 25% tariffs" in text
    assert "North Korea" not in text.split("WORDS:")[1].split("MOVES:")[0]


def test_a_move_must_be_big_or_shared_to_count_and_only_the_windows_own_moves_count():
    bars = _bars()
    t = bars["Gold"]["open_time"]
    c = bars["Gold"]["close"].to_numpy().copy()
    k = int(np.flatnonzero(t == at("2026-10-19 12:00"))[0])   # the day before the window
    c[k:] *= 0.99
    bars["Gold"] = bars["Gold"].assign(close=c)
    usd = bars["Gold"].assign(close=150 * (bars["Gold"]["close"] / 4000) ** -0.3)  # moves with Gold
    two = {"Gold": bars["Gold"], "USDJPY": usd}
    mv = bn.find_all_moves(two, [RELEASE, LATER])
    text = bn.report(LINES, two, [RELEASE, LATER], START, END)
    assert (pd.to_datetime(mv["start"], utc=True) < START).any()      # found ...
    assert "Mon 19 Oct" not in text                                   # ... but not counted
    s = mv[(pd.to_datetime(mv["start"], utc=True) >= START)]
    table = bn.sizes_table(s.reset_index(drop=True), START, END)
    assert list(table["Gulf (one)"]) == ["caught"] * 4                 # -0.8% = 3.2x Gold's floor
    assert table.loc["1x floor", "two markets, a day"] > 0            # USDJPY moved with it
    assert table.loc["1x floor", "Gulf (two)"] == "caught"
    small = _bars()
    c = small["Gold"]["close"].to_numpy().copy()
    k = int(np.flatnonzero(small["Gold"]["open_time"] == at("2026-10-20 09:00"))[0])
    c[k:] *= 0.997                                   # -0.3% = 1.2x the floor: counted at 1x only
    small["Gold"] = small["Gold"].assign(close=c)
    one = bn.sizes_table(bn.find_all_moves(small, [RELEASE, LATER]), START, END)
    assert set(one["Gulf (two)"]) == {"missed"}                       # Gold alone
    assert one.loc["1x floor", "one market, a day"] == round(2 / 3, 1)
    assert one.loc["1.5x floor", "one market, a day"] == round(1 / 3, 1)
