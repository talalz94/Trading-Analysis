"""
The breaking-news tiers (quant/news/headlines.py) and experiments/breaking_news_check.py, on
hand-built candles and real lines from the fortnight measured (19 Oct - 2 Nov 2026). The promises:
TOP lines are the few critical whatever the market does; BREAKING lines are wider but go only with
a sharp move; talk, reposts, printed results, a price "hit" and Russia/Ukraine strikes are neither;
a move near a scheduled release is its result; the live rule sends a TOP line at once and a
BREAKING line once its move is known, one per story an hour, at most 8 a day.
"""
import numpy as np
import pandas as pd
import pytest

from experiments import breaking_news_check as bn
from quant.news import headlines as H


def at(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize("text, want", [
    # TOP: critical whatever the market does
    ("US State Dept is preparing to issue an ordered departure of some personnel", (1, "evacuation")),
    ("US troops and diplomats evacuating Middle East posts - NBC", (1, "evacuation")),
    ("Japan MOF: intervened in the FX market", (1, "central bank")),
    ("Fed announces emergency rate cut", (1, "central bank")),
    ("Russia declares war on NATO member", (1, "war declared")),            # Russia, but TOP
    # BREAKING: only with a sharp move
    ("US Embassy posts alert over Houthis attacks against Saudi Arabia", (2, "attack")),
    ("Yemen's Houthis attacked Saudi Arabia's Ras Tanura oil refinery - Statement.", (2, "attack")),
    ("Saudi-led coalition: intercepted and destroyed ballistic missile fired by Yemen's Houthis "
     "toward Riyadh", (2, "attack")),
    ("IRGC: 4th tanker explodes after hitting mine in Strait of Hormuz - IRIB", (2, "attack")),
    ("Iranian missiles hit US base in Qatar", (2, "attack")),
    ("Iran closes the Strait of Hormuz to tankers", (2, "hormuz")),
    ("Trump imposes 25% tariffs on Mexico", (2, "policy")),
    ("Binance halts withdrawals after exploit", (2, "crypto")),
    ("Ceasefire agreed between Israel and Iran", (2, "de-escalation")),
    # neither
    ("North Korea test-fired a newly designed advanced air-defence missile system - KCNA", None),
    ("US Treasury: Imposing sanctions against 4 India-based individuals and 9 companies", None),
    ("The UAE mulls a third Crude oil pipeline to avoid the Strait of Hormuz.", None),
    ("Iran could close the Strait of Hormuz", None),                         # talk
    ("ICYMI: Pakistan-bound LNG tanker attacked near Hormuz despite Iran clearance", None),
    ("US CPI MoM Actual 0.5% (Forecast 0.3%)", None),                        # a scheduled result
    ("Gold hit a record high as Gulf investors bought", None),               # a price "hit"
    ("Fifteen injured in Russian strikes on Ukraine's Kyiv overnight: mayor", None),
    ("Exxon CEO: Hormuz crude getting back close to pre-war levels.", None),
    ("South Korea presidential office holds emergency meeting on North Korea missile launch", None),
])
def test_each_line_lands_in_its_tier(text, want):
    assert H.breaking_tier(text) == want


def test_a_markets_floor_comes_from_its_symbol():
    assert H.move_floor("XAUUSD") == 0.0025 and H.move_floor("BTCUSDT") == 0.005
    assert H.move_floor("USDJPY") == 0.0012 and H.move_floor("US500") is None


def test_one_story_is_told_apart_from_another():
    a = "US troops and diplomats evacuating Middle East posts - NBC"
    assert H.same_story(a, a)
    assert not H.same_story(a, "Trump imposes 25% tariffs on Mexico")


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


LINES = bn.with_tiers(pd.DataFrame([
    (at("2026-10-20 10:00"), "North Korea test-fired a newly designed advanced air-defence "
                             "missile system - KCNA"),
    (at("2026-10-21 18:30"), "Gold steady ahead of US data"),          # earlier, not critical
    (at("2026-10-21 18:34:15"), "US Embassy posts alert over Houthis attacks against Saudi Arabia"),
    (at("2026-10-21 18:47:50"), "US Officials: US State Dept is preparing to issue an ordered "
                                "departure of some personnel and family members"),
    (RELEASE + pd.Timedelta(seconds=5), "US CPI MoM Actual 0.5% (Forecast 0.3%)"),
    (at("2026-10-22 15:00"), "Trump imposes 25% tariffs on Mexico"),   # BREAKING, no move
], columns=["t", "text"]))


def test_a_move_right_after_a_scheduled_release_is_its_result_and_left_out():
    mv = bn.find_all_moves(_bars(), [RELEASE, LATER])
    assert len(mv) == 3
    gulf, cpi, later = mv.iloc[0], mv.iloc[1], mv.iloc[2]
    assert at("2026-10-21 18:39") <= gulf["start"] <= at("2026-10-21 18:45") and not gulf["release"]
    assert cpi["release"] and later["release"] and later["start"] > LATER
    assert (bn.alerts(LINES, mv)["way"] == "move").sum() == 1   # the results send nothing


def test_each_way_is_counted_per_day_and_timed_as_it_would_have_gone():
    mv = bn.find_all_moves(_bars(), [RELEASE, LATER])
    al = bn.alerts(LINES, mv)
    days = bn.per_day(al, START, END)
    assert list(days["words"][:3]) == [0, 2, 1]               # embassy + departure; tariffs
    assert list(days["move"][:3]) == [0, 1, 0]
    assert list(days["live"][:3]) == [0, 2, 0]                # the tariffs line had no move
    move = al[al["way"] == "move"].iloc[0]
    assert move["at"] == mv.iloc[0]["start"] + H.MOVE_KNOWN   # sent once the move is known
    assert "Embassy" in move["text"]                          # the flagged line near it, first


def test_the_live_rule_sends_top_at_once_and_breaking_once_its_move_is_known():
    mv = bn.find_all_moves(_bars(), [RELEASE, LATER])
    lv = bn.live(LINES, mv)
    top = lv[lv["tier"] == 1].iloc[0]
    assert top["at"] == at("2026-10-21 18:47:50") and top["rule"] == "evacuation"
    brk = lv[lv["tier"] == 2].iloc[0]
    assert brk["at"] == mv.iloc[0]["start"] + H.MOVE_KNOWN and brk["market"].startswith("Gold -")
    m = mv.iloc[0]["start"]
    for t in (m - H.MOVE_AFTER - pd.Timedelta(minutes=1),     # the move came too long after it
              m + H.MOVE_BEFORE + pd.Timedelta(minutes=1)):   # ... or too long before it
        one = bn.with_tiers(pd.DataFrame([(t, "Iranian missiles hit US base in Qatar")],
                                         columns=["t", "text"]))
        assert bn.live(one, mv).empty
    inside = bn.with_tiers(pd.DataFrame([(m + H.MOVE_BEFORE, "Iranian missiles hit US base in Qatar")],
                                        columns=["t", "text"]))
    assert len(bn.live(inside, mv)) == 1


def test_one_story_goes_once_an_hour_and_no_more_than_eight_a_day():
    twice = bn.with_tiers(pd.DataFrame([(at("2026-10-21 20:53"), "US troops and diplomats "
                                         "evacuating Middle East posts - NBC")] * 2 +
                                       [(at("2026-10-21 22:10"), "US troops and diplomats "
                                         "evacuating Middle East posts - NBC")],
                                       columns=["t", "text"]))
    assert list(bn.live(twice, bn.find_all_moves({}, []))["at"]) == [at("2026-10-21 20:53"),
                                                                    at("2026-10-21 22:10")]
    tops = ["Fed announces emergency rate cut", "Japan MOF: intervened in the FX market",
            "US troops and diplomats evacuating Middle East posts", "Martial law declared in Seoul",
            "Coup attempt under way in Ankara", "ECB holds emergency meeting on markets",
            "Nuclear test detected near Punggye-ri", "Ordered departure for embassy staff in Doha",
            "BoJ conducted intervention overnight", "Embassy in Tel Aviv closed until further notice",
            "SNB emergency session on franc", "Declares war on neighbour in televised address"]
    many = bn.with_tiers(pd.DataFrame([(at("2026-10-21 00:00") + pd.Timedelta(minutes=5 * i), s)
                                       for i, s in enumerate(tops)], columns=["t", "text"]))
    assert (many["tier"] == 1).all()                          # twelve different TOP stories
    assert len(bn.live(many, bn.find_all_moves({}, []))) == H.BREAKING_PER_DAY


def test_the_gulf_story_is_found_by_every_way_that_catches_it():
    al = bn.alerts(LINES, bn.find_all_moves(_bars(), [RELEASE, LATER]))
    got = bn.first_alert(al)
    assert got["words"]["at"] == at("2026-10-21 18:34:15")    # 10 min before the drop
    assert got["move"] is not None and got["live"] is not None
    assert got["live"]["at"] <= at("2026-10-21 18:50")
    none = bn.first_alert(al.assign(at=lambda d: d["at"] + pd.Timedelta(days=5)))
    assert all(v is None for v in none.values())


def test_the_report_prints_the_counts_the_story_and_every_flagged_line(tmp_path):
    p = tmp_path / "FinancialJuice.csv"
    pd.DataFrame({"id": range(len(LINES)), "t": LINES["t"].astype(str),
                  "title": ["FinancialJuice: " + s + "|FJ" for s in LINES["text"]]}).to_csv(p)
    lines = bn.load_lines(p)
    assert list(lines["text"]) == list(LINES["text"])        # the tags are trimmed
    text = bn.report(lines, _bars(), [RELEASE, LATER], START, END)
    assert "ALERTS A DAY" in text and "a day (mean)" in text
    assert "words  18:34:15  US Embassy posts alert" in text
    assert "T1 [evacuation]  US Officials: US State Dept" in text.split("LIVE:")[1]
    assert "T2 [policy]  Trump imposes 25% tariffs" in text.split("WORDS:")[1]
    assert "Trump imposes" not in text.split("LIVE:")[1].split("WORDS:")[0]
    assert "North Korea" not in text.split("WORDS:")[1]


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
