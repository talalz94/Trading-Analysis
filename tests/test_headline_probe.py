"""
The headline probe and its report (experiments/headline_probe.py, headline_report.py) -- a test of
free news sources before any is wired into the alerts. What they record, and how they judge it,
on hand-built feeds and a fake clock: no network.
"""
import numpy as np
import pandas as pd
import pytest

from experiments import headline_report as hr
from experiments.headline_probe import Probe, Source, parse_rss, parse_telegram

RSS = b"""\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>t</title>
<item><title>FinancialJuice: US Officials: US State Dept is preparing to issue an ordered
 departure at US embassies across the Gulf</title><link>https://x/1</link><guid>g1</guid>
 <pubDate>Wed, 21 Oct 2026 18:47:50 GMT</pubDate></item>
<item><title>Kenya - Level 3: Reconsider Travel</title><link>https://x/2</link>
 <pubDate>Mon, 19 Oct 2026</pubDate></item>
</channel></rss>"""

TELEGRAM = """<html><body>
<div class="tgme_widget_message_wrap"><div class="tgme_widget_message js-widget_message"
 data-post="WalterBloomberg/101"><div class="tgme_widget_message_text js-message_text" dir="auto">
*US PREPARING ORDERED DEPARTURE FROM GULF EMBASSIES: OFFICIALS<br/>&amp; FAMILIES</div>
<div class="tgme_widget_message_footer"><a class="tgme_widget_message_date" href="x">
<time datetime="2026-10-21T18:46:05+00:00" class="time">18:46</time></a></div></div></div>
<div class="tgme_widget_message_wrap"><div class="tgme_widget_message js-widget_message"
 data-post="WalterBloomberg/102"><div class="tgme_widget_message_photo"></div>
<time datetime="2026-10-21T18:50:00+00:00" class="time">18:50</time></div></div>
</body></html>"""


def at(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture(autouse=True)
def _no_real_calendar(monkeypatch, tmp_path):
    """The report reads this PC's calendar and alerts database by default: not in a test."""
    monkeypatch.setattr(hr, "CALENDARS", ())
    monkeypatch.setattr(hr, "ALERTS_DB", tmp_path / "no_alerts.db")


def test_rss_reads_bytes_with_a_byte_order_mark_and_refuses_a_made_up_time():
    # the Fed's feed starts with a BOM and declares no charset: decoded as text, line 1 failed
    a, b = parse_rss(RSS)
    assert a["id"] == "g1" and a["published"] == at("2026-10-21 18:47:50").isoformat()
    assert "ordered departure at US embassies" in a["title"] and "\n" not in a["title"]
    assert b["id"] == "https://x/2" and b["published"] is None      # a date with no time of day


def test_telegram_page_gives_each_messages_own_time_and_skips_a_wordless_post():
    (m,) = parse_telegram(TELEGRAM.encode())
    assert m["id"] == "WalterBloomberg/101" and m["published"] == at("2026-10-21 18:46:05").isoformat()
    assert m["title"] == "*US PREPARING ORDERED DEPARTURE FROM GULF EMBASSIES: OFFICIALS & FAMILIES"


class Clock:
    def __init__(self, t):
        self.t = at(t)

    def __call__(self):
        return self.t


def _probe(tmp_path, pages, clock):
    """A probe over one RSS source whose successive downloads are `pages` (bytes, or an
    Exception to raise, or a status int)."""
    queue = list(pages)

    def fetch(url, headers):
        p = queue.pop(0)
        if isinstance(p, Exception):
            raise p
        if isinstance(p, int):
            return p, b"", {}
        return 200, p, {"ETag": "v1"}

    src = Source("feed", "rss", "https://x/feed", 60, "squawk")
    return Probe(tmp_path / "p.sqlite", sources=[src], fetch=fetch, clock=clock), src


def test_a_headline_is_stored_once_and_the_first_polls_backlog_is_not_timed(tmp_path):
    clock = Clock("2026-10-21 18:40")
    p, src = _probe(tmp_path, [RSS, RSS], clock)
    assert p.poll(src)["new"] == 2
    clock.t = at("2026-10-21 18:41")
    r = p.poll(src)
    assert r["ok"] and r["n"] == 2 and r["new"] == 0                    # seen again, stored once
    rows = p.db.execute("SELECT item_id, backlog, first_seen FROM items").fetchall()
    assert {(i, b) for i, b, _ in rows} == {("g1", 1), ("https://x/2", 1)}
    assert all(f.startswith("2026-10-21T18:40") for _, _, f in rows)


def test_a_failing_source_backs_off_and_is_recorded_and_a_success_resets_it(tmp_path):
    clock = Clock("2026-10-21 18:40")
    p, src = _probe(tmp_path, [ConnectionError("down"), 503, RSS, 304], clock)
    waits = []
    for _ in range(4):
        r = p.poll(src)
        waits.append((p.due["feed"] - clock.t).total_seconds())
        clock.t = p.due["feed"]
    assert waits == [120, 240, 60, 60]                    # 2x, 4x, then back to every 60 s
    polls = p.db.execute("SELECT ok, status, error FROM polls ORDER BY rowid").fetchall()
    assert [o for o, _, _ in polls] == [0, 0, 1, 1]
    assert "down" in polls[0][2] and polls[1][1] == 503
    assert p.cache["feed"] == {"If-None-Match": "v1"}     # the next ask is conditional


def test_the_test_window_counts_from_the_first_start_not_a_restart(tmp_path):
    clock = Clock("2026-10-21 18:00")
    p, _ = _probe(tmp_path, [RSS] * 50, clock)

    def sleep(s):
        clock.t += pd.Timedelta(seconds=s)

    p.run(days=1 / 24, sleep=sleep, log=lambda *a: None)                 # one hour
    assert clock.t >= at("2026-10-21 19:00")
    clock.t = at("2026-10-21 18:30")                                     # a restart inside it
    q, _ = _probe(tmp_path, [RSS] * 50, clock)
    assert q.meta("start") == at("2026-10-21 18:00").isoformat()
    q.run(days=1 / 24, sleep=sleep, log=lambda *a: None)
    assert clock.t < at("2026-10-21 19:02")                              # not 19:30


def _items(rows):
    d = pd.DataFrame(rows, columns=["source", "title", "printed", "seen"])
    d["published"] = pd.to_datetime(d["printed"], utc=True)
    d["first_seen"] = pd.to_datetime(d["seen"], utc=True)
    d["backlog"] = 0
    return d


GULF = [
    ("walter_telegram", "*US PREPARING ORDERED DEPARTURE FROM GULF EMBASSIES: OFFICIALS",
     "2026-10-21 18:46:05", "2026-10-21 18:46:40"),
    ("fj_rss", "FinancialJuice: US Officials: US State Dept is preparing to issue an ordered "
               "departure of some personnel and family members at US embassies across the Gulf",
     "2026-10-21 18:47:50", "2026-10-21 18:48:30"),
    ("gnews_security", "US Officials: State Dept Preparing To Issue Ordered Departure Of Some "
                       "Personnel At US Embassies Across The Gulf - Forex Factory",
     "2026-10-21 18:48:38", "2026-10-21 18:52:00"),
    ("investinglive", "US State Department is to prepared to issue a order departure at US "
                      "embassies in the Gulf", "2026-10-21 19:13:21", "2026-10-21 19:14:00"),
    ("fj_rss", "FinancialJuice: China PBOC expected to set yuan midpoint at 6.6756",
     "2026-10-21 18:50:00", "2026-10-21 18:50:30"),
    ("gnews_security", "The Current State of the Iran War - UCLA",              # dated hours before
     "2026-10-21 12:00:00", "2026-10-21 18:55:00"),
]


def test_one_story_across_sources_is_grouped_and_each_sources_lag_measured():
    d = hr.timed(_items(GULF))
    s = hr.stories(d)
    gulf = s[d["title"].str.contains("ORDERED DEPARTURE|ordered departure|order departure")]
    assert gulf.nunique() == 1 and s.nunique() == 3                    # Gulf, PBOC, Iran war
    t = hr.speed_table(d)
    assert t.loc["walter_telegram", "first_%"] == 100 and t.loc["walter_telegram", "lag_med_s"] == 0
    assert t.loc["fj_rss", "lag_med_s"] == 105                         # 18:47:50 - 18:46:05
    assert t.loc["investinglive", "lag_med_s"] == 27 * 60 + 16


def test_a_printed_time_far_before_we_saw_it_is_not_trusted():
    d = hr.timed(_items(GULF))
    old = d[d["title"].str.startswith("The Current State")].iloc[0]
    assert pd.isna(old["printed"]) and old["t"] == at("2026-10-21 18:55")


def _bars(jump_at="2026-10-21 18:45", size=-0.008, days=2):
    t = pd.date_range("2026-10-20 00:00", periods=days * 1440, freq="1min", tz="UTC")
    c = 4100 + np.random.default_rng(0).normal(0, 0.3, len(t)).cumsum() * 0.1
    i = int(np.flatnonzero(t == at(jump_at))[0])
    c[i:] += c[i] * size                                 # one sharp step down
    return pd.DataFrame({"open_time": t, "close": c})


def test_a_sharp_move_is_found_once_at_its_first_minute():
    m = hr.find_moves(_bars(), "Gold")
    assert len(m) == 1
    assert at("2026-10-21 18:40") <= m["start"].iat[0] <= at("2026-10-21 18:45")
    assert m["move_pct"].iat[0] < -0.5


def test_coverage_says_which_source_had_a_matching_headline_and_how_late():
    moves = pd.DataFrame([dict(market="Gold", start=at("2026-10-21 18:44"), move_pct=-0.8)])
    d = hr.timed(_items(GULF))
    per_move, summ = hr.coverage(moves, d, ["walter_telegram", "fj_rss", "investinglive", "fed"])
    row = per_move.iloc[0]
    assert row["walter_telegram"] == pytest.approx(2.1) and row["fj_rss"] == pytest.approx(3.8)
    assert np.isnan(row["investinglive"])                # 29 min after: outside the window
    assert np.isnan(row["fed"]) and summ.loc["fed", "covered_pct"] == 0
    assert "ORDERED DEPARTURE" in row["first_headline"]


TIMES = [(2, 13), (5, 40), (9, 7), (12, 31), (14, 46), (17, 2), (19, 55), (22, 18)]
MOVES = pd.DataFrame(dict(market="Gold", move_pct=-0.5, start=[
    at(f"2026-10-{day} {h:02d}:{m:02d}") for day in (21, 22) for h, m in TIMES]))
SPAN = (at("2026-10-21 00:00"), at("2026-10-22 23:59"))


def _sources():
    busy = pd.date_range(*SPAN, freq="10min")            # about gold every 10 minutes, all day
    rows = [("busy", "Gold prices steady as traders wait", t, t) for t in busy]
    rows += [("sharp", "Gold drops after missile strike", m - pd.Timedelta(minutes=2),
              m - pd.Timedelta(minutes=2)) for m in MOVES["start"]]   # only just before a move
    rows += [("offtopic", "Football results", m, m) for m in MOVES["start"]]
    return hr.timed(_items(rows))


def test_a_source_that_posts_all_the_time_covers_every_move_and_gets_no_credit():
    # The trap the baseline exists for: Google News posts 300-600 headlines a day, so a "matching"
    # headline sits next to almost ANY moment -- covered 100% is what random moments get too
    _, summ = hr.coverage(MOVES, _sources(), ["busy", "sharp", "offtopic"], *SPAN, draws=200)
    busy = summ.loc["busy"]
    assert busy["covered_pct"] == 100 and busy["chance_day_pct"] >= 95
    assert busy["p_day"] > 0.5 and busy["verdict"] == "chance" and busy["above_chance"] <= 5


def test_a_source_that_posts_only_before_moves_beats_both_baselines():
    _, summ = hr.coverage(MOVES, _sources(), ["busy", "sharp", "offtopic"], *SPAN, draws=200)
    sharp = summ.loc["sharp"]
    assert sharp["covered_pct"] == 100
    assert sharp["chance_day_pct"] < 30 and sharp["chance_near_pct"] < 30
    assert sharp["p_day"] < 0.05 and sharp["p_near"] < 0.05 and sharp["verdict"] == "CREDIT"
    assert summ.index[0] == "sharp"                       # ranked by how far above chance
    off = summ.loc["offtopic"]                            # right on time, wrong words: nothing
    assert off["covered_pct"] == 0 and off["verdict"] == "chance"


def test_the_baseline_uses_the_moves_own_matching_rule():
    # at the moves themselves the baseline's rule gives exactly the real coverage; moved 45
    # minutes later (headline 47 min before: outside -15/+20), nothing matches and p is 1
    d = _sources()
    c = hr.chance(MOVES, d, ["sharp"], *SPAN, draws=50)
    assert c.loc["sharp", "real_n"] == len(MOVES)
    late = MOVES.assign(start=MOVES["start"] + pd.Timedelta(minutes=45))
    c = hr.chance(late, d, ["sharp"], *SPAN, draws=50)
    assert c.loc["sharp", "real_n"] == 0 and c.loc["sharp", "p_day"] == 1.0
    c = hr.chance(MOVES, d, ["offtopic"], *SPAN, draws=50)    # on time, but not about gold
    assert c.loc["offtopic", "real_n"] == 0 and c.loc["offtopic", "chance_day_pct"] == 0


def test_a_source_busy_only_in_the_us_session_is_not_credited_for_moves_there():
    # Moves crowd the US session, and so do headlines. Against random moments of the WHOLE day
    # such a source looks prescient (p_day tiny); against moments 1-3 hours away -- still the
    # session -- it is not. That is why both baselines must be beaten
    moves = pd.DataFrame(dict(market="Gold", move_pct=-0.5, start=[
        at(f"2026-10-{day} {h}") for day in (21, 22) for h in ("14:10", "15:20", "16:30",
                                                              "17:15", "17:50")]))
    rows = []
    for day in (21, 22):
        for t in pd.date_range(f"2026-10-{day} 12:00", f"2026-10-{day} 19:55", freq="5min",
                               tz="UTC"):
            rows.append(("session", "Gold traders eye the dollar", t, t))
    d = hr.timed(_items(rows))
    _, summ = hr.coverage(moves, d, ["session"], *SPAN, draws=200)
    row = summ.loc["session"]
    assert row["covered_pct"] == 100 and row["p_day"] < 0.05     # fooled by the day alone ...
    assert 60 <= row["chance_near_pct"] < 100 and row["p_near"] >= 0.05   # ... not by the hour
    assert row["verdict"] == "chance"                             # a little above it is not enough
    assert row["above_chance"] == row["covered_pct"] - max(row["chance_day_pct"],
                                                            row["chance_near_pct"])


def test_the_report_runs_end_to_end_on_a_recorded_database(tmp_path):
    later = RSS.replace(b"</channel>", b"<item><title>US missile strike on Houthi sites</title>"
                        b"<guid>g9</guid><pubDate>Wed, 21 Oct 2026 18:49:00 GMT</pubDate></item>"
                        b"</channel>")
    clock = Clock("2026-10-21 18:40")
    p, src = _probe(tmp_path, [RSS, later], clock)
    p.meta("start", clock.t.isoformat())
    p.poll(src)
    empty = hr.report(tmp_path / "p.sqlite", prices=False, out=tmp_path / "r.txt")
    assert "0 timed headlines" in empty                  # the first minutes: nothing timed yet
    clock.t = at("2026-10-21 18:50")
    p.poll(src)
    text = hr.report(tmp_path / "p.sqlite", prices=False, out=tmp_path / "r.txt")
    assert "1 timed headlines" in text and "RELIABILITY" in text and "SPEED" in text
    assert "P1 security" in text                          # the missile strike is flagged
    assert (tmp_path / "r.txt").read_text(encoding="utf-8") == text


def test_the_report_prints_coverage_against_chance(tmp_path, monkeypatch):
    # the whole report with prices, the candle download replaced by a sharp Gold step at 18:45
    later = RSS.replace(b"</channel>", b"<item><title>US missile strike on Houthi sites</title>"
                        b"<guid>g9</guid><pubDate>Wed, 21 Oct 2026 18:44:00 GMT</pubDate></item>"
                        b"</channel>")
    clock = Clock("2026-10-21 18:00")
    p, src = _probe(tmp_path, [RSS, later], clock)
    p.meta("start", clock.t.isoformat())
    p.poll(src)
    clock.t = at("2026-10-21 18:50")
    p.poll(src)
    bars = _bars().assign(open=lambda b: b["close"], high=lambda b: b["close"],
                          low=lambda b: b["close"])
    monkeypatch.setattr(hr, "fetch_bars", lambda start, end: {"Gold": bars})
    monkeypatch.setattr(hr, "CHANCE_DRAWS", 50)
    text = hr.report(tmp_path / "p.sqlite", prices=True, out=tmp_path / "r.txt")
    assert "COVERAGE: 1 sharp moves" in text and "chance_day_pct" in text
    assert "chance_near_pct" in text and "verdict" in text and "CREDIT = beats both" in text


# ── CRITICAL RELEASES ─────────────────────────────────────────────────────────────────────────
CRIT = pd.DataFrame([
    dict(t=at("2026-10-28 18:00"), currency="USD", title="Federal Funds Rate", level="critical",
         actual="4.00%", forecast="4.00%"),
    dict(t=at("2026-10-28 18:00"), currency="USD", title="FOMC Statement", level="critical",
         actual=None, forecast=None),
    dict(t=at("2026-10-30 03:02"), currency="JPY", title="BOJ Policy Rate", level="critical",
         actual=None, forecast="<1.25%"),          # Forex Factory's time is a guess for the BOJ
])
CRIT_NEWS = [
    ("fj_rss", "FinancialJuice: FOMC decision due shortly",                     # a preview
     "2026-10-28 17:58:00", "2026-10-28 17:58:40"),
    ("fj_rss", "FinancialJuice: FED LEAVES RATES UNCHANGED AT 3.75%-4.00%",
     "2026-10-28 18:00:04", "2026-10-28 18:00:40"),
    ("walter_telegram", "*FED HOLDS RATES STEADY", "2026-10-28 18:00:09", "2026-10-28 18:00:20"),
    ("walter_telegram", "*FED: VOTE 10-2", "2026-10-28 18:00:30", "2026-10-28 18:00:45"),
    ("investinglive", "Fed leaves rates unchanged, as expected",
     "2026-10-28 18:07:00", "2026-10-28 18:07:30"),
    ("slow", "Fed holds rates", "2026-10-28 18:11:00", "2026-10-28 18:11:30"),   # too late
    ("gnews_gold", "Gold steady ahead of central bank news - Reuters",          # not the Fed
     "2026-10-28 18:00:30", "2026-10-28 18:01:00"),
    ("fj_rss", "FinancialJuice: BOJ KEEPS RATES ON HOLD AT 1.00%",
     "2026-10-30 03:41:00", "2026-10-30 03:41:30"),
]
CRIT_SOURCES = ["fj_rss", "walter_telegram", "investinglive", "slow", "gnews_gold"]


def test_a_critical_release_gets_each_sources_first_headline_in_seconds():
    g = hr.release_groups(CRIT)
    assert list(g["titles"]) == ["Federal Funds Rate + FOMC Statement", "BOJ Policy Rate"]
    assert g["numbers"].iat[0] == "Federal Funds Rate 4.00% (forecast 4.00%)"
    r = hr.release_delivery(g, hr.timed(_items(CRIT_NEWS)), CRIT_SOURCES).set_index(
        ["release", "source"])
    fj, walter = r.loc[(0, "fj_rss")], r.loc[(0, "walter_telegram")]
    assert (fj["delay_s"], fj["seen_s"], fj["number_s"]) == (4, 40, 4)   # not the 17:58 preview
    assert walter["delay_s"] == 9 and walter["number_s"] == 30   # its first had no number
    assert r.loc[(0, "investinglive"), "delay_s"] == 420
    assert np.isnan(r.loc[(0, "investinglive"), "number_s"])
    assert (0, "slow") not in r.index and (0, "gnews_gold") not in r.index
    assert r.loc[(1, "fj_rss"), "delay_s"] == 39 * 60          # the BOJ has no set time


def test_carrying_a_release_is_judged_against_the_same_window_on_other_days():
    g = hr.release_groups(CRIT.iloc[:2])
    busy = [("busy", f"Fed speaker number {k} on the outlook", f"2026-10-{day} 18:03:00",
             f"2026-10-{day} 18:03:20") for k, day in enumerate(range(22, 31))]
    d = hr.timed(_items(CRIT_NEWS + busy))
    c = hr.release_chance(g, d, ["busy", "fj_rss"], at("2026-10-22 00:00"), at("2026-10-30 23:00"))
    assert c["busy"] == 100 and c["fj_rss"] == 0       # busy "carries" 18:00 every day


def test_our_result_email_is_read_from_the_alerts_database(tmp_path):
    import sqlite3
    from quant.alerts.store import SCHEMA
    db = tmp_path / "alerts.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    for i, (tf, sent) in enumerate([("result", "2026-10-28T18:01:05+00:00"),
                                    ("T-15", "2026-10-28T17:45:20+00:00")], start=1):
        con.execute("INSERT INTO outbox (id, sent_at) VALUES (?, ?)", (i, sent))
        con.execute("INSERT INTO events (id, alert, tf, market, bar_time, outbox_id) VALUES "
                    "(?, 'news', ?, 'USD', '2026-10-28T18:00:00+00:00', ?)", (f"e{i}", tf, i))
    con.commit()
    con.close()
    e = hr.alert_emails(hr.release_groups(CRIT), db)
    assert e.iat[0] == 65 and np.isnan(e.iat[1])       # the result, not the reminder; no BOJ
    assert hr.alert_emails(hr.release_groups(CRIT), tmp_path / "none.db").isna().all()


def test_a_release_move_is_measured_against_the_same_minute_on_other_days():
    t = pd.date_range("2026-10-26 00:00", "2026-10-29 23:59", freq="1min", tz="UTC")
    c = np.where(np.arange(len(t)) % 2 == 1, 4000.4, 4000.0)       # a wiggle every minute
    i = int(np.flatnonzero(t == at("2026-10-28 18:00"))[0])
    c[i:] = c[i:] * 0.99                                         # -1% at the release
    bars = pd.DataFrame({"open_time": t, "close": c})
    rx = hr.reaction(bars, at("2026-10-28 18:00"), t[0], t[-1])
    move = 3960.0 / 4000.4 - 1
    usual = 1 - 4000.0 / 4000.4
    for h in (1, 5):
        assert rx[h][0] == pytest.approx(100 * move)
        assert rx[h][1] == pytest.approx(abs(move) / usual)
    shut = hr.reaction(bars[bars["open_time"].dt.day != 28], at("2026-10-28 18:00"), t[0], t[-1])
    assert np.isnan(shut[1][0])


def test_the_report_leads_with_the_critical_releases(tmp_path, monkeypatch):
    later = RSS.replace(b"</channel>", b"<item><title>US NON-FARM PAYROLLS 254K VS 150K "
                        b"EXPECTED</title><guid>g9</guid><pubDate>Wed, 21 Oct 2026 18:30:05 GMT"
                        b"</pubDate></item></channel>")
    clock = Clock("2026-10-21 18:00")
    p, src = _probe(tmp_path, [RSS, later], clock)
    p.meta("start", clock.t.isoformat())
    p.poll(src)
    clock.t = at("2026-10-21 18:50")
    p.poll(src)
    nfp = pd.DataFrame([dict(t=at("2026-10-21 18:30"), currency="USD", level="critical",
                             title="Non-Farm Employment Change", actual="254K", forecast="150K")])
    seen = {}

    def calendar(start, end, levels, roots):
        seen.update(start=start, end=end, levels=levels)
        return nfp, "test calendar"
    monkeypatch.setattr(hr, "load_calendar", calendar)
    text = hr.report(tmp_path / "p.sqlite", prices=False, out=tmp_path / "r.txt",
                     critical_only=True)
    assert "CRITICAL RELEASES: 1 in the test" in text and "254K (forecast 150K)" in text
    assert "feed                      5 s" in text and "RELIABILITY" not in text
    assert seen["end"] == at("2026-10-21 18:40") and seen["levels"] == ("critical",)
    full = hr.report(tmp_path / "p.sqlite", prices=False, out=tmp_path / "r.txt")
    assert full.index("CRITICAL RELEASES") < full.index("RELIABILITY")


def test_a_broken_calendar_never_loses_the_rest_of_the_report(tmp_path, monkeypatch):
    clock = Clock("2026-10-21 18:40")
    p, src = _probe(tmp_path, [RSS], clock)
    p.meta("start", clock.t.isoformat())
    p.poll(src)

    def broken(*a):
        raise OSError("disk")
    monkeypatch.setattr(hr, "load_calendar", broken)
    text = hr.report(tmp_path / "p.sqlite", prices=False, out=tmp_path / "r.txt")
    assert "the calendar could not be read (OSError: disk)" in text and "RELIABILITY" in text
