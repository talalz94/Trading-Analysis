"""
News notices in the alerts monitor -- on a fake calendar and a fake clock, so every test states what
the calendar said and what time it was. The promises: each notice goes ONCE; a reminder is never
sent after its release, and after an outage only the latest one still ahead goes; a digest outside
its window is not sent at all; results carry the surprise and its meaning; a moved release is
reported; a news failure never stops a candle alert; and an email without news is unchanged.
"""
import json

import numpy as np
import pandas as pd
import pytest

from quant.alerts import REGISTRY, ConfigError, Detector, load_config, tick
from quant.alerts.feeds import FakeFeed
from quant.alerts.news import FakeNews, parse_news, reminder_offsets, NewsConfigError
from quant.alerts.notify import ListNotifier
from quant.alerts.store import State
from quant.data.base import TIME_COL
from quant.news.calendar import finish


class Spike(Detector):
    kind = "spike"

    def validate(self, p):
        pass

    def warmup_bars(self, p):
        return 5

    def evaluate(self, frame, p, tf):
        return [dict(i=int(i), what="spike", side="bull")
                for i in np.flatnonzero(frame["high"].to_numpy() >= 999)]


@pytest.fixture(autouse=True)
def _spike(monkeypatch):
    monkeypatch.setitem(REGISTRY, "spike", Spike())


_OPEN = []


@pytest.fixture(autouse=True)
def _close():
    yield
    while _OPEN:
        _OPEN.pop().close()


def at(s):
    return pd.Timestamp(s, tz="UTC")


def rel(t, title="CPI m/m", cur="USD", impact="high", actual=None, forecast="0.3%",
        previous="0.2%", q="exact"):
    return dict(t=at(t), time_q=q, currency=cur, title=title, impact=impact, actual=actual,
                forecast=forecast, previous=previous, source="ff_feed")


def calendar(*rows):
    return finish(pd.DataFrame(list(rows)))


def bars(spikes=(), start="2026-10-10", end="2026-10-16"):
    t = pd.date_range(start, end, freq="5min", tz="UTC", inclusive="left")
    c = 100 + np.random.default_rng(0).normal(0, 0.1, len(t)).cumsum()
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({TIME_COL: t, "open": o, "high": np.maximum(o, c) + 0.05,
                       "low": np.minimum(o, c) - 0.05, "close": c, "volume": 1.0})
    for s in spikes:
        df.loc[df[TIME_COL] == at(s), "high"] = 1000.0
    return df


NEWS = dict(on=True, currencies={"USD": "medium", "EUR": "high"}, timezone="Asia/Karachi",
            week_ahead=None, day_ahead=None, remind_before={"critical": (60, 15), "high": (15,)},
            results="high", warn_alerts=30)


class Rig:
    def __init__(self, tmp_path, news_cfg=None, events=None, alerts=(), spikes=(), delivery=None,
                 categories=None, **fake):
        tmp_path.mkdir(parents=True, exist_ok=True)
        p = tmp_path / "cfg.py"
        cfg = dict(NEWS, **(news_cfg or {}))
        p.write_text(f"MARKETS = {{'A': dict(symbol='XAUUSD', source='dukascopy')}}\n"
                     f"ALERTS = {list(alerts)!r}\nNEWS = {cfg!r}\n"
                     + (f"DELIVERY = {delivery!r}\n" if delivery else "")
                     + (f"CATEGORIES = {categories!r}\n" if categories else ""))
        self.cfg = load_config(p)
        self.state = State(tmp_path / "state")
        _OPEN.append(self.state)
        self.mail = ListNotifier()
        self.feed = FakeFeed({"A": bars(spikes)})
        self.news = FakeNews(events if events is not None else calendar(), **fake)

    def tick(self, when):
        return tick(self.cfg, self.state, self.mail, self.feed, now=at(when), news=self.news)

    def subjects(self):
        return [s for s, _, _ in self.mail.sent]

    def rows(self, **where):
        q = "SELECT * FROM events" + (" WHERE " + " AND ".join(f"{k}=?" for k in where)
                                      if where else "")
        return [dict(r) for r in self.state.db.execute(q, tuple(where.values()))]


# ── config ──────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad, says", [
    (dict(remind=5), "unknown settings"),
    (dict(timezone="Mars/Olympus"), "unknown zone"),
    (dict(currencies={"XYZ": "high"}), "unknown currency"),
    (dict(currencies={"USD": "red"}), "level"),
    (dict(week_ahead="Someday 20:00"), "week_ahead"),
    (dict(day_ahead="8pm"), "HH:MM"),
    (dict(remind_before={"high": (0,)}), "1..2880"),
    (dict(remind_before={"none": (15,)}), "level"),
    (dict(results="red"), "results"),
    (dict(headlines=("reuters",)), "headlines"),
    (dict(headlines=("financialjuice", "financialjuice")), "headlines"),
    (dict(instant="yes"), "instant"),
    (dict(breaking=("reuters",)), "breaking"),
])
def test_a_news_setting_mistake_is_refused_by_name(bad, says):
    with pytest.raises(NewsConfigError, match=says):
        parse_news(dict(NEWS, **bad))


def test_news_is_off_without_a_block_and_its_name_is_reserved(tmp_path):
    p = tmp_path / "c.py"
    p.write_text("MARKETS = {'A': dict(symbol='XAUUSD', source='dukascopy')}\nALERTS = []\n")
    cfg = load_config(p)
    assert not cfg.news.on and "News" not in cfg.categories
    p.write_text("MARKETS = {'A': dict(symbol='XAUUSD', source='dukascopy')}\n"
                 "ALERTS = [dict(name='news', kind='spike', tfs='1h')]\n")
    with pytest.raises(ConfigError, match="reserved"):
        load_config(p)


def test_the_shipped_config_has_news_on_and_first():
    cfg = load_config()
    assert cfg.news.on and cfg.categories[0] == "News"


def test_reminder_offsets_come_from_the_nearest_level_at_or_below():
    nc = parse_news(dict(NEWS, remind_before={"high": (15,)}))
    assert reminder_offsets(nc, "critical") == (15,)
    assert reminder_offsets(nc, "high") == (15,)
    assert reminder_offsets(nc, "medium") == ()


def test_a_medium_level_reminds_of_medium_releases_and_leaves_high_ones_their_own():
    nc = parse_news(dict(NEWS, remind_before={"high": (60, 15), "medium": (15,)}))
    assert reminder_offsets(nc, "medium") == (15,)
    assert reminder_offsets(nc, "high") == reminder_offsets(nc, "critical") == (60, 15)
    assert reminder_offsets(nc, "low") == ()


def test_a_medium_us_release_gets_its_one_reminder_15_minutes_ahead(tmp_path):
    # 23 Oct 2026: the 13:45 flash PMIs (medium) broke Gold's rising line a minute later. USD is
    # covered at medium, so they are reminded of at 13:30; a medium EUR release is not covered
    ev = calendar(rel("2026-10-23 13:45", "Flash Manufacturing PMI", impact="medium"),
                  rel("2026-10-23 13:45", "Flash Services PMI", impact="medium"),
                  rel("2026-10-23 13:45", "German Thing", cur="EUR", impact="medium"))
    r = Rig(tmp_path, events=ev, news_cfg=dict(remind_before={"high": (60, 15), "medium": (15,)}))
    sent = []
    for t in pd.date_range("2026-10-23 12:30", "2026-10-23 13:50", freq="1min", tz="UTC"):
        n = len(r.mail.sent)
        tick(r.cfg, r.state, r.mail, r.feed, now=t, news=r.news)
        sent += [t] * (len(r.mail.sent) - n)
    assert sent == [pd.Timestamp("2026-10-23 13:30", tz="UTC")]
    assert "Flash Manufacturing PMI" in r.mail.sent[0][0] and "in 15 min" in r.mail.sent[0][0]
    assert "German" not in r.mail.sent[0][1]


# ── reminders ───────────────────────────────────────────────────────────────────────────────
def test_reminders_go_once_each_and_never_after_the_release(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")))
    for m in ("11:29", "11:30", "11:31", "12:14", "12:15", "12:16", "12:29", "12:31", "13:00"):
        r.tick(f"2026-10-14 {m}")
    s = r.subjects()
    assert len(s) == 2
    assert "USD CPI m/m in 1 h" in s[0] and "in 15 min" in s[1] and "⚡" in s[0]


def test_after_an_outage_only_the_latest_reminder_still_ahead_goes(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")))
    r.tick("2026-10-14 12:20")
    r.tick("2026-10-14 12:21")
    assert len(r.subjects()) == 1 and "in 10 min" in r.subjects()[0]
    assert [x["status"] for x in r.rows(tf="T-60")] == ["skipped"]


def test_a_monitor_started_after_the_release_sends_no_reminder(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")))
    r.tick("2026-10-14 12:31")
    assert r.subjects() == []


def test_releases_at_the_same_minute_share_one_reminder_and_levels_decide(tmp_path):
    ev = calendar(rel("2026-10-14 12:30"), rel("2026-10-14 12:30", "Core CPI m/m"),
                  rel("2026-10-14 12:30", "Unemployment Claims", impact="medium"),
                  rel("2026-10-14 14:00", "ISM Services PMI"),                     # high: 15 only
                  rel("2026-10-14 15:00", "Some Medium Thing", impact="medium"),  # none
                  rel("2026-10-14 12:30", "German Thing", cur="EUR", impact="medium"))  # not covered
    r = Rig(tmp_path, events=ev)
    for t in pd.date_range("2026-10-14 11:00", "2026-10-14 15:30", freq="1min", tz="UTC"):
        tick(r.cfg, r.state, r.mail, r.feed, now=t, news=r.news)
    s = r.subjects()
    assert len(s) == 3, s
    assert "CPI m/m +2 in 1 h" in s[0]                       # one email, all three titles
    assert "Core CPI m/m" in r.mail.sent[0][1] and "Unemployment Claims" in r.mail.sent[0][1]
    assert "German" not in "".join(x[1] for x in r.mail.sent)
    assert "ISM Services PMI in 15 min" in s[2]


# ── results ─────────────────────────────────────────────────────────────────────────────────
def test_a_result_says_what_the_number_means_for_the_currency(tmp_path):
    ev = calendar(rel("2026-10-14 12:30", actual="0.5%", forecast="0.3%"),
                  rel("2026-10-14 12:30", "Unemployment Rate", actual="4.4%", forecast="4.2%"))
    r = Rig(tmp_path, events=ev, news_cfg=dict(remind_before={}))
    r.tick("2026-10-14 12:33")
    r.tick("2026-10-14 12:40")
    assert len(r.mail.sent) == 1
    subject, text, _ = r.mail.sent[0]
    assert "0.5% vs 0.3%" in subject
    assert "better for USD" in text and "worse for USD" in text


def test_results_wait_poll_and_send_a_partial_set_only_after_it_settles(tmp_path):
    ev = calendar(rel("2026-10-14 12:30", actual="0.5%"), rel("2026-10-14 12:30", "Core CPI m/m"))
    r = Rig(tmp_path, events=ev, news_cfg=dict(remind_before={}))
    r.tick("2026-10-14 12:33")
    assert r.subjects() == [] and r.news.polls == [at("2026-10-14 12:33")]
    r.tick("2026-10-14 12:51")                               # 21 min: send what there is
    assert len(r.subjects()) == 1


def test_results_need_a_source_and_never_cover_old_releases(tmp_path):
    ev = calendar(rel("2026-10-14 12:30", actual="0.5%"))
    r = Rig(tmp_path, events=ev, news_cfg=dict(remind_before={}), results=False)
    r.tick("2026-10-14 12:33")
    assert r.subjects() == []
    r2 = Rig(tmp_path / "b", events=ev, news_cfg=dict(remind_before={}))
    r2.tick("2026-10-14 14:00")                              # an hour and a half late
    assert r2.subjects() == []


def test_a_late_number_goes_only_if_it_was_awaited(tmp_path):
    # FRED can post an hour after the release: a number that arrives 90 min late still goes,
    # because this monitor was waiting for it -- but the same number found after the PC was off
    # does not (the test above)
    ev = calendar(rel("2026-10-14 12:30"))
    r = Rig(tmp_path, events=ev, news_cfg=dict(remind_before={}))
    r.tick("2026-10-14 12:35")
    r.tick("2026-10-14 13:30")
    assert r.subjects() == [] and len(r.news.polls) == 2
    assert list(r.news.pending[-1]["title"]) == ["CPI m/m"]  # what to ask the agencies for
    r.news.ev = calendar(rel("2026-10-14 12:30", actual="0.5%"))
    r.tick("2026-10-14 14:00")
    assert len(r.subjects()) == 1 and "0.5%" in r.subjects()[0]
    r.tick("2026-10-14 14:05")
    assert len(r.subjects()) == 1                            # once


def test_a_release_no_source_answers_does_not_hold_up_its_group(tmp_path):
    # ISM has no free official feed: CPI goes as soon as IT is in, not 20 min later
    ev = calendar(rel("2026-10-14 12:30", actual="0.5%"),
                  rel("2026-10-14 12:30", "ISM Thing"))
    r = Rig(tmp_path, events=ev, news_cfg=dict(remind_before={}), unresolvable={"ISM Thing"})
    r.tick("2026-10-14 12:33")
    assert len(r.subjects()) == 1 and r.news.polls == []
    # resolvable, it waits for the settle
    r2 = Rig(tmp_path / "b", events=ev, news_cfg=dict(remind_before={}))
    r2.tick("2026-10-14 12:33")
    assert r2.subjects() == []


# ── digests ─────────────────────────────────────────────────────────────────────────────────
def test_the_day_ahead_goes_once_in_its_window_and_waits_for_the_calendar(tmp_path):
    ev = calendar(rel("2026-10-15 12:30"), rel("2026-10-15 07:00", "GDP m/m", cur="EUR"))
    # 20:00 Karachi = 15:00 UTC; tomorrow there = 15 Oct
    r = Rig(tmp_path, events=ev, news_cfg=dict(day_ahead="20:00", remind_before={}),
            covered_until="2026-10-18 04:00")
    r.tick("2026-10-14 14:59")
    r.tick("2026-10-14 15:00")
    r.tick("2026-10-14 18:00")
    assert len(r.subjects()) == 1 and "Tomorrow, Thu 15 Oct -- 2 releases" in r.subjects()[0]
    # too late: past the 6 h window, nothing at all
    r2 = Rig(tmp_path / "b", events=ev, news_cfg=dict(day_ahead="20:00", remind_before={}),
             covered_until="2026-10-18 04:00")
    r2.tick("2026-10-14 21:01")
    assert r2.subjects() == []
    # the calendar does not reach tomorrow yet: wait, then send with a warning after 5 h
    r3 = Rig(tmp_path / "c", events=ev, news_cfg=dict(day_ahead="20:00", remind_before={}),
             covered_until="2026-10-14 04:00")
    r3.tick("2026-10-14 15:00")
    assert r3.subjects() == []
    r3.tick("2026-10-14 20:00")
    assert len(r3.subjects()) == 1 and "not complete" in r3.mail.sent[0][1]


def test_the_week_ahead_lists_only_high_and_critical(tmp_path):
    ev = calendar(rel("2026-10-14 12:30"), rel("2026-10-13 14:00", "Philly", impact="medium"),
                  rel("2026-10-12 08:00", "ECB Thing", cur="EUR"))
    # Sunday 11 Oct 20:00 Karachi = 15:00 UTC
    r = Rig(tmp_path, events=ev, news_cfg=dict(week_ahead="Sun 20:00", remind_before={}),
            covered_until="2026-10-18 04:00")
    r.tick("2026-10-11 15:00")
    r.tick("2026-10-11 16:00")
    assert len(r.subjects()) == 1 and "2 releases" in r.subjects()[0]
    assert "Philly" not in r.mail.sent[0][1]


# ── a moved release, a stale calendar ───────────────────────────────────────────────────────
def test_a_moved_or_dropped_release_is_reported(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30"), rel("2026-10-14 14:00", "ISM"),
                                      rel("2026-10-14 16:00", "Downgraded")),
            news_cfg=dict(remind_before={}))
    r.tick("2026-10-13 10:00")
    assert r.subjects() == []
    r.news.ev = calendar(rel("2026-10-14 13:30"), rel("2026-10-14 16:00", "Downgraded",
                                                      impact="medium"))
    r.tick("2026-10-13 11:00")
    text = "\n".join(t for _, t, _ in r.mail.sent)
    assert "CPI m/m MOVED" in text
    assert "ISM" in text and "no longer on the calendar" in text
    assert "Downgraded" not in text                          # still there, merely medium now
    n = len(r.mail.sent)
    r.tick("2026-10-13 12:00")
    assert len(r.mail.sent) == n


def test_a_calendar_that_stopped_updating_is_reported_once_a_day(tmp_path):
    r = Rig(tmp_path, age=pd.Timedelta(hours=7))
    r.tick("2026-10-13 10:00")
    r.tick("2026-10-13 11:00")
    assert len(r.subjects()) == 1 and "not updating" in r.subjects()[0]
    r.tick("2026-10-14 10:00")
    assert len(r.subjects()) == 2


# ── with the candle alerts ──────────────────────────────────────────────────────────────────
SPIKE = dict(name="spike", kind="spike", tfs="5m", priority="urgent", markets="all")


def test_a_candle_alert_near_high_news_says_so(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), alerts=[SPIKE],
            spikes=["2026-10-14 12:10"], news_cfg=dict(remind_before={}))
    r.tick("2026-10-14 11:00")
    r.tick("2026-10-14 12:17")
    spike = r.rows(alert="spike")
    assert len(spike) == 1
    assert json.loads(spike[0]["detail"])["note"] == "news: USD CPI m/m in 15 min"
    assert "news: USD CPI m/m in 15 min" in r.mail.sent[-1][1]


def test_a_news_failure_never_stops_a_candle_alert(tmp_path):
    r = Rig(tmp_path, alerts=[SPIKE], spikes=["2026-10-14 12:10"])

    def boom(now):
        raise RuntimeError("calendar exploded")
    r.news.refresh = boom
    r.tick("2026-10-14 11:00")
    rep = r.tick("2026-10-14 12:17")
    assert "calendar exploded" in rep.errors["news"]
    assert len(r.rows(alert="spike")) == 1 and len(r.mail.sent) == 1


def test_a_result_poll_waits_for_the_candle_emails_and_its_number_goes_the_same_tick(tmp_path):
    # 14 Oct 2026: polling the agencies at the release took 40 s, and it ran BEFORE the candles
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), alerts=[SPIKE],
            spikes=["2026-10-14 12:30"], news_cfg=dict(remind_before={}))
    r.tick("2026-10-14 12:00")
    seen = []

    def poll(now, pending=None):
        seen.append(len(r.mail.sent))
        r.news.ev = calendar(rel("2026-10-14 12:30", actual="0.5%"))   # the number lands
        return {}
    r.news.poll_results = poll
    r.tick("2026-10-14 12:37")
    assert seen == [1]                     # the spike's email had gone before the poll began
    assert len(r.mail.sent) == 2 and "0.5%" in r.mail.sent[1][0]


def test_slow_news_fetches_wait_until_the_candle_emails_are_out(tmp_path):
    r = Rig(tmp_path, alerts=[SPIKE], spikes=["2026-10-14 12:10"])
    seen = []

    def background(now):
        seen.append(len(r.mail.sent))
        raise RuntimeError("FRED is down")
    r.news.background = background
    r.tick("2026-10-14 11:00")
    rep = r.tick("2026-10-14 12:17")
    assert seen == [0, 1]                  # the spike's email had gone before the fetch began
    assert "FRED is down" in rep.errors["news background"] and len(r.mail.sent) == 1


def test_news_and_signals_on_one_tick_share_one_email(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), alerts=[SPIKE],
            spikes=["2026-10-14 12:05"])
    r.tick("2026-10-14 11:00")
    r.tick("2026-10-14 12:15")
    subject, text, html = r.mail.sent[-1]
    assert "+1 news" in subject and text.startswith("NEWS")
    assert html.index(">News</h3>") < html.index("spike")


# ── on a 15-minute clock (DELIVERY every_minutes) ───────────────────────────────────────────
def test_on_a_clock_each_reminder_rides_a_boundary_email_early_never_late(tmp_path):
    # 12:40 is off the quarter hour: its 60 / 30 / 15 minute reminders fall at 11:40 / 12:10 /
    # 12:25, and each goes in the email of the boundary BEFORE that -- never after it
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:40")),
            news_cfg=dict(remind_before={"high": (60, 30, 15)}),
            delivery=dict(every_minutes=15))
    sent = []
    for t in pd.date_range(at("2026-10-14 11:20"), at("2026-10-14 12:45"), freq="1min"):
        n = len(r.mail.sent)
        r.tick(f"{t:%Y-%m-%d %H:%M}")
        sent += [t] * (len(r.mail.sent) - n)
    assert sent == [at("2026-10-14 11:32"), at("2026-10-14 12:02"), at("2026-10-14 12:17")]
    assert all("USD CPI m/m in" in s for s in r.subjects())
    assert [x["status"] for x in r.rows(what="reminder")] == ["sent"] * 3


def test_a_day_of_reminders_is_seven_emails_and_all_before_the_release(tmp_path):
    offs = (1440, 720, 360, 120, 60, 30, 15)
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")),
            news_cfg=dict(remind_before={"high": offs}), delivery=dict(every_minutes=15))
    for t in pd.date_range(at("2026-10-13 12:00"), at("2026-10-14 12:45"), freq="5min"):
        r.tick(f"{t:%Y-%m-%d %H:%M}")
    rows = sorted(r.rows(what="reminder"), key=lambda x: int(x["tf"][2:]), reverse=True)
    assert [x["tf"] for x in rows] == [f"T-{k}" for k in offs]
    assert len(r.mail.sent) == 7 and all(x["status"] == "sent" for x in rows)
    assert all(pd.Timestamp(x["created"]) < at("2026-10-14 12:30") for x in rows)


def test_news_goes_last_when_placed_after_the_other_sections(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), alerts=[SPIKE],
            spikes=["2026-10-14 12:05"], categories=("Alerts", "News"))
    r.tick("2026-10-14 11:00")
    r.tick("2026-10-14 12:15")
    subject, text, html = r.mail.sent[-1]
    assert "+1 news" in subject and not text.startswith("NEWS")
    assert text.index("spike") < text.index("NEWS") < text.index("-- quant alerts")
    assert html.index("spike") < html.index(">News</h3>")


def test_on_a_clock_a_result_polled_at_the_boundary_rides_in_that_email(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), alerts=[SPIKE],
            spikes=["2026-10-14 12:30"], news_cfg=dict(remind_before={}),
            delivery=dict(every_minutes=15))
    r.tick("2026-10-14 12:00")

    def poll(now, pending=None):
        r.news.ev = calendar(rel("2026-10-14 12:30", actual="0.5%"))   # the number lands
        return {}
    r.news.poll_results = poll
    r.tick("2026-10-14 12:47")               # the 12:45 boundary: candles, then the poll, then ONE
    assert len(r.mail.sent) == 1             # email with both
    subject, text, _ = r.mail.sent[0]
    assert "spike" in text and "0.5%" in text


def test_an_instant_alert_does_not_wait_for_the_results_poll_nor_take_its_boundary(tmp_path):
    # a 5m break closing at 12:35 while the 12:30 CPI result is awaited: the break goes first, at
    # 12:35:20 BEFORE the poll (which may take its whole 25 s budget), and the number the poll
    # brings still goes in that tick -- the instant email did not spend the 12:30 boundary
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")),
            alerts=[dict(SPIKE, priority="instant")], spikes=["2026-10-14 12:30"],
            news_cfg=dict(remind_before={}), delivery=dict(every_minutes=15))
    r.tick("2026-10-14 12:00")
    order = []

    def poll(now, pending=None):
        order.append(("poll", len(r.mail.sent)))
        r.news.ev = calendar(rel("2026-10-14 12:30", actual="0.5%"))
        return {}
    r.news.poll_results = poll
    r.tick("2026-10-14 12:35:20")
    assert order == [("poll", 1)]                   # one email out before the poll began
    assert len(r.mail.sent) == 2
    (s1, t1, _), (s2, t2, _) = r.mail.sent
    assert "spike" in t1 and "0.5%" not in t1
    assert "0.5%" in t2 and "spike" not in t2


@pytest.mark.parametrize("minutes, says", [
    (15, "15 min"), (60, "1 h"), (150, "2 h 30 min"), (720, "12 h"), (1440, "1 day"),
    (1560, "1 day 2 h"), (2880, "2 days"), (0, "0 min"),
])
def test_a_wait_reads_in_days_hours_and_minutes(minutes, says):
    from quant.alerts.news import span
    assert span(minutes) == says


def test_a_day_ahead_reminder_says_so_in_its_subject(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")),
            news_cfg=dict(remind_before={"high": (1440, 720)}))
    r.tick("2026-10-13 12:30")
    r.tick("2026-10-14 00:30")
    s = r.subjects()
    assert "USD CPI m/m in 1 day" in s[0] and "USD CPI m/m in 12 h" in s[1]


# ── the tiered subject (DELIVERY subject="tiers") ───────────────────────────────────────────
@pytest.mark.parametrize("cur, title, k, want", [
    ("USD", "CPI m/m", 15, 1),                          # critical
    ("USD", "ISM Services PMI", 15, 1),                 # high, and US
    ("USD", "ISM Services PMI", 120, 2),
    ("USD", "CPI m/m", 1440, 3),                        # a day out: for the record
    ("GBP", "CPI y/y", 30, 1),                          # critical anywhere
    ("EUR", "German ZEW Economic Sentiment", 15, 2),    # high, not US
    ("EUR", "German ZEW Economic Sentiment", 120, 3),
])
def test_a_news_reminder_is_tiered_by_level_currency_and_nearness(tmp_path, cur, title, k, want):
    from quant.alerts.compose import TIER_TAG, tier
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30", title=title, cur=cur)),
            news_cfg=dict(currencies={"USD": "medium", "EUR": "high", "GBP": "high"},
                          remind_before={"high": (k,)}),
            delivery=dict(subject="tiers"))
    t = at("2026-10-14 12:30") - pd.Timedelta(minutes=k)
    r.tick(f"{t:%Y-%m-%d %H:%M}")
    (row,) = r.rows(what="reminder")
    assert tier(row) == want
    assert r.subjects()[0].startswith(f"[Alerts] {TIER_TAG[want]} · {cur} {title} in ")


# ── results from headlines (quant/news/headlines.py) ───────────────────────────────────────
HL = dict(remind_before={}, headlines=("financialjuice", "financialjuice_telegram",
                                       "walterbloomberg"), instant=True)


def hl(t, text):
    return dict(id=f"{t}|{text[:12]}", title=text, link="", published=at(t).isoformat())


CPI_LINE = hl("2026-10-14 12:30:51", "FinancialJuice: US CPI Actual 0.5% (Forecast 0.3%, "
                                     "Previous 0.2%)")


@pytest.mark.parametrize("instant, sent_at", [(True, "2026-10-14 12:31:20"),
                                              (False, "2026-10-14 12:32:20")])
def test_a_headline_result_goes_at_once_when_instant_and_asks_the_next_source_only_if_needed(
        tmp_path, instant, sent_at):
    # CPI at 12:30, FinancialJuice types it 51 s later: the 12:30:20 look finds nothing and asks
    # the backups too; the 12:31:20 look finds it and asks no further. Instant, it goes then;
    # otherwise it rides the 12:30 boundary's email, which waits for the bars (settle 90 s)
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), news_cfg=dict(HL, instant=instant),
            delivery=dict(every_minutes=5))
    r.tick("2026-10-14 12:00")
    r.tick("2026-10-14 12:30:20")
    assert r.news.asked == ["financialjuice", "financialjuice_telegram", "walterbloomberg"]
    r.news.lines = {"financialjuice": [CPI_LINE]}
    sent = []
    for t in ("2026-10-14 12:31:20", "2026-10-14 12:32:20", "2026-10-14 12:35:20"):
        n = len(r.mail.sent)
        r.tick(t)
        sent += [t] * (len(r.mail.sent) - n)
    assert sent == [sent_at]
    assert r.news.asked[3:] == ["financialjuice"]           # found: no backup, and never again
    subject, text, _ = r.mail.sent[0]
    assert "USD CPI m/m: US CPI Actual 0.5% (Forecast 0.3%, Previous 0.2%)" in subject
    assert "+51 s  FinancialJuice: US CPI Actual 0.5%" in text
    assert "calendar forecast: CPI m/m 0.3%" in text and "not the official number" in text
    (row,) = r.rows(tf="headline")
    assert row["priority"] == ("instant" if instant else "urgent") and row["status"] == "sent"


def test_a_failing_source_hands_over_to_the_next_and_a_release_goes_once(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), news_cfg=HL,
            lines={"financialjuice": IOError("FinancialJuice: HTTP 503"),
                   "financialjuice_telegram": [hl("2026-10-14 12:31:02", "‼ BREAKING: US CPI "
                                                  "ACTUAL 0.5% (FORECAST 0.3%) $MACRO|FJ")]})
    r.tick("2026-10-14 12:00")
    rep = r.tick("2026-10-14 12:31:20")
    assert r.news.asked == ["financialjuice", "financialjuice_telegram"]
    assert "HTTP 503" in rep.errors["news headlines financialjuice"]
    (subject, text, _), = r.mail.sent
    assert "+62 s  FinancialJuice (Telegram): US CPI ACTUAL 0.5% (FORECAST 0.3%)" in text
    r.tick("2026-10-14 12:32:20")
    assert len(r.news.asked) == 2 and len(r.mail.sent) == 1


def test_only_a_covered_release_in_its_window_with_words_is_asked_about(tmp_path):
    ev = calendar(rel("2026-10-14 12:30", "German Ifo Business Climate", cur="EUR"),  # no words
                  rel("2026-10-14 12:30", "Cash Rate", cur="AUD"),                    # not covered
                  rel("2026-10-14 12:30", "CPI m/m", impact="low"),                   # below results
                  rel("2026-10-14 13:00"))                                            # not yet out
    r = Rig(tmp_path, events=ev, news_cfg=HL)
    r.tick("2026-10-14 12:31")
    assert r.news.asked == []
    r2 = Rig(tmp_path / "b", events=calendar(rel("2026-10-14 12:30")), news_cfg=HL)
    r2.tick("2026-10-14 12:41")                             # inside 10 min (+1 for the last look)
    r2.tick("2026-10-14 12:42")                             # past it
    assert r2.news.asked == ["financialjuice", "financialjuice_telegram", "walterbloomberg"]


@pytest.mark.parametrize("said, mailed, note", [
    ("US CPI MoM Actual 0.5% (Forecast 0.3%)", False, None),
    ("US CPI MoM Actual 0.4% (Forecast 0.3%)", True, "it differs from the headline"),
    ("US CPI hotter than expected, core sticky", True, "(the official number)"),
])
def test_the_official_number_is_mailed_only_when_the_headline_did_not_print_it(
        tmp_path, said, mailed, note):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), news_cfg=HL,
            delivery=dict(every_minutes=5),
            lines={"financialjuice": [hl("2026-10-14 12:30:04", "FinancialJuice: " + said)]})
    r.tick("2026-10-14 12:00")
    r.tick("2026-10-14 12:30:20")
    assert len(r.mail.sent) == 1 and said in r.mail.sent[0][1]
    r.news.ev = calendar(rel("2026-10-14 12:30", actual="0.5%"))     # the agencies' number lands
    r.tick("2026-10-14 12:33:20")                                    # off the clock: instant
    (official,) = r.rows(tf="result")
    assert official["status"] == ("sent" if mailed else "skipped")
    assert len(r.mail.sent) == 1 + mailed
    if mailed:
        assert "0.5% vs 0.3%" in r.mail.sent[1][0] and note in r.mail.sent[1][1]


def test_an_official_number_that_came_first_leaves_no_headline_to_send(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30", actual="0.5%")), news_cfg=HL,
            lines={"financialjuice": [CPI_LINE]})
    r.tick("2026-10-14 12:00")
    r.tick("2026-10-14 12:31:20")
    assert [x["tf"] for x in r.rows(what="result")] == ["result"] and r.news.asked == []
    assert len(r.mail.sent) == 1


def test_instant_news_reminders_carry_the_instant_priority(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")),
            news_cfg=dict(HL, remind_before={"high": (15,)}))
    r.tick("2026-10-14 12:15")
    (row,) = r.rows(what="reminder")
    assert row["priority"] == "instant" and len(r.mail.sent) == 1


def test_the_monitor_reads_a_headline_source_through_its_shared_connections(tmp_path, monkeypatch):
    from quant.alerts import feeds
    from quant.alerts.news import LiveNews
    from quant.news import headlines as H
    calls = []

    class Resp:
        status_code = 200
        content = (b"<rss><channel><item><title>US CPI Actual 0.5%</title><guid>9</guid>"
                   b"<pubDate>Wed, 14 Oct 2026 12:30:04 GMT</pubDate></item></channel></rss>")

    class Session:
        def get(self, url, headers=None, timeout=None):
            calls.append((url, timeout))
            return Resp()
    monkeypatch.setattr(feeds, "session", lambda: Session())
    got = LiveNews(tmp_path / "news").headlines("financialjuice")
    assert got[0]["title"] == "US CPI Actual 0.5%" and got[0]["id"] == "9"
    assert calls == [(H.SOURCES["financialjuice"].url, H.TIMEOUT)]


def test_the_trendlines_config_takes_headline_results_instantly():
    from pathlib import Path
    from quant.news.headlines import DEFAULT_ORDER
    cfg = load_config(Path(__file__).resolve().parents[1] / "alerts" / "alerts_config_trendlines.py")
    assert cfg.news.headlines == DEFAULT_ORDER and cfg.news.instant is True
    assert cfg.news.breaking == ("financialjuice", "financialjuice_telegram")


def test_a_headline_subject_quotes_the_line_about_its_most_important_release(tmp_path):
    # claims (high) and GDP (critical) at 12:30: FinancialJuice types claims first, but the
    # subject leads with GDP, the group's most important title, and quotes GDP's own line
    ev = calendar(rel("2026-10-14 12:30", "Unemployment Claims", forecast="225K"),
                  rel("2026-10-14 12:30", "Advance GDP q/q", forecast="3.2%"))
    r = Rig(tmp_path, events=ev, news_cfg=HL, lines={"financialjuice": [
        hl("2026-10-14 12:30:03", "FinancialJuice: US Initial Jobless Claims Actual 220K "
                                  "(Forecast 225K)"),
        hl("2026-10-14 12:30:05", "FinancialJuice: US GDP Advance Actual 3.4% (Forecast 3.2%)")]})
    r.tick("2026-10-14 12:00")
    r.tick("2026-10-14 12:30:20")
    (subject, text, _), = r.mail.sent
    assert "USD Advance GDP q/q: US GDP Advance Actual 3.4% (Forecast 3.2%)" in subject
    assert text.index("Jobless Claims") < text.index("US GDP Advance")    # lines keep their order


# ── breaking news, not on the calendar (quant/news/headlines.py tiers) ─────────────────────
BR = dict(remind_before={}, breaking=("financialjuice", "financialjuice_telegram"), instant=True)
DEPARTURE = "FinancialJuice: US Officials: US State Dept is preparing to issue an ordered departure"
EMBASSY = "FinancialJuice: US Embassy posts alert over Houthis attacks against Saudi Arabia"


def _a_1m(r, end, jumps=None, start="2026-10-20 00:00"):
    """Market A's 1m candles in the monitor's own store up to `end`: calm, but for `jumps`."""
    t = pd.date_range(at(start), at(end), freq="1min", tz="UTC", inclusive="left")
    ret = np.random.default_rng(1).normal(0, 0.0001, len(t))
    for when, x in (jumps or {}).items():
        ret[t.get_indexer([at(when)])] = x
    c = 4000 * np.exp(np.cumsum(ret))
    r.state.save_bars("A", pd.DataFrame({TIME_COL: t, "open": c, "high": c, "low": c, "close": c,
                                         "volume": 1.0}), base="1m")


def test_a_top_breaking_line_goes_at_once_and_once(tmp_path):
    r = Rig(tmp_path, news_cfg=BR, delivery=dict(every_minutes=5))
    r.tick("2026-10-21 18:40:20")
    r.news.lines = {"financialjuice": [hl("2026-10-21 18:47:50", DEPARTURE)]}
    r.tick("2026-10-21 18:48:20")                             # off the clock: instant
    (subject, text, _), = r.mail.sent
    assert subject.startswith("[Alerts]") and "BREAKING: US Officials: US State Dept" in subject
    assert "FinancialJuice:" in text and "ordered departure" in text
    r.tick("2026-10-21 18:49:20")
    assert len(r.mail.sent) == 1                              # once
    (row,) = r.rows(tf="breaking")
    from quant.alerts.compose import tier
    assert row["priority"] == "instant" and tier(row) == 1


def test_a_breaking_line_waits_for_a_sharp_move_and_goes_with_it(tmp_path):
    r = Rig(tmp_path, news_cfg=BR, lines={"financialjuice": [hl("2026-10-21 18:34:15", EMBASSY)]})
    _a_1m(r, "2026-10-21 18:35")
    r.tick("2026-10-21 18:35:20")
    assert r.subjects() == []                                 # no move yet: it waits
    _a_1m(r, "2026-10-21 18:50", {"2026-10-21 18:44": -0.008})
    r.tick("2026-10-21 18:50:20")
    (subject, text, _), = r.mail.sent
    assert "BREAKING: US Embassy posts alert" in subject and "· A -0.7" in subject
    assert "in 5 min from" in text


def test_a_breaking_line_whose_move_never_comes_is_dropped(tmp_path):
    r = Rig(tmp_path, news_cfg=BR, lines={"financialjuice": [hl("2026-10-21 18:34:15", EMBASSY)]})
    for t in ("18:35", "18:51", "18:55"):
        _a_1m(r, f"2026-10-21 {t}")
        r.tick(f"2026-10-21 {t}:20")
    assert json.loads(r.state.get("news_breaking_waiting")) == []     # given up at 18:54
    _a_1m(r, "2026-10-21 19:06", {"2026-10-21 19:00": -0.008})       # a move, too late
    r.tick("2026-10-21 19:06:20")
    assert r.subjects() == []


def test_an_old_line_a_repeat_and_talk_are_not_sent(tmp_path):
    r = Rig(tmp_path, news_cfg=BR, lines={"financialjuice": [
        hl("2026-10-21 18:20", DEPARTURE),                    # 28 min old at first sight
        hl("2026-10-21 18:47", "FinancialJuice: Iran could close the Strait of Hormuz"),
        hl("2026-10-21 18:47:30", "FinancialJuice: Fed announces emergency rate cut")]})
    r.tick("2026-10-21 18:48:20")
    assert [s for s in r.subjects() if "BREAKING" in s] == [
        r.subjects()[0]] and "emergency rate cut" in r.subjects()[0]
    r.news.lines["financialjuice"].append(hl("2026-10-21 18:55", "FinancialJuice: Fed announces "
                                                                  "an emergency rate cut of 50 bp"))
    r.tick("2026-10-21 18:55:20")
    assert len(r.mail.sent) == 1                              # the same story within the hour


def test_a_failing_source_hands_breaking_news_to_the_next(tmp_path):
    r = Rig(tmp_path, news_cfg=BR, lines={"financialjuice": IOError("FinancialJuice: HTTP 429"),
                                          "financialjuice_telegram": [hl("2026-10-21 18:48:14",
                                                                         DEPARTURE + "|FJ")]})
    rep = r.tick("2026-10-21 18:48:20")
    assert r.news.asked == ["financialjuice", "financialjuice_telegram"]
    assert "HTTP 429" in rep.errors["news breaking financialjuice"]
    assert "FinancialJuice (Telegram):" in r.mail.sent[0][1]


def test_no_more_than_eight_breaking_alerts_a_day_and_none_when_off(tmp_path):
    tops = ["Fed announces emergency rate cut", "Japan MOF: intervened in the FX market",
            "US troops and diplomats evacuating Middle East posts", "Martial law declared in Seoul",
            "Coup attempt under way in Ankara", "ECB holds emergency meeting on markets",
            "Nuclear test detected near Punggye-ri", "Ordered departure for embassy staff in Doha",
            "Embassy in Tel Aviv closed until further notice"]
    r = Rig(tmp_path, news_cfg=BR, lines={"financialjuice": [
        hl(f"2026-10-21 18:{40 + i}:00", s) for i, s in enumerate(tops)]})
    r.tick("2026-10-21 18:49:20")
    st = [x["status"] for x in r.rows(tf="breaking")]
    assert st.count("sent") == 8 and st.count("skipped") == 1
    off = Rig(tmp_path / "off", news_cfg=dict(remind_before={}),
              lines={"financialjuice": [hl("2026-10-21 18:47:50", DEPARTURE)]})
    off.tick("2026-10-21 18:48:20")
    assert off.news.asked == [] and off.subjects() == []


def test_the_results_and_the_breaking_news_share_one_download_a_tick(tmp_path, monkeypatch):
    from quant.alerts import feeds
    from quant.alerts.news import LiveNews
    calls = []

    class Resp:
        status_code = 200
        content = (b"<rss><channel><item><title>US CPI MoM Actual 0.5%</title><guid>9</guid>"
                   b"<pubDate>Wed, 14 Oct 2026 12:30:04 GMT</pubDate></item></channel></rss>")

    class Session:
        def get(self, url, headers=None, timeout=None):
            calls.append(url)
            return Resp()
    monkeypatch.setattr(feeds, "session", lambda: Session())
    monkeypatch.setattr("quant.news.update.update_feed", lambda *a, **k: "fresh")
    n = LiveNews(tmp_path / "news")
    n.refresh(at("2026-10-14 12:30:20"))
    n.headlines("financialjuice")
    n.headlines("financialjuice")
    assert len(calls) == 1                                    # FinancialJuice: 429 on a second
    n.refresh(at("2026-10-14 12:31:20"))
    n.headlines("financialjuice")
    assert len(calls) == 2                                    # the next tick asks again


def test_every_headline_download_the_monitor_makes_is_kept_in_its_archive(tmp_path, monkeypatch,
                                                                          caplog):
    import sqlite3
    from quant.alerts import feeds
    from quant.alerts.news import LiveNews
    answers = [(200, b"<rss><channel><item><title>US CPI MoM Actual 0.5%</title><guid>9</guid>"
                     b"<pubDate>Wed, 14 Oct 2026 12:30:04 GMT</pubDate></item></channel></rss>"),
               (429, b""),
               (200, b"<rss><channel><item><title>US CPI MoM Actual 0.5%</title><guid>9</guid>"
                     b"<pubDate>Wed, 14 Oct 2026 12:30:04 GMT</pubDate></item><item><title>"
                     b"Iran says talks are over</title><guid>10</guid><pubDate>Wed, 14 Oct 2026 "
                     b"12:31:50 GMT</pubDate></item></channel></rss>")]

    class Session:
        def get(self, url, headers=None, timeout=None):
            status, body = answers.pop(0)
            return type("R", (), dict(status_code=status, content=body))()
    monkeypatch.setattr(feeds, "session", lambda: Session())
    monkeypatch.setattr("quant.news.update.update_feed", lambda *a, **k: "fresh")
    n = LiveNews(tmp_path / "news")
    for tick in ("12:30:20", "12:31:20", "12:32:20"):
        n.refresh(at(f"2026-10-14 {tick}"))
        try:
            n.headlines("financialjuice")
            n.headlines("financialjuice")                       # the tick's cached copy: no row
        except OSError:
            pass
    assert n.archive == tmp_path / "news" / "headlines.sqlite"
    db = sqlite3.connect(str(n.archive))
    assert db.execute("SELECT ok, status, n, new, error FROM polls ORDER BY rowid").fetchall() == [
        (1, 200, 1, 1, None), (0, 429, 0, 0, "OSError: FinancialJuice: HTTP 429"),
        (1, 200, 2, 1, None)]
    assert db.execute("SELECT item_id, title, backlog FROM items ORDER BY rowid").fetchall() == [
        ("9", "US CPI MoM Actual 0.5%", 1), ("10", "Iran says talks are over", 0)]
    db.close()
    # an archive that cannot be written costs nothing but one line in the log
    n2 = LiveNews(tmp_path / "news2")
    n2.archive = tmp_path / "news" / "headlines.sqlite" / "not-a-folder" / "x.sqlite"
    answers[:] = [(200, b"<rss><channel><item><title>A</title><guid>1</guid></item></channel>"
                        b"</rss>")] * 2
    with caplog.at_level("WARNING", logger="quant.alerts"):
        assert [x["title"] for x in n2.headlines("financialjuice")] == ["A"]
        n2.refresh(at("2026-10-14 12:33:20"))
        assert [x["title"] for x in n2.headlines("financialjuice")] == ["A"]
    assert sum("headline archive" in r.getMessage() for r in caplog.records) == 1


# ── 6 Nov 2026: the US jobs report, Canada's beside it, and a release's own move ───────────────
def fj_2026_11_06():
    """FinancialJuice's real lines of 6 Nov 2026 13:20-13:36 UTC (tests/data), read back from the
    server's headline archive: the US jobs report and Canada's, both at 13:30."""
    from pathlib import Path
    rows = (Path(__file__).parent / "data" / "financialjuice_2026-11-06.tsv").read_text(
        encoding="utf-8").splitlines()
    return [dict(id=f"fj{k}", title=s, link="", published=at(f"2026-11-06 {t}").isoformat())
            for k, (t, s) in enumerate(r.split("\t", 1) for r in rows if not r.startswith("#"))]


TANKER = ("FinancialJuice: One seafarer dead in the November 5th explosion on a Greek-operated oil "
          "products tanker in the Gulf, tanker was likely struck by unidentified mi")


def _jobs_day(cur="USD"):   # Forex Factory's own order: earnings, payrolls, the jobless rate
    return [rel("2026-11-06 13:30", "Average Hourly Earnings m/m", cur=cur),
            rel("2026-11-06 13:30", "Non-Farm Employment Change", cur=cur, forecast="110K"),
            rel("2026-11-06 13:30", "Unemployment Rate", cur=cur, forecast="4.3%")]


def test_releases_sharing_a_level_lead_with_the_most_important():
    from quant.alerts.news import _groups
    from quant.news.catalog import CRITICAL, lead_rank
    ev = calendar(*_jobs_day(), rel("2026-11-05 12:00", "Monetary Policy Summary", cur="GBP"),
                  rel("2026-11-05 12:00", "Official Bank Rate", cur="GBP", forecast="4%"),
                  rel("2026-11-06 13:30", "Building Permits", impact="medium"))
    assert [list(g["title"]) for _, _, g in _groups(ev)] == [
        ["Official Bank Rate", "Monetary Policy Summary"],
        ["Non-Farm Employment Change", "Unemployment Rate", "Average Hourly Earnings m/m",
         "Building Permits"]]
    assert lead_rank("USD", "Building Permits") == len(CRITICAL["USD"])


def test_on_jobs_day_the_headline_quotes_the_us_lines_and_leads_with_payrolls(tmp_path):
    r = Rig(tmp_path, events=calendar(*_jobs_day()), news_cfg=HL,
            lines={"financialjuice": fj_2026_11_06()})
    r.tick("2026-11-06 13:00")
    r.tick("2026-11-06 13:30:20")
    (subject, text, _), = r.mail.sent
    assert ("USD Non-Farm Employment Change: US Nonfarm Payrolls Actual 165k (Forecast 70k, "
            "Previous 29k)") in subject
    assert "Canadian" not in text and "US Unemployment Rate Actual 4.1%" in text


@pytest.mark.parametrize("release, sent", [("high", 0), ("medium", 0), ("low", 1), (None, 1)])
def test_a_scheduled_releases_own_move_confirms_no_breaking_news(tmp_path, release, sent):
    # 6 Nov 2026: a 13:22 BREAKING line (a tanker explosion) went out at 13:31:20 on the jobs
    # report's own move. A medium+ release at 13:30 makes that move its result, not news
    ev = (calendar(rel("2026-11-06 13:30", "Non-Farm Employment Change", impact=release))
          if release else calendar())
    r = Rig(tmp_path, news_cfg=BR, events=ev,
            lines={"financialjuice": [hl("2026-11-06 13:22:32", TANKER)]})
    _a_1m(r, "2026-11-06 13:23", start="2026-11-05 00:00")
    r.tick("2026-11-06 13:23:20")
    _a_1m(r, "2026-11-06 13:32", {"2026-11-06 13:30": -0.008}, start="2026-11-05 00:00")
    r.tick("2026-11-06 13:32:20")
    assert len([s for s in r.subjects() if "BREAKING" in s]) == sent


def test_on_jobs_day_all_three_numbers_go_in_the_first_email_and_none_twice(tmp_path):
    # 6 Nov 2026, the real lines: the monthly earnings line is quoted (it was the yearly one),
    # so the agencies' numbers three minutes later add nothing and are not mailed again
    r = Rig(tmp_path, events=calendar(*_jobs_day()), news_cfg=HL, delivery=dict(every_minutes=5),
            lines={"financialjuice": fj_2026_11_06()})
    r.tick("2026-11-06 13:00")
    r.tick("2026-11-06 13:30:20")
    (_, text, _), = r.mail.sent
    assert "US Average Earnings MoM Actual 0.4%" in text and "YoY" not in text
    r.news.ev = calendar(*[dict(e, actual=a) for e, a in zip(_jobs_day(), ("0.4%", "165K", "4.1%"))])
    r.tick("2026-11-06 13:33:20")
    (official,) = r.rows(tf="result")
    assert official["status"] == "skipped" and len(r.mail.sent) == 1


def test_on_cpi_day_the_subject_quotes_the_headline_monthly_line_not_core(tmp_path):
    # Forex Factory lists core first, and so may the desk type it: "CPI m/m" leads the email and
    # must quote its own line, not "Core CPI MoM" typed a second earlier
    ev = calendar(*[rel("2026-11-10 13:30", ti, forecast=f) for ti, f in (
        ("Core CPI m/m", "0.3%"), ("Core CPI y/y", "3.0%"), ("CPI m/m", "0.3%"), ("CPI y/y", "2.8%"))])
    lines = [hl("2026-11-10 13:30:01", "FinancialJuice: US Core CPI MoM Actual 0.3% (Forecast 0.3%)"),
             hl("2026-11-10 13:30:01", "FinancialJuice: US Core CPI YoY Actual 3.0% (Forecast 3.0%)"),
             hl("2026-11-10 13:30:02", "FinancialJuice: US CPI YoY Actual 2.9% (Forecast 2.8%)"),
             hl("2026-11-10 13:30:02", "FinancialJuice: US CPI MoM Actual 0.4% (Forecast 0.3%)")]
    r = Rig(tmp_path, events=ev, news_cfg=HL, lines={"financialjuice": lines})
    r.tick("2026-11-10 13:00")
    r.tick("2026-11-10 13:30:20")
    (subject, text, _), = r.mail.sent
    assert "USD CPI m/m: US CPI MoM Actual 0.4% (Forecast 0.3%)" in subject
    assert all(x["title"].replace("FinancialJuice: ", "") in text for x in lines)   # all four
