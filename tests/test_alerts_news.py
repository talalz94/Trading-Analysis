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
