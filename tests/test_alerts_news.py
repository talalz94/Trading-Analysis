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
    def __init__(self, tmp_path, news_cfg=None, events=None, alerts=(), spikes=(), **fake):
        tmp_path.mkdir(parents=True, exist_ok=True)
        p = tmp_path / "cfg.py"
        cfg = dict(NEWS, **(news_cfg or {}))
        p.write_text(f"MARKETS = {{'A': dict(symbol='XAUUSD', source='dukascopy')}}\n"
                     f"ALERTS = {list(alerts)!r}\nNEWS = {cfg!r}\n")
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
    assert "USD CPI m/m in 60 min" in s[0] and "in 15 min" in s[1] and "⚡" in s[0]


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
    assert "CPI m/m +2 in 60 min" in s[0]                    # one email, all three titles
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


def test_news_and_signals_on_one_tick_share_one_email(tmp_path):
    r = Rig(tmp_path, events=calendar(rel("2026-10-14 12:30")), alerts=[SPIKE],
            spikes=["2026-10-14 12:05"])
    r.tick("2026-10-14 11:00")
    r.tick("2026-10-14 12:15")
    subject, text, html = r.mail.sent[-1]
    assert "+1 news" in subject and text.startswith("NEWS")
    assert html.index(">News</h3>") < html.index("spike")
