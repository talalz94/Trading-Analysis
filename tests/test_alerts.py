"""
quant.alerts -- the monitor must fire every alert ONCE, never on a candle still forming, never mail
the history it was switched on into, and survive outages of the feed, the mail server and itself.

Everything runs on synthetic bars through FakeFeed and ListNotifier, with an explicit `now`, so each
test states exactly what the clock said and what the feed had.
"""
import numpy as np
import pandas as pd
import pytest

from quant.alerts import REGISTRY, ConfigError, Detector, load_config, preview, tick
from quant.alerts.compose import compose
from quant.alerts.config import DEFAULT_CONFIG
from quant.alerts.detectors import CandleDetector
from quant.alerts.engine import final_bars, tf_bars
from quant.alerts.feeds import FakeFeed
from quant.alerts.notify import ListNotifier, NotConfigured, SmtpNotifier
from quant.alerts.runner import acquire_lock
from quant.alerts.store import State
from quant.data.base import TIME_COL
from quant.data.timeframe import resample_ohlcv


class Spike(Detector):
    """Fires on any bar whose high reaches `level` -- a marker the tests plant by hand."""
    kind = "spike"

    def validate(self, p):
        if "level" not in p:
            raise KeyError("spike alerts need level=")

    def warmup_bars(self, p):
        return 5

    def evaluate(self, frame, p):
        return [dict(i=int(i), what="spike", side="bull")
                for i in np.flatnonzero(frame["high"].to_numpy() >= p["level"])]


@pytest.fixture(autouse=True)
def _spike(monkeypatch):
    monkeypatch.setitem(REGISTRY, "spike", Spike())


_OPEN = []


@pytest.fixture(autouse=True)
def _close_states():
    yield
    while _OPEN:
        _OPEN.pop().close()


def bars(spikes=(), start="2026-05-30", end="2026-06-08", stop=None, seed=0):
    t = pd.date_range(start, end, freq="5min", tz="UTC", inclusive="left")
    c = 100 + np.random.default_rng(seed).normal(0, 0.1, len(t)).cumsum()
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({TIME_COL: t, "open": o, "high": np.maximum(o, c) + 0.05,
                       "low": np.minimum(o, c) - 0.05, "close": c, "volume": 1.0})
    for s in spikes:
        df.loc[df[TIME_COL] == pd.Timestamp(s, tz="UTC"), "high"] = 1000.0
    if stop is not None:                             # the market shuts here (a weekend)
        df = df[df[TIME_COL] < pd.Timestamp(stop, tz="UTC")]
    return df.reset_index(drop=True)


def config(tmp_path, alerts, delivery=None, categories=("Spikes", "Other")):
    p = tmp_path / "cfg.py"
    markets = {"A": dict(symbol="AAAUSD", source="dukascopy"),
               "B": dict(symbol="BBBUSDT", source="binance")}
    p.write_text(f"MARKETS = {markets!r}\nCATEGORIES = {categories!r}\n"
                 f"ALERTS = {alerts!r}\nDELIVERY = {delivery or {}!r}\n")
    return load_config(p)


def spike_alert(**kw):
    a = dict(name="spike", category="Spikes", kind="spike", level=999, markets="all",
             tfs=("1h",), priority="normal")
    a.update(kw)
    return a


def at(s):
    return pd.Timestamp(s, tz="UTC")


class Rig:
    def __init__(self, tmp_path, alerts, a=(), b=(), delivery=None, fail=0, **bar_kw):
        self.cfg = config(tmp_path, alerts, delivery)
        self.state = State(tmp_path / "state")
        _OPEN.append(self.state)
        self.feed = FakeFeed({"A": bars(a, **bar_kw), "B": bars(b, seed=1, **bar_kw)})
        self.mail = ListNotifier(fail=fail)

    def tick(self, when):
        return tick(self.cfg, self.state, self.mail, self.feed, now=at(when))

    def events(self):
        return [dict(r) for r in self.state.db.execute("SELECT * FROM events ORDER BY bar_time")]


# ── config ──────────────────────────────────────────────────────────────────────────────────
def test_the_shipped_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert set(cfg.markets) == {"Gold", "USDJPY", "BTC"}
    assert {tf for a in cfg.alerts for tf in a.tfs} == {"15m", "1h", "4h"}


@pytest.mark.parametrize("bad, says", [
    (dict(markets=("A", "Z")), "'Z' is neither a market nor a group"),
    (dict(tfs=("1h", "3h")), "tfs must be"),
    (dict(kind="nope"), "kind must be one of"),
    (dict(priority="asap"), "priority must be"),
    (dict(level=None, kind="candle", patterns=("engulfing_bull", "engulfed")), "engulfed"),
])
def test_a_config_mistake_is_refused_by_name(tmp_path, bad, says):
    with pytest.raises(ConfigError, match=says):
        config(tmp_path, [spike_alert(**bad)])


# ── firing ──────────────────────────────────────────────────────────────────────────────────
def test_the_first_tick_starts_from_now_and_mails_no_history(tmp_path):
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-02 10:20", "2026-06-02 22:20"])
    rep = r.tick("2026-06-03 00:01:30")
    assert len(rep.started) == 2 and rep.events == 0 and not r.mail.sent
    assert r.state.cursor("spike", "A", "1h") == at("2026-06-02 23:00")


def test_an_alert_fires_once_on_its_close_and_never_again(tmp_path):
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-03 00:20"])
    r.tick("2026-06-03 00:01:30")
    for t in ("2026-06-03 00:30", "2026-06-03 00:59", "2026-06-03 01:01:00"):
        assert r.tick(t).events == 0                 # the 00:00 bar is forming, or not settled
    rep = r.tick("2026-06-03 01:01:30")
    assert rep.events == 1 and len(r.mail.sent) == 1
    assert "spike" in r.mail.sent[0][1] and "A " in r.mail.sent[0][1]
    for t in ("2026-06-03 01:02:30", "2026-06-03 01:30", "2026-06-03 02:01:30"):
        r.tick(t)
    assert len(r.events()) == 1 and len(r.mail.sent) == 1


def test_rejudging_a_bar_after_a_crash_cannot_send_it_twice(tmp_path):
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-03 00:20"])
    r.tick("2026-06-03 00:01:30")
    r.tick("2026-06-03 01:01:30")
    with r.state:                                    # as if the cursor had not been saved
        r.state.set_cursor("spike", "A", "1h", at("2026-06-02 23:00"))
    assert r.tick("2026-06-03 01:03").events == 0
    assert len(r.mail.sent) == 1


def test_a_market_is_fetched_only_when_one_of_its_bars_is_due(tmp_path):
    r = Rig(tmp_path, [spike_alert()])
    r.tick("2026-06-03 00:01:30")
    n = len(r.feed.calls)
    for t in ("2026-06-03 00:15", "2026-06-03 00:30", "2026-06-03 01:01:00"):
        r.tick(t)
    assert len(r.feed.calls) == n                    # nothing could have closed
    r.tick("2026-06-03 01:01:30")
    assert len(r.feed.calls) == n + 2                # A and B, once each
    start = r.feed.calls[-1][1]
    assert start >= at("2026-06-02 23:45")           # only the new bars are asked for


def test_a_bar_is_final_only_once_closed_settled_and_complete(tmp_path):
    d = config(tmp_path, [spike_alert()]).delivery                 # settle 90 s, grace 20 min
    b = bars()
    b = b[b[TIME_COL] <= at("2026-06-03 01:25")]                  # even a feed that leaks the
    last = lambda now, upto="2026-06-03 01:25": final_bars(       # noqa  forming 5m bar
        b[b[TIME_COL] <= at(upto)], "1h", at(now), d)[TIME_COL].iloc[-1]
    assert last("2026-06-03 01:00:30", "2026-06-03 00:55") == at("2026-06-02 23:00")  # settling
    assert last("2026-06-03 01:01:30", "2026-06-03 00:55") == at("2026-06-03 00:00")
    assert last("2026-06-03 01:27") == at("2026-06-03 00:00")     # 01:00 bar is still forming
    assert last("2026-06-03 01:05", "2026-06-03 00:45") == at("2026-06-02 23:00")  # 00:50, 00:55
    assert last("2026-06-03 01:20", "2026-06-03 00:45") == at("2026-06-03 00:00")  # never came


def test_a_shut_market_bar_is_judged_after_the_grace_not_before(tmp_path):
    # the market stops at 21:00 Friday: the 20:00-00:00 4h bar never gets its last 5m bar
    r = Rig(tmp_path, [spike_alert(tfs=("4h",), markets=("A",))], a=["2026-06-05 20:30"],
            stop="2026-06-05 21:00")
    r.tick("2026-06-05 20:01:30")
    assert r.tick("2026-06-06 00:01:30").events == 0      # data does not reach 00:00
    assert r.tick("2026-06-06 00:15").events == 0         # and the grace has not run out
    r.feed.fail = {"A"}
    r.tick("2026-06-06 00:21")                           # feed down: nothing is judged
    r.feed.fail = set()
    assert r.events() == []
    assert r.tick("2026-06-06 00:23").events == 1        # it answered, 20+ min past the close


def test_a_shut_market_is_polled_a_few_times_an_hour_not_every_minute(tmp_path):
    r = Rig(tmp_path, [spike_alert(tfs=("15m",), markets=("A",))], stop="2026-06-05 21:00")
    r.tick("2026-06-05 20:01:30")
    n = len(r.feed.calls)
    for t in pd.date_range("2026-06-06 01:00", "2026-06-06 03:00", freq="1min"):
        r.tick(t)
    assert len(r.feed.calls) - n <= 12               # 120 ticks; backing off to every 15 min


# ── delivery ────────────────────────────────────────────────────────────────────────────────
def test_urgent_alerts_on_one_bar_close_arrive_in_one_email(tmp_path):
    r = Rig(tmp_path, [spike_alert(priority="urgent")], a=["2026-06-03 00:20"],
            b=["2026-06-03 00:40"])
    r.tick("2026-06-03 00:01:30")
    r.tick("2026-06-03 01:01:30")
    assert len(r.mail.sent) == 1
    subject, text, _ = r.mail.sent[0]
    assert "2 signals (2 urgent)" in subject and "A, B" in subject


def test_normal_alerts_wait_for_the_digest(tmp_path):
    r = Rig(tmp_path, [spike_alert(tfs=("15m",))], delivery=dict(digest_minutes=30),
            a=["2026-06-03 00:05", "2026-06-03 00:20"])
    r.tick("2026-06-03 00:01:30")
    r.tick("2026-06-03 00:16:30")
    assert len(r.mail.sent) == 1                     # the first is never held back
    r.tick("2026-06-03 00:31:30")
    r.tick("2026-06-03 00:40")
    assert len(r.mail.sent) == 1 and len(r.events()) == 2
    r.tick("2026-06-03 00:46:30")
    assert len(r.mail.sent) == 2 and "1 signal " in r.mail.sent[1][0]


def test_an_urgent_email_carries_the_pending_digest_with_it(tmp_path):
    r = Rig(tmp_path, [spike_alert(tfs=("15m",)),
                       spike_alert(name="big", priority="urgent", tfs=("1h",))],
            delivery=dict(digest_minutes=60), a=["2026-06-03 00:05", "2026-06-03 00:50"])
    r.tick("2026-06-03 00:01:30")
    r.tick("2026-06-03 00:16:30")                    # first email: the 00:05 spike on 15m
    r.tick("2026-06-03 01:01:30")                    # 00:45 15m (normal) + 00:00 1h (urgent)
    assert len(r.mail.sent) == 2
    assert "2 signals (1 urgent)" in r.mail.sent[1][0]


def test_a_failed_send_is_retried_with_backoff_and_sent_once(tmp_path):
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-03 00:20"], fail=2)
    r.tick("2026-06-03 00:01:30")
    assert "email 1" in r.tick("2026-06-03 01:01:30").errors
    r.tick("2026-06-03 01:02:00")                    # next try is 01:02:30
    assert r.mail.fail == 1
    r.tick("2026-06-03 01:02:30")                    # fails again; next try 01:04:30
    r.tick("2026-06-03 01:04:00")
    assert not r.mail.sent
    assert r.tick("2026-06-03 01:04:30").sent == 1
    r.tick("2026-06-03 01:10")
    assert len(r.mail.sent) == 1 and r.events()[0]["status"] == "sent"


def test_after_an_outage_old_signals_are_reported_as_late_not_urgent(tmp_path):
    r = Rig(tmp_path, [spike_alert(priority="urgent")],
            a=["2026-06-03 00:20", "2026-06-03 01:20", "2026-06-03 02:20"])
    r.tick("2026-06-03 00:01:30")
    rep = r.tick("2026-06-03 03:10")                 # the PC was off for three hours
    assert rep.events == 3
    assert [e["late"] for e in r.events()] == [1, 1, 0]
    subject, text, _ = r.mail.sent[0]
    assert "(1 urgent)" in subject
    assert text.index("SPIKES") < text.index("WHILE OFFLINE")


def test_a_failing_feed_skips_only_its_market_and_backs_off(tmp_path):
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-03 00:20"], b=["2026-06-03 00:20"])
    r.tick("2026-06-03 00:01:30")
    r.feed.fail = {"B"}
    rep = r.tick("2026-06-03 01:01:30")
    assert "B" in rep.errors and rep.events == 1
    r.feed.fail = set()
    n = len(r.feed.calls)
    r.tick("2026-06-03 01:02:00")                    # B is backing off for a minute
    assert all(m != "B" for m, *_ in r.feed.calls[n:])
    assert r.tick("2026-06-03 01:03:00").events == 1  # B caught up, nothing lost


def test_cooldown_keeps_an_alert_quiet_for_n_bars(tmp_path):
    r = Rig(tmp_path, [spike_alert(markets=("A",), cooldown_bars=3)],
            a=["2026-06-03 00:20", "2026-06-03 02:20", "2026-06-03 04:20"])
    r.tick("2026-06-03 00:01:30")
    r.tick("2026-06-03 05:01:30")
    assert [e["bar_time"][11:16] for e in r.events()] == ["00:00", "04:00"]


def test_categories_follow_the_config_order(tmp_path):
    cfg = config(tmp_path, [spike_alert()], categories=("Spikes", "Other"))
    ev = lambda cat, m: dict(category=cat, market=m, tf="1h", priority="normal", late=0,  # noqa
                             what="x", side="", price=1.0, detail="{}",
                             close_time="2026-06-03T01:00:00+00:00")
    _, text, html = compose([ev("New", "C"), ev("Other", "B"), ev("Spikes", "A")], cfg,
                            at("2026-06-03 01:02"))
    assert text.index("SPIKES") < text.index("OTHER") < text.index("NEW")
    assert html.index(">Spikes<") < html.index(">Other<")


# ── correctness of the data path ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tf, rule", [("15m", "15min"), ("1h", "1h"), ("4h", "4h")])
def test_timeframe_bars_match_the_research_resampler(tf, rule):
    b = bars()
    mine = tf_bars(b, tf)
    ref = resample_ohlcv(b, rule)[mine.columns]
    pd.testing.assert_frame_equal(mine, ref, check_dtype=False)


def test_the_candle_detector_never_repaints():
    rng = np.random.default_rng(3)
    n = 800
    c = 100 + rng.normal(0, 1, n).cumsum()
    o = c + rng.normal(0, 0.8, n)
    f = pd.DataFrame({TIME_COL: pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
                      "open": o, "high": np.maximum(o, c) + rng.exponential(0.6, n),
                      "low": np.minimum(o, c) - rng.exponential(0.6, n), "close": c,
                      "volume": 1.0})
    p = dict(patterns=("engulfing_bull", "engulfing_bear", "hammer", "shooting_star"))
    full = {(h["i"], h["what"]) for h in CandleDetector().evaluate(f, p)}
    assert len(full) > 20
    for k in (100, 333, 517, 799):
        cut = {(h["i"], h["what"]) for h in CandleDetector().evaluate(f.iloc[:k], p)}
        assert cut == {x for x in full if x[0] < k}, k


def test_preview_reports_the_window_and_writes_no_state(tmp_path):
    cfg = config(tmp_path, [spike_alert()])
    feed = FakeFeed({"A": bars(["2026-06-01 10:20", "2026-06-02 22:20"]), "B": bars(seed=1)})
    events, subject, text, html, errors = preview(cfg, feed, hours=24, now=at("2026-06-03 00:30"))
    assert [e["bar_time"][:16] for e in events] == ["2026-06-02T22:00"]
    assert "Preview" in text and not errors
    assert not (tmp_path / "state").exists()


# ── the process ─────────────────────────────────────────────────────────────────────────────
def test_only_one_monitor_can_hold_the_lock(tmp_path):
    first = acquire_lock(tmp_path / "alerts.lock")
    assert first is not None
    assert acquire_lock(tmp_path / "alerts.lock") is None
    first.close()
    again = acquire_lock(tmp_path / "alerts.lock")
    assert again is not None
    again.close()


def test_the_loop_survives_a_broken_config_and_a_second_copy_exits(tmp_path, monkeypatch):
    from quant.alerts import runner
    config(tmp_path, [spike_alert()])
    cfg_file, state_dir = tmp_path / "cfg.py", tmp_path / "state"
    two = {"A": bars(), "B": bars(seed=1)}

    class Stop(Exception):
        pass

    def sleep(_):
        if not cfg_file.read_text().startswith("ALERTS = ["):
            cfg_file.write_text("ALERTS = [")          # a half-saved edit
        else:
            raise Stop
    monkeypatch.setattr(runner.time, "sleep", sleep)
    try:
        with pytest.raises(Stop):
            runner.run_forever(cfg_file, state_dir, ListNotifier(), FakeFeed(two))
        log = (state_dir / "alerts.log").read_text(encoding="utf-8")
        assert "keeping the last good one" in log and "does not run" in log
        assert log.count("tick failed") == 0
        st = State(state_dir)
        _OPEN.append(st)
        assert st.get("heartbeat") is not None           # it ticked on the old config

        held = acquire_lock(state_dir / "alerts.lock")
        assert runner.run_forever(cfg_file, state_dir, ListNotifier(), FakeFeed(two)) == 0
        held.close()
    finally:
        for h in list(runner.log.handlers):
            runner.log.removeHandler(h)
            h.close()


def test_email_settings_come_only_from_the_environment(monkeypatch):
    for k in ("ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(NotConfigured, match="ALERT_SMTP_USER, ALERT_SMTP_PASSWORD"):
        SmtpNotifier.from_env()
    monkeypatch.setenv("ALERT_SMTP_USER", "me@example.com")
    monkeypatch.setenv("ALERT_SMTP_PASSWORD", "x")
    n = SmtpNotifier.from_env()
    assert n.to == ["me@example.com"] and n.host == "smtp.gmail.com" and n.port == 587
