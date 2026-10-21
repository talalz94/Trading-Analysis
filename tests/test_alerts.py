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

    def evaluate(self, frame, p, tf):
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
    markets = {"A": dict(symbol="XAUUSD", source="dukascopy"),
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
def test_the_shipped_config_loads_and_every_alert_in_it_is_valid():
    cfg = load_config(DEFAULT_CONFIG)                # checks the OFF alerts too
    assert cfg.markets and (cfg.alerts or cfg.off)


def test_the_trendline_config_loads_and_fits_its_lines_strict():
    # the ready-to-switch copy: it has to load as it stands, since its instructions say copy it
    # over alerts_config.py, and its whole point is the strict fit
    cfg = load_config(DEFAULT_CONFIG.with_name("alerts_config_trendlines.py"))
    lines = [a for a in cfg.alerts + cfg.off if a.kind == "trendline"]
    assert [a for a in cfg.alerts if a.kind == "trendline"]
    assert all(a.params.get("fit") == "strict" for a in lines)
    # what the user asked for: no candle alerts, lines on 1m-4h, a 15-minute clock, news last
    assert {a.kind for a in cfg.alerts} == {"trendline", "chart"}
    assert all(set(a.tfs) == {"1m", "5m", "15m", "1h", "4h"} for a in cfg.alerts
               if a.kind == "trendline")
    assert cfg.delivery.every_minutes == 15 and cfg.delivery.collapse_repeats
    assert cfg.categories[-1] == "News" and cfg.news.on and cfg.markets
    assert cfg.news.remind_before["high"] == (1440, 720, 360, 120, 60, 30, 15)


@pytest.mark.parametrize("bad, says", [
    (dict(markets=("A", "Z")), "'Z' is neither a market nor a group"),
    (dict(tfs=("1h", "3h")), "tfs must be"),
    (dict(kind="nope"), "kind must be one of"),
    (dict(priority="asap"), "priority must be"),
])
def test_a_config_mistake_is_refused_by_name(tmp_path, bad, says):
    with pytest.raises(ConfigError, match=says):
        config(tmp_path, [spike_alert(**bad)])


def plain(kind, **kw):
    return dict(name="x", kind=kind, markets="all", tfs="1h", **kw)


@pytest.mark.parametrize("alert, says", [
    (plain("candle", patterns=("engulfing_bull", "engulfed")), "engulfed"),
    (plain("rsi", perod=14), r"no setting \['perod'\]"),            # a typo is not a default
    (plain("rsi", above=30, below=70), "below < above"),
    (plain("ma_cross", fast=200, slow=50), "fast must be shorter"),
    (plain("fvg", event="sweep"), "formed | filled | both"),
    (plain("trendline", direction="sideways"), "direction must be"),
    (plain("setup", setup="time_entry_long@v1"), "unknown setup"),        # the metronome
    (plain("setup", setup="bos_long@v1", options=dict(zz_mult=2)), r"no option \['zz_mult'\]"),
    (dict(plain("rsi"), on=False, perod=14), "perod"),                  # checked while OFF too
])
def test_every_kind_refuses_a_bad_setting_by_name(tmp_path, alert, says):
    with pytest.raises(ConfigError, match=says):
        config(tmp_path, [alert])


def test_two_alerts_cannot_share_a_name(tmp_path):
    with pytest.raises(ConfigError, match="two alerts are named 'spike'"):
        config(tmp_path, [spike_alert(), spike_alert(tfs="4h")])


@pytest.mark.parametrize("market, ok", [
    (dict(symbol="US500", source="dukascopy"), "E_SandP-500"),          # a short name
    (dict(symbol="cocoa.cmd/usd", source="dukascopy"), "COCOA.CMD/USD"),  # Dukascopy's own code
    (dict(symbol="EURJPY", source="dukascopy"), "EUR/JPY"),             # any 6-letter FX pair
    (dict(symbol="ABCXYZ", source="dukascopy"), None),
    (dict(symbol="NOTHING", source="dukascopy"), None),
    (dict(symbol="btc-usdt", source="binance"), None),
])
def test_a_market_must_be_an_instrument_its_source_has(tmp_path, market, ok):
    from quant.data.dukascopy import _instrument
    p = tmp_path / "cfg.py"
    p.write_text(f"MARKETS = {{'M': {market!r}}}\nALERTS = {[plain('candle', patterns='all')]!r}\n")
    if ok is None:
        with pytest.raises(ConfigError, match="'M'"):
            load_config(p)
    else:
        assert _instrument(load_config(p).markets["M"].symbol) == ok


def test_every_short_name_is_a_real_dukascopy_instrument():
    from quant.data.dukascopy import _SYMBOL_MAP, dukascopy_instruments
    assert not [v for v in _SYMBOL_MAP.values() if v not in dukascopy_instruments()]


def test_an_alert_switched_off_is_not_run_and_comes_back_without_the_gap(tmp_path):
    on = [spike_alert(markets=("A",))]
    off = [dict(spike_alert(markets=("A",)), on=False)]
    r = Rig(tmp_path, on, a=["2026-06-03 02:20", "2026-06-03 05:20"])
    r.tick("2026-06-03 00:01:30")
    r.cfg = config(tmp_path, off)
    assert r.cfg.alerts == () and [a.name for a in r.cfg.off] == ["spike"]
    r.tick("2026-06-03 03:01:30")                    # the 02:00 spike: switched off, not judged
    r.cfg = config(tmp_path, on)
    r.tick("2026-06-03 04:01:30")                    # back on: starts from now, nothing mailed
    assert r.events() == []
    assert r.tick("2026-06-03 06:01:30").events == 1  # and fires again from then on


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


# ── 1m bars and the 15-minute clock ─────────────────────────────────────────────────────────
def bars1(spikes=(), start="2026-06-02 12:00", end="2026-06-03 06:00", seed=3):
    t = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    c = 100 + np.random.default_rng(seed).normal(0, 0.05, len(t)).cumsum()
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({TIME_COL: t, "open": o, "high": np.maximum(o, c) + 0.02,
                       "low": np.minimum(o, c) - 0.02, "close": c, "volume": 1.0})
    for s in spikes:
        df.loc[df[TIME_COL] == at(s), "high"] = 1000.0
    return df


class ClockRig(Rig):
    """A Rig whose market B also has 1m bars, on a 15-minute clock unless told otherwise."""

    def __init__(self, tmp_path, alerts, spikes_1m=(), delivery=None, **kw):
        super().__init__(tmp_path, alerts, delivery=delivery or dict(every_minutes=15), **kw)
        self.feed = FakeFeed(self.feed.bars, bars_1m={"B": bars1(spikes_1m)})
        self.when = {}                               # email index -> the tick that sent it

    def run(self, start, end, step="1min"):
        for t in pd.date_range(at(start), at(end), freq=step):
            n = len(self.mail.sent)
            tick(self.cfg, self.state, self.mail, self.feed, now=t)
            for i in range(n, len(self.mail.sent)):
                self.when[i] = t


ONE_MIN = dict(name="1m spike", category="Spikes", kind="spike", level=999, markets=("B",),
               tfs="1m", priority="urgent")


def test_a_1m_alert_reads_1m_bars_and_waits_for_the_quarter_hour(tmp_path):
    r = ClockRig(tmp_path, [ONE_MIN], spikes_1m=["2026-06-03 00:03", "2026-06-03 00:07",
                                                 "2026-06-03 00:11"])
    r.run("2026-06-03 00:00:20", "2026-06-03 00:32:20")
    assert len(r.events()) == 3 and {e["tf"] for e in r.events()} == {"1m"}
    # ONE email, at the first look after 00:15's bars are final -- not one per 1m candle,
    # and not at 00:04 when the first of them closed
    assert len(r.mail.sent) == 1 and r.when[0] == at("2026-06-03 00:17:20")
    calls = [c for c in r.feed.calls if c[-1] == "1m"]
    assert calls and len(calls) == len(r.feed.calls)        # only the 1m feed: no 5m alert here
    asked = sorted({c[2] for c in calls})
    assert all(t.minute % 15 == 2 for t in asked[1:])      # after the first look: boundaries only
    assert r.state.bars("B", "1m").shape[0] > 0 and r.state.bars("B").empty


def test_on_a_clock_one_email_per_boundary_even_for_urgent_alerts(tmp_path):
    alerts = [spike_alert(name="5m urgent", tfs="5m", priority="urgent", markets=("A",)),
              spike_alert(name="1h", tfs="1h", markets=("A",))]
    r = ClockRig(tmp_path, alerts, a=["2026-06-03 00:05", "2026-06-03 00:20", "2026-06-03 00:40",
                                      "2026-06-03 01:10"])
    r.run("2026-06-03 00:00:20", "2026-06-03 02:00:20")
    sent = [r.when[i] for i in sorted(r.when)]
    assert sent and all(t.minute % 15 == 2 and t.second == 20 for t in sent)
    assert len({t.floor("15min") for t in sent}) == len(sent)        # never two in one slot
    # four 5m spikes; on 1h the 00:00 bar (three of them) -- 01:00's closes after the run
    assert len(r.events()) == 5 and all(e["status"] == "sent" for e in r.events())
    # the 00:05 bar closed at 00:10 and is URGENT, yet it waits for the 00:15 boundary
    assert r.when[0] == at("2026-06-03 00:17:20") and "00:10 UTC" in r.mail.sent[0][1]
    # the 1h bar 00:00 closes at 01:00 and rides with the 5m bars that closed by then
    one_am = [i for i, t in r.when.items() if t == at("2026-06-03 01:02:20")]
    assert one_am and "  1h  " in r.mail.sent[one_am[0]][1]


def test_on_a_clock_a_signal_found_after_its_boundary_email_waits_for_the_next(tmp_path):
    # the feed fails at the 00:15 boundary and answers a minute later: what it brings must not
    # make a second email in the same quarter hour
    r = ClockRig(tmp_path, [spike_alert(tfs="5m", markets=("A",), priority="urgent")],
                 a=["2026-06-03 00:05"])
    r.run("2026-06-03 00:00:20", "2026-06-03 00:16:20")
    r.feed.fail = {"A"}
    r.run("2026-06-03 00:17:20", "2026-06-03 00:17:20")
    r.feed.fail = set()
    r.run("2026-06-03 00:18:20", "2026-06-03 00:33:20")
    assert len(r.events()) == 1 and r.events()[0]["created"] > "2026-06-03T00:17:20"   # backoff
    assert [r.when[i] for i in sorted(r.when)] == [at("2026-06-03 00:32:20")]


def test_repeats_in_one_window_are_one_row(tmp_path):
    spikes = ["2026-06-03 00:03", "2026-06-03 00:07", "2026-06-03 00:11"]
    r = ClockRig(tmp_path, [ONE_MIN], spikes_1m=spikes,
                 delivery=dict(every_minutes=15, collapse_repeats=True))
    r.run("2026-06-03 00:00:20", "2026-06-03 00:17:20")
    text = r.mail.sent[0][1]
    rows = [line for line in text.splitlines() if line.lstrip("! ").startswith("B ")]
    assert len(rows) == 1 and "3 times, 00:04-00:12 UTC" in rows[0]
    assert "1 signal " in r.mail.sent[0][0]
    # a preview spanning several windows keeps them apart
    (tmp_path / "p").mkdir()
    r2 = ClockRig(tmp_path / "p", [ONE_MIN], spikes_1m=spikes + ["2026-06-03 00:21"],
                  delivery=dict(every_minutes=15, collapse_repeats=True))
    events, subject, text, html, errors = preview(r2.cfg, r2.feed, hours=2,
                                                  now=at("2026-06-03 00:40"))
    assert len(events) == 4 and not errors
    rows = [line for line in text.splitlines() if line.lstrip("! ").startswith("B ")]
    assert len(rows) == 2 and "3 times" in text


@pytest.mark.parametrize("delivery, says", [
    (dict(subject="loud"), "subject must be"),
    (dict(charts=11), "charts must be"),
    (dict(every_minutes=7), "every_minutes"),
    (dict(every_minute=15), r"unknown settings \['every_minute'\]"),
])
def test_a_delivery_mistake_is_refused_by_name(tmp_path, delivery, says):
    with pytest.raises(ConfigError, match=says):
        config(tmp_path, [spike_alert()], delivery)


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


def synthetic(n=1500, seed=7):
    """Hourly candles with trends, volatility regimes and wicks: enough structure that every
    kind fires on it, so a no-repaint check is never vacuous."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.25, n // 60 + 1), 60)[:n]
    vol = np.repeat(rng.uniform(0.4, 1.6, n // 40 + 1), 40)[:n]
    c = 1000 + np.cumsum(drift + rng.normal(0, 1, n) * vol)
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.2, n) * vol
    t = pd.date_range("2026-01-05", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({TIME_COL: t, "t": t, "open": o,
                         "high": np.maximum(o, c) + rng.exponential(0.5, n) * vol,
                         "low": np.minimum(o, c) - rng.exponential(0.5, n) * vol, "close": c,
                         "volume": rng.lognormal(3, 0.6, n)})


def _kind_cases():
    from quant.alerts.detectors import setup_keys
    cases = [(k, dict(d.example)) for k, d in REGISTRY.items() if k not in ("spike", "setup")]
    cases += [("chart", dict(patterns="all", event="both")),
              ("trendline", dict(event="both", closes=2, full=1, colour=2)),
              ("trendline", dict(event="break", closes=1, confirm=3)),
              ("trendline", dict(event="break", closes=0, full=2)),
              ("trendline", dict(event="both", closes=1, fit="strict", touch_tol=0.3)),
              ("sr_zone", dict(event="break", closes=2, full=1, colour=1)),
              ("chart", dict(patterns="all")), ("sr_zone", dict(event="both")),
              ("fvg", dict(event="both")), ("ma_cross", dict(fast=20, slow=50)),
              ("rsi", dict(when="enter")), ("macd", dict(cross="zero")),
              ("stochastic", dict(above=100, below=0))]
    cases += [("setup", dict(setup=s)) for s in setup_keys()]
    return cases


@pytest.mark.parametrize("kind, params", _kind_cases(),
                         ids=lambda x: x if isinstance(x, str) else ",".join(
                             f"{k}={v}" for k, v in x.items())[:40])
def test_no_alert_kind_ever_repaints(kind, params):
    """Run on the candles up to bar i, a kind must say exactly what it says about them when run
    on all of them. Cut AT the signals -- data ending on the very bar that fired -- because a
    lookahead of one bar changes nothing but the last candle, and three fixed cuts let a .shift(-1)
    through on three kinds."""
    det = REGISTRY[kind]
    det.validate(params)
    f = synthetic()
    key = lambda h: (h["i"], h["what"], h.get("note", ""))          # noqa: E731
    full = {key(h) for h in det.evaluate(f, params, "1h")}
    if kind != "setup":                              # a few setups are rare on any 1,500 bars
        assert full, f"{kind} never fired, so this test would prove nothing"
    rng = np.random.default_rng(0)
    ends = sorted({i + 1 for i, *_ in full})
    cuts = set(rng.choice(ends, min(25, len(ends)), replace=False).tolist()) if ends else set()
    cuts |= set(rng.integers(600, len(f), 10).tolist())
    for k in sorted(cuts):
        cut = {key(h) for h in det.evaluate(f.iloc[:k].reset_index(drop=True), params, "1h")}
        want = {x for x in full if x[0] < k}
        assert cut == want, (kind, k, sorted(cut ^ want)[:4])


def test_a_break_rule_counts_from_the_first_close_through():
    """By hand, on a level at 100 that is broken downward twice."""
    from quant.alerts.detectors import _sustained
    c = np.array([101, 99.0, 98.5, 100.5, 98.0, 97.0, 96.0, 95.5])
    o = c + 0.4                                       # every candle red
    h = np.array([102, 100.5, 99.5, 101, 100.2, 99.0, 96.5, 96.0])
    l = c - 0.5
    first = np.r_[False, (c[1:] < 100) & (c[:-1] >= 100)]           # bars 1 and 4
    flat, zero = np.full(8, 100.0), np.zeros(8)

    def at(**rule):
        return [j for _, j in _sustained(first, flat, zero, o, h, l, c, False, **rule)]

    assert at(closes=1, full=0, colour=0) == [1, 4]   # the first close through, each time
    assert at(closes=2, full=0, colour=0) == [2, 5]   # bar 3 closed back: the first break died
    assert at(closes=3, full=0, colour=0) == [6]
    assert at(closes=1, full=2, colour=0) == [6]      # whole candles: highs 99.0, 96.5 < 100
    assert at(closes=2, full=0, colour=0, confirm=2) == [6]   # 96 closes below bar 5's low 96.5
    o[5] = c[5] - 0.4                                 # bar 5 is GREEN, though still below 100
    assert at(closes=1, full=0, colour=2) == [2, 7]   # red 1-2; then red 4, GREEN 5 resets,
                                                      # red 6 and 7 -- not 5, as without colour
    assert at(closes=2, full=0, colour=0) == [2, 5]   # colour off: the green candle is irrelevant
    # a SLOPED line is carried along its slope: a support falling 2 a bar is at 98 on bar 2, so
    # the 98.5 close there is back ABOVE it and the first break dies -- flat, it survived
    pairs = _sustained(first, flat, np.full(8, -2.0), o, h, l, c, False,
                       closes=2, full=0, colour=0)
    assert pairs == [(4, 5)]                          # flat gave [(1, 2), (4, 5)]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_closes_rule_is_exactly_the_research_rule(n):
    """closes=N must mean what notebook 21's tl_close_n means, or an alert and a backtest
    would disagree about the same break."""
    from quant.indicators import add_trendlines
    from quant.scenarios.entries import _tl_entry
    f = synthetic()
    d = add_trendlines(f[["open", "high", "low", "close"]].copy())
    q = dict(tl_prefixes=("",), tl_close_n=n, tl_extend=True)
    research = set(np.flatnonzero(_tl_entry(q, "long")(d) | _tl_entry(q, "short")(d)))
    alert = {h["i"] for h in REGISTRY["trendline"].evaluate(f, dict(closes=n), "1h")}
    assert research and alert == research


def test_a_trendline_forms_on_the_bar_it_is_first_published_and_a_break_names_it():
    from quant.indicators import add_trendlines, trendline_table
    f = synthetic()
    lines = trendline_table(add_trendlines(f[["open", "high", "low", "close"]].copy()))
    hits = REGISTRY["trendline"].evaluate(f, dict(event="both", closes=2), "1h")
    formed = sorted(h["i"] for h in hits if h["what"].endswith("formed"))
    assert formed == sorted(lines["born_i"].astype(int))
    breaks = [h for h in hits if h["what"].startswith("broke")]
    assert breaks and all("line formed" in h["note"] and "2 closes" in h["note"] for h in breaks)


def test_the_chart_formed_hook_changes_nothing_and_every_breakout_formed_first():
    from quant.indicators import add_chart_patterns
    f = synthetic(2000)[["open", "high", "low", "close", "volume"]]
    formed = []
    plain = add_chart_patterns(f)
    hooked = add_chart_patterns(f, _formed=formed)
    pd.testing.assert_frame_equal(plain, hooked)
    hits = REGISTRY["chart"].evaluate(f, dict(patterns="all", event="both"), "1h")
    made = [(h["i"], h["what"].replace(" formed", "")) for h in hits if h["what"].endswith("formed")]
    outs = [(h["i"], h["what"].replace(" breakout", "")) for h in hits
            if h["what"].endswith("breakout")]
    assert outs and made
    assert all(any(m == name and i < j for i, m in made) for j, name in outs)


def test_setups_fire_on_this_data_too():
    from quant.alerts.detectors import setup_keys
    f = synthetic()
    fired = [s for s in setup_keys()
             if REGISTRY["setup"].evaluate(f, dict(setup=s), "1h")]
    assert len(fired) >= len(setup_keys()) - 3


def test_a_signal_carries_its_note_into_the_email(tmp_path):
    cfg = config(tmp_path, [spike_alert()])
    e = dict(category="Spikes", market="A", tf="1h", priority="normal", late=0, what="x",
             side="bull", price=1.0, detail='{"note": "target 84,120"}',
             close_time="2026-06-03T01:00:00+00:00")
    _, text, html = compose([e], cfg, at("2026-06-03 01:02"))
    assert "(target 84,120)" in text and "target 84,120</span>" in html


def test_the_catalogue_doc_is_what_the_code_accepts(monkeypatch):
    from quant.alerts.catalogue import DOC, catalogue
    monkeypatch.delitem(REGISTRY, "spike")           # this file's own test kind
    assert DOC.read_text(encoding="utf-8") == catalogue(markdown=True), \
        "docs/ALERTS_CATALOGUE.md is stale: python -m quant.alerts list --write-docs"


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
    full = {(h["i"], h["what"]) for h in CandleDetector().evaluate(f, p, "1h")}
    assert len(full) > 20
    for k in (100, 333, 517, 799):
        cut = {(h["i"], h["what"]) for h in CandleDetector().evaluate(f.iloc[:k], p, "1h")}
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


def test_the_lock_names_its_holder_and_a_second_copy_says_so(tmp_path, capsys):
    # on a server, a second copy that exits in silence looks like a broken service
    import os
    from quant.alerts import runner
    held = acquire_lock(tmp_path / "state" / "alerts.lock")
    try:
        assert runner.lock_holder(tmp_path / "state" / "alerts.lock") in (str(os.getpid()), "?")
        assert runner.run_forever(tmp_path / "cfg.py", tmp_path / "state", ListNotifier(),
                                  FakeFeed({})) == 0
    finally:
        held.close()
    out = capsys.readouterr().out
    assert "another monitor" in out and "already running" in out
    if os.name != "nt":                              # Windows locks the bytes against reading
        assert f"pid {os.getpid()}" in out


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


def test_a_hanging_feed_times_out_by_name_and_is_not_asked_twice(monkeypatch):
    import threading
    from quant.alerts import feeds
    from quant.alerts.config import Market
    gate = threading.Event()
    monkeypatch.setitem(feeds._FETCHERS, ("binance", "5m"), lambda *a: gate.wait(5))
    feed, m, now = feeds.Feed(timeout=0.2), Market("B", "BBBUSDT", "binance", ""), at("2026-06-03")
    try:
        first = feed.fetch([m], {"B": now - pd.Timedelta(minutes=10)}, now)["B"]
        again = feed.fetch([m], {"B": now - pd.Timedelta(minutes=10)}, now)["B"]
    finally:
        gate.set()
    assert isinstance(first, TimeoutError) and "no answer in" in str(first)
    assert "still hanging" in str(again)              # no second thread beside the stuck one


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body

    def json(self):
        return self._body

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_binance_refusing_a_server_falls_back_to_its_market_data_host(monkeypatch):
    """A US server (Google's free VM) gets HTTP 451 from api.binance.com; the same candles come
    from data-api.binance.vision, and the host that answered is used from then on."""
    import requests
    from quant.alerts import feeds
    t0 = int(at("2026-06-03 00:00").timestamp() * 1000)
    kline = [[t0 + k * 300_000, "1", "2", "0.5", "1.5", "10"] + [0] * 6 for k in range(3)]
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(url.rsplit("/api/v3", 1)[0])
        return _Resp(451) if "api.binance.com" in url else _Resp(200, kline)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(feeds, "_binance_host", [0])
    bars_ = feeds.binance_5m("BTCUSDT", at("2026-06-03 00:00"), at("2026-06-03 00:20"))
    assert len(bars_) == 3 and calls == ["https://api.binance.com", "https://data-api.binance.vision"]
    feeds.binance_5m("BTCUSDT", at("2026-06-03 00:00"), at("2026-06-03 00:20"))
    assert calls[2:] == ["https://data-api.binance.vision"]          # remembered: one call

    monkeypatch.setattr(requests, "get", lambda url, params=None, timeout=None: _Resp(451))
    monkeypatch.setattr(feeds, "_binance_host", [0])
    with pytest.raises(requests.HTTPError, match="refuses this server on every host"):
        feeds.binance_5m("BTCUSDT", at("2026-06-03 00:00"), at("2026-06-03 00:20"))


def test_a_months_long_first_download_gets_a_longer_timeout():
    from quant.alerts.feeds import Feed
    f, now = Feed(timeout=90), at("2026-06-03")
    assert f.timeout_for(now - pd.Timedelta(minutes=15), now) < 91      # a normal poll
    assert f.timeout_for(now - pd.Timedelta(days=147), now) > 250       # 42k bars, ~43 requests


def test_email_settings_come_only_from_the_environment(monkeypatch):
    for k in ("ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(NotConfigured, match="ALERT_SMTP_USER, ALERT_SMTP_PASSWORD"):
        SmtpNotifier.from_env()
    monkeypatch.setenv("ALERT_SMTP_USER", "me@example.com")
    monkeypatch.setenv("ALERT_SMTP_PASSWORD", "x")
    n = SmtpNotifier.from_env()
    assert n.to == ["me@example.com"] and n.host == "smtp.gmail.com" and n.port == 587


# ── from an email row to a chart (python -m quant.alerts show REF) ─────────────────────────
def test_every_email_row_carries_a_ref_that_names_its_alert(tmp_path):
    from quant.alerts.show import find_alert, make_ref, parse_ref
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-03 00:20"])
    r.tick("2026-06-03 00:01:30")
    r.tick("2026-06-03 01:01:30")
    ref = make_ref("A", "1h", at("2026-06-03 01:00"), "spike")
    assert ref == "a.1h.202606030100.spike" and f"ref {ref}" in r.mail.sent[0][1]
    assert ref in r.mail.sent[0][2]
    assert parse_ref(ref.upper())[1:3] == ("1h", at("2026-06-03 01:00"))
    _, mk, alert = find_alert(ref, r.cfg.path)
    assert (mk.name, alert.name) == ("A", "spike")


@pytest.mark.parametrize("ref, says", [
    ("a.1h.20260603.spike", "not an alert ref"),
    ("a.3h.202606030100.spike", "not an alert ref"),
    ("a.1h.202606030100.nothing", "no alert 'nothing'"),
    ("zz.1h.202606030100.spike", "for market 'zz'"),
])
def test_a_ref_that_names_nothing_is_refused_by_name(tmp_path, ref, says):
    from quant.alerts.show import find_alert
    config(tmp_path, [spike_alert()])
    with pytest.raises((ValueError, LookupError), match=says):
        find_alert(ref, tmp_path / "cfg.py")


def test_show_re_runs_the_alert_on_its_own_candle_and_draws_it(tmp_path):
    from quant.alerts.show import alert_chart
    r = Rig(tmp_path, [spike_alert()], a=["2026-06-03 00:20"])
    c = alert_chart("a.1h.202606030100.spike", config=r.cfg.path, feed=r.feed,
                    now=at("2026-06-03 06:00"), write=False)
    assert c.reproduced and c.hits[0]["what"] == "spike"
    assert c.frame[TIME_COL].iloc[c.k] == at("2026-06-03 00:00")       # the candle that closed
    assert c.frame[TIME_COL].iloc[-1] < at("2026-06-03 06:00")        # nothing still forming
    names = [tr.name for tr in c.fig.data]
    assert names[0] == "A 1h" and "the alert's candle (its close)" in names
    quiet = alert_chart("a.1h.202606030200.spike", config=r.cfg.path, feed=r.feed,
                        now=at("2026-06-03 06:00"), write=False)
    assert not quiet.reproduced                       # nothing fired there: said, not hidden


def test_every_trendline_alert_the_monitor_sends_redraws_with_its_own_line(tmp_path):
    # the real case: 1m lines on the 15-minute clock, where the monitor judges a quarter hour of
    # candles at once -- show must re-run from the same window or the line can differ
    from quant.alerts.show import alert_chart, make_ref
    tl = dict(name="lines", kind="trendline", event="both", fit="strict", touch_tol=0.3,
              markets=("B",), tfs="1m")
    r = ClockRig(tmp_path, [tl])
    r.run("2026-06-02 21:00:20", "2026-06-03 05:47:20")
    ev = r.events()
    assert len(ev) >= 3, "no trendline alerts to redraw"
    for e in ev:
        c = alert_chart(make_ref(e["market"], e["tf"], pd.Timestamp(e["close_time"]), e["alert"]),
                        config=r.cfg.path, feed=r.feed, now=at("2026-06-03 06:00"), write=False)
        assert e["what"] in [h["what"] for h in c.hits], (e["close_time"], e["what"])
        assert any("line" in (tr.name or "") and "touches" in (tr.name or "")
                   for tr in c.fig.data)


# ── the tiered subject (DELIVERY subject="tiers") ───────────────────────────────────────────
@pytest.mark.parametrize("tf, what, late, want", [
    ("1h", "broke a support trendline", 0, 1), ("4h", "falling wedge breakout", 0, 1),
    ("15m", "broke a resistance trendline", 0, 2), ("1h", "support trendline formed", 0, 2),
    ("4h", "rising wedge formed", 0, 2), ("1m", "broke a support trendline", 0, 3),
    ("5m", "falling wedge breakout", 0, 3), ("15m", "rising wedge formed", 0, 3),
    ("1h", "broke a support trendline", 1, 3),          # found late: for the record
])
def test_a_signal_is_tiered_by_what_it_is_and_its_timeframe(tf, what, late, want):
    from quant.alerts.compose import tier
    assert tier(dict(alert="x", market="BTC", tf=tf, what=what, late=late, detail="{}")) == want


def test_a_tiered_subject_leads_with_the_most_important_row(tmp_path):
    from quant.alerts.compose import compose
    rows = [("1m", "support trendline formed"), ("1h", "broke a support trendline"),
            ("15m", "rising wedge formed"), ("1h", "support trendline formed"),
            ("4h", "rising wedge formed")]                 # two P2: the coarser one leads
    ev = [dict(alert="lines", category="Spikes", priority="normal", market="BTC", tf=tf, what=w,
               side="", price=1.0, close_time="2026-06-03T01:00:00+00:00", late=0, detail="{}")
          for tf, w in rows]
    tiered = compose(ev, config(tmp_path, [spike_alert()], dict(subject="tiers")),
                     at("2026-06-03 01:02"))[0]
    assert tiered == ("[Alerts] 🔴 P1 · BTC 1h broke support line · BTC 4h rising wedge formed"
                      " · +3 more · 01:02 UTC")
    classic = compose(ev, config(tmp_path, [spike_alert()]), at("2026-06-03 01:02"))[0]
    assert classic.startswith("[Alerts] 5 signals")      # the default is what it always was


# ── pictures in the email (DELIVERY charts=N) ───────────────────────────────────────────────
def _picture_rig(tmp_path, charts, **kw):
    # one 15-minute email with two rows: three 1m spikes (one row, its latest) and a 5m spike
    return ClockRig(tmp_path, [ONE_MIN, spike_alert(name="5m spike", tfs="5m", markets=("B",))],
                    spikes_1m=["2026-06-03 00:03", "2026-06-03 00:07", "2026-06-03 00:11"],
                    b=["2026-06-03 00:05"],
                    delivery=dict(every_minutes=15, collapse_repeats=True, charts=charts), **kw)


def test_an_email_carries_pictures_of_its_top_rows_ranked(tmp_path):
    r = _picture_rig(tmp_path, charts=3)
    r.run("2026-06-03 00:00:20", "2026-06-03 00:17:20")
    (subject, text, html), imgs = r.mail.sent[0], r.mail.images[0]
    assert len(imgs) == 2                              # one per ROW, not per repeat
    assert all(png.startswith(b"\x89PNG") and 5_000 < len(png) < 150_000 for _, png in imgs)
    assert [html.count(f"cid:{cid}") for cid, _ in imgs] == [1, 1]
    assert html.index(">Charts</h3>") < html.index("<table")    # at the top, above the list
    (tmp_path / "one").mkdir()
    one = _picture_rig(tmp_path / "one", charts=1)
    one.run("2026-06-03 00:00:20", "2026-06-03 00:17:20")
    assert len(one.mail.images[0]) == 1 and "B 5m spike" in one.mail.sent[0][2].split("<img")[0]


def test_charts_off_is_the_email_it_always_was(tmp_path):
    r = _picture_rig(tmp_path, charts=0)
    r.run("2026-06-03 00:00:20", "2026-06-03 00:17:20")
    assert r.mail.images[0] == [] and ">Charts</h3>" not in r.mail.sent[0][2]


def test_without_matplotlib_or_with_a_broken_chart_the_email_still_goes(tmp_path, monkeypatch):
    import sys
    from quant.alerts import snapshot
    monkeypatch.setitem(sys.modules, "matplotlib", None)       # as on a server without it
    r = _picture_rig(tmp_path, charts=3)
    reps = []
    for t in pd.date_range(at("2026-06-03 00:00:20"), at("2026-06-03 00:17:20"), freq="1min"):
        reps.append(tick(r.cfg, r.state, r.mail, r.feed, now=t))
    assert len(r.mail.sent) == 1 and r.mail.images[0] == []
    assert any("matplotlib is not installed" in x.errors.get("charts", "") for x in reps)
    monkeypatch.delitem(sys.modules, "matplotlib")
    monkeypatch.setattr(snapshot, "render_png", lambda *a, **k: 1 / 0)
    (tmp_path / "b").mkdir()
    r2 = _picture_rig(tmp_path / "b", charts=3)
    r2.run("2026-06-03 00:00:20", "2026-06-03 00:17:20")
    assert len(r2.mail.sent) == 1 and r2.mail.images[0] == []


def test_a_retried_email_keeps_its_pictures(tmp_path):
    r = _picture_rig(tmp_path, charts=3, fail=1)
    r.run("2026-06-03 00:00:20", "2026-06-03 00:20:20")
    assert len(r.mail.sent) == 1 and len(r.mail.images[0]) == 2   # sent on the retry, intact


def test_the_smtp_message_carries_the_pictures_inline():
    msg = SmtpNotifier("me@x.com", "pw", "me@x.com").build(
        "s", "text", '<p><img src="cid:chart0.1@quant-alerts"></p>',
        images=[("chart0.1@quant-alerts", b"\x89PNG fake")])
    parts = {p.get_content_type(): p for p in msg.walk()}
    assert "multipart/related" in parts and "text/html" in parts and "text/plain" in parts
    assert parts["image/png"]["Content-ID"] == "<chart0.1@quant-alerts>"


def test_a_pattern_line_is_drawn_candle_by_candle_from_its_own_first_touch():
    # two endpoints on a time axis bent the line wherever the market shut: 20 points off on Gold
    import sys as _sys
    _sys.path.insert(0, "tests")
    from test_patterns import _ascending_triangle_frame
    from quant.alerts.config import Alert
    from quant.alerts.show import _shape_marks
    f = _ascending_triangle_frame()
    f["t"] = f[TIME_COL]
    alert = Alert(name="x", category="Chart patterns", kind="chart", markets=("A",), tfs=("1h",),
                  params=dict(patterns=("ascending_triangle",), event="formed"))
    det = REGISTRY["chart"]
    hits = det.evaluate(f, alert.params, "1h")
    assert hits, "no triangle formed on the hand-built candles"
    k = hits[0]["i"]
    marks, first = _shape_marks(alert, f.iloc[:k + 1].reset_index(drop=True), 0, len(f) - 1,
                                [h for h in hits if h["i"] == k])
    lines = [m for m in marks if m.kind == "line" and len(set(m.ys)) > 1]
    touches = [m for m in marks if m.kind == "points"]
    assert lines and touches
    for m in lines:
        assert np.all(np.diff(m.xs) == 1)                         # every candle, no jumps
    assert {m.xs[0] for m in lines} == {min(t.xs) for t in touches}  # each from ITS first touch


def test_a_repeating_signal_is_drawn_once_at_its_latest_candle_in_any_order(tmp_path):
    # the preview hands its rows over grouped by alert, not in time order
    from quant.alerts.snapshot import email_charts
    r = _picture_rig(tmp_path, charts=3)
    ev = [dict(alert="1m spike", market="B", tf="1m", what="spike", late=0, priority="urgent",
               close_time=f"2026-06-03T00:{m}:00+00:00", detail="{}") for m in ("12", "04", "08")]
    charts, err = email_charts(ev, r.cfg, lambda m, b: r.feed.bars_1m[m], at("2026-06-03 00:20"))
    assert err is None and [c["close"] for c in charts] == ["2026-06-03T00:12:00+00:00"]
