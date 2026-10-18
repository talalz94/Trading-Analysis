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
    assert {a.kind for a in cfg.alerts} >= {"trendline", "candle", "chart"}
    assert cfg.news.on and cfg.markets


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
    monkeypatch.setitem(feeds._FETCHERS, "binance", lambda *a: gate.wait(5))
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
