"""
US actuals from the agencies (quant/news/official.py) -- on a fake FRED and a fake BLS, so every test
states what the agency had published on which day. The promises: a number is printed the way Forex
Factory prints it (half-up, never "-0.0"); a release is read as it was published THAT DAY, not from
today's revised history; GDP's three stages come from the release days, so an unchanged estimate
keeps its stage; the Fed's rate is the one in force the day after the decision; a live reply counts
only when it is NEWER than what was already out; a FRED number posted on a weekend is tied back to
its release; the agencies' number wins the merge; the history resumes from the last HISTORY row and
the live polls are spaced and capped. The last test reads the real store, when there is one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.news import official as O
from quant.news import update as U
from quant.news.calendar import NewsStore, merge_rows

NY = "America/New_York"
PCT = O.Spec("X", "X", "pch", 1, "%")
BY = O.specs_by_title()


def ts(d):
    return pd.Timestamp(d)


def ny(s):
    return pd.Timestamp(s, tz=NY).tz_convert("UTC")


class FakeFred:
    """{series: {day: {period: value}}} = what FRED showed from each day on. Answers the three
    calls official.py makes: first prints (output_type=4), the series as of chosen days
    (output_type=2), and the series now with each value's realtime_start."""

    def __init__(self, vint):
        self.vint = {s: {d: {ts(p): v for p, v in obs.items()} for d, obs in days.items()}
                     for s, days in vint.items()}
        self.calls = []

    def as_of(self, series, day):
        out = {}
        for d in sorted(self.vint[series]):
            if d <= day:
                out.update(self.vint[series][d])
        return out

    def __call__(self, path, key, timeout=30, **p):
        self.calls.append((path, p))
        assert path == "series/observations"
        s = p["series_id"]
        days = sorted(self.vint[s])
        periods = sorted({q for d in days for q in self.vint[s][d]})
        if p.get("output_type") == 4:
            obs = []
            for q in periods:
                d = next(d for d in days if q in self.vint[s][d])
                obs.append(dict(date=str(q.date()), realtime_start=d,
                                value=str(self.vint[s][d][q])))
            return dict(observations=obs)
        if p.get("output_type") == 2:
            want = p["vintage_dates"].split(",")
            rows = []
            for q in periods:
                r = dict(date=str(q.date()))
                for d in want:
                    v = self.as_of(s, d).get(q)
                    r[f"{s}_{d.replace('-', '')}"] = "." if v is None else str(v)
                rows.append(r)
            return dict(observations=rows)
        now = self.as_of(s, "9999-12-31")
        obs = []
        for q in sorted(now):
            changed = [d for d in days if q in self.vint[s][d]]
            obs.append(dict(date=str(q.date()), realtime_start=changed[-1], value=str(now[q])))
        return dict(observations=obs)


# ── printing ────────────────────────────────────────────────────────────────────────────────
def test_numbers_print_the_way_forex_factory_prints_them():
    assert O.fmt(0.25, PCT) == "0.3%"                       # half-up; round() gives 0.2
    assert O.fmt(0.05, PCT) == "0.1%"
    assert O.round_half_up(2.675, 2) == 2.68                # 2.675 is 2.67499.. in binary
    assert O.fmt(-0.04, PCT) == "0.0%"                      # never "-0.0%"
    assert O.fmt(206.0, BY["Non-Farm Employment Change"]) == "206K"
    assert O.fmt(231500.0, BY["Unemployment Claims"]) == "232K"
    assert O.fmt(7600.0, BY["JOLTS Job Openings"]) == "7.60M"
    assert O.fmt(5.0, BY["Federal Funds Rate"]) == "5.00%"
    assert O.fmt(np.nan, PCT) is None


def test_each_kind_of_number_is_computed_from_the_levels():
    s = pd.Series([100.0, 100.4], index=[ts("2024-05-01"), ts("2024-06-01")])
    p, v = O.compute(s, "pch")
    assert p == ts("2024-06-01") and v == pytest.approx(0.4)
    assert O.compute(s, "chg")[1] == pytest.approx(0.4)
    assert O.compute(s, "lin")[1] == 100.4
    y = pd.Series([300.0, 303.0, 309.0],
                  index=[ts("2023-06-01"), ts("2024-05-01"), ts("2024-06-01")])
    assert O.compute(y, "pc1")[1] == pytest.approx(3.0)     # against the SAME month a year ago
    assert np.isnan(O.compute(y.drop(ts("2023-06-01")), "pc1")[1])
    assert np.isnan(O.compute(s.iloc[1:], "pch")[1])        # nothing to compare with


# ── history ─────────────────────────────────────────────────────────────────────────────────
PAYEMS = {"PAYEMS": {
    "2024-06-07": {"2024-04-01": 158100, "2024-05-01": 158372},
    "2024-07-05": {"2024-05-01": 158432, "2024-06-01": 158638},
    "2024-08-02": {"2024-06-01": 158609, "2024-07-01": 158723},
}}


def test_payrolls_are_read_as_published_that_day(monkeypatch):
    # June: 158,638 - 158,432 (May as revised IN THE JUNE REPORT) = 206K. July: 158,723 - 158,609
    # (June as revised in the July report) = 114K -- from today's history it would be 85K
    fake = FakeFred(PAYEMS)
    monkeypatch.setattr(O, "_fred", fake)
    h = O.history("k", "2024-07-01", specs=[BY["Non-Farm Employment Change"]])
    assert list(h["actual"]) == ["206K", "114K"]
    assert list(h["t"]) == [ny("2024-07-05 08:30"), ny("2024-08-02 08:30")]
    assert set(h["time_q"]) == {"official"} and set(h["source"]) == {"official"}
    assert list(h["period"]) == [ts("2024-06-01"), ts("2024-07-01")]
    # incremental: only what is newer than the last day stored
    h2 = O.history("k", "2024-07-01", specs=[BY["Non-Farm Employment Change"]],
                   after={"PAYEMS": "2024-07-05"})
    assert list(h2["actual"]) == ["114K"]


def test_an_unrevised_series_uses_first_prints_and_y_y_reaches_back_a_year(monkeypatch):
    fake = FakeFred({"CPIAUCNS": {
        "2023-07-12": {"2023-06-01": 305.1},
        "2024-07-11": {"2024-06-01": 314.2},
    }})
    monkeypatch.setattr(O, "_fred", fake)
    h = O.history("k", "2024-07-01", specs=[BY["CPI y/y"]])
    assert list(h["actual"]) == ["3.0%"]                    # 314.2 / 305.1 - 1 = 2.98%
    assert not any(p.get("output_type") == 2 for _, p in fake.calls)


def test_two_months_published_on_one_day_print_the_later_one(monkeypatch):
    # JOLTS caught up after the 2025 shutdown: September and October together on 9 December
    fake = FakeFred({"JTSJOL": {
        "2025-09-30": {"2025-08-01": 7227},
        "2025-12-09": {"2025-09-01": 7658, "2025-10-01": 7670},
    }})
    monkeypatch.setattr(O, "_fred", fake)
    h = O.history("k", "2025-09-01", specs=[BY["JOLTS Job Openings"]])
    assert list(zip(h["actual"], h["period"])) == [("7.23M", ts("2025-08-01")),
                                                   ("7.67M", ts("2025-10-01"))]


def test_the_start_of_fred_s_vintage_record_is_not_a_release(monkeypatch):
    obs = [dict(date=f"2009-{m:02d}-01", realtime_start="2010-01-01", value="1") for m in range(1, 6)]
    obs += [dict(date="2010-01-01", realtime_start="2010-02-05", value="2")]
    monkeypatch.setattr(O, "_fred", lambda *a, **k: dict(observations=obs))
    fp = O.first_prints("k", "S", "2009-01-01")
    assert list(fp["day"]) == ["2010-02-05"]


def test_gdp_stages_come_from_the_release_days_even_when_an_estimate_is_unchanged(monkeypatch):
    # the second estimate repeated the first: FRED records no revision that day
    fake = FakeFred({"A191RL1Q225SBEA": {
        "2024-10-30": {"2024-07-01": 2.8},
        "2024-12-19": {"2024-07-01": 3.1},
    }})
    monkeypatch.setattr(O, "_fred", fake)
    days = {"2024-10-30": "Advance GDP q/q", "2024-11-27": "Prelim GDP q/q",
            "2024-12-19": "Final GDP q/q"}
    h = O.history("k", "2024-10-01", specs=[BY["Advance GDP q/q"]], gdp_days=days)
    assert list(zip(h["title"], h["actual"])) == [("Advance GDP q/q", "2.8%"),
                                                  ("Prelim GDP q/q", "2.8%"),
                                                  ("Final GDP q/q", "3.1%")]


def test_the_rate_is_the_one_in_force_the_day_after_the_decision(monkeypatch):
    fake = FakeFred({"DFEDTARU": {"2024-09-19": {"2024-09-16": 5.5, "2024-09-17": 5.5,
                                                 "2024-09-18": 5.5, "2024-09-19": 5.0}}})
    monkeypatch.setattr(O, "_fred", fake)
    h = O.history("k", "2024-09-01", specs=[BY["Federal Funds Rate"]], fomc_days=["2024-09-18"])
    assert list(h["actual"]) == ["5.00%"]
    assert list(h["t"]) == [ny("2024-09-18 14:00")]


@pytest.mark.parametrize("text, upper", [
    ("decided to lower the target range for the federal funds rate by 1/2 percentage point to "
     "4-3/4 to 5 percent.", 5.0),
    ("decided to maintain the target range for the federal funds rate at 0 to 1/4 percent", 0.25),
    ("to raise the target range for the <a href='x'>federal funds rate</a>\n to 5-1/4 to 5-1/2 "
     "percent", 5.5),
    ("lower the target range for the federal funds rate to 4.25 to 4.50 percent", 4.5),
    ("The Committee will continue to monitor the implications of incoming information", None),
])
def test_the_fed_statement_gives_the_upper_bound(text, upper):
    assert O.parse_fed_statement(f"<html><p>{text}</p></html>") == upper


# ── live ────────────────────────────────────────────────────────────────────────────────────
def pend(t, title):
    return pd.DataFrame([dict(t=pd.Timestamp(t, tz="UTC"), title=title, currency="USD")])


def test_a_live_bls_number_counts_only_when_its_period_is_newer(monkeypatch):
    lv = {"CES0000000001": pd.Series([158609.0, 158723.0],
                                     index=[ts("2024-06-01"), ts("2024-07-01")])}
    monkeypatch.setattr(O, "bls_levels", lambda ids, year, key=None, **kw: lv)
    p = pend("2024-08-02 12:30", "Non-Farm Employment Change")
    title = "Non-Farm Employment Change"
    rows, _ = O.live(None, p, {title: ts("2024-06-01")})
    assert list(rows["actual"]) == ["114K"] and rows["t"].iloc[0] == p["t"].iloc[0]
    rows, _ = O.live(None, p, {title: ts("2024-07-01")})   # BLS still shows the old report
    assert rows.empty
    rows, notes = O.live(None, p, {})                      # unknown baseline: refuse, say so
    assert rows.empty and "no baseline" in notes[0]


def test_bls_is_asked_two_years_back_and_only_for_months(monkeypatch):
    seen = {}

    class R:
        status_code = 200

        def json(self):
            return dict(status="REQUEST_SUCCEEDED", Results=dict(series=[dict(
                seriesID="CUUR0000SA0",
                data=[dict(year="2025", period="M13", value="320"),   # an annual average
                      dict(year="2025", period="M12", value="321.0"),
                      dict(year="2024", period="M12", value="312.0")])]))

    def post(url, json=None, **k):
        seen.update(json)
        return R()

    import requests
    monkeypatch.setattr(requests, "post", post)
    out = O.bls_levels(["CUUR0000SA0"], 2026)
    assert seen["startyear"] == "2024"          # December's y/y comes out in January
    assert list(out["CUUR0000SA0"].index) == [ts("2024-12-01"), ts("2025-12-01")]


def test_a_live_fred_number_counts_only_when_published_on_the_release_day(monkeypatch):
    s = pd.Series([700.0, 707.0], index=[ts("2024-06-01"), ts("2024-07-01")])
    title = "Retail Sales m/m"
    p = pend("2024-08-15 12:30", title)
    monkeypatch.setattr(O, "fred_now", lambda k, se, st, **kw: (s, "2024-08-14"))
    assert O.live("k", p, {})[0].empty
    monkeypatch.setattr(O, "fred_now", lambda k, se, st, **kw: (s, "2024-08-15"))
    assert list(O.live("k", p, {})[0]["actual"]) == ["1.0%"]


def test_when_bls_is_off_limits_the_number_comes_from_fred(monkeypatch):
    s = pd.Series([158609.0, 158723.0], index=[ts("2024-06-01"), ts("2024-07-01")])
    monkeypatch.setattr(O, "bls_levels", lambda *a, **k: pytest.fail("BLS asked over its cap"))
    monkeypatch.setattr(O, "fred_now", lambda k, se, st, **kw: (s, "2024-08-02"))
    p = pend("2024-08-02 12:30", "Non-Farm Employment Change")
    rows, _ = O.live("k", p, {"Non-Farm Employment Change": ts("2024-06-01")}, bls_ok=False)
    assert list(rows["actual"]) == ["114K"]


# ── the calendar ────────────────────────────────────────────────────────────────────────────
def test_a_weekend_posting_keeps_its_date_only():
    spec = BY["Philly Fed Manufacturing Index"]
    sun = O._row(spec.title, "2022-11-27", -19.4, spec, "2022-11-01")
    thu = O._row(spec.title, "2022-11-17", -19.4, spec, "2022-11-01")
    assert sun["time_q"] == "date" and thu["time_q"] == "official"
    assert thu["t"] == ny("2022-11-17 08:30")


def row(t, title, source, actual=None, q="exact"):
    return dict(t=pd.Timestamp(t, tz="UTC"), time_q=q, currency="USD", title=title, impact="high",
                actual=actual, forecast="0.3%", previous="0.2%", source=source)


def test_a_number_fred_posted_late_is_tied_to_its_release():
    spec = BY["Philly Fed Manufacturing Index"]
    late = O._row(spec.title, "2022-11-27", -19.4, spec, "2022-11-01")
    rows = pd.DataFrame([row("2022-11-17 13:30", spec.title, "ff_archive"),
                         {k: late[k] for k in late if k not in ("period", "series")}])
    ev = merge_rows(rows)
    assert len(ev) == 1
    assert ev["t"].iloc[0] == pd.Timestamp("2022-11-17 13:30", tz="UTC")
    assert ev["actual"].iloc[0] == "-19.4"
    # too far from any release: left where it is
    far = pd.DataFrame([row("2022-10-20 12:30", spec.title, "ff_archive"),
                        {k: late[k] for k in late if k not in ("period", "series")}])
    assert len(merge_rows(far)) == 2


def test_the_agencies_number_wins_the_merge():
    rows = pd.DataFrame([row("2022-01-26 19:00", "Federal Funds Rate", "ff_archive", "<0.25%"),
                         row("2022-01-26 19:00", "Federal Funds Rate", "official", "0.25%",
                             q="official")])
    assert list(merge_rows(rows)["actual"]) == ["0.25%"]


def test_an_event_only_the_agencies_report_takes_forex_factory_s_last_rating():
    # after the archive ends, only the agency number exists for claims: without a rating the
    # event would be level "none" and never shown
    off = {**O._row("Unemployment Claims", "2025-06-05", 248000, BY["Unemployment Claims"],
                    "2025-05-31")}
    rows = pd.DataFrame([row("2025-03-06 13:30", "Unemployment Claims", "ff_archive"),
                         row("2025-04-24 12:30", "Unemployment Claims", "ff_archive"),
                         {k: off[k] for k in off if k not in ("period", "series")},
                         row("2025-09-04 12:30", "Unemployment Claims", "ff_archive")])
    rows.loc[0, "impact"], rows.loc[1, "impact"], rows.loc[3, "impact"] = "medium", "high", "low"
    ev = merge_rows(rows).sort_values("t")
    # the LAST rating before it, not the next one after
    assert list(ev["impact"]) == ["medium", "high", "high", "low"]
    assert ev["actual"].iloc[2] == "248K"
    # never rated before: the first rating after it
    rows2 = rows.drop(index=3).copy()
    rows2.loc[[0, 1], "t"] = rows2.loc[[0, 1], "t"] + pd.DateOffset(years=1)
    assert merge_rows(rows2).sort_values("t")["impact"].iloc[0] == "medium"


# ── the store ───────────────────────────────────────────────────────────────────────────────
def hist_row(day, actual, series="PAYEMS", title="Non-Farm Employment Change"):
    return dict(t=ny(f"{day} 08:30"), time_q="official", currency="USD", title=title,
                impact="high", actual=actual, forecast=None, previous=None, source="official",
                period=ts(day) - pd.DateOffset(months=1), series=series)


def test_the_history_resumes_from_its_own_last_row_not_a_live_one(tmp_path, monkeypatch):
    st = NewsStore(tmp_path)
    calls = []

    def fake_history(key, start, **kw):
        calls.append(kw["after"])
        return pd.DataFrame([hist_row("2024-07-05", "206K")] if len(calls) == 1 else
                            [hist_row("2024-08-02", "114K"), hist_row("2024-09-06", "142K")])

    from quant.news import official
    monkeypatch.setattr(official, "history", fake_history)
    now = pd.Timestamp("2024-09-10", tz="UTC")
    U.update_official(st, "k", now)
    assert calls == [{}]
    assert U.update_official(st, "k", now + pd.Timedelta(hours=1)) == "official: fresh"
    # caught live after time offline: it must not make the history skip August
    U._merge_official(st, pd.DataFrame([hist_row("2024-09-06", "142K")]), "live")
    U.update_official(st, "k", now + pd.Timedelta(hours=7))
    assert calls[-1] == {"PAYEMS": "2024-07-05"}
    got = st.read("official")
    assert list(got["actual"]) == ["206K", "114K", "142K"]
    assert list(got["via"]) == ["history"] * 3              # the history row replaced the live one


def test_live_polls_are_spaced_and_bls_is_capped(tmp_path, monkeypatch):
    st = NewsStore(tmp_path)
    monkeypatch.delenv("BLS_API_KEY", raising=False)
    monkeypatch.setattr(U, "_baseline", lambda *a: ts("2024-06-01"))
    seen = []

    def fake_live(key, pending, base, bls_key=None, bls_ok=True):
        seen.append(bls_ok)
        return pd.DataFrame(columns=O.ROW_COLS + ["period", "series"]), []

    monkeypatch.setattr(O, "live", fake_live)
    now = pd.Timestamp("2024-08-02 12:35", tz="UTC")
    p = pend("2024-08-02 12:30", "Non-Farm Employment Change")
    st.set_meta(bls_count={"2024-08-02": U.BLS_DAILY_CAP[0] - 1})
    U.poll_official(st, "k", p, now)
    assert seen == [True]
    assert U.poll_official(st, "k", p, now + pd.Timedelta(seconds=30)).endswith("between polls")
    U.poll_official(st, "k", p, now + pd.Timedelta(seconds=46))
    assert seen == [True, False]                            # the cap was reached
    # a release more than 3 h old, or not yet out, or not a US one: nothing asked
    for t in ("2024-08-02 09:00", "2024-08-02 14:00"):
        U.poll_official(st, "k", pend(t, "Non-Farm Employment Change"), now + pd.Timedelta(hours=1))
    assert len(seen) == 2


def test_between_daily_sweeps_only_series_with_a_release_since_are_asked(tmp_path, monkeypatch):
    st = NewsStore(tmp_path)
    U._merge_official(st, pd.DataFrame([
        hist_row("2024-08-02", "114K"),
        hist_row("2024-07-11", "-0.1%", series="CPIAUCSL", title="CPI m/m")]), "history")
    st.write("events", pd.DataFrame([
        dict(t=ny("2024-08-02 08:30"), currency="USD", title="Non-Farm Employment Change"),
        dict(t=ny("2024-08-14 08:30"), currency="USD", title="CPI m/m")]))
    asked = []

    def fake_history(key, start, specs=(), **kw):
        asked.append({sp.fred for sp in specs})
        return pd.DataFrame(columns=O.ROW_COLS + ["period", "series"])

    monkeypatch.setattr(O, "history", fake_history)
    now = pd.Timestamp("2024-08-20", tz="UTC")
    U.update_official(st, "k", now)                       # the first run is a full sweep
    assert "PAYEMS" in asked[-1] and len(asked[-1]) == len(O.SPECS)
    U.update_official(st, "k", now + pd.Timedelta(hours=7))
    assert "CPIAUCSL" in asked[-1] and "PAYEMS" not in asked[-1]   # payrolls: nothing new
    U.update_official(st, "k", now + pd.Timedelta(hours=25))
    assert len(asked[-1]) == len(O.SPECS)                 # once a day, every series


def test_a_live_poll_stops_when_its_time_is_spent(monkeypatch):
    s = pd.Series([700.0, 707.0], index=[ts("2024-06-01"), ts("2024-07-01")])
    monkeypatch.setattr(O, "fred_now", lambda *a, **k: (s, "2024-08-15"))
    p = pd.concat([pend("2024-08-15 12:30", "Retail Sales m/m"),
                   pend("2024-08-15 12:30", "Core Retail Sales m/m")])
    assert len(O.live("k", p, {})[0]) == 2
    rows, notes = O.live("k", p, {}, budget=-1)
    assert rows.empty and "time budget" in notes[0]


def test_a_source_that_fails_is_not_asked_again_every_minute(tmp_path, monkeypatch):
    from quant.news import sources
    st = NewsStore(tmp_path)
    calls = []

    def down(*a, **k):
        calls.append(1)
        raise sources.SourceError("HTTP 503")

    monkeypatch.setattr(sources, "fred_rows", down)
    monkeypatch.setattr(sources, "fetch_ff_week", down)
    now = pd.Timestamp("2026-10-14 12:00", tz="UTC")
    for fn in (lambda t: U.update_fred(st, "k", t), lambda t: U.update_feed(st, t)):
        calls.clear()
        with pytest.raises(sources.SourceError):
            fn(now)
        assert "backing off" in fn(now + pd.Timedelta(minutes=1))
        assert len(calls) == 1
    # the feed (the reminders need it) is retried after 5 minutes, FRED after 30
    with pytest.raises(sources.SourceError):
        U.update_feed(st, now + pd.Timedelta(minutes=6))
    assert "backing off" in U.update_fred(st, "k", now + pd.Timedelta(minutes=6))
    with pytest.raises(sources.SourceError):
        U.update_fred(st, "k", now + pd.Timedelta(minutes=31))


def test_parallel_fetches_never_exceed_fred_s_rate(monkeypatch):
    # the workers only hide latency: every request still waits for its slot, whichever thread
    import time
    from concurrent.futures import ThreadPoolExecutor
    from quant.news import sources

    class R:
        status_code = 200

        def json(self):
            return {}

    stamps = []
    monkeypatch.setattr(sources, "_get", lambda *a, **k: stamps.append(time.monotonic()) or R())
    monkeypatch.setattr(sources, "FRED_PER_SECOND", 20.0)
    monkeypatch.setattr(sources, "_fred_next", [0.0])
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: sources._fred("series/observations", "k"), range(8)))
    gaps = np.diff(sorted(stamps))
    assert len(stamps) == 8 and gaps.min() >= 0.05 * 0.9      # 20 a second, never faster


# ── the real thing ──────────────────────────────────────────────────────────────────────────
def test_the_stored_history_matches_forex_factory_s_printed_numbers():
    st = NewsStore()
    off, arc = st.read("official"), st.read("archive")
    if off is None or arc is None:
        pytest.skip("no official history / archive in data/news")
    v = O.validate(off, arc).set_index("title")
    v = v[v["both"] > 0]
    assert len(v) >= 20
    fed = v.loc["Federal Funds Rate"]
    assert fed["match"] >= 0.95                     # FF printed "<0.25%" in January 2022
    assert (v.drop(index="Federal Funds Rate")["match"] == 1.0).all(), v[v["match"] < 1]
