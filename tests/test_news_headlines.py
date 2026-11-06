"""
The squawk headlines the alerts quote as a release's first result (quant/news/headlines.py), on
lines copied from the headline test's own records (22-30 Oct 2026). The promises: a line counts
only inside its release's window and in its words; a printed result ranks first; the source's tags
are trimmed and its words are not; the number a line gives as the result is read in its printed
units; a failed or empty source raises, so the next one is asked.
"""
import pandas as pd
import pytest

from quant.news import headlines as H


def at(s):
    return pd.Timestamp(s, tz="UTC")


def line(t, text, i=None):
    return dict(id=i or text[:20], title=text, link="", published=at(t).isoformat())


GDP_DAY = [  # 29 Oct 2026, the 12:30 UTC US GDP + PCE release, as FinancialJuice printed it
    line("2026-10-29 12:28:00", "FinancialJuice: US GDP expected at 3.2% -- preview"),
    line("2026-10-29 12:30:07", "FinancialJuice: US Core PCE Prices Advance Actual 2.5% "
                                "(Forecast 2.4%, Previous 3.3%)"),
    line("2026-10-29 12:30:09", "FinancialJuice: US GDP Advance Actual 3.4% (Forecast 3.2%, "
                                "Previous 3.8%)"),
    line("2026-10-29 12:31:30", "FinancialJuice: Traders now see GDP strength lifting yields"),
    line("2026-10-29 12:45:00", "FinancialJuice: US GDP details: consumer spending up"),
]


def test_a_line_counts_only_inside_its_window_and_in_its_words():
    w = H.window("USD", ["Advance GDP q/q"])
    got = H.matching(GDP_DAY, at("2026-10-29 12:30"), *w)
    texts = [x["title"] for x in got]
    assert "preview" not in " ".join(texts)                 # 2 min before: not the result
    assert not any("12:45" in x["published"] for x in got)  # past 10 minutes
    assert not any("PCE" in x for x in texts)               # GDP's words, not PCE's
    assert texts == [GDP_DAY[2]["title"], GDP_DAY[3]["title"]]


def test_a_printed_result_ranks_before_a_number_and_a_number_before_words():
    w = H.window("USD", ["Advance GDP q/q", "Core PCE Price Index m/m"])
    got = H.matching(GDP_DAY, at("2026-10-29 12:30"), *w)
    assert [x["title"].split(": ", 1)[1][:12] for x in got] == \
        ["US Core PCE ", "US GDP Advan", "Traders now "]       # both "Actual" lines, then words
    late_number = [line("2026-10-29 12:30:01", "Fed's Warsh: rates stay where they are"),
                   line("2026-10-29 12:30:05", "Fed Funds Rate kept at 4%")]
    fed = H.matching(late_number, at("2026-10-29 12:30"), *H.window("USD", ["Federal Funds Rate"]))
    assert fed[0]["title"].endswith("4%")


def test_a_release_no_rule_names_gets_no_window_and_the_boj_gets_two_hours():
    assert H.window("USD", ["Treasury Currency Report"]) is None
    words, since, until = H.window("JPY", ["BOJ Policy Rate"])
    assert since == pd.Timedelta(seconds=-30) and until == pd.Timedelta(hours=2)
    assert H.window("USD", ["CPI m/m"])[2] == pd.Timedelta(minutes=10)


@pytest.mark.parametrize("raw, clean", [
    ("FinancialJuice: BoC Rate Decision Actual 2.5% (Forecast 2.5%, Previous 2.25%)",
     "BoC Rate Decision Actual 2.5% (Forecast 2.5%, Previous 2.25%)"),
    ("‼ BREAKING: BOC RATE DECISION ACTUAL 2.5% (FORECAST 2.5%, PREVIOUS 2.25%) $MACRO|FJ",
     "BOC RATE DECISION ACTUAL 2.5% (FORECAST 2.5%, PREVIOUS 2.25%)"),
    ("❗ ECB INTEREST RATE ACTUAL 2.65% (FORECAST 2.65%, PREVIOUS 2.65%) $MACRO|FJ",
     "ECB INTEREST RATE ACTUAL 2.65% (FORECAST 2.65%, PREVIOUS 2.65%)"),
    ("*FED LEAVES INTEREST RATE UNCHANGED (@WalterBloomberg)", "FED LEAVES INTEREST RATE UNCHANGED"),
    ("BOJ holds short-term interest rate target at 1.25%|FJ",
     "BOJ holds short-term interest rate target at 1.25%"),
])
def test_a_line_loses_its_tags_and_keeps_its_words(raw, clean):
    assert H.clean(raw) == clean


@pytest.mark.parametrize("text, number", [
    ("US Interest Rate Decision Actual 4% (Forecast 4%, Previous 4%)", 4.0),
    ("US 30-Year Mortgage Rate Actual 6.25% (Previous 6.31%)", 6.25),   # not the 30
    ("US ADVANCE Q3 GDP +3.4% (CONSENSUS +3.2%)", 3.4),          # Q3 is not the number
    ("US Non-Farm Payrolls Actual 245K (Forecast 180K, Previous 150K)", 245_000.0),
    ("US Initial Jobless Claims Actual -1.2% (Forecast 0.5%)", -1.2),
    ("New Zealand's Reserve Bank keeps OCR steady at 2.75%", 2.75),
    ("FED LEAVES INTEREST RATE UNCHANGED", None),
])
def test_the_number_a_line_gives_as_the_result(text, number):
    assert H.headline_number(text) == number


def test_the_official_number_is_checked_against_the_lines_about_its_own_title():
    lines = [dict(text=H.clean(x["title"])) for x in GDP_DAY[1:3]]
    assert H.agrees("3.4%", "USD", "Advance GDP q/q", lines) is True
    assert H.agrees("3.5%", "USD", "Advance GDP q/q", lines) is False
    # the GDP report's quarterly "Core PCE Prices" is not the monthly Core PCE Price Index m/m
    assert H.agrees("2.5%", "USD", "Core PCE Price Index m/m", lines) is None
    assert H.agrees("3.4%", "USD", "Advance GDP q/q", lines[:1]) is None   # only the PCE line
    assert H.agrees("4.00%", "USD", "Federal Funds Rate",
                    [dict(text="US Interest Rate Decision Actual 4% (Forecast 4%)")]) is True
    assert H.agrees(None, "USD", "Advance GDP q/q", lines) is None


JOBS_DAY = [dict(text=s) for s in (           # three numbers typed within seconds of each other
    "US Unemployment Rate Actual 4.2% (Forecast 4.2%, Previous 4.2%)",
    "US Non-Farm Payrolls Actual 64K (Forecast 63K, Previous 29K)",
    "US Average Hourly Earnings MoM Actual 0.3% (Forecast 0.3%, Previous 0.1%)",
    "US Average Hourly Earnings YoY Actual 3.8% (Forecast 3.7%, Previous 3.7%)")]
CPI_DAY = [dict(text=s) for s in (
    "US CPI MoM Actual 0.3% (Forecast 0.3%)", "US CPI YoY Actual 3.1% (Forecast 3.0%)",
    "US Core CPI MoM Actual 0.4% (Forecast 0.3%)", "US Core CPI YoY Actual 3.0% (Forecast 3.0%)")]


@pytest.mark.parametrize("lines, title, actual, want", [
    (JOBS_DAY, "Non-Farm Employment Change", "64K", True),     # not the 4.2 typed first
    (JOBS_DAY, "Non-Farm Employment Change", "65K", False),
    (JOBS_DAY, "Unemployment Rate", "4.2%", True),
    (JOBS_DAY, "Average Hourly Earnings m/m", "0.3%", True),
    (JOBS_DAY, "Average Hourly Earnings m/m", "3.8%", False),   # that is the yearly line
    (CPI_DAY, "CPI m/m", "0.3%", True),
    (CPI_DAY, "CPI m/m", "0.4%", False),                        # 0.4 is CORE's
    (CPI_DAY, "Core CPI m/m", "0.4%", True),
    (CPI_DAY, "CPI y/y", "3.1%", True),
    (CPI_DAY, "CPI y/y", "0.3%", False),                        # 0.3 is the MONTHLY line's
    (CPI_DAY, "Core CPI y/y", "3.0%", True),
    ([dict(text="US GDP Price Index Advance Actual 2.9%")], "Advance GDP q/q", "3.4%", None),
    (JOBS_DAY[:1], "Non-Farm Employment Change", "64K", None),  # only the jobless RATE typed
    ([dict(text="US Private Non-Farm Payrolls Actual 50K"),       # two lines about payrolls:
      dict(text="US Non-Farm Payrolls Actual 64K")],              # either may be the number
     "Non-Farm Employment Change", "64K", True),
])
def test_on_a_day_of_several_numbers_each_is_checked_against_its_own_line(lines, title, actual,
                                                                          want):
    assert H.agrees(actual, "USD", title, lines) is want


RSS = (b"\xef\xbb\xbf<?xml version='1.0'?><rss><channel>"
       b"<item><title>FinancialJuice: BoC Rate Decision Actual 2.5%</title><guid>1</guid>"
       b"<pubDate>Wed, 28 Oct 2026 13:45:00 GMT</pubDate></item></channel></rss>")


def test_a_source_is_read_as_bytes_and_a_failed_or_empty_one_raises():
    seen = []

    def get(url, headers, timeout):
        seen.append((url, headers["User-Agent"], timeout))
        return 200, RSS
    got = H.fetch("financialjuice", get)
    assert got[0]["published"] == "2026-10-28T13:45:00+00:00" and got[0]["id"] == "1"
    assert seen[0][0] == H.SOURCES["financialjuice"].url and seen[0][2] == H.TIMEOUT
    with pytest.raises(IOError, match="HTTP 503"):
        H.fetch("financialjuice", lambda u, h, t: (503, b""))
    with pytest.raises(ValueError, match="no headlines"):
        H.fetch("walterbloomberg", lambda u, h, t: (200, b"<html>moved</html>"))


# ── the archive: every line the alerts read ─────────────────────────────────────────────────────
ARCH = [dict(x, id=f"fj{k}") for k, x in enumerate(GDP_DAY)]   # each line its own id


def _rows(db, sql):
    import sqlite3
    c = sqlite3.connect(str(db))
    try:
        return c.execute(sql).fetchall()
    finally:
        c.close()


def test_the_archive_keeps_each_line_once_and_times_only_lines_after_the_first_download(tmp_path):
    db = tmp_path / "news" / "headlines.sqlite"
    a = [ARCH[0], ARCH[1]]
    assert H.archive(db, "financialjuice", a, at("2026-10-29 12:29:20"), 310) == 2
    assert H.archive(db, "financialjuice", ARCH[:3], at("2026-10-29 12:30:20"), 290) == 1
    assert H.archive(db, "financialjuice", ARCH[:3], at("2026-10-29 12:31:20"), 300) == 0
    items = _rows(db, "SELECT item_id, backlog, first_seen FROM items ORDER BY first_seen")
    assert [(i, b) for i, b, _ in items] == [(ARCH[0]["id"], 1), (ARCH[1]["id"], 1),
                                             (ARCH[2]["id"], 0)]
    assert items[2][2] == at("2026-10-29 12:30:20").isoformat()   # seen on the download it came
    polls = _rows(db, "SELECT ok, status, n, new, ms FROM polls ORDER BY at")
    assert polls == [(1, 200, 2, 2, 310), (1, 200, 3, 1, 290), (1, 200, 3, 0, 300)]
    assert _rows(db, "SELECT value FROM meta WHERE key = 'start'") == [
        (at("2026-10-29 12:29:20").isoformat(),)]


def test_a_failed_download_is_recorded_and_the_next_good_one_is_still_the_backlog(tmp_path):
    db = tmp_path / "headlines.sqlite"
    assert H.archive(db, "financialjuice", None, at("2026-10-29 12:29:20"), 8000, 0,
                     "ReadTimeout: read timed out") == 0
    assert H.archive(db, "financialjuice", None, at("2026-10-29 12:30:20"), 120, 429,
                     "OSError: FinancialJuice RSS: HTTP 429") == 0
    assert H.archive(db, "financialjuice", ARCH[:2], at("2026-10-29 12:31:20"), 300) == 2
    assert _rows(db, "SELECT ok, status, n, error FROM polls ORDER BY at") == [
        (0, 0, 0, "ReadTimeout: read timed out"), (0, 429, 0, "OSError: FinancialJuice RSS: HTTP 429"),
        (1, 200, 2, None)]
    assert _rows(db, "SELECT DISTINCT backlog FROM items") == [(1,)]   # no good download before


def test_each_source_has_its_own_backlog(tmp_path):
    db = tmp_path / "headlines.sqlite"
    H.archive(db, "financialjuice", ARCH[:1], at("2026-10-29 12:29:20"), 300)
    H.archive(db, "financialjuice_telegram", ARCH[:1], at("2026-10-29 12:30:20"), 300)
    H.archive(db, "financialjuice_telegram", ARCH[:2], at("2026-10-29 12:31:20"), 300)
    assert _rows(db, "SELECT source, backlog FROM items ORDER BY source, first_seen") == [
        ("financialjuice", 1), ("financialjuice_telegram", 1), ("financialjuice_telegram", 0)]


def test_an_archive_that_cannot_be_written_never_raises(tmp_path):
    blocker = tmp_path / "news"
    blocker.write_text("a file where the folder should be")
    assert H.archive(blocker / "headlines.sqlite", "financialjuice", ARCH,
                     at("2026-10-29 12:30:20"), 300) is None


def test_the_archive_is_the_headline_probes_format_and_its_report_reads_it(tmp_path):
    from experiments import headline_probe, headline_report as hr
    assert headline_probe.SCHEMA is H.ARCHIVE_SCHEMA                  # one copy of the format
    db = tmp_path / "headlines.sqlite"
    H.archive(db, "financialjuice", ARCH[:2], at("2026-10-29 12:29:20"), 300)
    H.archive(db, "financialjuice", ARCH, at("2026-10-29 12:46:20"), 300)
    items, polls, start = hr.load(db)
    assert len(items) == 5 and len(polls) == 2 and start == at("2026-10-29 12:29:20")
    t = hr.timed(items)
    assert list(t["item_id"]) == [x["id"] for x in ARCH[2:]]       # the backlog is not timed
    assert t["printed"].notna().all()                                 # printed times believed


def test_the_archives_report_is_written_where_asked_and_leaves_the_probes_alone(tmp_path,
                                                                                monkeypatch):
    from experiments import headline_report as hr
    monkeypatch.setattr(hr, "CALENDARS", (tmp_path / "no-calendar",))
    monkeypatch.setattr(hr, "ALERTS_DB", tmp_path / "no-alerts.db")
    monkeypatch.setattr(hr, "OUT", tmp_path / "probe" / "report.txt")
    db = tmp_path / "headlines.sqlite"
    H.archive(db, "financialjuice", ARCH[:2], at("2026-10-29 12:29:20"), 300)
    H.archive(db, "financialjuice", ARCH, at("2026-10-29 12:46:20"), 300)
    out = tmp_path / "archive_report.txt"
    hr.main(["--db", str(db), "--no-prices", "--out", str(out)])
    assert out.read_text(encoding="utf-8").startswith("HEADLINE PROBE")
    assert "3 timed headlines from 1 sources" in out.read_text(encoding="utf-8")
    assert not (tmp_path / "probe" / "report.txt").exists()
