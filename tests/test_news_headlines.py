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
    assert H.agrees("2.5%", "USD", "Core PCE Price Index m/m", lines) is True
    assert H.agrees("3.4%", "USD", "Advance GDP q/q", lines[:1]) is None   # only the PCE line
    assert H.agrees("4.00%", "USD", "Federal Funds Rate",
                    [dict(text="US Interest Rate Decision Actual 4% (Forecast 4%)")]) is True
    assert H.agrees(None, "USD", "Advance GDP q/q", lines) is None


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
