"""
A key never sits in a log or a state file. A failed FRED download's error message carries the
whole URL, api_key included, and that message used to be stored in the calendar's meta.json,
printed into data/news/update.log by the scheduled task, and logged by the monitor (2 Nov 2026).
The promises: the message is blanked wherever it is written; a value written before the rule is
scrubbed at the next write; the monitor's log blanks a traceback too.
"""
import json

import pandas as pd
import requests

from quant.logging_utils import redact

KEY = "abc123notarealkey"
URL_ERROR = (f"HTTPSConnectionPool(host='api.stlouisfed.org', port=443): Max retries exceeded "
             f"with url: /fred/series/observations?api_key={KEY}&file_type=json&series_id=CPIAUCSL "
             f"(Caused by SSLError(SSLEOFError(8, 'EOF occurred')))")


def test_a_key_in_a_url_is_blanked_and_the_rest_kept():
    out = redact(URL_ERROR)
    assert KEY not in out and "api_key=***&file_type=json&series_id=CPIAUCSL" in out
    assert redact("apikey=1 access_token=2 token=3 key=keep") == \
        "apikey=*** access_token=*** token=*** key=keep"


def test_the_calendar_never_writes_a_failed_fred_call_with_its_key(tmp_path, monkeypatch):
    from quant.news import update as U
    from quant.news.calendar import NewsStore
    st = NewsStore(tmp_path)
    monkeypatch.setattr(U, "keys", lambda: {"fred": KEY, "jblanked": None, "bls": None})
    for name in ("update_feed", "update_fomc", "update_official", "poll_official"):
        monkeypatch.setattr(U, name, lambda *a, **k: "ok")

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError(URL_ERROR)
    monkeypatch.setattr(U, "update_fred", boom)
    out = U.update_all(st, pd.Timestamp("2026-11-02 10:00", tz="UTC"), rebuild=False)
    line = next(x for x in out if x.startswith("fred:"))
    assert "FAILED ConnectionError" in line and KEY not in line and "api_key=***" in line
    text = (tmp_path / "meta.json").read_text(encoding="utf-8")
    assert KEY not in text and "api_key=***" in json.loads(text)["fred_error"]


def test_a_key_stored_before_the_rule_is_scrubbed_at_the_next_write(tmp_path):
    from quant.news.calendar import NewsStore
    (tmp_path / "meta.json").write_text(json.dumps({"official_error": URL_ERROR, "n": 3,
                                                    "feed_weeks": {"2026-11-01": 1}}))
    st = NewsStore(tmp_path)
    st.set_meta(feed_ok="2026-11-02T10:00:00+00:00")
    m = st.meta()
    assert KEY not in json.dumps(m) and m["n"] == 3 and m["feed_weeks"] == {"2026-11-01": 1}


def test_the_monitors_log_blanks_a_key_in_a_message_and_in_a_traceback(tmp_path):
    from quant.alerts import runner
    for h in list(runner.log.handlers):
        runner.log.removeHandler(h)
    try:
        runner.setup_logging(tmp_path)
        runner.log.warning("news fred: %s", URL_ERROR)
        try:
            raise requests.exceptions.ConnectionError(URL_ERROR)
        except Exception:
            runner.log.exception("tick failed")
        for h in runner.log.handlers:
            h.flush()
        text = (tmp_path / "alerts.log").read_text(encoding="utf-8")
        assert text.count("api_key=***") == 2 and KEY not in text   # message, traceback
    finally:
        for h in list(runner.log.handlers):
            h.close()
            runner.log.removeHandler(h)


def test_the_monitors_news_errors_are_blanked_at_the_source(tmp_path, monkeypatch):
    from quant.alerts.news import LiveNews
    from quant.news import update as U
    monkeypatch.setattr(U, "keys", lambda: {"fred": KEY, "jblanked": None, "bls": None})
    monkeypatch.setattr(U, "update_fomc", lambda *a, **k: "ok")
    monkeypatch.setattr(U, "update_official", lambda *a, **k: "ok")

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError(URL_ERROR)
    monkeypatch.setattr(U, "update_fred", boom)
    errors = LiveNews(tmp_path / "news").background(pd.Timestamp("2026-11-02 10:00", tz="UTC"))
    assert KEY not in errors["news fred"] and "api_key=***" in errors["news fred"]
