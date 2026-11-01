"""
experiments/telegram_check.py -- reading a public Telegram channel's history back through its
web pages, and timing each channel against the events. Hand-built pages, no network.
"""
import re

import pandas as pd

from experiments import headline_report as hr
from experiments import telegram_check as tc


def at(s):
    return pd.Timestamp(s, tz="UTC")


def _page(name, posts, title=None, subs=None):
    """A t.me/s page: posts = [(id, "2026-10-21 18:46:05", text or None for a photo)], with the
    channel's header (its title and subscriber count) when given."""
    out = ["<html><head>"]
    if title:
        out.append(f'<meta property="og:title" content="{title}">')
    out.append("</head><body>")
    if subs:
        out.append(f'<div class="tgme_channel_info_counter"><span class="counter_value">{subs}'
                   f'</span> <span class="counter_type">subscribers</span></div>')
    for i, when, text in posts:
        body = (f'<div class="tgme_widget_message_text js-message_text" dir="auto">{text}</div>'
                if text else '<div class="tgme_widget_message_photo"></div>')
        out.append(f'<div class="tgme_widget_message_wrap"><div class="tgme_widget_message '
                   f'js-widget_message" data-post="{name}/{i}">{body}<div class="tgme_widget_'
                   f'message_footer"><a class="tgme_widget_message_date" href="x"><time '
                   f'datetime="{at(when).isoformat()}" class="time">x</time></a></div></div></div>')
    return ("".join(out) + "</body></html>").encode()


class Web:
    """Pages by URL; anything else is a channel with no web preview. Records every request."""

    def __init__(self, pages, status=None):
        self.pages, self.status, self.asked = pages, dict(status or {}), []

    def __call__(self, url, headers):
        self.asked.append(url)
        if url in self.status:
            code = self.status[url].pop(0) if isinstance(self.status[url], list) else \
                self.status[url]
            if isinstance(self.status[url], list) and not self.status[url]:
                del self.status[url]
            if code != 200:
                return code, b"", {"Retry-After": "5"}
        return 200, self.pages.get(url, b"<html>no preview</html>"), {}


def test_history_pages_back_until_the_start_and_photos_still_count_for_paging():
    web = Web({                       # each page's OLDEST post is a photo: paging must use it
        "https://t.me/s/Chan": _page("Chan", [(47, "2026-10-21 19:00", None),
                                             (48, "2026-10-21 19:10", "C"),
                                             (50, "2026-10-21 19:20", "D")]),
        "https://t.me/s/Chan?before=47": _page("Chan", [(44, "2026-10-21 18:00", None),
                                                        (45, "2026-10-21 18:30", "B")]),
        "https://t.me/s/Chan?before=44": _page("Chan", [(40, "2026-10-19 23:00", "old"),
                                                        (43, "2026-10-20 09:00", "A")]),
    })
    sleeps = []
    d, st = tc.history("Chan", at("2026-10-20"), fetch=web, sleep=sleeps.append)
    assert st == "ok" and list(d["title"]) == ["old", "A", "B", "C", "D"]   # no photo rows
    assert web.asked[-1].endswith("?before=44") and len(web.asked) == 3     # stops past START
    assert sleeps == [tc.PAUSE_S, tc.PAUSE_S]                               # paced


def test_a_channel_with_no_public_page_or_an_error_is_said_so():
    assert tc.history("Nobody", at("2026-10-20"), fetch=Web({}), sleep=lambda s: None)[1] \
        == "not a public channel"
    web = Web({}, status={"https://t.me/s/Gone": 404})
    assert tc.history("Gone", at("2026-10-20"), fetch=web, sleep=lambda s: None)[1] == "HTTP 404"

    def down(url, headers):
        raise ConnectionError("no route")
    st = tc.history("Any", at("2026-10-20"), fetch=down, sleep=lambda s: None)[1]
    assert "ConnectionError" in st


def test_telegrams_slow_down_is_waited_out():
    web = Web({"https://t.me/s/Chan": _page("Chan", [(1, "2026-10-21 19:00", "only post")])},
              status={"https://t.me/s/Chan": [429, 200]})
    sleeps = []
    d, st = tc.history("Chan", at("2026-10-20"), fetch=web, sleep=sleeps.append)
    assert st == "ok" and list(d["title"]) == ["only post"] and sleeps == [5.0]


def test_each_channel_is_timed_against_the_events_and_its_chance(tmp_path, monkeypatch):
    fomc = pd.DataFrame([dict(t=at("2026-10-28 18:00"), currency="USD", level="critical",
                              title="Federal Funds Rate", actual="4.00%", forecast="4.00%")])
    monkeypatch.setattr(hr, "load_calendar", lambda s, e, lv, roots: (fomc, "test calendar"))
    fast = _page("Fast", [(1, "2026-10-19 10:00", "a post from before the window"),
                          (2, "2026-10-21 17:52", "US Embassy security alert for Saudi Arabia"),
                          (3, "2026-10-21 18:47", "US ORDERED DEPARTURE FROM GULF EMBASSIES"),
                          (4, "2026-10-28 18:00:04", "FED LEAVES RATES UNCHANGED AT 4.00%")],
                 title="Fast News &amp; Co", subs="12.3K")
    chatty = _page("Chatty", [(k, f"2026-10-{d} 18:02", f"Fed speaker {k} on the outlook")
                              for k, d in enumerate(range(20, 31), start=1)])
    web = Web({"https://t.me/s/Fast": fast, "https://t.me/s/Chatty": chatty})
    text = tc.check(["Fast", "Chatty", "Nobody"], fetch=web, sleep=lambda s: None,
                    probe_db=None, calendars=(), out_dir=tmp_path, log=lambda *a: None)
    assert "Nobody" in text and "not a public channel" in text
    fast_row = [ln for ln in text[:text.index("\nEVENTS")].splitlines()
                if ln.startswith("Fast ")][0]
    assert "Fast News & Co" in fast_row and "12.3K" in fast_row  # what the name turned out to be
    assert re.search(r"\bok\s+3\b", fast_row)                    # the 19 Oct post is not counted
    gulf = text[text.index("Gulf embassies"):text.index("Federal Funds Rate")]
    assert "Fast" in gulf and "-3120 s" in gulf                 # the warning, 52 min before
    fed = text[text.index("Federal Funds Rate"):text.index("PER CHANNEL")]
    assert "+4 s" in fed and "+120 s" in fed                    # Fast 18:00:04, Chatty 18:02
    table = text[text.index("PER CHANNEL"):]
    row = {ln.split()[0]: ln.split() for ln in table.splitlines()[1:] if ln.strip()}
    # Chatty's "Fed" at 18:02 every day fills the 18:00 window on 10 of the 11 other days (none on
    # the 31st) and the Gulf window on none of its 11: 10 of 22. Fast says nothing on other days
    assert row["Chatty"][4] == "45.0" and row["Fast"][4] == "0.0"
    assert row["Fast"][1:3] == ["2/2", "2"]                      # both events, first on both
    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == text
    assert (tmp_path / "Fast.csv").is_file()
