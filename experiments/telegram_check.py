"""
TELEGRAM CHANNELS -- which free public channels carry the news that moves markets, how fast, and
how much else they post. RUN IT ON THE SERVER: Telegram does not open from Pakistan, so the dev
PC cannot reach it. No login and no key: a public channel has a web page, t.me/s/<name>, showing
its latest messages with Telegram's own time, and ?before=<id> pages back through its history --
so a channel is judged on news that already happened, without waiting a week to record it.

For each channel, back to START:
  posts a day, and how many of them the draft keyword rules flag (how noisy it would be)
  per EVENT -- every critical release in the calendar, plus the unscheduled ones in UNSCHEDULED:
  its first post about it, in seconds after the event by Telegram's clock (negative = before),
  and the post itself, to check by eye that it is really about it
  CHANCE -- the same words in the same window on the other days: a channel that "carries" 18:00
  every day carries nothing
FinancialJuice's RSS from the headline test (22-30 Oct) rides along as the benchmark.

    PYTHONPATH=. .venv/bin/python experiments/telegram_check.py
    PYTHONPATH=. .venv/bin/python experiments/telegram_check.py --channels WalterBloomberg,bloomberg
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from experiments import headline_report as hr
from experiments.headline_probe import UA, http_get, parse_telegram

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "telegram_check"
START = pd.Timestamp("2026-10-20", tz="UTC")
END = pd.Timestamp("2026-11-01", tz="UTC")
PAUSE_S = 1.0                     # between page requests: a reader, not a scraper
MAX_PAGES = 200                   # ~20 posts a page: 4,000 posts is enough for 12 days
BENCHMARK = "fj_rss (test)"

# Free public channels to try. Only the first two were in the headline test; the rest are names
# that could not be checked from here (Telegram does not open from Pakistan), so the run says
# which exist. Add any channel by its t.me name.
CANDIDATES = [
    ("WalterBloomberg", "Bloomberg-terminal headlines; in the 8.5-day test"),
    ("FinancialJuice", "FinancialJuice's own channel; in the test"),
    ("bloomberg", "Bloomberg's official channel, about 3 a day"),
    ("investingcom", "Investing.com"),
    ("fxstreetnews", "FXStreet"),
    ("forexlive", "ForexLive, now investingLive"),
    ("WatcherGuru", "'JUST IN' headlines: markets, crypto, politics"),
    ("unusual_whales", "markets"),
    ("zerohedge", "markets"),
    ("Cointelegraph", "crypto news"),
    ("disclosetv", "breaking news, tabloid style"),
    ("insiderpaper", "breaking news"),
    ("BNONews", "breaking news"),
    ("FaytuksNetwork", "geopolitics"),
    ("ClashReport", "conflict news"),
    ("IranIntl_En", "Iran International, English"),
]

# News with no calendar time: the moment the market moved, the words a post about it uses, and
# how far before and after a post still counts (an early warning is the point here).
UNSCHEDULED = [
    dict(t=pd.Timestamp("2026-10-21 18:44", tz="UTC"), currency="-", level="unscheduled",
         titles="Gulf embassies: US ordered departure (gold fell from 18:44 UTC)",
         numbers="FinancialJuice RSS had it at 18:47:50",
         words=r"ordered departure|departure of|evacuat|embass|security alert|u\.?s\.? mission",
         since=pd.Timedelta(minutes=-90), until=pd.Timedelta(minutes=60)),
]

_POST_ID = re.compile(r'data-post="[^"/]+/(\d+)"')


def _get(url: str, fetch: Callable, sleep: Callable, tries: int = 4):
    """-> (status, body). Waits out a 429 (Telegram's "slow down") rather than giving up."""
    status, body = 0, b""
    for _ in range(tries):
        try:
            status, body, headers = fetch(url, {"User-Agent": UA})
        except Exception as e:                       # no network, a timeout: report, go on
            return 0, f"{type(e).__name__}: {e}".encode()
        if status != 429:
            return status, body
        try:
            wait = float((headers or {}).get("Retry-After", 20))
        except (TypeError, ValueError):
            wait = 20.0
        sleep(wait)
    return status, body


def history(name: str, start=START, fetch: Callable = http_get, sleep: Callable = time.sleep,
            pause: float = PAUSE_S, max_pages: int = MAX_PAGES):
    """A public channel's posts back to `start`, newest page first, each page asking for the
    posts before the oldest one seen (photo-only posts have no words but still count for paging).
    -> (DataFrame [id, t, title], what happened: ok | not a public channel | HTTP n | stopped)."""
    rows: List[dict] = []
    before = None
    for page in range(max_pages):
        url = f"https://t.me/s/{name}" + (f"?before={before}" if before else "")
        status, body = _get(url, fetch, sleep)
        if status != 200:
            if rows:
                return _frame(rows), f"ok, stopped at HTTP {status}"
            return _frame(rows), f"HTTP {status}" if status else body.decode()[:80]
        text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
        ids = [int(x) for x in _POST_ID.findall(text)]
        if not ids:
            return _frame(rows), "ok" if rows else "not a public channel"
        for m in parse_telegram(text):
            if m["published"]:
                rows.append(dict(id=int(m["id"].rsplit("/", 1)[1]),
                                 t=pd.Timestamp(m["published"]), title=m["title"]))
        oldest = min(ids)
        if oldest <= 1 or (rows and min(r["t"] for r in rows) < start):
            return _frame(rows), "ok"
        before = oldest
        sleep(pause)
    return _frame(rows), f"ok, stopped after {max_pages} pages"


def _frame(rows) -> pd.DataFrame:
    d = pd.DataFrame(rows, columns=["id", "t", "title"])
    return d.drop_duplicates("id").sort_values("t").reset_index(drop=True)


def _benchmark(probe_db) -> tuple:
    """FinancialJuice RSS as the headline test recorded it -> (rows as sources, start, end)."""
    if probe_db is None or not Path(probe_db).is_file():
        return pd.DataFrame(columns=["source", "title", "t", "seen"]), None, None
    items, polls, pstart = hr.load(probe_db)
    d = hr.timed(items)
    d = d[d["source"] == "fj_rss"][["title", "t", "seen"]].assign(source=BENCHMARK)
    return d, pstart, polls["at"].max()


def check(channels: List[str], start=START, end=END, fetch: Callable = http_get,
          sleep: Callable = time.sleep, probe_db=hr.DB, levels=("critical",), calendars=None,
          out_dir=OUT_DIR, log: Callable = print) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msgs, status = {}, {}
    for name in channels:
        log(f"reading {name} ...")
        df, st = history(name, start, fetch=fetch, sleep=sleep)
        msgs[name], status[name] = df[(df["t"] >= start) & (df["t"] < end)], st
        if len(df):
            df.to_csv(out_dir / f"{name}.csv", index=False)
    live = [n for n in channels if len(msgs[n])]
    d = pd.concat([m.assign(source=n, seen=m["t"])[["source", "title", "t", "seen"]]
                   for n, m in msgs.items() if len(m)] or
                  [pd.DataFrame(columns=["source", "title", "t", "seen"])], ignore_index=True)
    bench, b_start, b_end = _benchmark(probe_db)
    sources = live + ([BENCHMARK] if len(bench) else [])
    if len(bench):
        d = pd.concat([d, bench], ignore_index=True)
    ev, cal = hr.load_calendar(start, end - hr.CRIT_WINDOW[1], levels,
                               hr.CALENDARS if calendars is None else calendars)
    groups = pd.concat([pd.DataFrame(UNSCHEDULED), hr.release_groups(ev)], ignore_index=True)
    groups = groups.sort_values("t").reset_index(drop=True)
    deliv = hr.release_delivery(groups, d, sources)
    chance = hr.release_chance(groups, d, live, start, end)
    if len(bench):
        chance = pd.concat([chance, hr.release_chance(groups, d, [BENCHMARK], b_start, b_end)])
    days = (end - start).total_seconds() / 86400

    rows = {}
    for name in channels:
        m = msgs[name]
        span = (end - max(start, m["t"].min())).total_seconds() / 86400 if len(m) else np.nan
        flag = pd.Series(False, index=m.index)
        p1 = pd.Series(False, index=m.index)
        for rule, pat in hr.RULES.items():
            hit = m["title"].str.contains(pat, case=False, regex=True)
            flag |= hit
            if rule.startswith("P1"):
                p1 |= hit
        rows[name] = dict(status=status[name], posts=len(m),
                          per_day=round(len(m) / span, 1) if len(m) else 0,
                          flagged_per_day=round(flag.sum() / span, 1) if len(m) else 0,
                          p1_per_day=round(p1.sum() / span, 1) if len(m) else 0,
                          since=f"{m['t'].min():%d %b %H:%M}" if len(m) else "-")
    lines = [f"TELEGRAM CHANNELS, {start:%d %b} -> {end:%d %b} UTC ({days:.0f} days). Calendar: {cal}",
             "  flagged = posts the draft keyword rules would flag; p1 = the security and rates "
             "rules alone",
             pd.DataFrame(rows).T.to_string()]
    n = int(groups["words"].notna().sum())
    lines.append(f"\nEVENTS: {n} -- seconds after the event to each channel's first post about it "
                 "(Telegram's own clock; negative = before it)")
    for gi, g in groups.iterrows():
        lines.append(f"\n{g['t']:%a %d %b %H:%M} UTC  {g['currency']}  {g['titles']}  "
                     f"[{g['level']}]")
        if isinstance(g.get("numbers"), str) and g["numbers"]:
            lines.append(f"  {g['numbers']}")
        got = deliv[deliv["release"] == gi].sort_values("delay_s")
        for r in got.itertuples():
            lines.append(f"  {r.source:18s} {r.delay_s:+7.0f} s | "
                         f"{' '.join(str(r.headline).split())[:100]}")
        missing = [s for s in sources if s not in set(got["source"])]
        if missing:
            lines.append("  nothing: " + ", ".join(missing))
    summ = {}
    for s in sources:
        r = deliv[deliv["source"] == s]
        first = 0
        for gi in r["release"]:
            first += int(r[r["release"] == gi]["delay_s"].iat[0]
                         == deliv[deliv["release"] == gi]["delay_s"].min())
        summ[s] = dict(carried=f"{len(r)}/{n}", first=first,
                       median_s=round(r["delay_s"].median(), 0) if len(r) else np.nan,
                       chance_pct=chance.get(s, np.nan),
                       per_day=rows[s]["per_day"] if s in rows else np.nan)
    lines += ["\nPER CHANNEL: carried = posted about the event, first = before every other "
              "channel, chance_pct = the same words in the same window on the other days",
              pd.DataFrame(summ).T.sort_values(["first", "median_s"], ascending=[False, True])
              .to_string() if summ else "no channel had any posts"]
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--channels", default=",".join(c for c, _ in CANDIDATES),
                    help="t.me names, comma-separated (default: the CANDIDATES list)")
    ap.add_argument("--levels", default="critical", help="calendar levels, e.g. critical,high")
    a = ap.parse_args(argv)
    print(check([c.strip() for c in a.channels.split(",") if c.strip()],
                levels=tuple(x.strip() for x in a.levels.split(","))))
    print(f"\n(also written to {OUT_DIR / 'report.txt'}, each channel's posts beside it)")


if __name__ == "__main__":
    main()
