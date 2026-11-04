"""
BREAKING NEWS CHECK -- three ways to catch critical news that is NOT on the calendar, replayed on the
days the server recorded, BEFORE any of them is wired into the alerts.

  words         a FinancialJuice line that meets the strict BREAKING list (quant/news/headlines.py)
  move          a sharp move on Gold, USDJPY or BTC -- 5 minutes at least its floor and 6x the
                usual (headline_report.find_moves) -- sent with FinancialJuice's lines around it
  words + move  a BREAKING line from 15 min before a sharp move to 10 min after it

A move near a scheduled high/critical release (any currency) -- the release from 5 min before the
move's window opens to 15 min after -- is that release's RESULT, which the headline results already
send: it is counted apart and left out.
Per way: alerts a day, every one listed, and whether it caught the 21 Oct Gulf story (gold fell from
18:44 UTC; FinancialJuice's feed had "ordered departure" at 18:47:50).

A move is KNOWN 5 minutes after it starts (the 5-minute return that defines it), so that is when
the move ways could send it; a words alert goes when its line is printed.

Reads, on the server: FinancialJuice's Telegram posts saved by experiments/telegram_check.py
(results/telegram_check/FinancialJuice.csv, 20 Oct - 1 Nov 2026: the same desk as the RSS feed,
~40 s later), the 1m candles of the three markets (the alerts' own feed) and the monitor's calendar.

    PYTHONPATH=. .venv/bin/python experiments/breaking_news_check.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiments import headline_report as hr
from quant.news import headlines as H

HERE = Path(__file__).resolve().parent
LINES = HERE / "results" / "telegram_check" / "FinancialJuice.csv"
OUT = HERE / "results" / "breaking_news_check" / "report.txt"
BEFORE = pd.Timedelta(minutes=15)          # a BREAKING line this long before a move counts for it
AFTER = pd.Timedelta(minutes=10)           # ... or this long after it
KNOWN = pd.Timedelta(minutes=5)            # a move is known 5 min after it starts
RELEASE_AFTER = pd.Timedelta(minutes=15)   # a move this soon after a scheduled release is its result
GULF = pd.Timestamp("2026-10-21 18:44", tz="UTC")
GULF_WINDOW = (pd.Timedelta(minutes=-30), pd.Timedelta(minutes=60))
WAYS = ("words", "move", "words + move")


def load_lines(path=LINES) -> pd.DataFrame:
    """FinancialJuice's posts: t (UTC), text (as a person reads it), rule (BREAKING name or None)."""
    d = pd.read_csv(path)
    d["t"] = pd.to_datetime(d["t"], utc=True, format="ISO8601")
    d["text"] = d["title"].astype(str).map(H.clean)
    d["rule"] = d["text"].map(H.breaking_rule)
    return d.sort_values("t").reset_index(drop=True)[["t", "text", "rule"]]


def find_all_moves(bars: Dict[str, pd.DataFrame], releases) -> pd.DataFrame:
    """Every sharp move per market; `release` = it starts within 15 min after a scheduled one."""
    parts = [hr.find_moves(b, m) for m, b in bars.items() if len(b)]
    mv = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=["market", "start", "move_pct"]))
    releases = list(releases)
    rel = np.sort(hr._ns(pd.Series(releases))) if releases else np.array([], dtype="int64")
    st = hr._ns(mv["start"]) if len(mv) else np.array([], dtype="int64")
    # a move is dated from the START of its 5-minute window, which can open a few minutes before
    # the release that caused it: a release from 5 min before that start to 15 min after counts
    lo = np.searchsorted(rel, st - KNOWN.value, side="left")
    mv["release"] = ((lo < len(rel)) & (rel[np.minimum(lo, len(rel) - 1)]
                                        <= st + RELEASE_AFTER.value)) if len(rel) else False
    return mv.sort_values("start").reset_index(drop=True)


SIZES = (1.0, 1.5, 2.0, 3.0)               # a move at least this many times its floor
TOGETHER = pd.Timedelta(minutes=5)         # another market moving this close counts as "together"


def annotate(mv: pd.DataFrame) -> pd.DataFrame:
    """The moves that were not releases, with `size` (times its market's floor) and `together`
    (another market's sharp move started within 5 minutes)."""
    m = mv[~mv["release"]].copy()
    m["size"] = [abs(p) / 100 / hr.MOVE_FLOOR[mk] for mk, p in zip(m["market"], m["move_pct"])]
    s = pd.to_datetime(m["start"], utc=True)
    m["together"] = [bool(((m["market"] != mk) & ((s - t).abs() <= TOGETHER)).any())
                     for mk, t in zip(m["market"], s)]
    return m


def sizes_table(mv: pd.DataFrame, start, end) -> pd.DataFrame:
    """How big a move must be: alerts a day when it is at least N x its floor (headline_report's
    MOVE_FLOOR), alone or with a second market's sharp move within 5 minutes -- and whether each
    rule caught the Gulf story. Releases are left out, as everywhere here."""
    days = max((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(days=1), 1e-9)
    m = annotate(mv)
    rows = []
    for k in SIZES:
        big = m[m["size"] >= k] if len(m) else m
        two = big[big["together"]] if len(big) else big
        gulf = (lambda d: bool(len(d)) and bool(((pd.to_datetime(d["start"], utc=True) >=
                                                  GULF - pd.Timedelta(minutes=10)) &
                                                 (pd.to_datetime(d["start"], utc=True) <=
                                                  GULF + pd.Timedelta(minutes=30))).any()))
        rows.append({"move at least": f"{k:g}x floor", "one market, a day": round(len(big) / days, 1),
                     "two markets, a day": round(len(two) / days, 1),
                     "Gulf (one)": "caught" if gulf(big) else "missed",
                     "Gulf (two)": "caught" if gulf(two) else "missed"})
    return pd.DataFrame(rows).set_index("move at least")


def near(lines: pd.DataFrame, at, before=BEFORE, after=AFTER) -> pd.DataFrame:
    """The lines from `before` ahead of `at` to `after` past it, BREAKING ones first."""
    w = lines[(lines["t"] >= at - before) & (lines["t"] <= at + after)]
    return w.assign(_k=w["rule"].isna()).sort_values(["_k", "t"]).drop(columns="_k")


def alerts(lines: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    """One row per alert each way would have sent: way, at (when it would go), market, text."""
    rows = [dict(way="words", at=r.t, market="", text=r.text, rule=r.rule)
            for r in lines[lines["rule"].notna()].itertuples()]
    for m in mv[~mv["release"]].itertuples():
        known = pd.Timestamp(m.start) + KNOWN
        around = near(lines, pd.Timestamp(m.start))
        top = around["text"].iat[0] if len(around) else "(no FinancialJuice line near it)"
        rows.append(dict(way="move", at=known, market=f"{m.market} {m.move_pct:+.2f}%", text=top,
                         rule=around["rule"].iat[0] if len(around) else None))
        hit = around[around["rule"].notna()]
        if len(hit):
            rows.append(dict(way="words + move", at=max(known, hit["t"].iat[0]),
                             market=f"{m.market} {m.move_pct:+.2f}%", text=hit["text"].iat[0],
                             rule=hit["rule"].iat[0]))
    return (pd.DataFrame(rows, columns=["way", "at", "market", "text", "rule"])
            .sort_values("at").reset_index(drop=True))


def per_day(al: pd.DataFrame, start, end) -> pd.DataFrame:
    """Alerts each way would have sent, per UTC day of the window."""
    days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"),
                         freq="D", inclusive="left")
    t = pd.DataFrame(0, index=[d.strftime("%a %d %b") for d in days], columns=list(WAYS))
    for r in al.itertuples():
        k = pd.Timestamp(r.at).floor("D").strftime("%a %d %b")
        if k in t.index:
            t.loc[k, r.way] += 1
    t.loc["a day (mean)"] = (t.sum() / max(len(days), 1)).round(1)
    return t


def first_alert(al: pd.DataFrame, at=GULF, window=GULF_WINDOW) -> Dict[str, Optional[dict]]:
    """Per way, its first alert inside `window` around `at` -- or None: did it catch the story?"""
    w = al[(al["at"] >= at + window[0]) & (al["at"] <= at + window[1])]
    out = {}
    for way in WAYS:
        g = w[w["way"] == way]
        out[way] = g.iloc[0].to_dict() if len(g) else None
    return out


def biggest_move(bars: Dict[str, pd.DataFrame], at, start, end) -> str:
    """The largest 15-minute reaction of the three markets after `at`, as 'Gold -0.42% (8.1x)'."""
    best = None
    for m, b in bars.items():
        if not len(b):
            continue
        pct, times = hr.reaction(b, pd.Timestamp(at).ceil("min"), start, end, horizons=(15,))[15]
        if np.isfinite(times) and (best is None or times > best[2]):
            best = (m, pct, times)
    return "-" if best is None else f"{best[0]} {best[1]:+.2f}% ({best[2]:.1f}x)"


def report(lines: pd.DataFrame, bars: Dict[str, pd.DataFrame], releases, start, end) -> str:
    mv = find_all_moves(bars, releases)
    if len(mv):                       # the candles reach a day back for the usual move: not ours
        s = pd.to_datetime(mv["start"], utc=True)
        mv = mv[(s >= pd.Timestamp(start)) & (s < pd.Timestamp(end))].reset_index(drop=True)
    al = alerts(lines, mv)
    days = (pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(days=1)
    out = [f"BREAKING NEWS CHECK, {pd.Timestamp(start):%d %b} -> {pd.Timestamp(end):%d %b %Y} UTC "
           f"({days:.0f} days): {len(lines)} FinancialJuice lines, {len(mv)} sharp moves on "
           f"{', '.join(bars) or 'no markets'} ({int(mv['release'].sum())} right after a scheduled "
           f"release: results, left out)",
           "  words = a line in the strict BREAKING list; move = a sharp move, sent with the "
           "nearest line; words + move = both within -15/+10 min",
           "\nALERTS A DAY", per_day(al, start, end).to_string(),
           "\nHOW BIG A MOVE: alerts a day if it must be at least N x its floor (Gold "
           f"{100 * hr.MOVE_FLOOR['Gold']:g}%, USDJPY {100 * hr.MOVE_FLOOR['USDJPY']:g}%, BTC "
           f"{100 * hr.MOVE_FLOOR['BTC']:g}% in 5 min), alone or with a second market within 5 min",
           sizes_table(mv, start, end).to_string(),
           "\nTHE 21 OCT GULF STORY (gold fell from 18:44 UTC)"]
    for way, a in first_alert(al).items():
        out.append(f"  {way:13s} " + ("missed" if a is None else
                                      f"{pd.Timestamp(a['at']):%H:%M:%S}  {a['text'][:90]}"))
    out.append("\nWORDS: every line the BREAKING list flags, and the biggest move of the three "
               "markets in the 15 min after it (x = times the usual for that minute)")
    for r in al[al["way"] == "words"].itertuples():
        out.append(f"  {pd.Timestamp(r.at):%a %d %b %H:%M}  [{r.rule}]  {r.text[:100]}  "
                   f"-> {biggest_move(bars, r.at, start, end) if bars else '-'}")
    out.append("\nMOVES: the sharp moves that were not releases and either were at least 1.5x "
               "their floor or came with a second market, with the FinancialJuice lines 15 min "
               "before to 10 min after (* = in the BREAKING list)")
    big = annotate(mv)
    big = big[(big["size"] >= 1.5) | big["together"]]
    for m in big.itertuples():
        out.append(f"  {pd.Timestamp(m.start):%a %d %b %H:%M}  {m.market:6s} {m.move_pct:+.2f}%  "
                   f"({m.size:.1f}x floor{', with another market' if m.together else ''})")
        for r in near(lines, pd.Timestamp(m.start)).head(3).itertuples():
            out.append(f"      {'*' if r.rule else ' '} {pd.Timestamp(r.t):%H:%M}  {r.text[:100]}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lines", default=str(LINES), help="FinancialJuice posts (telegram_check CSV)")
    ap.add_argument("--no-prices", action="store_true", help="words only: no candles fetched")
    a = ap.parse_args(argv)
    lines = load_lines(a.lines)
    start, end = lines["t"].min().floor("D"), lines["t"].max().ceil("D")
    bars = {} if a.no_prices else hr.fetch_bars(start, end)
    # medium releases too: ISM or the oil inventories move markets, and that is not breaking news
    ev, _ = hr.load_calendar(start, end, levels=("medium", "high", "critical"))
    text = report(lines, bars, ev["t"] if len(ev) else [], start, end)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"\n(also written to {OUT})")


if __name__ == "__main__":
    main()
