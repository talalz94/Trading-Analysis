"""
BREAKING NEWS CHECK -- ways to catch critical news that is NOT on the calendar, replayed on the days
the server recorded, before (and now beside) the live alert.

  words   a FinancialJuice line in either tier of quant/news/headlines.py (TOP or BREAKING)
  move    a sharp move on Gold, USDJPY or BTC (headlines.sharp_moves), sent once KNOWN (5 min in)
          with the nearest lines
  live    what the alert sends (NEWS breaking=): a TOP line at once; a BREAKING line only with a
          sharp move from 10 min before it to 15 min after, sent once the move is known; the same
          story again within an hour is not sent; at most 8 a day

The first run (5 Nov 2026, FinancialJuice's 2,880 posts of 19 Oct - 2 Nov, a fortnight of war around
Iran and the Gulf): words 4.6 a day, a third of them routine Russia/Ukraine strikes and statements;
move 4.2 a day, and at 1.5x the floor it missed the Gulf story; words + move 0.5 a day, Gulf caught.
Hence the two tiers.

A move near a scheduled medium+ release (any currency) -- the release from 5 min before the move's
window opens to 15 min after -- is that release's RESULT, which the headline results already send:
it is left out. Per way: alerts a day, every one listed, and whether it caught the 21 Oct Gulf story
(gold fell from 18:44 UTC; FinancialJuice's feed had "ordered departure" at 18:47:50).

Reads, on the server: FinancialJuice's Telegram posts saved by experiments/telegram_check.py
(results/telegram_check/FinancialJuice.csv, the same desk as the RSS feed ~40 s later), the 1m
candles of the three markets (the alerts' own feed) and the monitor's calendar.

    PYTHONPATH=. .venv/bin/python experiments/breaking_news_check.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from experiments import headline_report as hr
from quant.news import headlines as H

HERE = Path(__file__).resolve().parent
LINES = HERE / "results" / "telegram_check" / "FinancialJuice.csv"
OUT = HERE / "results" / "breaking_news_check" / "report.txt"
NEAR = (pd.Timedelta(minutes=-15), pd.Timedelta(minutes=10))   # lines shown beside a move
RELEASE_AFTER = pd.Timedelta(minutes=15)   # a move this soon after a scheduled release is its result
GULF = pd.Timestamp("2026-10-21 18:44", tz="UTC")
GULF_WINDOW = (pd.Timedelta(minutes=-30), pd.Timedelta(minutes=60))
WAYS = ("words", "move", "live")
SIZES = (1.0, 1.5, 2.0, 3.0)               # a move at least this many times its floor
TOGETHER = pd.Timedelta(minutes=5)         # another market moving this close counts as "together"


def load_lines(path=LINES) -> pd.DataFrame:
    """FinancialJuice's posts: t (UTC), text (as a person reads it), tier (1 / 2 / None), rule."""
    d = pd.read_csv(path)
    d["t"] = pd.to_datetime(d["t"], utc=True, format="ISO8601")
    d["text"] = d["title"].astype(str).map(H.clean)
    return with_tiers(d.sort_values("t").reset_index(drop=True)[["t", "text"]])


def with_tiers(lines: pd.DataFrame) -> pd.DataFrame:
    tiers = lines["text"].map(H.breaking_tier)
    return lines.assign(tier=[t[0] if t else None for t in tiers],
                        rule=[t[1] if t else None for t in tiers])


def find_all_moves(bars: Dict[str, pd.DataFrame], releases) -> pd.DataFrame:
    """Every sharp move per market; `release` = a scheduled release from 5 min before the move's
    window opens (a move is dated from the START of its 5 minutes, which can open a few minutes
    before the release that caused it) to 15 min after."""
    parts = [hr.find_moves(b, m) for m, b in bars.items() if len(b)]
    mv = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=["market", "start", "move_pct"]))
    releases = list(releases)
    rel = np.sort(hr._ns(pd.Series(releases))) if releases else np.array([], dtype="int64")
    st = hr._ns(mv["start"]) if len(mv) else np.array([], dtype="int64")
    lo = np.searchsorted(rel, st - H.MOVE_KNOWN.value, side="left")
    mv["release"] = ((lo < len(rel)) & (rel[np.minimum(lo, len(rel) - 1)]
                                        <= st + RELEASE_AFTER.value)) if len(rel) else False
    return mv.sort_values("start").reset_index(drop=True)


def near(lines: pd.DataFrame, at, window=NEAR) -> pd.DataFrame:
    """The lines from 15 min before `at` to 10 min after it, flagged ones first."""
    w = lines[(lines["t"] >= at + window[0]) & (lines["t"] <= at + window[1])]
    return w.assign(_k=w["tier"].isna()).sort_values(["_k", "t"]).drop(columns="_k")


def live(lines: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    """What the live alert sends, from the same core rules: a TOP line at its time; a BREAKING
    line once a confirming move is known; one per story an hour; at most BREAKING_PER_DAY a day."""
    moves = mv[~mv["release"]]
    starts = pd.to_datetime(moves["start"], utc=True)
    rows = []
    for r in lines[lines["tier"].notna()].itertuples():
        if r.tier == 1:
            when, market = r.t, ""
        else:
            ok = moves[(starts >= r.t - H.MOVE_BEFORE) & (starts <= r.t + H.MOVE_AFTER)]
            if not len(ok):
                continue
            m = ok.iloc[0]
            when = max(r.t, pd.Timestamp(m["start"]) + H.MOVE_KNOWN)
            market = f"{m['market']} {m['move_pct']:+.2f}%"
        rows.append(dict(at=when, t=r.t, text=r.text, rule=r.rule, tier=int(r.tier), market=market))
    out, sent = [], []
    for r in sorted(rows, key=lambda x: x["at"]):
        if any(H.same_story(r["text"], s["text"]) and r["at"] - s["at"] <= H.STORY_GAP for s in sent):
            continue
        day = pd.Timestamp(r["at"]).floor("D")
        if sum(pd.Timestamp(s["at"]).floor("D") == day for s in sent) >= H.BREAKING_PER_DAY:
            continue
        sent.append(r)
        out.append(dict(r, way="live"))
    return pd.DataFrame(out, columns=["way", "at", "t", "market", "text", "rule", "tier"])


def alerts(lines: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    """One row per alert each way would have sent: way, at (when it would go), market, text."""
    rows = [dict(way="words", at=r.t, market="", text=r.text, rule=r.rule)
            for r in lines[lines["tier"].notna()].itertuples()]
    for m in mv[~mv["release"]].itertuples():
        around = near(lines, pd.Timestamp(m.start))
        rows.append(dict(way="move", at=pd.Timestamp(m.start) + H.MOVE_KNOWN,
                         market=f"{m.market} {m.move_pct:+.2f}%",
                         text=around["text"].iat[0] if len(around)
                         else "(no FinancialJuice line near it)",
                         rule=around["rule"].iat[0] if len(around) else None))
    lv = live(lines, mv)
    rows += lv[["way", "at", "market", "text", "rule"]].to_dict("records")
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
    """How big a move must be: alerts a day when it is at least N x its floor, alone or with a
    second market's sharp move within 5 minutes -- and whether each rule caught the Gulf story."""
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
           "  words = a line in either tier; move = a sharp move, sent with the nearest line; "
           "live = what the alert sends (a TOP line at once, a BREAKING line only with a move)",
           "\nALERTS A DAY", per_day(al, start, end).to_string(),
           "\nHOW BIG A MOVE: alerts a day if it must be at least N x its floor (Gold "
           f"{100 * hr.MOVE_FLOOR['Gold']:g}%, USDJPY {100 * hr.MOVE_FLOOR['USDJPY']:g}%, BTC "
           f"{100 * hr.MOVE_FLOOR['BTC']:g}% in 5 min), alone or with a second market within 5 min",
           sizes_table(mv, start, end).to_string(),
           "\nTHE 21 OCT GULF STORY (gold fell from 18:44 UTC)"]
    for way, a in first_alert(al).items():
        out.append(f"  {way:6s} " + ("missed" if a is None else
                                     f"{pd.Timestamp(a['at']):%H:%M:%S}  {a['text'][:90]}"))
    out.append("\nLIVE: every alert the live rule would have sent (T1 = at once, T2 = with the move)")
    for r in live(lines, mv).itertuples():
        out.append(f"  {pd.Timestamp(r.at):%a %d %b %H:%M}  T{r.tier} [{r.rule}]  {r.text[:100]}"
                   + (f"  ({r.market})" if r.market else ""))
    out.append("\nWORDS: every line either tier flags (T1 / T2), and the biggest move of the three "
               "markets in the 15 min after it (x = times the usual for that minute)")
    for r in lines[lines["tier"].notna()].itertuples():
        out.append(f"  {pd.Timestamp(r.t):%a %d %b %H:%M}  T{int(r.tier)} [{r.rule}]  {r.text[:100]}"
                   f"  -> {biggest_move(bars, r.t, start, end) if bars else '-'}")
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
