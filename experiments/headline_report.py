"""
HEADLINE REPORT -- what the headline probe (experiments/headline_probe.py) recorded, judged per
source on the four questions that decide the headlines layer:

  SPEED        on every story two or more sources carried, how far each was behind the first
  COVERAGE     every sharp Gold / USDJPY / BTC move: which sources had a matching headline, and
               how long before or after the move started -- against CHANCE: the same matching on
               random moments (the move's own day, and 1-3 h either side), because a busy source
               "covers" moves by luck. Only a source beating both is credited
  NOISE        headlines a day, and how many the draft keyword rules would flag
  RELIABILITY  failed downloads, the longest silence, the printed time vs when we first saw it

Two clocks per headline, because they answer different questions:
  printed  the source's own time (Telegram: when Telegram received the message) -- how fast the
           SOURCE is
  seen     when the probe first had it -- how fast it would reach US, polling included. A printed
           time far before we saw it (a Google News article dated hours earlier) is not trusted.

    python experiments/headline_report.py               summary on screen, all of it in report.txt
    python experiments/headline_report.py --no-prices   skip the sharp moves (no candle download)
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE / "results" / "headline_probe" / "probe.sqlite"
OUT = HERE / "results" / "headline_probe" / "report.txt"

STORY_GAP = pd.Timedelta(minutes=45)      # a headline joins a story seen this recently
MIN_SHARED, MIN_OVERLAP = 3, 0.5          # words in common, and as a share of the shorter title
TRUST_PRINTED = (pd.Timedelta(seconds=-120), pd.Timedelta(minutes=30))

STOP = set("""the a an and or of to in on at for by with from as is are was were be been it its
this that these those after before over under into amid says said say will would could should may
can not no new more than about than us u.s up down out off has have had who what when where why how
financialjuice breaking report reports sources source official officials""".split())

# the draft rules -- the same ones run over 21 Oct's FinancialJuice feed (26 of 100 flagged)
RULES = {
    "P1 security": r"ordered departure|evacuat|embass(?:y|ies)|missile|airstrike|strikes? on|"
                   r"attack(?:s|ed)? |invasion|nuclear|houthi|hormuz|ceasefire|\bwar\b",
    "P1 rates": r"\bfed\b.*(?:cut|hike|emergency)|fomc|powell|warsh|rate decision",
    "P2 japan": r"\bboj\b|ueda|intervention|japan(?:'s)? (?:mof|finance)",
    "P2 trade": r"tariff|sanction",
    "P2 crypto": r"bitcoin|crypto|\bsec\b|\betf\b",
}
RELEASES = r"|cpi|ppi|retail sales|payroll|jobs|claims|gdp|pmi|inflation"   # scheduled news
MARKET_WORDS = {
    "Gold": r"gold|war|attack|missile|strike|embass|evacuat|departure|iran|israel|houthi|gulf|"
            r"\bfed\b|fomc|powell|warsh|tariff|sanction|dollar|yield|treasur|trump|ceasefire|oil"
            + RELEASES,
    "USDJPY": r"\bboj\b|ueda|japan|yen|intervention|\bmof\b|\bfed\b|fomc|powell|warsh|"
              r"yield|treasur|dollar|tariff|trump" + RELEASES,
    "BTC": r"bitcoin|btc|crypto|\bsec\b|\betf\b|stablecoin|\bfed\b|fomc|powell|tariff|trump|war"
           + RELEASES,
}
MOVE_FLOOR = {"Gold": 0.0025, "USDJPY": 0.0012, "BTC": 0.005}   # a 5-minute move at least this
MOVE_K = 6.0                               # ... and K times the day's typical 5-minute move
MOVE_WINDOW = (pd.Timedelta(minutes=-15), pd.Timedelta(minutes=20))
# THE CHANCE BASELINE: a busy source "covers" moves by luck (Google News, 300-600 headlines a day,
# matched BTC's biggest drop to a headline about Ukraine), so coverage counts only above what the
# same matching gives RANDOM moments: on the move's own day (the project's within-day shuffle),
# and 1-3 hours either side of it (keeps the hour: moves and headlines both crowd the US session)
CHANCE_DRAWS = 500
NEAR = (pd.Timedelta(hours=1), pd.Timedelta(hours=3))
CREDIT_P = 0.05                            # credited only below this against BOTH baselines


def load(db_path=DB):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    items = pd.read_sql("SELECT * FROM items", db)
    polls = pd.read_sql("SELECT * FROM polls", db)
    start = db.execute("SELECT value FROM meta WHERE key = 'start'").fetchone()
    for c in ("published", "first_seen"):
        items[c] = pd.to_datetime(items[c], utc=True, format="ISO8601")
    polls["at"] = pd.to_datetime(polls["at"], utc=True, format="ISO8601")
    return items, polls, (pd.Timestamp(start[0]) if start else polls["at"].min())


def timed(items: pd.DataFrame) -> pd.DataFrame:
    """The headlines that can be timed (not the backlog of a first poll), with `seen` and
    `printed` (NaT where the source's own time is missing or not believable)."""
    d = items[items["backlog"] == 0].copy()
    d["seen"] = d["first_seen"]
    gap = d["first_seen"] - d["published"]
    ok = d["published"].notna() & (gap >= TRUST_PRINTED[0]) & (gap <= TRUST_PRINTED[1])
    d["printed"] = d["published"].where(ok)
    d["t"] = d["printed"].fillna(d["seen"])
    return d.sort_values("t").reset_index(drop=True)


def words(title: str, source: str = "") -> frozenset:
    t = title
    if source.startswith("gnews") and " - " in t:
        t = t.rsplit(" - ", 1)[0]                        # Google News appends " - Publisher"
    t = re.sub(r"^(financialjuice|breaking|\*+)\s*:?\s*", "", t.strip(), flags=re.I)
    toks = re.findall(r"[a-z0-9$%.]+", t.lower())
    return frozenset(w.strip(".") for w in toks if len(w.strip(".")) >= 3 and w not in STOP)


def stories(d: pd.DataFrame) -> pd.Series:
    """Group headlines about the same thing: shares >= 3 words AND >= half the shorter title's
    words with a story active in the last 45 minutes. -> story number per row of `d`."""
    index: Dict[str, set] = {}                 # word -> stories using it
    toks: List[frozenset] = []                 # a story's words (its first five headlines)
    last: List[pd.Timestamp] = []
    size: List[int] = []
    out = []
    for t, title, src in zip(d["t"], d["title"], d["source"]):
        w = words(title, src)
        best, score = None, 0.0
        if len(w) >= MIN_SHARED:
            cand: Dict[int, int] = {}
            for x in w:
                for s in index.get(x, ()):
                    if t - last[s] <= STORY_GAP:
                        cand[s] = cand.get(s, 0) + 1
            for s, shared in cand.items():
                ov = shared / min(len(w), len(toks[s]))
                if shared >= MIN_SHARED and ov >= MIN_OVERLAP and ov > score:
                    best, score = s, ov
        if best is None:
            best = len(toks)
            toks.append(w)
            last.append(t)
            size.append(0)
            for x in w:
                index.setdefault(x, set()).add(best)
        else:
            last[best] = max(last[best], t)
            if size[best] < 5:
                new = w - toks[best]
                toks[best] = toks[best] | w
                for x in new:
                    index.setdefault(x, set()).add(best)
        size[best] += 1
        out.append(best)
    return pd.Series(out, index=d.index)


def speed_table(d: pd.DataFrame) -> pd.DataFrame:
    """Per source, over the stories two or more sources carried: how many it carried, how often
    it was first, and its median / 90th-percentile lag behind the first (printed and seen)."""
    d = d.assign(story=stories(d))
    first = d.groupby(["story", "source"]).agg(t=("t", "min"), seen=("seen", "min")).reset_index()
    multi = first.groupby("story")["source"].transform("nunique") >= 2
    f = first[multi].copy()
    f["lag"] = (f["t"] - f.groupby("story")["t"].transform("min")).dt.total_seconds()
    f["lag_seen"] = (f["seen"] - f.groupby("story")["seen"].transform("min")).dt.total_seconds()
    g = f.groupby("source")
    return pd.DataFrame({
        "stories": g.size(),
        "first_%": g["lag"].apply(lambda x: 100 * (x == 0).mean()).round(0),
        "lag_med_s": g["lag"].median().round(0),
        "lag_p90_s": g["lag"].quantile(0.9).round(0),
        "seen_med_s": g["lag_seen"].median().round(0),
    }).sort_values("lag_med_s")


def reliability_table(items, polls, start, end) -> pd.DataFrame:
    days = max((end - start).total_seconds() / 86400, 1e-9)
    rows = {}
    for src, p in polls.sort_values("at").groupby("source"):
        ok = p[p["ok"] == 1]["at"]
        gaps = ok.diff().dt.total_seconds() / 60
        it = items[(items["source"] == src) & (items["backlog"] == 0)]
        lag = (it["first_seen"] - it["published"]).dt.total_seconds().dropna()
        rows[src] = dict(polls=len(p), ok_pct=round(100 * p["ok"].mean(), 1),
                         longest_gap_min=round(gaps.max(), 0) if len(gaps) else np.nan,
                         ms_med=int(p["ms"].median()), per_day=round(len(it) / days, 0),
                         printed_to_seen_med_s=round(lag.median(), 0) if len(lag) else np.nan,
                         printed_to_seen_p90_s=round(lag.quantile(0.9), 0) if len(lag) else np.nan)
    return pd.DataFrame(rows).T


def noise_table(d: pd.DataFrame, days: float) -> pd.DataFrame:
    out = {}
    for src, g in d.groupby("source"):
        row = {}
        for name, pat in RULES.items():
            row[name] = round(g["title"].str.contains(pat, case=False, regex=True).sum() / days, 1)
        row["any"] = round(sum(row.values()), 1)
        out[src] = row
    if not out:                                      # nothing timed yet: the first minutes
        return pd.DataFrame(columns=[*RULES, "any"])
    return pd.DataFrame(out).T.sort_values("any", ascending=False)


def find_moves(bars: pd.DataFrame, market: str) -> pd.DataFrame:
    """Sharp moves in 1m candles: a 5-minute return at least MOVE_FLOOR and MOVE_K times the
    median 5-minute move of the day before. One move per 30 minutes (its first minute)."""
    b = bars.sort_values("open_time").reset_index(drop=True)
    c = b["close"].to_numpy(float)
    r5 = np.full(len(c), np.nan)
    r5[:-5] = c[5:] / c[:-5] - 1                         # from this candle's close, 5 later
    typical = pd.Series(np.abs(r5)).shift(1).rolling(1440, min_periods=240).median().to_numpy()
    hit = np.abs(r5) >= np.maximum(MOVE_FLOOR[market], MOVE_K * np.nan_to_num(typical, nan=np.inf))
    out, until = [], None
    for i in np.flatnonzero(hit):
        t = b["open_time"].iat[i] + pd.Timedelta(minutes=1)    # the move starts at this close
        if until is not None and t < until:
            continue
        out.append(dict(market=market, start=t, move_pct=round(100 * r5[i], 3)))
        until = t + pd.Timedelta(minutes=30)
    return pd.DataFrame(out, columns=["market", "start", "move_pct"])


def _ns(times: pd.Series) -> np.ndarray:
    """Timestamps as int64 nanoseconds, whatever resolution pandas parsed them at."""
    return pd.to_datetime(times, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def _covered(times: np.ndarray, at: np.ndarray) -> np.ndarray:
    """For each moment in `at` (ns): is one of the sorted `times` (ns) inside its window?"""
    if not len(times):
        return np.zeros(np.shape(at), bool)
    i = np.searchsorted(times, at + MOVE_WINDOW[0].value, side="left")
    return (i < len(times)) & (times[np.minimum(i, len(times) - 1)] <= at + MOVE_WINDOW[1].value)


def chance(moves: pd.DataFrame, d: pd.DataFrame, sources: List[str], start, end,
           draws: int = CHANCE_DRAWS, seed: int = 0) -> pd.DataFrame:
    """Per source, the coverage RANDOM moments get with exactly the moves' matching rule (the
    window, the market's words, the source), against two baselines:
      day   a random moment on each move's own UTC day (the project's within-day shuffle)
      near  a random moment 1-3 hours before or after the move (keeps the time of day)
    each kept inside the recorded period so its whole window could hold headlines. p_* = the
    share of draws covering at least as many moves as the real moves did (with the +1 that keeps
    a p-value off zero). -> real_n, chance_day_pct, chance_near_pct, p_day, p_near per source."""
    rng = np.random.default_rng(seed)
    t0 = _ns(moves["start"])
    lo = (pd.Timestamp(start) - MOVE_WINDOW[0]).value              # whole window recorded
    hi = (pd.Timestamp(end) - MOVE_WINDOW[1]).value
    day0 = _ns(moves["start"].dt.floor("D"))
    a = np.clip(day0, lo, hi)
    b = np.clip(day0 + pd.Timedelta(days=1).value, lo, hi)
    b = np.where(b > a, b, a + 1)
    day = a + rng.random((draws, len(t0))) * (b - a)
    off = NEAR[0].value + rng.random((draws, len(t0))) * (NEAR[1] - NEAR[0]).value
    sign = np.where(rng.random((draws, len(t0))) < 0.5, -1, 1)
    near = t0 + sign * off
    near = np.where((near < lo) | (near > hi), t0 - sign * off, near)   # the other side instead
    near = np.clip(near, lo, hi)
    rows = {}
    for s in sources:
        real = np.zeros(len(t0), bool)
        dd = np.zeros((draws, len(t0)), bool)
        nn = np.zeros((draws, len(t0)), bool)
        for mk in moves["market"].unique():
            cols = (moves["market"] == mk).to_numpy()
            src = d[(d["source"] == s) & d["title"].str.contains(MARKET_WORDS[mk], case=False,
                                                                 regex=True)]
            times = np.sort(_ns(src["t"]))
            real[cols] = _covered(times, t0[cols])
            dd[:, cols] = _covered(times, day[:, cols])
            nn[:, cols] = _covered(times, near[:, cols])
        k = real.sum()
        rows[s] = dict(real_n=int(k),
                       chance_day_pct=round(100 * dd.mean(), 0),
                       chance_near_pct=round(100 * nn.mean(), 0),
                       p_day=round((1 + (dd.sum(axis=1) >= k).sum()) / (1 + draws), 3),
                       p_near=round((1 + (nn.sum(axis=1) >= k).sum()) / (1 + draws), 3))
    return pd.DataFrame(rows).T


def coverage(moves: pd.DataFrame, d: pd.DataFrame, sources: List[str], start=None, end=None,
             draws: int = CHANCE_DRAWS):
    """-> (per-move table of each source's first matching headline, minutes from the move's
    start; per-source coverage share and median lag). Given the recorded period (`start`,
    `end`), each source's coverage also gets its CHANCE baselines (see chance()) and a verdict:
    CREDIT only where it beats both with p < CREDIT_P."""
    rows = []
    for m in moves.itertuples():
        win = d[(d["t"] >= m.start + MOVE_WINDOW[0]) & (d["t"] <= m.start + MOVE_WINDOW[1])]
        win = win[win["title"].str.contains(MARKET_WORDS[m.market], case=False, regex=True)]
        row = dict(start=m.start, market=m.market, move_pct=m.move_pct)
        for s in sources:
            w = win[win["source"] == s]
            row[s] = (round((w["t"].min() - m.start).total_seconds() / 60, 1)
                      if len(w) else np.nan)
        first = win.sort_values("t").head(1)
        row["first_headline"] = first["title"].iat[0][:90] if len(first) else ""
        rows.append(row)
    per_move = pd.DataFrame(rows)
    if per_move.empty:
        return per_move, pd.DataFrame()
    summ = pd.DataFrame({s: dict(covered_pct=round(100 * per_move[s].notna().mean(), 0),
                                 lag_med_min=round(per_move[s].dropna().median(), 1))
                         for s in sources}).T.sort_values("covered_pct", ascending=False)
    if start is None or end is None:
        return per_move, summ
    c = chance(moves, d, sources, start, end, draws=draws)
    summ = summ.join(c[["chance_day_pct", "chance_near_pct", "p_day", "p_near"]])
    summ["above_chance"] = summ["covered_pct"] - summ[["chance_day_pct", "chance_near_pct"]].max(
        axis=1)
    ok = (summ["p_day"] < CREDIT_P) & (summ["p_near"] < CREDIT_P) & (summ["above_chance"] > 0)
    summ["verdict"] = np.where(ok, "CREDIT", "chance")
    cols = ["covered_pct", "chance_day_pct", "chance_near_pct", "above_chance", "p_day",
            "p_near", "verdict", "lag_med_min"]
    return per_move, summ[cols].sort_values("above_chance", ascending=False)


def fetch_bars(start, end) -> Dict[str, pd.DataFrame]:
    """1m candles of the three alert markets for the test window, from the alerts' own feed."""
    from quant.alerts.config import Market
    from quant.alerts.feeds import Feed
    mk = [Market("Gold", "XAUUSD", "dukascopy", ""), Market("USDJPY", "USDJPY", "dukascopy", ""),
          Market("BTC", "BTCUSDT", "binance", "")]
    got: Dict[str, pd.DataFrame] = {}
    for _ in range(3):                                   # Dukascopy sometimes fails once
        need = [m for m in mk if m.name not in got]
        if not need:
            break
        r = Feed().fetch(need, {m.name: start - pd.Timedelta(days=1) for m in need}, end,
                         base="1m")
        got.update({k: v for k, v in r.items() if isinstance(v, pd.DataFrame) and len(v)})
    return got


def report(db_path=DB, prices: bool = True, out=OUT) -> str:
    items, polls, start = load(db_path)
    end = polls["at"].max()
    days = max((end - start).total_seconds() / 86400, 1e-9)
    d = timed(items)
    sources = sorted(polls["source"].unique())
    parts = [f"HEADLINE PROBE {start:%d %b %H:%M} -> {end:%d %b %H:%M} UTC ({days:.1f} days): "
             f"{len(d)} timed headlines from {len(sources)} sources",
             "\nRELIABILITY", reliability_table(items, polls, start, end).to_string(),
             "\nSPEED: seconds behind the first source, on stories 2+ sources carried",
             speed_table(d).to_string(),
             "\nNOISE: headlines a day the draft keyword rules flag",
             noise_table(d, days).to_string()]
    if prices:
        bars = fetch_bars(start, end)
        # moves are judged against the day BEFORE (fetch_bars reaches back one): find them on all
        # the candles, then keep those inside the test -- cutting the candles first left the
        # first 4 hours with no typical move to compare against, so none could be found there
        found = [(lambda f: f[f["start"] >= start])(find_moves(b, m)) for m, b in bars.items()]
        moves = pd.concat(found, ignore_index=True) if found else find_moves(
            pd.DataFrame(columns=["open_time", "close"]), "Gold")
        per_move, summ = coverage(moves.sort_values("start"), d, sources, start, end)
        parts += [f"\nCOVERAGE: {len(moves)} sharp moves -- minutes from the move's start to a "
                  "matching headline (negative = the headline came first)",
                  f"  chance = the same matching on random moments: on the move's own day, and "
                  f"1-3 h either side ({CHANCE_DRAWS} draws each). p = how often chance covered "
                  f"as many moves.\n  CREDIT = beats both with p < {CREDIT_P}; a busy source "
                  f"'covers' moves by luck, so covered_pct alone means nothing",
                  summ.to_string() if len(summ) else "no sharp moves",
                  "\nTHE BIGGEST MOVES",
                  (per_move.reindex(per_move["move_pct"].abs().sort_values(ascending=False).index)
                   .head(15).to_string(index=False) if len(per_move) else "")]
    text = "\n".join(parts)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--no-prices", action="store_true", help="skip the sharp-move coverage")
    a = ap.parse_args(argv)
    print(report(a.db, prices=not a.no_prices))
    print(f"\n(also written to {OUT})")


if __name__ == "__main__":
    main()
