"""
JJ Simon's A+ setup -- the EXPECTED NEWS REVERSION -- on Nasdaq 100 one-minute data.

His words (docs/SESSION_NOTES.md): on 8:30 a.m. "red folder" days (CPI, PPI ...) the news is
"priced in" -- there is a forecast and the number is usually close to it -- so the huge candle
coming off the release is "unfair" and price reverts to where it stood BEFORE the news. He calls
it the strongest trade he takes and risks the most on it. Outsiders -- a number far from its
forecast -- just keep going, and he says they are obvious on the chart.

The rule, with everything not about the news taken unchanged from experiments/nq_open_reversion:
  news day     a US event of high or critical impact scheduled at 8:30 New York (quant.news)
  fair price   the CLOSE of the 8:29 candle -- the last price before the release
  window       entries on candles from 8:31 to 10:59 (the 8:30 candle IS the news); his example
               trades start right after the release and run to 11:00
  side         back TOWARD fair price, with at least 80% of the target between entry and fair
  entry        his displacement or break of structure, on the candle close (the same code)
  stop/target  25 / 38 points scaled to price, one position, 3 losses end the day, flat by 15:59
  costs        NQ futures, 0.725 points a trade

Arms, fixed before any result was seen:
  news_all       every 8:30 red-folder day
  news_aplus     only CPI, PPI and payrolls days (the ones he names)
  news_from_open every 8:30 red day, but entering only 9:30-11:00 (his session window)
  priced_in      news_all, only days whose 8:30 surprises were all small (|z| < 1.5)
  outsiders      news_all, only days with a large surprise (|z| >= 1.5) -- he expects these to fail
Nulls:
  metro_toward   news days, the same side/room/stop/target/3-loss, entering every 5 minutes
  metro_away     the same metronome trading AWAY from the pre-news price
  quiet_days     his rules on days with NO high-impact US news at all, the 8:29 close as "fair"
                 -- if news days do not beat ordinary days, the news is not what matters
z is each release's surprise over the size of its OWN past surprises (quant.news.surprise_z), so a
day is called an outsider only from information older than it.

TWO SCOPES, because the calendar is not equally complete everywhere:
  complete   every red-folder release (claims, durable goods, Empire ...) is known only while the
             archive runs, 2021-11 .. 2025-04. The arms above, as registered.
  official   2021-11 .. 2026-09 on the releases the OFFICIAL sources cover through the whole span
             -- FRED's CPI, PPI, payrolls, retail sales, PCE and GDP days plus the Fed's own
             decisions -- so both halves use ONE definition. A "quiet" day here has none of those,
             though it may hold a lesser red release (claims) the later period cannot see.
             Surprises exist only while the archive runs, so the outsider arms stop there.
Train 2022Q1-2024Q2 / forward 2024Q3-2026Q3, as in the open test.

Run:  python -m experiments.nq_news_reversion      (needs data/nas100_1m from the open test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import nq_open_reversion as nq

OUT = nq.Path(__file__).resolve().parent / "results_nq_news_reversion.csv"
OUT_Q = nq.Path(__file__).resolve().parent / "results_nq_news_reversion_quarters.csv"
OUT_FULL = nq.Path(__file__).resolve().parent / "results_nq_news_reversion_official.csv"
OUT_FULL_Q = nq.Path(__file__).resolve().parent / "results_nq_news_reversion_official_quarters.csv"

NEWS_MIN = 8 * 60 + 30
FAIR_MIN = NEWS_MIN - 1                       # the 8:29 candle: its close is the pre-news price
WIN_START, WIN_END = NEWS_MIN + 1, 11 * 60    # 8:31 .. 10:59 candles
DAY_START, DAY_END = 8 * 60, 16 * 60
APLUS = {"CPI m/m", "Core CPI m/m", "CPI y/y", "Core CPI y/y", "PPI m/m", "Core PPI m/m",
         "Non-Farm Employment Change"}
OUTSIDER_Z = 1.5


def _official_titles() -> set:
    """Every title the official sources carry through the whole span (see the docstring)."""
    from quant.news.sources import FRED_RELEASES
    out = {"Federal Funds Rate", "FOMC Statement", "FOMC Press Conference",
           "FOMC Economic Projections"}
    for _, titles in FRED_RELEASES.values():
        for t in titles:
            out |= ({f"{s} GDP q/q" for s in ("Advance", "Prelim", "Final")}
                    if t == "GDP q/q" else {t})
    return out
ARMS = ("news_all", "news_aplus", "news_from_open", "priced_in", "outsiders",
        "metro_toward", "metro_away", "quiet_days")


# ── which days ───────────────────────────────────────────────────────────────────────────────
def news_days(events: pd.DataFrame, titles=None) -> pd.DataFrame:
    """One row per New York date with US high/critical news: whether any was at 8:30 (`red830`),
    whether CPI/PPI/payrolls were (`aplus`), and the largest |surprise z| among the 8:30 ones
    (`max_z`, NaN when none has a z). `titles` keeps only those releases (the official scope)."""
    ev = events[(events["currency"] == "USD") & events["level"].isin(["high", "critical"])
                & events["time_q"].isin(["exact", "official", "typical"])].copy()
    if titles is not None:
        ev = ev[ev["title"].isin(titles)]
    et = pd.to_datetime(ev["t"], utc=True).dt.tz_convert("America/New_York")
    ev["day"] = et.dt.date
    ev["at830"] = (et.dt.hour * 60 + et.dt.minute) == NEWS_MIN
    g = ev.groupby("day")
    out = pd.DataFrame({"any_red": g.size() > 0,
                        "red830": g["at830"].any(),
                        "aplus": ev[ev["at830"] & ev["title"].isin(APLUS)].groupby("day").size()
                        .reindex(g.size().index).fillna(0) > 0,
                        "max_z": ev[ev["at830"]].assign(az=ev["surprise_z"].abs())
                        .groupby("day")["az"].max().reindex(g.size().index),
                        "titles": ev[ev["at830"]].groupby("day")["title"]
                        .agg(lambda s: ", ".join(sorted(set(s)))).reindex(g.size().index)})
    return out


def sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Bars 8:00-15:59 New York, each day's pre-news price (the 8:29 close) and price scale. A day
    with no 8:29 candle is dropped. The scale is the open test's: price over the median 9:30 open
    of the video's months, so 25 points mean the same thing in 2022 and 2026."""
    et = df["open_time"].dt.tz_convert("America/New_York")
    m = et.dt.hour * 60 + et.dt.minute
    ref_rows = (m == nq.OPEN_MIN) & et.dt.strftime("%Y-%m").isin(nq.REF_MONTHS)
    ref = df.loc[ref_rows, "open"].median()
    out = df.assign(et=et, minute=m, day=et.dt.date)
    out = out[(m >= DAY_START) & (m < DAY_END)].reset_index(drop=True)
    fair = out.loc[out["minute"] == FAIR_MIN].set_index("day")["close"]
    out = out[out["day"].isin(fair.index)].reset_index(drop=True)
    out["fair"] = out["day"].map(fair)
    out["scale"] = out["fair"] / ref
    out["window"] = (out["minute"] >= WIN_START) & (out["minute"] < WIN_END)
    out["window_open"] = (out["minute"] >= nq.OPEN_MIN) & (out["minute"] < WIN_END)
    out["last"] = out["day"].ne(out["day"].shift(-1))
    out["t"] = out["open_time"]
    return out


def gates(s: pd.DataFrame, toward: bool = True, window: str = "window"):
    c, f = s["close"].to_numpy(float), s["fair"].to_numpy(float)
    room = nq.ROOM * nq.TARGET_PTS * s["scale"].to_numpy(float)
    w = s[window].to_numpy()
    above, below = c - f >= room, f - c >= room
    return (w & above, w & below) if toward else (w & below, w & above)


def day_mask(s: pd.DataFrame, days: pd.DataFrame, arm: str) -> np.ndarray:
    """Which bars belong to the arm's days."""
    d = s["day"]
    red = set(days.index[days["red830"]])
    if arm == "quiet_days":
        return (~d.isin(set(days.index[days["any_red"]]))).to_numpy()
    if arm == "news_aplus":
        return d.isin(set(days.index[days["aplus"]])).to_numpy()
    if arm == "priced_in":
        return d.isin(set(days.index[days["red830"] & (days["max_z"] < OUTSIDER_Z)])).to_numpy()
    if arm == "outsiders":
        return d.isin(set(days.index[days["red830"] & (days["max_z"] >= OUTSIDER_Z)])).to_numpy()
    return d.isin(red).to_numpy()


def arm_entries(s: pd.DataFrame, days: pd.DataFrame, arm: str, sig: dict):
    on = day_mask(s, days, arm)
    gs, gl = gates(s, toward=(arm != "metro_away"),
                   window="window_open" if arm == "news_from_open" else "window")
    gs, gl = gs & on, gl & on
    if arm.startswith("metro"):
        tick = ((s["minute"] - WIN_START) % nq.METRO_EVERY == 0).to_numpy()
        return gs & tick, gl & tick
    return gs & (sig["disp_s"] | sig["bos_s"]), gl & (sig["disp_l"] | sig["bos_l"])


# ── running and judging ──────────────────────────────────────────────────────────────────────
def coverage(events: pd.DataFrame, titles=None, complete: bool = True) -> tuple:
    """First and last day the calendar can speak for. `complete`: only while the ARCHIVE runs,
    since every red-folder release is known only there; otherwise as far as `titles` reach."""
    d = news_days(events, titles)
    d = d[d["red830"]]
    if not len(d):
        return None, None
    last = max(d.index)
    if complete and "source" in events.columns:
        arch = events[events["source"].astype(str).str.contains("ff_archive")]
        if len(arch):
            last = min(last, pd.to_datetime(arch["t"], utc=True).max()
                       .tz_convert("America/New_York").date())
    return min(d.index), last


def run(events: pd.DataFrame = None, scope: str = "complete") -> pd.DataFrame:
    """`scope` "complete" (every red release, to the archive's end) or "official" (the releases
    the official sources cover, the whole span) -- see the module docstring."""
    from quant.news import load_events
    if scope not in ("complete", "official"):
        raise ValueError(f"scope must be complete | official, got {scope!r}")
    if events is None:
        events = load_events("2021-11-01", "2026-10-01")
    titles = _official_titles() if scope == "official" else None
    days = news_days(events, titles)
    first, last = coverage(events, titles, complete=(scope == "complete"))
    s = sessions(nq.load())
    # only days the calendar can speak for: a "quiet" day after its last date is merely unknown
    s = s[(s["day"] >= first) & (s["day"] <= last)].reset_index(drop=True)
    last = min(last, max(s["day"]))
    sig = nq.entry_signals(s)
    trades = {arm: nq.simulate(s, *arm_entries(s, days, arm, sig), arm=arm) for arm in ARMS}
    table = nq.summarize(trades)
    table.insert(0, "calendar", f"{scope}: {first} .. {last}")
    out, out_q = (OUT, OUT_Q) if scope == "complete" else (OUT_FULL, OUT_FULL_Q)
    table.to_csv(out)
    q = pd.concat([t.groupby("quarter")["R"].agg(["mean", "size"]).add_prefix(a + "_")
                   for a, t in trades.items() if len(t)], axis=1)
    q.to_csv(out_q)
    run.trades = trades
    return table


def verdict(table: pd.DataFrame) -> str:
    g = table.loc
    lines = [
        f"Calendar covers {table['calendar'].iloc[0]}",
        f"Reversion bias on news days (metronome toward vs away, R/trade): train "
        f"{g['metro_toward'].train_R:+.3f} vs {g['metro_away'].train_R:+.3f}, forward "
        f"{g['metro_toward'].fwd_R:+.3f} vs {g['metro_away'].fwd_R:+.3f}",
        f"His rules on news days: {g['news_all'].R_net:+.3f} R/trade over {g['news_all'].trades} "
        f"trades (t {g['news_all'].t}); vs metronome train {g['news_all'].vs_metro_train:+.3f}, "
        f"forward {g['news_all'].vs_metro_fwd:+.3f}",
        f"News days vs quiet days (his rules): {g['news_all'].R_net:+.3f} vs "
        f"{g['quiet_days'].R_net:+.3f} R/trade",
        f"Priced in vs outsiders: {g['priced_in'].R_net:+.3f} ({g['priced_in'].trades}) vs "
        f"{g['outsiders'].R_net:+.3f} ({g['outsiders'].trades})",
        f"CPI/PPI/payrolls only: {g['news_aplus'].R_net:+.3f} over {g['news_aplus'].trades}; "
        f"from the 9:30 open only: {g['news_from_open'].R_net:+.3f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for sc in ("complete", "official"):
        tbl = run(scope=sc)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(tbl.drop(columns="calendar"))
        print()
        print(verdict(tbl))
        print()
