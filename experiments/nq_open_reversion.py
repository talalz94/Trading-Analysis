"""
JJ Simon's "fair pricing theory" -- the NEW YORK OPEN REVERSION -- on Nasdaq 100 one-minute data.

Source: a 2026 podcast; the full summary is in docs/SESSION_NOTES.md. He trades NQ futures and says
the edge per trade is "slight"; most of his money comes from running it across 40+ prop accounts.
This asks only the research question: is there a bias back to the 9:30 open, and do his entries
capture it after costs?

The rule, as he states it (all on 1-minute candles, Eastern time):
  fair price   the 9:30 opening price
  window       entries 9:30-11:00 only -- the first 90 minutes
  side         back TOWARD fair price: short while above it, long while below it
  entry        on a candle CLOSE, either
                 displacement -- body bigger than the previous candle's, closing beyond its wick,
                                 and the previous candle is the opposite colour
                 break of structure -- a close beyond the latest swing, i.e. a candle whose wick
                                 reaches beyond both its neighbours
  room         at least 80% of the target still between the entry and fair price
  stop/target  static 25 / 38 points, his evaluation setting. Those are 2026 points (the video's
               trades are July-August 2026, Nasdaq ~30,000), so they are SCALED to each day's price:
               in 2022, at ~12,000, the same stop is ~10 points. Fixed points would have been a stop
               2.5x wider relative to the market in 2022 than the one he uses.
  one position at a time; three losses in a row ends the session; flat by 15:59
  costs        NQ futures: one tick (0.25) of slippage each side + $4.50 commission round trip
               = 0.725 points per trade

Arms, fixed before any result was seen:
  his_rules     displacement OR break of structure, everything above
  bos_only      break of structure only (he calls it the stronger entry)
  disp_only     displacement only (the weaker, more frequent one)
  target_fair   his_rules, but the target is fair price itself (his funded-account style)
  no_3loss      his_rules without the three-losses stop
Nulls -- the rules that bind research here:
  metro_toward  the same side, room, stop, target and 3-loss rule, entering EVERY 5 MINUTES instead
                of on his signals. If his entries beat this, they add something.
  metro_away    the same metronome trading AWAY from fair price. If toward does not beat away, there
                is no reversion bias to capture in the first place.
Train 2022Q1-2024Q2 picks nothing (nothing is tuned) but must agree with forward 2024Q3-2026Q3.

Run:  python -m experiments.nq_open_reversion        (downloads ~15 min of data the first time)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nas100_1m"                      # one parquet per month (git-ignored)
OUT = Path(__file__).resolve().parent / "results_nq_open_reversion.csv"
OUT_Q = Path(__file__).resolve().parent / "results_nq_open_reversion_quarters.csv"

START, END = "2021-12", "2026-09"                       # months downloaded (inclusive)
TRAIN, FORWARD = ("2022Q1", "2024Q2"), ("2024Q3", "2026Q3")
OPEN_MIN, WINDOW_END, DAY_START, DAY_END = 9 * 60 + 30, 11 * 60, 9 * 60, 16 * 60
STOP_PTS, TARGET_PTS, ROOM = 25.0, 38.0, 0.8            # his evaluation numbers, 2026 points
REF_MONTHS = ("2026-07", "2026-08")                     # when the video's trades were taken
COST_PTS = 2 * 0.25 + 4.50 / 20.0                       # 1 tick each side + commission, NQ $20/pt
METRO_EVERY = 5


# ── data ─────────────────────────────────────────────────────────────────────────────────────
def _fetch_month(month: str) -> pd.DataFrame:
    import dukascopy_python as d
    a = pd.Timestamp(month + "-01", tz="UTC")
    b = a + pd.offsets.MonthBegin(1)
    raw = d.fetch("E_NQ-100", d.INTERVAL_MIN_1, d.OFFER_SIDE_BID,
                  a.to_pydatetime(), b.to_pydatetime(), max_retries=3)
    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"timestamp": "open_time"})
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df[(df["open_time"] >= a) & (df["open_time"] < b)][
        ["open_time", "open", "high", "low", "close"]]


def download(start: str = START, end: str = END, timeout: float = 300) -> None:
    """One parquet per month. A month already on disk is skipped, so a stalled run resumes; each
    fetch runs under a timeout (a Dukascopy request once hung for 20 minutes with no error)."""
    import logging
    logging.getLogger("DUKASCRIPT").addFilter(lambda r: r.levelno >= logging.WARNING)
    DATA.mkdir(parents=True, exist_ok=True)
    months = [str(p) for p in pd.period_range(start, end, freq="M")]
    pool = ThreadPoolExecutor(max_workers=1)
    for m in months:
        path = DATA / f"{m}.parquet"
        if path.exists():
            continue
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                df = pool.submit(_fetch_month, m).result(timeout=timeout)
                df.to_parquet(path, index=False)
                print(f"  {m}: {len(df):,} bars in {time.perf_counter() - t0:.0f}s", flush=True)
                break
            except Exception as e:
                print(f"  {m}: attempt {attempt + 1} failed -- {type(e).__name__} {e}", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                pool = ThreadPoolExecutor(max_workers=1)
        else:
            raise RuntimeError(f"{m} could not be downloaded")


def load() -> pd.DataFrame:
    files = sorted(DATA.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data in {DATA}: run download() first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


# ── the New York session ─────────────────────────────────────────────────────────────────────
def sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Bars from 9:00 to 15:59 Eastern, with each day's fair price and its price scale. Days with
    no 9:30 bar (holidays, data gaps) are dropped. Daylight saving is handled by the time zone."""
    et = df["open_time"].dt.tz_convert("America/New_York")
    m = et.dt.hour * 60 + et.dt.minute
    out = df.assign(et=et, minute=m, day=et.dt.date)
    out = out[(m >= DAY_START) & (m < DAY_END)].reset_index(drop=True)
    fair = out.loc[out["minute"] == OPEN_MIN].set_index("day")["open"]
    out = out[out["day"].isin(fair.index)].reset_index(drop=True)
    out["fair"] = out["day"].map(fair)
    ref = out.loc[out["et"].dt.strftime("%Y-%m").isin(REF_MONTHS), "fair"].median()
    out["scale"] = out["fair"] / ref                     # 1.0 at the video's price level
    out["window"] = (out["minute"] >= OPEN_MIN) & (out["minute"] < WINDOW_END)
    out["last"] = out["day"].ne(out["day"].shift(-1))    # the day's final bar: be flat
    out["t"] = out["open_time"]
    return out


def entry_signals(s: pd.DataFrame) -> dict:
    """His two entries, both judged on the candle's close, never across a day boundary."""
    o, h, l, c = (s[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    same = s["day"].eq(s["day"].shift(1)).to_numpy()     # bar i-1 is the same day
    po, ph, pl, pc = (np.r_[np.nan, x[:-1]] for x in (o, h, l, c))
    body, pbody = np.abs(c - o), np.abs(pc - po)
    with np.errstate(invalid="ignore"):
        disp_s = same & (pc > po) & (c < o) & (body > pbody) & (c < pl)
        disp_l = same & (pc < po) & (c > o) & (body > pbody) & (c > ph)

    # a swing low: a wick below both neighbours, KNOWN once the next candle has closed
    def swing_level(x, low: bool):
        prev = pd.Series(x).groupby(s["day"].to_numpy()).shift(1).to_numpy()
        nxt = pd.Series(x).groupby(s["day"].to_numpy()).shift(-1).to_numpy()
        with np.errstate(invalid="ignore"):
            flag = (x < prev) & (x < nxt) if low else (x > prev) & (x > nxt)
        lvl = pd.Series(np.where(flag, x, np.nan))
        # available from the confirming candle (j+1) on, within the day only
        return lvl.groupby(s["day"].to_numpy()).shift(1).groupby(s["day"].to_numpy()).ffill().to_numpy()

    sl, sh = swing_level(l, True), swing_level(h, False)
    with np.errstate(invalid="ignore"):
        bos_s = same & (c < sl) & (pc >= sl)              # the first close below the latest low
        bos_l = same & (c > sh) & (pc <= sh)
    return dict(disp_s=disp_s, disp_l=disp_l, bos_s=bos_s, bos_l=bos_l)


def gates(s: pd.DataFrame, toward: bool = True):
    """(short allowed, long allowed): in the window, on the chosen side of fair price, with at
    least ROOM x the target still between the close and fair price."""
    c, f = s["close"].to_numpy(float), s["fair"].to_numpy(float)
    room = ROOM * TARGET_PTS * s["scale"].to_numpy(float)
    w = s["window"].to_numpy()
    above, below = c - f >= room, f - c >= room
    return (w & above, w & below) if toward else (w & below, w & above)


ARMS = ("his_rules", "bos_only", "disp_only", "target_fair", "no_3loss",
        "metro_toward", "metro_away")


def arm_signals(s: pd.DataFrame, arm: str, sig: dict):
    gs, gl = gates(s, toward=(arm != "metro_away"))
    if arm.startswith("metro"):
        tick = ((s["minute"] - OPEN_MIN) % METRO_EVERY == 0).to_numpy()
        return gs & tick, gl & tick
    if arm == "bos_only":
        return gs & sig["bos_s"], gl & sig["bos_l"]
    if arm == "disp_only":
        return gs & sig["disp_s"], gl & sig["disp_l"]
    return gs & (sig["disp_s"] | sig["bos_s"]), gl & (sig["disp_l"] | sig["bos_l"])


def run_arm(s: pd.DataFrame, arm: str, sig: dict) -> pd.DataFrame:
    """The engine, then the three-losses rule (see `simulate`)."""
    es, el = arm_signals(s, arm, sig)
    return simulate(s, es, el, arm=arm, target_fair=(arm == "target_fair"),
                    three_loss=(arm != "no_3loss"))


def simulate(s: pd.DataFrame, es, el, *, arm: str = "", target_fair: bool = False,
             three_loss: bool = True) -> pd.DataFrame:
    """Trade the entries `es` (short) / `el` (long) with his stop, target, costs and one position
    at a time, flat at each session's last bar; then the three-losses rule. With one position at a
    time, dropping a session's trades after its third loss in a row is exactly the same as having
    stopped there. Shared with experiments/nq_news_reversion.py."""
    from quant.engine import BacktestConfig, run_backtest
    from quant.engine.config import Signals
    d = s.copy()
    stop = STOP_PTS * d["scale"]
    d["stop_l"], d["stop_s"] = d["close"] - stop, d["close"] + stop
    if target_fair:
        d["tgt_l"] = d["tgt_s"] = d["fair"]
    else:
        d["tgt_l"], d["tgt_s"] = (d["close"] + TARGET_PTS * d["scale"],
                                  d["close"] - TARGET_PTS * d["scale"])
    flat = d["last"].to_numpy()
    cfg = BacktestConfig(initial_cash=1e9, sizing_mode="cash", sizing_value=1e6, allow_short=True,
                         max_open_trades=1, exit_enabled=True, sl_mode="ref_col",
                         sl_ref_long_col="stop_l", sl_ref_short_col="stop_s",
                         tp_ref_long_col="tgt_l", tp_ref_short_col="tgt_s")
    res = run_backtest(d, Signals(entry_long=el, exit_long=flat, entry_short=es, exit_short=flat),
                       cfg)
    tr = res.trades.copy()
    if tr.empty:
        return tr
    sign = np.where(tr["side"] == "long", 1.0, -1.0)
    risk = STOP_PTS * d["scale"].to_numpy()[tr["entry_i"].to_numpy(int)]
    tr["pts"] = (tr["exit_price"] - tr["entry_price"]) * sign
    tr["R_gross"] = tr["pts"] / risk
    tr["R"] = (tr["pts"] - COST_PTS) / risk
    tr["day"] = d["day"].to_numpy()[tr["entry_i"].to_numpy(int)]
    tr["quarter"] = pd.PeriodIndex(pd.to_datetime(tr["day"]), freq="Q").astype(str)
    tr["arm"] = arm
    if three_loss:
        keep, run_, last_day = [], 0, None
        for day, r in zip(tr["day"], tr["R"]):
            if day != last_day:
                run_, last_day = 0, day
            ok = run_ < 3
            keep.append(ok)
            if ok:
                run_ = run_ + 1 if r < 0 else 0
        tr = tr[np.array(keep)].reset_index(drop=True)
    return tr


# ── judging ──────────────────────────────────────────────────────────────────────────────────
def _t(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def _half(tr, span):
    q = tr["quarter"]
    return tr[(q >= span[0]) & (q <= span[1])]


def summarize(trades: dict) -> pd.DataFrame:
    rows = []
    base = _half(trades["metro_toward"], (TRAIN[0], FORWARD[1]))
    for arm, tr in trades.items():
        tr = _half(tr, (TRAIN[0], FORWARD[1]))            # Dec 2021 is warm-up only
        tr_tr, tr_fw = _half(tr, TRAIN), _half(tr, FORWARD)
        bq = tr.groupby("quarter")["R"].mean()
        top = np.sort(tr["R"].to_numpy())[::-1]
        rows.append(dict(
            arm=arm, trades=len(tr), days=tr["day"].nunique(),
            win_pct=round(100 * (tr["R"] > 0).mean(), 1),
            R_gross=round(tr["R_gross"].mean(), 4), R_net=round(tr["R"].mean(), 4),
            t=round(_t(tr["R"]), 2),
            train_R=round(tr_tr["R"].mean(), 4), fwd_R=round(tr_fw["R"].mean(), 4),
            fwd_t=round(_t(tr_fw["R"]), 2),
            quarters_pos=f"{int((bq > 0).sum())}/{len(bq)}",
            trim3_R=round(top[3:].mean(), 4) if len(top) > 3 else np.nan,
            vs_metro_train=round(tr_tr["R"].mean() - _half(base, TRAIN)["R"].mean(), 4),
            vs_metro_fwd=round(tr_fw["R"].mean() - _half(base, FORWARD)["R"].mean(), 4),
            total_R=round(tr["R"].sum(), 1)))
    return pd.DataFrame(rows).set_index("arm")


def run(download_first: bool = True) -> pd.DataFrame:
    if download_first:
        download()
    s = sessions(load())
    sig = entry_signals(s)
    trades = {arm: run_arm(s, arm, sig) for arm in ARMS}
    table = summarize(trades)
    table.to_csv(OUT)
    q = pd.concat([t.groupby("quarter")["R"].agg(["mean", "size"]).add_prefix(a + "_")
                   for a, t in trades.items()], axis=1)
    q.to_csv(OUT_Q)
    return table


def verdict(table: pd.DataFrame) -> str:
    t, a = table.loc["metro_toward"], table.loc["metro_away"]
    h = table.loc["his_rules"]
    lines = [
        f"Reversion bias (metronome toward vs away, R/trade net): train {t.train_R:+.3f} vs "
        f"{a.train_R:+.3f}, forward {t.fwd_R:+.3f} vs {a.fwd_R:+.3f}",
        f"His entries vs the metronome toward fair price: train {h.vs_metro_train:+.3f}, "
        f"forward {h.vs_metro_fwd:+.3f} R/trade",
        f"His rules after costs: {h.R_net:+.3f} R/trade over {h.trades} trades (t {h.t}), "
        f"train {h.train_R:+.3f}, forward {h.fwd_R:+.3f}, quarters positive {h.quarters_pos}",
        f"Break-even win rate for 25/38 after costs: "
        f"{(STOP_PTS + COST_PTS) / (STOP_PTS + TARGET_PTS):.1%} at the video's price level",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    tbl = run(download_first="--no-download" not in sys.argv)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tbl)
    print()
    print(verdict(tbl))
