"""
TRENDLINE SETTINGS SEARCH — what works, per timeframe, per instrument, and across timeframes.

Four stages, each answering a different question, and deliberately separated because they have
different failure modes:

  1. QUALITY (outcome-blind).  Which settings produce well-formed, frequent lines? Judged on line
     count, touches, and how often a line is actually broken rather than fading away. Never looks at
     forward returns. This is the only stage whose answer is stable — see the warning below.

  2. SCREEN (outcome-based).   After a break, does price keep going? Measured three ways at once,
     because each controls a different artifact:
        edge_fix      forward move in units of the window's MEDIAN ATR — a denominator that cannot
                      move with the signal (an ATR-at-the-bar denominator inflated a result 5x
                      earlier in this project)
        edge_matched  against bars in the same trailing-move x volatility bucket. A break bar is a
                      big-move bar BY CONSTRUCTION, so without this the measurement is momentum
                      wearing a trendline costume
        vs NULL       the same detector on within-day-shuffled bars, where nothing is findable
     plus the hit rate, which is the plainest reading of "does it point the right way".

  3. MULTI-TIMEFRAME.          Does a break confirmed on two timeframes beat a break on one? The
     higher timeframe supplies context, the lower the trigger. Compared against the SAME
     single-timeframe entry at matched trade counts, so a filter cannot win by simply trading less.

  4. CONFIRMATION.             Survivors re-measured on the fastest timeframe and the other
     instrument, which is where cross-instrument results have died before.

WHY THE STAGES ARE SEPARATE, AND A WARNING ABOUT STAGE 1's ANSWER BEING THE ONLY STABLE ONE:
re-running an identical grid after a detector fix moved the best-SCORING cell for 13 of 13 chart
patterns, including ones whose ranking across all 108 settings correlated at 0.98-0.999. The score
surface is flat and its peak is noise. So stage 1 selects settings, and stages 2-4 report whether
those settings carry signal — the argmax of stage 2 is NOT a recommendation.

Nothing in this project has yet shown a validated directional edge, and a trendline break is a close
relative of the level-break family already measured across 320 cells and found to carry none. The
expected result here is a clean negative; the point is to know rather than to assume.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                       # noqa: E402
from quant.indicators import add_atr, add_trendlines, trendline_table  # noqa: E402
from experiments.null_control import shuffle_within_day                # noqa: E402
from experiments.phase9_long_fair_test import WINDOWS                  # noqa: E402

OUT_DIR = "experiments"
HORIZONS = [10, 20, 40]
TRAIL, NBIN = 20, 5
MIN_BREAKS = 30            # per (instrument, tf, setting) pooled, below this a number is noise

INSTRUMENTS = [("BTCUSDT", "binance"), ("XAUUSD", "dukascopy")]
TIMEFRAMES = ["1h", "15m", "5m"]          # 1m is added for survivors only, in stage 4

#: 144 combinations. Sized against a MEASURED cost, not a guess: at 324 combinations stage 1 alone
#: ran 5 hours (0.22s per detection per 4,400 bars, and 5m windows carry 52,000). Trimming
#: `line_tol_atr` and `lookback_pivots` to their end points brings the whole run to ~3 hours, which
#: costs very little — the finding that the score surface is flat and its argmax is noise means a
#: finer grid buys precision on a quantity that is not stable anyway.
GRID = dict(
    atr_mult=[2.5, 3.5, 5.0],
    min_touches=[3, 4],
    line_tol_atr=[0.4, 1.0],
    min_span_bars=[20, 40, 80],
    break_buffer_atr=[0.0, 0.3],
    lookback_pivots=[12, 24],
)


# ---------------------------------------------------------------------------
def _combos() -> List[dict]:
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def _load(sym: str, src: str, tf: str, a: str, b: str) -> Optional[pd.DataFrame]:
    try:
        d = get_ohlcv(sym, tf, start=a, end=b, source=src, tz="UTC")
    except Exception:
        return None
    if d is None or len(d) < 800:
        return None
    return add_atr(d, 14).reset_index(drop=True)


def _bins(x: np.ndarray, k: int) -> np.ndarray:
    b = np.zeros(len(x), int)
    ok = np.isfinite(x)
    if ok.sum() > k:
        qs = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, k + 1)[1:-1]))
        b[ok] = np.searchsorted(qs, x[ok])
    return np.clip(b, 0, k - 1)


def _forward(d: pd.DataFrame):
    """Forward moves and the matched-control cell for every bar. Denominators cannot move."""
    c = d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    med = float(np.nanmedian(atr))
    out = {}
    for h in HORIZONS:
        fwd = np.r_[c[h:], np.full(h, np.nan)] - c
        out[h] = (fwd, fwd / med, np.isfinite(fwd))
    trail = np.r_[np.full(TRAIL, np.nan), c[TRAIL:] - c[:-TRAIL]] / med
    cell = _bins(trail, NBIN) * NBIN + _bins(atr, NBIN)
    return out, cell, np.isfinite(trail), med


def measure_breaks(d: pd.DataFrame, cfg: dict) -> List[dict]:
    """One detection pass -> quality stats and, for each horizon, the three edge measures."""
    o = add_trendlines(d, **cfg)
    tl = trendline_table(o)
    fwd_all, cell, okt, med = _forward(d)

    up = o["tl_break_up"].to_numpy(bool)
    dn = o["tl_break_dn"].to_numpy(bool)
    rows = []
    for side, m0, sgn in (("up", up, +1), ("dn", dn, -1)):
        rec = dict(side=side, n_breaks=int(m0.sum()), n_lines=len(tl),
                   broke_frac=float(tl["broke"].mean()) if len(tl) else np.nan,
                   median_touches=float(tl["touches"].median()) if len(tl) else np.nan,
                   pct_bars_live=float(np.isfinite(
                       o["tl_res" if side == "up" else "tl_sup"].to_numpy(float)).mean()))
        for h in HORIZONS:
            fwd, f_fix, ok = fwd_all[h]
            m = m0 & ok
            n = int(m.sum())
            if n == 0:
                continue
            base = float(np.nanmean(f_fix[ok]))
            mm = m & okt
            cmean = np.full(NBIN * NBIN, np.nan)
            for k in range(NBIN * NBIN):
                s = ok & okt & (cell == k)
                if s.sum() >= 30:
                    cmean[k] = f_fix[s].mean()
            mm = mm & np.isfinite(cmean[cell])
            rec[f"n{h}"] = n
            rec[f"edge{h}"] = float((sgn * f_fix[m]).mean() - sgn * base)
            rec[f"matched{h}"] = (float(sgn * (f_fix[mm].mean() - cmean[cell[mm]].mean()))
                                  if mm.any() else np.nan)
            rec[f"hit{h}"] = float((sgn * fwd[m] > 0).mean())
        rows.append(rec)
    return rows


def sweep(shuffle_seed: Optional[int] = None, timeframes=None, combos=None,
          instruments=None, tag: str = "") -> pd.DataFrame:
    combos = combos or _combos()
    tfs = timeframes or TIMEFRAMES
    insts = instruments or INSTRUMENTS
    total = len(insts) * len(tfs) * len(WINDOWS)
    frames, t0, done = [], time.time(), 0
    for sym, src in insts:
        for tf in tfs:
            for wi, (k, (a, b)) in enumerate(WINDOWS.items(), 1):
                d = _load(sym, src, tf, a, b)
                done += 1
                if d is None:
                    print(f"  [{done}/{total}] {tag}{sym} {tf} {k}: no data", flush=True)
                    continue
                if shuffle_seed is not None:
                    d = add_atr(shuffle_within_day(d, seed=shuffle_seed + wi), 14).reset_index(drop=True)
                for cfg in combos:
                    for r in measure_breaks(d, cfg):
                        r.update(symbol=sym, tf=tf, window=k, **cfg)
                        frames.append(r)
                print(f"  [{done}/{total}] {tag}{sym} {tf} {k} "
                      f"({len(d):,} bars, {time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# stage 1 — quality, outcome-blind
# ---------------------------------------------------------------------------
def stage1_quality(raw: pd.DataFrame) -> pd.DataFrame:
    """Which settings give well-formed, frequent lines? Never looks at forward returns.

    A line that is never broken faded into irrelevance; a line broken instantly was never respected.
    A healthy setting sits between, on enough lines per window to mean something.
    """
    keys = ["symbol", "tf"] + list(GRID)
    g = (raw.groupby(keys)
            .agg(breaks=("n_breaks", "sum"), lines=("n_lines", "sum"),
                 broke_frac=("broke_frac", "median"), touches=("median_touches", "median"),
                 pct_live=("pct_bars_live", "median"),
                 worst_window=("n_breaks", "min"), windows=("window", "nunique"))
            .reset_index())
    # outcome-blind score: enough breaks in EVERY window, a sane break rate, real touch support
    g["quality_ok"] = ((g.worst_window >= 3) & (g.broke_frac.between(0.25, 0.85))
                       & (g.touches >= 3))
    return g


def stage1_pick(q: pd.DataFrame, per_tf: int = 3) -> pd.DataFrame:
    """Per (symbol, tf): the strictest settings that pass quality, most lines breaking the tie."""
    out = []
    for (sym, tf), sub in q.groupby(["symbol", "tf"]):
        ok = sub[sub.quality_ok]
        if not len(ok):
            ok = sub.nlargest(per_tf, "breaks")
        ok = ok.sort_values(["min_touches", "atr_mult", "min_span_bars"], ascending=False)
        out.append(ok.head(per_tf))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# stage 2 — the screen, differenced against the null
# ---------------------------------------------------------------------------
def stage2_screen(real: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol", "tf", "side"] + list(GRID)
    agg = {}
    for h in HORIZONS:
        agg[f"n{h}"] = (f"n{h}", "sum")
        agg[f"edge{h}"] = (f"edge{h}", "median")
        agg[f"matched{h}"] = (f"matched{h}", "median")
        agg[f"hit{h}"] = (f"hit{h}", "median")
        agg[f"pos{h}"] = (f"matched{h}", lambda s: int((s > 0).sum()))
    agg["windows"] = ("window", "nunique")
    r = real.groupby(keys).agg(**agg).reset_index()
    n = null.groupby(keys).agg(**agg).reset_index()
    j = r.merge(n, on=keys, suffixes=("", "_null"), how="left")
    for h in HORIZONS:
        j[f"vs_null{h}"] = j[f"matched{h}"] - j[f"matched{h}_null"]
    return j


# ---------------------------------------------------------------------------
# stage 3 — multi-timeframe alignment
# ---------------------------------------------------------------------------
def stage3_mtf(sym: str, src: str, pairs, cfg: dict) -> List[dict]:
    """Does higher-timeframe CONTEXT change what a lower-timeframe break is worth?

    The obvious test — require a break on both timeframes — cannot be answered at this event rate.
    Measured on BTC over six months: 1h produces 6 up-breaks and 14 down-breaks, so a 15m break
    landing inside an 1h confirmation window happens 0-1 times. That is arithmetic, not a null
    result, and reporting it as one would be wrong.

    So the higher timeframe supplies a STATE rather than an event, which every lower-timeframe break
    can be classified by and which therefore splits hundreds of events instead of a handful:
        aligned      the HTF line on the same side is sloping WITH the break direction
        against      it is sloping the other way
    Both groups are reported with their counts, so a difference can be judged at comparable trade
    counts rather than by one group simply trading less — methodology rule 7.
    """
    rows = []
    for hi_tf, lo_tf in pairs:
        for k, (a, b) in WINDOWS.items():
            lo = _load(sym, src, lo_tf, a, b)
            hi = _load(sym, src, hi_tf, a, b)
            if lo is None or hi is None:
                continue
            o_lo = add_trendlines(lo, **cfg)
            o_hi = add_trendlines(hi, **cfg)
            # shift one HTF bar before carrying the state down: the value must have been knowable
            h = o_hi[["t", "tl_sup_slope", "tl_res_slope"]].copy()
            h[["tl_sup_slope", "tl_res_slope"]] = h[["tl_sup_slope", "tl_res_slope"]].shift(1)
            m = pd.merge_asof(o_lo.sort_values("t"), h.sort_values("t"), on="t",
                              direction="backward", suffixes=("", "_hi"))
            fwd_all, cell, okt, med = _forward(lo)
            fwd, f_fix, ok = fwd_all[20]
            base = float(np.nanmean(f_fix[ok]))
            for side, sgn, col in (("up", +1, "tl_res_slope_hi"), ("dn", -1, "tl_sup_slope_hi")):
                brk = m[f"tl_break_{side}"].to_numpy(bool) & ok
                slope = m[col].to_numpy(float) if col in m.columns else np.full(len(m), np.nan)
                have = np.isfinite(slope)
                groups = {
                    "all": brk,
                    "htf_aligned": brk & have & (np.sign(slope) == sgn),
                    "htf_against": brk & have & (np.sign(slope) == -sgn),
                }
                for label, mask in groups.items():
                    if mask.sum() < 5:
                        continue
                    rows.append(dict(symbol=sym, hi_tf=hi_tf, lo_tf=lo_tf, window=k, side=side,
                                     kind=label, n=int(mask.sum()),
                                     edge20=float((sgn * f_fix[mask]).mean() - sgn * base),
                                     hit20=float((sgn * fwd[mask] > 0).mean())))
    return rows


def prewarm(timeframes) -> None:
    """Touch every (instrument, timeframe, window) once so the long run never blocks on the network.

    Dukascopy loads can trigger an incremental fetch mid-sweep. Unattended, that turns a compute job
    into one that stalls on a socket.
    """
    print("pre-warming caches so nothing fetches mid-run...", flush=True)
    for sym, src in INSTRUMENTS:
        for tf in timeframes:
            for k, (a, b) in WINDOWS.items():
                d = _load(sym, src, tf, a, b)
                print(f"   {sym} {tf} {k}: {0 if d is None else len(d):,} bars", flush=True)


# ---------------------------------------------------------------------------
def main(quick: bool = False):
    t0 = time.time()
    combos = _combos()[:4] if quick else _combos()
    tfs = ["1h"] if quick else TIMEFRAMES
    pd.set_option("display.width", 240)

    def hdr(s):
        print("\n" + "=" * 112); print(s); print("=" * 112, flush=True)

    # 1m is deliberately EXCLUDED. Its cache covers only part of the nine-window span, so
    # including it turned a compute job into 1,052 network pages per window -- over four hours of
    # fetching before any measurement began. And 1m carries a 10.8%-of-R cost floor that this
    # project has already shown no entry family clears, so it is the worst hours-per-insight trade
    # available. Fetch it separately if you want it: `quant fetch --symbol BTCUSDT --tf 1m`.
    prewarm(tfs)

    hdr(f"STAGE 1 — QUALITY SWEEP (outcome-blind): {len(combos)} settings x {len(tfs)} timeframes "
        f"x {len(INSTRUMENTS)} instruments x {len(WINDOWS)} windows")
    real = sweep(timeframes=tfs, combos=combos, tag="real ")
    real.to_csv(f"{OUT_DIR}/results_trendline_real.csv", index=False)
    q = stage1_quality(real)
    q.to_csv(f"{OUT_DIR}/results_trendline_quality.csv", index=False)
    print("\nsettings passing the quality gate, per symbol x timeframe:")
    print(q.groupby(["symbol", "tf"]).quality_ok.agg(["sum", "size"]).to_string())
    picks = stage1_pick(q)
    print("\nSELECTED (strictest that pass; these are the STABLE answer):")
    cols = ["symbol", "tf"] + list(GRID) + ["breaks", "lines", "broke_frac", "touches", "pct_live"]
    print(picks[cols].round(3).to_string(index=False))
    picks.to_csv(f"{OUT_DIR}/results_trendline_picks.csv", index=False)

    hdr("STAGE 2 — SCREEN: does a break predict anything? (null-differenced)")
    sel = [dict(zip(GRID, v)) for v in
           picks[list(GRID)].drop_duplicates().itertuples(index=False, name=None)]
    print(f"screening {len(sel)} distinct settings on real bars, then on shuffled bars", flush=True)
    r2 = sweep(timeframes=tfs, combos=sel, tag="real ")
    n2 = sweep(timeframes=tfs, combos=sel, shuffle_seed=7000, tag="null ")
    scr = stage2_screen(r2, n2)
    scr.to_csv(f"{OUT_DIR}/results_trendline_screen.csv", index=False)
    keep = scr[scr.n20 >= MIN_BREAKS]
    show = ["symbol", "tf", "side", "atr_mult", "min_touches", "n20", "edge20", "matched20",
            "matched20_null", "vs_null20", "hit20", "pos20", "windows"]
    print("\nby symbol x timeframe (median over settings), h=20:")
    print(keep.groupby(["symbol", "tf", "side"])[
        ["n20", "edge20", "matched20", "matched20_null", "vs_null20", "hit20"]]
        .median().round(4).to_string())
    print("\ntop cells by real-minus-null:")
    print(keep.nlargest(12, "vs_null20")[show].round(4).to_string(index=False))
    print(f"\n  cells with n>={MIN_BREAKS}: {len(keep)}   beating their null: "
          f"{int((keep.vs_null20 > 0).sum())} ({100*(keep.vs_null20 > 0).mean():.0f}%)  chance = 50%")
    print(f"  median hit rate: real {keep.hit20.median():.4f}   (0.50 = coin flip)")

    hdr("STAGE 3 — MULTI-TIMEFRAME: does higher-timeframe CONTEXT change a break's value?")
    pairs = [("1h", "15m"), ("15m", "5m"), ("1h", "5m")]
    best = sel[0] if sel else dict(atr_mult=3.5, min_touches=3, line_tol_atr=0.6,
                                   min_span_bars=40, break_buffer_atr=0.0)
    mrows = []
    for sym, src in INSTRUMENTS:
        mrows += stage3_mtf(sym, src, pairs if not quick else pairs[:1], best)
    mtf = pd.DataFrame(mrows)
    if len(mtf):
        mtf.to_csv(f"{OUT_DIR}/results_trendline_mtf.csv", index=False)
        piv = (mtf.groupby(["symbol", "hi_tf", "lo_tf", "side", "kind"])
                  .agg(n=("n", "sum"), edge20=("edge20", "median"), hit20=("hit20", "median"))
                  .reset_index())
        print(piv.round(4).to_string(index=False))
        w = piv.pivot_table(index=["symbol", "hi_tf", "lo_tf", "side"], columns="kind",
                            values=["edge20", "n"])
        if ("edge20", "htf_aligned") in w.columns and ("edge20", "htf_against") in w.columns:
            w["gain"] = w[("edge20", "htf_aligned")] - w[("edge20", "htf_against")]
            print("\n  HTF-ALIGNED minus HTF-AGAINST (positive = the higher timeframe helps):")
            print(w["gain"].round(4).to_string())
            print(f"\n  pairs where alignment helps: {int((w['gain'] > 0).sum())} of {len(w)}")
            print("  Judge against the n columns: a split that helps only by trading far less")
            print("  has not been shown to help at all.")
    else:
        print("  no multi-timeframe rows produced")

    hdr("STAGE 4 — CONFIRMATION: the selected settings on the SECOND instrument")
    if not quick and len(sel):
        r4 = sweep(timeframes=tfs, combos=sel[:6],
                   instruments=[i for i in INSTRUMENTS if i[0] != "BTCUSDT"], tag="conf ")
        if len(r4):
            r4.to_csv(f"{OUT_DIR}/results_trendline_1m.csv", index=False)
            g4 = (r4.groupby(["symbol", "side"])
                    .agg(n=("n20", "sum"), edge20=("edge20", "median"),
                         matched20=("matched20", "median"), hit20=("hit20", "median"))
                    .reset_index())
            print(g4.round(4).to_string(index=False))
            print("\n  On 1m the spread is 10.8% of one R, so a directional edge has to clear that")
            print("  before anything else. Read `hit20` first: below 0.50 nothing else matters.")
    else:
        print("  skipped in --quick")

    hdr("VERDICT")
    if len(keep):
        beat = int((keep.vs_null20 > 0).sum())
        print(f"  {beat} of {len(keep)} cells beat their shuffled null ({100*beat/len(keep):.0f}%, "
              f"chance 50%)")
        print(f"  median hit rate {keep.hit20.median():.4f}; a directional edge needs this ABOVE 0.50")
        cons = keep[(keep.vs_null20 > 0) & (keep.pos20 >= keep.windows - 1)]
        print(f"  positive AND consistent in >=8 of 9 windows: {len(cons)}   "
              f"chance ~ {len(keep) * 0.0195:.1f}")
    print(f"\n  total runtime {time.time()-t0:.0f}s")
    print(f"  results -> {OUT_DIR}/results_trendline_*.csv")


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
