"""
S/R ZONE SETTINGS SEARCH — which settings give levels price actually respects, per timeframe.

WHAT IS BEING MEASURED. A support/resistance zone is only worth drawing if price REACTS at it more
often than at an arbitrary price. So the question is never "how many zones" or "how often does
price touch one" -- a chart covered in lines is touched constantly -- but:

    when price comes back to a zone, does it HOLD more often than it would at a level that
    carries no information?

A TEST is price returning to a live zone after having moved at least GAP_ATR away from it. The
OUTCOME is decided by whichever comes first within HORIZON bars:

    hold    price closes K_ATR back on the side it approached from
    break   price closes K_ATR through to the other side
    neither (excluded from the rate)

HOLD RATE = holds / (holds + breaks). Its level means nothing on its own -- a driftless market
gives ~0.5 at ANY price -- so it is always read against two nulls:

    vs SHUFFLE  the same detector, same settings, on within-day-shuffled bars (the project's
                mandated null, experiments/null_control.py). Real structure is destroyed; drift,
                volatility and candle shapes survive. "Does this procedure find anything real?"
    vs SHIFT    the SAME zones on the SAME bars, each displaced by a random 1-3 ATR. "Is THIS
                price special, or would any nearby price have done as well?"

EDGE = hold rate minus the null's. Positive means the zones carry information.

THE EVALUATION PARAMETERS ARE FIXED BEFOREHAND and are not searched over -- TOL 0.25 ATR, GAP 1.0,
K 1.0, HORIZON 24 bars. Tuning the yardstick alongside the thing being measured is how a flat score
surface acquires a peak. A robustness pass re-scores the picks at K 1.5 / HORIZON 48.

SELECTION, AND WHY IT IS NOT A PLAIN ARGMAX. experiments/trendline_settings_search.py records that
re-running an identical grid moved the best-SCORING cell for 13 of 13 patterns: the surface is flat
and its peak is noise. So:
  * settings are ranked on the TRAIN windows (2022H1-2024H1) and the pick is REPORTED on the
    FORWARD windows (2024H2-2026H1), which it never saw;
  * the rank correlation between train and forward is printed -- if it is near zero, the ranking
    carries no information and the pick is a coin toss, whatever its number says;
  * survivors (edge > 0 in every window) are counted against n_configs x 0.5^n_windows.

THE KNOB. Per timeframe, three presets for `SR_SENSITIVITY`:
    normal  the best forward-validated setting
    tight   the best setting drawing at most ~60% of normal's zones -- fewer, sharper levels
    loose   the best setting drawing at least ~150% of normal's -- more levels, more context
Each is the best AT ITS CLUTTER LEVEL, so moving the knob trades coverage for clarity rather than
trading good settings for bad ones.

Run:  python -m experiments.sr_zone_search           (~8 min)
      python -m experiments.sr_zone_search --picks-only   re-select from the saved results
Out:  experiments/results_sr_zones.csv               every (instrument, tf, setting, window)
      experiments/results_sr_zones_picks.csv         the three presets per timeframe, with evidence
"""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv, resample_ohlcv                        # noqa: E402
from quant.indicators.structure import _confirmed_pivots, _sr_core      # noqa: E402
from quant.indicators.volatility import atr as _atr                     # noqa: E402
from experiments.null_control import shuffle_within_day                 # noqa: E402
from experiments.phase9_long_fair_test import WINDOWS                   # noqa: E402

OUT_DIR = "experiments"

# ---- the yardstick: fixed, never searched -------------------------------------------------------
TOL_ATR, GAP_ATR, K_ATR, HORIZON = 0.25, 1.0, 1.0, 24
ROBUST_K, ROBUST_H = 1.5, 48
MIN_TESTS = 30                     # holds + breaks per window, below this a rate is noise
MAX_ZONES = 40                     # what the chart uses

# ---- the grid -----------------------------------------------------------------------------------
LR = (5, 10, 20)                   # left = right bars a pivot needs
CLUSTER = (0.08, 0.15, 0.25, 0.40)  # % of price
MIN_T = (2, 3, 4)                  # pivots before a zone is drawn
BREAK = (0, 2, 4, 8)               # closes beyond before a zone is retired; 0 = never
ANCHOR = (False, True)

TRAIN = ("2022H1_bear", "2022H2_bear", "2023H1_bull", "2023H2_bull", "2024H1_bull")
FORWARD = ("2024H2_bull", "2025H1_bull", "2025H2_bear", "2026H1_bear")
TFS = ("5m", "15m", "1h", "4h")


@njit(cache=True)
def _score(seg_zone, seg_start, seg_end, seg_px, high, low, close, atr, win,
           n_win, tol, gap, k, horizon):
    """Holds, breaks and 'neither' per window. Segments must be sorted by (zone, start) so a
    zone's ARMED state carries across its own price revisions -- a revision is the same level
    nudged, not a new level, and re-arming at every revision would double-count one approach."""
    n = close.shape[0]
    hold = np.zeros(n_win, np.int64)
    brk = np.zeros(n_win, np.int64)
    nei = np.zeros(n_win, np.int64)
    armed = False
    prev = -1
    for s in range(seg_zone.shape[0]):
        if seg_zone[s] != prev:
            armed = False
            prev = seg_zone[s]
        lv = seg_px[s]
        a0 = max(seg_start[s], 1)
        for i in range(a0, seg_end[s] + 1):
            a = atr[i]
            if not (a > 0.0):
                continue
            if low[i] <= lv <= high[i]:
                d = 0.0
            elif lv > high[i]:
                d = lv - high[i]
            else:
                d = low[i] - lv
            if not armed:
                if d >= gap * a:
                    armed = True
                continue
            if d > tol * a:
                continue
            side = 1 if close[i - 1] > lv else (-1 if close[i - 1] < lv else 0)
            armed = False
            if side == 0:
                continue
            w = win[i]
            if w < 0:
                continue
            ka = k * a
            res = 0
            for j in range(i, min(n, i + horizon + 1)):
                away = (close[j] - lv) * side
                if away >= ka:
                    res = 1
                    break
                if away <= -ka:
                    res = -1
                    break
            if res == 1:
                hold[w] += 1
            elif res == -1:
                brk[w] += 1
            else:
                nei[w] += 1
    return hold, brk, nei


def _windows_of(t: pd.Series, names) -> np.ndarray:
    """Bar -> index into `names`, or -1 for bars outside every window."""
    out = np.full(len(t), -1, np.int64)
    tv = t.to_numpy()
    for w, nm in enumerate(names):
        a, b = WINDOWS[nm]
        m = (tv >= np.datetime64(pd.Timestamp(a, tz="UTC").tz_localize(None))) & \
            (tv < np.datetime64(pd.Timestamp(b, tz="UTC").tz_localize(None)))
        out[m] = w
    return out


def _frame(sym: str, tf: str) -> pd.DataFrame:
    """Full cached history for one timeframe. 4h is built from 1h: the cached 4h file only goes
    back a year, and resample_ohlcv is exact going DOWN in resolution."""
    if tf == "4h":
        d = get_ohlcv(sym, "1h", start="2021-12-01", end="2026-09-01", tz="UTC")
        d = resample_ohlcv(d, "240min")
    else:
        d = get_ohlcv(sym, tf, start="2021-12-01", end="2026-09-01", tz="UTC")
    d = d.reset_index(drop=True)
    d["t"] = pd.to_datetime(d["open_time"] if "open_time" in d else d["t"], utc=True)
    return d


def _segments(pivots, close, cluster, break_bars, anchor, min_t):
    zo, st, en, px, tc, *_ = _sr_core(pivots, close, cluster / 100.0, MAX_ZONES,
                                      int(break_bars), bool(anchor))
    keep = tc >= min_t
    zo, st, en, px = zo[keep], st[keep], en[keep], px[keep]
    order = np.lexsort((st, zo))
    return zo[order], st[order], en[order], px[order]


NEAR_ATR = 20.0                   # "on screen": zones within this many ATR of the close


@njit(cache=True)
def _near_live(st, en, px, close, atr, n):
    """Zones live AND within NEAR_ATR of the close, per bar. Counting every live zone overstated
    clutter badly: with retirement freeing slots, zones price trended away from linger 25% off --
    never drawn (the chart filters by proximity), never tested, never blocking anything."""
    cnt = np.zeros(n, np.int64)
    for s in range(st.shape[0]):
        for i in range(st[s], en[s] + 1):
            a = atr[i]
            if a > 0.0 and abs(px[s] - close[i]) <= NEAR_ATR * a:
                cnt[i] += 1
    return cnt


def _live_mean(st, en, px, close, atr, win, n_win):
    """Mean number of zones a chart would actually DRAW per bar, per window."""
    live = _near_live(st, en, px, close, atr, close.shape[0])
    out = np.full(n_win, np.nan)
    for w in range(n_win):
        m = win == w
        if m.any():
            out[w] = live[m].mean()
    return out


def run_one(sym: str, tf: str, names, seed: int = 0) -> pd.DataFrame:
    real = _frame(sym, tf)
    shuf = shuffle_within_day(real, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for tag, d in (("real", real), ("shuffle", shuf)):
        h, lo, c = (d[x].to_numpy(np.float64) for x in ("high", "low", "close"))
        a = np.asarray(_atr(d, 14), np.float64)
        win = _windows_of(d["t"].dt.tz_localize(None), names)
        piv = {lr: _confirmed_pivots(d, lr, lr).to_numpy(np.float64) for lr in LR}
        for lr, cl, mt, bb, an in itertools.product(LR, CLUSTER, MIN_T, BREAK, ANCHOR):
            zo, st, en, px = _segments(piv[lr], c, cl, bb, an, mt)
            hold, brk, nei = _score(zo, st, en, px, h, lo, c, a, win, len(names),
                                    TOL_ATR, GAP_ATR, K_ATR, HORIZON)
            live = _live_mean(st, en, px, c, a, win, len(names))
            key = dict(sym=sym, tf=tf, lr=lr, cluster=cl, min_t=mt, break_bars=bb, anchor=an)
            if tag == "real":
                # SHIFT null: the same zones, each moved a random 1-3 ATR, on the same bars
                off = np.zeros(len(px))
                for z in np.unique(zo):
                    m = zo == z
                    s0 = st[m][0]
                    off[m] = rng.choice((-1, 1)) * rng.uniform(1, 3) * max(a[s0], 1e-12)
                sh, sb, _ = _score(zo, st, en, px + off, h, lo, c, a, win, len(names),
                                   TOL_ATR, GAP_ATR, K_ATR, HORIZON)
                rh, rb, _ = _score(zo, st, en, px, h, lo, c, a, win, len(names),
                                   TOL_ATR, GAP_ATR, ROBUST_K, ROBUST_H)
            for w, nm in enumerate(names):
                r = dict(key, window=nm, data=tag, hold=int(hold[w]), brk=int(brk[w]),
                         neither=int(nei[w]), live=float(live[w]))
                if tag == "real":
                    r.update(shift_hold=int(sh[w]), shift_brk=int(sb[w]),
                             robust_hold=int(rh[w]), robust_brk=int(rb[w]))
                rows.append(r)
    return pd.DataFrame(rows)


def _rate(h, b):
    n = h + b
    return np.where(n >= MIN_TESTS, h / np.maximum(n, 1), np.nan)


def summarise(df: pd.DataFrame, names) -> pd.DataFrame:
    keys = ["sym", "tf", "lr", "cluster", "min_t", "break_bars", "anchor", "window"]
    re = df[df.data == "real"].set_index(keys)
    nu = df[df.data == "shuffle"].set_index(keys)
    out = re[["hold", "brk", "live", "shift_hold", "shift_brk",
              "robust_hold", "robust_brk"]].copy()
    out["rate"] = _rate(out.hold, out.brk)
    out["null_rate"] = _rate(nu.hold.reindex(out.index), nu.brk.reindex(out.index))
    out["shift_rate"] = _rate(out.shift_hold, out.shift_brk)
    out["edge"] = out.rate - out.null_rate
    out["edge_shift"] = out.rate - out.shift_rate
    out["robust_rate"] = _rate(out.robust_hold, out.robust_brk)
    return out.reset_index()


# THE KNOB IS A CLARITY DIAL. Chosen AFTER the first run showed that no setting makes the LEVEL
# itself predictive: against the SHIFT null (the same zones moved 1-3 ATR) the mean edge is
# negative on 5m, 15m and 1h -- a price next to a zone holds as well as the zone does. What beats
# the SHUFFLE null is most likely generic short-term mean reversion that real BTC has and shuffled
# bars lack, which any nearby level would capture. So "best" cannot mean "most predictive", and
# ranking every setting on edge picked degenerate ones -- 15m "normal" drew 0.5 lines and went
# negative forward. Each level is instead a BAND of lines on screen, and edge only breaks ties
# inside it:
BANDS = {"tight": (2, 4), "normal": (4, 8), "loose": (8, 16)}   # zones within NEAR_ATR of price
LEGACY = dict(lr=10, cluster=0.15, min_t=2, break_bars=0, anchor=False)


def _table(s: pd.DataFrame, tf: str, train, forward) -> pd.DataFrame:
    """One row per setting: train and forward edge, the shift-null edge, windows positive, and
    the clutter it draws."""
    cfg = ["lr", "cluster", "min_t", "break_bars", "anchor"]
    rows = []
    for k, x in s[s.tf == tf].groupby(cfg):
        tr, fw = x[x.window.isin(train)], x[x.window.isin(forward)]
        rows.append(dict(zip(cfg, k), train_edge=tr.edge.mean(), fwd_edge=fw.edge.mean(),
                         fwd_edge_shift=fw.edge_shift.mean(),
                         train_n=int(tr.edge.notna().sum()), fwd_n=int(fw.edge.notna().sum()),
                         all_pos=int((x.edge > 0).sum()), all_valid=int(x.edge.notna().sum()),
                         live=x.live.mean()))
    return pd.DataFrame(rows)


def pick(s: pd.DataFrame, tf: str, train, forward) -> dict:
    """The three presets for one timeframe, and how much the ranking behind them can be trusted.

    Candidates are ANCHORED only: anchoring moved edge by +-0.003 on every timeframe -- noise --
    and it is what makes a zone's width honest, so it costs nothing to require. Within each
    clutter band the setting with the best TRAIN edge wins, and its FORWARD edge is reported on
    windows it never saw. A band with no candidate that has enough tests is re-tried with the
    test requirement relaxed by one window, then falls back to the closest band member, because a
    timeframe with few bars (4h) cannot fill every band at full strictness.
    """
    t = _table(s, tf, train, forward)
    strict = t[(t.train_n >= len(train) - 1) & (t.fwd_n >= len(forward) - 1)]
    loose_n = t[(t.train_n >= len(train) - 2) & (t.fwd_n >= len(forward) - 2)]
    rho = strict[["train_edge", "fwd_edge"]].rank().corr().iloc[0, 1] if len(strict) > 2 else np.nan
    nwin = len(train) + len(forward)
    out = dict(tf=tf, rho=float(rho), n_cfg=len(strict),
               survivors=int((strict.all_pos == strict.all_valid).sum()),
               chance=round(len(strict) * 0.5 ** nwin, 2))
    anch = lambda d: d[d.anchor.astype(bool)]
    for lvl, (lo, hi) in BANDS.items():
        r, how = None, "strict"
        for pool, tag in ((anch(strict), "strict"), (anch(loose_n), "relaxed")):
            b = pool[(pool.live >= lo) & (pool.live < hi)]
            if len(b):
                r, how = b.sort_values("train_edge", ascending=False).iloc[0], tag
                break
        if r is None:
            pool = anch(loose_n)
            r = pool.iloc[(pool.live - (lo + hi) / 2).abs().argsort()[:1]].iloc[0]
            how = "nearest"
        out[lvl] = (r, how)
    leg = t[(t.lr == LEGACY["lr"]) & (t.cluster == LEGACY["cluster"])
            & (t.min_t == LEGACY["min_t"]) & (t.break_bars == LEGACY["break_bars"])
            & (~t.anchor.astype(bool))]
    out["legacy_live"] = float(leg.live.iloc[0]) if len(leg) else np.nan
    return out


def report(s: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for tf in TFS:
        p = pick(s, tf, TRAIN, FORWARD)
        trust = ("STABLE" if p["rho"] >= 0.5 else "weak" if p["rho"] >= 0.2 else "NOISE")
        print(f"\n=== {tf}   train->forward rank corr {p['rho']:+.2f} ({trust})   survivors "
              f"{p['survivors']} of {p['n_cfg']} (chance {p['chance']})   "
              f"legacy draws {p['legacy_live']:.1f} lines")
        for lvl in BANDS:
            r, how = p[lvl]
            print(f"  {lvl:<7} lr={int(r.lr):>2} cluster={r.cluster:.2f} min_t={int(r.min_t)} "
                  f"break={int(r.break_bars)}  lines {r.live:4.1f}  train {r.train_edge:+.3f}  "
                  f"FORWARD {r.fwd_edge:+.3f}  vs-shift {r.fwd_edge_shift:+.3f}  "
                  f"pos {int(r.all_pos)}/{int(r.all_valid)}  [{how}]")
            picks.append(dict(tf=tf, level=lvl, lr=int(r.lr), cluster=float(r.cluster),
                              min_t=int(r.min_t), break_bars=int(r.break_bars), anchor=True,
                              lines=round(float(r.live), 1), train_edge=r.train_edge,
                              fwd_edge=r.fwd_edge, fwd_edge_shift=r.fwd_edge_shift,
                              windows_positive=f"{int(r.all_pos)}/{int(r.all_valid)}",
                              selection=how, rho=p["rho"], trust=trust,
                              survivors=p["survivors"], chance=p["chance"],
                              legacy_lines=round(p["legacy_live"], 1)))
    out = pd.DataFrame(picks)
    out.to_csv(f"{OUT_DIR}/results_sr_zones_picks.csv", index=False)
    return out


def main():
    import sys
    t0 = time.time()
    if "--picks-only" in sys.argv:
        s = pd.read_csv(f"{OUT_DIR}/results_sr_zones.csv")
    else:
        names = TRAIN + FORWARD
        parts = []
        for tf in TFS:
            print(f"BTCUSDT {tf} ...", flush=True)
            parts.append(run_one("BTCUSDT", tf, names))
        s = summarise(pd.concat(parts, ignore_index=True), names)
        s.to_csv(f"{OUT_DIR}/results_sr_zones.csv", index=False)
    report(s)
    print(f"\n{time.time() - t0:.0f}s  ->  {OUT_DIR}/results_sr_zones*.csv")


if __name__ == "__main__":
    main()
