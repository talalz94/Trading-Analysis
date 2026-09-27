"""
LEVEL AGE STUDY — do OLD S/R zones and fair value gaps hold better, worse, or the same as young ones?

WHY. Two age limits on the chart were never measured: FVG_MAX_AGE_BARS = 500 was chosen to bound the
mitigation scan (a speed limit), and ZONE_LOOKBACK = 2000 was a workaround for the legacy eviction
bug that the S/R study later fixed. Either can throw away the level price comes back to months later
("BTC is back at 86k") -- or keep clutter that means nothing. Only a measurement separates the two.

WHAT. Same yardstick as experiments/sr_zone_search.py, fixed beforehand and not searched:
    a TEST is price reaching a live level; HOLD if it then closes K_ATR back on the side it came
    from, BREAK if it closes K_ATR through to the other side, first within HORIZON bars.
Every test is tagged with the level's AGE in calendar days, and bucketed. Each bucket is read
against the SHIFT null -- the same levels on the same bars, displaced 1-3 ATR -- because hold rate
alone rises and falls with the market. EDGE = hold rate minus the shifted levels' hold rate.
Positive means an old level's exact price still matters; zero means any nearby price did as well.

    S/R   the tuned "normal" preset per timeframe, on UNBOUNDED history (no lookback), so zones can
          be as old as they survive. 1h/4h retire a zone after 8 closes beyond; they are also run
          with retirement OFF, because retirement is exactly what would kill an old level that
          price broke, left, and returned to.
    FVG   gaps >= 0.5 ATR (the notebooks' gate), no age limit; the test is the FIRST return into
          the gap. Band geometry: hold = close K_ATR beyond the near edge, break = K_ATR beyond the
          far edge. The shifted band has the same size, so its bias cancels in the edge.

Split TRAIN (2022H1-2024H1) / FORWARD (2024H2-2026H1): an age effect that flips sign between the
halves is noise. MIN_TESTS per cell or the rate is not reported.

Run:  python -m experiments.level_age_study        (~3 min)
Out:  experiments/results_level_age.csv            one row per (kind, tf, variant, phase, bucket)
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")
from quant.data.timeframe import tf_minutes                              # noqa: E402
from quant.indicators import sr_params                                   # noqa: E402
from quant.indicators.structure import _confirmed_pivots, _sr_core      # noqa: E402
from quant.indicators.volatility import atr as _atr                     # noqa: E402
from quant.indicators.fvg import fvg_table                              # noqa: E402
from experiments.sr_zone_search import (_frame, _windows_of, TRAIN, FORWARD, TFS,   # noqa: E402
                                        TOL_ATR, GAP_ATR, K_ATR, HORIZON, MAX_ZONES)

OUT = "experiments/results_level_age.csv"
BUCKETS = (0, 1, 7, 30, 90, 365, 10_000)          # days
LABELS = ("<1d", "1-7d", "7-30d", "30-90d", "90-365d", ">1y")
MIN_TESTS = 30
FVG_MIN_SIZE_ATR = 0.5


# ---- S/R: the study's scorer, emitting one record per test ------------------------------------
@njit(cache=True)
def _sr_tests(zone, start, end, px, birth, high, low, close, atr, tol, gap, k, horizon):
    n = close.shape[0]
    cap = 4_000_000
    t_i = np.empty(cap, np.int64)
    t_age = np.empty(cap, np.int64)
    t_res = np.empty(cap, np.int8)
    m = 0
    armed = False
    prev = -1
    for s in range(zone.shape[0]):
        if zone[s] != prev:
            armed = False
            prev = zone[s]
        lv = px[s]
        for i in range(max(start[s], 1), end[s] + 1):
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
            res = 0
            for j in range(i, min(n, i + horizon + 1)):
                away = (close[j] - lv) * side
                if away >= k * a:
                    res = 1
                    break
                if away <= -k * a:
                    res = -1
                    break
            if m < cap:
                t_i[m] = i
                t_age[m] = i - birth[s]
                t_res[m] = res
                m += 1
    return t_i[:m], t_age[:m], t_res[:m]


def _sr(d, h, lo, c, a, tf, break_bars, rng):
    p = sr_params(tf, "normal")
    piv = _confirmed_pivots(d, p["left"], p["right"]).to_numpy(np.float64)
    zo, st, en, px, tc, *_ = _sr_core(piv, c, p["cluster_pct"] / 100.0, MAX_ZONES,
                                      int(break_bars), True)
    # AGE counts from the zone's FIRST pivot -- when the level was first established -- not from
    # the bar it earned enough touches to be drawn
    first = pd.Series(st).groupby(zo).min()
    birth = first.reindex(zo).to_numpy(np.int64)
    keep = tc >= p["min_touches"]
    zo, st, en, px, birth = zo[keep], st[keep], en[keep], px[keep], birth[keep]
    o = np.lexsort((st, zo))
    zo, st, en, px, birth = zo[o], st[o], en[o], px[o], birth[o]
    off = np.zeros(len(px))
    for z in np.unique(zo):
        mz = zo == z
        off[mz] = rng.choice((-1, 1)) * rng.uniform(1, 3) * max(a[st[mz][0]], 1e-12)
    real = _sr_tests(zo, st, en, px, birth, h, lo, c, a, TOL_ATR, GAP_ATR, K_ATR, HORIZON)
    shift = _sr_tests(zo, st, en, px + off, birth, h, lo, c, a, TOL_ATR, GAP_ATR, K_ATR, HORIZON)
    return real, shift


# ---- FVG: the first return into each gap ------------------------------------------------------
@njit(cache=True)
def _fvg_tests(born, top, bot, high, low, close, atr, k, horizon):
    """First bar after birth that trades into [bot, top] from outside it; outcome from there."""
    n = close.shape[0]
    g = born.shape[0]
    t_i = np.full(g, -1, np.int64)
    t_res = np.zeros(g, np.int8)
    for q in range(g):
        hi, lo = top[q], bot[q]
        for i in range(born[q] + 1, n):
            if low[i] <= hi and high[i] >= lo:
                pc = close[i - 1]
                side = 1 if pc > hi else (-1 if pc < lo else 0)
                if side == 0:
                    break                       # the band was already around price: no approach
                a = atr[i]
                if not (a > 0.0):
                    break
                near = hi if side == 1 else lo
                far = lo if side == 1 else hi
                res = 0
                for j in range(i, min(n, i + horizon + 1)):
                    if (close[j] - near) * side >= k * a:
                        res = 1
                        break
                    if (close[j] - far) * side <= -k * a:
                        res = -1
                        break
                t_i[q] = i
                t_res[q] = res
                break
    return t_i, t_res


def _fvg(d, h, lo, c, a, rng):
    # max_age_bars=1 only keeps the library's own mitigation scan cheap -- it is not used here;
    # detection and the size gate do not depend on it
    t = fvg_table(d[["open", "high", "low", "close"]].copy(), min_size_atr=FVG_MIN_SIZE_ATR,
                  max_age_bars=1)
    born = t["born_i"].to_numpy(np.int64)
    top = t["top"].to_numpy(np.float64)
    bot = t["bot"].to_numpy(np.float64)
    off = rng.choice((-1, 1), len(born)) * rng.uniform(1, 3, len(born)) * a[born]
    ri, rr = _fvg_tests(born, top, bot, h, lo, c, a, K_ATR, HORIZON)
    si, sr = _fvg_tests(born, top + off, bot + off, h, lo, c, a, K_ATR, HORIZON)
    keep_r, keep_s = ri >= 0, si >= 0
    return ((ri[keep_r], (ri - born)[keep_r], rr[keep_r]),
            (si[keep_s], (si - born)[keep_s], sr[keep_s]))


# ---- tabulate ---------------------------------------------------------------------------------
def _cells(tests, win, bar_min, phase_of):
    i, age, res = tests
    days = age * bar_min / 1440.0
    b = np.digitize(days, BUCKETS[1:-1], right=False)
    ph = phase_of[win[i]] if len(i) else np.array([], object)
    df = pd.DataFrame({"phase": ph, "bucket": np.asarray(LABELS, object)[b], "res": res})
    df = df[df.phase != ""]
    g = df.groupby(["phase", "bucket"]).res
    return pd.DataFrame({"hold": g.apply(lambda r: int((r == 1).sum())),
                         "brk": g.apply(lambda r: int((r == -1).sum()))})


def main():
    t0 = time.time()
    names = TRAIN + FORWARD
    phase_of = np.array(["train"] * len(TRAIN) + ["forward"] * len(FORWARD) + [""], object)
    rng = np.random.default_rng(0)
    rows = []
    for tf in TFS:
        d = _frame("BTCUSDT", tf)
        h, lo, c = (d[x].to_numpy(np.float64) for x in ("high", "low", "close"))
        a = np.asarray(_atr(d, 14), np.float64)
        win = _windows_of(d["t"].dt.tz_localize(None), names)     # -1 -> phase_of[-1] == ""
        bm = tf_minutes(tf)
        runs = [("sr", "normal", *_sr(d, h, lo, c, a, tf, sr_params(tf, "normal")["break_bars"], rng))]
        if sr_params(tf, "normal")["break_bars"]:
            runs.append(("sr", "no-retire", *_sr(d, h, lo, c, a, tf, 0, rng)))
        runs.append(("fvg", ">=0.5ATR", *_fvg(d, h, lo, c, a, rng)))
        for kind, var, real, shift in runs:
            r = _cells(real, win, bm, phase_of)
            s = _cells(shift, win, bm, phase_of).add_prefix("shift_")
            m = r.join(s, how="outer").fillna(0).reset_index()
            m.insert(0, "variant", var)
            m.insert(0, "tf", tf)
            m.insert(0, "kind", kind)
            rows.append(m)
        print(f"{tf} done  {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    n, sn = out.hold + out.brk, out.shift_hold + out.shift_brk
    out["tests"] = n.astype(int)
    out["hold_rate"] = np.where(n >= MIN_TESTS, out.hold / n.clip(lower=1), np.nan)
    out["shift_rate"] = np.where(sn >= MIN_TESTS, out.shift_hold / sn.clip(lower=1), np.nan)
    out["edge"] = out.hold_rate - out.shift_rate
    out["bucket"] = pd.Categorical(out.bucket, LABELS, ordered=True)
    out = out.sort_values(["kind", "tf", "variant", "phase", "bucket"])
    out.to_csv(OUT, index=False)
    report(out)
    print(f"\n{time.time() - t0:.0f}s -> {OUT}")


def report(out):
    for (kind, tf, var), g in out.groupby(["kind", "tf", "variant"], sort=False):
        p = g.pivot_table(index="bucket", columns="phase", values=["tests", "edge"],
                          aggfunc="first", observed=False)
        print(f"\n=== {kind} {tf} {var}")
        print(f"  {'age':<8} {'train n':>8} {'edge':>7}   {'fwd n':>7} {'edge':>7}")
        for b in LABELS:
            def cell(ph):
                try:
                    n = p[("tests", ph)][b]
                    return (0 if pd.isna(n) else int(n)), p[("edge", ph)][b]
                except KeyError:
                    return 0, np.nan
            (tn, te), (fn, fe) = cell("train"), cell("forward")
            print(f"  {b:<8} {tn:>8} {te:>+7.3f}   {fn:>7} {fe:>+7.3f}")


if __name__ == "__main__":
    main()
