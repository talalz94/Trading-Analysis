"""
Market structure: swing highs/lows (pivots), lookahead-safe last-swing levels, and classic
(floor-trader) pivot points. The last_swing_* columns are the intended ref_col stop sources.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.timeframe import resample_ohlcv

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def w(f):
            return f
        return w(a[0]) if a and callable(a[0]) else w


def add_swings(df: pd.DataFrame, left: int = 10, right: int = 10, *,
               prefix: str = "swing") -> pd.DataFrame:
    """Detect pivot highs/lows and expose lookahead-safe rolling swing levels.

    A pivot high at bar p is the local max over [p-left, p+right]; it is only *confirmed*
    `right` bars later. `{prefix}_last_high` / `{prefix}_last_low` forward-fill the last
    confirmed pivot and are safe to use as stop references (no future leakage).
    """
    out = df.copy()
    w = int(left) + int(right) + 1
    high, low = out["high"], out["low"]

    roll_max = high.rolling(w, center=True, min_periods=w).max()
    roll_min = low.rolling(w, center=True, min_periods=w).min()
    is_ph = (high == roll_max)
    is_pl = (low == roll_min)

    ph_price = high.where(is_ph)
    pl_price = low.where(is_pl)

    # confirmed (and thus usable) `right` bars after the pivot bar
    out[f"{prefix}_high"] = is_ph.shift(right, fill_value=False)
    out[f"{prefix}_low"] = is_pl.shift(right, fill_value=False)
    out[f"{prefix}_last_high"] = ph_price.shift(right).ffill()
    out[f"{prefix}_last_low"] = pl_price.shift(right).ffill()
    return out


def _confirmed_pivot_seq(df: pd.DataFrame, left: int, right: int):
    """Confirmed pivots in chronological order: list of (confirm_bar, price, is_high).

    A pivot at bar p is only KNOWN at p+right, so it is emitted at its confirmation bar — the
    sequence therefore contains nothing that wasn't visible at the time.
    """
    w = int(left) + int(right) + 1
    high, low = df["high"], df["low"]
    is_ph = (high == high.rolling(w, center=True, min_periods=w).max())
    is_pl = (low == low.rolling(w, center=True, min_periods=w).min())
    ph = high.where(is_ph).shift(right)
    pl = low.where(is_pl).shift(right)
    ph_v, pl_v = ph.to_numpy(np.float64), pl.to_numpy(np.float64)
    seq = []
    for i in range(len(df)):
        if ph_v[i] == ph_v[i]:
            seq.append((i, float(ph_v[i]), True))
        if pl_v[i] == pl_v[i]:
            seq.append((i, float(pl_v[i]), False))
    return seq


def add_trend_structure(df: pd.DataFrame, left: int = 10, right: int = 10, *,
                        prefix: str = "struct") -> pd.DataFrame:
    """Classify market structure from confirmed swings: **HH/HL = up, LH/LL = down, else sideways**.

    Each confirmed pivot is labelled against the previous pivot of the SAME kind:
      highs -> 'HH' (higher high) or 'LH' (lower high)
      lows  -> 'HL' (higher low)  or 'LL' (lower low)
    The regime is then: up-trend while the latest high is HH *and* the latest low is HL;
    down-trend while LH *and* LL; anything else (mixed) is a range.

    Adds:
      {prefix}_label   'HH'|'HL'|'LH'|'LL' at each pivot's CONFIRMATION bar (None elsewhere)
      {prefix}_price   that pivot's price (NaN elsewhere) — for plotting the label
      {prefix}_trend    1 up / -1 down / 0 sideways, forward-filled per bar
    Anti-lookahead: a pivot only exists from the bar it is confirmed, so the regime updates late
    (by `right` bars) exactly as it would live.
    """
    out = df.copy()
    n = len(out)
    label = np.full(n, None, dtype=object)
    price = np.full(n, np.nan)
    trend = np.zeros(n, np.int8)

    prev_high = prev_low = np.nan
    hi_state = lo_state = ""        # 'HH'/'LH' and 'HL'/'LL'
    cur = 0
    seq = _confirmed_pivot_seq(out, left, right)
    k = 0
    for i in range(n):
        while k < len(seq) and seq[k][0] == i:
            _, px, is_high = seq[k]
            if is_high:
                if prev_high == prev_high:
                    hi_state = "HH" if px > prev_high else "LH"
                    label[i] = hi_state
                    price[i] = px
                prev_high = px
            else:
                if prev_low == prev_low:
                    lo_state = "HL" if px > prev_low else "LL"
                    label[i] = lo_state
                    price[i] = px
                prev_low = px
            if hi_state == "HH" and lo_state == "HL":
                cur = 1
            elif hi_state == "LH" and lo_state == "LL":
                cur = -1
            else:
                cur = 0
            k += 1
        trend[i] = cur

    out[f"{prefix}_label"] = label
    out[f"{prefix}_price"] = price
    out[f"{prefix}_trend"] = trend
    return out


@njit(cache=True)
def _sr_scan(pivot_px, close, cluster_frac, max_zones):
    """Causal support/resistance scan. `pivot_px[i]` is a newly-CONFIRMED pivot price at bar i
    (NaN if none). Zones accumulate as pivots cluster (within cluster_frac of an existing zone);
    each zone tracks a running mean price and a touch count. Per bar, report the nearest zone at
    or below close (support) and at or above close (resistance) with their touch counts. Only
    information available up to bar i is used (no lookahead)."""
    n = close.shape[0]
    zp = np.zeros(max_zones)          # zone price (running mean)
    zt = np.zeros(max_zones, np.int64)  # zone touch count
    zlast = np.zeros(max_zones, np.int64)  # last-updated bar (for eviction)
    nz = 0

    sup = np.full(n, np.nan)
    res = np.full(n, np.nan)
    sup_t = np.zeros(n, np.int64)
    res_t = np.zeros(n, np.int64)

    for i in range(n):
        p = pivot_px[i]
        if p == p and p > 0.0:        # a pivot confirmed on this bar
            best = -1
            best_d = 1e300
            for z in range(nz):
                d = abs(zp[z] - p)
                if d <= cluster_frac * p and d < best_d:
                    best_d = d
                    best = z
            if best >= 0:
                zt[best] += 1
                zp[best] += (p - zp[best]) / zt[best]   # incremental mean
                zlast[best] = i
            elif nz < max_zones:
                zp[nz] = p; zt[nz] = 1; zlast[nz] = i; nz += 1
            else:                     # full: evict the least-touched, oldest zone
                worst = 0
                for z in range(1, nz):
                    if zt[z] < zt[worst] or (zt[z] == zt[worst] and zlast[z] < zlast[worst]):
                        worst = z
                zp[worst] = p; zt[worst] = 1; zlast[worst] = i

        c = close[i]
        s_px = -1.0; s_ct = 0; r_px = -1.0; r_ct = 0
        for z in range(nz):
            if zp[z] <= c:
                if s_px < 0.0 or zp[z] > s_px:
                    s_px = zp[z]; s_ct = zt[z]
            else:
                if r_px < 0.0 or zp[z] < r_px:
                    r_px = zp[z]; r_ct = zt[z]
        if s_px > 0.0:
            sup[i] = s_px; sup_t[i] = s_ct
        if r_px > 0.0:
            res[i] = r_px; res_t[i] = r_ct
    return sup, res, sup_t, res_t


@njit(cache=True)
def _sgn(x):
    return 1 if x > 0.0 else (-1 if x < 0.0 else 0)


@njit(cache=True)
def _sr_core(pivot_px, close, cluster_frac, max_zones, break_bars, anchor):
    """The zone engine with the two corrections the legacy scans lack. ONE implementation behind
    both `add_sr_zones` and `sr_zone_table`, so the per-bar levels a strategy reads and the lines a
    chart draws cannot disagree -- the legacy pair were documented as mirrors but evicted with
    different tie-breaks, so at capacity they could hold different zones.

    ANCHORED (`anchor`). A pivot joins a zone only if the zone's whole band -- min to max of every
    pivot in it, including the new one -- stays within `cluster_frac` of price. The legacy test
    compared against the zone's RUNNING MEAN, which moves as pivots join, so a zone could creep:
    measured on BTC 5m at ~63k, where 0.15% is ~94 points, the strongest zones spanned 149-157.

    RETIRED ON A BREAK (`break_bars` > 0). Each zone remembers which side price sat on when a pivot
    last touched it. `break_bars` consecutive closes on the OTHER side retire it; a close back on
    the original side resets the count, so a wick-and-reclaim does not kill a level. The legacy
    scans never looked at price crossing a zone at all -- 48 of 54 zones on one chart were levels
    price had broken and held beyond for an hour or more.

    EVICTION, when all `max_zones` slots are live, drops the zone FARTHEST FROM PRICE. The legacy
    scans dropped the least-touched, which protects old strong levels and kills every new one --
    on a trending instrument the table filled with unreachable levels (BTC 4h, 2022-26: 40 live
    zones a median 67% from price, 1 within 5%) and new zones next to price were evicted as they
    formed. Zone ids are STABLE and unique; the legacy table reused the slot index, so an evicted
    zone and its replacement shared an id and the chart's one-line-per-zone step dropped one.

    Returns the segments (zone, start, end, price, touches, lo, hi) plus the nearest support and
    resistance per bar with their touch counts. Strictly causal: bar i uses nothing after bar i.
    """
    n = close.shape[0]
    zp = np.zeros(max_zones)
    zt = np.zeros(max_zones, np.int64)
    zlo = np.zeros(max_zones)
    zhi = np.zeros(max_zones)
    zlast = np.zeros(max_zones, np.int64)
    zseg = np.zeros(max_zones, np.int64)
    zid = np.zeros(max_zones, np.int64)
    zside = np.zeros(max_zones, np.int64)
    zcnt = np.zeros(max_zones, np.int64)
    live = np.zeros(max_zones, np.bool_)
    nz = 0
    nid = 0
    npiv = 0
    for i in range(n):
        if pivot_px[i] == pivot_px[i] and pivot_px[i] > 0.0:
            npiv += 1
    cap = 2 * npiv + max_zones + 8
    o_zone = np.empty(cap, np.int64)
    o_start = np.empty(cap, np.int64)
    o_end = np.empty(cap, np.int64)
    o_px = np.empty(cap)
    o_t = np.empty(cap, np.int64)
    o_lo = np.empty(cap)
    o_hi = np.empty(cap)
    ns = 0
    sup = np.full(n, np.nan)
    res = np.full(n, np.nan)
    sup_t = np.zeros(n, np.int64)
    res_t = np.zeros(n, np.int64)

    for i in range(n):
        c = close[i]
        p = pivot_px[i]
        if p == p and p > 0.0:
            best = -1
            best_d = 1e300
            for z in range(nz):
                if not live[z]:
                    continue
                d = abs(zp[z] - p)
                if anchor:
                    ok = (max(zhi[z], p) - min(zlo[z], p)) <= cluster_frac * p
                else:
                    ok = d <= cluster_frac * p
                if ok and d < best_d:
                    best_d = d
                    best = z
            if best >= 0:
                z = best
                if i - 1 >= zseg[z]:                 # the old level held THROUGH i-1
                    o_zone[ns] = zid[z]
                    o_start[ns] = zseg[z]
                    o_end[ns] = i - 1
                    o_px[ns] = zp[z]
                    o_t[ns] = zt[z]
                    o_lo[ns] = zlo[z]
                    o_hi[ns] = zhi[z]
                    ns += 1
                zt[z] += 1
                zp[z] += (p - zp[z]) / zt[z]
                zlo[z] = min(zlo[z], p)
                zhi[z] = max(zhi[z], p)
                zlast[z] = i
                zseg[z] = i
                zcnt[z] = 0
                sd = _sgn(c - zp[z])
                if sd != 0:
                    zside[z] = sd
            else:
                slot = -1
                for z in range(nz):
                    if not live[z]:
                        slot = z
                        break
                if slot < 0 and nz < max_zones:
                    slot = nz
                    nz += 1
                if slot < 0:
                    # EVICT THE FARTHEST FROM PRICE, not the least-touched. The legacy rule kept
                    # old heavily-touched zones and threw away every new one, so on a trending
                    # instrument the table filled with levels nobody could reach: BTC 4h over
                    # 2022-26 ended with 40 live zones a median 67% from price, 1 within 5%.
                    # The chart hides distant zones anyway -- the damage was the SLOTS they held.
                    w = 0
                    for z in range(1, nz):
                        if abs(zp[z] - c) > abs(zp[w] - c):
                            w = z
                    if i - 1 >= zseg[w]:
                        o_zone[ns] = zid[w]
                        o_start[ns] = zseg[w]
                        o_end[ns] = i - 1
                        o_px[ns] = zp[w]
                        o_t[ns] = zt[w]
                        o_lo[ns] = zlo[w]
                        o_hi[ns] = zhi[w]
                        ns += 1
                    slot = w
                zp[slot] = p
                zt[slot] = 1
                zlo[slot] = p
                zhi[slot] = p
                zlast[slot] = i
                zseg[slot] = i
                zcnt[slot] = 0
                live[slot] = True
                zid[slot] = nid
                nid += 1
                zside[slot] = _sgn(c - p)

        if break_bars > 0:
            for z in range(nz):
                if not live[z]:
                    continue
                sd = _sgn(c - zp[z])
                if zside[z] == 0:                    # formed exactly on the close: adopt a side
                    if sd != 0:
                        zside[z] = sd
                    continue
                if sd == -zside[z]:
                    zcnt[z] += 1
                elif sd == zside[z]:
                    zcnt[z] = 0
                if zcnt[z] >= break_bars:
                    o_zone[ns] = zid[z]
                    o_start[ns] = zseg[z]
                    o_end[ns] = i
                    o_px[ns] = zp[z]
                    o_t[ns] = zt[z]
                    o_lo[ns] = zlo[z]
                    o_hi[ns] = zhi[z]
                    ns += 1
                    live[z] = False

        s_px = -1.0
        s_ct = 0
        r_px = -1.0
        r_ct = 0
        for z in range(nz):
            if not live[z]:
                continue
            if zp[z] <= c:
                if s_px < 0.0 or zp[z] > s_px:
                    s_px = zp[z]
                    s_ct = zt[z]
            else:
                if r_px < 0.0 or zp[z] < r_px:
                    r_px = zp[z]
                    r_ct = zt[z]
        if s_px > 0.0:
            sup[i] = s_px
            sup_t[i] = s_ct
        if r_px > 0.0:
            res[i] = r_px
            res_t[i] = r_ct

    for z in range(nz):
        if live[z] and n - 1 >= zseg[z]:
            o_zone[ns] = zid[z]
            o_start[ns] = zseg[z]
            o_end[ns] = n - 1
            o_px[ns] = zp[z]
            o_t[ns] = zt[z]
            o_lo[ns] = zlo[z]
            o_hi[ns] = zhi[z]
            ns += 1
    return (o_zone[:ns], o_start[:ns], o_end[:ns], o_px[:ns], o_t[:ns], o_lo[:ns], o_hi[:ns],
            sup, res, sup_t, res_t)


# ---- THE KNOB -----------------------------------------------------------------------------------
# Per-timeframe presets for `sr_params(tf, sensitivity)`, chosen by experiments/sr_zone_search.py
# (BTCUSDT, 2022-2026, the project's 9 standard windows; selected on 2022H1-2024H1, reported on
# 2024H2-2026H1). Evidence per row: experiments/results_sr_zones_picks.csv. Verdict and caveats:
# docs/WHAT_WORKS.md. DO NOT HAND-EDIT a value without re-running the study.
#
# IT IS A CLARITY DIAL, NOT A QUALITY DIAL. No setting made the LEVEL itself predictive: against
# the same zones moved 1-3 ATR, price holds about as often (edge ~0, often negative). So each
# level is a band of lines on screen -- tight 2-4, normal 4-8, loose 8-16 zones within 20 ATR of
# price -- and edge only breaks ties inside a band. The old defaults drew 18.3 / 21.1 / 17.2 / 8.3
# lines on 5m / 15m / 1h / 4h; "normal" draws about a quarter of that, with no measurable change
# in how often price reacts at them. The ranking behind the tie-breaks is STABLE on 5m (train->
# forward rank corr +0.81) and 15m (+0.66) and NOISE on 1h (+0.02) and 4h (+0.16): on those two,
# any setting in a band is as good as any other. Retirement (break_bars) is chosen by the data,
# not forced -- on 5m and 15m it measurably lowers the reaction rate (broken levels keep acting as
# levels: role reversal), so "normal" there keeps them and gets its clarity from tighter clusters.
SR_LEVELS = ("tight", "normal", "loose")
SR_LEGACY = dict(left=10, right=10, cluster_pct=0.15, min_touches=2, break_bars=0, anchor=False)
SR_TUNED: dict = {
    "5m": {
        "tight": dict(left=5, right=5, cluster_pct=0.25, min_touches=4, break_bars=4, anchor=True),   # 2.7 lines
        "normal": dict(left=20, right=20, cluster_pct=0.08, min_touches=4, break_bars=0, anchor=True),   # 5.5 lines
        "loose": dict(left=5, right=5, cluster_pct=0.4, min_touches=4, break_bars=0, anchor=True),   # 15.0 lines
    },
    "15m": {
        "tight": dict(left=20, right=20, cluster_pct=0.08, min_touches=4, break_bars=0, anchor=True),   # 2.3 lines
        "normal": dict(left=10, right=10, cluster_pct=0.08, min_touches=4, break_bars=0, anchor=True),   # 4.8 lines
        "loose": dict(left=10, right=10, cluster_pct=0.08, min_touches=3, break_bars=0, anchor=True),   # 9.6 lines
    },
    "1h": {
        "tight": dict(left=10, right=10, cluster_pct=0.08, min_touches=2, break_bars=8, anchor=True),   # 2.0 lines
        "normal": dict(left=10, right=10, cluster_pct=0.25, min_touches=2, break_bars=8, anchor=True),   # 4.3 lines
        "loose": dict(left=20, right=20, cluster_pct=0.15, min_touches=2, break_bars=0, anchor=True),   # 10.5 lines
    },
    "4h": {
        "tight": dict(left=5, right=5, cluster_pct=0.25, min_touches=2, break_bars=4, anchor=True),   # 3.2 lines
        "normal": dict(left=5, right=5, cluster_pct=0.25, min_touches=2, break_bars=8, anchor=True),   # 4.2 lines
        "loose": dict(left=10, right=10, cluster_pct=0.15, min_touches=2, break_bars=0, anchor=True),   # 8.3 lines
    },
}


def sr_params(tf, sensitivity="normal", **overrides) -> dict:
    """The S/R settings for one timeframe, from the tuned table.

    `sensitivity` is the knob: "tight" (fewer, sharper levels), "normal" (the forward-validated
    best) or "loose" (more levels, more context) -- each is the best setting AT ITS CLUTTER LEVEL,
    so turning it trades coverage for clarity rather than good settings for bad ones.
    `sensitivity=None` is LEGACY: the pre-2026-09-27 values, no anchoring, no retirement, which
    take the original code path and reproduce the old zones byte for byte.

    A timeframe that was not tuned borrows the nearest one that was, by bar length. Any keyword in
    `overrides` that is not None wins over the table, so an explicit setting in a config is never
    silently replaced.
    """
    if sensitivity is None:
        out = dict(SR_LEGACY)
    else:
        lvl = str(sensitivity).lower()
        if lvl not in SR_LEVELS:
            raise ValueError(f"sensitivity must be one of {SR_LEVELS} or None, got {sensitivity!r}")
        out = dict(SR_TUNED[_nearest_tuned_tf(tf)][lvl])
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _nearest_tuned_tf(tf) -> str:
    """The tuned timeframe closest to `tf` in log bar length -- 30m borrows 15m or 1h, not 4h."""
    from ..data.timeframe import tf_minutes
    if tf in SR_TUNED:
        return tf
    try:
        m = max(tf_minutes(tf), 1)
    except Exception:
        return "5m"
    return min(SR_TUNED, key=lambda k: abs(np.log(tf_minutes(k)) - np.log(m)))


def _legacy(break_bars, anchor) -> bool:
    """The pre-2026-09-27 behaviour, kept on its ORIGINAL code path so every existing caller --
    strategies and scenarios whose results are recorded in docs/ -- stays byte-identical."""
    return int(break_bars or 0) <= 0 and not anchor


def add_sr_zones(df: pd.DataFrame, left: int = 10, right: int = 10, *, cluster_pct: float = 0.15,
                 max_zones: int = 128, prefix: str = "sr", break_bars: int = 0,
                 anchor: bool = False) -> pd.DataFrame:
    """Detect swing-based support/resistance ZONES and expose the nearest ones per bar.

    Swing pivots (confirmed `right` bars later, like `add_swings`) are clustered into zones when
    they fall within `cluster_pct`% of each other; a zone's *touch count* is how many pivots landed
    in it — a proxy for strength (major = many touches, minor = few). Anti-lookahead: a zone only
    exists after its pivots are confirmed. Adds:
      `{prefix}_support` / `{prefix}_resistance`         nearest zone below / above the close
      `{prefix}_support_touches` / `{prefix}_resistance_touches`   their strength (touch counts)

    Filter to 'major' levels downstream with the touch columns; run on a higher timeframe (via the
    builder's `sr_zones(tf=...)`) for major structural levels.
    """
    out = df.copy()
    w = int(left) + int(right) + 1
    high, low = out["high"], out["low"]
    roll_max = high.rolling(w, center=True, min_periods=w).max()
    roll_min = low.rolling(w, center=True, min_periods=w).min()
    ph = high.where(high == roll_max)
    pl = low.where(low == roll_min)
    # confirmed `right` bars after the pivot bar; combine highs & lows into one level stream
    conf = ph.shift(right).combine_first(pl.shift(right))
    pivot_px = conf.to_numpy(np.float64)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(np.float64)
    if _legacy(break_bars, anchor):
        sup, res, sup_t, res_t = _sr_scan(pivot_px, close, float(cluster_pct) / 100.0,
                                          int(max_zones))
    else:
        *_, sup, res, sup_t, res_t = _sr_core(pivot_px, close, float(cluster_pct) / 100.0,
                                              int(max_zones), int(break_bars), bool(anchor))
    out[f"{prefix}_support"] = sup
    out[f"{prefix}_resistance"] = res
    out[f"{prefix}_support_touches"] = sup_t
    out[f"{prefix}_resistance_touches"] = res_t
    return out


def _confirmed_pivots(df: pd.DataFrame, left: int, right: int) -> pd.Series:
    """Prices of swing pivots at the bar they become CONFIRMED (NaN elsewhere)."""
    w = int(left) + int(right) + 1
    high, low = df["high"], df["low"]
    ph = high.where(high == high.rolling(w, center=True, min_periods=w).max())
    pl = low.where(low == low.rolling(w, center=True, min_periods=w).min())
    return ph.shift(right).combine_first(pl.shift(right))


def sr_zone_table(df: pd.DataFrame, left: int = 10, right: int = 10, *, cluster_pct: float = 0.15,
                  max_zones: int = 128, time_col: str = "t", break_bars: int = 0,
                  anchor: bool = False) -> pd.DataFrame:
    """The causal S/R zone *timeline* — for plotting zones exactly as they evolved.

    Mirrors `add_sr_zones`' clustering, but instead of the nearest level per bar it returns one row
    per (zone, price-revision): the bar range over which that zone sat at that price, plus its touch
    count at the time. A zone appears only from the bar its first pivot was confirmed (no lookahead)
    and its level nudges as later pivots merge into it — so plotting these segments shows what was
    actually known at each point, not a flattering end-state snapshot.

    Columns: zone, start_i, end_i, start_t, end_t, price, touches
    Only loops over confirmed pivots (sparse), so it is cheap even on millions of bars.
    """
    n = len(df)
    conf = _confirmed_pivots(df, left, right).to_numpy(np.float64)
    times = df[time_col].reset_index(drop=True) if time_col in df.columns else None
    frac = float(cluster_pct) / 100.0

    if not _legacy(break_bars, anchor):
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
        zo, st, en, px, tc, lo, hi, *_ = _sr_core(conf, close, frac, int(max_zones),
                                                  int(break_bars), bool(anchor))
        out = pd.DataFrame({"zone": zo, "start_i": st, "end_i": en, "price": px,
                            "touches": tc, "lo": lo, "hi": hi})
        out = out.sort_values(["start_i", "zone"], kind="stable").reset_index(drop=True)
        if not out.empty and times is not None:
            out["start_t"] = times.iloc[out["start_i"].to_numpy()].to_numpy()
            out["end_t"] = times.iloc[out["end_i"].to_numpy()].to_numpy()
        return out

    zones: list[dict] = []      # {price, touches, seg_start}
    rows: list[dict] = []

    def _close(z: dict, zid: int, end_i: int):
        # a segment is [seg_start, end_i]; skip empty ones (a zone revised on the same bar it began)
        if end_i >= z["seg_start"]:
            rows.append({"zone": zid, "start_i": z["seg_start"], "end_i": end_i,
                         "price": z["price"], "touches": z["touches"]})

    for i in range(n):
        p = conf[i]
        if not (p == p) or p <= 0.0:
            continue
        best, best_d = -1, np.inf
        for zi, z in enumerate(zones):
            d = abs(z["price"] - p)
            if d <= frac * p and d < best_d:
                best, best_d = zi, d
        if best >= 0:
            z = zones[best]
            _close(z, best, i - 1)                    # old level held only THROUGH i-1
            z["touches"] += 1
            z["price"] += (p - z["price"]) / z["touches"]
            z["seg_start"] = i                        # new level takes over AT bar i
        elif len(zones) < max_zones:
            zones.append({"price": p, "touches": 1, "seg_start": i})
        else:                                        # evict the weakest zone
            wi = min(range(len(zones)), key=lambda k: (zones[k]["touches"], -zones[k]["seg_start"]))
            _close(zones[wi], wi, i - 1)
            zones[wi] = {"price": p, "touches": 1, "seg_start": i}

    for zi, z in enumerate(zones):
        _close(z, zi, n - 1)

    out = pd.DataFrame(rows, columns=["zone", "start_i", "end_i", "price", "touches"])
    if not out.empty and times is not None:
        out["start_t"] = times.iloc[out["start_i"].to_numpy()].to_numpy()
        out["end_t"] = times.iloc[out["end_i"].to_numpy()].to_numpy()
    return out


def add_pivot_points(df: pd.DataFrame, rule: str = "1D", *, prefix: str = "piv") -> pd.DataFrame:
    """Classic floor-trader pivot points from the PRIOR `rule` period (default daily).

    Adds {prefix}_pp/_r1/_s1/_r2/_s2, aligned onto the base grid and shifted one period so
    only the previous period's levels are visible (lookahead-safe).
    """
    out = df.copy()
    agg = resample_ohlcv(out, rule)
    pp = (agg["high"] + agg["low"] + agg["close"]) / 3.0
    r1 = 2 * pp - agg["low"]
    s1 = 2 * pp - agg["high"]
    r2 = pp + (agg["high"] - agg["low"])
    s2 = pp - (agg["high"] - agg["low"])
    lv = pd.DataFrame({
        "open_time": agg["open_time"],
        f"{prefix}_pp": pp, f"{prefix}_r1": r1, f"{prefix}_s1": s1,
        f"{prefix}_r2": r2, f"{prefix}_s2": s2,
    })
    # shift by one period so the CURRENT period sees only the PRIOR period's levels
    for c in [c for c in lv.columns if c != "open_time"]:
        lv[c] = lv[c].shift(1)
    merged = pd.merge_asof(out.sort_values("open_time"), lv.sort_values("open_time"),
                           on="open_time", direction="backward")
    return merged


def add_zigzag_structure(df: pd.DataFrame, *, atr_mult: float = 1.5, atr_period: int = 14,
                         pct: Optional[float] = None, prefix: str = "zz") -> pd.DataFrame:
    """**Adaptive** market structure via an ATR-scaled ZigZag (directional-change swings).

    Why this rather than a fixed `left/right` fractal: a fractal pivot needs a FIXED number of bars
    on both sides, so its lag is constant regardless of what price is doing, and its sensitivity is
    wrong whenever volatility changes. A ZigZag instead confirms a swing the moment price retraces
    from the running extreme by a **volatility-scaled threshold** (`atr_mult` x ATR, or a fixed
    `pct`). Consequences:

      * self-scaling — the same setting works across regimes and instruments;
      * variable, usually SHORTER lag in a trend (a sharp reversal confirms immediately) and longer
        in chop (which is correct: chop should not manufacture structure);
      * no arbitrary lookback to tune — one threshold with a physical meaning.

    This is the standard "directional change" event definition from the microstructure literature
    and what most adaptive ZigZag/structure indicators implement.

    Causal: a pivot is emitted at the bar its retracement threshold is met, never at the extreme
    itself. `{prefix}_lag` records how many bars back the extreme actually was.

    Adds (all at the CONFIRMATION bar; NaN/None elsewhere):
      {prefix}_label 'HH'|'HL'|'LH'|'LL'   {prefix}_price  the pivot price
      {prefix}_pivot_i  bar index of the extreme     {prefix}_lag  confirm_bar - extreme_bar
      {prefix}_trend  1 up / -1 down / 0 sideways, forward-filled
    """
    out = df.copy()
    n = len(out)
    high = out["high"].to_numpy(np.float64)
    low = out["low"].to_numpy(np.float64)
    if pct is not None:
        thr = out["close"].to_numpy(np.float64) * (float(pct) / 100.0)
    else:
        from .volatility import atr as _atr
        thr = _atr(out, atr_period).to_numpy(np.float64) * float(atr_mult)

    label = np.full(n, None, dtype=object)
    price = np.full(n, np.nan)
    piv_i = np.full(n, -1, np.int64)
    lag = np.full(n, np.nan)
    trend = np.zeros(n, np.int8)

    direction = 0            # +1 = seeking a high, -1 = seeking a low
    ext_px, ext_i = np.nan, -1
    prev_high = prev_low = np.nan
    hi_state = lo_state = ""
    cur = 0

    for i in range(n):
        t = thr[i]
        if not (t == t) or t <= 0:
            trend[i] = cur
            continue
        if direction == 0:                      # bootstrap
            if ext_i < 0:
                ext_px, ext_i, direction = high[i], i, 1
            trend[i] = cur
            continue

        if direction == 1:                       # tracking a swing HIGH
            if high[i] >= ext_px:
                ext_px, ext_i = high[i], i
            elif ext_px - low[i] >= t:           # retraced enough -> the high is confirmed
                if prev_high == prev_high:
                    hi_state = "HH" if ext_px > prev_high else "LH"
                    label[i] = hi_state
                    price[i] = ext_px
                    piv_i[i] = ext_i
                    lag[i] = i - ext_i
                prev_high = ext_px
                direction, ext_px, ext_i = -1, low[i], i
        else:                                    # tracking a swing LOW
            if low[i] <= ext_px:
                ext_px, ext_i = low[i], i
            elif high[i] - ext_px >= t:
                if prev_low == prev_low:
                    lo_state = "HL" if ext_px > prev_low else "LL"
                    label[i] = lo_state
                    price[i] = ext_px
                    piv_i[i] = ext_i
                    lag[i] = i - ext_i
                prev_low = ext_px
                direction, ext_px, ext_i = 1, high[i], i

        if hi_state == "HH" and lo_state == "HL":
            cur = 1
        elif hi_state == "LH" and lo_state == "LL":
            cur = -1
        else:
            cur = 0
        trend[i] = cur

    out[f"{prefix}_label"] = label
    out[f"{prefix}_price"] = price
    out[f"{prefix}_pivot_i"] = piv_i
    out[f"{prefix}_lag"] = lag
    out[f"{prefix}_trend"] = trend
    # Running last CONFIRMED swing high / low, forward-filled from the confirmation bar onward.
    # These are the levels a break-of-structure entry trades against; carrying them per bar keeps
    # that rule a plain column comparison instead of a lookup loop.
    lab = pd.Series(label, index=out.index)
    px = pd.Series(price, index=out.index, dtype="float64")
    out[f"{prefix}_last_high"] = px.where(lab.isin(("HH", "LH"))).ffill().to_numpy()
    out[f"{prefix}_last_low"] = px.where(lab.isin(("HL", "LL"))).ffill().to_numpy()
    return out
