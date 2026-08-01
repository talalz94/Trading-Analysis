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


def add_sr_zones(df: pd.DataFrame, left: int = 10, right: int = 10, *, cluster_pct: float = 0.15,
                 max_zones: int = 128, prefix: str = "sr") -> pd.DataFrame:
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
    sup, res, sup_t, res_t = _sr_scan(pivot_px, close, float(cluster_pct) / 100.0, int(max_zones))
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
                  max_zones: int = 128, time_col: str = "t") -> pd.DataFrame:
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
