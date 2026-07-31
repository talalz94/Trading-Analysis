"""Vectorized oscillators: RSI (Wilder), MACD, Stochastic. Compute only (no plotting)."""
from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (0-100). Uses Wilder smoothing (ewm alpha=1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def add_rsi(df: pd.DataFrame, period: int = 14, *, source: str = "close",
            prefix: str = "rsi") -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_{int(period)}"] = rsi(out[source], period)
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
             *, source: str = "close", prefix: str = "macd") -> pd.DataFrame:
    """Adds {prefix}, {prefix}_signal, {prefix}_hist."""
    out = df.copy()
    src = out[source]
    ema_fast = src.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = src.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    out[prefix] = macd_line
    out[f"{prefix}_signal"] = sig
    out[f"{prefix}_hist"] = macd_line - sig
    return out


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3,
                   *, prefix: str = "stoch") -> pd.DataFrame:
    """Adds {prefix}_k, {prefix}_d (fast %K smoothed by `smooth`, %D = SMA of %K)."""
    out = df.copy()
    ll = out["low"].rolling(k, min_periods=k).min()
    hh = out["high"].rolling(k, min_periods=k).max()
    raw_k = 100.0 * (out["close"] - ll) / (hh - ll)
    k_s = raw_k.rolling(smooth, min_periods=smooth).mean()
    out[f"{prefix}_k"] = k_s
    out[f"{prefix}_d"] = k_s.rolling(d, min_periods=d).mean()
    return out


def add_rsi_divergence(df: pd.DataFrame, period: int = 14, left: int = 5, right: int = 5, *,
                       max_gap: int = 120, prefix: str = "rsi") -> pd.DataFrame:
    """Classic RSI divergence between consecutive CONFIRMED price pivots.

    bearish : price makes a HIGHER high while RSI makes a LOWER high  (upside momentum fading)
    bullish : price makes a LOWER low   while RSI makes a HIGHER low  (downside momentum fading)

    Compared pivots must be at most `max_gap` bars apart, so far-apart peaks aren't linked.
    Anti-lookahead: a pivot only exists `right` bars after it forms, so a divergence is flagged at
    the later pivot's CONFIRMATION bar — the same bar you could have acted on it.

    Adds `{prefix}_div_bear` / `{prefix}_div_bull` (bool). Use `rsi_divergence_table` to draw the
    connecting lines.
    """
    import numpy as np
    out = df.copy() if f"{prefix}_{int(period)}" in df.columns else add_rsi(df, period, prefix=prefix)
    r = out[f"{prefix}_{int(period)}"].to_numpy(dtype=float)
    n = len(out)
    w = int(left) + int(right) + 1
    high, low = out["high"], out["low"]
    ph = high.where(high == high.rolling(w, center=True, min_periods=w).max()).shift(right)
    pl = low.where(low == low.rolling(w, center=True, min_periods=w).min()).shift(right)
    ph_v, pl_v = ph.to_numpy(dtype=float), pl.to_numpy(dtype=float)

    bear = np.zeros(n, bool)
    bull = np.zeros(n, bool)
    rows = []
    last_h = last_l = None                      # (confirm_bar, pivot_bar, price, rsi)
    for i in range(n):
        if ph_v[i] == ph_v[i]:
            pb = i - int(right)
            cur = (i, pb, ph_v[i], r[pb])
            if last_h is not None and (pb - last_h[1]) <= max_gap:
                if cur[2] > last_h[2] and cur[3] == cur[3] and last_h[3] == last_h[3] \
                        and cur[3] < last_h[3]:
                    bear[i] = True
                    rows.append({"kind": "bear", "i0": last_h[1], "i1": pb,
                                 "px0": last_h[2], "px1": cur[2],
                                 "rsi0": last_h[3], "rsi1": cur[3], "confirm_i": i})
            last_h = cur
        if pl_v[i] == pl_v[i]:
            pb = i - int(right)
            cur = (i, pb, pl_v[i], r[pb])
            if last_l is not None and (pb - last_l[1]) <= max_gap:
                if cur[2] < last_l[2] and cur[3] == cur[3] and last_l[3] == last_l[3] \
                        and cur[3] > last_l[3]:
                    bull[i] = True
                    rows.append({"kind": "bull", "i0": last_l[1], "i1": pb,
                                 "px0": last_l[2], "px1": cur[2],
                                 "rsi0": last_l[3], "rsi1": cur[3], "confirm_i": i})
            last_l = cur
    out[f"{prefix}_div_bear"] = bear
    out[f"{prefix}_div_bull"] = bull
    out.attrs["rsi_div_rows"] = rows
    return out


def rsi_divergence_table(df: pd.DataFrame, period: int = 14, left: int = 5, right: int = 5, *,
                         max_gap: int = 120, prefix: str = "rsi") -> "pd.DataFrame":
    """The divergence segments as a table (kind, i0, i1, px0, px1, rsi0, rsi1) — for plotting the
    lines on both the price pane and the RSI pane."""
    out = add_rsi_divergence(df, period, left, right, max_gap=max_gap, prefix=prefix)
    return pd.DataFrame(out.attrs.get("rsi_div_rows", []),
                        columns=["kind", "i0", "i1", "px0", "px1", "rsi0", "rsi1", "confirm_i"])


def add_rsi_divergence_realtime(df: pd.DataFrame, period: int = 14, left: int = 10, *,
                                max_gap: int = 120, prefix: str = "rsi") -> pd.DataFrame:
    """**Zero-lag (provisional) RSI divergence.**

    A confirmed pivot cannot exist until `right` bars have passed — that lag is definitional, not an
    implementation flaw. This variant removes it by firing on the RUNNING extreme instead:

      bearish : this bar's high is the highest of the last `left` bars AND exceeds the last confirmed
                swing high, while RSI is BELOW its value at that swing high.
      bullish : mirror image on lows.

    Nothing here looks forward, so a signal appears on the very bar it becomes true — but it is
    *provisional*: price may keep running and never leave a pivot there. Treat it as an early warning
    and use `add_rsi_divergence` (confirmed) for anything that must not be revised.

    Adds `{prefix}_rtdiv_bear` / `{prefix}_rtdiv_bull`.
    """
    import numpy as np
    out = df.copy() if f"{prefix}_{int(period)}" in df.columns else add_rsi(df, period, prefix=prefix)
    r = out[f"{prefix}_{int(period)}"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    n = len(out)
    L = int(left)
    roll_hi = out["high"].rolling(L, min_periods=L).max().to_numpy(dtype=float)
    roll_lo = out["low"].rolling(L, min_periods=L).min().to_numpy(dtype=float)

    bear = np.zeros(n, bool)
    bull = np.zeros(n, bool)
    # reference = the last CONFIRMED swing (needs `left` bars each side, but it is in the past
    # already, so waiting for it costs nothing at signal time)
    ref_h_px = ref_h_rsi = ref_h_i = np.nan
    ref_l_px = ref_l_rsi = ref_l_i = np.nan
    w = 2 * L + 1
    ph = out["high"].where(out["high"] == out["high"].rolling(w, center=True, min_periods=w).max())
    pl = out["low"].where(out["low"] == out["low"].rolling(w, center=True, min_periods=w).min())
    ph_v = ph.shift(L).to_numpy(dtype=float)
    pl_v = pl.shift(L).to_numpy(dtype=float)

    for i in range(n):
        if ph_v[i] == ph_v[i]:
            ref_h_px, ref_h_rsi, ref_h_i = ph_v[i], r[i - L], i - L
        if pl_v[i] == pl_v[i]:
            ref_l_px, ref_l_rsi, ref_l_i = pl_v[i], r[i - L], i - L
        if r[i] != r[i]:
            continue
        if ref_h_px == ref_h_px and high[i] >= roll_hi[i] and (i - ref_h_i) <= max_gap:
            if high[i] > ref_h_px and r[i] < ref_h_rsi:
                bear[i] = True
        if ref_l_px == ref_l_px and low[i] <= roll_lo[i] and (i - ref_l_i) <= max_gap:
            if low[i] < ref_l_px and r[i] > ref_l_rsi:
                bull[i] = True
    out[f"{prefix}_rtdiv_bear"] = bear
    out[f"{prefix}_rtdiv_bull"] = bull
    return out
