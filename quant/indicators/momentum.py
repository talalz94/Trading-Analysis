"""
Squeeze Momentum Indicator [LazyBear] — faithful port of the TradingView Pine study "SQZMOM_LB".

Two parts:
  * momentum histogram = linear-regression of (close - midline) over `kc_length` bars, coloured by
    sign + rising/falling (4 states);
  * a squeeze state (on / off / none) from whether the Bollinger Bands sit INSIDE the Keltner
    Channel, drawn as coloured dots on the zero line.

Matches LazyBear's original exactly, including its documented quirk: the BB deviation uses the KC
multiplier (`kc_mult`), so the `bb_mult` input is present-but-unused — the same as the TradingView
indicator. (Set `faithful=False` to use `bb_mult` for the BB instead.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .overlays import ema as _ema
from .volatility import atr as _atr


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=int(n)).mean()


def add_ema_gradient(df: pd.DataFrame, period: int = 200, smooth: int = 9, lookback: int = 5,
                     atr_period: int = 14, source: str = "close", *,
                     prefix: str = "grad") -> pd.DataFrame:
    """Slope ('gradient') of a smoothed EMA — the oscillator shown in the scenario screenshots.

    Line = EMA(source, period) optionally smoothed by EMA(smooth). The gradient is the per-bar slope
    of that line over `lookback` bars, normalised by ATR so it's a bounded, regime-independent number
    (≈ ±1): >0 = line rising (up-trend), <0 = falling. Use a small threshold (e.g. ±0.1) to call it
    flat. Adds:
      {prefix}_line   the (smoothed) EMA line itself — the red dotted trend line on price
      {prefix}        the normalised gradient (bottom pane)
    """
    out = df.copy()
    line = _ema(out[source], period)
    if smooth:
        line = _ema(line, smooth)
    slope = line.diff(int(lookback)) / float(lookback)
    a = _atr(out, atr_period).replace(0.0, np.nan)
    out[f"{prefix}_line"] = line
    out[f"{prefix}"] = slope / a
    return out


def _stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=int(n)).std(ddof=0)   # Pine stdev = population (ddof=0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = (h.iloc[0] - l.iloc[0])                        # first bar: no prior close
    return tr


def _linreg_endpoint(y: np.ndarray, n: int) -> np.ndarray:
    """Vectorized Pine `linreg(y, n, 0)`: value of the least-squares line at the current bar,
    fit over the last `n` points (t = 0 oldest .. n-1 newest). NaN until the window is full."""
    y = np.asarray(y, dtype=float)
    L = len(y)
    out = np.full(L, np.nan)
    if L < n:
        return out
    pos = np.arange(L, dtype=float)
    valid = ~np.isnan(y)
    yf = np.where(valid, y, 0.0)

    def rsum(a):
        c = np.cumsum(np.insert(a, 0, 0.0))
        r = np.full(L, np.nan)
        r[n - 1:] = c[n:] - c[:-n]
        return r

    Sy = rsum(yf)
    Sjy = rsum(pos * yf)
    Svalid = rsum(valid.astype(float))
    start = pos - (n - 1)                       # window start index at each bar
    sum_ty = Sjy - start * Sy                   # Σ k*y over the window (k = 0..n-1)
    sum_t = n * (n - 1) / 2.0
    sum_t2 = (n - 1) * n * (2 * n - 1) / 6.0
    denom = n * sum_t2 - sum_t ** 2
    slope = (n * sum_ty - sum_t * Sy) / denom
    val = Sy / n + slope * (n - 1) / 2.0        # intercept + slope*(n-1)
    return np.where(Svalid == n, val, np.nan)


def add_squeeze_momentum(df: pd.DataFrame, bb_length: int = 20, bb_mult: float = 2.0,
                         kc_length: int = 20, kc_mult: float = 1.5, use_tr: bool = True,
                         *, faithful: bool = True, prefix: str = "sqz") -> pd.DataFrame:
    """Add Squeeze Momentum columns (LazyBear SQZMOM_LB). Inputs mirror the TradingView dialog:
    BB Length/MultFactor, KC Length/MultFactor, Use TrueRange (KC).

    Adds:
      {prefix}_mom          momentum histogram value (linreg)
      {prefix}_on           squeeze ON  (BB inside KC — volatility compressed)
      {prefix}_off          squeeze OFF (BB outside KC — released)
      {prefix}_no           no squeeze
      {prefix}_state        'up_rising' | 'up_falling' | 'down_falling' | 'down_rising' (for colour)
    """
    out = df.copy()
    src = out["close"]

    # Bollinger Bands. LazyBear's original uses kc_mult for the deviation (bb_mult is unused) —
    # replicate that when faithful=True so the squeeze matches TradingView exactly.
    basis = _sma(src, bb_length)
    dev = (kc_mult if faithful else bb_mult) * _stdev(src, bb_length)
    upper_bb, lower_bb = basis + dev, basis - dev

    # Keltner Channel
    ma = _sma(src, kc_length)
    rng = _true_range(out) if use_tr else (out["high"] - out["low"])
    rangema = _sma(rng, kc_length)
    upper_kc, lower_kc = ma + rangema * kc_mult, ma - rangema * kc_mult

    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = ~sqz_on & ~sqz_off

    # Momentum: linreg of (close - midline) over kc_length
    hh = out["high"].rolling(int(kc_length), min_periods=int(kc_length)).max()
    ll = out["low"].rolling(int(kc_length), min_periods=int(kc_length)).min()
    mid = ((hh + ll) / 2.0 + _sma(src, kc_length)) / 2.0
    val = _linreg_endpoint((src - mid).to_numpy(), int(kc_length))

    prev = np.r_[np.nan, val[:-1]]
    up = val > 0
    rising = val > prev
    state = np.where(up, np.where(rising, "up_rising", "up_falling"),
                     np.where(~rising, "down_falling", "down_rising"))
    state = np.where(np.isnan(val), None, state)

    out[f"{prefix}_mom"] = val
    out[f"{prefix}_on"] = sqz_on.to_numpy()
    out[f"{prefix}_off"] = sqz_off.to_numpy()
    out[f"{prefix}_no"] = no_sqz.to_numpy()
    out[f"{prefix}_state"] = state
    return out


def add_wavetrend(df: pd.DataFrame, channel_length: int = 10, average_length: int = 21,
                  ob1: float = 60, ob2: float = 53, os1: float = -60, os2: float = -53,
                  *, prefix: str = "wt") -> pd.DataFrame:
    """WaveTrend Oscillator [LazyBear] (the WT_CROSS_LB study). Inputs mirror the TradingView
    dialog: Channel Length, Average Length, and the Over Bought/Sold 1 & 2 levels.

    Adds:
      {prefix}1        WT1 = ema(ci, average_length)   (the fast wave, green line)
      {prefix}2        WT2 = sma(WT1, 4)               (signal, red line)
      {prefix}_diff    WT1 - WT2                       (blue momentum area)
      {prefix}_cross_up / {prefix}_cross_down          (WT1 crosses WT2 — the dots)
    The OB/OS levels are returned via `.attrs` so the chart can draw the guide lines.
    """
    out = df.copy()
    ap = (out["high"] + out["low"] + out["close"]) / 3.0        # hlc3
    esa = _ema(ap, channel_length)
    d = _ema((ap - esa).abs(), channel_length)
    ci = (ap - esa) / (0.015 * d.replace(0.0, np.nan))          # avoid div-by-zero
    tci = _ema(ci, average_length)
    wt1 = tci
    wt2 = _sma(wt1, 4)

    out[f"{prefix}1"] = wt1
    out[f"{prefix}2"] = wt2
    out[f"{prefix}_diff"] = wt1 - wt2
    diff = wt1 - wt2
    out[f"{prefix}_cross_up"] = ((diff > 0) & (diff.shift(1) <= 0)).to_numpy()
    out[f"{prefix}_cross_down"] = ((diff < 0) & (diff.shift(1) >= 0)).to_numpy()
    out.attrs["wt_levels"] = dict(ob1=ob1, ob2=ob2, os1=os1, os2=os2)
    return out
