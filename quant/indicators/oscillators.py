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
