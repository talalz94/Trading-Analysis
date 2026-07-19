import time
import re
import requests
import pandas as pd
from typing import Optional, Literal, List

Market = Literal["spot", "usdm_futures", "coinm_futures"]

VALID_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

BASES = {
    "spot": ("https://data-api.binance.vision", "/api/v3/klines", 1000),
    "usdm_futures": ("https://fapi.binance.com", "/fapi/v1/klines", 1500),
    "coinm_futures": ("https://dapi.binance.com", "/dapi/v1/klines", 1500),
}

def _to_ms(dt: str) -> int:
    ts = pd.to_datetime(dt, utc=True)
    return int(ts.value // 1_000_000)  # ns -> ms

def fetch_binance_klines(
    symbol: str,
    interval: str,
    start: str,
    end: Optional[str] = None,
    market: Market = "spot",
    pause_s: float = 0.15,
    verbose: bool = False,
) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Use one of: {sorted(VALID_INTERVALS)}")

    base, path, limit = BASES[market]
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else None

    rows = []
    req = 0

    if verbose:
        print(f"[START] {market=} {symbol=} {interval=} {start=} {end=} {limit=}")

    while True:
        req += 1
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms

        r = requests.get(base + path, params=params, timeout=30)
        if r.status_code != 200:
            if verbose:
                print(f" -> ERROR {r.status_code}: {r.text[:200]}")
            r.raise_for_status()

        data = r.json()
        if not data:
            break

        rows.extend(data)

        last_open = data[-1][0]
        returned = len(data)
        start_ms = last_open + 1

        if returned < limit:
            break
        if end_ms is not None and start_ms >= end_ms:
            break

        time.sleep(pause_s)

    cols = [
        "open_time_ms","open","high","low","close","volume",
        "close_time_ms","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote","num_trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    return df[[
        "open_time","open","high","low","close","volume",
        "quote_volume","num_trades","taker_buy_base","taker_buy_quote"
    ]]

def parse_show_last(show_last: str) -> pd.Timedelta:
    s = show_last.strip()
    if re.search(r"week(s)?$", s):
        n = int("".join(ch for ch in s if ch.isdigit()) or "1")
        return pd.Timedelta(days=7*n)
    if re.search(r"month(s)?$", s) or s.endswith("M"):
        n = int("".join(ch for ch in s if ch.isdigit()) or "1")
        return pd.Timedelta(days=30*n)
    if s.endswith("w"):
        n = int(s[:-1])
        return pd.Timedelta(days=7*n)
    return pd.Timedelta(s)