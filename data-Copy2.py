import os
import time
import re
import logging
from pathlib import Path
import requests
import pandas as pd
from typing import Optional, Literal, List, Tuple, Union

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


def _make_logger(name: str = "charting.binance_cache") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def _to_ms(dt: Union[str, pd.Timestamp]) -> int:
    ts = pd.to_datetime(dt, utc=True)
    return int(ts.value // 1_000_000)


def interval_to_timedelta(interval: str) -> Optional[pd.Timedelta]:
    # returns None for 1M (variable)
    if interval.endswith("m"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return pd.Timedelta(days=int(interval[:-1]))
    if interval.endswith("w"):
        return pd.Timedelta(weeks=int(interval[:-1]))
    if interval == "1M":
        return None
    return None


def _resolve_cache_dir(cache_dir: Union[str, Path, None]) -> Path:
    if cache_dir is None:
        cache_dir = Path.cwd() / "data"
    else:
        cache_dir = Path(cache_dir)

    # If user provided relative path, resolve from current working directory
    if not cache_dir.is_absolute():
        cache_dir = (Path.cwd() / cache_dir).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(cache_dir: Path, market: str, symbol: str, interval: str) -> Path:
    fname = f"binance_{market}_{symbol.upper()}_{interval}.parquet"
    return cache_dir / fname


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    # Prefer parquet; fallback to pickle
    try:
        df = pd.read_parquet(path)
    except Exception:
        pkl = path.with_suffix(".pkl")
        if pkl.exists():
            try:
                df = pd.read_pickle(pkl)
            except Exception:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    return df


def _write_cache(df: pd.DataFrame, path: Path, logger: logging.Logger) -> str:
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)

    # Try parquet first (fast). If missing engine, fallback to pickle.
    try:
        df.to_parquet(path, index=False)
        # cleanup old pkl
        pkl = path.with_suffix(".pkl")
        if pkl.exists():
            pkl.unlink()
        return "parquet"
    except Exception as e:
        logger.warning(f"Parquet write failed ({type(e).__name__}: {e}). Falling back to pickle.")
        pkl = path.with_suffix(".pkl")
        df.to_pickle(pkl)
        return "pickle"


def _fetch_binance_klines_uncached(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int],
    market: Market,
    pause_s: float,
    logger: logging.Logger,
    debug_pages: bool,
) -> pd.DataFrame:
    base, path, limit = BASES[market]

    rows = []
    req = 0

    while True:
        req += 1
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms

        r = requests.get(base + path, params=params, timeout=30)
        if r.status_code != 200:
            logger.error(f"HTTP {r.status_code} | {r.text[:200]}")
            r.raise_for_status()

        data = r.json()
        if not data:
            break

        rows.extend(data)

        last_open = data[-1][0]
        returned = len(data)
        start_ms = last_open + 1

        if debug_pages:
            first_open = pd.to_datetime(data[0][0], unit="ms", utc=True)
            last_open_dt = pd.to_datetime(last_open, unit="ms", utc=True)
            logger.debug(f"Page {req:03d}: {returned} rows ({first_open} -> {last_open_dt})")

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


def _find_gaps(df: pd.DataFrame, delta: pd.Timedelta) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []
    ot = df["open_time"].sort_values()
    diffs = ot.diff()
    gap_idx = diffs[diffs > (delta * 1.5)].index
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for idx in gap_idx:
        i = ot.index.get_loc(idx)
        prev_t = ot.iloc[i - 1]
        next_t = ot.iloc[i]
        s = prev_t + delta
        e = next_t - delta
        if s <= e:
            gaps.append((s, e))
    return gaps


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start: str,
    end: Optional[str] = None,
    market: Market = "spot",
    pause_s: float = 0.15,
    cache: bool = True,
    cache_dir: Union[str, Path, None] = None,
    max_staleness_intervals: int = 2,
    log_level: str = "INFO",
    debug_pages: bool = False,
) -> pd.DataFrame:
    """
    Cached Binance klines fetch.

    - Saves to <cache_dir>/binance_<market>_<symbol>_<interval>.parquet (or .pkl fallback).
    - Fetches only missing ranges based on requested [start, end] and cache coverage.
    - Always merges, de-dups, sorts, and saves cache after fetching.
    """
    logger = _make_logger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    symbol = symbol.upper().strip()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Use one of: {sorted(VALID_INTERVALS)}")

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True) if end else None
    delta = interval_to_timedelta(interval)

    step = 0
    def log(msg: str, level: int = logging.INFO):
        nonlocal step
        step += 1
        logger.log(level, f"{step:02d}. {msg}")

    if not cache:
        log("Cache disabled; fetching directly from API.", logging.INFO)
        return _fetch_binance_klines_uncached(
            symbol=symbol,
            interval=interval,
            start_ms=_to_ms(start_ts),
            end_ms=_to_ms(end_ts) if end_ts is not None else None,
            market=market,
            pause_s=pause_s,
            logger=logger,
            debug_pages=debug_pages,
        )

    cache_dir_p = _resolve_cache_dir(cache_dir)
    path = _cache_path(cache_dir_p, market, symbol, interval)

    log(f"Parameters: market={market}, symbol={symbol}, interval={interval}, start={start_ts}, end={end_ts or 'LATEST'}")
    log(f"Cache directory: {cache_dir_p} (cwd={Path.cwd()})")
    log(f"Cache file: {path}")

    df_cache = _read_cache(path)
    if df_cache.empty:
        log("Cache: MISS (no local data found).", logging.INFO)
    else:
        cmin = df_cache["open_time"].min()
        cmax = df_cache["open_time"].max()
        log(f"Cache: HIT | rows={len(df_cache):,} | range={cmin} -> {cmax}")

    fetch_ranges: List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]] = []

    if df_cache.empty:
        fetch_ranges.append((start_ts, end_ts))
    else:
        cmin = df_cache["open_time"].min()
        cmax = df_cache["open_time"].max()

        # Missing before requested start
        if start_ts < cmin:
            end_before = cmin - (delta or pd.Timedelta(milliseconds=1))
            fetch_ranges.append((start_ts, end_before))

        # Missing after (if end specified)
        if end_ts is not None:
            if cmax < end_ts:
                fetch_ranges.append((cmax + (delta or pd.Timedelta(milliseconds=1)), end_ts))
        else:
            # end = latest; fetch if stale
            if delta is not None:
                now = pd.Timestamp.utcnow()
                if (now - cmax) > (delta * max_staleness_intervals):
                    fetch_ranges.append((cmax + delta, None))
            else:
                fetch_ranges.append((cmax, None))

        # Fill internal gaps
        if delta is not None:
            gaps = _find_gaps(df_cache, delta)
            if gaps:
                log(f"Detected {len(gaps)} internal gap(s). Will attempt to backfill.", logging.INFO)
            for gs, ge in gaps:
                # Only fetch gaps that overlap requested window
                if ge < start_ts:
                    continue
                if end_ts is not None and gs > end_ts:
                    continue
                fetch_ranges.append((max(gs, start_ts), min(ge, end_ts) if end_ts is not None else ge))

    # Normalize/compact ranges (optional): remove invalid ranges
    cleaned: List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]] = []
    for s, e in fetch_ranges:
        if e is not None and e < s:
            continue
        cleaned.append((pd.to_datetime(s, utc=True), pd.to_datetime(e, utc=True) if e is not None else None))
    fetch_ranges = cleaned

    if not fetch_ranges:
        log("No API fetch needed; cache already covers requested range.", logging.INFO)
    else:
        for i, (s, e) in enumerate(fetch_ranges, start=1):
            log(f"Fetch plan {i}/{len(fetch_ranges)}: {s} -> {e if e is not None else 'LATEST'}", logging.INFO)

    new_parts = []
    for i, (s, e) in enumerate(fetch_ranges, start=1):
        part = _fetch_binance_klines_uncached(
            symbol=symbol,
            interval=interval,
            start_ms=_to_ms(s),
            end_ms=_to_ms(e) if e is not None else None,
            market=market,
            pause_s=pause_s,
            logger=logger,
            debug_pages=debug_pages,
        )
        if not part.empty:
            log(f"Fetched {len(part):,} rows for range {i}.", logging.INFO)
            new_parts.append(part)
        else:
            log(f"Fetched 0 rows for range {i}.", logging.INFO)

    # Merge + persist
    if new_parts:
        df_all = pd.concat([df_cache] + new_parts, ignore_index=True) if not df_cache.empty else pd.concat(new_parts, ignore_index=True)
        df_all = df_all.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    else:
        df_all = df_cache.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)

    if df_all.empty:
        log("Final dataset is empty (no data fetched).", logging.WARNING)
        return df_all

    fmt = _write_cache(df_all, path, logger)
    log(f"Cache saved ({fmt}) | rows={len(df_all):,} | range={df_all['open_time'].min()} -> {df_all['open_time'].max()}", logging.INFO)

    # Return requested slice
    if end_ts is None:
        out = df_all[df_all["open_time"] >= start_ts].reset_index(drop=True)
    else:
        out = df_all[(df_all["open_time"] >= start_ts) & (df_all["open_time"] <= end_ts)].reset_index(drop=True)

    log(f"Returning slice | rows={len(out):,} | range={out['open_time'].min()} -> {out['open_time'].max()}", logging.INFO)
    return out


def cache_info(symbol: str, interval: str, market: Market = "spot", cache_dir: Union[str, Path, None] = None) -> dict:
    cache_dir_p = _resolve_cache_dir(cache_dir)
    path = _cache_path(cache_dir_p, market, symbol.upper(), interval)
    df = _read_cache(path)
    info = {
        "cache_dir": str(cache_dir_p),
        "cache_path": str(path),
        "exists_parquet": path.exists(),
        "exists_pickle": path.with_suffix(".pkl").exists(),
        "rows": int(len(df)) if not df.empty else 0,
        "min_open_time": str(df["open_time"].min()) if not df.empty else None,
        "max_open_time": str(df["open_time"].max()) if not df.empty else None,
    }
    return info


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