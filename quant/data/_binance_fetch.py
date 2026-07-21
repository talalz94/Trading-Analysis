"""
Cached Binance klines fetcher (absorbed from the original, proven data.py).

Incremental: loads the main cache + any partial checkpoints, fetches only missing ranges,
saves every successful API page as a durable partial chunk, then consolidates. Resilient to
rate limits / network errors (exponential backoff) and resumes from the last saved candle.
Public entry point: `fetch_binance_klines(...)`. Cache file layout matches quant.data.store:
`data/binance_{market}_{SYMBOL}_{interval}.parquet`.
"""
import os
import time
import re
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
import requests
import pandas as pd
from typing import Optional, Literal, List, Tuple, Union, Callable

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


# -----------------------------------------------------------------------------
# Logging / stats
# -----------------------------------------------------------------------------

def _make_logger(name: str = "charting.binance_cache") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


@dataclass
class FetchStats:
    requests: int = 0          # HTTP attempts, including retries
    api_pages: int = 0         # successful pages
    retries: int = 0
    rows_fetched: int = 0
    partial_saves: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at


# -----------------------------------------------------------------------------
# Time / interval helpers
# -----------------------------------------------------------------------------

def _to_ms(dt: Union[str, pd.Timestamp]) -> int:
    ts = pd.to_datetime(dt, utc=True)
    return int(ts.value // 1_000_000)


def interval_to_timedelta(interval: str) -> Optional[pd.Timedelta]:
    # returns None for 1M because calendar month is variable
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


def _estimate_pages(start_ts: pd.Timestamp, end_ts: Optional[pd.Timestamp], interval: str, limit: int) -> Optional[int]:
    delta = interval_to_timedelta(interval)
    if end_ts is None or delta is None:
        return None
    if end_ts <= start_ts:
        return 1
    bars = int(((end_ts - start_ts) / delta)) + 1
    return max(1, math.ceil(bars / limit))


# -----------------------------------------------------------------------------
# Cache path / read / write
# -----------------------------------------------------------------------------

def _resolve_cache_dir(cache_dir: Union[str, Path, None]) -> Path:
    if cache_dir is None:
        cache_dir = Path.cwd() / "data"
    else:
        cache_dir = Path(cache_dir)

    if not cache_dir.is_absolute():
        cache_dir = (Path.cwd() / cache_dir).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(cache_dir: Path, market: str, symbol: str, interval: str) -> Path:
    return cache_dir / f"binance_{market}_{symbol.upper()}_{interval}.parquet"


def _partial_dir(path: Path) -> Path:
    return path.parent / ".partials" / path.stem


def _read_one_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".pkl":
            df = pd.read_pickle(path)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    return df


def _read_main_cache(path: Path) -> pd.DataFrame:
    # prefer parquet, fallback to pickle
    df = _read_one_file(path)
    if not df.empty:
        return df

    pkl = path.with_suffix(".pkl")
    return _read_one_file(pkl)


def _read_partial_cache(path: Path) -> Tuple[pd.DataFrame, int]:
    pdir = _partial_dir(path)
    if not pdir.exists():
        return pd.DataFrame(), 0

    files = sorted(list(pdir.glob("*.parquet")) + list(pdir.glob("*.pkl")))
    parts = []
    for f in files:
        d = _read_one_file(f)
        if not d.empty:
            parts.append(d)

    if not parts:
        return pd.DataFrame(), len(files)

    df = pd.concat(parts, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    return df, len(files)


def _merge_dedupe_sort(*dfs: pd.DataFrame) -> pd.DataFrame:
    parts = [d for d in dfs if d is not None and not d.empty]
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out = out.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    return out


def _read_cache_with_partials(path: Path) -> Tuple[pd.DataFrame, int, int]:
    main = _read_main_cache(path)
    partials, partial_files = _read_partial_cache(path)
    merged = _merge_dedupe_sort(main, partials)
    return merged, len(main), partial_files


def _atomic_write_parquet_or_pickle(df: pd.DataFrame, path: Path, logger: logging.Logger) -> str:
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Try parquet first
    try:
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

        # remove stale pickle if main parquet succeeded
        pkl = path.with_suffix(".pkl")
        if pkl.exists():
            pkl.unlink()

        return "parquet"
    except Exception as e:
        logger.warning(f"Parquet write failed ({type(e).__name__}: {e}). Falling back to pickle.")

        pkl = path.with_suffix(".pkl")
        tmp = pkl.with_suffix(".tmp.pkl")
        df.to_pickle(tmp)
        os.replace(tmp, pkl)
        return "pickle"


def _write_main_cache(df: pd.DataFrame, path: Path, logger: logging.Logger) -> str:
    return _atomic_write_parquet_or_pickle(df, path, logger)


def _write_partial_page(df_page: pd.DataFrame, path: Path, logger: logging.Logger) -> str:
    if df_page.empty:
        return "none"

    pdir = _partial_dir(path)
    pdir.mkdir(parents=True, exist_ok=True)

    first_ms = _to_ms(df_page["open_time"].min())
    last_ms = _to_ms(df_page["open_time"].max())
    part_path = pdir / f"part_{first_ms}_{last_ms}.parquet"

    return _atomic_write_parquet_or_pickle(df_page, part_path, logger)


def _cleanup_partials(path: Path, logger: logging.Logger) -> None:
    pdir = _partial_dir(path)
    if not pdir.exists():
        return

    files = sorted(list(pdir.glob("*.parquet")) + list(pdir.glob("*.pkl")) + list(pdir.glob("*.tmp.*")))
    for f in files:
        try:
            f.unlink()
        except Exception as e:
            logger.warning(f"Could not remove partial file {f}: {e}")

    try:
        pdir.rmdir()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Binance parsing / HTTP
# -----------------------------------------------------------------------------

def _parse_klines_page(data: list) -> pd.DataFrame:
    cols = [
        "open_time_ms","open","high","low","close","volume",
        "close_time_ms","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]

    df = pd.DataFrame(data, columns=cols)
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


def _retry_sleep_seconds(response: Optional[requests.Response], attempt: int, backoff_base_s: float, backoff_max_s: float) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), backoff_max_s)
            except ValueError:
                pass

    return min(backoff_base_s * (2 ** max(0, attempt - 1)), backoff_max_s)


def _get_klines_page_with_retries(
    url: str,
    params: dict,
    logger: logging.Logger,
    stats: FetchStats,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    timeout_s: int,
) -> list:
    retry_statuses = {418, 429, 500, 502, 503, 504}

    for attempt in range(0, max_retries + 1):
        try:
            stats.requests += 1
            r = requests.get(url, params=params, timeout=timeout_s)

            if r.status_code == 200:
                return r.json()

            if r.status_code in retry_statuses:
                if attempt >= max_retries:
                    logger.error(
                        f"HTTP {r.status_code} after {stats.requests} request attempts. "
                        f"Progress already checkpointed through last successful page."
                    )
                    r.raise_for_status()

                stats.retries += 1
                sleep_s = _retry_sleep_seconds(r, attempt + 1, backoff_base_s, backoff_max_s)
                logger.warning(
                    f"HTTP {r.status_code}; retry {attempt + 1}/{max_retries} in {sleep_s:.1f}s "
                    f"(requests={stats.requests}, pages={stats.api_pages}, rows={stats.rows_fetched:,})."
                )
                time.sleep(sleep_s)
                continue

            logger.error(f"HTTP {r.status_code} | {r.text[:250]}")
            r.raise_for_status()

        except requests.RequestException as e:
            if attempt >= max_retries:
                logger.error(
                    f"Request failed after {stats.requests} request attempts: {type(e).__name__}: {e}. "
                    f"Progress already checkpointed through last successful page."
                )
                raise

            stats.retries += 1
            sleep_s = min(backoff_base_s * (2 ** max(0, attempt)), backoff_max_s)
            logger.warning(
                f"Request error {type(e).__name__}; retry {attempt + 1}/{max_retries} in {sleep_s:.1f}s "
                f"(requests={stats.requests}, pages={stats.api_pages}, rows={stats.rows_fetched:,})."
            )
            time.sleep(sleep_s)

    return []


def _maybe_tqdm(enabled: bool, total: Optional[int], desc: str):
    if not enabled:
        return None

    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit="page", leave=True)
    except Exception:
        return None


def _fetch_binance_klines_uncached(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int],
    market: Market,
    pause_s: float,
    logger: logging.Logger,
    stats: FetchStats,
    debug_pages: bool,
    on_page: Optional[Callable[[pd.DataFrame], None]] = None,
    progress: bool = True,
    progress_total_pages: Optional[int] = None,
    progress_log_every: int = 10,
    max_retries: int = 8,
    backoff_base_s: float = 2.0,
    backoff_max_s: float = 90.0,
    timeout_s: int = 30,
) -> pd.DataFrame:
    base, path, limit = BASES[market]
    url = base + path

    parts = []
    pbar = _maybe_tqdm(progress, progress_total_pages, f"{symbol} {interval}")

    try:
        while True:
            params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
            if end_ms is not None:
                params["endTime"] = end_ms

            data = _get_klines_page_with_retries(
                url=url,
                params=params,
                logger=logger,
                stats=stats,
                max_retries=max_retries,
                backoff_base_s=backoff_base_s,
                backoff_max_s=backoff_max_s,
                timeout_s=timeout_s,
            )

            if not data:
                break

            page_df = _parse_klines_page(data)
            if page_df.empty:
                break

            # checkpoint immediately after every successful page
            if on_page is not None:
                on_page(page_df)

            parts.append(page_df)

            stats.api_pages += 1
            stats.rows_fetched += len(page_df)

            last_open_ms = int(data[-1][0])
            returned = len(data)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    rows=f"{stats.rows_fetched:,}",
                    req=stats.requests,
                    retries=stats.retries,
                    saved=stats.partial_saves,
                )
            elif debug_pages or (stats.api_pages % max(1, progress_log_every) == 0):
                first_open = page_df["open_time"].min()
                last_open = page_df["open_time"].max()
                logger.info(
                    f"Progress: pages={stats.api_pages:,}, rows={stats.rows_fetched:,}, "
                    f"requests={stats.requests:,}, retries={stats.retries:,}, "
                    f"last_page={first_open} -> {last_open}, elapsed={stats.elapsed_s:.1f}s"
                )

            start_ms = last_open_ms + 1

            if returned < limit:
                break

            if end_ms is not None and start_ms >= end_ms:
                break

            time.sleep(pause_s)

    finally:
        if pbar is not None:
            pbar.close()

    if not parts:
        return pd.DataFrame()

    return _merge_dedupe_sort(*parts)


# -----------------------------------------------------------------------------
# Gap detection / fetch plan
# -----------------------------------------------------------------------------

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


def _clean_fetch_ranges(
    ranges: List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]
) -> List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]:
    cleaned = []
    for s, e in ranges:
        s = pd.to_datetime(s, utc=True)
        e = pd.to_datetime(e, utc=True) if e is not None else None
        if e is not None and e < s:
            continue
        cleaned.append((s, e))
    return cleaned


# -----------------------------------------------------------------------------
# Public fetch
# -----------------------------------------------------------------------------

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

    # resilience / retry
    max_retries: int = 8,
    backoff_base_s: float = 2.0,
    backoff_max_s: float = 90.0,
    timeout_s: int = 30,
    return_partial_on_error: bool = False,

    # progress
    progress: bool = True,
    progress_log_every: int = 10,
) -> pd.DataFrame:
    """
    Cached Binance klines fetch with durable partial checkpoints.

    Main behavior:
      - Loads main cache + any unfinished partial pages.
      - Fetches only missing ranges.
      - Saves every successful API page into a partial cache immediately.
      - On successful completion, consolidates partial files into the main cache.
      - If rate limit / network error kills the run, the partial pages remain and
        the next run resumes from the last saved candle.
    """
    logger = _make_logger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    symbol = symbol.upper().strip()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Use one of: {sorted(VALID_INTERVALS)}")

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True) if end else None
    delta = interval_to_timedelta(interval)

    stats = FetchStats()

    step = 0
    def log(msg: str, level: int = logging.INFO):
        nonlocal step
        step += 1
        logger.log(level, f"{step:02d}. {msg}")

    if not cache:
        log("Cache disabled; fetching directly from API.")
        base, _, limit = BASES[market]
        total_pages = _estimate_pages(start_ts, end_ts, interval, limit)
        return _fetch_binance_klines_uncached(
            symbol=symbol,
            interval=interval,
            start_ms=_to_ms(start_ts),
            end_ms=_to_ms(end_ts) if end_ts is not None else None,
            market=market,
            pause_s=pause_s,
            logger=logger,
            stats=stats,
            debug_pages=debug_pages,
            on_page=None,
            progress=progress,
            progress_total_pages=total_pages,
            progress_log_every=progress_log_every,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            timeout_s=timeout_s,
        )

    cache_dir_p = _resolve_cache_dir(cache_dir)
    path = _cache_path(cache_dir_p, market, symbol, interval)

    log(f"Parameters: market={market}, symbol={symbol}, interval={interval}, start={start_ts}, end={end_ts or 'LATEST'}")
    log(f"Cache directory: {cache_dir_p} (cwd={Path.cwd()})")
    log(f"Main cache file: {path}")
    log(f"Partial checkpoint dir: {_partial_dir(path)}")

    df_cache, main_rows, partial_files = _read_cache_with_partials(path)

    if df_cache.empty:
        log("Cache: MISS (no main cache or partial checkpoints found).")
    else:
        log(
            f"Cache: HIT | rows={len(df_cache):,} "
            f"(main_rows={main_rows:,}, partial_files={partial_files}) | "
            f"range={df_cache['open_time'].min()} -> {df_cache['open_time'].max()}"
        )

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

        # Missing after
        if end_ts is not None:
            if cmax < end_ts:
                fetch_ranges.append((cmax + (delta or pd.Timedelta(milliseconds=1)), end_ts))
        else:
            # end=None means latest; refresh only if stale
            if delta is not None:
                now = pd.Timestamp.now(tz="UTC")
                if (now - cmax) > (delta * max_staleness_intervals):
                    fetch_ranges.append((cmax + delta, None))
            else:
                fetch_ranges.append((cmax, None))

        # Internal gaps
        if delta is not None:
            gaps = _find_gaps(df_cache, delta)
            if gaps:
                log(f"Detected {len(gaps)} internal gap(s); will backfill gaps that overlap requested window.")
            for gs, ge in gaps:
                if ge < start_ts:
                    continue
                if end_ts is not None and gs > end_ts:
                    continue
                fetch_ranges.append((max(gs, start_ts), min(ge, end_ts) if end_ts is not None else ge))

    fetch_ranges = _clean_fetch_ranges(fetch_ranges)

    if not fetch_ranges:
        log("No API fetch needed; local cache already covers requested range.")
    else:
        for i, (s, e) in enumerate(fetch_ranges, start=1):
            log(f"Fetch plan {i}/{len(fetch_ranges)}: {s} -> {e if e is not None else 'LATEST'}")

    # Checkpoint callback: save every page immediately as a partial chunk
    def checkpoint_page(page_df: pd.DataFrame) -> None:
        fmt = _write_partial_page(page_df, path, logger)
        stats.partial_saves += 1
        if debug_pages:
            logger.debug(
                f"Checkpoint saved ({fmt}) | rows={len(page_df):,} | "
                f"{page_df['open_time'].min()} -> {page_df['open_time'].max()}"
            )

    # Fetch missing ranges. If interrupted, partial chunks already exist.
    try:
        base, _, limit = BASES[market]

        for i, (s, e) in enumerate(fetch_ranges, start=1):
            total_pages = _estimate_pages(s, e, interval, limit)
            log(f"Starting fetch range {i}/{len(fetch_ranges)}.")

            before_rows = stats.rows_fetched

            _fetch_binance_klines_uncached(
                symbol=symbol,
                interval=interval,
                start_ms=_to_ms(s),
                end_ms=_to_ms(e) if e is not None else None,
                market=market,
                pause_s=pause_s,
                logger=logger,
                stats=stats,
                debug_pages=debug_pages,
                on_page=checkpoint_page,
                progress=progress,
                progress_total_pages=total_pages,
                progress_log_every=progress_log_every,
                max_retries=max_retries,
                backoff_base_s=backoff_base_s,
                backoff_max_s=backoff_max_s,
                timeout_s=timeout_s,
            )

            range_rows = stats.rows_fetched - before_rows
            log(
                f"Completed fetch range {i}/{len(fetch_ranges)} | rows={range_rows:,} | "
                f"total_rows_fetched={stats.rows_fetched:,} | requests={stats.requests:,} | "
                f"retries={stats.retries:,} | elapsed={stats.elapsed_s:.1f}s"
            )

    except BaseException as e:
        # BaseException so KeyboardInterrupt also logs the safe checkpoint state.
        log(
            f"Fetch interrupted: {type(e).__name__}: {e}. "
            f"Successful pages have been saved as partial checkpoints. "
            f"Next run will load them and continue.",
            logging.ERROR,
        )

        # Show current cache status after interruption
        safe_df, _, pf = _read_cache_with_partials(path)
        if not safe_df.empty:
            log(
                f"Safe local progress now available | rows={len(safe_df):,} | partial_files={pf} | "
                f"range={safe_df['open_time'].min()} -> {safe_df['open_time'].max()}",
                logging.ERROR,
            )

        if return_partial_on_error:
            log("return_partial_on_error=True; returning available local slice instead of raising.")
            return _return_requested_slice(safe_df, start_ts, end_ts, log)

        raise

    # Consolidate main cache + partials into one main file after successful fetch
    df_all, _, partial_files_after = _read_cache_with_partials(path)

    if df_all.empty:
        log("Final dataset is empty; no data available.", logging.WARNING)
        return df_all

    fmt = _write_main_cache(df_all, path, logger)
    _cleanup_partials(path, logger)

    log(
        f"Cache consolidated and saved ({fmt}) | rows={len(df_all):,} | "
        f"range={df_all['open_time'].min()} -> {df_all['open_time'].max()} | "
        f"cleared_partial_files={partial_files_after}"
    )

    log(
        f"Fetch summary | new_rows={stats.rows_fetched:,} | successful_pages={stats.api_pages:,} | "
        f"requests={stats.requests:,} | retries={stats.retries:,} | "
        f"partial_saves={stats.partial_saves:,} | elapsed={stats.elapsed_s:.1f}s"
    )

    return _return_requested_slice(df_all, start_ts, end_ts, log)


def _return_requested_slice(
    df_all: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: Optional[pd.Timestamp],
    log_func: Optional[Callable[[str, int], None]] = None,
) -> pd.DataFrame:
    if df_all.empty:
        return df_all

    if end_ts is None:
        out = df_all[df_all["open_time"] >= start_ts].reset_index(drop=True)
    else:
        out = df_all[(df_all["open_time"] >= start_ts) & (df_all["open_time"] <= end_ts)].reset_index(drop=True)

    if log_func is not None:
        if out.empty:
            log_func("Returning slice | rows=0", logging.WARNING)
        else:
            log_func(
                f"Returning slice | rows={len(out):,} | "
                f"range={out['open_time'].min()} -> {out['open_time'].max()}",
                logging.INFO,
            )

    return out


def cache_info(symbol: str, interval: str, market: Market = "spot", cache_dir: Union[str, Path, None] = None) -> dict:
    cache_dir_p = _resolve_cache_dir(cache_dir)
    path = _cache_path(cache_dir_p, market, symbol.upper(), interval)
    df, main_rows, partial_files = _read_cache_with_partials(path)

    return {
        "cache_dir": str(cache_dir_p),
        "main_cache_path": str(path),
        "partial_dir": str(_partial_dir(path)),
        "exists_parquet": path.exists(),
        "exists_pickle": path.with_suffix(".pkl").exists(),
        "partial_files": int(partial_files),
        "rows_total_available": int(len(df)) if not df.empty else 0,
        "rows_main_cache": int(main_rows),
        "min_open_time": str(df["open_time"].min()) if not df.empty else None,
        "max_open_time": str(df["open_time"].max()) if not df.empty else None,
    }


def parse_show_last(show_last: str) -> pd.Timedelta:
    s = show_last.strip()
    if re.search(r"week(s)?$", s):
        n = int("".join(ch for ch in s if ch.isdigit()) or "1")
        return pd.Timedelta(days=7 * n)
    if re.search(r"month(s)?$", s) or s.endswith("M"):
        n = int("".join(ch for ch in s if ch.isdigit()) or "1")
        return pd.Timedelta(days=30 * n)
    if s.endswith("w"):
        n = int(s[:-1])
        return pd.Timedelta(days=7 * n)
    return pd.Timedelta(s)