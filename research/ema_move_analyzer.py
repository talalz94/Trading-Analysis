from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmaMoveSpec:
    """
    Defines one EMA move study.

    fast_col:
      The EMA you expect to rise early, e.g. 1m EMA50.

    slow_col:
      The slower EMA you compare against, e.g. 1m EMA100.

    dominance_cols:
      Optional higher-timeframe EMAs to compare against.
      Example: 5m EMA100, 15m EMA100.
    """
    name: str
    fast_col: str
    slow_col: Optional[str] = None
    dominance_cols: Sequence[str] = ()


def _safe_div(num, den):
    return np.where(np.abs(den) > 1e-12, num / den, np.nan)


def _future_max(series: pd.Series, bars: int) -> pd.Series:
    return series.shift(-1).rolling(int(bars), min_periods=1).max().shift(-(int(bars) - 1))


def _future_min(series: pd.Series, bars: int) -> pd.Series:
    return series.shift(-1).rolling(int(bars), min_periods=1).min().shift(-(int(bars) - 1))


def add_ema_move_features(
    df: pd.DataFrame,
    specs: Sequence[EmaMoveSpec],
    slope_bars: Sequence[int] = (3, 5, 10, 15, 30),
    future_bars: Sequence[int] = (5, 10, 15, 30, 60),
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """
    Adds EMA strength features and future-return labels.

    Main features:
      {name}__slope_pct_{n}
      {name}__slope_per_bar_pct_{n}
      {name}__slope_accel_pct_{n}
      {name}__spread_pct
      {name}__spread_change_pct_{n}
      {name}__dominance_pct__<col>

    Future labels:
      fwd_close_ret_pct_{n}
      fwd_max_up_pct_{n}
      fwd_max_down_pct_{n}
    """
    out = df.copy()

    for spec in specs:
        if spec.fast_col not in out.columns:
            raise KeyError(f"fast_col missing: {spec.fast_col}")

        fast = pd.to_numeric(out[spec.fast_col], errors="coerce")

        for n in slope_bars:
            n = int(n)

            slope_col = f"{spec.name}__slope_pct_{n}"
            slope_per_bar_col = f"{spec.name}__slope_per_bar_pct_{n}"
            accel_col = f"{spec.name}__slope_accel_pct_{n}"

            out[slope_col] = _safe_div(fast - fast.shift(n), fast.shift(n)) * 100.0
            out[slope_per_bar_col] = out[slope_col] / n
            out[accel_col] = out[slope_col] - out[slope_col].shift(n)

        if spec.slow_col is not None:
            if spec.slow_col not in out.columns:
                raise KeyError(f"slow_col missing: {spec.slow_col}")

            slow = pd.to_numeric(out[spec.slow_col], errors="coerce")

            out[f"{spec.name}__spread_pct"] = _safe_div(fast - slow, slow) * 100.0

            for n in slope_bars:
                n = int(n)
                out[f"{spec.name}__spread_change_pct_{n}"] = (
                    out[f"{spec.name}__spread_pct"] - out[f"{spec.name}__spread_pct"].shift(n)
                )

        for dom_col in spec.dominance_cols:
            if dom_col not in out.columns:
                raise KeyError(f"dominance_col missing: {dom_col}")

            dom = pd.to_numeric(out[dom_col], errors="coerce")
            clean_name = dom_col.replace("__", "_").replace("/", "_")
            out[f"{spec.name}__dominance_pct__{clean_name}"] = _safe_div(fast - dom, dom) * 100.0

    close = pd.to_numeric(out[price_col], errors="coerce")
    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")

    for n in future_bars:
        n = int(n)

        out[f"fwd_close_ret_pct_{n}"] = _safe_div(close.shift(-n) - close, close) * 100.0
        out[f"fwd_max_up_pct_{n}"] = _safe_div(_future_max(high, n) - close, close) * 100.0
        out[f"fwd_max_down_pct_{n}"] = _safe_div(_future_min(low, n) - close, close) * 100.0

    return out


def summarize_feature_buckets(
    df: pd.DataFrame,
    feature_col: str,
    fwd_bars: int,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Shows how future returns behave as the feature gets stronger.

    Useful for answering:
      "When EMA slope is in the top 20%, do future returns improve?"
    """
    work = df[[feature_col, f"fwd_close_ret_pct_{fwd_bars}", f"fwd_max_up_pct_{fwd_bars}", f"fwd_max_down_pct_{fwd_bars}"]].dropna().copy()

    if work.empty:
        return pd.DataFrame()

    work["bucket"] = pd.qcut(work[feature_col], q=bins, duplicates="drop")

    result = (
        work.groupby("bucket", observed=True)
        .agg(
            events=(feature_col, "size"),
            feature_min=(feature_col, "min"),
            feature_median=(feature_col, "median"),
            feature_max=(feature_col, "max"),
            avg_fwd_close_ret=(f"fwd_close_ret_pct_{fwd_bars}", "mean"),
            median_fwd_close_ret=(f"fwd_close_ret_pct_{fwd_bars}", "median"),
            win_rate=(f"fwd_close_ret_pct_{fwd_bars}", lambda s: float((s > 0).mean() * 100)),
            avg_max_up=(f"fwd_max_up_pct_{fwd_bars}", "mean"),
            avg_max_down=(f"fwd_max_down_pct_{fwd_bars}", "mean"),
        )
        .reset_index()
    )

    return result


def grid_search_static_thresholds(
    df: pd.DataFrame,
    slope_col: str,
    spread_col: Optional[str],
    fwd_bars: int,
    slope_quantiles: Sequence[float] = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    spread_quantiles: Sequence[float] = (0.50, 0.60, 0.70, 0.80),
    target_move_pct: float = 0.20,
    min_events: int = 30,
) -> pd.DataFrame:
    """
    Static threshold search.

    Example:
      EMA50 slope over 10 bars >= X
      AND EMA50-EMA100 spread >= Y

    Reports what happened over the next fwd_bars.
    """
    needed = [slope_col, f"fwd_close_ret_pct_{fwd_bars}", f"fwd_max_up_pct_{fwd_bars}", f"fwd_max_down_pct_{fwd_bars}"]
    if spread_col:
        needed.append(spread_col)

    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if work.empty:
        return pd.DataFrame()

    slope_thresholds = work[slope_col].quantile(list(slope_quantiles)).dropna().unique()

    if spread_col:
        spread_thresholds = work[spread_col].quantile(list(spread_quantiles)).dropna().unique()
    else:
        spread_thresholds = [None]

    rows = []

    for slope_thr in slope_thresholds:
        for spread_thr in spread_thresholds:
            cond = work[slope_col] >= slope_thr

            if spread_col and spread_thr is not None:
                cond = cond & (work[spread_col] >= spread_thr)

            sample = work[cond]

            if len(sample) < min_events:
                continue

            fwd = sample[f"fwd_close_ret_pct_{fwd_bars}"]
            mfe = sample[f"fwd_max_up_pct_{fwd_bars}"]
            mae = sample[f"fwd_max_down_pct_{fwd_bars}"]

            wins = fwd[fwd > 0]
            losses = fwd[fwd <= 0]

            rows.append({
                "slope_col": slope_col,
                "spread_col": spread_col,
                "fwd_bars": fwd_bars,
                "slope_threshold": float(slope_thr),
                "spread_threshold": float(spread_thr) if spread_thr is not None else np.nan,
                "events": int(len(sample)),
                "avg_fwd_close_ret": float(fwd.mean()),
                "median_fwd_close_ret": float(fwd.median()),
                "win_rate": float((fwd > 0).mean() * 100),
                "target_hit_rate": float((mfe >= target_move_pct).mean() * 100),
                "avg_max_up": float(mfe.mean()),
                "avg_max_down": float(mae.mean()),
                "avg_winner": float(wins.mean()) if len(wins) else np.nan,
                "avg_loser": float(losses.mean()) if len(losses) else np.nan,
                "score": float(fwd.mean() * np.sqrt(len(sample))),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        ["score", "target_hit_rate", "avg_fwd_close_ret"],
        ascending=False,
    ).reset_index(drop=True)


def grid_search_dynamic_thresholds(
    df: pd.DataFrame,
    slope_col: str,
    spread_col: Optional[str],
    fwd_bars: int,
    rolling_windows: Sequence[int] = (500, 1000, 2000),
    slope_quantiles: Sequence[float] = (0.70, 0.80, 0.85, 0.90),
    spread_quantiles: Sequence[float] = (0.50, 0.60, 0.70),
    target_move_pct: float = 0.20,
    min_events: int = 30,
) -> pd.DataFrame:
    """
    Dynamic threshold search.

    Example:
      EMA50 slope is above its rolling 80th percentile
      AND EMA spread is above its rolling 60th percentile.

    Uses shift(1), so thresholds use only past data.
    """
    rows = []

    base_cols = [slope_col, f"fwd_close_ret_pct_{fwd_bars}", f"fwd_max_up_pct_{fwd_bars}", f"fwd_max_down_pct_{fwd_bars}"]
    if spread_col:
        base_cols.append(spread_col)

    work = df[base_cols].replace([np.inf, -np.inf], np.nan).copy()

    for window in rolling_windows:
        window = int(window)

        for slope_q in slope_quantiles:
            slope_thr = work[slope_col].rolling(window, min_periods=max(50, window // 5)).quantile(slope_q).shift(1)
            cond_base = work[slope_col] >= slope_thr

            if spread_col:
                for spread_q in spread_quantiles:
                    spread_thr = work[spread_col].rolling(window, min_periods=max(50, window // 5)).quantile(spread_q).shift(1)
                    cond = cond_base & (work[spread_col] >= spread_thr)

                    sample = work[cond].dropna()

                    if len(sample) < min_events:
                        continue

                    fwd = sample[f"fwd_close_ret_pct_{fwd_bars}"]
                    mfe = sample[f"fwd_max_up_pct_{fwd_bars}"]
                    mae = sample[f"fwd_max_down_pct_{fwd_bars}"]

                    wins = fwd[fwd > 0]
                    losses = fwd[fwd <= 0]

                    rows.append({
                        "mode": "dynamic",
                        "slope_col": slope_col,
                        "spread_col": spread_col,
                        "fwd_bars": fwd_bars,
                        "rolling_window": window,
                        "slope_quantile": slope_q,
                        "spread_quantile": spread_q,
                        "events": int(len(sample)),
                        "avg_fwd_close_ret": float(fwd.mean()),
                        "median_fwd_close_ret": float(fwd.median()),
                        "win_rate": float((fwd > 0).mean() * 100),
                        "target_hit_rate": float((mfe >= target_move_pct).mean() * 100),
                        "avg_max_up": float(mfe.mean()),
                        "avg_max_down": float(mae.mean()),
                        "avg_winner": float(wins.mean()) if len(wins) else np.nan,
                        "avg_loser": float(losses.mean()) if len(losses) else np.nan,
                        "score": float(fwd.mean() * np.sqrt(len(sample))),
                    })

            else:
                sample = work[cond_base].dropna()

                if len(sample) < min_events:
                    continue

                fwd = sample[f"fwd_close_ret_pct_{fwd_bars}"]
                mfe = sample[f"fwd_max_up_pct_{fwd_bars}"]
                mae = sample[f"fwd_max_down_pct_{fwd_bars}"]

                wins = fwd[fwd > 0]
                losses = fwd[fwd <= 0]

                rows.append({
                    "mode": "dynamic",
                    "slope_col": slope_col,
                    "spread_col": None,
                    "fwd_bars": fwd_bars,
                    "rolling_window": window,
                    "slope_quantile": slope_q,
                    "spread_quantile": np.nan,
                    "events": int(len(sample)),
                    "avg_fwd_close_ret": float(fwd.mean()),
                    "median_fwd_close_ret": float(fwd.median()),
                    "win_rate": float((fwd > 0).mean() * 100),
                    "target_hit_rate": float((mfe >= target_move_pct).mean() * 100),
                    "avg_max_up": float(mfe.mean()),
                    "avg_max_down": float(mae.mean()),
                    "avg_winner": float(wins.mean()) if len(wins) else np.nan,
                    "avg_loser": float(losses.mean()) if len(losses) else np.nan,
                    "score": float(fwd.mean() * np.sqrt(len(sample))),
                })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        ["score", "target_hit_rate", "avg_fwd_close_ret"],
        ascending=False,
    ).reset_index(drop=True)

def analyze_ema_move_strength(
    df: pd.DataFrame,
    specs: Sequence[EmaMoveSpec],
    slope_bars: Sequence[int] = (3, 5, 10, 15, 30),
    future_bars: Sequence[int] = (5, 10, 15, 30, 60),
    target_move_pct: float = 0.20,
    min_events: int = 30,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full analysis runner with progress logs.

    Returns:
      features_df
      bucket_stats
      static_thresholds
      dynamic_thresholds
    """
    import time

    t0 = time.perf_counter()

    def log(msg: str):
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"[EMA ANALYZER | {elapsed:,.1f}s] {msg}")

    specs = list(specs)
    slope_bars = tuple(int(x) for x in slope_bars)
    future_bars = tuple(int(x) for x in future_bars)

    total_jobs = len(specs) * len(slope_bars) * len(future_bars)
    job_no = 0

    log(
        f"Started | rows={len(df):,} | specs={len(specs)} | "
        f"slope_bars={slope_bars} | future_bars={future_bars} | jobs={total_jobs:,}"
    )

    log("Adding EMA move features and forward-return labels...")
    features = add_ema_move_features(
        df=df,
        specs=specs,
        slope_bars=slope_bars,
        future_bars=future_bars,
    )
    log(f"Feature generation complete | cols={len(features.columns):,}")

    bucket_frames = []
    static_frames = []
    dynamic_frames = []

    for spec_i, spec in enumerate(specs, start=1):
        spread_col = f"{spec.name}__spread_pct" if spec.slow_col is not None else None

        log(f"Processing spec {spec_i}/{len(specs)} | {spec.name}")

        for n in slope_bars:
            slope_col = f"{spec.name}__slope_pct_{int(n)}"

            for fwd in future_bars:
                job_no += 1

                log(
                    f"Job {job_no}/{total_jobs} | "
                    f"slope={slope_col} | spread={spread_col} | fwd_bars={fwd}"
                )

                step_t = time.perf_counter()

                bucket = summarize_feature_buckets(
                    features,
                    feature_col=slope_col,
                    fwd_bars=int(fwd),
                    bins=10,
                )

                if not bucket.empty:
                    bucket.insert(0, "feature_col", slope_col)
                    bucket.insert(1, "fwd_bars", int(fwd))
                    bucket_frames.append(bucket)

                log(f"  bucket stats done | rows={len(bucket):,}")

                static_result = grid_search_static_thresholds(
                    features,
                    slope_col=slope_col,
                    spread_col=spread_col,
                    fwd_bars=int(fwd),
                    target_move_pct=target_move_pct,
                    min_events=min_events,
                )

                if not static_result.empty:
                    static_frames.append(static_result)

                log(f"  static grid done | rows={len(static_result):,}")

                dynamic_result = grid_search_dynamic_thresholds(
                    features,
                    slope_col=slope_col,
                    spread_col=spread_col,
                    fwd_bars=int(fwd),
                    target_move_pct=target_move_pct,
                    min_events=min_events,
                )

                if not dynamic_result.empty:
                    dynamic_frames.append(dynamic_result)

                step_elapsed = time.perf_counter() - step_t

                log(
                    f"  dynamic grid done | rows={len(dynamic_result):,} | "
                    f"job_time={step_elapsed:,.1f}s"
                )

    bucket_stats = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
    static_thresholds = pd.concat(static_frames, ignore_index=True) if static_frames else pd.DataFrame()
    dynamic_thresholds = pd.concat(dynamic_frames, ignore_index=True) if dynamic_frames else pd.DataFrame()

    log(
        f"Finished | total_time={time.perf_counter() - t0:,.1f}s | "
        f"bucket_rows={len(bucket_stats):,} | "
        f"static_rows={len(static_thresholds):,} | "
        f"dynamic_rows={len(dynamic_thresholds):,}"
    )

    return {
        "features_df": features,
        "bucket_stats": bucket_stats,
        "static_thresholds": static_thresholds,
        "dynamic_thresholds": dynamic_thresholds,
    }