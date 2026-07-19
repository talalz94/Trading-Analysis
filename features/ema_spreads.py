
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmaSpreadSpec:
    name: str
    left: str
    right: str


def add_ema_spread_features(
    df: pd.DataFrame,
    specs: Sequence[EmaSpreadSpec],
    dynamic_windows: Tuple[int, ...] = (1000,),
    dynamic_quantiles: Tuple[float, ...] = (0.80,),
    min_periods_ratio: float = 0.20,
) -> pd.DataFrame:
    """
    Adds EMA spread features.

    For each pair:
      {name}__abs = left EMA - right EMA
      {name}__pct = (left EMA - right EMA) / right EMA * 100

    Also adds dynamic rolling percentile thresholds:
      {name}__pct_q80_w1000
      {name}__pct_q85_w2000
      etc.

    Dynamic thresholds are shifted by 1 candle to avoid lookahead.
    """
    out = df.copy()

    for spec in specs:
        if spec.left not in out.columns:
            raise KeyError(f"Missing left EMA column: {spec.left}")

        if spec.right not in out.columns:
            raise KeyError(f"Missing right EMA column: {spec.right}")

        left = pd.to_numeric(out[spec.left], errors="coerce")
        right = pd.to_numeric(out[spec.right], errors="coerce")

        abs_col = f"{spec.name}__abs"
        pct_col = f"{spec.name}__pct"

        out[abs_col] = left - right
        out[pct_col] = np.where(
            right.abs() > 1e-12,
            (left - right) / right * 100.0,
            np.nan,
        )

        for window in dynamic_windows:
            window = int(window)
            min_periods = max(50, int(window * min_periods_ratio))

            for q in dynamic_quantiles:
                q_int = int(round(q * 100))
                thr_col = f"{pct_col}__q{q_int}_w{window}"

                out[thr_col] = (
                    out[pct_col]
                    .rolling(window, min_periods=min_periods)
                    .quantile(q)
                    .shift(1)
                )

    return out