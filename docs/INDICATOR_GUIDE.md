# Indicator & Signal Guide (`quant`)

Reference for the compute + signal layers. Pair with [`../CLAUDE.md`](../CLAUDE.md) and
[`../README.md`](../README.md). All indicators are **vectorized** (recursive parts JIT-compiled with
numba) and do **compute only** — rendering lives in `quant.viz`.

## 1. Two layers

| Layer | Module | Shape | Use for |
|---|---|---|---|
| **Indicators** | `quant.indicators` | `add_*(df, ...) -> df` (new columns) | EMA, RSI, MACD, Stochastic, ATR, Supertrend, Heikin Ashi, swings, pivots |
| **Signals** | `quant.signals` | `fn(df, ...) -> bool ndarray` | vectorized predicates + time filters that compose into entry/exit arrays |

Indicators add columns; signals turn columns into boolean arrays; a `Signals(entry_long=…, exit_long=…)`
object feeds the engine. Same representation for single runs and sweeps.

## 2. Indicator reference (`quant.indicators`)

| Function | Adds columns | Notes |
|---|---|---|
| `add_emas(df, [50,200])` | `ema_50`, `ema_200` | `ema(series, p)` / `sma(series, p)` also exported |
| `add_smas(df, [20])` | `sma_20` | |
| `add_rsi(df, 14)` | `rsi_14` | Wilder smoothing |
| `add_macd(df, 12, 26, 9)` | `macd`, `macd_signal`, `macd_hist` | |
| `add_stochastic(df, 14, 3, 3)` | `stoch_k`, `stoch_d` | |
| `add_atr(df, 14)` | `atr_14` | Wilder ATR |
| `add_supertrend(df, 10, 3.0)` | `st`, `st_dir` (+1/−1), `st_up` | recursive part in numba |
| `add_heikin_ashi(df)` | `ha_open/high/low/close`, `ha_green` | recursive `ha_open` in numba |
| `add_swings(df, left, right)` | `swing_high/low` (confirmed), `swing_last_high/low` | last-swing levels are lookahead-safe → use as ref_col stops |
| `add_pivot_points(df, "1D")` | `piv_pp/r1/s1/r2/s2` | prior-period floor-trader pivots, lookahead-safe |

MTF (`quant.data`): `resample_ohlcv(df, "5min")`, `align_timeframes(base, {…})`, and
`build_mtf(base, {"5min": lambda d: add_emas(d,[50,200])})` → columns like `5min__ema_50` on the
base grid, anti-lookahead.

## 3. Signal primitives (`quant.signals`)

Each returns a full boolean numpy array. A "ref" is a column name, scalar, Series, or array.

- **Compare:** `above(df,a,b)`, `below`, `above_all(df,a,[refs])`, `below_all`
- **Cross:** `cross_up(df,a,b)`, `cross_down`, `crossed_up_within(df,a,b,lookback)`
- **Persistence:** `last_all_above(df,x,level,n)`, `last_all_below`, `prev_all_above/below` (exclude current)
- **Candles:** `is_green/is_red`, `consecutive_green(df,n)`, `consecutive_red`
- **Trend/ribbon:** `rising(df,x,n)`, `falling`, `refs_ordered(df,[a,b,c], descending=True)`
- **Combine:** `all_of(*m)`, `any_of(*m)`, `none_of(*m)`
- **Time filters:** `in_session(df,'london')`, `hour_between(df,8,12)`, `between_times(df,'13:30','20:00')`,
  `weekday_in(df,[0,1,2,3,4])`, `not_weekend(df)` (sessions/hours evaluate in the `t` column tz; pass `tz=`).

Every `Strategy` also gets time filtering for free via fields `session` / `hours` / `weekdays` /
`avoid_weekends`, applied to entries in `signals()`.

## 4. Building a signal (example)

```python
from quant import signals as S
from quant.indicators import add_emas
from quant.engine import Signals

df = add_emas(df, [20, 50])
entry = S.all_of(
    S.cross_up(df, "close", "ema_20"),          # price crosses EMA20
    S.last_all_above(df, "close", "ema_50", 3), # above EMA50 for 3 bars
)
exit_ = S.cross_down(df, "close", "ema_20")
sig = Signals(entry_long=entry, exit_long=exit_)
```

## 5. Add a new indicator

1. Add a vectorized `add_myind(df, ...) -> df` in the right `quant/indicators/*.py` (or a new module),
   creating clearly-named columns. Keep it numpy/pandas-vectorized; if a recursive loop is
   unavoidable, write the kernel as an `@njit` function over numpy arrays (see `volatility.py`,
   `candles.py`).
2. Export it in `quant/indicators/__init__.py`.
3. Use it from a strategy's `prepare()`, then reference its columns in `build_signals()`.

## 6. Add a new signal primitive

Add a numpy-native function to `quant/signals/primitives.py` (use the `_arr` resolver and the
`_roll_all` cumulative-sum window helper for O(n) rolling conditions), export it in
`quant/signals/__init__.py`, and add a unit test in `tests/test_signals.py`.
