# Indicator & Feature Usage Guide

> Reference for the indicator/feature layer. Pair with [`../CLAUDE.md`](../CLAUDE.md) (orientation)
> and [`ARCHITECTURE.md`](ARCHITECTURE.md) (where this layer is heading).
>
> Everything here describes the **current** system. Column names and config keys are extracted
> directly from the source and are accurate as of 2026-07.

---

## 1. Two ways to add signals to a DataFrame

| Layer | Where | Shape | Registered? | Use for |
|---|---|---|---|---|
| **Indicators** | `indicators/*.py` | class with `compute()` + plotly `add_traces()` | Yes — `INDICATOR_REGISTRY` | Classic TA (EMA, MACD, RSI, Supertrend, S/R, …) |
| **Features** | `features/*.py` | plain function `add_*_features(df, specs) -> df` | No — imported directly | Derived signals on top of indicators (EMA spreads/compression, cross-through, stochastic states) |

Indicators go through the pipeline; features are called explicitly after alignment.

## 2. The indicator contract

Every indicator is a class (duck-typed to the `Indicator` Protocol in `indicators_load.py`):

```python
class MyIndicator:
    name = "my_indicator"        # registry key
    is_overlay = True            # True = draw on price panel; False = own subpanel
    row_weight = 0.0             # subpanel height weight (0 for overlays)

    @staticmethod
    def compute(df, cfg: dict, tag: str) -> tuple[pd.DataFrame, list[str]]:
        out = df.copy()
        out[f"{tag}__VALUE"] = ...          # create columns namespaced by tag
        return out, [f"{tag}__VALUE"]       # (df, list_of_created_columns)

    @staticmethod
    def add_traces(fig, df, cfg, tag, row, price_row) -> None: ...   # plotly rendering
    @staticmethod
    def yaxis_title(cfg, tag) -> str: ...
```

**Column naming:** `col(tag, base)` → `f"{tag}__{base}"`. After `align_timeframes`, a column from
another timeframe becomes `f"{tf}__{tag}__{base}"` (e.g. `5m__ema__EMA_50`).

**`tag`** is a unique instance id you pick. It lets you run the same indicator multiple times with
different configs (e.g. `ema_fast` vs `ema_slow`). Tags must be unique within a `build_feature_df` call.

## 3. Using indicators via the pipeline

```python
from pipeline import build_feature_df, IndicatorSpec

specs = [
    IndicatorSpec(name="moving_average", tag="ema", config={"type": "ema", "periods": [50, 100, 200]}),
    IndicatorSpec(name="macd",           tag="macd", config={"fast": 12, "slow": 26, "signal": 9}),
    IndicatorSpec(name="rsi_divergence", tag="rsi14", config={"length": 14}),
]
df, specs, ma_cols = build_feature_df(raw_df, tz="Asia/Karachi", ma_windows=[20, 50], indicators=specs)
# creates: ema__EMA_50, ema__EMA_100, ema__EMA_200, macd__MACD, macd__SIGNAL, macd__HIST,
#          rsi14__RSI, rsi14__BULL_DIV, rsi14__BEAR_DIV, MA20, MA50, ...
```

Multi-timeframe: run `build_feature_df` per timeframe, then merge with `align_timeframes` (see
CLAUDE.md §4). HTF feature columns are automatically shifted +1 candle to prevent lookahead.

## 4. Indicator reference

Legend: **overlay** = drawn on price; **panel** = separate subplot. ✅ vectorized / ⚠ contains
per-bar Python loop (slow — see §7).

### moving_average ✅ (overlay)
Config: `type` `"sma"|"ema"`, `period` int **or** `periods` list, `source` (default `"close"`),
`min_periods`, `adjust` (ema only).
Output: `{tag}__SMA_{p}` or `{tag}__EMA_{p}` per period.

### macd ✅ (panel)
Config: `fast` (12), `slow` (26), `signal` (9).
Output: `{tag}__MACD`, `{tag}__SIGNAL`, `{tag}__HIST`.

### stochastic ✅ (panel)
Config: `k_length` (14), `d_length` (3), `smooth` (3), `show_levels`.
Output: `{tag}__K`, `{tag}__D`.

### bollinger_bands ✅ (overlay)
Config: `length` (20), `stdev` (2.0), `fill`.
Output: `{tag}__MID`, `{tag}__UP`, `{tag}__LO`.

### momentum ✅ (panel)
Config: `length`, `mode` (`"diff"` / roc).
Output: `{tag}__MOM`.

### volume_ma ✅ (panel)
Config: `ma_length`.
Output: `{tag}__VOL_MA`.

### supertrend ⚠ (overlay)
Config: `length` (10), `multiplier` (3.0), `show_markers`, `marker_*`.
Output: `{tag}__ST` (line), `{tag}__DIR` (+1/−1 trend), `{tag}__BUY`, `{tag}__SELL`,
`{tag}__ST_BUY_LINE`, `{tag}__ST_SELL_LINE`.
Hot loops: `supertrend.py:32`, `:49` (recursive band/direction). numba candidate.

### rsi_divergence ⚠ (panel)
Config: `length` (14), `pivot_lookback`, `ob_level`, `os_level`, `min_rsi_delta`, `zone_mode`,
plus many label/marker display keys.
Output: `{tag}__RSI`, `{tag}__BULL_DIV`, `{tag}__BEAR_DIV`, `{tag}__BULL_RSI_LINE`,
`{tag}__BEAR_RSI_LINE`, `{tag}__BULL_START_RSI`, `{tag}__BEAR_START_RSI`.
Hot loop: Wilder smoothing `.iloc[i]` at `rsi_divergence.py:40`.

### market_structure ⚠ (overlay)
Config: `left` (pivot left bars), `right` (confirmation bars), `min_swing_pct`, `high_col`, `low_col`.
Output: `{tag}__SWING_HIGH`, `{tag}__SWING_LOW` (bool at confirmation), `{tag}__SWING_HIGH_PRICE`,
`{tag}__SWING_LOW_PRICE`, `{tag}__LAST_SWING_HIGH`, `{tag}__LAST_SWING_LOW` (forward-filled level),
`{tag}__HH_PRICE`, `{tag}__HL_PRICE`, `{tag}__LH_PRICE`, `{tag}__LL_PRICE`.
> `LAST_SWING_LOW`/`HIGH` are the columns used for **structure-based stop losses**
> (`StopLossConfig(mode="ref_col", ref_col="5m__ms__LAST_SWING_LOW")`).
Hot loop: `market_structure.py:78` (single pass; mild).

### support_resistance ⚠ (overlay) — slowest indicator
Config: `left`, `right`, `min_touches`, `tolerance_pct`, `near_pct`, `level_method`, `selection`,
`max_levels`, `max_clusters`, `breakout_basis`, `breakout_buffer_pct`, `freeze_level_after_activation`,
`lookback_bars`, `min_bars_between_touches`, `overlap_pct`, … (see source for the full set).
Output: `{tag}__NEAREST_SUPPORT`, `{tag}__NEAREST_RESISTANCE`, `{tag}__NEAR_SUPPORT`,
`{tag}__NEAR_RESISTANCE`, `{tag}__DIST_TO_SUPPORT_PCT`, `{tag}__DIST_TO_RESISTANCE_PCT`,
`{tag}__PIVOT_HIGH(_PRICE)`, `{tag}__PIVOT_LOW(_PRICE)`, plus per-zone `_ZONE_LOW/_ZONE_HIGH/_TOUCHES`.
Hot loop: per-bar cluster maintenance at `support_resistance.py:425`. Top numba/rewrite priority.

### trend_channels ⚠ (overlay)
Config: `swing_detection_window`, `min_touches`, `max_channel_angle`, `projection_bars`,
`enable_dynamic_updates`, `min_history_bars`, `max_points`, `touch_tolerance`, `timeframe`, display keys.
Output: `{tag}__Ch Up/Lo/Mid` (+ `(hist)` variants), `{tag}__BreakUp`, `{tag}__BreakDn`.
Hot loops: `trend_channels.py:287` (`while`), O(swing²) fits at `:195/:233`.

### precomputed (plot-only, not a real indicator)
Does **not** compute anything — it validates that named columns already exist and renders them.
Produced by `precomputed_factory.py` to plot already-aligned cross-timeframe columns.

## 5. Features layer (`features/*.py`)

Called explicitly on the aligned df; all vectorized. Each takes a spec dataclass + returns df with
new `{name}__{suffix}` columns.

- **`ema_spreads.add_ema_spread_features(df, [EmaSpreadSpec(...)])`** — abs/pct spread between two EMAs
  + rolling-quantile thresholds (uses `shift(1)`, anti-lookahead). Cols: `{name}__spread`,
  `{name}__spread_pct`, `{name}__thr_*`.
- **`ema_compression.add_ema_compression_features(df, [EMACompressionSpec(...)])`** — basket range %,
  bullish/bearish ordering, compression→expansion breakout flags. Cols: `{name}__range_pct`,
  `{name}__bull_breakout`, `{name}__bear_breakout`, `{name}__ordered_*`. Config: `cols`,
  `compress_thr`, `expand_thr`, `lookback`, `require_bullish_order`, `fresh_only`.
- **`cross_through.add_cross_through_features(df, [CrossThroughSpec(...)])`** — counts how many
  reference lines a series has crossed within a lookback (cumsum + rolling-any). 
- **`stochastic_signals.add_stochastic_signal_features(df, ...)`** — above/below/cross state,
  active-cross via `np.maximum.accumulate`.
- **`ema_diagnostics` / `ema_diagnostic_plots`** — pair/group spread + cut-through rank/speed
  diagnostics and their plotly panels.

## 6. Vectorized rule/feature helpers

`simulation/rule_features.py` holds **vectorized** equivalents of common rule predicates — the
seed of the future unified signal layer:
- `add_last_n_compare(df, left, op, right, n, out_col)` — condition true for the last *n* bars.
- `add_cross_compare(df, left, direction, right, out_col, lookback)` — up/down cross as a boolean column.

These return whole boolean columns (fast), unlike the per-bar `RuleContextMixin` predicates.

## 7. Performance notes

- **Vectorized already:** moving_average, macd, stochastic, bollinger_bands, momentum, volume_ma,
  and all `features/*`. These are fine at scale.
- **Per-bar loops (slow), ranked worst-first:** `support_resistance` → `trend_channels` →
  `supertrend` → `rsi_divergence` → `market_structure`. These dominate feature-build time on 1m data
  and are the top targets for numba/vectorization (see ARCHITECTURE.md → Indicator engine).
- No TA-Lib / pandas-ta / bottleneck is used; everything is hand-rolled pandas/numpy. `numba` is
  installed but not yet used.

## 8. How to add a new indicator (today)

1. Create `indicators/my_ind.py` with a class following §2 (`name`, `compute`, `add_traces`).
2. Import it in `indicators_load.py` and add it to `INDICATOR_REGISTRY`.
   ⚠ Do **not** rely on `indicators/__init__.py`'s registry — it is stale (10 vs 12 entries).
3. Use it via `IndicatorSpec(name="my_ind", tag="x", config={...})`.
4. Prefer vectorized numpy/pandas. If a recursive loop is unavoidable, write the kernel as a
   `@njit` numba function over numpy arrays, not a pandas `.iloc[i]` loop.

> In the target architecture (ARCHITECTURE.md), `compute()` (data) and `add_traces()` (viz) are
> split so indicators no longer depend on plotly, and indicator outputs become cacheable.
