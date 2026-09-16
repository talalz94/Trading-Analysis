# CLAUDE.md — Project Orientation (full context for a fresh session)

> Read this first. It is the single source of truth for picking up this project cold — no prior
> conversation needed. Keep it current when the architecture changes.
> **Companion docs:** [`README.md`](README.md) (user guide) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> (design + roadmap) · [`docs/INDICATOR_GUIDE.md`](docs/INDICATOR_GUIDE.md) · [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md)
> · [`experiments/README.md`](experiments/README.md) · **[`docs/NOTEBOOK_GUIDE.md`](docs/NOTEBOOK_GUIDE.md)
> (where notebook / config / helper code belongs — read before touching a notebook)** ·
> **[`docs/SESSION_NOTES.md`](docs/SESSION_NOTES.md)
> (research findings, gotchas, next steps — read this for running context).**
>
> **Strategy search:** [`docs/WHAT_WORKS.md`](docs/WHAT_WORKS.md) is the short verdict — read it
> before proposing any new strategy idea, because most of them have already been tested and killed.
> [`docs/SEARCH_PLAN.md`](docs/SEARCH_PLAN.md) is the plan that was executed and
> [`docs/STRATEGY_SEARCH_LOG.md`](docs/STRATEGY_SEARCH_LOG.md) the full phase-by-phase evidence.
> **Four methodology rules now bind all research here:** difference every forward-return screen
> against a within-day-shuffled null (`experiments/null_control.py`); make every entry compete
> against the `time_entry` metronome; count survivors against `n_configs x 0.5^n_windows`; and trim
> the top K trades, not just the top one, before believing a fat-tailed result.

---

## 1. What this is

`quant` — a Python **quantitative research & backtesting platform**. Purpose: **fast strategy
research and backtesting**, testing thousands→millions of parameter combinations over years of
1-minute data. **Not for live trading.** Focus asset: **gold** (Binance PAXGUSDT/XAUTUSDT proxies,
and **true spot XAU/USD via Dukascopy**); the data layer is source-agnostic for other assets.

Design in one line: **vectorized signal generation + a numba-JIT position/PnL engine**. Signals are
whole-series boolean numpy arrays computed once; execution runs in a compiled kernel. Result: a
1-year 1-minute backtest runs in ~0.05–0.25 s (kernel ~6–48 ms); sweeps are ~200× faster than the
original per-bar simulator (which has been removed).

History: refactored from an organically-grown notebook codebase (now deleted; the user keeps an
external backup). Everything lives in `quant/` + `experiments/`.

## 2. Repo / project status

- **GitHub:** https://github.com/talalz94/Trading-Analysis · default branch **`main`**.
- Git user: Talal. Commit trailer used: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Secrets:** only from `.env` (git-ignored). `config.py`/`quant/config.py` load it. History is
  clean of keys. The user was advised to **rotate the old Binance keys** (read-only market keys;
  not needed for data — the public API is used).
- **Not committed (git-ignored):** `.env`, `data/` (parquet cache), `**/reports/`,
  `**/experiments/results/`, `__pycache__`, notebooks' checkpoints.
- **Tests:** `python -m pytest tests/` → **534 passing**. Keep them green.

## 3. Architecture & data flow

```
quant.data.get_ohlcv(symbol, tf, start, end, source, market, tz)      # cached, incremental, pushdown
   -> pandas OHLCV (open_time tz-aware UTC + 't' display-tz column)
strategy.prepare(df)          # add vectorized indicator columns
strategy.signals(df)          # -> Signals(entry/exit bool arrays); base applies time filters
run_backtest(df, signals, cfg)  # @njit kernel: SL/TP/trailing/partials/sizing/margin/costs -> SimResult
   -> SimResult(trades, equity_curve, stats)   # stats incl Sharpe/Sortino/Calmar/…
quant.viz.ResearchChart / quant.optimize.run_grid / experiments.Experiment
```

## 4. Package layout (complete, current)

```
quant/
├── config.py            # Settings + .env loading (secrets from env only)
├── logging_utils.py     # logger + tqdm helpers
├── cli.py               # `quant fetch|backtest|report` (entry point: quant = quant.cli:main)
├── data/
│   ├── base.py          #   DataSource protocol + OHLCV contract + validate_ohlcv
│   ├── binance.py       #   BinanceSource + get_source registry (binance, dukascopy)
│   ├── _binance_fetch.py#   proven incremental Binance klines fetcher (retries, partial checkpoints)
│   ├── dukascopy.py     #   DukascopySource — true spot XAU/USD + FX (needs dukascopy-python)
│   ├── cache.py         #   generic incremental Parquet cache for range-based providers
│   ├── store.py         #   pushdown load (column projection + date range) via polars/pyarrow
│   ├── loader.py        #   get_ohlcv(): fast cache path OR incremental fetch
│   └── timeframe.py     #   resample_ohlcv + align_timeframes + build_mtf (anti-lookahead) +
│                        #     tf_frames() = one FRAME per timeframe with equal history IN BARS
│                        #     (build_mtf gives multi-timeframe COLUMNS; drawing a 4h shape needs
│                        #     the 4h FRAME, because its slopes are per-4h-bar), tf_minutes(),
│                        #     bar_interval() = a bar's duration read off the frame, not a label
├── indicators/          # vectorized compute ONLY (no plotting); recursive parts in numba
│   ├── overlays.py      #   ema, sma, add_emas, add_smas + add_mtf_emas(base, frames, {tf:[periods]})
│   │                    #     -> {tf}__ema_{p} on the base grid, shift_bars=1 so a 15m EMA is
│   │                    #     invisible until its bar closed. Takes the LOADED frames rather than
│   │                    #     resampling like build_mtf: an EMA200 wants 200 bars of its own
│   │                    #     timeframe, and a short base frame silently yields NaN
│   ├── oscillators.py   #   add_rsi (Wilder), add_macd, add_stochastic, add_rsi_divergence
│   │                    #     (+ rsi_divergence_table; lag == `right` bars exactly, `left` is free)
│   │                    #     and add_rsi_divergence_realtime (zero-lag but PROVISIONAL, ~53% ratified)
│   ├── candles.py       #   add_heikin_ashi (recursive ha_open in numba) + add_candle_runs ->
│   │                    #     up_run/dn_run = consecutive green/red candles ending here (a doji
│   │                    #     ends both). The COUNT, not a fixed-n predicate, so one column serves
│   │                    #     every n AND a 15m run can be merged onto a 5m grid -- where
│   │                    #     consecutive_green() would count 5m candles whoever's line it was
│   ├── candle_patterns.py #  62 CANDLESTICK PATTERNS as bool cols `cdl_<name>`, flagged on the bar
│   │                    #     the pattern COMPLETES. Scale-free (ratios + a rolling avg body), with
│   │                    #     trend context so hammer/hanging-man and inv-hammer/shooting-star are
│   │                    #     separated. `gap_mode='none'` default: BTC has 0.000% true gaps, so the
│   │                    #     textbook gap tests can never fire (7 patterns stay dead by design).
│   │                    #     Adds cdl_bull_n / cdl_bear_n / cdl_score to use the whole set at once.
│   ├── chart_patterns.py #   22 CHART PATTERNS (double/triple top+bottom, H&S + inverse, wedges,
│   │                    #     triangles, rectangles, flags, pennants, broadening) over ZIGZAG pivots.
│   │                    #     Two-stage and strictly causal: a shape is registered when its pivots
│   │                    #     confirm, and fires only when price CLOSES through its boundary (which
│   │                    #     may SLOPE - a triangle breaks its rising line, not a flat proxy). Each
│   │                    #     hit publishes pat_entry / pat_stop / pat_target (measured move) plus
│   │                    #     forward-filled `_ff` copies for sl_mode='ref_col' and signal TPs.
│   │                    #     TRENDLINES ARE FITTED, not assumed: _best_line runs RANSAC over pivot
│   │                    #     pairs, counts pivots within line_tol_atr as TOUCHES, and requires an
│   │                    #     ENVELOPE (nothing pokes through). A line through 2 points is not a
│   │                    #     trendline - two points always fit - and that was the original bug.
│   │                    #     PatternSpec + resolve_specs give PER-PATTERN strictness, with
│   │                    #     sensitivity='major'|'normal'|'minor' as the coarse dial.
│   │                    #     chart_pattern_shapes() returns the fitted boundaries for drawing;
│   │                    #     TUNED_SPECS / tuned_specs() are the Phase-15 per-pattern calibration and
│   │                    #     chart_pattern_shapes_tuned() merges the passes they need (atr_mult sizes
│   │                    #     the zigzag per PASS, so differing values cannot share one call).
│   │                    #     They are tuned for WELL-FORMED, FREQUENT shapes, NOT for direction:
│   │                    #     hit rate 0.48, below a coin flip. See docs/WHAT_WORKS.md.
│   │                    #     CHANNELS (channel_up/down/flat) = two PARALLEL boundaries holding a
│   │                    #     constant width; width_tol_frac IS the parallelism test and
│   │                    #     converge_min_frac makes channels and wedges mutually exclusive (a
│   │                    #     parallel channel that narrowed by a hair used to be called a wedge).
│   │                    #     slope_sym_max stops a near-flat line paired with a steep one being
│   │                    #     called SYMMETRICAL (measured ratios reached 17:1). Flags/pennants are
│   │                    #     renamed from channels/rectangles/symmetrical triangles after a pole.
│   ├── fvg.py           #   FAIR VALUE GAPS -- the three-candle imbalance. add_fvgs() flags
│   │                    #     bullish `low[i] > high[i-2]` and bearish `high[i] < low[i-2]`,
│   │                    #     PUBLISHED FROM BAR i (the third candle), never from the middle
│   │                    #     one where it visually sits -- that would be a two-bar lookahead
│   │                    #     drawn on the chart. `top` is always the HIGHER edge; writing the
│   │                    #     bearish band in candle order instead inverted every one of them.
│   │                    #     THREE mitigation thresholds, all recorded for every gap and
│   │                    #     `mitigation=` picking which retires it: touch / ce (the midpoint,
│   │                    #     default) / full. Gates are ATR-relative (min_size_atr,
│   │                    #     min_disp_atr on the MIDDLE candle's body). COLUMNS are counts and
│   │                    #     distances -- fvg_bull_n/_bear_n live now, plus the nearest one's
│   │                    #     top/bot/age/size_atr/dist_atr -- not a level, for the reason
│   │                    #     add_candle_runs gives: one column then serves every threshold and
│   │                    #     a 1h count can ride onto a 5m grid. mtf_fvgs() merges layers with
│   │                    #     the usual shift_bars=1; fvg_table() is one row per gap for drawing.
│   │                    #     Cost: 230k 1m bars -> 58,776 gaps in 270 ms (mitigation is a numba
│   │                    #     scan bounded by max_age_bars). NO EDGE as a trade filter -- see
│   │                    #     docs/WHAT_WORKS.md
│   ├── trendlines.py    #   STANDALONE TRENDLINES — add_trendlines() fits the live support and
│   │                    #     resistance line bar by bar (same RANSAC + envelope, same causality:
│   │                    #     published only from the bar its last pivot CONFIRMED) and flags
│   │                    #     tl_break_up / tl_break_dn on the close through it. A broken line is
│   │                    #     retired. trendline_table() gives one row per line for drawing.
│   │                    #     mtf_trendlines(df_1m, frames) fits lines on EVERY timeframe and
│   │                    #     merges them onto the base grid (align_timeframes shift_bars=1, so a
│   │                    #     1h line is invisible until its bar closed) -> tl_res/5m__tl_res/...;
│   │                    #     each frame's own up_run/dn_run ride along, so "three red 15m candles"
│   │                    #     is answerable on a 5m grid; trendline_level_cols() names the levels,
│   │                    #     base first then coarsest;
│   │                    #     add_trendline_ids() -> tl_res_n/tl_sup_n = WHICH line is live per bar,
│   │                    #     so a trade can name the exact line it broke, not just its timeframe.
│   ├── volatility.py    #   atr, add_atr, add_supertrend (recursive part in numba)
│   └── structure.py     #   add_swings (swing_last_low/high = ref_col stop sources), add_pivot_points,
│                        #     add_trend_structure (HH/HL/LH/LL from fixed fractals) +
│                        #     add_zigzag_structure (ATR-adaptive swings; ~1/3 the lag — preferred),
│                        #     add_sr_zones (nearest S/R per bar) + sr_zone_table (causal S/R zone timeline for plotting)
│                        #     zigzag also exposes zz_last_high/zz_last_low (running confirmed pivots -> BOS)
├── signals/
│   ├── primitives.py    #   numpy-native bool predicates (cross_up, last_all_above, consecutive_*,
│   │                    #     refs_ordered, rising/falling, becomes_true (rising edge), all_of/…)
│   └── time_filters.py  #   in_session, hour_between, between_times, weekday_in, not_weekend
├── engine/
│   ├── kernel.py        #   @njit run_kernel (the hot core) + equity_stats reducer + _fill_px/_costs
│   ├── config.py        #   BacktestConfig (fully documented), TakeProfit, Signals
│   └── run.py           #   run_backtest(df, signals, cfg) -> SimResult ; invoke_kernel helper
├── strategies/
│   ├── base.py          #   Strategy ABC (prepare + build_signals; signals() adds time filters)
│   ├── ema_cross.py     #   configurable price-vs-EMA (cross/close/full_candle, confirm_n, HA, HTF
│   │                    #     bias, exit_mode opposite/none/ha_flip/below_ema, one_entry_per_cross
│   │                    #     = one entry per EMA-cross episode) — the variant explorer
│   ├── indicators.py    #   TAGGED indicator handles: MA/EMA/SMA/RSI/ATR/HeikinAshi/Swings/SRZones
│   │                    #     — tag IS the column name (no collisions), `tf=` for higher-timeframe,
│   │                    #     multi-column handles expose sub-cols (sw.last_low, sr.support)
│   ├── conditions.py    #   composable Cond algebra (& | ~) + builders (above/full_above/full_below/
│   │                    #     green/red/ha_red/big_candle/near_support/bounce_off_support…) +
│   │                    #     indicator specs (ema[+smooth]/ma/mtf_ema/sr_zones/heikin_ashi/swings/
│   │                    #     atr) + StopSpec builders (pct_stop/ref_col_stop/price_abs_stop)
│   │                    #     candle(name) · candle_any(...) · candle_bias('bull', min_n, net=)
│   │                    #     chart_pattern(name) · chart_pattern_any('bull'|'bear')
│   │                    #     + candle_patterns()/chart_patterns() Indicator specs (auto-prepared)
│   ├── regimes.py       #   REGIME CONDITIONERS as Conds (AND onto any entry via entry_filter=):
│   │                    #     vol_regime(high/mid/low) · trendiness(trending/choppy, Kaufman ER) ·
│   │                    #     vol_shock(within=N) = ANNOUNCEMENT PROXY · range_position · CONDITIONERS dict.
│   │                    #     Thresholds are ROLLING QUANTILES of the instrument's own history (shifted
│   │                    #     one bar), never absolute numbers — an absolute cut becomes a date filter.
│   ├── rulespec.py      #   A ONE-LINE RULE LANGUAGE for notebook configs:
│   │                    #     "2 candles EMA_5m_200 close above" · "long: 3 EMA_15m_50 formed
│   │                    #     fully below red" · "1 SUP_15m touched". The Cond algebra can express
│   │                    #     all of these, but only as code, and a config is not the place for it.
│   │                    #     <n> <ref> <close|full|touch> [above|below] [red|green|NA], order-free,
│   │                    #     filler words ignored; `long:`/`short:` scopes a rule to one side.
│   │                    #     refs: EMA_<tf>_<period> · RES_/SUP_<tf> · any column on the frame.
│   │                    #     resolve_ref() maps a ref to its column (the chart's own tf is
│   │                    #     unprefixed), required_emas() says which columns to add first,
│   │                    #     describe_rules() shows what each sentence became, build_rules(side)
│   │                    #     returns None when no rule applies -- NOT an empty all/any, because
│   │                    #     guessing would turn "no short rules" into "no short may trade".
│   ├── fvg_veto.py      #   "there is an unfilled gap behind this move" as a Cond, for
│   │                    #     entry_filter=. fvg_blocks(side) is TRUE where a trade should be
│   │                    #     skipped, fvg_clear(side) its negation -- entry_filter is a MAY-ENTER
│   │                    #     condition, so handing it the veto itself would trade exactly the
│   │                    #     setups it was built to avoid and nothing would look wrong. TWO
│   │                    #     MODES, both built because they are exact mirrors and only
│   │                    #     measurement separates them: "retrace" (a bullish gap BELOW blocks a
│   │                    #     long -- price returns to sweep it) and "barrier" (a bearish gap
│   │                    #     ABOVE blocks a long -- buying into a seller's zone). Gates:
│   │                    #     max_age_bars (in the gap's OWN timeframe's bars), min_size_atr,
│   │                    #     max_dist_atr, timeframes. They describe the NEAREST live gap, so
│   │                    #     filter the population at DETECTION instead.
│   │                    #     A ONE-LINE LANGUAGE sits on top, for configs: "short: opposite FVG
│   │                    #     15m+1h within 20 candles min 0.8 ATR near 3 ATR" -- parse_fvg_rule /
│   │                    #     build_fvg_rules / describe_fvg_rules, order-free and filler-tolerant
│   │                    #     like rulespec. "OPPOSITE" IS THE WORD THAT READS TWO WAYS, and taking
│   │                    #     it backwards inverts the filter: it means the gap whose FILL moves
│   │                    #     price AGAINST the trade, so for a SHORT it is a BEARISH gap above,
│   │                    #     not a bullish one. describe_fvg_rules() prints the resolved direction
│   │                    #     per side rather than leaving it to a docstring
│   ├── builder.py       #   RuleStrategy + Rule: declarative entry->exit(->stop) rules with `once`
│   │                    #     (one per episode), `rearm`, `after` sequencing, per-rule exits+stops
│   └── ema_ribbon, rsi, macd, heikin_ashi, supertrend, mtf, key_level  (+ REGISTRY)
├── analytics/
│   ├── metrics.py       #   compute_stats (return, win/loss, PF, expectancy, Sharpe/Sortino/Calmar,
│   │                    #     drawdown, recovery, side splits)
│   ├── attribution.py   #   by_hour/weekday/month/session/regime + monthly_returns
│   └── fast.py          #   array-native stats for sweeps (no DataFrame build) + equity_stats use
├── viz/
│   ├── export.py        #   open_in_browser(fig), heikin_ashi_candles(df) — keeps notebooks helper-free
│   ├── responsive.py    #   ResearchChart / price_chart / equity_chart (plotly-resampler; millions of pts)
│   ├── charts.py        #   static price+trades, equity+drawdown (HTML-export safe)
│   ├── asof.py          #   REWIND — AS_OF_JS, injected by open_in_browser(extra_js=). Arm the
│   │                    #     button, click a candle, and every gap, level, trendline and trade
│   │                    #     snaps back to what was knowable THERE. No kernel, no round trip.
│   │                    #     THE WHOLE TOOL IS ONE CLIP: every rewindable trace is None-separated
│   │                    #     segments, so "drop segments starting after k, clip the rest at k with
│   │                    #     y interpolated" handles every shape without special-casing -- a gap
│   │                    #     box ends at k (so one MITIGATED LATER still reads open, which is the
│   │                    #     point), a trendline keeps its slope, a later trade disappears.
│   │                    #     Candles and EMAs are deliberately left running: you study a bar to
│   │                    #     see what happened NEXT. Reads gd.layout.meta.rewind, so it carries no
│   │                    #     second copy of the data and is inert on a figure built without
│   │                    #     rewind=True. TWO BROWSER TRAPS, both found by clicking a real
│   │                    #     chart: plotly.py 6 base64-encodes numeric columns, so it reads
│   │                    #     through gd._fullData (gd.data[i].y is not always an array); and a
│   │                    #     CLICK returns a timezone-NAIVE string, which new Date() parses as
│   │                    #     LOCAL -- on a UTC+5 machine that cut the chart five hours early, so
│   │                    #     the cut is taken from the point's stored x instead. TWO CLIPS, not
│   │                    #     one: a LINE is truncated with y interpolated (moving x alone tilts a
│   │                    #     trendline), a BOX is CLAMPED -- a gap is a closed polygon and
│   │                    #     truncating it kept only the bottom edge, so every rewound gap
│   │                    #     collapsed into a flat line. Clamped, its height is always the gap's
│   │                    #     real height and only its length follows the click.
│   ├── studies.py       #   pattern_figure (candles + chart-pattern SHAPES: pivots, both trendlines,
│   │                    #     neckline and measured-move target, one legend toggle each).
│   │                    #     rewind=True names the clippable traces in layout.meta (everything
│   │                    #     except the candles and the EMAs) and draws S/R as ONE SEGMENT PER
│   │                    #     PRICE REVISION instead of one line at its final level -- the normal
│   │                    #     chart paints today's level across the whole window, which is a quiet
│   │                    #     lookahead when the point is to see what was known earlier.
│   │                    #     S/R ZONES ARE PER TIMEFRAME: sr_frames={"1h": df_1h} clusters
│   │                    #     levels on each frame IN ITS OWN BAR SPACE -- a level fitted on
│   │                    #     1h pivots is not the same level as one fitted on 5m pivots --
│   │                    #     and each layer gets a legend row, a cascade entry and a zone id
│   │                    #     (`4h·Z8`). WEIGHT and brightness carry the timeframe, coarsest
│   │                    #     heaviest, the same convention the trendlines use; the shade is
│   │                    #     ANCHORED on the chart's own layer so that layer looks identical
│   │                    #     whatever else is loaded (indexing by position made the same
│   │                    #     levels change colour when a 4h frame was added). An `S/R:`
│   │                    #     BUTTON ROW isolates one timeframe, like the gaps have,
│   │                    #     pattern_figure(trendlines=True) adds the standalone trendlines with an
│   │                    #     X at each break, on their own legend toggles; emas=, view_start=
│   │                    #     (warm-up is COMPUTED but not DRAWN), line_picker= (CASCADING dropdowns:
│   │                    #     the family menu rewrites the instance menu via the layout patch),
│   │                    #     tf_frames={"1h": df_1h} draws OTHER TIMEFRAMES' patterns and
│   │                    #     trendlines on one chart -- each layer detected in its OWN bar space
│   │                    #     and placed by TIMESTAMP, the only coordinate the frames share, so no
│   │                    #     index remapping. HTF EVENTS are stamped at the bar CLOSE (bars carry
│   │                    #     their open, so a 4h star at the open is a 4h lookahead); geometry
│   │                    #     stays at the open. WEIGHT encodes timeframe, colour still means
│   │                    #     direction, only the chart's own timeframe is visible at open. The
│   │                    #     THREE cascading menus: timeframe -> pattern -> occurrence, each
│   │                    #     narrowing the next (plotly menus do NOT scroll -- they clip, so a
│   │                    #     51-row list had unreachable entries). MULTI-SELECT is the LEGEND:
│   │                    #     choosing a family promotes its occurrences to their own entries, so
│   │                    #     you can keep 2 of 5. ACROSS families needs legend_instances=[...],
│   │                    #     which pins a family so every occurrence is a permanent, independent
│   │                    #     legend entry -- menu buttons are STATELESS, so a press replaces the
│   │                    #     selection and cannot accumulate ("name"=chart tf, "5m::name",
│   │                    #     "*::name", "5m", "all"). Caps bound DRAWN and LISTED together -- trimming
│   │                    #     only the menu left a legend saying (11) beside a dropdown of 9.
│   │                    #     max_picker_
│   │                    #     instances= / max_shapes_per_tf= bound it; mtf_emas=("1h",50,100,200)
│   │                    #     draws OTHER TIMEFRAMES' EMAs as STEPS that change only at their own
│   │                    #     bar close (an interpolated 1h EMA shows a number nobody could read);
│   │                    #     reuses a tf_frames frame for the history, warns when it cannot warm
│   │                    #     up rather than drawing an all-NaN line, hue=timeframe weight=period.
│   │                    #     A timeframe FINER than the chart is REFUSED unless you hand it a real
│   │                    #     frame (`ema_frames={"1m": df_1m}`): resample_ohlcv only works DOWNWARD,
│   │                    #     so a 1m EMA taken off a 5m chart was the 5m EMA under another name --
│   │                    #     identical on 100% of bars and 310 points out, with nothing saying so.
│   │                    #     mtf_ema_timeframes() names the frames a spec needs; emas_on=False starts
│   │                    #     them all `legendonly` (still listed, still individually clickable) and an
│   │                    #     "EMAs on/off" button pair does it live. legend.itemdoubleclick="toggle",
│   │                    #     because plotly's default ISOLATES the entry -- double-clicking an EMA
│   │                    #     used to take the isolated trade and its trendline off the chart with it;
│   │                    #     EVERYTHING IS NAMED: a line is `5m·S7` (trendline_table's `id`, one
│   │                    #     source so hover/legend/menu/trade all agree), a pattern `1h·wedge#2`,
│   │                    #     a trade `#12`; the id leads every hover, so a stop line says which
│   │                    #     trade owns it. Two more menus pick longs/shorts/winners/losers then
│   │                    #     ONE trade, showing that trade AND the line it broke and nothing else;
│   │                    #     a traded line is never dropped by the caps.
│   │                    #     The isolated trade draws entry/stop/target as THREE traces in the
│   │                    #     grouped view's colours (grey/red/green) -- one trace meant one colour,
│   │                    #     so a losing trade showed two red lines and the stop looked wrong-sided.
│   │                    #     A stop that walked to breakeven before being hit is tagged `sl@be`,
│   │                    #     because its exit sits at entry, nowhere near the stop drawn at entry.
│   │                    #     trades=/trade_tp_rr= overlay a backtest -- entry marker (side, R,
│   │                    #     close reason, trigger), entry level, stop and target running to the
│   │                    #     exit, and an exit marker coloured by outcome; y_from_price= scales the
│   │                    #     y-axis to the CANDLES, not to geometry that may be far taller.
│   │                    #     xaxis.autorangeoptions.minallowed is EPOCH MILLISECONDS, never a
│   │                    #     date string: plotly.js cannot compare a string there, autorange
│   │                    #     yielded NOTHING and the axis fell back to its empty default of
│   │                    #     2000-01-01..2001-01-01, so Autoscale threw the chart 26 years back.
│   │                    #     The `FVGs:`/`S/R:` rows open on the button that is actually TRUE
│   │                    #     (`none` when the layers start legendonly, -1 when mixed),
│   │                    #     zone_lookback_bars= (S/R zones from BOUNDED history, IN EACH
│   │                    #     LAYER'S OWN BARS -- warm-up makes them stale in a trend: 1 of 40
│   │                    #     near price vs 38 at a 2000-bar lookback), max_sr_picker=,
│   │                    #     and a STAR at each pattern's confirmation bar),
│   │                    #     add_pattern_overlays (candlestick + chart-pattern markers and their
│   │                    #     entry/stop/target levels; one legend group per pattern, default OFF),
│   │                    #     studies_figure (SQZMOM+WaveTrend), scenario_figure (fully-annotated setup
│   │                    #     chart: candles+EMA line+S/R zones+entries/stops/exits+gradient pane,
│   │                    #     ATR bands, HH/HL structure, RSI+divergence, HIGHER-TF EMA line+pane),
│   │                    #     fvgs=True draws FAIR VALUE GAPS as filled boxes, ONE TRACE PAIR
│   │                    #     PER LAYER however many boxes (a fill="toself" Scattergl takes any
│   │                    #     number of None-separated polygons) -- trace count is what decides
│   │                    #     whether the chart opens, since every picker button carries a
│   │                    #     visibility vector as long as the trace list. Drawn BEFORE every
│   │                    #     other layer so the boxes sit behind the lines, and tagged
│   │                    #     meta="fvg". Hover lives on an invisible centroid marker, because a
│   │                    #     filled polygon only reports its outline. fvg_frames= is for
│   │                    #     timeframes you want GAPS from but not patterns or trendlines --
│   │                    #     kept out of tf_frames on purpose, since that is what gets fitted
│   │                    #     and traded. max_fvg_picker= isolatable gaps PER TIMEFRAME PER
│   │                    #     DIRECTION (one shared pool was drained by the coarsest layer),
│   │                    #     biggest-first, capped before drawing so legend and menu agree.
│   │                    #     An `FVGs: all | 4h | 1h | 15m | 5m | 1m | none` BUTTON ROW
│   │                    #     isolates one timeframe's gaps in a single press -- `restyle`
│   │                    #     takes a LIST of visibilities beside a list of trace indices, so
│   │                    #     one button shows one layer and hides the rest, which a single
│   │                    #     style value cannot express. Gap timeframes may be FINER than the
│   │                    #     chart (1m gaps on a 5m chart): each layer is detected on its own
│   │                    #     real bars, so nothing is resampled and the EMA trap cannot recur;
│   │                    #     zoom_to_trade, CROSSHAIR_JS (full-height cross-pane cursor)
│   └── heatmaps.py      #   monthly-returns, hour×weekday, parameter-sweep heatmaps
├── scenarios/           # VERSIONED, COMPOSABLE setups: Scenario × Stop × TP × Secondary × params
│   ├── base.py          #   Scenario/StopMethod/TPMethod/SecondaryMethod ABCs + registries +
│   │                    #     Preset (JSON I/O) + compose()/run()/run_pieces() -> ScenarioRun
│   ├── entries.py       #   ENTRY FAMILIES as primary triggers, each registered long AND short:
│   │                    #     level_event (break/fade/sweep/retest of any level) · bos/choch (zigzag) ·
│   │                    #     vol_thrust · squeeze · wt_cross · rsi_div · vwap_rev · grad_extreme ·
│   │                    #     time_entry = NULL BENCHMARK (enter every N bars, no signal — any entry
│   │                    #     that can't beat a metronome on the same exits contributed nothing) ·
│   │                    #     candle_pattern (named subset or the whole library via its score) ·
│   │                    #     chart_pattern (fires on the neckline break, not on shape completion) ·
│   │                    #     mtf_trendline = a FITTED trendline from ANY timeframe giving way
│   │                    #     (long through resistance, short through support). Three combinable
│   │                    #     conditions: tl_close_n closes beyond / tl_full_m whole candles beyond
│   │                    #     / tl_touch_k prior touches / tl_dir_n candles of the TRADE'S OWN
│   │                    #     COLOUR (red for a short, green for a long -- NOT the same test, a
│   │                    #     candle can close through a support and still be green; tl_dir_src =
│   │                    #     "chart" the execution frame or "line" the line's own timeframe) /
│   │                    #     tl_dir_full_n n candles BOTH the right colour AND wholly clear of the
│   │                    #     line (== tl_full_m + tl_dir_n under "all", but survives "any"),
│   │                    #     joined by tl_combine all|any. EACH takes
│   │                    #     a DICT keyed by timeframe ({"5m":4,"15m":5,"*":2}) via tl_per_tf(),
│   │                    #     because two 15m closes is half an hour and two 1m closes is two
│   │                    #     minutes; experiments/trendline_wait_search.py sweeps it. The level
│   │                    #     is HELD for the confirmation window because add_trendlines retires a
│   │                    #     line the moment it breaks, and tl_extend (default True) carries it
│   │                    #     along its OWN SLOPE rather than flat -- a rising support keeps rising,
│   │                    #     line the moment it breaks -- without that, n>1 fired ZERO times on
│   │                    #     the base timeframe. trendline_trigger() labels each trade with WHICH
│   │                    #     timeframe fired, since the entry is an OR across them.
│   │                    #     tl_confirm_n MOVES THE ENTRY OFF THE BREAK CANDLE: a break bar that is
│   │                    #     huge next to its neighbours and closes a whole new body beyond the
│   │                    #     line looks decisive and is the shape that gets swept, so its close is
│   │                    #     the worst price of the move. Within n CHART candles one of the trade's
│   │                    #     OWN COLOUR must close beyond the WHOLE trigger candle (below its low
│   │                    #     for a short); the entry is on that close, and no confirmation means no
│   │                    #     trade. It is also the ONLY way an entry can happen later than the
│   │                    #     break -- tl_close_n asks for n CONSECUTIVE closes, a different
│   │                    #     condition. The line's ID is held for the trigger window PLUS the wait,
│   │                    #     or a delayed entry lands where the id is gone and every trade reads `?`
│   ├── support_break.py #   support_break_short v1.0 core / v1.1 +HH-HL structure / v1.2 no-structure
│   │                    #     / v1.3 nearest-trough — short below a down-sloping smoothed 200-EMA on a
│   │                    #     support break; entry_mode = break_close|wait_n|n_red|pullback (Q2);
│   │                    #     LEVEL B exposed as `break_level` (latched per episode) + break_mode =
│   │                    #     close_below|full_below; candle_src = real|ha for every candle test
│   ├── stops.py         #   pct / swing_high / prev_high / leg_high / high_of_n / zone_top / atr /
│   │                    #     pattern_stop (the chart pattern's own invalidation level)
│   │                    #     (Q1: where the stop goes — see the LAG note in §12)
│   ├── takeprofits.py   #   rr / fixed_pct / trailing / opposite_candle / n_green_exit /
│   │                    #     back_above_ema / next_zone / rr_partials / be_ladder (breakeven and/or
│   │                    #     partial at N R, then a target and/or a trail) / pattern_target
│   │                    #     (the chart pattern's measured move)  (Q3; signal- or cfg-based)
│   └── secondaries.py   #   POST-ENTRY rules: shaky_trend_bail@v1 (arm when price closes above/touches
│                        #     the EMA, then bail on the first opposite candle IF in profit) ·
│                        #     entry_retest_bail@v1 = arm from the trade's OWN path: sec_touch_n bars
│                        #     that traded back THROUGH its entry price, or sec_wrong_m candles
│                        #     formed WHOLLY on its losing side. Neither is expressible as a global
│                        #     signal array (the level is that trade's entry), so the KERNEL counts
│                        #     them per trade -- o_touch/o_wrong beside o_armed. sec_min_pct is the
│                        #     move the exit must clear: the engine's gate is "any favourable move",
│                        #     which on 5m exited BELOW COST on 3 of 10 bail-outs.
├── research/            # NOTEBOOK-FACING ORCHESTRATION — the wiring a notebook would
│                        #   otherwise spell out in a cell. A notebook here reads as
│                        #   SETTINGS -> ONE CALL -> A CHART, so the loading, the column
│                        #   assembly and the scenario plumbing live here where they are
│                        #   imported, tested and versioned. It CALLS indicators, scenarios
│                        #   and viz; it holds no strategy logic, no indicator maths, no
│                        #   drawing, and NO DEFAULT for any setting a config owns — a value
│                        #   stated in two places is the 8a8b8de bug. Settings arrive as the
│                        #   config MODULE, read through _get(cfg, name, default) so a config
│                        #   predating a setting degrades instead of raising.
│   └── fvg_lab.py       #   fvg_lab(CFG) -> FvgLab: loads, detects gaps on every timeframe,
│                        #     and runs the sim TWICE (with the veto and without, same frame
│                        #     and params) because a veto-only run cannot distinguish a filter
│                        #     that removed losers from one that removed trades at random.
│                        #     FOUR frame sets, kept apart on purpose: LINE frames are fitted
│                        #     and TRADED, GAP frames are drawn and may veto, S/R frames are
│                        #     clustered and drawn, EMA frames feed mtf_emas. One setting per
│                        #     job, so adding 4h to see its levels cannot start trading its
│                        #     trendlines -- and MTF_EMAS naming a timeframe none of the others
│                        #     mention makes the lab LOAD it, rather than the user having to add
│                        #     "1m" to an unrelated setting to satisfy the finer-than-chart
│                        #     refusal.
│                        #     AS_OF is JUST AN EARLIER END: every detector here is already
│                        #     causal, so stopping the data at a bar gives the exact
│                        #     point-in-time answer with no new logic. `.end` reports what was
│                        #     actually used. Slower than the browser tool and not clickable,
│                        #     but nothing is filtered and it works inline.
│                        #     .trades/.all/.blocked/.gaps/.end/.census()/.summary()/.figure()
├── optimize/
│   ├── grid.py          #   run_grid (bounded-parallel sweep; fast array path; backend threading/loky)
│   └── search.py        #   optuna_search (optional; Bayesian)
└── reporting/report.py  #   summary / print_summary / to_html (standalone offline report)

experiments/             # INFERENCE layer — find best settings for an idea (SEPARATE from core)
├── base.py              #   Experiment (strategy_space + cfg_space, objective, run() -> results/)
├── fvg_veto_search.py    # does an unfilled FVG behind a break say skip it? 24 settings x 6
│                        #   windows, scored by R PER TRADE (a veto only removes trades, so total
│                        #   R falls even when it does nothing) with the top winner trimmed, and
│                        #   the chance baseline printed. Answer: 0 clear all 6, 4 clear >=4 where
│                        #   chance predicts 8.2. Below chance -- see docs/WHAT_WORKS.md
├── trendline_wait_search.py  # how many candles to wait before trading a break, PER TIMEFRAME:
│                        #   6 windows over 5.5 months, scored by WINDOWS POSITIVE not total R,
│                        #   with the chance baseline printed beside it
├── ema_mtf.py · session_timing.py · exit_design.py   # worked examples (each with a description)
└── README.md
scenarios/presets/       # tuned, named, versioned scenario configs (JSON; git-tracked, reloadable)
notebooks/               # nb18_config.py = notebook 18's SETTINGS, owned by the user. The
                         #   notebook is regenerated from a script; the script used to hardcode
                         #   every value, so adding one setting silently re-imposed the others
                         #   (8a8b8de changed TF/LINE_TFS/WARMUP_START unasked). The notebook now
                         #   READS settings and states none — tests/test_notebook_config.py fails
                         #   if any assignment creeps back into a cell. NEVER regenerate that file.
                         #   A setting APPENDED there is invisible to an editor tab the user
                         #   already had open — saving that stale buffer writes the old tail
                         #   back, which is how EMAS_ON vanished and the notebook died on a
                         #   NameError. So cell 2 carries SETTING_DEFAULTS: a setting the
                         #   config predates falls back and prints the line to paste in.
                         # 20_rewind + nb20_config.py = notebook 19's chart plus the REWIND
                         #   tool. The config is a verified COPY of 19's (a test fails if any
                         #   value drifts) plus REWIND and AS_OF, so the point-in-time view is
                         #   the only difference between them. Notebook 19 is untouched.
                         # 19_fvg_viewer + nb19_config.py = FAIR VALUE GAPS from several
                         #   timeframes on one chart, plus a COPY of notebook 18's sim with the
                         #   FVG veto added and a with/without comparison. SEVEN code cells and
                         #   36 lines -- all wiring is quant/research/fvg_lab.py, and the tests
                         #   cap it there (no cell may define a function; 9 cells / 60 lines).
                         #   A copy rather than a change to 18, so nothing already works can
                         #   regress; the trading
                         #   settings are identical, which makes the veto the only difference.
                         #   FVG_VETO_TFS excludes the chart's own timeframe by DEFAULT: 5m
                         #   produces ~2,000 gaps with ~96 live at once, so including it makes
                         #   the veto a constant (it blocked all 8 trades)
                         # 18_trendline_trading_sim = TRADES multi-timeframe trendline breaks on
                         #   1m (entry rule, previous-high/low stop, BE@1R + 3R target), with the
                         #   trades on the chart and the cost floor measured on them
                         # 17_multi_timeframe_patterns = ONE chart, patterns from SEVERAL
                         #   timeframes (tf_frames=); the window_holds diagnostic says up front
                         #   whether a timeframe's shapes can even fit on screen
                         # 16_trendline_research = reads experiments/trendline_settings_search
                         #   (144 settings x 3 TFs x 2 instruments x 9 windows, 4 stages)
                         # 15_tuned_pattern_viewer = all 22 chart patterns at their CALIBRATED
                         #   per-pattern settings (quant.indicators.TUNED_SPECS) + trendlines, one toggle each
                         # 14_pattern_viewer = chart-pattern SHAPES + S/R zones, toggled from the legend
                         # 01_research_cycle · 02_inference_experiments · 03_ema_cross_study (Q1-Q9)
                         #   · 04_trend_runner_leverage · 05_simple_1m_ema_breakout ·
                         #   06_ema_breakout_candles · 07_rule_builder · 08_sr_zones_view ·
                         #   09_squeeze_momentum · 10_scenario_lab (versioned setups + annotated chart)
scripts/fetch_gold_dukascopy.py   # resumable true-spot XAUUSD fetch (1m/5m/15m/1h/4h)
examples/gold_ema_demo.py · tests/ (10 files) · docs/ · data/ (cache, git-ignored)
```

### Scenario framework (`quant/scenarios`) — versioned, composable setups

A scenario run = **`Scenario`** (entry logic) × **`StopMethod`** × **`TPMethod`** ×
**`SecondaryMethod`** (optional post-entry rule) × **params**, each piece named + **versioned**
(`name@version`) and registered, bundled into a **`Preset`** (JSON under `scenarios/presets/`) that
reproduces the exact backtest on any symbol/TF. Reuses the engine + `RuleStrategy`/conditions + S/R
zones + the EMA-gradient indicator. `run_pieces(df, scen, stop, tp, secondary=…, params=…, cfg=…)` or
`run(preset)` → `ScenarioRun` (`.result/.trades/.stats/.df/.params`); draw it with
`quant.viz.scenario_figure(run)`. Signal-based TPs (`next_zone`, `opposite_candle`, `back_above_ema`)
use `Signals.exit_short`; cfg-based TPs (rr/trailing/partials) use existing `BacktestConfig`. Only the
**secondary** needed an engine change (per-trade `o_armed` + in-profit gate — see §5); it is
flag-guarded, so runs without a secondary stay byte-identical. Each piece `prepare()`s what it needs
(e.g. `next_zone` adds S/R zones if the scenario didn't), so any combination is runnable.

**Level B** (the level whose break triggers the entry) is exposed as a `break_level` column, latched
at the first break of each episode — an S/R support steps down the moment price closes through it, so
the live value stops naming the level that broke. `scenario_figure` draws it per trade (amber dashed,
own legend group). `break_mode` = `close_below` (default) | `full_below`.

True absolute multi-zone partials (`zone_partials`) would need a per-trade TP-ref kernel branch
(mirror of the `sl_ref_short_col` path) — deferred, not yet built.

## 5. The engine (`quant/engine`) — capabilities

`BacktestConfig` (every field documented in its docstring — do `help(BacktestConfig)`):

- **Stop loss** (`sl_mode`): `none` | `entry_pct` | `price_abs` | `ref_col` (structure level from a
  column, e.g. `sl_ref_long_col="swing_last_low"`, with `sl_buffer_pct`, `sl_max_ref_risk_pct`,
  fallback). **Trailing** (`trail_mode` `pct`/`price_abs`, ratchets on high-water mark).
- **Take profit**: `take_profits=(TakeProfit(mode, value, close_pct, move_stop_mode, move_stop_value), …)`
  laddered/partial; or convenience `tp_mode`/`tp_value`. Modes: `entry_pct`|`price_abs`|`rr`.
  **TP prices are FIXED at entry** (rr uses the original stop distance) — intentional, documented.
- **Sizing** (`sizing_mode`): `cash` | `risk_pct_equity` (compounds) | `risk_amount` | `lots`.
- **Costs (Exness-style):** `spread` = fixed bid/ask WIDTH in price units (buy at ask, sell at bid;
  cost = spread × qty = spread × contract_size × lots — scales with volume automatically),
  `commission_per_lot` (per side), `fee_bps` (% per side), `slippage_bps`.
- **Leverage/margin (opt-in `margin_enabled`):** `leverage`, `contract_size` (gold=100 oz/lot),
  used/free margin, free-margin gate on entry, **stop-out liquidation** at `stop_out_level`% margin
  level (close reason `margin_call`). Non-margin path uses `allow_leverage`/`max_notional_pct` cap.
- **Fills:** `max_open_trades` (multi-position), `allow_short`, `intrabar_priority`
  (`stop_first`/`take_profit_first`), force-close at end.
- **Per-rule ("leg") exits + stops:** `Signals` may carry `entry_leg_*`/`exit_legs_*` (2-D
  [n_legs, n]) and `stop_specs` (per-leg); the kernel tags each trade with the rule that opened it
  (`o_leg`) and closes it on THAT rule's exit and (if set) THAT rule's stop method. `n_legs==1` +
  `per_leg_stops==0` take the original code paths — byte-identical (golden tests pass), no per-bar
  overhead. Populated automatically by `RuleStrategy` only when rules use different exits/stops.
- **Secondary ("alt") exit — per-trade armed state:** `Signals` may carry `exit_arm_*` / `exit_alt_*`
  (+ `alt_replace_tp`). `exit_arm_*` arms an OPEN trade on any bar **after its own entry bar** (the
  kernel tracks `o_armed` per trade; arming runs before entries open, so a trade can neither arm on
  its entry bar nor inherit arming from before it existed). Once armed, `exit_alt_*` closes it **only
  while it is in profit** (measured at the exit fill). `alt_replace_tp` additionally drops the TP
  ladder for armed trades; the stop/trailing are never affected. Expresses "since THIS trade opened",
  which a global signal array cannot. With no `exit_alt_*` the whole branch is skipped —
  byte-identical (golden tests pass).
- **Take-profit levels may close 0%** and still move the stop (`TakeProfit(mode, value, close_pct=0,
  move_stop_mode='breakeven')`) — that is how a plain "go to breakeven at 1R" is expressed. Such a
  level books no fill; levels that DO close something are unchanged (golden parity).
- **Time stop (`max_bars_in_trade`)**: close any trade still open N bars after entry, at that bar's
  close. Checked AFTER stop/TP and exit signals, so a trade closing for a real reason keeps that
  reason. `0` = off and the whole block is skipped — byte-identical (golden tests pass).
- **Self-arming (`arm_touch_n` / `arm_wrong_m` on `Signals`)**: a trade arms itself from its OWN
  path — bars that traded back through its entry price, or candles formed wholly on its losing side
  — because a global array cannot name a per-trade level. `alt_min_pct` is how far price must have
  moved your way before the secondary may close, since "any favourable move" routinely exits for
  less than the round trip. All three default to 0 and the block is skipped — byte-identical.
- Close-reason codes: `signal`, `stop_loss`, `take_profit`, `forced_close_end`, `margin_call`,
  `alt_exit` (closed by the secondary rule), `time_stop`.

`SimResult` = `trades` (DataFrame; incl. per-trade `stop_price`), `equity_curve`
(t/equity/open_trades/drawdown), `stats` (dict),
`elapsed_s` (kernel time).

## 6. Key conventions & invariants (DO NOT BREAK)

- **Time:** `open_time` tz-aware UTC (cache); `t` = display-tz copy used by sim/plots/filters.
- **OHLCV** columns lowercase `open/high/low/close/volume`.
- **Indicator columns** named (e.g. `ema_50`, `rsi_14`, `macd`, `st_dir`, `swing_last_low`); MTF
  columns prefixed `{tf}__` (e.g. `5min__ema_50`).
- **Signals are the single representation** for manual runs AND sweeps (numpy bool arrays). Never
  express a strategy twice.
- **Anti-lookahead:** HTF features shifted +1 HTF bar before as-of merge (`data/timeframe.py`);
  swing/pivot levels exposed only after confirmation (`indicators/structure.py`). Preserve this.
- **Golden parity:** the numba kernel was validated at exact per-trade parity vs the original
  simulator; locked by `tests/test_engine_golden.py`. **Zero-cost / non-margin defaults must stay
  byte-identical** — when extending the kernel, keep those paths unchanged and re-run golden tests.
- **Performance:** vectorized numpy / numba over Python loops; never `df.iloc[i]` in hot paths; keep
  the sweep path array-native (no per-combo DataFrames); hoist config-derived kernel args out of loops.

## 7. Data layer specifics

- `get_ohlcv(symbol, tf, start, end, source="binance", market="spot", tz="UTC", refresh=False)`.
- Sources: **`binance`** (crypto + PAXG/XAUT gold proxies) and **`dukascopy`** (true spot XAUUSD/FX
  from 2003; `pip install "quant[data]"`; live fetch verified working). Add a provider by
  implementing `DataSource.fetch` and routing ranges through `quant.data.cache.incremental_fetch`.
- Cache file: `data/{source}_{market}_{SYMBOL}_{tf}.parquet` (+ `data/.partials/` for Binance
  checkpoints). Incremental (only missing ranges fetched); reads use column+date pushdown.
- MTF: `build_mtf(df, {"5min": lambda d: add_emas(d,[50,200])})` → `5min__ema_50` on the base grid.

## 8. Optimization & experiments

- **Sweeps:** `quant.optimize.run_grid(df, StrategyCls, grid, cfg, valid_fn=…, keep_stats=…)` —
  precomputes indicators once, array-native stats, bounded parallel (`n_jobs=CPU−2`, threading).
  `backend="loky"` exists but is usually **slower** here (pickles the frame). **Optuna** via
  `optimize.search.optuna_search` for large continuous spaces.
- **Speed levers for millions of combos:** subset-then-refine (shorter window / coarser TF first),
  then confirm on full 1m; use Optuna over brute grid. Rough: full-year 1m ~130 ms/combo threaded;
  3-month window ~30 ms/combo.
- **Experiments/inference** (`experiments/`): declare `strategy_space` (strategy fields incl
  `session`/`hours`) + `cfg_space` (BacktestConfig fields incl SL/TP/leverage), an objective, and a
  `description`. `Experiment.run()` sweeps (grouping by strategy so signals compute once, cfg
  variants reuse them), ranks, and writes `experiments/results/<name>/` (results.csv, best.json,
  report.md). **Never put experiment logic in `quant/` core.** See `docs/EXPERIMENT_GUIDE.md`.

## 9. Environment

- Python **3.13**, Windows (PowerShell primary; Bash available). Run from repo root with
  `PYTHONPATH=.` or `pip install -e .`.
- **The Jupyter kernel IS the repo interpreter — keep it that way.** A user-level kernelspec at
  `~/AppData/Roaming/jupyter/kernels/python3` points `python3` at **C:\Python313\python.exe**,
  overriding the system one that shipped with Python 3.9. Re-register with
  `python -m ipykernel install --user --name python3 --display-name "Python 3 (3.13 - quant)"`.
  Before this, notebooks ran 3.9 / pandas 1.5.3 / plotly 5.14.1 while the tests ran 3.13 /
  pandas 2.3.3 / plotly 6.3.1 — so the notebooks exercised a configuration nothing tested, and
  `xaxis.autorangeoptions` (plotly 5.19+) imported fine here and raised there. Output was proved
  IDENTICAL across both to six decimals, so the upgrade changed no number; it is ~2.5x faster to
  build a chart. The autorangeoptions call is still wrapped, and the geometry is still clipped to
  the window, so an older kernel degrades rather than breaks. ALWAYS execute a notebook with
  `jupyter nbconvert --execute` before believing a viz change works — and check nbconvert actually
  WROTE, since a failed run leaves the previous outputs in place and an error scan then reads clean.
- Installed & used: pandas, numpy, pyarrow, polars, **numba**, plotly, **plotly-resampler**,
  tsdownsample, joblib, scipy, tqdm, requests, dukascopy-python. Optional: optuna.
- Data cached: BTC/PAXG 1m ≈ 742k rows (2025-01 → 2026-05), 5m/15m/1h derived. Fits in RAM; the
  challenge is compute, solved by the JIT engine.

## 10. How to run

```python
from quant.data import get_ohlcv
from quant.strategies import EmaRibbon
from quant.engine import BacktestConfig
df  = get_ohlcv("PAXGUSDT", "1m", start="2025-06-01", end="2026-05-31", tz="UTC")   # or source="dukascopy", "XAUUSD"
cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, spread=0.20, exit_enabled=True,
                     sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
                     sizing_mode="risk_pct_equity", sizing_value=1.0)
res = EmaRibbon(fast=50, slow=200, confirm_n=5).backtest(df, cfg)
print(res.stats)
```
- Notebooks: `01_research_cycle` · `02_inference_experiments` · `03_ema_cross_study` ·
  `04_trend_runner_leverage` · `05_simple_1m_ema_breakout` · `06_ema_breakout_candles_demo`
  (candle chart + HA toggle) · `07_rule_builder` (declarative
  RuleStrategy) · `08_sr_zones_view` (200-MA strategy + causal S/R zone overlay) ·
  `09_squeeze_momentum` (SQZMOM + WaveTrend + crosshair) · `14_pattern_viewer` (one shared
  detector config) · `15_tuned_pattern_viewer` (per-pattern calibrated settings) · `10_scenario_lab` (versioned
  Scenario×Stop×TP setups, fully-annotated `scenario_figure`, variant sweep answering the 3 Qs, Preset I/O).
- CLI: `quant backtest --symbol PAXGUSDT --tf 1m --start 2025-06-01 --strategy ema_ribbon --params '{"fast":50,"slow":200,"confirm_n":5}'`
- Demo: `python examples/gold_ema_demo.py` · Tests: `python -m pytest tests/`
- New strategy = one dataclass in `quant/strategies/` (`prepare` + `build_signals`); sweep with `run_grid`.

## 11. Tests (what's covered)

`tests/`: `test_engine_golden.py` (numeric regression vs legacy-validated engine),
`test_leverage.py` (margin/stop-out/lots), `test_costs.py` (spread/commission — incl the EUR/USD
$12 example), `test_signals.py` (primitive correctness), `test_strategies.py` (all 7 run),
`test_analytics.py` (attribution + reporting), `test_viz.py` (responsive chart resamples; PER-TIMEFRAME S/R ZONES — each timeframe gets its own layer, weight strictly increases with it, the chart's own layer is byte-identical whether or not other layers are loaded, the timeframe button row isolates with a PER-TRACE visibility list, and every zone is named `5m·Z3` and hoverable off a midpoint marker),
`test_dukascopy.py` (provider mapping + incremental cache, mocked fetch), `test_builder.py`
(RuleStrategy conditions, `after`-sequencing, per-rule leg exits + stops with single-leg golden
parity, causal S/R zones, MTF bias, big-move, configurable MA, tagged indicator handles),
`test_scenarios.py` (breakdown conditions, every scenario version runs short-only with stops above
entry, full stop×TP + entry-mode matrix, LEVEL-B exposure/latching + both break_modes, the pivot-lag
property (leg_high <= swing_high), Heikin-Ashi candle source, engine-level secondary semantics
(in-profit gate, arm-only-after-entry, sec_replace_tp), Preset JSON round-trip incl. `secondary`,
shipped presets load+run), `test_patterns.py` (candlestick shapes from hand-built candles, hammer vs hanging man separated by trend, three-candle patterns flagged on their LAST bar, gap patterns impossible without gaps, score columns consistent, **CAUSALITY by truncation** for both libraries — running on the first k bars must reproduce the full run's first k results — chart patterns confirm only past the neckline with a valid stop/target geometry, and the pattern scenarios/stop/TP run end to end), `test_levels_entries.py` (reference-level CAUSALITY — prev-day/week/session
levels come from the completed prior period, running session extremes are expanding not final, the
opening range stays hidden until it completes, VWAP starts at the first bar's typical price; exact
`level_events` classification; zigzag running pivots; every entry family runs long+short with stops on
the correct side; time stop caps `bars_held`, is byte-identical when off, and never overrides a real
close reason), `test_fvg.py` (fair value gaps — the band's orientation (`top` is the higher edge, the bug that inverted every bearish gap), publication from the THIRD candle, CAUSALITY BY TRUNCATION, the three mitigation thresholds, the ATR gates, the live-count columns, the higher-timeframe shift, the two veto modes being exact mirrors, and that a whole layer of boxes costs one trace pair while leaving every existing trace untouched), `test_trendline_sim.py` (multi-timeframe trendline entries and the trade overlay — a higher timeframe's level stays invisible until its bar closed, the confirmation rule behaves the same on every timeframe, an isolated trade keeps its EMAs, and an EMA finer than the chart is refused rather than resampled), `test_rewind.py` (the point-in-time view — rewind=False is inert, the deny-list offers every trace WITH A BIRTH and never the candles or EMAs, rewind keeps every S/R revision and each one stops where the next starts, AS_OF is a truncation not a filter, and the injected script is inert on a figure that was not built for it), `test_rulespec.py` (the one-line rule language), `test_notebook_config.py` (no notebook cell may ASSIGN a setting). **534 total.**

## 12. Known limitations / honest caveats

- Data providers: only **binance** + **dukascopy** implemented (OANDA/CSV/metals-API planned).
- Leverage: models used/free margin + stop-out, but **no swap/overnight financing**; stop-out
  liquidates **all** positions (not partial); spread is a fixed width (no dynamic/time-varying spread).
- **Pivot levels carry a confirmation lag** (by design, anti-lookahead): `add_swings`/`swing_high@v1`/
  `prev_high@v1` need `lr` bars to the RIGHT of a pivot before it exists, so the level in use is always
  ≥ `lr` bars old and can sit surprisingly far from the entry. Lowering `lr` does not remove it. Use
  `leg_high@v1` (running max since price closed below the EMA) or `high_of_n@v1` for a lag-free level.
  Worked example in `docs/SESSION_NOTES.md`.
- No **walk-forward / out-of-sample** automation yet (the experiment guide describes doing it manually).
- Responsive chart interactivity needs a **live Jupyter kernel** (static HTML export keeps only the
  initial view); static charts are the fallback.
- Very large sweeps ("millions") realistically need Optuna + subset-then-refine, not brute force on
  full-year 1m (see §8).

## 13. Roadmap (next candidates)

OANDA practice provider; CSV/TradingView import; walk-forward/OOS tooling; swap/rollover + partial
stop-out; (if needed) per-strategy fused-numba signal generation for extreme sweeps; add a `LICENSE`.

## 14. Working agreements for AI sessions

- Never commit/print secrets; secrets come only from `.env`. If exposed, advise rotation (user's action).
- Keep `tests/` green; keep golden parity (zero-cost/non-margin paths byte-identical); add
  parity/golden tests when extending the kernel.
- Preserve anti-lookahead + the incremental cache.
- Put experiment-specific logic in `experiments/`, never in `quant/` core.
- Update this file + README + docs when architecture changes. Commit in logical units; push only
  when asked (remote `origin` → GitHub `main`).
- **Read [`docs/NOTEBOOK_GUIDE.md`](docs/NOTEBOOK_GUIDE.md) before touching a notebook or a
  config.** Two rules it enforces with tests: a cell may never ASSIGN a setting, and a cell may
  never DEFINE a function — orchestration belongs in `quant/research/`. It carries the table of
  where each kind of code goes, how to lay out a config file, and the value-preservation check
  to run after re-arranging one.
- **Never change a setting the user did not ask you to change.** Notebooks here are regenerated from
  build scripts, and a build script that states values will re-impose them on every rebuild — which
  is how `TF`, `LINE_TFS` and `WARMUP_START` were silently altered in 8a8b8de. Settings belong in a
  file the generator cannot write (`notebooks/nb18_config.py`); when adding a new one, APPEND it
  there and leave every existing line alone.
