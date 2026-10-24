# The chart guide — every layer, every pattern, and the setting behind each

The research charts (`notebooks/23_break_confirmation_news.ipynb`, `lab.ipynb`,
`market_view.ipynb`, notebooks 18–22) are all drawn by one function, `quant.viz.pattern_figure`,
so a layer looks and behaves the same everywhere. This page lists what can be on a chart, which
setting draws it, and the control that switches it on the chart itself. The names below are the
exact strings the settings take.

## The layers

| layer | what it shows | setting (in the notebook's config) | on the chart |
|---|---|---|---|
| candles | price | `TF`, the window | — |
| trendlines | fitted support / resistance lines per timeframe, an ✕ where one broke | `LINE_TFS`, `TRENDLINE`, `SHOW_TRENDLINES` | legend; the timeframe → line menus, top centre |
| S/R zones | horizontal levels price has turned at, per timeframe | `SR_TFS`, `SR_SENSITIVITY`, `SR_MAX_ZONES`, `SR_BREAK_BARS`, `SR_STYLE`, `SR_ROLES`, `SR_COLOURS` | `S/R:` button row, top left |
| fair value gaps | three-candle imbalances, boxed until filled | `FVG_TFS` and the `FVG_*` settings | `FVGs:` button row, top left |
| EMAs | moving averages of the chart's and other timeframes | `EMAS`, `MTF_EMAS`, `EMAS_ON` | `EMAs on / off`, top right; legend |
| chart patterns | the shapes below, with pivots, both lines, neckline and target | `CHART_PATTERNS`, `PATTERNS_TUNED` | pattern menus, top centre; legend |
| candlestick patterns | a ▲ under / ▼ over the bar a pattern completes on | `CANDLE_PATTERNS` | legend |
| news | economic releases along the bottom; hover for forecast, actual, meaning | `NEWS`, `NEWS_SHOW`, `NEWS_TZ` | `News:` dropdown, bottom right |
| previous period high / low | PDH / PDL, PWH / PWL, PMH / PML, PYH / PYL as dotted steps | `PERIOD_LEVELS`, `PERIOD_LEVELS_ON`, `LEVELS_TZ` | `Levels:` dropdown; legend |
| market sessions | each market's hours as a ribbon along the top; optional line at each open | `SESSIONS`, `SESSIONS_ON`, `SESSION_OPEN_LINES` | `Sessions:` dropdown; legend |
| trades | entries, stops, targets, exits with their tags | the strategy settings; `EXIT_LABELS` | trade menus, top centre |
| rejected setups | breaks that did not trade, and why | `SHOW_REJECTED` | legend |
| rewind | everything back to what was known at a clicked candle | `REWIND` | `Rewind` button, bottom left |

Every layer that is not asked for is not drawn, and a chart built without the newer settings is
exactly what it was before they existed.

## Support and resistance

A zone is a price where several swing highs or lows clustered: `min_touches` pivots within the
clustering tolerance (`cluster_pct` of price) of each other. `SR_SENSITIVITY` picks tuned values
per timeframe: `"tight"` (fewer, sharper), `"normal"`, `"loose"` (more context).

* **`SR_STYLE = "band"`** draws the zone as the band of prices it stands for, its clustering
  tolerance widened to every pivot that joined it, instead of one line at the average. The S/R
  study found a level 1–3 ATR away holds about as well as the line itself, so zones mark areas.
* **`SR_ROLES = True`** colours a zone by what it is at each moment: **blue = support** while
  price closes above it, **orange = resistance** while price closes below. The colour changes at
  the close that crossed it, so a broken support turning into resistance shows as a change of
  colour. Blue and orange are used so they never read as the gaps' green / red.
* **`SR_BREAK_BARS`** retires a zone once price has closed beyond it that many bars in a row (a
  close back resets the count). It can be set per timeframe: `{"5m": 12, "15m": 4, "1h": 2,
  "4h": 2}` is about an hour on each. `None` keeps the tuned table, where 5m and 15m never
  retire. Measured on BTC 5m, 1 Sep – 19 Oct 2026: the 5m layer drew 63 zones with 16 live at a
  time, almost all of them levels price had already cut through. Retired after an hour beyond,
  the 1h and 4h layers keep about five live zones each, and the 5m layer keeps very few, because
  a 5m level rarely survives that long.
* **`SR_MAX_ZONES`** caps the live zones per timeframe; when full, the one farthest from price
  goes. A zone that was retired or pushed out ends where it did, it does not run on to the right
  edge.
* **Zones from the warm-up.** A zone still live when the chart opens is drawn from the first
  candle, and one that died during the warm-up is not drawn at all. (With `REWIND` on, those dead
  ones used to be drawn from the first candle *backwards* into the warm-up, a fan of levels that
  all stopped at the view start and looked as though the chart had failed to carry them forward.)

## Trendlines

A trendline is fitted from the zigzag's swing highs (resistance) or lows (support): the line that
touches the most of them, at least `min_touches`, within `line_tol_atr` ATR. The settings are the
`TRENDLINE` dict in the config; two fits are available.

* **`fit="ransac"`** (the default, and what every recorded result was measured with) checks only
  the swings against the line and then refits it by least squares. A candle the zigzag never
  called a swing, such as an outside bar, can cross it, and the touches float on either side. On
  BTC 15m, Sep 2026, line S15 had a wick 2.2 ATR through it and touches at −0.40 / +0.17 / +0.46 /
  −0.23 ATR; 17 of the window's 55 15m lines had a candle close through them before they were
  even drawn.
* **`fit="strict"`** draws the line exactly through two real swing extremes and keeps it only if
  every candle from its first touch to the moment it is drawn is clear: no close beyond it, and no
  wick beyond it by more than `wick_tol_atr` (0.15). Pair it with `line_tol_atr=0.3`, so a third
  swing counts as a touch only when it visibly sits on the line. It finds fewer lines (15m: 34
  instead of 55 over that window) and every one of them is clean. Over 19 quarters it changes
  notebook 21's results by nothing measurable (−0.02R a trade, t −1.0): a cleaner chart, not an
  edge.

* **`fit="bestfit"`** (opt-in, researched, not used anywhere yet) is the line an eye draws through
  swings that are not quite collinear: the least-squares line through the touches, each within
  `line_tol_atr`, with the strict candle test but `closes_through` closes (1) allowed beyond it.

Two more settings decide WHICH line you see and when it BREAKS. Only one support and one resistance
line are live at a time, and by default a new line needs MORE touches to replace the live one;
`prefer_newer=True` lets one with as many touches in when its last touch is more recent.
`break_sustain_bars=N` keeps a line alive through a close beyond it until N more closes in a row
confirm the break; a close back restarts the count (0: the first close retires the line).
`min_span_frac` (0.5) is how much of the stretch covered by the last 16 swing points a line's
touches must span; after big swings that stretch is hours long and a short clean line is refused
(1/3 lets it in).

Notebook 23 and the market view use `fit="strict"`; notebook 21 and the strategy file keep the
default. The trendline alerts take the same choices as `fit=`, `touch_tol=`, `wick_tol=`,
`prefer_newer=`, `hold=` (= `break_sustain_bars`) and `span=` (= `min_span_frac`);
`alerts/alerts_config_trendlines.py` uses strict everywhere, and `prefer_newer`, `hold=1` and
`span=1/3` on 1m (docs/ALERTS.md).

## Previous period highs and lows

The previous **day, week, month and year** high and low, each drawn across the period it applies
to, so a dotted step marks where yesterday's high stood all of today. They are complete before
the period starts, so nothing on them is known late. "Previous" means the previous period that
traded: Monday's previous day on gold is Friday. `LEVELS_TZ` sets where a day starts: `"UTC"`, or
`"America/New_York"` for a New York trading day. They are read off 1h bars, which give the exact
high and low of any day, week, month or year whose boundary falls on the hour.

## Market sessions

`SESSIONS = {name: (timezone, opens, closes)}`, each in that market's own clock, so daylight
saving is that market's. London opens at 08:00 London time all year, which is 07:00 UTC in summer
and 08:00 in winter. The defaults are the usual FX sessions (Sydney 07–16, Tokyo 09–18, London
08–17, New York 08–17); for the stock-market open on Nasdaq, use
`"New York": ("America/New_York", "09:30", "16:00")`. The lines at each open
(`SESSION_OPEN_LINES`) start visible on a chart of a week or less and hidden on a longer one,
where two lines a day become a curtain; `Sessions: + opens` shows them.

## Chart patterns — `CHART_PATTERNS`

Detected on the chart's timeframe and on every `LINE_TFS` timeframe. Set `CHART_PATTERNS = "all"`
for every one, or a list of names. `PATTERNS_TUNED = True` uses the calibrated per-pattern
settings, which give fewer, cleaner shapes. Measured: they are tuned for well-formed shapes, not
direction. Their hit rate is 0.48, so treat them as context, not signals (docs/WHAT_WORKS.md).

| name | direction | what it is |
|---|---|---|
| `double_top` | bearish | two highs at about the same price; confirmed when price closes below the low between them |
| `double_bottom` | bullish | two lows at about the same price; confirmed on a close above the high between them |
| `triple_top` | bearish | three highs at about the same price, then a close below the neckline |
| `triple_bottom` | bullish | three lows at about the same price, then a close above the neckline |
| `head_shoulders` | bearish | a higher high between two lower ones; confirmed below the neckline |
| `inverse_head_shoulders` | bullish | a lower low between two higher ones; confirmed above the neckline |
| `rising_wedge` | bearish | two rising lines converging, the lower one steeper; breaks down |
| `falling_wedge` | bullish | two falling lines converging, the upper one steeper; breaks up |
| `ascending_triangle` | bullish | a flat top over rising lows; breaks up through the top |
| `descending_triangle` | bearish | a flat bottom under falling highs; breaks down through the bottom |
| `symmetrical_triangle` | neutral | falling highs and rising lows of similar slope; either way |
| `rectangle_bull` | bullish | a flat range between parallel lines, broken upward |
| `rectangle_bear` | bearish | a flat range between parallel lines, broken downward |
| `bull_flag` | bullish | a sharp rise (the pole), then a short channel down; breaks up |
| `bear_flag` | bearish | a sharp fall, then a short channel up; breaks down |
| `bull_pennant` | bullish | a sharp rise, then a small symmetrical triangle; breaks up |
| `bear_pennant` | bearish | a sharp fall, then a small symmetrical triangle; breaks down |
| `broadening_top` | bearish | highs rising and lows falling, after a rise; breaks down |
| `broadening_bottom` | bullish | highs rising and lows falling, after a fall; breaks up |
| `channel_up` | neutral | two parallel rising lines of constant width |
| `channel_down` | neutral | two parallel falling lines of constant width |
| `channel_flat` | neutral | two parallel flat lines of constant width |

## Candlestick patterns — `CANDLE_PATTERNS`

Flagged on the bar the pattern completes on (never earlier). `CANDLE_PATTERNS = "all"` marks every
one, which on 1m or 5m is a lot; a short list is usually what you want. ▲ bullish, ▼ bearish,
• neutral. Patterns that need a price gap (abandoned baby, tasuki gaps, kickers) rarely or never
fire on a market that trades around the clock.

| | | |
|---|---|---|
| `doji` • | `kicker_bear` ▼ | `three_black_crows` ▼ |
| `long_legged_doji` • | `piercing_line` ▲ | `identical_three_crows` ▼ |
| `dragonfly_doji` ▲ | `dark_cloud_cover` ▼ | `three_inside_up` ▲ |
| `gravestone_doji` ▼ | `tweezer_bottom` ▲ | `three_inside_down` ▼ |
| `hammer` ▲ | `tweezer_top` ▼ | `three_outside_up` ▲ |
| `hanging_man` ▼ | `matching_low` ▲ | `three_outside_down` ▼ |
| `inverted_hammer` ▲ | `counterattack_bull` ▲ | `three_line_strike_bull` ▲ |
| `shooting_star` ▼ | `counterattack_bear` ▼ | `three_line_strike_bear` ▼ |
| `marubozu_bull` ▲ | `on_neck` ▼ | `rising_three_methods` ▲ |
| `marubozu_bear` ▼ | `thrusting` ▼ | `falling_three_methods` ▼ |
| `spinning_top_bull` ▲ | `separating_lines_bull` ▲ | `upside_tasuki_gap` ▲ |
| `spinning_top_bear` ▼ | `separating_lines_bear` ▼ | `downside_tasuki_gap` ▼ |
| `belt_hold_bull` ▲ | `inside_bar` • | `stick_sandwich` ▲ |
| `belt_hold_bear` ▼ | `outside_bar` • | `advance_block` ▼ |
| `engulfing_bull` ▲ | `morning_star` ▲ | `deliberation` ▼ |
| `engulfing_bear` ▼ | `evening_star` ▼ | `homing_pigeon` ▲ |
| `harami_bull` ▲ | `morning_doji_star` ▲ | `three_stars_in_the_south` ▲ |
| `harami_bear` ▼ | `evening_doji_star` ▼ | `ladder_bottom` ▲ |
| `harami_cross_bull` ▲ | `abandoned_baby_bull` ▲ | `hikkake_bull` ▲ |
| `harami_cross_bear` ▼ | `abandoned_baby_bear` ▼ | `hikkake_bear` ▼ |
| `kicker_bull` ▲ | `three_white_soldiers` ▲ |  |

## Where each notebook keeps these settings

| notebook | settings file |
|---|---|
| `23_break_confirmation_news` | `notebooks/nb23_config.py` (§7 S/R, §9 news, §10 patterns, §11 market hours & levels) |
| `market_view` | `notebooks/market_view_config.py` (add any of the settings above to it) |
| `lab` | `notebooks/lab_config.py` |
| `21_break_confirmation` | `notebooks/nb21_config.py` |
