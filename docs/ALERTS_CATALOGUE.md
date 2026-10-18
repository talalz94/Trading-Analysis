# Alerts catalogue: everything an alert can be set on

Generated from the code by `python -m quant.alerts list --write-docs`, so it lists exactly what `alerts/alerts_config.py` accepts. Do not edit it by hand. How to use it is in [ALERTS.md](ALERTS.md).

> Nothing here has a measured trading edge. The candle and chart patterns were tested and are no better than a coin flip (docs/WHAT_WORKS.md). An alert says *look at this chart*, not *trade it*.

## 1. Instruments (MARKETS)

source="binance": any Binance SPOT symbol. A wrong one shows as a feed error in `python -m quant.alerts status`. For example:

`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `PAXGUSDT (gold-backed token)`, `XAUTUSDT (gold-backed token)`



source="dukascopy": 1,380 instruments, priced on the BID side. Use a short name below, any 6-letter FX pair, or Dukascopy's own code exactly as listed.

### Short names

`XAUUSD = XAU/USD`, `XAGUSD = XAG/USD`, `EURUSD = EUR/USD`, `GBPUSD = GBP/USD`, `USDJPY = USD/JPY`, `AUDUSD = AUD/USD`, `USDCAD = USD/CAD`, `USDCHF = USD/CHF`, `WTI = E_Light`, `BRENT = E_Brent`, `XPTUSD = XPT.CMD/USD`, `XPDUSD = XPD.CMD/USD`, `COPPER = COPPER.CMD/USD`, `NATGAS = GAS.CMD/USD`, `US500 = E_SandP-500`, `NAS100 = E_NQ-100`, `US30 = E_D&J-Ind`, `US2000 = USSC2000.IDX/USD`, `GER40 = E_DAAX`, `UK100 = E_Futsee-100`, `FRA40 = E_CAAC-40`, `EU50 = E_DJE50XX`, `JPN225 = E_N225Jap`, `HK50 = E_H-Kong`, `AUS200 = E_XJO-ASX`, `DXY = DOLLAR.IDX/USD`

### FX pairs and spot metals (70)

`AUDCAD`, `AUDCHF`, `AUDJPY`, `AUDNZD`, `AUDSGD`, `AUDUSD`, `CADCHF`, `CADHKD`, `CADJPY`, `CHFJPY`, `CHFPLN`, `CHFSGD`, `EURAUD`, `EURCAD`, `EURCHF`, `EURCZK`, `EURDKK`, `EURGBP`, `EURHKD`, `EURHUF`, `EURJPY`, `EURMXN`, `EURNOK`, `EURNZD`, `EURPLN`, `EURRUB`, `EURSEK`, `EURSGD`, `EURTRY`, `EURUSD`, `EURZAR`, `GBPAUD`, `GBPCAD`, `GBPCHF`, `GBPJPY`, `GBPNZD`, `GBPUSD`, `HKDJPY`, `MXNJPY`, `NZDCAD`, `NZDCHF`, `NZDJPY`, `NZDSGD`, `NZDUSD`, `SGDJPY`, `TRYJPY`, `USDBRL`, `USDCAD`, `USDCHF`, `USDCNH`, `USDCZK`, `USDDKK`, `USDHKD`, `USDHUF`, `USDILS`, `USDJPY`, `USDMXN`, `USDNOK`, `USDNZD`, `USDPLN`, `USDRON`, `USDRUB`, `USDSEK`, `USDSGD`, `USDTHB`, `USDTRY`, `USDZAR`, `XAGUSD`, `XAUUSD`, `ZARJPY`

### Crypto (CFDs) (36)

`ADA/USD`, `AVE/USD`, `BAT/USD`, `BCH/CHF`, `BCH/EUR`, `BCH/GBP`, `BCH/USD`, `BTC/CHF`, `BTC/EUR`, `BTC/GBP`, `BTC/USD`, `CMP/USD`, `DSH/USD`, `ENJ/USD`, `EOS/USD`, `ETH/CHF`, `ETH/EUR`, `ETH/GBP`, `ETH/USD`, `LNK/USD`, `LTC/CHF`, `LTC/EUR`, `LTC/GBP`, `LTC/USD`, `MAT/USD`, `MKR/USD`, `TRX/USD`, `UNI/USD`, `UST/USD`, `XLM/CHF`, `XLM/EUR`, `XLM/GBP`, `XLM/USD`, `XMR/USD`, `XRP/USD`, `YFI/USD`

### Stock indices (24)

`CHI.IDX/USD`, `DOLLAR.IDX/USD`, `E_CAAC-40`, `E_D&J-Ind`, `E_DAAX`, `E_DJE50XX`, `E_Futsee-100`, `E_H-Kong`, `E_IBC-MAC`, `E_N225Jap`, `E_NQ-100`, `E_SWMI`, `E_SandP-500`, `E_XJO-ASX`, `IND.IDX/USD`, `ITA.IDX/EUR`, `NLD.IDX/EUR`, `PLN.IDX/PLN`, `PRT.IDX/EUR`, `RUSSELL.IDX/USD`, `SGD.IDX/SGD`, `SOA.IDX/ZAR`, `USSC2000.IDX/USD`, `VOL.IDX/USD`

### Commodities (13)

`COCOA.CMD/USD`, `COFFEE.CMD/USX`, `COPPER.CMD/USD`, `COTTON.CMD/USX`, `DIESEL.CMD/USD`, `E_Brent`, `E_Light`, `GAS.CMD/USD`, `OJUICE.CMD/USX`, `SOYBEAN.CMD/USX`, `SUGAR.CMD/USD`, `XPD.CMD/USD`, `XPT.CMD/USD`

### Bonds (3)

`BUND.TR/EUR`, `UKGILT.TR/GBP`, `USTBOND.TR/USD`

### ETFs (87) and single stocks (1,147)

Too many to list. Use Dukascopy's own code, e.g. `AAPL.US/USD`, `TSLA.US/USD`, `BMW.DE/EUR`. Stocks by market: US 630, UK 117, Japan 56, Italy 49, Germany 44, France 43, Switzerland 32, Mexico 29, Hong Kong 27, Spain 27, Sweden 21, Netherlands 19, Norway 14, Austria 9, Denmark 8, Finland 8, Belgium 7, Ireland 5, Portugal 2.

## 2. Timeframes (tfs)

`5m`, `15m`, `30m`, `1h`, `2h`, `4h`



Every market is fetched at 5m and the rest are built from it. Candles start on UTC boundaries: 4h candles open at 00, 04, 08 ... UTC.

## 3. Candlestick patterns (62), for kind="candle"

Use the names in patterns=(...), or patterns="all" / "bullish" / "bearish".

### Bullish (29)

`dragonfly_doji`, `hammer`, `inverted_hammer`, `marubozu_bull`, `spinning_top_bull`, `belt_hold_bull`, `engulfing_bull`, `harami_bull`, `harami_cross_bull`, `kicker_bull`, `piercing_line`, `tweezer_bottom`, `matching_low`, `counterattack_bull`, `separating_lines_bull`, `morning_star`, `morning_doji_star`, `abandoned_baby_bull`, `three_white_soldiers`, `three_inside_up`, `three_outside_up`, `three_line_strike_bull`, `rising_three_methods`, `upside_tasuki_gap`, `stick_sandwich`, `homing_pigeon`, `three_stars_in_the_south`, `ladder_bottom`, `hikkake_bull`

### Bearish (29)

`gravestone_doji`, `hanging_man`, `shooting_star`, `marubozu_bear`, `spinning_top_bear`, `belt_hold_bear`, `engulfing_bear`, `harami_bear`, `harami_cross_bear`, `kicker_bear`, `dark_cloud_cover`, `tweezer_top`, `counterattack_bear`, `on_neck`, `thrusting`, `separating_lines_bear`, `evening_star`, `evening_doji_star`, `abandoned_baby_bear`, `three_black_crows`, `identical_three_crows`, `three_inside_down`, `three_outside_down`, `three_line_strike_bear`, `falling_three_methods`, `downside_tasuki_gap`, `advance_block`, `deliberation`, `hikkake_bear`

### Either way (4)

`doji`, `long_legged_doji`, `inside_bar`, `outside_bar`

## 4. Chart patterns (22), for kind="chart"

Use the names in patterns=(...), or patterns="all" / "bullish" / "bearish".

### Bullish (9)

`double_bottom`, `triple_bottom`, `inverse_head_shoulders`, `falling_wedge`, `ascending_triangle`, `rectangle_bull`, `bull_flag`, `bull_pennant`, `broadening_bottom`

### Bearish (9)

`double_top`, `triple_top`, `head_shoulders`, `rising_wedge`, `descending_triangle`, `rectangle_bear`, `bear_flag`, `bear_pennant`, `broadening_top`

### Either way (4)

`symmetrical_triangle`, `channel_up`, `channel_down`, `channel_flat`

## 5. Alert kinds (15)

kind= picks what to look for. Each kind takes only its own settings, and any other setting is refused by name. direction= is always both | up | down.

### Candle patterns

**`candle`**: a candlestick pattern completes (62 of them; flagged on the pattern's LAST candle)

| setting | default | meaning |
|---|---|---|
| `patterns` | `None` | names, or 'all' / 'bullish' / 'bearish' |

```python
dict(on=True, name="...", kind="candle", patterns=('engulfing_bull', 'engulfing_bear'), markets="all", tfs="1h")
```

### Chart patterns

**`chart`**: a chart pattern FORMS (its swings are in place and it starts waiting for a break), and/or BREAKS OUT: price closes through its neckline / boundary (22 of them)

| setting | default | meaning |
|---|---|---|
| `patterns` | `'all'` | names, or 'all' / 'bullish' / 'bearish' |
| `event` | `'breakout'` | formed \| breakout \| both |
| `sensitivity` | `'normal'` | major \| normal \| minor -- bigger or smaller swings |

```python
dict(on=True, name="...", kind="chart", patterns=('double_top', 'double_bottom', 'head_shoulders', 'inverse_head_shoulders'), event='both', markets="all", tfs="1h")
```

### Levels & lines

**`trendline`**: a FITTED trendline (N touches) FORMS, and/or price BREAKS it by a rule: N closes, N whole candles and/or N candles of the break's colour beyond it

| setting | default | meaning |
|---|---|---|
| `event` | `'break'` | formed \| break \| both |
| `direction` | `'both'` | up = resistance lines (broken upward) · down = support lines |
| `closes` | `1` | N closes in a row beyond it (1 = the first close through) |
| `full` | `0` | N whole candles in a row beyond it, wicks included (0 = off) |
| `colour` | `0` | N candles in a row of the break's colour: red down, green up (0 = off) |
| `confirm` | `0` | then, within N candles, one of the break's colour must close beyond the WHOLE break candle (notebook 21's rule; 0 = off) |
| `min_touches` | `3` | pivots the line must touch |
| `swing_size` | `3.5` | zigzag pivot size in ATRs: bigger = longer, fewer lines |
| `fit` | `'ransac'` | ransac = the default fit \| strict = exactly through two swing extremes, and no candle closes through it before it is drawn |
| `touch_tol` | `0.6` | how far off the line a pivot may sit and still count as a touch, in ATR (0.3 with strict) |
| `wick_tol` | `0.15` | strict only: how far a wick may poke through the line, in ATR |

```python
dict(on=True, name="...", kind="trendline", event='both', closes=2, markets="all", tfs="1h")
```

**`sr_zone`**: a support / resistance zone (clustered swing highs and lows) is BROKEN by a rule (N closes / whole candles / candles of the break's colour), or price BOUNCES off it

| setting | default | meaning |
|---|---|---|
| `event` | `'break'` | break \| bounce \| both |
| `direction` | `'both'` | up = resistance broken / support held · down = the mirror |
| `closes` | `1` | N closes in a row beyond it (1 = the first close through) |
| `full` | `0` | N whole candles in a row beyond it, wicks included (0 = off) |
| `colour` | `0` | N candles in a row of the break's colour: red down, green up (0 = off) |
| `sensitivity` | `'normal'` | tight \| normal \| loose -- fewer, stronger zones to more |
| `min_touches` | `None` | swings a zone needs; None = the tuned value for the tf |

```python
dict(on=True, name="...", kind="sr_zone", event='break', closes=2, sensitivity='normal', markets="all", tfs="1h")
```

**`fvg`**: a fair value gap (three-candle imbalance) FORMS, or a live one gets FILLED

| setting | default | meaning |
|---|---|---|
| `event` | `'formed'` | formed \| filled \| both |
| `direction` | `'both'` | up = bullish gaps · down = bearish gaps |
| `min_size_atr` | `0.25` | smallest gap, in ATRs |
| `mitigation` | `'ce'` | what counts as filled: touch \| ce (midpoint) \| full |

```python
dict(on=True, name="...", kind="fvg", event='formed', min_size_atr=0.5, markets="all", tfs="1h")
```

### Indicators

**`ma_cross`**: price closes across a moving average (period=), or a fast MA crosses a slow one (fast= + slow=: golden / death cross)

| setting | default | meaning |
|---|---|---|
| `ma` | `'ema'` | ema \| sma |
| `period` | `200` | price vs this MA (ignored when fast/slow are set) |
| `fast` | `None` | fast MA period, for an MA-vs-MA cross |
| `slow` | `None` | slow MA period |
| `direction` | `'both'` | up = crosses above · down = crosses below |

```python
dict(on=True, name="...", kind="ma_cross", ma='ema', period=200, markets="all", tfs="1h")
```

**`rsi`**: RSI crosses into overbought / oversold (when='enter'), or back out of it ('exit')

| setting | default | meaning |
|---|---|---|
| `period` | `14` | RSI length |
| `above` | `70` | overbought line |
| `below` | `30` | oversold line |
| `when` | `'enter'` | enter \| exit |
| `direction` | `'both'` | up = crossing upward · down = crossing downward |

```python
dict(on=True, name="...", kind="rsi", period=14, above=70, below=30, when='exit', markets="all", tfs="1h")
```

**`rsi_divergence`**: price makes a new swing high / low that RSI does not confirm; known `right` bars after the swing, so it always arrives that late

| setting | default | meaning |
|---|---|---|
| `period` | `14` | RSI length |
| `left` | `5` | bars left of a swing |
| `right` | `5` | bars right of a swing = how late it is known |
| `direction` | `'both'` | up = bullish divergence · down = bearish |

```python
dict(on=True, name="...", kind="rsi_divergence", period=14, left=5, right=5, markets="all", tfs="1h")
```

**`macd`**: the MACD line crosses its signal line (cross='signal') or zero (cross='zero')

| setting | default | meaning |
|---|---|---|
| `fast` | `12` | fast EMA length |
| `slow` | `26` | slow EMA length |
| `signal` | `9` | signal line length |
| `cross` | `'signal'` | signal \| zero |
| `direction` | `'both'` | up = bullish cross · down = bearish |

```python
dict(on=True, name="...", kind="macd", fast=12, slow=26, signal=9, cross='signal', markets="all", tfs="1h")
```

**`stochastic`**: %K crosses %D -- upward only while oversold (below `below`), downward only while overbought (above `above`)

| setting | default | meaning |
|---|---|---|
| `k` | `14` | %K lookback |
| `d` | `3` | %D smoothing |
| `smooth` | `3` | %K smoothing |
| `above` | `80` | overbought line (100 = any downward cross) |
| `below` | `20` | oversold line (0 = any upward cross) |
| `direction` | `'both'` | up = bullish cross · down = bearish |

```python
dict(on=True, name="...", kind="stochastic", k=14, d=3, smooth=3, above=80, below=20, markets="all", tfs="1h")
```

**`supertrend`**: Supertrend flips direction

| setting | default | meaning |
|---|---|---|
| `period` | `10` | ATR length |
| `multiplier` | `3.0` | band width in ATRs |
| `direction` | `'both'` | up = flips bullish · down = flips bearish |

```python
dict(on=True, name="...", kind="supertrend", period=10, multiplier=3.0, markets="all", tfs="1h")
```

**`heikin_ashi`**: Heikin-Ashi candles change colour after at least `after` candles of the other

| setting | default | meaning |
|---|---|---|
| `after` | `3` | candles of the other colour before the flip |
| `direction` | `'both'` | up = turns green · down = turns red |

```python
dict(on=True, name="...", kind="heikin_ashi", after=3, markets="all", tfs="1h")
```

**`candle_run`**: N candles in a row of one colour (fires once, on the Nth)

| setting | default | meaning |
|---|---|---|
| `n` | `4` | candles in a row |
| `direction` | `'both'` | up = green run · down = red run |

```python
dict(on=True, name="...", kind="candle_run", n=4, markets="all", tfs="1h")
```

**`big_candle`**: a candle whose high-low range is at least `atr_mult` x the ATR before it

| setting | default | meaning |
|---|---|---|
| `atr_mult` | `2.5` | range / ATR |
| `atr_period` | `14` | ATR length |
| `direction` | `'both'` | up = green candle · down = red candle |

```python
dict(on=True, name="...", kind="big_candle", atr_mult=2.5, markets="all", tfs="1h")
```

### Setups

**`setup`**: a research ENTRY SETUP from quant/scenarios would open a trade here (one per setup episode, exactly as the backtest takes it)

| setting | default | meaning |
|---|---|---|
| `setup` | `None` | a key from the Setups list, e.g. 'bos_long@v1' |
| `options` | `{}` | that setup's own settings to change, as a dict |

```python
dict(on=True, name="...", kind="setup", setup='bos_long@v1', markets="all", tfs="1h")
```

## 6. Setups (28), for kind="setup"

The research entry setups in quant/scenarios. An alert fires on the bar the backtest would have entered, once per setup episode. Change a setup's own settings with options={...}, e.g. `dict(kind="setup", setup="rsi_div_long@v1", options=dict(div_right=3))`. Left out: time_entry (the metronome benchmark) and mtf_trendline, which needs several timeframes' lines at once.

| setup | side | what it is |
|---|---|---|
| `bos_long@v1` | long | break of the last confirmed zigzag swing, with the trend (bos) or against it (choch) |
| `bos_short@v1` | short | break of the last confirmed zigzag swing, with the trend (bos) or against it (choch) |
| `candle_pattern_long@v1` | long | candlestick patterns as the trigger: a named subset, or the whole library through its bull/bear score |
| `candle_pattern_short@v1` | short | candlestick patterns as the trigger: a named subset, or the whole library through its bull/bear score |
| `chart_pattern_long@v1` | long | classical chart patterns (double top/bottom, H&S, wedges, triangles, flags) fired on the bar their neckline BREAKS, never when the shape first looks complete |
| `chart_pattern_short@v1` | short | classical chart patterns (double top/bottom, H&S, wedges, triangles, flags) fired on the bar their neckline BREAKS, never when the shape first looks complete |
| `compression_break_long@v1` | long | break out of (or fade) a VOLATILITY-COMPRESSED range |
| `compression_break_short@v1` | short | break out of (or fade) a VOLATILITY-COMPRESSED range |
| `grad_extreme_long@v1` | long | EMA gradient beyond its own trailing quantile — trend continuation |
| `grad_extreme_short@v1` | short | EMA gradient beyond its own trailing quantile — trend continuation |
| `level_event_long@v1` | long | interaction with a reference level: break / fade / sweep / retest |
| `level_event_short@v1` | short | interaction with a reference level: break / fade / sweep / retest |
| `resistance_break_long@v1.0` | long | LONG mirror: close above the smoothed EMA + a break of a resistance zone |
| `resistance_break_long@v1.2` | long | LONG mirror, no S/R zones: N green candles above the EMA + a break of the highest high of the prior N bars. |
| `rsi_div_long@v1` | long | RSI divergence; div_mode='confirmed' lags by div_right bars, 'realtime' is zero-lag but only ~53% ratified |
| `rsi_div_short@v1` | short | RSI divergence; div_mode='confirmed' lags by div_right bars, 'realtime' is zero-lag but only ~53% ratified |
| `squeeze_long@v1` | long | squeeze fires off, traded in the direction of the momentum histogram |
| `squeeze_short@v1` | short | squeeze fires off, traded in the direction of the momentum histogram |
| `support_break_short@v1.0` | short | Core: close below the smoothed 200-EMA + a break of a support zone |
| `support_break_short@v1.1` | short | Structure: v1.0 AND the prior swings tagged a resistance zone above (a topped-out HH/HL run) before the support break. |
| `support_break_short@v1.2` | short | No-structure: no S/R zones — N consecutive red candles below the EMA and a break of the lowest low of the prior N bars. |
| `support_break_short@v1.3` | short | Nearest-trough: break the LOWER of the last two swing troughs (a hand-drawn support) — for charts where the auto S/R zone is weak/absent. |
| `vol_thrust_long@v1` | long | volume z-score spike (or relative-to-time-of-day) with a candle in the traded direction |
| `vol_thrust_short@v1` | short | volume z-score spike (or relative-to-time-of-day) with a candle in the traded direction |
| `vwap_rev_long@v1` | long | price beyond N sigma of session VWAP and starting back toward it |
| `vwap_rev_short@v1` | short | price beyond N sigma of session VWAP and starting back toward it |
| `wt_cross_long@v1` | long | WaveTrend fast/slow cross while the slow line is beyond the OB/OS level |
| `wt_cross_short@v1` | short | WaveTrend fast/slow cross while the slow line is beyond the OB/OS level |

## 7. News (the NEWS block)

Economic-calendar notices, driven by the clock rather than by candles: the week ahead, tomorrow's releases, reminders before a release, the actual against the forecast once it is out, and a release that moved. Times are Forex Factory's.

| setting | meaning |
|---|---|
| `on` | True = news notices on |
| `currencies` | {currency: lowest level kept}, e.g. {'USD': 'medium', 'EUR': 'high'}; '*' = every other currency |
| `timezone` | the zone your emails show times in, e.g. 'Asia/Karachi' |
| `week_ahead` | 'Sun 20:00' -- the week's high/critical news, once a week; None = off |
| `day_ahead` | '20:00' -- tomorrow's covered news, every evening; None = off |
| `remind_before` | {level: minutes before}, e.g. {'critical': (60, 15), 'high': (15,)} |
| `results` | lowest level whose actual-vs-forecast is emailed (needs JBLANKED_API_KEY); None = off |
| `warn_alerts` | a candle alert within this many minutes of high news says so; 0 = off |

### Currencies

`USD`, `EUR`, `GBP`, `JPY`, `AUD`, `CAD`, `CHF`, `NZD`, `CNY`

### Levels, lowest to highest

`low`, `medium`, `high`, `critical`



critical = a high-impact release on this list (never a medium one):

  USD: CPI m/m, CPI y/y, Core CPI m/m, Core CPI y/y, Non-Farm Employment Change, Unemployment Rate, Average Hourly Earnings m/m, Federal Funds Rate, FOMC Statement, FOMC Press Conference, FOMC Economic Projections, Core PCE Price Index m/m, PPI m/m, Core PPI m/m, Retail Sales m/m, Core Retail Sales m/m, Advance GDP q/q
  EUR: Main Refinancing Rate, Monetary Policy Statement, ECB Press Conference, CPI Flash Estimate y/y, Core CPI Flash Estimate y/y
  GBP: Official Bank Rate, Monetary Policy Summary, CPI y/y
  JPY: BOJ Policy Rate, Monetary Policy Statement, BOJ Press Conference
  CAD: Overnight Rate, BOC Rate Statement, CPI m/m, Employment Change, Unemployment Rate
  AUD: Cash Rate, RBA Rate Statement, CPI q/q, Employment Change
  NZD: Official Cash Rate, RBNZ Rate Statement
  CHF: SNB Policy Rate
