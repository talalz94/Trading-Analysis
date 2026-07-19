# Experiment: ema_and_exit_search

Best EMA ribbon (fast/slow/confirm) x stop/target for gold, by Sharpe.

- **Asset / timeframe:** PAXGUSDT 1m  (2026-01-01 → 2026-05-31)
- **Objective:** maximize **sharpe**  (min 20 trades)
- **Search space:** strategy={'fast': [20, 50, 100], 'slow': [150, 200], 'confirm_n': [1, 3, 5]} ; cfg={'sl_value': [0.4, 0.6, 1.0], 'tp_value': [1.5, 2.0, 3.0]}
- **Trials ranked:** 162

## Best settings
```
best sharpe = -0.02296070016049723
strategy params: {'fast': 100.0, 'slow': 200.0, 'confirm_n': 1.0}
config params:   {'sl_value': 1.0, 'tp_value': 3.0}
```

### Reproduce
```python
from quant.strategies import EmaRibbon
from quant.engine import BacktestConfig
strat = EmaRibbon(**{'fast': 100.0, 'slow': 200.0, 'confirm_n': 1.0})
cfg = <base_cfg with {'sl_value': 1.0, 'tp_value': 3.0}>
res = strat.backtest(df, cfg)
```

## Top 15 results
```
 fast  slow  confirm_n  sl_value  tp_value  num_trades  total_return_pct  win_rate_pct  profit_factor    sharpe  max_drawdown_pct
  100   200          1       1.0       3.0        96.0         -1.669291     29.166667       0.980512 -0.022961         20.481378
   50   150          3       1.0       3.0        96.0         -5.444389     28.125000       0.937413 -0.393376         24.515302
   50   200          3       1.0       3.0        96.0         -5.444394     28.125000       0.937207 -0.396172         23.616869
   50   150          5       1.0       3.0        93.0         -5.835696     27.956989       0.930760 -0.433371         24.827691
   50   200          5       1.0       3.0        93.0         -5.835900     27.956989       0.930787 -0.437685         24.827681
  100   200          3       1.0       3.0        93.0         -5.882499     27.956989       0.930123 -0.477024         24.717938
  100   150          3       1.0       3.0        93.0         -5.882414     27.956989       0.930423 -0.481103         25.603391
  100   150          3       1.0       2.0       110.0         -5.914821     38.181818       0.931547 -0.505168         24.651174
  100   200          3       1.0       2.0       114.0         -5.958883     37.719298       0.932613 -0.510004         23.279055
   20   200          3       1.0       3.0       101.0         -7.308202     27.722772       0.919176 -0.555615         24.609229
   20   200          5       1.0       3.0       101.0         -7.308190     27.722772       0.919176 -0.556232         24.609230
   20   150          5       1.0       3.0       103.0         -9.475088     27.184466       0.894942 -0.752314         26.022921
  100   150          5       1.0       2.0       107.0         -8.170255     37.383178       0.901876 -0.775350         25.537369
  100   150          1       1.0       3.0       100.0         -9.858715     27.000000       0.886679 -0.802355         27.017978
  100   150          1       1.0       2.0       136.0        -10.646097     36.764706       0.899709 -0.937099         27.114004
```

*Full ranked results in `results.csv`; best in `best.json`.*
