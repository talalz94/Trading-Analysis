# Backup — S/R settings before the zone tuning study (2026-09-27)

Taken immediately before three changes to support/resistance zones:

1. **Anchored clustering** — a zone may no longer grow wider than `cluster_pct`.
   Before, a pivot joined a zone if it sat within `cluster_pct` of the zone's *running mean*,
   and since the mean moves as pivots join, zones crept: at ~63k (0.15% = ~94 pts) the
   strongest 5m zones spanned 149, 149 and 157 points.
2. **Break retirement** — a zone is retired once price closes beyond it for N consecutive bars.
   Before, crossing a zone did nothing; 48 of 54 zones drawn on the notebook-21 chart were levels
   price had broken through and held beyond for an hour or more.
3. **A tuned, per-timeframe sensitivity knob** (`SR_SENSITIVITY`) chosen by a forward test.

## What is here

The four notebook configs exactly as they stood. Their S/R settings at the time:

| setting | value (nb18–21) |
|---|---|
| ZONES_LEFT / ZONES_RIGHT | 10 / 10 |
| SR_CLUSTER_PCT | 0.15 |
| SR_MIN_TOUCHES | 2 |
| SR_MAX_ZONES | 40 |
| ZONE_LOOKBACK | 2000 |

Code state: commit `b1eb3088484451e8e541f7cf15886dae1502ad06`.

## How to get the old behaviour back

Nothing needs restoring from here to do that. In any notebook config:

```python
SR_SENSITIVITY = None     # legacy: your explicit ZONES_* values, no anchoring, no retirement
```

That reproduces the pre-change zones byte for byte (a test holds it). To restore a whole config
file, copy it back from this folder.
