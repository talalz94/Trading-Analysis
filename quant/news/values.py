"""
From the strings a calendar prints ("0.3%", "245K", "-1.2B", "4.35|2.3") to numbers, and from
actual-vs-forecast to a SURPRISE and a SENTIMENT.

A value keeps its printed unit: "245K" is 245,000 and "0.3%" is 0.3, so an actual and a forecast
of the same event subtract cleanly. What is NOT a number stays NaN rather than being guessed:
  "<0.1%"     a bound, not a value
  "0-0-9"     a vote split (Bank of England MPC)
  "Pass"      a referendum
"4.35|2.3" is a bond auction's yield | bid-to-cover; the yield is the headline, so it is taken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_RX = r"^\s*(?P<sign>-)?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%+|[KkMBT])?\s*$"


def parse_values(s) -> np.ndarray:
    """Vectorised: an array-like of printed values -> float64 (NaN where it is not a number)."""
    s = pd.Series(s, dtype="object").astype("string").str.split("|").str[0]
    m = s.str.extract(_RX)
    v = pd.to_numeric(m["num"], errors="coerce")
    v = v.where(m["sign"].isna(), -v)
    unit = m["unit"].fillna("").str.upper().str[:1]
    mult = unit.map(_MULT).astype(float).fillna(1.0)
    return (v * mult).to_numpy(dtype=float, na_value=np.nan)


def parse_value(x) -> float:
    return float(parse_values([x])[0])


def sentiment(surprise, direction) -> np.ndarray:
    """"better" / "worse" for the CURRENCY / "as expected" / "" (no forecast, or no direction).

    `direction` is +1 when a higher actual is good for the currency (catalog.direction)."""
    s = np.asarray(surprise, dtype=float)
    d = np.asarray(direction, dtype=float)
    out = np.full(s.shape, "", dtype=object)
    ok = np.isfinite(s) & (d != 0)
    same = ok & (np.abs(s) < 1e-9)
    signed = np.where(ok, np.sign(s) * d, 0)
    out[ok & (signed > 0)] = "better"
    out[ok & (signed < 0)] = "worse"
    out[same] = "as expected"
    return out


def surprise_z(events: pd.DataFrame, *, min_history: int = 8) -> np.ndarray:
    """Each release's surprise in units of THAT EVENT'S PAST surprises (strictly earlier ones).

    Causal by construction: release i is scaled by the spread of releases 0..i-1 of the same
    (currency, title), so a z-score never knows about a surprise that came after it. Fewer than
    `min_history` earlier surprises -> NaN, because the spread of three numbers is noise."""
    e = events[["currency", "title", "t", "surprise"]].copy()
    e["_i"] = np.arange(len(e))
    e = e.sort_values(["currency", "title", "t"], kind="stable")
    s = e["surprise"].astype(float)
    # root mean square of the PREVIOUS surprises (their typical size around zero, which is what
    # "how far from the forecast" means -- not their spread around their own mean)
    sq = (s ** 2).where(s.notna())
    cnt = s.notna().astype(float).groupby([e["currency"], e["title"]], sort=False).cumsum()
    ssum = sq.fillna(0.0).groupby([e["currency"], e["title"]], sort=False).cumsum()
    prev_cnt = cnt - s.notna().astype(float)
    prev_sum = ssum - sq.fillna(0.0)
    rms = np.sqrt(prev_sum / prev_cnt.where(prev_cnt > 0))
    z = s / rms.where(rms > 0)
    z = z.where(prev_cnt >= min_history)
    out = np.full(len(events), np.nan)
    out[e["_i"].to_numpy()] = z.to_numpy(dtype=float)
    return out
