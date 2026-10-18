"""
The chart layers a config can ask for beyond a strategy's own lines, gaps and trades: chart and
candlestick patterns, the S/R style, the previous period highs and lows, and the market sessions.

    kw = chart_layers(cfg, symbol=..., source=..., view_start=..., end=...)
    pattern_figure(df, ..., **kw)

One function for the lab (fvg_lab) and the market view, so a setting means the same thing in
every notebook. Every setting is optional and its absence draws exactly what was drawn before:

    CHART_PATTERNS      [] | ["double_top", ...] | "all"   shapes on every fitted timeframe
    PATTERNS_TUNED      True = the calibrated per-pattern settings (quant.indicators.TUNED_SPECS)
    CANDLE_PATTERNS     [] | ["hammer", "engulfing_bull", ...] | "all"   markers on the chart TF
    SR_STYLE            "line" | "band";  SR_ROLES  True = support blue / resistance orange
    SR_COLOURS          {"support": "#...", "resistance": "#..."}
    PERIOD_LEVELS       ("D", "W", "M", "Y") available;  PERIOD_LEVELS_ON  shown at open
    LEVELS_TZ           where a day / week / month / year starts ("UTC")
    SESSIONS            {name: (tz, "HH:MM", "HH:MM")};  SESSIONS_ON;  SESSION_OPEN_LINES

The period levels need history the chart does not load -- last year's high for the yearly pair --
so they are read off 1h bars fetched from the start of the period before the first one on screen,
and only when a level is asked for. 1h bars give the exact high and low of any day, week, month or
year whose boundary falls on the hour, which is every timezone but a handful.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .fvg_lab import _get


def _level_frame(symbol: str, source: str, view_start, end, periods, tz: str) -> pd.DataFrame:
    """1h bars from the start of the period BEFORE the first visible one of the coarsest period."""
    from ..data import get_ohlcv
    vs = pd.Timestamp(view_start)
    vs = vs.tz_localize("UTC") if vs.tzinfo is None else vs.tz_convert("UTC")
    local = vs.tz_convert(tz).tz_localize(None)
    back = {"D": local.normalize() - pd.DateOffset(days=4),       # Monday's "yesterday" is Friday
            "W": local.normalize() - pd.DateOffset(weeks=2),
            "M": (local.to_period("M").start_time - pd.DateOffset(months=1)),
            "Y": (local.to_period("Y").start_time - pd.DateOffset(years=1))}
    start = min(back[p] for p in periods) - pd.Timedelta(days=1)
    df = get_ohlcv(symbol, "1h", start=start.strftime("%Y-%m-%d"), end=end, source=source,
                   tz="UTC")
    return df.reset_index(drop=True)


def chart_layers(cfg, *, symbol: str, source: str, view_start, end=None,
                 level_frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """pattern_figure keywords for the layers `cfg` asks for. `end=None` = up to now."""
    from ..indicators import PERIODS, period_levels, session_windows
    c, kw = cfg, {}

    pats = _get(c, "CHART_PATTERNS", None)
    if pats:
        kw["patterns"] = None if pats == "all" else list(pats)
        kw["tuned"] = bool(_get(c, "PATTERNS_TUNED", True))
    cdl = _get(c, "CANDLE_PATTERNS", None)
    if cdl:
        kw["candle_patterns"] = cdl

    style = _get(c, "SR_STYLE", None)
    if style is not None:
        if style not in ("line", "band"):
            raise ValueError(f"SR_STYLE must be 'line' or 'band', got {style!r}")
        kw["sr_style"] = style
    if _get(c, "SR_ROLES", False):
        kw["sr_roles"] = True
        kw["sr_colours"] = _get(c, "SR_COLOURS", None)

    periods = tuple(_get(c, "PERIOD_LEVELS", ()) or ())
    if periods:
        bad = sorted(set(periods) - set(PERIODS))
        if bad:
            raise ValueError(f"PERIOD_LEVELS has {bad}; it takes {list(PERIODS)}")
        tz = _get(c, "LEVELS_TZ", "UTC")
        src = level_frame if level_frame is not None else _level_frame(
            symbol, source, view_start, end, periods, tz)
        kw["period_levels"] = {p: period_levels(src, p, tz=tz) for p in periods}
        kw["levels_show"] = tuple(p for p in (_get(c, "PERIOD_LEVELS_ON", ("D",)) or ())
                                  if p in periods)

    sess = _get(c, "SESSIONS", None)
    if sess:
        for name, spec in sess.items():
            if not (isinstance(spec, (tuple, list)) and len(spec) == 3):
                raise ValueError(f"SESSIONS[{name!r}] must be (timezone, 'HH:MM', 'HH:MM')")
        hi = pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end)
        hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi
        kw["sessions"] = session_windows(sess, view_start, hi + pd.Timedelta(days=1))
        kw["sessions_show"] = bool(_get(c, "SESSIONS_ON", True))
        lines = tuple(_get(c, "SESSION_OPEN_LINES", ()) or ())
        unknown = sorted(set(lines) - set(sess))
        if unknown:
            raise ValueError(f"SESSION_OPEN_LINES names {unknown}, not in SESSIONS {list(sess)}")
        kw["session_open_lines"] = lines
    return kw
