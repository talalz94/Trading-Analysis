"""
Notebook 19's machinery: load, detect, trade twice, draw.

`fvg_lab(CFG)` does everything the notebook used to spell out in a forty-line cell, and hands back
an object whose attributes are the things worth looking at. The notebook is then three calls.

    lab = fvg_lab(CFG)     # loads, detects, runs the sim with AND without the veto
    lab.summary()          # what was found and what the veto did
    lab.figure()           # the chart

RUN TWICE, ALWAYS. A veto can only REMOVE trades, so a run with it is uninterpretable on its own —
you cannot tell a filter that removed losers from one that removed trades at random. `lab.all` is
the unfiltered run and `lab.trades` the filtered one, from the same frame and the same parameters,
so the difference is the veto and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _get(cfg, name, default=None):
    """A setting off the config module, or `default` if that config predates it.

    Mirrors the notebook's own SETTING_DEFAULTS fallback, for the same reason: a setting appended
    to a config file is invisible to an editor tab that was already open, and a stale save drops
    it. Dying on AttributeError two calls later is a worse way to find that out.
    """
    return getattr(cfg, name, default)


@dataclass
class FvgLab:
    """Everything notebook 19 looks at, already computed."""

    cfg: Any
    df: pd.DataFrame                      # the chart timeframe, warm-up included
    D: pd.DataFrame                       # + trendlines, EMAs and the gap columns
    line_frames: Dict[str, pd.DataFrame]  # fitted and TRADED
    gap_frames: Dict[str, pd.DataFrame]   # drawn, and allowed to veto — never fitted
    sr_frames: Dict[str, pd.DataFrame]    # levels are clustered on these — never fitted or traded
    ema_frames: Dict[str, pd.DataFrame]   # every frame an EMA needs, including EMA-only timeframes
    fvg_kwargs: Dict[str, Any]
    params: Dict[str, Any]
    trades: pd.DataFrame                  # with the veto, if it is on
    all: pd.DataFrame                     # without it, always
    gaps: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ---------------------------------------------------------------- what happened
    @property
    def blocked(self) -> pd.DataFrame:
        """The trades the veto removed — the only part of a filter worth reading."""
        if not len(self.all):
            return self.all
        kept = set(zip(self.trades["entry_time"], self.trades["side"]))
        keys = self.all.set_index(["entry_time", "side"]).index
        return self.all[~keys.isin(kept)]

    def census(self) -> pd.DataFrame:
        """One row per timeframe: how many gaps, how many still open, how fast they fill.

        The number that decides whether a gap rule can discriminate at all. A timeframe producing
        a gap every few bars has one live on almost every bar, so any "is a gap present" test on it
        is a constant rather than a filter.
        """
        rows = []
        for tf, t in self.gaps.items():
            if not len(t):
                rows.append(dict(tf=tf, gaps=0, open=0, median_size_atr=np.nan,
                                 median_bars_to_fill=np.nan))
                continue
            filled = t["dead_i"] >= 0
            rows.append(dict(
                tf=tf, gaps=len(t), open=int((~filled).sum()),
                median_size_atr=round(float(t["size_atr"].median()), 2),
                median_bars_to_fill=float((t.loc[filled, "dead_i"]
                                           - t.loc[filled, "born_i"]).median())))
        return pd.DataFrame(rows)

    def summary(self) -> None:
        """Print the three things worth knowing, and nothing else."""
        c = self.cfg
        print(self.census().to_string(index=False))
        t, a = self.trades, self.all
        if not len(a):
            print("\nno trades in this window")
            return
        if not _get(c, "USE_FVG_VETO", False):
            print(f"\n{len(a)} trades   {a.R.sum():+.1f}R   {100 * (a.pnl > 0).mean():.0f}% win"
                  f"    (veto off)")
            return
        b = self.blocked
        print(f"\nveto {c.FVG_VETO_MODE} on {'+'.join(map(str, _get(c, 'FVG_VETO_TFS') or ['all']))}"
              f"   {len(a)} trades -> {len(t)}  ({len(b)} blocked)\n")
        print(f"{'':9s} {'n':>3s} {'R':>7s} {'R/trade':>8s} {'win%':>5s}")
        for nm, x in (("kept", t), ("blocked", b), ("all", a)):
            if len(x):
                print(f"{nm:9s} {len(x):3d} {x.R.sum():+7.2f} {x.R.mean():+8.2f} "
                      f"{100 * (x.pnl > 0).mean():5.0f}")
            else:
                print(f"{nm:9s} {0:3d} {'-':>7s} {'-':>8s} {'-':>5s}")
        print("\nR PER TRADE is the column that matters: a veto only removes trades, so removing "
              "them\nat random lowers total R while leaving the per-trade number alone. And see "
              "docs/WHAT_WORKS.md\nbefore reading anything into a window this small.")

    # ---------------------------------------------------------------- the chart
    def figure(self, **overrides):
        """The chart, with every setting the config asked for. `overrides` go to `pattern_figure`."""
        from ..viz import pattern_figure

        c = self.cfg
        kw = dict(
            patterns=[], trendlines=_get(c, "SHOW_TRENDLINES", True),
            trendline_kwargs=c.TRENDLINE, tf_frames=self.line_frames,
            fvg_frames=self.gap_frames, fvgs=_get(c, "FVG_DRAW", True),
            fvg_kwargs=self.fvg_kwargs, fvg_extend=_get(c, "FVG_EXTEND", True),
            fvg_show=tuple(_get(c, "FVG_SHOW", ("bull", "bear"))),
            max_fvg_picker=_get(c, "MAX_FVG_PICKER", 6),
            view_start=c.VIEW_START, emas=c.EMAS, mtf_emas=c.MTF_EMAS,
            emas_on=_get(c, "EMAS_ON", True), chart_tf=c.TF,
            ema_frames=self.ema_frames,
            sr_zones=_get(c, "SHOW_SR_ZONES", False), sr_frames=self.sr_frames,
            max_sr_picker=_get(c, "MAX_SR_PICKER", 6),
            zone_lookback_bars=_get(c, "ZONE_LOOKBACK", 2000),
            zones_left=_get(c, "ZONES_LEFT", 10), zones_right=_get(c, "ZONES_RIGHT", 10),
            cluster_pct=_get(c, "SR_CLUSTER_PCT", 0.15),
            min_touches=_get(c, "SR_MIN_TOUCHES", 2), max_zones=_get(c, "SR_MAX_ZONES", 40),
            max_picker_instances=_get(c, "MAX_PICKER", 24),
            max_shapes_per_tf=_get(c, "MAX_SHAPES_TF", 12),
            max_trade_picker=_get(c, "MAX_TRADE_PICKER", 40),
            trades=self.trades, trade_tp_rr=c.TP_RR,
            start_visible=_get(c, "START_VISIBLE", ("trades", "fvg_bull", "fvg_bear")),
            height=_get(c, "HEIGHT", 900),
            title=(f"{c.SYMBOL} {c.TF}  {c.VIEW_START} to {c.END}  ·  "
                   f"{'/'.join((c.TF,) + tuple(_get(c, 'FVG_TFS', ())))} fair value gaps"
                   f"  ·  {len(self.trades)} trades"))
        kw.update(overrides)
        return pattern_figure(self.D, **kw)


def fvg_lab(cfg) -> FvgLab:
    """Load, detect, run the sim with and without the veto, and return it all.

    `cfg` is the notebook's settings MODULE. Every value is read from it; nothing is defaulted here
    that a config owns.
    """
    from ..data import get_ohlcv, tf_frames
    from ..indicators import add_mtf_emas, fvg_table, mtf_fvgs
    from ..indicators.trendlines import mtf_trendlines
    from ..scenarios import run_pieces
    from ..scenarios.entries import trendline_trigger
    from ..strategies import fvg_clear
    from ..strategies.conditions import raw
    from ..strategies.rulespec import build_rules, required_emas
    from ..viz import mtf_ema_timeframes

    c = cfg
    src = _get(c, "SOURCE", "binance")
    df = get_ohlcv(c.SYMBOL, c.TF, start=c.WARMUP_START, end=c.END, source=src,
                   tz="UTC").reset_index(drop=True)

    # LINE frames are fitted and therefore TRADED. GAP frames are drawn and may veto, never fitted.
    # Two separate settings on purpose: adding 4h to LINE_TFS just to see its gaps would quietly
    # start trading 4h trendlines, which is a change to the strategy made by a drawing setting.
    line_tfs = tuple(_get(c, "LINE_TFS", ()) or ())
    gap_tfs = tuple(_get(c, "FVG_TFS", ()) or ())
    warm = _get(c, "TF_WARMUP_BARS", 800)
    line_frames = (tf_frames(c.SYMBOL, line_tfs, view_start=c.VIEW_START, end=c.END,
                             warmup_bars=warm, source=src) if line_tfs else {})
    gap_frames = (tf_frames(c.SYMBOL, gap_tfs, view_start=c.VIEW_START, end=c.END,
                            warmup_bars=warm, source=src) if gap_tfs else {})
    # S/R layers: clustered, drawn, and nothing else. A level fitted on 1h pivots is a different
    # level from one fitted on 5m pivots, so each timeframe needs its own frame -- but none of them
    # is ever fitted for trendlines or traded.
    sr_tfs = tuple(_get(c, "SR_TFS", ()) or ()) if _get(c, "SHOW_SR_ZONES", False) else ()
    sr_frames = {k: (gap_frames[k] if k in gap_frames else line_frames[k])
                 for k in sr_tfs if k in gap_frames or k in line_frames}
    _need = [k for k in sr_tfs if k not in sr_frames]
    if _need:
        sr_frames.update(tf_frames(c.SYMBOL, tuple(_need), view_start=c.VIEW_START, end=c.END,
                                   warmup_bars=warm, source=src))

    fvg_kwargs = dict(mitigation=c.FVG_MITIGATION, min_size_atr=c.FVG_MIN_SIZE_ATR,
                      min_disp_atr=_get(c, "FVG_MIN_DISP_ATR", 0.0),
                      max_age_bars=_get(c, "FVG_MAX_AGE_BARS", 500),
                      atr_period=_get(c, "FVG_ATR_PERIOD", 14))

    # EVERY FRAME AN EMA NEEDS. MTF_EMAS is its own setting, so it may name a timeframe that is
    # neither a line, a gap nor an S/R layer -- and one FINER than the chart cannot be resampled
    # into existence, so pattern_figure refuses it rather than returning the chart's own EMA under
    # another name. Load it here instead of making the user add "1m" to an unrelated setting.
    ema_frames = {**line_frames, **gap_frames, **sr_frames}
    _ema_only = [k for k in mtf_ema_timeframes(c.MTF_EMAS, chart_tf=c.TF) if k not in ema_frames]
    if _ema_only:
        ema_frames.update(tf_frames(c.SYMBOL, tuple(_ema_only), view_start=c.VIEW_START,
                                    end=c.END, warmup_bars=warm, source=src))

    D = mtf_trendlines(df, line_frames, **c.TRENDLINE)
    D = add_mtf_emas(D, ema_frames, required_emas(c.ENTRY_RULES, chart_tf=c.TF))
    fv = mtf_fvgs(df, gap_frames, **fvg_kwargs)      # gap columns, with the usual +1 HTF shift
    for col in [x for x in fv.columns if "fvg_" in x]:
        D[col] = fv[col].to_numpy()

    tf_names = {"": c.TF, **{f"{k}__": k for k in line_frames}}
    params = dict(
        tl_prefixes=tuple(tf_names), tl_tf_names=tf_names,
        tl_close_n=c.TL_CLOSE_N, tl_full_m=c.TL_FULL_M, tl_touch_k=c.TL_TOUCH_K,
        tl_dir_n=c.TL_DIR_N, tl_dir_src=c.TL_DIR_SRC, tl_dir_full_n=c.TL_DIR_FULL_N,
        tl_extend=c.TL_EXTEND, tl_combine=c.TL_COMBINE, tl_tol_atr=c.TL_TOL_ATR,
        stop_n=c.STOP_N, stop_buffer_pct=c.STOP_BUFFER_PCT,
        be_at_rr=c.BE_AT_RR, tp_rr=c.TP_RR,
        sec_touch_n=c.SEC_TOUCH_N, sec_wrong_m=c.SEC_WRONG_M, sec_opp_n=c.SEC_OPP_N,
        sec_min_pct=c.SEC_MIN_PCT, sec_replace_tp=c.SEC_REPLACE_TP)
    secondary = "entry_retest_bail@v1" if _get(c, "USE_RETEST_BAIL", False) else None

    # the warm-up feeds the lines, the gaps and the stop lookback; it is not traded
    vs = pd.Timestamp(c.VIEW_START, tz="UTC")
    in_window = raw(lambda d: (pd.to_datetime(d["t"]) >= vs).to_numpy(), "in_window")

    def _filter(side, veto):
        # None means "no rule for this side", which is NOT the same as "nothing passes"
        extra = build_rules(c.ENTRY_RULES, side, chart_tf=c.TF,
                            combine=_get(c, "RULES_COMBINE", "all"))
        out = in_window if extra is None else (in_window & extra)
        if veto:
            out = out & fvg_clear(side, mode=c.FVG_VETO_MODE,
                                  timeframes=_get(c, "FVG_VETO_TFS"),
                                  max_age_bars=_get(c, "FVG_VETO_MAX_AGE", 0),
                                  min_size_atr=_get(c, "FVG_VETO_MIN_SIZE", 0.0),
                                  max_dist_atr=_get(c, "FVG_VETO_MAX_DIST", 0.0))
        return out

    sides = [s for s, on in (("long", _get(c, "TRADE_LONG", True)),
                             ("short", _get(c, "TRADE_SHORT", True))) if on]

    def _run(veto):
        runs = {s: run_pieces(D, f"mtf_trendline_{s}@v1", "high_of_n@v1", "be_ladder@v1",
                              secondary=secondary, params=params, cfg=c.COSTS,
                              entry_filter=_filter(s, veto)) for s in sides}
        parts = [trendline_trigger(r.df, r.trades, params, prefix_names=tf_names)
                 for r in runs.values()]
        t = (pd.concat(parts, ignore_index=True).sort_values("entry_time")
             .reset_index(drop=True))
        risk = (t["entry_price"] - t["stop_price"]).abs()
        t["R"] = t["pnl"] / (risk * t["qty"])
        t["risk_%"] = 100 * risk / t["entry_price"]
        t["tid"] = [f"#{i + 1:02d}" for i in range(len(t))]
        return t

    everything = _run(False)
    filtered = _run(True) if _get(c, "USE_FVG_VETO", False) else everything
    assert not len(filtered) or filtered["entry_time"].min() >= vs, "a trade started in the warm-up"

    gaps = {c.TF: fvg_table(df, **fvg_kwargs)}
    for tf, f in gap_frames.items():
        gaps[tf] = fvg_table(f.reset_index(drop=True), **fvg_kwargs)

    return FvgLab(cfg=c, df=df, D=D, line_frames=line_frames, gap_frames=gap_frames,
                  sr_frames=sr_frames, ema_frames=ema_frames,
                  fvg_kwargs=fvg_kwargs, params=params, trades=filtered, all=everything,
                  gaps=gaps)
