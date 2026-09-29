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


# The two optional exit rules' settings.
EXIT_KEYS = frozenset({"USE_TP_MOVE", "TP_MOVE", "USE_FVG_EXIT", "FVG_EXIT"})

# EVERY setting read only when TRADING, never when loading, fitting or detecting. A change to these
# alone can reuse every frame, line and gap already built -- `FvgLab.retrade` -- so an A/B of exits,
# confirmations, stops or costs costs one sim per arm instead of a full reload. A setting the loader
# reads (TRENDLINE, ENTRY_RULES for its EMAs, the FVG detection settings, the window) must never be
# here: re-trading would silently keep the old frames. test_workspace.py parses the loader's source
# and fails if it reads any of these.
RETRADE_KEYS = EXIT_KEYS | frozenset({
    "COSTS", "TRADE_LONG", "TRADE_SHORT", "RULES_COMBINE", "TRADE_TFS",
    "STOP_N", "STOP_BUFFER_PCT", "BE_AT_RR", "TP_RR",
    "USE_RETEST_BAIL", "SEC_TOUCH_N", "SEC_WRONG_M", "SEC_OPP_N", "SEC_MIN_PCT", "SEC_REPLACE_TP",
    "TL_CLOSE_N", "TL_FULL_M", "TL_TOUCH_K", "TL_DIR_N", "TL_DIR_SRC", "TL_DIR_FULL_N",
    "TL_COMBINE", "TL_TOL_ATR", "TL_EXTEND", "TL_ONCE_PER_LINE",
    "TL_CONFIRM_N", "TL_CONFIRM_ALL", "TL_CONFIRM_MODE", "TL_CONFIRM_COLOUR", "TL_CONFIRM_CANCEL",
    "TL_CONFIRM_FROM",
    "USE_FVG_VETO", "FVG_VETO_MODE", "FVG_VETO_TFS", "FVG_VETO_MAX_AGE", "FVG_VETO_MIN_SIZE",
    "FVG_VETO_MAX_DIST", "FVG_RULES",
})


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
    end: str = ""                         # the effective right edge: AS_OF when set, else END
    _base: Dict[str, Any] = field(default_factory=dict, repr=False)   # what retrade() reuses

    def retrade(self, **overrides) -> "FvgLab":
        """The same study with only TRADING settings changed (`RETRADE_KEYS`), re-using every
        frame, line and gap already built. Anything else could change what is loaded or fitted, so
        it refuses rather than return a lab that silently still reflects the old value."""
        bad = sorted(set(overrides) - RETRADE_KEYS)
        if bad:
            raise KeyError(f"retrade() cannot change {bad}: the loader reads them. Run "
                           f"fvg_lab(variant(cfg, ...)) instead")
        from .variant import variant
        return _trade(variant(self.cfg, **overrides), self._base)

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
                rows.append(dict(tf=tf, gaps=0, open=0, expired=0, median_size_atr=np.nan,
                                 median_bars_to_fill=np.nan))
                continue
            filled = t["dead_i"] >= 0
            why = t["end_reason"] if "end_reason" in t.columns else pd.Series("open", t.index)
            rows.append(dict(
                # "open" meant "never filled", which silently included every gap that had simply
                # grown too old to count. They are different things and now two columns.
                tf=tf, gaps=len(t), open=int((why == "open").sum()),
                expired=int((why == "expired").sum()),
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

    @property
    def rejected(self) -> pd.DataFrame:
        """Every trendline break that did NOT become a trade, with the reason it was turned down.

        Only judgeable evidence about a filter. Computed on demand, because it re-runs the entry
        conditions and most callers never look at it.
        """
        from .rejected import rejected_setups
        return rejected_setups(self)

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
            # THE KNOB, plus only the S/R settings the config actually STATES -- None for the
            # rest, so they come from the tuned table rather than from a default written here
            # (a default stated in two places is the 8a8b8de bug). A config that predates the
            # knob gets "normal"; SR_SENSITIVITY = None restores the legacy zones exactly.
            sr_sensitivity=_get(c, "SR_SENSITIVITY", "normal"),
            zones_left=_get(c, "ZONES_LEFT", None), zones_right=_get(c, "ZONES_RIGHT", None),
            cluster_pct=_get(c, "SR_CLUSTER_PCT", None),
            min_touches=_get(c, "SR_MIN_TOUCHES", None),
            sr_break_bars=_get(c, "SR_BREAK_BARS", None), sr_anchor=_get(c, "SR_ANCHOR", None),
            max_zones=_get(c, "SR_MAX_ZONES", 40),
            max_picker_instances=_get(c, "MAX_PICKER", 24),
            max_shapes_per_tf=_get(c, "MAX_SHAPES_TF", 12),
            max_trade_picker=_get(c, "MAX_TRADE_PICKER", 40),
            trades=self.trades, trade_tp_rr=c.TP_RR, ref_target_label="FVG exit",
            exit_labels=_get(c, "EXIT_LABELS", False),
            start_visible=_get(c, "START_VISIBLE", ("trades", "fvg_bull", "fvg_bear")),
            height=_get(c, "HEIGHT", 900),
            rewind=_get(c, "REWIND", False),
            # SHOW_REJECTED is opt-in: computing it re-runs the entry conditions, and a chart that
            # nobody asked to see them on should not pay for it. The layer starts legendonly even
            # when it is on, so the chart still opens on price and trades.
            rejected=self.rejected if _get(c, "SHOW_REJECTED", False) else None,
            title=(f"{c.SYMBOL} {c.TF}  {c.VIEW_START} to {self.end}"
                   f"{'  (AS OF)' if _get(c, 'AS_OF', None) else ''}  ·  "
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
    from ..strategies.rulespec import required_emas
    from ..viz import mtf_ema_timeframes

    c = cfg
    src = _get(c, "SOURCE", "binance")
    # AS_OF IS JUST AN EARLIER END. Every detector, level and entry in this project is already
    # causal, so "what was knowable at that bar" needs no new logic -- stop the data there and the
    # whole pipeline produces the point-in-time answer. Exact and reproducible; it costs a re-run,
    # which is the trade against the browser tool's instant-but-filtered rewind.
    end = _get(c, "AS_OF", None) or c.END
    df = get_ohlcv(c.SYMBOL, c.TF, start=c.WARMUP_START, end=end, source=src,
                   tz="UTC").reset_index(drop=True)

    # LINE frames are fitted and therefore TRADED. GAP frames are drawn and may veto, never fitted.
    # Two separate settings on purpose: adding 4h to LINE_TFS just to see its gaps would quietly
    # start trading 4h trendlines, which is a change to the strategy made by a drawing setting.
    line_tfs = tuple(_get(c, "LINE_TFS", ()) or ())
    gap_tfs = tuple(_get(c, "FVG_TFS", ()) or ())
    warm = _get(c, "TF_WARMUP_BARS", 800)
    line_frames = (tf_frames(c.SYMBOL, line_tfs, view_start=c.VIEW_START, end=end,
                             warmup_bars=warm, source=src) if line_tfs else {})
    gap_frames = (tf_frames(c.SYMBOL, gap_tfs, view_start=c.VIEW_START, end=end,
                            warmup_bars=warm, source=src) if gap_tfs else {})
    # S/R layers: clustered, drawn, and nothing else. A level fitted on 1h pivots is a different
    # level from one fitted on 5m pivots, so each timeframe needs its own frame -- but none of them
    # is ever fitted for trendlines or traded.
    sr_tfs = tuple(_get(c, "SR_TFS", ()) or ()) if _get(c, "SHOW_SR_ZONES", False) else ()
    sr_frames = {k: (gap_frames[k] if k in gap_frames else line_frames[k])
                 for k in sr_tfs if k in gap_frames or k in line_frames}
    _need = [k for k in sr_tfs if k not in sr_frames]
    if _need:
        sr_frames.update(tf_frames(c.SYMBOL, tuple(_need), view_start=c.VIEW_START, end=end,
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
                                    end=end, warmup_bars=warm, source=src))

    D = mtf_trendlines(df, line_frames, **c.TRENDLINE)
    D = add_mtf_emas(D, ema_frames, required_emas(c.ENTRY_RULES, chart_tf=c.TF))
    fv = mtf_fvgs(df, gap_frames, **fvg_kwargs)      # gap columns, with the usual +1 HTF shift
    for col in [x for x in fv.columns if "fvg_" in x]:
        D[col] = fv[col].to_numpy()

    gaps = {c.TF: fvg_table(df, **fvg_kwargs)}
    for tf, f in gap_frames.items():
        gaps[tf] = fvg_table(f.reset_index(drop=True), **fvg_kwargs)

    base = dict(df=df, D=D, line_frames=line_frames, gap_frames=gap_frames, sr_frames=sr_frames,
                ema_frames=ema_frames, fvg_kwargs=fvg_kwargs, gaps=gaps, end=end, src=src,
                warm=warm)
    return _trade(c, base)


def _trade(c, base) -> FvgLab:
    """Everything after loading: the entry conditions, the exits, the sim. Reads the config for the
    trading settings only, so `retrade` can call it again with different exits."""
    from ..data import tf_frames
    from ..indicators import fvg_on_grid
    from ..scenarios import resolve_stop, run_pieces
    from ..scenarios.entries import trendline_trigger
    from ..strategies import build_fvg_rules, fvg_clear, fvg_target
    from ..strategies.conditions import raw
    from ..strategies.rulespec import build_rules

    df, D = base["df"], base["D"]
    line_frames, gap_frames = base["line_frames"], base["gap_frames"]
    fvg_kwargs, end, src, warm = base["fvg_kwargs"], base["end"], base["src"], base["warm"]

    tf_names = {"": c.TF, **{f"{k}__": k for k in line_frames}}
    # WHICH FITTED TIMEFRAMES MAY TRADE. Every timeframe in tf_names is fitted and drawn; TRADE_TFS
    # picks the ones whose breaks become entries. None (a config that predates it) = all of them,
    # which is exactly what every earlier notebook did -- the chart's own timeframe included, with
    # no way to leave it out.
    trade_tfs = _get(c, "TRADE_TFS", None)
    if trade_tfs is not None:
        unknown = sorted(set(trade_tfs) - set(tf_names.values()))
        if unknown:
            raise ValueError(f"TRADE_TFS names {unknown}, whose lines are not fitted: a timeframe "
                             f"trades only if it is TF ({c.TF}) or in LINE_TFS {tuple(line_frames)}")
    traded = tuple(p for p, tf in tf_names.items() if trade_tfs is None or tf in trade_tfs)
    params = dict(
        tl_prefixes=traded, tl_tf_names=tf_names,
        tl_close_n=c.TL_CLOSE_N, tl_full_m=c.TL_FULL_M, tl_touch_k=c.TL_TOUCH_K,
        tl_dir_n=c.TL_DIR_N, tl_dir_src=c.TL_DIR_SRC, tl_dir_full_n=c.TL_DIR_FULL_N,
        tl_extend=c.TL_EXTEND, tl_combine=c.TL_COMBINE, tl_tol_atr=c.TL_TOL_ATR,
        tl_once_per_line=_get(c, "TL_ONCE_PER_LINE", False),
        tl_confirm_n=_get(c, "TL_CONFIRM_N", 0),
        tl_confirm_all=_get(c, "TL_CONFIRM_ALL", False),
        tl_confirm_mode=_get(c, "TL_CONFIRM_MODE", "close"),
        tl_confirm_colour=_get(c, "TL_CONFIRM_COLOUR", True),
        tl_confirm_cancel=_get(c, "TL_CONFIRM_CANCEL", False),
        tl_confirm_from=_get(c, "TL_CONFIRM_FROM", "every"),
        stop_n=c.STOP_N, stop_buffer_pct=c.STOP_BUFFER_PCT,
        be_at_rr=c.BE_AT_RR, tp_rr=c.TP_RR,
        sec_touch_n=c.SEC_TOUCH_N, sec_wrong_m=c.SEC_WRONG_M, sec_opp_n=c.SEC_OPP_N,
        sec_min_pct=c.SEC_MIN_PCT, sec_replace_tp=c.SEC_REPLACE_TP)
    secondary = "entry_retest_bail@v1" if _get(c, "USE_RETEST_BAIL", False) else None

    # ---- THE TWO OPTIONAL EXIT RULES. Both off (the default for a config that predates them) runs
    # the engine's original path, so the trades are exactly the ones before these existed.
    exit_cfg = {}
    if _get(c, "USE_TP_MOVE", False):
        m = c.TP_MOVE
        exit_cfg.update(tp_move_to=float(m["to"]), tp_move_near=float(m.get("near") or 0.0),
                        tp_move_back=float(m.get("back") or 0.0),
                        tp_move_after=int(m.get("after_bars") or 0))
    exit_gaps, fx = None, _get(c, "FVG_EXIT", None) or {}
    if _get(c, "USE_FVG_EXIT", False):
        # THE SAME GAPS THE CHART DRAWS: same frames where they exist, same detection settings
        # (mitigation, size gate, max age), so the gap a trade exits on is one you can see.
        tfs = tuple(fx.get("tfs") or ())
        frames = {k: (df if k == c.TF else gap_frames.get(k, line_frames.get(k))) for k in tfs}
        need = tuple(k for k, f in frames.items() if f is None)
        if need:
            frames.update(tf_frames(c.SYMBOL, need, view_start=c.VIEW_START, end=end,
                                    warmup_bars=warm, source=src))
        exit_gaps = fvg_on_grid(D, frames, **fvg_kwargs)

    # the warm-up feeds the lines, the gaps and the stop lookback; it is not traded
    vs = pd.Timestamp(c.VIEW_START, tz="UTC")
    in_window = raw(lambda d: (pd.to_datetime(d["t"]) >= vs).to_numpy(), "in_window")

    def _filter(side, veto):
        # None means "no rule for this side", which is NOT the same as "nothing passes"
        extra = build_rules(c.ENTRY_RULES, side, chart_tf=c.TF,
                            combine=_get(c, "RULES_COMBINE", "all"))
        out = in_window if extra is None else (in_window & extra)
        gap_rules = build_fvg_rules(_get(c, "FVG_RULES", None), side)
        if gap_rules is not None:
            out = out & gap_rules
        if veto:
            out = out & fvg_clear(side, mode=c.FVG_VETO_MODE,
                                  timeframes=_get(c, "FVG_VETO_TFS"),
                                  max_age_bars=_get(c, "FVG_VETO_MAX_AGE", 0),
                                  min_size_atr=_get(c, "FVG_VETO_MIN_SIZE", 0.0),
                                  max_dist_atr=_get(c, "FVG_VETO_MAX_DIST", 0.0))
        return out

    sides = [s for s, on in (("long", _get(c, "TRADE_LONG", True)),
                             ("short", _get(c, "TRADE_SHORT", True))) if on]

    gap_pick = {}                          # side -> the gap each entry bar's FVG exit chose

    def _frame_for(side):
        """D, plus this side's FVG-exit column when that rule is on, and the config to match."""
        cfg = {**c.COSTS, **exit_cfg}
        if exit_gaps is None:
            return D, cfg
        # the stop method's OWN reference column, so the R the gap is measured in is the trade's
        ref = resolve_stop("high_of_n@v1").prepare(D, {**params, "_side": side})["stop_extreme_n"]
        col = f"fvg_exit_{side}"
        lv, gap_pick[side] = fvg_target(D, exit_gaps, side=side,
                                        stop_ref=ref.to_numpy(np.float64),
                                        stop_buffer_pct=c.STOP_BUFFER_PCT, tp_rr=c.TP_RR,
                                        near_r=float(fx.get("near_r", 1.0)),
                                        at=fx.get("at", "ce"), return_gap=True)
        cfg[f"tp_ref_{side}_col"] = col
        return D.assign(**{col: lv}), cfg

    def _run(veto):
        runs = {}
        for s in sides:
            Ds, cfg_s = _frame_for(s)
            runs[s] = run_pieces(Ds, f"mtf_trendline_{s}@v1", "high_of_n@v1", "be_ladder@v1",
                                 secondary=secondary, params=params, cfg=cfg_s,
                                 entry_filter=_filter(s, veto))
        parts = [trendline_trigger(r.df, r.trades, params, prefix_names=tf_names)
                 for r in runs.values()]
        t = (pd.concat(parts, ignore_index=True).sort_values("entry_time")
             .reset_index(drop=True))
        risk = (t["entry_price"] - t["stop_price"]).abs()
        t["R"] = t["pnl"] / (risk * t["qty"])
        t["risk_%"] = 100 * risk / t["entry_price"]
        t["tid"] = [f"#{i + 1:02d}" for i in range(len(t))]
        # the engine's generic "second target" IS the FVG exit here -- say so in the table
        t["close_reason"] = t["close_reason"].replace({"ref_target": "fvg_exit"})
        if _get(c, "EXIT_LABELS", False):
            # a tag for the chart and a sentence for the hover, from the rules' own settings
            from .exit_labels import label_exits
            t = label_exits(t, D, c, exit_gaps, gap_pick)
        return t

    everything = _run(False)
    filtered = _run(True) if _get(c, "USE_FVG_VETO", False) else everything
    assert not len(filtered) or filtered["entry_time"].min() >= vs, "a trade started in the warm-up"

    return FvgLab(cfg=c, df=df, D=D, line_frames=line_frames, gap_frames=gap_frames,
                  sr_frames=base["sr_frames"], ema_frames=base["ema_frames"], end=str(end),
                  fvg_kwargs=fvg_kwargs, params=params, trades=filtered, all=everything,
                  gaps=base["gaps"], _base=base)
