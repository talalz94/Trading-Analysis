"""
WHY DID THIS TRADE CLOSE -- a short tag for the chart and a sentence for the hover.

The engine records a close REASON (`take_profit`, `tp_moved`, `ref_target`...), which says which
mechanism fired but not what happened: a moved target does not say whether a near miss or a long
wait moved it, or from where to where; an FVG exit does not say which gap. This reads the trade, the
bars and the settings, and says it in words:

    TP·MOVED   MOVED — target moved 3R -> 1.8R at 06-09 16:55: near miss, got 92% of the way,
               then a close gave back 43%
    TP·FVG     FVG — filled the 15m gap 60,812-61,256 to its middle (61,034), 0.5R short of 3R

Each rule's TAG comes from its own settings (`TP_MOVE["tag"]`, `FVG_EXIT["tag"]`), so what the chart
says is decided in the config. Everything is recomputed from what the trade and the bars already
hold -- the near-miss numbers use the engine's own definitions (best price over the bars after
entry up to the move, the close of the move bar) -- so the words cannot disagree with the sim.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["label_exits", "STANDARD_TAGS"]

STANDARD_TAGS = {"take_profit": "TP", "stop_loss": "SL", "forced_close_end": "open",
                 "time_stop": "time", "alt_exit": "bail", "signal": "exit", "margin_call": "margin"}
_AT = {"touch": "near edge", "ce": "middle", "full": "far edge"}


def _get(cfg, name, default=None):
    return getattr(cfg, name, default)


def label_exits(trades: pd.DataFrame, D: pd.DataFrame, cfg: Any,
                gaps: Optional[pd.DataFrame] = None,
                picks: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """`trades` with two columns added: `exit_tag` (short, drawn on the marker) and `exit_why`.

    `gaps` / `picks` are the FVG exit's gap table and, per side, the gap each entry bar chose --
    only needed when that rule is on.
    """
    t = trades.copy()
    if not len(t):
        return t.assign(exit_tag=pd.Series(dtype=str), exit_why=pd.Series(dtype=str))
    rr = float(cfg.TP_RR)
    mv = _get(cfg, "TP_MOVE", None) or {}
    fx = _get(cfg, "FVG_EXIT", None) or {}
    mtag, ftag = mv.get("tag", "MOVED"), fx.get("tag", "FVG")
    h = D["high"].to_numpy(float)
    lo = D["low"].to_numpy(float)
    cl = D["close"].to_numpy(float)
    tt = D["t"].astype(str).str[5:16].to_numpy()        # "06-09 16:55"

    tags, whys = [], []
    for r in t.itertuples():
        s = 1.0 if r.side == "long" else -1.0
        e, risk = float(r.entry_price), abs(float(r.entry_price) - float(r.stop_price))
        why = str(r.close_reason)
        tag = STANDARD_TAGS.get(why, why)
        cr = why
        ei, xi = int(r.entry_i), int(r.exit_i)

        # ---- the target, if it moved
        mi = getattr(r, "target_moved_i", -1)
        mi = -1 if mi is None or (isinstance(mi, float) and np.isnan(mi)) else int(mi)
        moved = ""
        if mi >= 0 and risk > 0:
            dist = rr * risk
            seg = slice(ei + 1, mi + 1)
            best = (h[seg].max() - e) if s > 0 else (e - lo[seg].min())
            now = s * (cl[mi] - e)
            because = []
            near, back = float(mv.get("near") or 0), float(mv.get("back") or 0)
            after = int(mv.get("after_bars") or 0)
            if near > 0 and best / dist >= near and (best - now) / dist >= back:
                because.append(f"near miss, got {best / dist:.0%} of the way, then a close gave "
                               f"back {(best - now) / dist:.0%}")
            if after > 0 and mi - ei >= after:
                because.append(f"held {mi - ei} bars")
            to_r = s * (float(r.target_moved_to) - e) / risk
            moved = (f"target moved {rr:g}R -> {to_r:.1f}R at {tt[mi]}"
                     + (": " + " and ".join(because) if because else ""))

        # ---- the gap, if the trade carried one
        gap = ""
        ref = getattr(r, "ref_target", np.nan)
        if picks is not None and gaps is not None and ref == ref:
            lab = int(picks.get(r.side, np.full(ei + 1, -1))[ei])
            if lab >= 0 and lab in gaps.index:
                g = gaps.loc[lab]
                short = s * (e + s * rr * risk - float(ref)) / risk
                gap = (f"the {g.tf} gap {g.bot:,.0f}-{g.top:,.0f} to its "
                       f"{_AT.get(fx.get('at', 'ce'), fx.get('at'))} ({float(ref):,.0f}), "
                       f"{abs(short):.1f}R {'short of' if short >= 0 else 'past'} the {rr:g}R target")

        if cr == "tp_moved":
            tag = f"TP·{mtag}"
            why = f"{mtag}: {moved}" + ("; price was already past it, closed at that close"
                                        if xi == mi else "")
        elif cr in ("ref_target", "fvg_exit"):
            tag = f"TP·{ftag}"
            why = f"{ftag}: filled {gap}" if gap else f"{ftag}: reached its gap level"
        elif cr == "take_profit":
            why = f"TP: reached the {rr:g}R target"
        elif cr == "stop_loss":
            walked = risk > 0 and abs(float(r.exit_price) - float(r.stop_price)) > 0.25 * risk
            tag = "BE" if walked else "SL"
            why = ("SL at breakeven: the stop had walked to entry" if walked else "SL: stopped out")
        elif cr == "forced_close_end":
            why = "still open at the end of the window"
        if cr != "tp_moved" and moved:
            why += f" ({moved})"
        if cr not in ("ref_target", "fvg_exit") and gap:
            why += f" (its {ftag} exit, {gap}, was not reached)"
        tags.append(tag)
        whys.append(why)
    t["exit_tag"] = tags
    t["exit_why"] = whys
    return t
