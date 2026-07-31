"""
The "support-break short" scenario family (versioned).

The setup, in words (from the chart the user described):
  price had been ABOVE a smoothed 200-EMA with a positive gradient, made higher-highs/higher-lows into
  a resistance zone with the lows trapped on a support, then rolled OVER — closing below the EMA (even
  full candles below) but *still no sell* — until it finally BREAKS THROUGH the support. That break is
  the short trigger.

Versions (kept side-by-side so we can choose per chart/TF):
  v1.0  core trigger   — below the EMA line + a support-zone break. Minimal, robust, structure-free
                         apart from the zone itself. Optional down-gradient gate (`grad_max`).
  v1.1  +structure     — v1.0 AND require the prior HH/HL run into a resistance zone (the "top" shape)
                         before the break.
  v1.2  no-structure   — no S/R zones: N red candles below the EMA + a break of the prior-N-bar low.
  v1.3  nearest-trough — like v1.0 but when no strong zone is near, break the *lower* of two nearby
                         troughs (a hand-drawn support when the auto-zone is weak).

All are ONE-entry-per-episode (re-arm when price closes back above the EMA line). `entry_mode` chooses
*when* to pull the trigger relative to the break — the user's Q2 — as a swept knob.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import signals as S
from ..indicators import add_ema_gradient, add_heikin_ashi, add_sr_zones, add_swings
from ..strategies.conditions import Cond, above, below, full_below, green, n_red_below, red
from .base import Scenario, register_scenario

# ---- shared defaults ------------------------------------------------------
# --- the tunable knob groups (every one is a `params=` key you can set from the notebook) ---
# EMA / gradient: which EMA line the setup uses, and the gradient threshold that gates entries.
#   grad_period/grad_smooth = the "200 EMA smoothed by 9" line;  grad_max = the THRESHOLD
#   (entry needs grad < grad_max; 0.0 = "any down-slope", -0.05 = "must be clearly falling",
#    None = don't gate on the gradient at all).
_GRAD = {"grad_prefix": "grad", "grad_period": 200, "grad_smooth": 9, "grad_lookback": 5,
         "grad_atr": 14, "grad_max": None}
_SR = {"sr_left": 10, "sr_right": 10, "sr_cluster_pct": 0.15, "sr_min_touches": 1}
# Entry timing + the candle COUNTS ("N") and candle TYPE used by those counts.
#   candle_src = 'real' | 'ha'   -> every candle-colour/length test below reads real or Heikin-Ashi
#   n_red      = N for entry_mode='n_red'      (N red candles below the EMA after the break)
#   wait_n     = N bars to wait for entry_mode='wait_n'
#   pullback_lookback = N bar window for entry_mode='pullback'
#   confirm_red (v1.2 only) = N red candles below the EMA required as context
_TIMING = {"entry_mode": "break_close", "wait_n": 3, "n_red": 2, "pullback_lookback": 5,
           "candle_src": "real"}
# "Level B" — the level whose break is the trigger. Every version writes it to a `break_level`
# column in prepare(), so it can be audited and drawn on the chart per trade.
#   break_mode = 'close_below' (the candle CLOSES below B — the default/original rule)
#              | 'full_below'  (the whole candle forms below B: high < B)
#   break_min_ema_dist_pct / break_max_ema_dist_pct — optional (default None = off) validity window
#     on |B - EMA line| / EMA x 100, to reject levels that sit too close to, or too far from, the EMA.
_BREAK = {"break_mode": "close_below", "break_min_ema_dist_pct": None,
          "break_max_ema_dist_pct": None}


def _grad_cols(p):
    pre = p["grad_prefix"]
    return pre, f"{pre}_line"


def _candle_cols(src):
    """(high, close) column names for the requested candle source."""
    return ("ha_high", "ha_close") if src == "ha" else ("high", "close")


def _latch_break_level(out: "pd.DataFrame", p: dict) -> "pd.DataFrame":
    """Latch LEVEL B — the level that was actually broken — for the whole episode.

    `break_level_raw` is the per-bar candidate (prior-bar support / prior-N low / trough). It is not
    enough on its own: an S/R support level steps DOWN to the next zone the instant price closes
    through it, so one bar later it no longer names the level that broke. We therefore freeze the
    candidate at the FIRST break of each episode and hold it until price closes back above the EMA
    line (the same episode boundary the entry rule re-arms on).

    This gives (a) a stable per-trade level to draw on the chart and (b) a meaningful reference for
    the `full_below` confirmation, which by nature fires at or after the break bar.
    """
    _pre, line = _grad_cols(p)
    raw = out["break_level_raw"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    ln = out[line].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        broke = (close < raw) & ~np.isnan(raw)
        reset = close >= ln                       # episode boundary (price back above the EMA line)
    first = S.arm_reset_trigger(broke, reset)     # one-shot: the first break of each episode
    ep = np.cumsum(reset)                         # episode id
    lvl = pd.Series(np.where(first, raw, np.nan)).groupby(ep).ffill()
    out["break_level"] = lvl.to_numpy()
    return out


def _break_cond(p) -> Cond:
    """The break of LEVEL B, read from the `break_level` column every version prepares.

    'close_below' (default) = the candle closes below B; 'full_below' = the whole candle is below B.
    Gated by `break_touches` (zone strength) when the version supplies it, and optionally by the
    level's distance from the EMA line.
    """
    mode = p.get("break_mode", "close_below")
    hi_c, cl_c = _candle_cols(p.get("candle_src", "real"))
    mt = int(p.get("sr_min_touches", 1))
    lo_d = p.get("break_min_ema_dist_pct")
    hi_d = p.get("break_max_ema_dist_pct")
    line = f"{p['grad_prefix']}_line"

    def fn(df):
        # close_below tests the live candidate (identical to the original rule); full_below tests the
        # LATCHED level that broke, so "the whole candle formed below B" stays meaningful after the
        # underlying S/R zone has stepped down.
        if mode == "full_below":
            lvl = df["break_level"].to_numpy(dtype=float)
            probe = df[hi_c].to_numpy()
        else:
            lvl = df["break_level_raw"].to_numpy(dtype=float)
            probe = df[cl_c].to_numpy()
        with np.errstate(invalid="ignore"):
            ok = (probe < lvl) & ~np.isnan(lvl)
            if "break_touches" in df.columns:
                t = df["break_touches"].to_numpy(dtype=float)
                ok = ok & (np.nan_to_num(t, nan=0.0) >= mt)
            if lo_d is not None or hi_d is not None:
                ln = df[line].to_numpy(dtype=float)
                dist = np.abs(lvl - ln) / ln * 100.0
                if lo_d is not None:
                    ok = ok & (dist >= float(lo_d))
                if hi_d is not None:
                    ok = ok & (dist <= float(hi_d))
        return ok

    return Cond(fn, f"break_{mode}(B)")


def _post_break_state(brk: Cond, reset: Cond) -> Cond:
    """Latched 'we are past a break this episode' — true from the break bar until price closes back
    above the EMA line (the episode reset)."""
    def fn(df):
        return S.latch(np.asarray(brk(df), bool), np.asarray(reset(df), bool))
    return Cond(fn, "post_break")


def _apply_timing(base_ctx: Cond, brk: Cond, reset: Cond, line: str, grad: str, p: dict) -> Cond:
    """Turn a 'context AND break' setup into the chosen entry-timing variant (Q2).

    break_close : fire on the break bar itself.
    wait_n      : fire `wait_n` bars after the break holds (still below the EMA).
    n_red       : fire once there are `n_red` consecutive red candles fully below the EMA after a break.
    pullback    : fire on the first red candle after a green pull-back up into the broken level.
    """
    mode = p["entry_mode"]
    src = p.get("candle_src", "real")
    if mode == "break_close":
        return base_ctx & brk

    post = _post_break_state(brk, reset)

    if mode == "wait_n":
        n = int(p["wait_n"])

        def fn(df):
            held = post(df).astype(float)
            run = np.zeros(len(held))
            # consecutive-bars-in-post-break count (cheap running length)
            c = 0
            for i, v in enumerate(held):
                c = c + 1 if v else 0
                run[i] = c
            return (run >= n) & below("close", line)(df)
        return base_ctx.__class__(fn, f"wait_n({n})")

    if mode == "n_red":
        return post & n_red_below(int(p["n_red"]), line, src=src) & below("close", line)

    if mode == "pullback":
        k = int(p["pullback_lookback"])

        def fn(df):
            in_break = post(df)
            greened = green(1, src=src)(df)
            # a green pull-back within the last k bars, then a red candle now
            recent_green = np.zeros(len(df), bool)
            gi = np.flatnonzero(greened)
            for j in gi:
                recent_green[j:j + k] = True
            return in_break & recent_green & red(1, src=src)(df)
        return base_ctx.__class__(fn, f"pullback({k})") & below("close", line)

    raise ValueError(f"unknown entry_mode {mode!r}: use break_close|wait_n|n_red|pullback")


# ---------------------------------------------------------------------------
# v1.0 — core trigger
# ---------------------------------------------------------------------------
class SupportBreakShortV10(Scenario):
    name, version = "support_break_short", "v1.0"
    description = "Core: close below the smoothed 200-EMA + a break of a support zone. One per episode."
    side = "short"
    default_params = {**_GRAD, **_SR, **_TIMING, **_BREAK}

    def prepare(self, df, p):
        out = add_ema_gradient(df, period=int(p["grad_period"]), smooth=int(p["grad_smooth"]),
                               lookback=int(p["grad_lookback"]), atr_period=int(p["grad_atr"]),
                               prefix=p["grad_prefix"])
        out = add_sr_zones(out, left=int(p["sr_left"]), right=int(p["sr_right"]),
                           cluster_pct=float(p["sr_cluster_pct"]), prefix="sr")
        if p.get("candle_src") == "ha":
            out = add_heikin_ashi(out)
        # LEVEL B = the support that was holding on the PRIOR bar (the zone table steps to the next
        # zone down the moment price closes through it, so we must look back one bar).
        out["break_level_raw"] = out["sr_support"].shift(1)
        out["break_touches"] = out["sr_support_touches"].shift(1)
        return _latch_break_level(out, p)

    def _context(self, p) -> Cond:
        pre, line = _grad_cols(p)
        ctx = below("close", line)                       # price has gone below the EMA line
        if p.get("grad_max") is not None:                # optional down-gradient gate
            ctx = ctx & below(pre, float(p["grad_max"]))
        return ctx

    def _break(self, p) -> Cond:
        return _break_cond(p)

    def entry(self, p) -> Cond:
        pre, line = _grad_cols(p)
        return _apply_timing(self._context(p), self._break(p), self.reset(p), line, pre, p)

    def reset(self, p) -> Cond:
        _pre, line = _grad_cols(p)
        return above("close", line)                      # re-arm when price closes back above the EMA


# ---------------------------------------------------------------------------
# v1.1 — + HH/HL structure into resistance before the break
# ---------------------------------------------------------------------------
class SupportBreakShortV11(SupportBreakShortV10):
    name, version = "support_break_short", "v1.1"
    description = ("Structure: v1.0 AND the prior swings tagged a resistance zone above (a topped-out "
                   "HH/HL run) before the support break.")
    default_params = {**SupportBreakShortV10.default_params,
                      "struct_lookback": 60, "struct_min_touches": 2}

    def prepare(self, df, p):
        out = super().prepare(df, p)
        out = add_swings(out, left=int(p["sr_left"]), right=int(p["sr_right"]))
        return out

    def _context(self, p) -> Cond:
        ctx = super()._context(p)
        look = int(p["struct_lookback"])
        mt = int(p["struct_min_touches"])

        def topped(df):
            # a resistance zone of >= mt touches was in force within the last `look` bars (the top)
            rt = df["sr_resistance_touches"].to_numpy()
            strong = np.nan_to_num(rt, nan=0.0) >= mt
            out = np.zeros(len(df), bool)
            idx = np.flatnonzero(strong)
            for j in idx:
                out[j:j + look] = True
            return out
        return ctx & Cond(topped, f"topped(<= {look}b,>= {mt}t)")


# ---------------------------------------------------------------------------
# v1.2 — no S/R structure: N red below the EMA + break of the prior-N low
# ---------------------------------------------------------------------------
class SupportBreakShortV12(SupportBreakShortV10):
    name, version = "support_break_short", "v1.2"
    description = ("No-structure: no S/R zones — N consecutive red candles below the EMA and a break "
                   "of the lowest low of the prior N bars.")
    default_params = {**_GRAD, **_TIMING, **_BREAK,
                      "prev_low_n": 20, "confirm_red": 2}

    def prepare(self, df, p):
        out = add_ema_gradient(df, period=int(p["grad_period"]), smooth=int(p["grad_smooth"]),
                               lookback=int(p["grad_lookback"]), atr_period=int(p["grad_atr"]),
                               prefix=p["grad_prefix"])
        if p.get("candle_src") == "ha":
            out = add_heikin_ashi(out)
        # LEVEL B = the lowest low of the prior `prev_low_n` bars (no S/R zones in this version)
        n = int(p["prev_low_n"])
        out["break_level_raw"] = out["low"].rolling(n, min_periods=n).min().shift(1)
        return _latch_break_level(out, p)

    def _context(self, p) -> Cond:
        pre, line = _grad_cols(p)
        src = p.get("candle_src", "real")
        ctx = full_below(line, n=1, src=src) & n_red_below(int(p["confirm_red"]), line, src=src)
        if p.get("grad_max") is not None:
            ctx = ctx & below(pre, float(p["grad_max"]))
        return ctx


# ---------------------------------------------------------------------------
# v1.3 — nearest-trough support when no strong zone is near
# ---------------------------------------------------------------------------
class SupportBreakShortV13(SupportBreakShortV10):
    name, version = "support_break_short", "v1.3"
    description = ("Nearest-trough: break the LOWER of the last two swing troughs (a hand-drawn "
                   "support) — for charts where the auto S/R zone is weak/absent.")
    default_params = {**_GRAD, **_SR, **_TIMING, **_BREAK, "trough_lr": 8}

    def prepare(self, df, p):
        out = super().prepare(df, p)
        lr = int(p["trough_lr"])
        out = add_swings(out, left=lr, right=lr)
        # lower of the last two confirmed troughs = the current confirmed swing low (already the most
        # recent), floored by the one before it — captured as the running min of swing lows over a
        # short memory so a fresh lower-low becomes the break level.
        sl = out["swing_last_low"]
        out["trough_support"] = sl.rolling(2, min_periods=1).min()
        # LEVEL B = the trough level held on the prior bar. No touches gate here: this version exists
        # precisely for charts where the clustered zone (and so its touch count) is weak/absent.
        out["break_level_raw"] = out["trough_support"].shift(1)
        out = out.drop(columns=["break_touches"], errors="ignore")
        return _latch_break_level(out, p)


register_scenario(SupportBreakShortV10())
register_scenario(SupportBreakShortV11())
register_scenario(SupportBreakShortV12())
register_scenario(SupportBreakShortV13())
