from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union, Any
import numpy as np

Ref = Union[str, int, float, Callable[..., Any]]

class RuleContextMixin:
    """
    Shared helper methods for simulation rule contexts.

    Requires the concrete context class to implement:
      self.v(col: str, shift: int = 0)
    """

    # ------------------------------------------------------------------
    # Generic value / boolean helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finite(x) -> bool:
        try:
            return bool(np.isfinite(x))
        except Exception:
            return False

    def flag(self, col: str, shift: int = 0) -> bool:
        val = self.v(col, shift=shift)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if self._finite(val):
            return bool(val)
        return False

    def flag_recent(self, col: str, lookback: int = 0, include_current: bool = True) -> bool:
        """
        True if boolean flag column was True recently.

        lookback=0 checks current only.
        lookback=3 checks current + previous 3 bars by default.
        """
        start_k = 0 if include_current else 1
        for k in range(start_k, int(lookback) + 1):
            if self.flag(col, shift=-k):
                return True
        return False

    def recent_value_when_flag(
        self,
        flag_col: str,
        value_col: str,
        predicate: Callable[[float], bool],
        lookback: int = 0,
        include_current: bool = True,
    ) -> bool:
        """
        Example:
          recent bull divergence where start RSI <= 30:
          c.recent_value_when_flag(
              "5m__rsi14__BULL_DIV",
              "5m__rsi14__BULL_START_RSI",
              lambda x: x <= 30,
              lookback=3
          )
        """
        start_k = 0 if include_current else 1
        for k in range(start_k, int(lookback) + 1):
            if self.flag(flag_col, shift=-k):
                val = self.v(value_col, shift=-k)
                if self._finite(val) and bool(predicate(float(val))):
                    return True
        return False

    # ------------------------------------------------------------------
    # Level-based helpers
    # ------------------------------------------------------------------
    def cross_up(self, col: str, level: float, lookback: int = 1, inclusive: bool = True) -> bool:
        cur = self.v(col, 0)
        if not self._finite(cur):
            return False

        cur_ok = (cur >= level) if inclusive else (cur > level)
        if not cur_ok:
            return False

        for k in range(1, int(lookback) + 1):
            prev = self.v(col, -k)
            if self._finite(prev) and prev < level:
                return True

        return False

    def cross_down(self, col: str, level: float, lookback: int = 1, inclusive: bool = True) -> bool:
        cur = self.v(col, 0)
        if not self._finite(cur):
            return False

        cur_ok = (cur <= level) if inclusive else (cur < level)
        if not cur_ok:
            return False

        for k in range(1, int(lookback) + 1):
            prev = self.v(col, -k)
            if self._finite(prev) and prev > level:
                return True

        return False

    def prev_all_below(self, col: str, level: float, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(1, int(n) + 1):
            val = self.v(col, -k)
            if not self._finite(val) or not (val < level):
                return False
        return True

    def prev_all_above(self, col: str, level: float, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(1, int(n) + 1):
            val = self.v(col, -k)
            if not self._finite(val) or not (val > level):
                return False
        return True

    def last_all_below(self, col: str, level: float, n: int) -> bool:
        """
        Includes current candle.

        n=5 checks current + previous 4.
        """
        if n <= 0:
            return True
        for k in range(0, int(n)):
            val = self.v(col, -k)
            if not self._finite(val) or not (val < level):
                return False
        return True

    def last_all_above(self, col: str, level: float, n: int) -> bool:
        """
        Includes current candle.

        n=5 checks current + previous 4.
        """
        if n <= 0:
            return True
        for k in range(0, int(n)):
            val = self.v(col, -k)
            if not self._finite(val) or not (val > level):
                return False
        return True

    def last_any_below(self, col: str, level: float, n: int) -> bool:
        for k in range(0, int(n)):
            val = self.v(col, -k)
            if self._finite(val) and val < level:
                return True
        return False

    def last_any_above(self, col: str, level: float, n: int) -> bool:
        for k in range(0, int(n)):
            val = self.v(col, -k)
            if self._finite(val) and val > level:
                return True
        return False

    # ------------------------------------------------------------------
    # Pair-based helpers: close vs MA, MACD vs signal, K vs D, etc.
    # ------------------------------------------------------------------
    def cross_up_pair(self, a: str, b: str, lookback: int = 1) -> bool:
        cur_a, cur_b = self.v(a, 0), self.v(b, 0)
        if not (self._finite(cur_a) and self._finite(cur_b)):
            return False
        if not (cur_a > cur_b):
            return False

        for k in range(1, int(lookback) + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if self._finite(pa) and self._finite(pb) and pa < pb:
                return True

        return False

    def cross_down_pair(self, a: str, b: str, lookback: int = 1) -> bool:
        cur_a, cur_b = self.v(a, 0), self.v(b, 0)
        if not (self._finite(cur_a) and self._finite(cur_b)):
            return False
        if not (cur_a < cur_b):
            return False

        for k in range(1, int(lookback) + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if self._finite(pa) and self._finite(pb) and pa > pb:
                return True

        return False

    def prev_all_below_pair(self, a: str, b: str, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(1, int(n) + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if not (self._finite(pa) and self._finite(pb) and pa < pb):
                return False
        return True

    def prev_all_above_pair(self, a: str, b: str, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(1, int(n) + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if not (self._finite(pa) and self._finite(pb) and pa > pb):
                return False
        return True

    def last_all_below_pair(self, a: str, b: str, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(0, int(n)):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if not (self._finite(pa) and self._finite(pb) and pa < pb):
                return False
        return True

    def last_all_above_pair(self, a: str, b: str, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(0, int(n)):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if not (self._finite(pa) and self._finite(pb) and pa > pb):
                return False
        return True

    # ------------------------------------------------------------------
    # Candle helpers
    # ------------------------------------------------------------------
    def candle_ohlc(
        self,
        shift: int = 0,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ):
        o = self.v(open_col, shift)
        h = self.v(high_col, shift)
        l = self.v(low_col, shift)
        c = self.v(close_col, shift)
        return o, h, l, c

    def candle_is_green(self, shift: int = 0, open_col: str = "open", close_col: str = "close") -> bool:
        o = self.v(open_col, shift)
        c = self.v(close_col, shift)
        return self._finite(o) and self._finite(c) and c > o

    def candle_is_red(self, shift: int = 0, open_col: str = "open", close_col: str = "close") -> bool:
        o = self.v(open_col, shift)
        c = self.v(close_col, shift)
        return self._finite(o) and self._finite(c) and c < o

    def last_n_candles(
        self,
        pattern: str,
        n: int,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        **kwargs,
    ) -> bool:
        """
        Includes current candle.

        Example:
          c.last_n_candles("green", n=3)
          c.last_n_candles("red", n=5)
        """
        if n <= 0:
            return True
        for k in range(0, int(n)):
            if not self.candle_pattern(
                pattern,
                shift=-k,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                **kwargs,
            ):
                return False
        return True

    def candle_any(
        self,
        patterns: Sequence[str],
        shift: int = 0,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        **kwargs,
    ) -> bool:
        return any(
            self.candle_pattern(
                p,
                shift=shift,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                **kwargs,
            )
            for p in patterns
        )

    def candle_pattern(
        self,
        pattern: str,
        shift: int = 0,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        **kwargs,
    ) -> bool:
        """
        Supported patterns:
          green, red, doji,
          hammer, green_hammer, red_hammer, bullish_hammer, bearish_hammer,
          shooting_star,
          bullish_engulfing, bearish_engulfing,
          bullish_harami, bearish_harami
        """
        p = pattern.lower().replace(" ", "_").replace("-", "_").strip()

        o, h, l, c = self.candle_ohlc(shift, open_col, high_col, low_col, close_col)

        if not all(self._finite(x) for x in [o, h, l, c]):
            return False

        rng = h - l
        if rng <= 0:
            return False

        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l

        is_green = c > o
        is_red = c < o

        if p in ("green", "bullish_candle"):
            return is_green

        if p in ("red", "bearish_candle"):
            return is_red

        if p == "doji":
            max_body_pct = float(kwargs.get("doji_max_body_pct", 0.10))
            return body <= rng * max_body_pct

        # Hammer-like candle
        if p in ("hammer", "green_hammer", "red_hammer", "bullish_hammer", "bearish_hammer"):
            lower_to_body = float(kwargs.get("hammer_lower_wick_to_body", 2.0))
            upper_to_body = float(kwargs.get("hammer_upper_wick_to_body_max", 0.75))
            max_body_pct = float(kwargs.get("hammer_body_max_range_pct", 0.40))

            # avoid body=0 division problem
            effective_body = max(body, rng * 0.02)

            hammer_shape = (
                lower >= lower_to_body * effective_body
                and upper <= upper_to_body * effective_body
                and body <= rng * max_body_pct
            )

            if not hammer_shape:
                return False

            if p in ("green_hammer", "bullish_hammer"):
                return is_green
            if p in ("red_hammer", "bearish_hammer"):
                return is_red
            return True

        if p == "shooting_star":
            upper_to_body = float(kwargs.get("star_upper_wick_to_body", 2.0))
            lower_to_body = float(kwargs.get("star_lower_wick_to_body_max", 0.75))
            max_body_pct = float(kwargs.get("star_body_max_range_pct", 0.40))

            effective_body = max(body, rng * 0.02)

            return (
                upper >= upper_to_body * effective_body
                and lower <= lower_to_body * effective_body
                and body <= rng * max_body_pct
            )

        # Two-candle patterns need previous candle
        po, ph, pl, pc = self.candle_ohlc(shift - 1, open_col, high_col, low_col, close_col)
        if not all(self._finite(x) for x in [po, ph, pl, pc]):
            return False

        prev_green = pc > po
        prev_red = pc < po

        cur_body_low = min(o, c)
        cur_body_high = max(o, c)
        prev_body_low = min(po, pc)
        prev_body_high = max(po, pc)

        if p == "bullish_engulfing":
            return (
                prev_red
                and is_green
                and cur_body_low <= prev_body_low
                and cur_body_high >= prev_body_high
            )

        if p == "bearish_engulfing":
            return (
                prev_green
                and is_red
                and cur_body_low <= prev_body_low
                and cur_body_high >= prev_body_high
            )

        if p == "bullish_harami":
            return (
                prev_red
                and is_green
                and cur_body_low >= prev_body_low
                and cur_body_high <= prev_body_high
            )

        if p == "bearish_harami":
            return (
                prev_green
                and is_red
                and cur_body_low >= prev_body_low
                and cur_body_high <= prev_body_high
            )

        raise ValueError(f"Unknown candle pattern: {pattern}")

    def ref(self, value: Ref, shift: int = 0):
        """
        Resolve a rule reference.
    
        Supported:
          80000                         -> static number
          "close"                       -> current row value from column
          "5m__ema__EMA_100"             -> current row value from column
          lambda c, shift: ...           -> custom dynamic reference
          lambda c: ...                  -> custom dynamic reference
        """
        if isinstance(value, str):
            return self.v(value, shift=shift)
    
        if callable(value):
            try:
                return value(self, shift)
            except TypeError:
                return value(self)
    
        return value
    
    
    def compare(self, left: Ref, op: str, right: Ref, shift: int = 0) -> bool:
        """
        Generic comparison:
          c.compare("5m__ema__EMA_50", "<", 80000)
          c.compare("5m__ema__EMA_50", "<", "5m__ema__EMA_100")
          c.compare("5m__ema__EMA_50", "<", "close")
        """
        l = self.ref(left, shift=shift)
        r = self.ref(right, shift=shift)
    
        if not (self._finite(l) and self._finite(r)):
            return False
    
        op = op.strip()
    
        if op == ">":
            return bool(l > r)
        if op == ">=":
            return bool(l >= r)
        if op == "<":
            return bool(l < r)
        if op == "<=":
            return bool(l <= r)
        if op == "==":
            return bool(l == r)
        if op == "!=":
            return bool(l != r)
    
        raise ValueError(f"Unsupported comparison operator: {op}")
    
    
    # Friendly aliases
    def gt(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, ">", right, shift=shift)
    
    def gte(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, ">=", right, shift=shift)
    
    def lt(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, "<", right, shift=shift)
    
    def lte(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, "<=", right, shift=shift)
    
    
    def prev_all_compare(self, left: Ref, op: str, right: Ref, n: int) -> bool:
        """
        Previous N completed candles only.
        Example:
          previous 3 candles: EMA50 < EMA100
        """
        if n <= 0:
            return True
    
        for k in range(1, int(n) + 1):
            if not self.compare(left, op, right, shift=-k):
                return False
    
        return True
    
    
    def last_all_compare(self, left: Ref, op: str, right: Ref, n: int) -> bool:
        """
        Includes current candle.
        n=3 means current candle + previous 2 candles.
        """
        if n <= 0:
            return True
    
        for k in range(0, int(n)):
            if not self.compare(left, op, right, shift=-k):
                return False
    
        return True
    
    
    def last_any_compare(self, left: Ref, op: str, right: Ref, n: int) -> bool:
        """
        Includes current candle.
        True if condition happened at least once in last N bars.
        """
        if n <= 0:
            return False
    
        for k in range(0, int(n)):
            if self.compare(left, op, right, shift=-k):
                return True
    
        return False
    
    
    def cross_up_ref(self, left: Ref, right: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        """
        Dynamic cross up:
          left currently above/right, and left was below right within lookback.
    
        Examples:
          c.cross_up_ref("rsi14__RSI", 20)
          c.cross_up_ref("close", "ema__EMA_50")
          c.cross_up_ref("5m__ema__EMA_50", "5m__ema__EMA_100")
        """
        l0 = self.ref(left, shift=0)
        r0 = self.ref(right, shift=0)
    
        if not (self._finite(l0) and self._finite(r0)):
            return False
    
        current_ok = (l0 >= r0) if inclusive else (l0 > r0)
        if not current_ok:
            return False
    
        for k in range(1, int(lookback) + 1):
            lk = self.ref(left, shift=-k)
            rk = self.ref(right, shift=-k)
    
            if self._finite(lk) and self._finite(rk) and lk < rk:
                return True
    
        return False
    
    
    def cross_down_ref(self, left: Ref, right: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        """
        Dynamic cross down:
          left currently below/right, and left was above right within lookback.
        """
        l0 = self.ref(left, shift=0)
        r0 = self.ref(right, shift=0)
    
        if not (self._finite(l0) and self._finite(r0)):
            return False
    
        current_ok = (l0 <= r0) if inclusive else (l0 < r0)
        if not current_ok:
            return False
    
        for k in range(1, int(lookback) + 1):
            lk = self.ref(left, shift=-k)
            rk = self.ref(right, shift=-k)
    
            if self._finite(lk) and self._finite(rk) and lk > rk:
                return True
    
        return False
    
    
    # Backward-compatible dynamic upgrades for existing helpers
    def cross_up(self, col: Ref, level: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        return self.cross_up_ref(col, level, lookback=lookback, inclusive=inclusive)
    
    def cross_down(self, col: Ref, level: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        return self.cross_down_ref(col, level, lookback=lookback, inclusive=inclusive)
    
    def prev_all_below(self, col: Ref, level: Ref, n: int) -> bool:
        return self.prev_all_compare(col, "<", level, n=n)
    
    def prev_all_above(self, col: Ref, level: Ref, n: int) -> bool:
        return self.prev_all_compare(col, ">", level, n=n)
    
    def last_all_below(self, col: Ref, level: Ref, n: int) -> bool:
        return self.last_all_compare(col, "<", level, n=n)
    
    def last_all_above(self, col: Ref, level: Ref, n: int) -> bool:
        return self.last_all_compare(col, ">", level, n=n)

    # ------------------------------------------------------------------
# Generic crossover + confirmation helpers
# ------------------------------------------------------------------

    def crossed_up_ref(
        self,
        left,
        right,
        cross_shift: int = 0,
        prev_op: str = "<=",
        cur_op: str = ">",
    ) -> bool:
        """
        True when `left` crossed above `right` on the candle at `cross_shift`.
    
        Example:
          cross_shift=0:
            previous candle: left <= right
            current candle:  left > right
    
          cross_shift=-1:
            candle -2: left <= right
            candle -1: left > right
    
        Works with:
          left/right = static value, column name, or callable.
        """
        return (
            self.compare(left, prev_op, right, shift=cross_shift - 1)
            and self.compare(left, cur_op, right, shift=cross_shift)
        )
    
    
    def crossed_down_ref(
        self,
        left,
        right,
        cross_shift: int = 0,
        prev_op: str = ">=",
        cur_op: str = "<",
    ) -> bool:
        """
        True when `left` crossed below `right` on the candle at `cross_shift`.
    
        Example:
          cross_shift=0:
            previous candle: left >= right
            current candle:  left < right
    
          cross_shift=-1:
            candle -2: left >= right
            candle -1: left < right
        """
        return (
            self.compare(left, prev_op, right, shift=cross_shift - 1)
            and self.compare(left, cur_op, right, shift=cross_shift)
        )
    
    def cross_then_confirm(
        self,
        left,
        direction: str,
        right,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        cross_shift: int | None = None,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        confirm_fn=None,
    ) -> bool:
        """
        Generic scalable helper:
    
        1) A crossover happens first.
        2) Then N confirmation candles must happen after the crossover.
    
        By default:
          confirm_shift=0 means the current candle is the final confirmation candle.
          confirm_bars=N means check the last N candles ending at confirm_shift.
          cross_shift=None means crossover is automatically placed one candle
          before the confirmation window.
    
        Example for confirm_bars=2:
          Candle -3: before cross
          Candle -2: crossover
          Candle -1: confirmation 1
          Candle  0: confirmation 2
    
        Example for confirm_bars=5:
          Candle -6: before cross
          Candle -5: crossover
          Candle -4,-3,-2,-1,0: five confirmations
        """
    
        direction = direction.lower().strip()
        confirm_bars = int(confirm_bars)
    
        if confirm_bars <= 0:
            raise ValueError("confirm_bars must be >= 1")
    
        # Auto-scalable behavior:
        # If final confirmation is candle 0 and confirm_bars=2,
        # confirmation window is -1, 0 and crossover is -2.
        if cross_shift is None:
            cross_shift = confirm_shift - confirm_bars
    
        first_confirm_shift = confirm_shift - confirm_bars + 1
        confirm_shifts = range(first_confirm_shift, confirm_shift + 1)
    
        # Make sure confirmation candles happen AFTER crossover candle.
        if first_confirm_shift <= cross_shift:
            raise ValueError(
                "Invalid confirmation window. Confirmation candles must come after the crossover candle. "
                f"Got cross_shift={cross_shift}, first_confirm_shift={first_confirm_shift}."
            )
    
        if direction == "up":
            crossed = self.crossed_up_ref(left, right, cross_shift=cross_shift)
            same_side_op = ">"
        elif direction == "down":
            crossed = self.crossed_down_ref(left, right, cross_shift=cross_shift)
            same_side_op = "<"
        else:
            raise ValueError("direction must be 'up' or 'down'.")
    
        if not crossed:
            return False
    
        for sh in confirm_shifts:
            if require_same_side and not self.compare(left, same_side_op, right, shift=sh):
                return False
    
            if confirm_pattern is not None:
                if not self.candle_pattern(confirm_pattern, shift=sh):
                    return False
    
            if confirm_fn is not None:
                try:
                    if not bool(confirm_fn(self, sh)):
                        return False
                except TypeError:
                    if not bool(confirm_fn(self)):
                        return False
    
        return True
    
    
    def cross_up_then_confirm(
        self,
        left,
        right,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        cross_shift: int | None = None,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        confirm_fn=None,
    ) -> bool:
        return self.cross_then_confirm(
            left=left,
            direction="up",
            right=right,
            confirm_bars=confirm_bars,
            confirm_shift=confirm_shift,
            cross_shift=cross_shift,
            confirm_pattern=confirm_pattern,
            require_same_side=require_same_side,
            confirm_fn=confirm_fn,
        )
    
    
    def cross_down_then_confirm(
        self,
        left,
        right,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        cross_shift: int | None = None,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        confirm_fn=None,
    ) -> bool:
        return self.cross_then_confirm(
            left=left,
            direction="down",
            right=right,
            confirm_bars=confirm_bars,
            confirm_shift=confirm_shift,
            cross_shift=cross_shift,
            confirm_pattern=confirm_pattern,
            require_same_side=require_same_side,
            confirm_fn=confirm_fn,
        )

# ------------------------------------------------------------------
# Cross happened earlier, then confirmation happens later
# ------------------------------------------------------------------

    def _crossed_direction_ref(self, left, direction: str, right, cross_shift: int) -> bool:
        direction = direction.lower().strip()
    
        if direction == "up":
            return self.crossed_up_ref(left, right, cross_shift=cross_shift)
    
        if direction == "down":
            return self.crossed_down_ref(left, right, cross_shift=cross_shift)
    
        raise ValueError("direction must be 'up' or 'down'.")
    
    
    def _opposite_direction(self, direction: str) -> str:
        direction = direction.lower().strip()
        if direction == "up":
            return "down"
        if direction == "down":
            return "up"
        raise ValueError("direction must be 'up' or 'down'.")
    
    
    def _confirm_window_ok(
        self,
        left,
        direction: str,
        right,
        confirm_bars: int,
        confirm_shift: int = 0,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        confirm_fn=None,
    ) -> bool:
        """
        Checks the confirmation candles.
    
        Example:
          confirm_bars=2, confirm_shift=0
          checks candle -1 and candle 0.
        """
        direction = direction.lower().strip()
        confirm_bars = int(confirm_bars)
    
        if confirm_bars <= 0:
            raise ValueError("confirm_bars must be >= 1.")
    
        if direction == "up":
            same_side_op = ">"
        elif direction == "down":
            same_side_op = "<"
        else:
            raise ValueError("direction must be 'up' or 'down'.")
    
        first_confirm_shift = confirm_shift - confirm_bars + 1
    
        for sh in range(first_confirm_shift, confirm_shift + 1):
            if require_same_side and not self.compare(left, same_side_op, right, shift=sh):
                return False
    
            if confirm_pattern is not None:
                if not self.candle_pattern(confirm_pattern, shift=sh):
                    return False
    
            if confirm_fn is not None:
                try:
                    if not bool(confirm_fn(self, sh)):
                        return False
                except TypeError:
                    if not bool(confirm_fn(self)):
                        return False
    
        return True
    
    
    def _has_opposite_cross_after(
        self,
        left,
        direction: str,
        right,
        cross_shift: int,
        end_shift: int = 0,
    ) -> bool:
        """
        Checks whether an opposite crossover happened after the original cross.
    
        Example:
          Long case:
            after close crossed above EMA, did close cross back below EMA?
        """
        opposite = self._opposite_direction(direction)
    
        # Example: cross_shift=-8, end_shift=0
        # check shifts -7, -6, ..., 0 for opposite cross.
        for sh in range(cross_shift + 1, end_shift + 1):
            if self._crossed_direction_ref(left, opposite, right, cross_shift=sh):
                return True
    
        return False
    
    
    def cross_with_later_confirm(
        self,
        left,
        direction: str,
        right,
        max_bars_since_cross: int = 20,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        require_no_opposite_cross: bool = True,
        confirm_fn=None,
    ) -> bool:
        """
        Scalable deferred-confirmation helper.
    
        Meaning:
          1) left crossed right sometime recently.
          2) price/indicator has not crossed back, if require_no_opposite_cross=True.
          3) the latest N candles confirm the direction.
    
        Example for short:
          direction="down"
          confirm_bars=2
          confirm_pattern="red"
    
          Means:
            close crossed below EMA recently,
            did not cross back above EMA,
            last 2 candles are red and still below EMA.
    
        This fixes the problem where the trade is missed if confirmation
        does not appear immediately after the crossover candle.
        """
        direction = direction.lower().strip()
        max_bars_since_cross = int(max_bars_since_cross)
        confirm_bars = int(confirm_bars)
    
        if max_bars_since_cross <= 0:
            raise ValueError("max_bars_since_cross must be >= 1.")
    
        if confirm_bars <= 0:
            raise ValueError("confirm_bars must be >= 1.")
    
        # First confirm candle must be after the cross.
        first_confirm_shift = confirm_shift - confirm_bars + 1
    
        # Confirmation itself must be valid right now.
        if not self._confirm_window_ok(
            left=left,
            direction=direction,
            right=right,
            confirm_bars=confirm_bars,
            confirm_shift=confirm_shift,
            confirm_pattern=confirm_pattern,
            require_same_side=require_same_side,
            confirm_fn=confirm_fn,
        ):
            return False
    
        # Search for the most recent valid cross before the confirmation window.
        # Example:
        #   confirm_bars=2, confirm_shift=0
        #   confirmation candles are -1 and 0
        #   latest allowed cross is -2
        latest_allowed_cross_shift = first_confirm_shift - 1
        oldest_allowed_cross_shift = -max_bars_since_cross
    
        for cross_shift in range(latest_allowed_cross_shift, oldest_allowed_cross_shift - 1, -1):
            crossed = self._crossed_direction_ref(
                left=left,
                direction=direction,
                right=right,
                cross_shift=cross_shift,
            )
    
            if not crossed:
                continue
    
            if require_no_opposite_cross:
                if self._has_opposite_cross_after(
                    left=left,
                    direction=direction,
                    right=right,
                    cross_shift=cross_shift,
                    end_shift=confirm_shift,
                ):
                    continue
    
            return True
    
        return False
    
    
    def cross_up_with_later_confirm(
        self,
        left,
        right,
        max_bars_since_cross: int = 20,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        require_no_opposite_cross: bool = True,
        confirm_fn=None,
    ) -> bool:
        return self.cross_with_later_confirm(
            left=left,
            direction="up",
            right=right,
            max_bars_since_cross=max_bars_since_cross,
            confirm_bars=confirm_bars,
            confirm_shift=confirm_shift,
            confirm_pattern=confirm_pattern,
            require_same_side=require_same_side,
            require_no_opposite_cross=require_no_opposite_cross,
            confirm_fn=confirm_fn,
        )
    
    
    def cross_down_with_later_confirm(
        self,
        left,
        right,
        max_bars_since_cross: int = 20,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        confirm_pattern: str | None = None,
        require_same_side: bool = True,
        require_no_opposite_cross: bool = True,
        confirm_fn=None,
    ) -> bool:
        return self.cross_with_later_confirm(
            left=left,
            direction="down",
            right=right,
            max_bars_since_cross=max_bars_since_cross,
            confirm_bars=confirm_bars,
            confirm_shift=confirm_shift,
            confirm_pattern=confirm_pattern,
            require_same_side=require_same_side,
            require_no_opposite_cross=require_no_opposite_cross,
            confirm_fn=confirm_fn,
        )

    def consecutive_candles(
        self,
        pattern: str,
        n: int,
        end_shift: int = 0,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        **kwargs,
    ) -> bool:
        """
        Checks whether the last N candles ending at `end_shift` match a candle pattern.
    
        end_shift=0:
          checks current candle and previous n-1 candles.
    
        Example:
          n=3, end_shift=0 checks shifts: -2, -1, 0
          n=3, end_shift=-1 checks shifts: -3, -2, -1
        """
        n = int(n)
        if n <= 0:
            return True
    
        first_shift = end_shift - n + 1
    
        for sh in range(first_shift, end_shift + 1):
            if not self.candle_pattern(
                pattern,
                shift=sh,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                **kwargs,
            ):
                return False
    
        return True
    
    
    def consecutive_green(self, n: int, end_shift: int = 0) -> bool:
        return self.consecutive_candles("green", n=n, end_shift=end_shift)
    
    
    def consecutive_red(self, n: int, end_shift: int = 0) -> bool:
        return self.consecutive_candles("red", n=n, end_shift=end_shift)
    
    
    def has_consecutive_candles(
        self,
        pattern: str,
        n: int,
        lookback: int,
        end_shift: int = 0,
        **kwargs,
    ) -> bool:
        """
        Searches for at least one run of N consecutive candles inside a wider lookback window.
    
        Example:
          has_consecutive_candles("green", n=3, lookback=10)
    
        Means:
          within the last 10 candles, there was at least one run of 3 green candles.
        """
        n = int(n)
        lookback = int(lookback)
    
        if n <= 0:
            return True
    
        if lookback < n:
            return False
    
        latest_end = end_shift
        earliest_end = end_shift - lookback + n
    
        for candidate_end in range(latest_end, earliest_end - 1, -1):
            if self.consecutive_candles(pattern, n=n, end_shift=candidate_end, **kwargs):
                return True
    
        return False