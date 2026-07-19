from __future__ import annotations

from typing import Callable, Optional, Sequence, Union, Any, Iterable
import numpy as np

Ref = Union[str, int, float, Callable[..., Any]]


class RuleContextMixin:
    """
    Shared helper methods for simulation rule contexts.

    Requires the concrete context class to implement:
      self.v(col: str, shift: int = 0)

    This mixin is intentionally generic. It supports:
      - static thresholds
      - dynamic column references
      - callable references
      - candle patterns
      - crossovers
      - delayed confirmations
      - multi-reference filters, e.g. close above all EMAs
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

    def flag(self, col: str, shift: int = 0) -> bool:
        val = self.v(col, shift=shift)

        if isinstance(val, (bool, np.bool_)):
            return bool(val)

        if self._finite(val):
            return bool(val)

        return False

    def flag_recent(self, col: str, lookback: int = 0, include_current: bool = True) -> bool:
        """
        True if a boolean flag column was True recently.

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
              lookback=3,
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
    # Generic comparisons
    # ------------------------------------------------------------------

    def compare(self, left: Ref, op: str, right: Ref, shift: int = 0) -> bool:
        """
        Generic comparison.

        Examples:
          c.compare("close", ">", "1m__ema__EMA_50")
          c.compare("5m__ema__EMA_50", "<", "5m__ema__EMA_100")
          c.compare("rsi14__RSI", ">=", 70)
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

    def gt(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, ">", right, shift=shift)

    def gte(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, ">=", right, shift=shift)

    def lt(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, "<", right, shift=shift)

    def lte(self, left: Ref, right: Ref, shift: int = 0) -> bool:
        return self.compare(left, "<=", right, shift=shift)

    # ------------------------------------------------------------------
    # Multi-reference comparison helpers
    # ------------------------------------------------------------------

    def all_compare(self, left: Ref, op: str, rights: Iterable[Ref], shift: int = 0) -> bool:
        """
        True if left satisfies the comparison against every item in rights.

        Example:
          c.all_compare("close", ">", ["1m__ema__EMA_50", "5m__ema__EMA_100"])
        """
        return all(self.compare(left, op, right, shift=shift) for right in rights)

    def any_compare(self, left: Ref, op: str, rights: Iterable[Ref], shift: int = 0) -> bool:
        """
        True if left satisfies the comparison against at least one item in rights.
        """
        return any(self.compare(left, op, right, shift=shift) for right in rights)

    def close_above_all(self, refs: Iterable[Ref], shift: int = 0) -> bool:
        return self.all_compare("close", ">", refs, shift=shift)

    def close_below_all(self, refs: Iterable[Ref], shift: int = 0) -> bool:
        return self.all_compare("close", "<", refs, shift=shift)

    def close_above_any(self, refs: Iterable[Ref], shift: int = 0) -> bool:
        return self.any_compare("close", ">", refs, shift=shift)

    def close_below_any(self, refs: Iterable[Ref], shift: int = 0) -> bool:
        return self.any_compare("close", "<", refs, shift=shift)

    # ------------------------------------------------------------------
    # N-candle comparisons
    # ------------------------------------------------------------------

    def prev_all_compare(self, left: Ref, op: str, right: Ref, n: int) -> bool:
        """
        Previous N completed candles only.

        n=3 checks shifts:
          -3, -2, -1
        """
        n = int(n)
        if n <= 0:
            return True

        for sh in range(-n, 0):
            if not self.compare(left, op, right, shift=sh):
                return False

        return True

    def last_all_compare(self, left: Ref, op: str, right: Ref, n: int, end_shift: int = 0) -> bool:
        """
        Includes the candle at end_shift.

        n=3, end_shift=0 checks shifts:
          -2, -1, 0
        """
        n = int(n)
        if n <= 0:
            return True

        first_shift = end_shift - n + 1

        for sh in range(first_shift, end_shift + 1):
            if not self.compare(left, op, right, shift=sh):
                return False

        return True

    def last_any_compare(self, left: Ref, op: str, right: Ref, n: int, end_shift: int = 0) -> bool:
        """
        True if condition happened at least once in the last N candles.
        Includes the candle at end_shift.
        """
        n = int(n)
        if n <= 0:
            return False

        first_shift = end_shift - n + 1

        for sh in range(first_shift, end_shift + 1):
            if self.compare(left, op, right, shift=sh):
                return True

        return False

    def last_all_compare_all(
        self,
        left: Ref,
        op: str,
        rights: Iterable[Ref],
        n: int,
        end_shift: int = 0,
    ) -> bool:
        """
        For the last N candles, left must satisfy comparison against all references.

        Example:
          Last 3 closes above all EMAs:
            c.last_all_compare_all("close", ">", EMA_FILTERS, n=3)
        """
        n = int(n)
        if n <= 0:
            return True

        first_shift = end_shift - n + 1

        for sh in range(first_shift, end_shift + 1):
            if not self.all_compare(left, op, rights, shift=sh):
                return False

        return True

    def prev_all_compare_all(self, left: Ref, op: str, rights: Iterable[Ref], n: int) -> bool:
        """
        Previous N completed candles only.
        """
        n = int(n)
        if n <= 0:
            return True

        for sh in range(-n, 0):
            if not self.all_compare(left, op, rights, shift=sh):
                return False

        return True

    def last_n_closes_above_all(self, refs: Iterable[Ref], n: int, end_shift: int = 0) -> bool:
        return self.last_all_compare_all("close", ">", refs, n=n, end_shift=end_shift)

    def last_n_closes_below_all(self, refs: Iterable[Ref], n: int, end_shift: int = 0) -> bool:
        return self.last_all_compare_all("close", "<", refs, n=n, end_shift=end_shift)

    def prev_n_closes_above_all(self, refs: Iterable[Ref], n: int) -> bool:
        return self.prev_all_compare_all("close", ">", refs, n=n)

    def prev_n_closes_below_all(self, refs: Iterable[Ref], n: int) -> bool:
        return self.prev_all_compare_all("close", "<", refs, n=n)

    # ------------------------------------------------------------------
    # Backward-compatible level-style helpers
    # ------------------------------------------------------------------

    def cross_up(self, col: Ref, level: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        return self.cross_up_ref(col, level, lookback=lookback, inclusive=inclusive)

    def cross_down(self, col: Ref, level: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        return self.cross_down_ref(col, level, lookback=lookback, inclusive=inclusive)

    def prev_all_below(self, col: Ref, level: Ref, n: int) -> bool:
        return self.prev_all_compare(col, "<", level, n=n)

    def prev_all_above(self, col: Ref, level: Ref, n: int) -> bool:
        return self.prev_all_compare(col, ">", level, n=n)

    def last_all_below(self, col: Ref, level: Ref, n: int, end_shift: int = 0) -> bool:
        return self.last_all_compare(col, "<", level, n=n, end_shift=end_shift)

    def last_all_above(self, col: Ref, level: Ref, n: int, end_shift: int = 0) -> bool:
        return self.last_all_compare(col, ">", level, n=n, end_shift=end_shift)

    def last_any_below(self, col: Ref, level: Ref, n: int, end_shift: int = 0) -> bool:
        return self.last_any_compare(col, "<", level, n=n, end_shift=end_shift)

    def last_any_above(self, col: Ref, level: Ref, n: int, end_shift: int = 0) -> bool:
        return self.last_any_compare(col, ">", level, n=n, end_shift=end_shift)

    # ------------------------------------------------------------------
    # Pair-style wrappers
    # ------------------------------------------------------------------

    def cross_up_pair(self, a: Ref, b: Ref, lookback: int = 1) -> bool:
        return self.cross_up_ref(a, b, lookback=lookback, inclusive=False)

    def cross_down_pair(self, a: Ref, b: Ref, lookback: int = 1) -> bool:
        return self.cross_down_ref(a, b, lookback=lookback, inclusive=False)

    def prev_all_below_pair(self, a: Ref, b: Ref, n: int) -> bool:
        return self.prev_all_compare(a, "<", b, n=n)

    def prev_all_above_pair(self, a: Ref, b: Ref, n: int) -> bool:
        return self.prev_all_compare(a, ">", b, n=n)

    def last_all_below_pair(self, a: Ref, b: Ref, n: int, end_shift: int = 0) -> bool:
        return self.last_all_compare(a, "<", b, n=n, end_shift=end_shift)

    def last_all_above_pair(self, a: Ref, b: Ref, n: int, end_shift: int = 0) -> bool:
        return self.last_all_compare(a, ">", b, n=n, end_shift=end_shift)

    # ------------------------------------------------------------------
    # Crossover helpers
    # ------------------------------------------------------------------

    def cross_up_ref(self, left: Ref, right: Ref, lookback: int = 1, inclusive: bool = True) -> bool:
        """
        Dynamic cross up:
          left is currently above/right,
          and left was below right within lookback candles.
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
          left is currently below/right,
          and left was above right within lookback candles.
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

    def crossed_up_ref(
        self,
        left: Ref,
        right: Ref,
        cross_shift: int = 0,
        prev_op: str = "<=",
        cur_op: str = ">",
    ) -> bool:
        """
        True when left crossed above right on the candle at cross_shift.

        cross_shift=0:
          candle -1: left <= right
          candle  0: left > right

        cross_shift=-1:
          candle -2: left <= right
          candle -1: left > right
        """
        return (
            self.compare(left, prev_op, right, shift=cross_shift - 1)
            and self.compare(left, cur_op, right, shift=cross_shift)
        )

    def crossed_down_ref(
        self,
        left: Ref,
        right: Ref,
        cross_shift: int = 0,
        prev_op: str = ">=",
        cur_op: str = "<",
    ) -> bool:
        """
        True when left crossed below right on the candle at cross_shift.
        """
        return (
            self.compare(left, prev_op, right, shift=cross_shift - 1)
            and self.compare(left, cur_op, right, shift=cross_shift)
        )

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

        if p in ("hammer", "green_hammer", "red_hammer", "bullish_hammer", "bearish_hammer"):
            lower_to_body = float(kwargs.get("hammer_lower_wick_to_body", 2.0))
            upper_to_body = float(kwargs.get("hammer_upper_wick_to_body_max", 0.75))
            max_body_pct = float(kwargs.get("hammer_body_max_range_pct", 0.40))

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

        po, ph, pl, pc = self.candle_ohlc(
            shift - 1,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        )

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
        Checks whether the last N candles ending at end_shift match a candle pattern.

        n=3, end_shift=0 checks:
          -2, -1, 0

        n=3, end_shift=-1 checks:
          -3, -2, -1
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

    def last_n_candles(self, pattern: str, n: int, **kwargs) -> bool:
        """
        Backward-compatible alias. Includes current candle.
        """
        return self.consecutive_candles(pattern, n=n, end_shift=0, **kwargs)

    def consecutive_green(self, n: int, end_shift: int = 0) -> bool:
        return self.consecutive_candles("green", n=n, end_shift=end_shift)

    def consecutive_red(self, n: int, end_shift: int = 0) -> bool:
        return self.consecutive_candles("red", n=n, end_shift=end_shift)

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
          c.has_consecutive_candles("green", n=3, lookback=10)
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

    # ------------------------------------------------------------------
    # Cross + immediate confirmation
    # ------------------------------------------------------------------

    def cross_then_confirm(
        self,
        left: Ref,
        direction: str,
        right: Ref,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        cross_shift: Optional[int] = None,
        confirm_pattern: Optional[str] = None,
        require_same_side: bool = True,
        confirm_fn=None,
    ) -> bool:
        """
        A crossover happens first, then N confirmation candles happen immediately after.
        """
        direction = direction.lower().strip()
        confirm_bars = int(confirm_bars)

        if confirm_bars <= 0:
            raise ValueError("confirm_bars must be >= 1.")

        if cross_shift is None:
            cross_shift = confirm_shift - confirm_bars

        first_confirm_shift = confirm_shift - confirm_bars + 1

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

        for sh in range(first_confirm_shift, confirm_shift + 1):
            if require_same_side and not self.compare(left, same_side_op, right, shift=sh):
                return False

            if confirm_pattern is not None and not self.candle_pattern(confirm_pattern, shift=sh):
                return False

            if confirm_fn is not None:
                try:
                    ok = bool(confirm_fn(self, sh))
                except TypeError:
                    ok = bool(confirm_fn(self))

                if not ok:
                    return False

        return True

    def cross_up_then_confirm(
        self,
        left: Ref,
        right: Ref,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        cross_shift: Optional[int] = None,
        confirm_pattern: Optional[str] = None,
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
        left: Ref,
        right: Ref,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        cross_shift: Optional[int] = None,
        confirm_pattern: Optional[str] = None,
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
    # Cross happened earlier, confirmation happens later
    # ------------------------------------------------------------------

    def _crossed_direction_ref(self, left: Ref, direction: str, right: Ref, cross_shift: int) -> bool:
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
        left: Ref,
        direction: str,
        right: Ref,
        confirm_bars: int,
        confirm_shift: int = 0,
        confirm_pattern: Optional[str] = None,
        require_same_side: bool = True,
        confirm_fn=None,
    ) -> bool:
        """
        Checks the confirmation candles.

        confirm_bars=2, confirm_shift=0 checks:
          -1, 0
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

            if confirm_pattern is not None and not self.candle_pattern(confirm_pattern, shift=sh):
                return False

            if confirm_fn is not None:
                try:
                    ok = bool(confirm_fn(self, sh))
                except TypeError:
                    ok = bool(confirm_fn(self))

                if not ok:
                    return False

        return True

    def _has_opposite_cross_after(
        self,
        left: Ref,
        direction: str,
        right: Ref,
        cross_shift: int,
        end_shift: int = 0,
    ) -> bool:
        opposite = self._opposite_direction(direction)

        for sh in range(cross_shift + 1, end_shift + 1):
            if self._crossed_direction_ref(left, opposite, right, cross_shift=sh):
                return True

        return False

    def cross_with_later_confirm(
        self,
        left: Ref,
        direction: str,
        right: Ref,
        max_bars_since_cross: int = 20,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        confirm_pattern: Optional[str] = None,
        require_same_side: bool = True,
        require_no_opposite_cross: bool = True,
        confirm_fn=None,
    ) -> bool:
        """
        Deferred-confirmation helper.

        Meaning:
          1) left crossed right sometime recently.
          2) left did not cross back, if require_no_opposite_cross=True.
          3) the latest N candles confirm the direction.

        This avoids missing trades when confirmation appears later than the crossover candle.
        """
        direction = direction.lower().strip()
        max_bars_since_cross = int(max_bars_since_cross)
        confirm_bars = int(confirm_bars)

        if max_bars_since_cross <= 0:
            raise ValueError("max_bars_since_cross must be >= 1.")

        if confirm_bars <= 0:
            raise ValueError("confirm_bars must be >= 1.")

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

        first_confirm_shift = confirm_shift - confirm_bars + 1
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
        left: Ref,
        right: Ref,
        max_bars_since_cross: int = 20,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        confirm_pattern: Optional[str] = None,
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
        left: Ref,
        right: Ref,
        max_bars_since_cross: int = 20,
        confirm_bars: int = 1,
        confirm_shift: int = 0,
        confirm_pattern: Optional[str] = None,
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

# ------------------------------------------------------------------
# Ordered reference / EMA ribbon helpers
# ------------------------------------------------------------------

    def refs_ordered(
        self,
        refs,
        direction: str = "desc",
        strict: bool = True,
        shift: int = 0,
    ) -> bool:
        """
        Checks whether references are ordered.
    
        direction="desc":
          refs[0] > refs[1] > refs[2]
    
        direction="asc":
          refs[0] < refs[1] < refs[2]
    
        Example bullish EMA ribbon:
          c.refs_ordered([EMA50, EMA100, EMA150], direction="desc")
    
        Example bearish EMA ribbon:
          c.refs_ordered([EMA50, EMA100, EMA150], direction="asc")
        """
        refs = list(refs)
    
        if len(refs) <= 1:
            return True
    
        direction = direction.lower().strip()
    
        if direction in ("desc", "down", "bullish", "above"):
            op = ">" if strict else ">="
        elif direction in ("asc", "up", "bearish", "below"):
            op = "<" if strict else "<="
        else:
            raise ValueError("direction must be 'desc'/'bullish' or 'asc'/'bearish'.")
    
        for left, right in zip(refs[:-1], refs[1:]):
            if not self.compare(left, op, right, shift=shift):
                return False
    
        return True
    
    
    def all_ref_groups_ordered(
        self,
        groups,
        direction: str = "desc",
        strict: bool = True,
        shift: int = 0,
    ) -> bool:
        """
        Checks multiple ordered reference groups.
    
        Example:
          c.all_ref_groups_ordered(
              [
                  [EMA50_1m, EMA100_1m, EMA150_1m],
                  [EMA50_5m, EMA100_5m, EMA150_5m],
              ],
              direction="desc"
          )
        """
        return all(
            self.refs_ordered(
                refs=group,
                direction=direction,
                strict=strict,
                shift=shift,
            )
            for group in groups
        ) 

# ------------------------------------------------------------------
# Cross-through helpers
# Example:
#   1m EMA50 crossed above at least 2 EMAs from a list in last 10 candles
# ------------------------------------------------------------------
    
    def crossed_through_refs(
        self,
        leader,
        refs,
        direction: str = "up",
        lookback: int = 10,
        include_current: bool = True,
        require_current_side: bool = True,
        unique: bool = True,
    ):
        """
        Returns the refs that `leader` crossed through within the recent lookback window.
    
        Parameters
        ----------
        leader:
          Column/value/callable that is doing the crossing.
          Example: "1m__ema__EMA_50"
    
        refs:
          List of columns/values/callables to check against.
          Example: [EMA100_1m, EMA150_1m, EMA100_5m]
    
        direction:
          "up"   -> leader crossed from below/equal to above ref
          "down" -> leader crossed from above/equal to below ref
    
        lookback:
          Number of candles to search.
    
          include_current=True, lookback=10 checks:
            candle 0, -1, -2, ..., -9
    
          include_current=False, lookback=10 checks:
            candle -1, -2, ..., -10
    
        require_current_side:
          If True:
            direction="up" requires leader currently above the crossed ref.
            direction="down" requires leader currently below the crossed ref.
    
          This prevents counting crosses that already reversed.
    
        unique:
          If True, each ref is counted once even if crossed multiple times.
          Usually this is what you want for strategy rules.
        """
        direction = direction.lower().strip()
        lookback = int(lookback)
    
        if lookback <= 0:
            return []
    
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'.")
    
        start_shift = 0 if include_current else -1
        end_shift = start_shift - lookback + 1
        shifts = range(start_shift, end_shift - 1, -1)
    
        crossed_refs = []
    
        for ref in refs:
            # Avoid self-comparison if the leader is accidentally included in refs.
            if isinstance(leader, str) and isinstance(ref, str) and leader == ref:
                continue
    
            matched = False
    
            for sh in shifts:
                if direction == "up":
                    crossed = self.crossed_up_ref(
                        leader,
                        ref,
                        cross_shift=sh,
                        prev_op="<=",
                        cur_op=">",
                    )
                    current_side_ok = self.gt(leader, ref, shift=0)
    
                else:
                    crossed = self.crossed_down_ref(
                        leader,
                        ref,
                        cross_shift=sh,
                        prev_op=">=",
                        cur_op="<",
                    )
                    current_side_ok = self.lt(leader, ref, shift=0)
    
                if crossed:
                    if require_current_side and not current_side_ok:
                        continue
    
                    crossed_refs.append(ref)
                    matched = True
    
                    if unique:
                        break
    
            # If unique=False, continue scanning and allow repeated events.
            if matched and unique:
                continue
    
        return crossed_refs
    
    
    def crossed_through_count(
        self,
        leader,
        refs,
        direction: str = "up",
        lookback: int = 10,
        include_current: bool = True,
        require_current_side: bool = True,
        unique: bool = True,
    ) -> int:
        """
        Count how many refs the leader crossed through.
        """
        return len(
            self.crossed_through_refs(
                leader=leader,
                refs=refs,
                direction=direction,
                lookback=lookback,
                include_current=include_current,
                require_current_side=require_current_side,
                unique=unique,
            )
        )
    
    
    def crossed_up_through_at_least(
        self,
        leader,
        refs,
        min_crosses: int,
        lookback: int,
        include_current: bool = True,
        require_current_side: bool = True,
        unique: bool = True,
    ) -> bool:
        """
        True if leader crossed upward through at least min_crosses refs.
        """
        return self.crossed_through_count(
            leader=leader,
            refs=refs,
            direction="up",
            lookback=lookback,
            include_current=include_current,
            require_current_side=require_current_side,
            unique=unique,
        ) >= int(min_crosses)
    
    
    def crossed_down_through_at_least(
        self,
        leader,
        refs,
        min_crosses: int,
        lookback: int,
        include_current: bool = True,
        require_current_side: bool = True,
        unique: bool = True,
    ) -> bool:
        """
        True if leader crossed downward through at least min_crosses refs.
        """
        return self.crossed_through_count(
            leader=leader,
            refs=refs,
            direction="down",
            lookback=lookback,
            include_current=include_current,
            require_current_side=require_current_side,
            unique=unique,
        ) >= int(min_crosses)