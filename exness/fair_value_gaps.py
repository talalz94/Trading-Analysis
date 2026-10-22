# indie:lang_version = 5
# FAIR VALUE GAPS -- the quant project's FVG (quant/indicators/fvg.py), ported to Indie for the
# Exness web terminal: Indicators -> Create (the code editor) -> paste this file -> Save -> Add to
# chart. It shows the gaps of the chart's own timeframe.
#
# THE RULES ARE THE RESEARCH CODE'S, value for value (tests/test_exness_fvg.py runs this file
# against quant.indicators.add_fvgs on real candles and requires the same gaps):
#   bullish  the 3rd candle's LOW is above the 1st candle's HIGH -> band (high[1st], low[3rd])
#   bearish  the 3rd candle's HIGH is below the 1st candle's LOW -> band (high[3rd], low[1st])
#   kept     if the band is at least `Min size` ATR tall (Wilder ATR) and, if set, the MIDDLE
#            candle's body is at least `Min displacement` ATR
#   born     at the CLOSE of the 3rd candle -- the box starts there, never at the middle candle,
#            where it only looks like it sits (drawing it there would be a two-candle lookahead)
#   dies     when a later candle trades back to the gap's near edge (touch), its midpoint (ce,
#            the default) or its far edge (full); or, never filled, after `Max age` candles
#
# NO REPAINTING. Gaps are created, judged and drawn only on a CLOSED candle: every historical
# candle, and the final update of the live one. Nothing changes while the live candle forms.
#
# WHAT EXNESS'S RUNTIME REQUIRES, each learnt from an error it raised:
#   - ALL STATE IN Var. Exness rolls the chart back whenever a candle is updated -- even a
#     candle that has already closed, when its value is corrected -- and re-runs the script.
#     A Var rolls back with it; a plain list does not, and then holds boxes the chart has
#     already removed ("cannot erase: this drawing was removed because the candle value
#     changed"). So every list below is a Var, and every draw / erase acts on a box read from one.
#   - AT MOST 100 DRAWING CHANGES PER UPDATE ("exceeded the limit 100"). Open boxes reach
#     `Reach ahead` candles past their last update and are redrawn only when the live candle
#     nears their edge; and each update makes at most 90 changes. Beyond that -- one huge candle
#     filling dozens of gaps -- the rest are drawn on the next candles. Only the DRAWING waits:
#     which gaps exist and which are filled is decided on the candle itself.
#   - TimeFrame has to_minutes, not to_seconds (a compile error).
#
# Defaults = notebooks/nb23_config.py: FVG_MITIGATION "ce", FVG_MIN_SIZE_ATR 0.5,
# FVG_MIN_DISP_ATR 0, FVG_MAX_AGE_BARS 1000, FVG_ATR_PERIOD 14.
from math import isnan
from indie import indicator, param, color, line_style, MainContext
from indie.algorithms import Atr
from indie.drawings import Rectangle, AbsolutePosition


@indicator('FVG (quant)', overlay_main_pane=True)
@param.str('mitigation', default='ce', options=['touch', 'ce', 'full'],
           title='Gap dies at: touch / ce (midpoint) / full')
@param.float('min_size_atr', default=0.5, min=0.0, max=20.0, step=0.05, title='Min size (ATR)')
@param.float('min_disp_atr', default=0.0, min=0.0, max=20.0, step=0.1,
             title='Min displacement: middle body (ATR), 0 = off')
@param.int('max_age', default=1000, min=0, max=100000, title='Max age (candles), 0 = no limit')
@param.int('atr_len', default=14, min=1, max=500, title='ATR length')
@param.bool('show_bull', default=True, title='Bullish gaps')
@param.bool('show_bear', default=True, title='Bearish gaps')
@param.bool('keep_dead', default=False, title='Keep filled / expired gaps on the chart')
@param.int('reach', default=20, min=3, max=500, title='Open boxes reach ahead (candles)')
@param.int('max_live', default=200, min=1, max=1000, title='Max open gaps drawn')
@param.int('max_dead', default=300, min=1, max=2000, title='Max filled gaps kept')
@param.color('bull_color', default=color.GREEN, title='Bullish colour')
@param.color('bear_color', default=color.RED, title='Bearish colour')
class Main(MainContext):
    def __init__(self, mitigation, min_size_atr, min_disp_atr, max_age, atr_len, show_bull,
                 show_bear, keep_dead, reach, max_live, max_dead, bull_color, bear_color):
        self._mode = mitigation
        self._min_size = min_size_atr
        self._min_disp = min_disp_atr
        self._max_age = max_age
        self._atr_len = atr_len
        self._show_bull = show_bull
        self._show_bear = show_bear
        self._keep_dead = keep_dead
        self._reach = float(reach)
        self._max_live = max_live
        self._max_dead = max_dead
        self._bull = bull_color
        self._bear = bear_color
        self._bar_secs = float(self.time_frame.to_minutes()) * 60.0
        # THE OPEN GAPS, one entry per list, oldest first -- all in Var (see the top of the file)
        kinds: list[int] = []
        tops: list[float] = []
        bots: list[float] = []
        borns: list[int] = []
        boxes: list[Rectangle] = []
        drawn: list[bool] = []
        dirty: list[bool] = []
        self._kind = self.new_var(kinds)     # +1 bullish, -1 bearish
        self._top = self.new_var(tops)       # always the HIGHER edge
        self._bot = self.new_var(bots)
        self._born = self.new_var(borns)     # bar_index of the 3rd candle
        self._box = self.new_var(boxes)
        self._drawn = self.new_var(drawn)    # is on the chart
        self._dirty = self.new_var(dirty)    # must be (re)drawn
        # filled / expired boxes still shown (keep_dead)
        d_boxes: list[Rectangle] = []
        d_drawn: list[bool] = []
        d_dirty: list[bool] = []
        self._dbox = self.new_var(d_boxes)
        self._ddrawn = self.new_var(d_drawn)
        self._ddirty = self.new_var(d_dirty)
        # boxes on the chart that must come off it
        trash: list[Rectangle] = []
        self._trash = self.new_var(trash)

    def calc(self):
        atr = Atr.new(self._atr_len)          # every update, so the ATR stays continuous
        if not self.is_closed_bar:
            return                           # the live candle: nothing changes until it closes
        close_t = self.time[0] + self._bar_secs
        self._retire(close_t)                # judge the open gaps on THIS candle first...
        self._detect(atr[0], close_t)        # ...then a gap this candle completes (not judged
                                             # on its own candle, as in the research code)
        self._renew(close_t)
        self._sync()

    def _sync(self) -> None:
        """Bring the chart in line with the state, at most 90 changes (Exness allows 100):
        erasures first, then redraws. What does not fit waits for the next candle."""
        left = 90
        while left > 0 and len(self._trash.get()) > 0:
            self.chart.erase(self._trash.get()[0])
            self._trash.get().pop(0)
            left -= 1
        for j in range(len(self._dbox.get())):
            if left <= 0:
                return
            if self._ddirty.get()[j]:
                self.chart.draw(self._dbox.get()[j])
                self._ddirty.get()[j] = False
                self._ddrawn.get()[j] = True
                left -= 1
        for i in range(len(self._box.get())):
            if left <= 0:
                return
            if self._dirty.get()[i]:
                self.chart.draw(self._box.get()[i])
                self._dirty.get()[i] = False
                self._drawn.get()[i] = True
                left -= 1

    def _renew(self, close_t: float) -> None:
        """Push forward the boxes the live candle is about to reach -- each about once every
        `reach` candles, so the redrawing spreads over candles instead of landing on one."""
        due = close_t + 2.0 * self._bar_secs
        for i in range(len(self._box.get())):
            if self._box.get()[i].point_b.time < due:
                self._box.get()[i].point_b.time = close_t + self._reach * self._bar_secs
                self._dirty.get()[i] = True

    def _level(self, i: int) -> float:
        """The price that kills gap i: near edge (touch), midpoint (ce) or far edge (full)."""
        top = self._top.get()[i]
        bot = self._bot.get()[i]
        level = 0.5 * (top + bot)
        if self._mode == 'touch':
            level = top if self._kind.get()[i] > 0 else bot
        elif self._mode == 'full':
            level = bot if self._kind.get()[i] > 0 else top
        return level

    def _retire(self, close_t: float) -> None:
        i = len(self._kind.get()) - 1
        while i >= 0:
            expired = self._max_age > 0 and self.bar_index - self._born.get()[i] > self._max_age
            filled = False
            if not expired:
                if self._kind.get()[i] > 0:
                    filled = self.low[0] <= self._level(i)     # a bullish gap is eaten from above
                else:
                    filled = self.high[0] >= self._level(i)    # a bearish gap from below
            if expired or filled:
                self._end(i, close_t)
            i -= 1

    def _end(self, i: int, close_t: float) -> None:
        if self._keep_dead:
            c = self._bull if self._kind.get()[i] > 0 else self._bear
            self._box.get()[i].point_b.time = close_t     # it stops on the candle that killed it
            self._box.get()[i].bg_color = c(0.06)
            self._box.get()[i].line_color = c(0.25)
            self._dbox.get().append(self._box.get()[i])
            self._ddrawn.get().append(self._drawn.get()[i])
            self._ddirty.get().append(True)
            if len(self._dbox.get()) > self._max_dead:   # too many kept: the OLDEST goes
                if self._ddrawn.get()[0]:
                    self._trash.get().append(self._dbox.get()[0])
                self._dbox.get().pop(0)
                self._ddrawn.get().pop(0)
                self._ddirty.get().pop(0)
        elif self._drawn.get()[i]:
            self._trash.get().append(self._box.get()[i])
        self._drop(i)

    def _drop(self, i: int) -> None:
        self._kind.get().pop(i)
        self._top.get().pop(i)
        self._bot.get().pop(i)
        self._born.get().pop(i)
        self._box.get().pop(i)
        self._drawn.get().pop(i)
        self._dirty.get().pop(i)

    def _detect(self, atr: float, close_t: float) -> None:
        if self.bar_index < 2 or isnan(atr) or atr <= 0:
            return
        kind = 0
        top = 0.0
        bot = 0.0
        if self.low[0] > self.high[2]:
            kind = 1
            top = self.low[0]
            bot = self.high[2]
        elif self.high[0] < self.low[2]:
            kind = -1
            top = self.low[2]
            bot = self.high[0]
        if kind == 0:
            return
        if (kind > 0 and not self._show_bull) or (kind < 0 and not self._show_bear):
            return
        if self._min_size > 0 and top - bot < self._min_size * atr:
            return
        if self._min_disp > 0 and abs(self.close[1] - self.open[1]) < self._min_disp * atr:
            return
        c = self._bull if kind > 0 else self._bear
        mid_line = c(0.6) if self._mode == 'ce' else color.TRANSPARENT
        self._box.get().append(Rectangle(
            AbsolutePosition(close_t, top),
            AbsolutePosition(close_t + self._reach * self._bar_secs, bot),
            line_style=line_style.SOLID, line_width=1, line_color=c(0.5), bg_color=c(0.15),
            middle_line_style=line_style.DOTTED, middle_line_width=1,
            middle_line_color=mid_line))
        self._kind.get().append(kind)
        self._top.get().append(top)
        self._bot.get().append(bot)
        self._born.get().append(self.bar_index)
        self._drawn.get().append(False)
        self._dirty.get().append(True)
        if len(self._box.get()) > self._max_live:     # too many open: the OLDEST goes
            if self._drawn.get()[0]:
                self._trash.get().append(self._box.get()[0])
            self._drop(0)
