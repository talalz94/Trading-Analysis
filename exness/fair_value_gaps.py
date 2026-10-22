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
# NO REPAINTING. A gap is created, and judged filled, only on a CLOSED candle: every historical
# candle, and the final update of the live one. While the live candle forms, the open boxes are
# only stretched to it -- nothing appears or disappears until it closes.
#
# Defaults = notebooks/nb23_config.py: FVG_MITIGATION "ce", FVG_MIN_SIZE_ATR 0.5,
# FVG_MIN_DISP_ATR 0, FVG_MAX_AGE_BARS 1000, FVG_ATR_PERIOD 14, FVG_EXTEND (open boxes run to now).
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
@param.int('max_live', default=200, min=1, max=1000, title='Max open gaps drawn')
@param.int('max_dead', default=300, min=1, max=2000, title='Max filled gaps kept')
@param.color('bull_color', default=color.GREEN, title='Bullish colour')
@param.color('bear_color', default=color.RED, title='Bearish colour')
class Main(MainContext):
    def __init__(self, mitigation, min_size_atr, min_disp_atr, max_age, atr_len, show_bull,
                 show_bear, keep_dead, max_live, max_dead, bull_color, bear_color):
        self._mode = mitigation
        self._min_size = min_size_atr
        self._min_disp = min_disp_atr
        self._max_age = max_age
        self._atr_len = atr_len
        self._show_bull = show_bull
        self._show_bear = show_bear
        self._keep_dead = keep_dead
        self._max_live = max_live
        self._max_dead = max_dead
        self._bull = bull_color
        self._bear = bear_color
        # to_minutes, not to_seconds: Exness's Indie (Oct 2026) has only __eq__ __le__ __lt__
        # to_minutes on TimeFrame -- to_seconds came in a later release and failed to compile
        self._bar_secs = float(self.time_frame.to_minutes()) * 60.0
        # the OPEN gaps, one entry per list each, oldest first
        self._kind: list[int] = []           # +1 bullish, -1 bearish
        self._top: list[float] = []          # always the HIGHER edge
        self._bot: list[float] = []
        self._born: list[int] = []           # bar_index of the 3rd candle
        self._box: list[Rectangle] = []
        self._dead: list[Rectangle] = []     # filled / expired boxes still shown (keep_dead)

    def calc(self):
        atr = Atr.new(self._atr_len)          # every candle, so the ATR stays continuous
        close_t = self.time[0] + self._bar_secs
        if self.is_closed_bar:
            self._retire(close_t)            # judge the open gaps on THIS candle first...
            self._detect(atr[0], close_t)    # ...then a gap this candle completes (not judged
                                             # on its own candle, as in the research code)
        if self.is_last_bar:                 # the open gaps run to the live candle
            for i in range(len(self._box)):
                self._box[i].point_b.time = close_t
                self.chart.draw(self._box[i])

    def _level(self, i: int) -> float:
        """The price that kills gap i: near edge (touch), midpoint (ce) or far edge (full)."""
        mid = 0.5 * (self._top[i] + self._bot[i])
        level = mid
        if self._mode == 'touch':
            level = self._top[i] if self._kind[i] > 0 else self._bot[i]
        elif self._mode == 'full':
            level = self._bot[i] if self._kind[i] > 0 else self._top[i]
        return level

    def _retire(self, close_t: float) -> None:
        i = len(self._kind) - 1
        while i >= 0:
            expired = self._max_age > 0 and self.bar_index - self._born[i] > self._max_age
            filled = False
            if not expired:
                if self._kind[i] > 0:
                    filled = self.low[0] <= self._level(i)     # a bullish gap is eaten from above
                else:
                    filled = self.high[0] >= self._level(i)    # a bearish gap from below
            if expired or filled:
                self._end(i, close_t)
            i -= 1

    def _end(self, i: int, close_t: float) -> None:
        box = self._box[i]
        if self._keep_dead:
            c = self._bull if self._kind[i] > 0 else self._bear
            box.point_b.time = close_t       # the box stops on the candle that killed it
            box.bg_color = c(0.06)
            box.line_color = c(0.25)
            self.chart.draw(box)
            self._dead.append(box)
            if len(self._dead) > self._max_dead:
                self.chart.erase(self._dead.pop(0))
        else:
            self.chart.erase(box)
        self._kind.pop(i)
        self._top.pop(i)
        self._bot.pop(i)
        self._born.pop(i)
        self._box.pop(i)

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
        box = Rectangle(AbsolutePosition(close_t, top), AbsolutePosition(close_t, bot),
                        line_style=line_style.SOLID, line_width=1, line_color=c(0.5),
                        bg_color=c(0.15), middle_line_style=line_style.DOTTED,
                        middle_line_width=1, middle_line_color=mid_line)
        self.chart.draw(box)
        self._kind.append(kind)
        self._top.append(top)
        self._bot.append(bot)
        self._born.append(self.bar_index)
        self._box.append(box)
        if len(self._box) > self._max_live:  # too many open: the OLDEST goes
            self.chart.erase(self._box[0])
            self._kind.pop(0)
            self._top.pop(0)
            self._bot.pop(0)
            self._born.pop(0)
            self._box.pop(0)
