# indie:lang_version = 5
# BIAS -- one line per timeframe (1m 5m 15m 1h 4h 1D) in the chart's top-right corner: strongly
# bullish, bullish, neutral, bearish or strongly bearish, and why. For the Exness web terminal:
# Indicators -> Create -> paste -> Save -> Add to chart. Paste exness/probe_timeframes.py first: it
# says whether Exness lets an indicator read other timeframes (this one needs it).
#
# THE RULE (quant.indicators.add_bias, which tests/test_exness_bias.py holds this to on every
# timeframe, to the last digit), on each timeframe's own candles:
#   swings  +1 after a higher high AND a higher low, -1 after a lower high AND a lower low, else 0
#           (the swings of the Structure indicator: `Swing size` x ATR)
#   EMAs    +1 when close > EMA 50 > EMA 200, -1 when close < EMA 50 < EMA 200, else 0
#   bias = swings + EMAs: +2 strongly bullish ... -2 strongly bearish
#
# SAFE: each timeframe's line uses its last CLOSED candle only -- the 1h line changes when a 1h
# candle closes, never while one forms, whatever the terminal shows meanwhile.
#
# WHAT IT IS NOT: a signal. It says what the market looks like on each timeframe. Tested as a
# trade filter in the research, an EMA bias and HH/HL structure added nothing (docs/WHAT_WORKS.md).
#
# Timeframes BELOW the chart's (1m on a 5m chart) need Indie 5.14; `Also lower timeframes` asks for
# them, and an older Indie refuses with an error -- the probe says which you have.
from math import isnan, nan
from indie import indicator, param, param_ref, color, MainContext, SecContext, TimeFrame
from indie.color import rgba
from indie.drawings import LabelRel, RelativePosition
from indie.drawings import vertical_anchor as va, horizontal_anchor as ha


@param_ref('swing_size')
@param_ref('fast')
@param_ref('slow')
class Bias(SecContext):
    """The bias on this context's own candles, after each closed candle."""
    def __init__(self, swing_size, fast, slow):
        self._swing = swing_size
        self._fast = fast
        self._slow = slow
        self._atr_len = 14
        a0 = 1.0 / float(self._atr_len)               # pandas' ewm(alpha=1/14, adjust=False),
        com = (1.0 - a0) / a0                         # bit for bit
        self._alpha = 1.0 / (1.0 + com)
        self._owf = 1.0 - self._alpha
        # pandas' ewm(span=n, adjust=False, min_periods=n): alpha = 1 / (1 + (n - 1) / 2)
        self._fa = 1.0 / (1.0 + (float(fast) - 1.0) / 2.0)
        self._sa = 1.0 / (1.0 + (float(slow) - 1.0) / 2.0)
        self._atr_w = self.new_var(0.0)
        self._atr_n = self.new_var(0)
        self._prev_close = self.new_var(0.0)
        self._ef = self.new_var(0.0)
        self._es = self.new_var(0.0)
        self._en = self.new_var(0)
        self._dir = self.new_var(0)
        self._ext_px = self.new_var(nan)
        self._ext_i = self.new_var(-1)
        self._prev_hi = self.new_var(nan)
        self._prev_lo = self.new_var(nan)
        self._hi_state = self.new_var(0)              # 2 HH, 1 LH
        self._lo_state = self.new_var(0)              # -1 HL, -2 LL
        self._trend = self.new_var(0)
        self._out = self.new_var([nan, nan, nan, nan, nan, nan, nan, nan])

    def calc(self):
        if self.is_closed_bar:
            self._step()
        o = self._out.get()
        return o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]

    def _ema(self, w: float, x: float, a: float) -> float:
        if w != x:
            w = (1.0 - a) * w + a * x
            w = w / ((1.0 - a) + a)
        return w

    def _step(self) -> None:
        i = self.bar_index
        c = self.close[0]
        # EMAs
        n = self._en.get() + 1
        self._en.set(n)
        if n == 1:
            self._ef.set(c)
            self._es.set(c)
        else:
            self._ef.set(self._ema(self._ef.get(), c, self._fa))
            self._es.set(self._ema(self._es.get(), c, self._sa))
        ef = nan
        es = nan
        if n >= self._fast:
            ef = self._ef.get()
        if n >= self._slow:
            es = self._es.get()
        side = 0
        if c > ef and ef > es:
            side = 1
        elif c < ef and ef < es:
            side = -1
        # swings: the research zigzag, keeping only what the trend needs
        a = self._atr()
        t = a * self._swing
        if not isnan(t) and t > 0:
            h = self.high[0]
            lo = self.low[0]
            d = self._dir.get()
            px = self._ext_px.get()
            if d == 0:
                if self._ext_i.get() < 0:
                    self._ext_px.set(h)
                    self._ext_i.set(i)
                    self._dir.set(1)
            elif d == 1:
                if h >= px:
                    self._ext_px.set(h)
                    self._ext_i.set(i)
                elif px - lo >= t:
                    if not isnan(self._prev_hi.get()):
                        s = 1
                        if px > self._prev_hi.get():
                            s = 2
                        self._hi_state.set(s)
                    self._prev_hi.set(px)
                    self._dir.set(-1)
                    self._ext_px.set(lo)
                    self._ext_i.set(i)
            else:
                if lo <= px:
                    self._ext_px.set(lo)
                    self._ext_i.set(i)
                elif h - px >= t:
                    if not isnan(self._prev_lo.get()):
                        s = -2
                        if px > self._prev_lo.get():
                            s = -1
                        self._lo_state.set(s)
                    self._prev_lo.set(px)
                    self._dir.set(1)
                    self._ext_px.set(h)
                    self._ext_i.set(i)
            if d != 0:
                cur = 0
                if self._hi_state.get() == 2 and self._lo_state.get() == -1:
                    cur = 1
                elif self._hi_state.get() == 1 and self._lo_state.get() == -2:
                    cur = -1
                self._trend.set(cur)
        tr = self._trend.get()
        self._out.set([float(tr + side), float(tr), float(side), ef, es, c, float(i),
                       self.time[0]])

    def _atr(self) -> float:
        h = self.high[0]
        lo = self.low[0]
        tr = h - lo
        n = self._atr_n.get()
        if n > 0:
            pc = self._prev_close.get()
            tr = max(tr, max(abs(h - pc), abs(lo - pc)))
        self._prev_close.set(self.close[0])
        n += 1
        self._atr_n.set(n)
        w = tr
        if n > 1:
            w = self._atr_w.get()
            if w != tr:
                w = self._owf * w + self._alpha * tr
                w = w / (self._owf + self._alpha)
        self._atr_w.set(w)
        res = nan
        if n >= self._atr_len:
            res = w
        return res


@indicator('Bias (quant)', overlay_main_pane=True)
@param.float('swing_size', default=3.5, min=0.5, max=20.0, step=0.1, title='Swing size (ATR)')
@param.int('fast', default=50, min=2, max=500, title='Fast EMA')
@param.int('slow', default=200, min=3, max=1000, title='Slow EMA')
@param.bool('try_lower', default=False, title='Also lower timeframes than the chart (needs Indie 5.14)')
@param.bool('show_1d', default=True, title='Show 1D')
@param.float('top', default=0.05, min=0.0, max=0.9, step=0.01, title='Panel position from the top')
class Main(MainContext):
    def __init__(self, swing_size, fast, slow, try_lower, show_1d, top):
        self._chart_m = self.time_frame.to_minutes()
        self._names = ['1m', '5m', '15m', '1h', '4h', '1D']
        self._mins = [1, 5, 15, 60, 240, 1440]
        self._fast = fast
        self._slow = slow
        on: list[bool] = []
        for s in range(6):
            ok = self._mins[s] >= self._chart_m or try_lower
            if s == 5:
                ok = ok and show_1d
            on.append(ok)
        self._on = on
        (self._a0, self._a1, self._a2, self._a3, self._a4, self._a5, self._a6,
         self._a7) = self.calc_on(Bias, time_frame=self._tf(0, try_lower))
        (self._b0, self._b1, self._b2, self._b3, self._b4, self._b5, self._b6,
         self._b7) = self.calc_on(Bias, time_frame=self._tf(1, try_lower))
        (self._c0, self._c1, self._c2, self._c3, self._c4, self._c5, self._c6,
         self._c7) = self.calc_on(Bias, time_frame=self._tf(2, try_lower))
        (self._d0, self._d1, self._d2, self._d3, self._d4, self._d5, self._d6,
         self._d7) = self.calc_on(Bias, time_frame=self._tf(3, try_lower))
        (self._e0, self._e1, self._e2, self._e3, self._e4, self._e5, self._e6,
         self._e7) = self.calc_on(Bias, time_frame=self._tf(4, try_lower))
        (self._f0, self._f1, self._f2, self._f3, self._f4, self._f5, self._f6,
         self._f7) = self.calc_on(Bias, time_frame=self._tf(5, try_lower))
        # the last state of each timeframe whose candle had CLOSED by a chart candle's close
        blank: list[float] = []
        for k in range(6 * 8):
            blank.append(nan)
        self._last = self.new_var(blank)
        rows: list[LabelRel] = []
        for r in range(6):
            rows.append(LabelRel('', RelativePosition(va.TOP, ha.RIGHT, top + 0.045 * float(r), 0.99),
                                 bg_color=rgba(128, 128, 128, 0.3),
                                 text_color=rgba(255, 255, 255, 1.0), font_size=12))
        self._rows = self.new_var(rows)
        self._shown = self.new_var(['', '', '', '', '', ''])

    def _tf(self, s: int, try_lower: bool) -> TimeFrame:
        t = TimeFrame.from_str(self._names[s])
        if t.to_minutes() < self._chart_m and not try_lower:
            return self.time_frame                     # never asked; that row says so
        return t

    def _vals(self, s: int) -> list[float]:
        if s == 0:
            return [self._a0[0], self._a1[0], self._a2[0], self._a3[0], self._a4[0], self._a5[0],
                    self._a6[0], self._a7[0]]
        if s == 1:
            return [self._b0[0], self._b1[0], self._b2[0], self._b3[0], self._b4[0], self._b5[0],
                    self._b6[0], self._b7[0]]
        if s == 2:
            return [self._c0[0], self._c1[0], self._c2[0], self._c3[0], self._c4[0], self._c5[0],
                    self._c6[0], self._c7[0]]
        if s == 3:
            return [self._d0[0], self._d1[0], self._d2[0], self._d3[0], self._d4[0], self._d5[0],
                    self._d6[0], self._d7[0]]
        if s == 4:
            return [self._e0[0], self._e1[0], self._e2[0], self._e3[0], self._e4[0], self._e5[0],
                    self._e6[0], self._e7[0]]
        return [self._f0[0], self._f1[0], self._f2[0], self._f3[0], self._f4[0], self._f5[0],
                self._f6[0], self._f7[0]]

    def calc(self):
        close_t = self.time[0] + float(self._chart_m) * 60.0
        if self.is_closed_bar:
            for s in range(6):
                if self._on[s]:
                    v = self._vals(s)
                    m = float(self._mins[s]) * 60.0
                    if not isnan(v[7]) and v[7] + m <= close_t:
                        for q in range(8):
                            self._last.get()[s * 8 + q] = v[q]
        if self.is_last_bar:
            self._panel()

    def _panel(self) -> None:
        for s in range(6):
            text = self._names[s] + '  '
            bg = rgba(128, 128, 128, 0.3)
            if not self._on[s]:
                text = text + '-  below this chart (setting: also lower timeframes)'
                if s == 5:
                    text = self._names[s] + '  -  off'
            elif isnan(self._last.get()[s * 8]):
                text = text + '-  no closed candle yet'
            else:
                b = int(self._last.get()[s * 8])
                st = int(self._last.get()[s * 8 + 1])
                em = int(self._last.get()[s * 8 + 2])
                word = 'Neutral'
                if b == 2:
                    word = 'Strongly bullish'
                    bg = color.GREEN(0.65)
                if b == 1:
                    word = 'Bullish'
                    bg = color.GREEN(0.35)
                if b == -1:
                    word = 'Bearish'
                    bg = color.RED(0.35)
                if b == -2:
                    word = 'Strongly bearish'
                    bg = color.RED(0.65)
                why = 'swings mixed'
                if st == 1:
                    why = 'HH + HL'
                if st == -1:
                    why = 'LH + LL'
                ema = 'EMAs mixed'
                if isnan(self._last.get()[s * 8 + 4]):
                    ema = 'EMAs warming up'
                elif em == 1:
                    ema = 'above ' + str(self._fast) + ' > ' + str(self._slow)
                elif em == -1:
                    ema = 'below ' + str(self._fast) + ' < ' + str(self._slow)
                text = text + word + '  (' + why + ', ' + ema + ')'
            if text != self._shown.get()[s]:
                lab = self._rows.get()[s]
                lab.text = text
                lab.bg_color = bg
                self.chart.draw(lab)
                self._shown.get()[s] = text
