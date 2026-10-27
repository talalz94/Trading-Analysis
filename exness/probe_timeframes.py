# indie:lang_version = 5
# TIMEFRAME PROBE -- paste this into the Exness code editor BEFORE the multi-timeframe indicators
# (trendlines_mtf, structure, bias). It draws nothing on the price; it writes a short report in the
# top-left corner, and tells you whether Exness gives an indicator what those three need:
#
#   1. other timeframes at all: 15m, 1h and 4h candles read from inside one indicator (Indie's
#      `calc_on`), with the constructs the real indicators use -- a secondary context written as a
#      class with its own memory (Var) and the chart's settings (`param_ref`), returning 16 values;
#   2. how much history each timeframe gets (a 4h line in the email alerts is fitted on ~505 4h
#      candles, about 3 months);
#   3. whether a chart candle ever sees a higher-timeframe candle that has NOT closed yet -- on
#      history ("seen early", must be 0) and live ("live: forming"; the real indicators ignore a
#      forming candle either way, this only says what Exness does);
#   4. whether the higher timeframes' candles start on round UTC times (4h at 00, 04, 08 ...),
#      which places a higher-timeframe line between its closes;
#   5. text labels (LabelAbs, LabelRel) with the arguments the real indicators use.
#
# Put it on a 1m or 5m chart. The last line says ALL GOOD, or what failed. If the editor refuses to
# save or add it, the error message is the answer: copy it back.
# `Also try 1m` asks for a timeframe FINER than the chart (needs Indie 5.14); on a 5m chart it
# fails loudly on an older Indie -- turn it on only to find that out.
from math import isnan, nan
from indie import indicator, param, param_ref, color, MainContext, SecContext, TimeFrame
from indie.color import rgba
from indie.drawings import LabelAbs, LabelRel, AbsolutePosition, RelativePosition
from indie.drawings import vertical_anchor as va, horizontal_anchor as ha


@param_ref('check_value')
class Candles(SecContext):
    """Runs on one timeframe's own candles and reports what it saw."""
    def __init__(self, check_value):
        self._n = self.new_var(0)                    # closed candles seen
        self._first = self.new_var(nan)              # open time of the first
        self._last = self.new_var(nan)               # open time of the latest closed one
        self._aligned = self.new_var(0)              # how many opened on a round UTC time
        self._echo = check_value

    def calc(self):
        if self.is_closed_bar:
            self._n.set(self._n.get() + 1)
            if isnan(self._first.get()):
                self._first.set(self.time[0])
            self._last.set(self.time[0])
            step = float(self.time_frame.to_minutes()) * 60.0
            if self.time[0] - step * float(int(self.time[0] / step)) == 0.0:
                self._aligned.set(self._aligned.get() + 1)
        return (float(self._n.get()), self._first.get(), self._last.get(),
                float(self._aligned.get()), float(self.time_frame.to_minutes()), self._echo,
                self.time[0], self.close[0], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)


@indicator('Timeframe probe', overlay_main_pane=True)
@param.float('check_value', default=7.0, min=0.0, max=100.0, step=1.0, title='Check value (leave at 7)')
@param.bool('try_lower', default=False, title='Also try 1m below the chart (needs Indie 5.14)')
class Main(MainContext):
    def __init__(self, check_value, try_lower):
        self._check = check_value
        self._chart_m = self.time_frame.to_minutes()
        self._names = ['15m', '1h', '4h', '1m']
        self._want = [15, 60, 240, 1]
        self._on = [self._chart_m < 15, self._chart_m < 60, self._chart_m < 240,
                    try_lower and self._chart_m > 1]
        tf_low = self.time_frame
        if try_lower and self._chart_m > 1:
            tf_low = TimeFrame.from_str('1m')
        (self._a0, self._a1, self._a2, self._a3, self._a4, self._a5, self._a6, self._a7,
         self._a8, self._a9, self._a10, self._a11, self._a12, self._a13, self._a14,
         self._a15) = self.calc_on(Candles, time_frame=self._tf('15m'))
        (self._b0, self._b1, self._b2, self._b3, self._b4, self._b5, self._b6, self._b7,
         self._b8, self._b9, self._b10, self._b11, self._b12, self._b13, self._b14,
         self._b15) = self.calc_on(Candles, time_frame=self._tf('1h'))
        (self._c0, self._c1, self._c2, self._c3, self._c4, self._c5, self._c6, self._c7,
         self._c8, self._c9, self._c10, self._c11, self._c12, self._c13, self._c14,
         self._c15) = self.calc_on(Candles, time_frame=self._tf('4h'))
        (self._d0, self._d1, self._d2, self._d3, self._d4, self._d5, self._d6, self._d7,
         self._d8, self._d9, self._d10, self._d11, self._d12, self._d13, self._d14,
         self._d15) = self.calc_on(Candles, time_frame=tf_low)
        self._n = self.new_var(0)                    # chart candles
        self._first = self.new_var(nan)
        self._early = self.new_var([0, 0, 0, 0])     # chart candles that saw an unclosed one
        self._forming = self.new_var([0, 0, 0, 0])   # live candles shown a forming one
        self._live = self.new_var(0)                 # live chart candles closed so far
        self._bad = self.new_var([0, 0, 0, 0])       # values that came back wrong
        blank: list[LabelRel] = []
        for r in range(6):
            blank.append(LabelRel('', RelativePosition(va.TOP, ha.LEFT, 0.04 + 0.045 * float(r), 0.01),
                                  bg_color=rgba(0, 0, 0, 0.6), text_color=rgba(255, 255, 255, 1.0), font_size=12))
        self._rows = self.new_var(blank)
        self._shown = self.new_var(['', '', '', '', '', ''])
        self._mark = self.new_var(LabelAbs('probe', AbsolutePosition(0.0, 0.0),
                                           bg_color=color.BLUE(0.3), text_color=rgba(255, 255, 255, 1.0),
                                           font_size=10))
        self._marked = self.new_var(False)

    def _tf(self, s: str) -> TimeFrame:
        t = TimeFrame.from_str(s)
        if t.to_minutes() < self._chart_m:
            return self.time_frame                     # never ask below the chart here
        return t

    def _vals(self, s: int) -> list[float]:
        if s == 0:
            return [self._a0[0], self._a1[0], self._a2[0], self._a3[0], self._a4[0], self._a5[0],
                    self._a6[0], self._a7[0], self._a8[0], self._a9[0], self._a10[0], self._a11[0],
                    self._a12[0], self._a13[0], self._a14[0], self._a15[0]]
        if s == 1:
            return [self._b0[0], self._b1[0], self._b2[0], self._b3[0], self._b4[0], self._b5[0],
                    self._b6[0], self._b7[0], self._b8[0], self._b9[0], self._b10[0], self._b11[0],
                    self._b12[0], self._b13[0], self._b14[0], self._b15[0]]
        if s == 2:
            return [self._c0[0], self._c1[0], self._c2[0], self._c3[0], self._c4[0], self._c5[0],
                    self._c6[0], self._c7[0], self._c8[0], self._c9[0], self._c10[0], self._c11[0],
                    self._c12[0], self._c13[0], self._c14[0], self._c15[0]]
        return [self._d0[0], self._d1[0], self._d2[0], self._d3[0], self._d4[0], self._d5[0],
                self._d6[0], self._d7[0], self._d8[0], self._d9[0], self._d10[0], self._d11[0],
                self._d12[0], self._d13[0], self._d14[0], self._d15[0]]

    def calc(self):
        close_t = self.time[0] + float(self._chart_m) * 60.0
        if self.is_closed_bar:
            self._n.set(self._n.get() + 1)
            if isnan(self._first.get()):
                self._first.set(self.time[0])
            if self.is_last_bar:
                self._live.set(self._live.get() + 1)
            for s in range(4):
                if self._on[s]:
                    v = self._vals(s)
                    m = float(self._want[s]) * 60.0
                    if not isnan(v[2]) and v[2] + m > close_t:
                        self._early.get()[s] = self._early.get()[s] + 1
                    if self.is_last_bar and not isnan(v[6]) and v[6] + m > close_t:
                        self._forming.get()[s] = self._forming.get()[s] + 1
                    ok = v[4] == float(self._want[s]) and v[5] == self._check
                    for q in range(8):
                        ok = ok and v[8 + q] == float(q + 1)
                    if not isnan(v[0]) and not ok:
                        self._bad.get()[s] = self._bad.get()[s] + 1
        if self.is_last_bar:
            self._report()

    def _report(self) -> None:
        days = (self.time[0] - self._first.get()) / 86400.0
        text: list[str] = []
        text.append('Timeframe probe: chart ' + self._name(self._chart_m) + ', '
                    + str(self._n.get()) + ' candles over ' + str(round(days, 1)) + ' days')
        good = True
        for s in range(4):
            name = self._names[s]
            if not self._on[s]:
                if s == 3:
                    text.append('1m below the chart: not asked (setting off, or a 1m chart)')
                else:
                    text.append(name + ': not above this chart, not asked')
                continue
            v = self._vals(s)
            if isnan(v[0]) or v[0] < 1.0:
                text.append(name + ': NO CANDLES')
                good = False
                continue
            span = (v[2] - v[1]) / 86400.0
            share = round(100.0 * v[3] / v[0])
            text.append(name + ': ' + str(int(v[0])) + ' candles, ' + str(round(span, 1))
                        + ' days; ' + str(int(share)) + '% start on round UTC times; seen early '
                        + str(self._early.get()[s]) + '; live: forming ' + str(self._forming.get()[s])
                        + ' of ' + str(self._live.get()) + '; wrong values ' + str(self._bad.get()[s]))
            if self._early.get()[s] > 0 or self._bad.get()[s] > 0:
                good = False
        if good:
            text.append('ALL GOOD: the multi-timeframe indicators can run here')
        else:
            text.append('PROBLEM: copy this panel back')
        for r in range(6):
            t = ''
            if r < len(text):
                t = text[r]
            if t != self._shown.get()[r]:
                lab = self._rows.get()[r]
                lab.text = t
                self.chart.draw(lab)
                self._shown.get()[r] = t
        if not self._marked.get():
            mk = self._mark.get()
            mk.position = AbsolutePosition(self.time[0], self.close[0])
            self.chart.draw(mk)
            self._marked.set(True)

    def _name(self, m: int) -> str:
        if m >= 1440 and m % 1440 == 0:
            return str(m // 1440) + 'D'
        if m >= 60 and m % 60 == 0:
            return str(m // 60) + 'h'
        return str(m) + 'm'
