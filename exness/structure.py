# indie:lang_version = 5
# STRUCTURE -- HH / HL / LH / LL swing labels for the chart's own timeframe AND 15m, 1h and 4h, on
# one chart, for the Exness web terminal: Indicators -> Create -> paste -> Save -> Add to chart.
# Paste exness/probe_timeframes.py first: it says whether Exness lets an indicator read other
# timeframes (this one needs it).
#
# THE SWINGS ARE THE RESEARCH CODE'S (quant.indicators.add_zigzag_structure, the same swings the
# trendlines are fitted through): a swing HIGH is confirmed once price falls `Swing size` x ATR
# below the highest high since the last swing low (and the mirror for lows), ATR = Wilder's 14,
# computed here as pandas does. A swing high above the previous one is HH, otherwise LH; a swing
# low above the previous one is HL, otherwise LL. tests/test_exness_structure.py holds every label
# to the research code on every timeframe.
#
# SAFE: a label appears on the candle that CONFIRMS the swing -- for a 1h swing, at the close of
# the 1h candle that confirms it, never earlier -- and never moves or disappears afterwards. A
# higher timeframe's candle is used only once it has closed, whatever the terminal shows while it
# forms. The label is drawn at the swing's own candle (for a 1h swing, at the chart candle that
# made that 1h candle's high or low), so it sits where the swing was, but it APPEARS later, when
# the swing is confirmed -- exactly as in the email alerts.
#
# DISTINGUISHABLE: green = HH / HL (rising), red = LH / LL (falling); a higher timeframe's label
# says its timeframe ('1h HH'), is larger, and sits further from the candle, so a 4h HH and a 15m
# HH at the same high stack instead of overlapping.
from math import isnan, nan
from indie import indicator, param, param_ref, color, MainContext, SecContext, TimeFrame
from indie.drawings import LabelAbs, AbsolutePosition


@param_ref('swing_size')
class Swings(SecContext):
    """The research zigzag on this context's own candles. Returns the swings found so far."""
    def __init__(self, swing_size):
        self._swing = swing_size
        self._atr_len = 14
        a0 = 1.0 / float(self._atr_len)               # pandas' ewm(alpha=1/14, adjust=False),
        com = (1.0 - a0) / a0                         # bit for bit
        self._alpha = 1.0 / (1.0 + com)
        self._owf = 1.0 - self._alpha
        self._atr_w = self.new_var(0.0)
        self._atr_n = self.new_var(0)
        self._prev_close = self.new_var(0.0)
        self._dir = self.new_var(0)                   # +1 seeking a high, -1 a low, 0 not started
        self._ext_px = self.new_var(nan)
        self._ext_i = self.new_var(-1)
        self._ext_t = self.new_var(nan)
        self._prev_hi = self.new_var(nan)
        self._prev_lo = self.new_var(nan)
        self._hi_state = self.new_var(0)              # 2 HH, 1 LH
        self._lo_state = self.new_var(0)              # -1 HL, -2 LL
        self._count = self.new_var(0)                 # swings labelled so far
        self._label = self.new_var(0)                 # the latest: 2 HH 1 LH -1 HL -2 LL
        self._px = self.new_var(nan)                  # its price
        self._pt = self.new_var(nan)                  # its candle's open time
        self._trend = self.new_var(0)                 # +1 HH+HL, -1 LH+LL, 0 mixed
        self._j = self.new_var(-1)                    # the latest closed candle, and its time
        self._tj = self.new_var(nan)
        # the state before the latest candle, for when it is computed again
        self._k_atr_w = self.new_var(self._atr_w.get())
        self._k_atr_n = self.new_var(self._atr_n.get())
        self._k_prev_close = self.new_var(self._prev_close.get())
        self._k_dir = self.new_var(self._dir.get())
        self._k_ext_px = self.new_var(self._ext_px.get())
        self._k_ext_i = self.new_var(self._ext_i.get())
        self._k_ext_t = self.new_var(self._ext_t.get())
        self._k_prev_hi = self.new_var(self._prev_hi.get())
        self._k_prev_lo = self.new_var(self._prev_lo.get())
        self._k_hi_state = self.new_var(self._hi_state.get())
        self._k_lo_state = self.new_var(self._lo_state.get())
        self._k_count = self.new_var(self._count.get())
        self._k_label = self.new_var(self._label.get())
        self._k_px = self.new_var(self._px.get())
        self._k_pt = self.new_var(self._pt.get())
        self._k_trend = self.new_var(self._trend.get())
        self._k_j = self.new_var(self._j.get())
        self._k_tj = self.new_var(self._tj.get())
        self._done_t = self.new_var(nan)        # the latest candle computed

    def calc(self):
        if self.is_closed_bar:
            self._once()
            self._step()
        return (float(self._count.get()), float(self._label.get()), self._px.get(),
                self._pt.get(), float(self._trend.get()), float(self._j.get()), self._tj.get())

    def _save(self) -> None:
        self._k_atr_w.set(self._atr_w.get())
        self._k_atr_n.set(self._atr_n.get())
        self._k_prev_close.set(self._prev_close.get())
        self._k_dir.set(self._dir.get())
        self._k_ext_px.set(self._ext_px.get())
        self._k_ext_i.set(self._ext_i.get())
        self._k_ext_t.set(self._ext_t.get())
        self._k_prev_hi.set(self._prev_hi.get())
        self._k_prev_lo.set(self._prev_lo.get())
        self._k_hi_state.set(self._hi_state.get())
        self._k_lo_state.set(self._lo_state.get())
        self._k_count.set(self._count.get())
        self._k_label.set(self._label.get())
        self._k_px.set(self._px.get())
        self._k_pt.set(self._pt.get())
        self._k_trend.set(self._trend.get())
        self._k_j.set(self._j.get())
        self._k_tj.set(self._tj.get())

    def _load(self) -> None:
        self._atr_w.set(self._k_atr_w.get())
        self._atr_n.set(self._k_atr_n.get())
        self._prev_close.set(self._k_prev_close.get())
        self._dir.set(self._k_dir.get())
        self._ext_px.set(self._k_ext_px.get())
        self._ext_i.set(self._k_ext_i.get())
        self._ext_t.set(self._k_ext_t.get())
        self._prev_hi.set(self._k_prev_hi.get())
        self._prev_lo.set(self._k_prev_lo.get())
        self._hi_state.set(self._k_hi_state.get())
        self._lo_state.set(self._k_lo_state.get())
        self._count.set(self._k_count.get())
        self._label.set(self._k_label.get())
        self._px.set(self._k_px.get())
        self._pt.set(self._k_pt.get())
        self._trend.set(self._k_trend.get())
        self._j.set(self._k_j.get())
        self._tj.set(self._k_tj.get())

    def _once(self) -> None:
        """Exness computes a timeframe's still-forming candle as if closed when a chart
        loads. If that candle is computed again when it really closes, start it from the
        state before its first time, so it counts once, with its final values."""
        if self.time[0] == self._done_t.get():
            self._load()
        else:
            self._save()
        self._done_t.set(self.time[0])

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

    def _step(self) -> None:
        i = self.bar_index
        a = self._atr()
        self._j.set(i)
        self._tj.set(self.time[0])
        t = a * self._swing
        if isnan(t) or t <= 0:
            return
        h = self.high[0]
        lo = self.low[0]
        d = self._dir.get()
        if d == 0:
            if self._ext_i.get() < 0:
                self._ext_px.set(h)
                self._ext_i.set(i)
                self._ext_t.set(self.time[0])
                self._dir.set(1)
            return
        px = self._ext_px.get()
        if d == 1:
            if h >= px:
                self._ext_px.set(h)
                self._ext_i.set(i)
                self._ext_t.set(self.time[0])
            elif px - lo >= t:
                if not isnan(self._prev_hi.get()):
                    s = 1
                    if px > self._prev_hi.get():
                        s = 2
                    self._hi_state.set(s)
                    self._emit(s, px)
                self._prev_hi.set(px)
                self._dir.set(-1)
                self._ext_px.set(lo)
                self._ext_i.set(i)
                self._ext_t.set(self.time[0])
        else:
            if lo <= px:
                self._ext_px.set(lo)
                self._ext_i.set(i)
                self._ext_t.set(self.time[0])
            elif h - px >= t:
                if not isnan(self._prev_lo.get()):
                    s = -2
                    if px > self._prev_lo.get():
                        s = -1
                    self._lo_state.set(s)
                    self._emit(s, px)
                self._prev_lo.set(px)
                self._dir.set(1)
                self._ext_px.set(h)
                self._ext_i.set(i)
                self._ext_t.set(self.time[0])
        cur = 0
        if self._hi_state.get() == 2 and self._lo_state.get() == -1:
            cur = 1
        elif self._hi_state.get() == 1 and self._lo_state.get() == -2:
            cur = -1
        self._trend.set(cur)

    def _emit(self, s: int, px: float) -> None:
        self._count.set(self._count.get() + 1)
        self._label.set(s)
        self._px.set(px)
        self._pt.set(self._ext_t.get())


@indicator('Structure (quant)', overlay_main_pane=True)
@param.float('swing_size', default=3.5, min=0.5, max=20.0, step=0.1, title='Swing size (ATR)')
@param.bool('show_chart', default=True, title="This chart's timeframe")
@param.bool('show_15m', default=True, title='15m')
@param.bool('show_1h', default=True, title='1h')
@param.bool('show_4h', default=True, title='4h')
@param.int('labels_kept', default=60, min=1, max=1000, title='Labels kept (each timeframe)')
@param.color('up_color', default=color.GREEN, title='HH / HL colour')
@param.color('down_color', default=color.RED, title='LH / LL colour')
class Main(MainContext):
    def __init__(self, swing_size, show_chart, show_15m, show_1h, show_4h, labels_kept, up_color,
                 down_color):
        self._chart_m = self.time_frame.to_minutes()
        self._secs = float(self._chart_m) * 60.0
        self._mins = [self._chart_m, 15, 60, 240]
        self._names = ['', '15m ', '1h ', '4h ']
        self._on = [show_chart, show_15m and self._chart_m < 15, show_1h and self._chart_m < 60,
                    show_4h and self._chart_m < 240]
        self._kept = labels_kept
        self._up = up_color
        self._down = down_color
        (self._a0, self._a1, self._a2, self._a3, self._a4, self._a5,
         self._a6) = self.calc_on(Swings, time_frame=self.time_frame)
        (self._b0, self._b1, self._b2, self._b3, self._b4, self._b5,
         self._b6) = self.calc_on(Swings, time_frame=self._tf('15m'))
        (self._c0, self._c1, self._c2, self._c3, self._c4, self._c5,
         self._c6) = self.calc_on(Swings, time_frame=self._tf('1h'))
        (self._d0, self._d1, self._d2, self._d3, self._d4, self._d5,
         self._d6) = self.calc_on(Swings, time_frame=self._tf('4h'))
        self._atr_len = 14
        a0 = 1.0 / float(self._atr_len)
        com = (1.0 - a0) / a0
        self._alpha = 1.0 / (1.0 + com)
        self._owf = 1.0 - self._alpha
        self._atr_w = self.new_var(0.0)
        self._atr_n = self.new_var(0)
        self._prev_close = self.new_var(0.0)
        self._seen = self.new_var([0, 0, 0, 0])       # swings already labelled, per timeframe
        # each higher timeframe's recent candles as the chart saw them: open time, and the chart
        # candle that made its high and its low -- where a swing of that timeframe is drawn
        bt: list[float] = []
        ht: list[float] = []
        hp: list[float] = []
        lt: list[float] = []
        lp: list[float] = []
        bs: list[int] = []
        self._bk_t = self.new_var(bt)
        self._bk_ht = self.new_var(ht)
        self._bk_hp = self.new_var(hp)
        self._bk_lt = self.new_var(lt)
        self._bk_lp = self.new_var(lp)
        self._bk_s = self.new_var(bs)
        labs: list[LabelAbs] = []
        f1: list[bool] = []
        f2: list[bool] = []
        o1: list[int] = []
        self._labs = self.new_var(labs)
        self._drawn = self.new_var(f1)
        self._dirty = self.new_var(f2)
        self._slot = self.new_var(o1)
        c1: list[int] = []
        p1: list[float] = []
        t1: list[float] = []
        self._code = self.new_var(c1)                 # per label: the swing it names --
        self._px = self.new_var(p1)                   # 2 HH 1 LH -1 HL -2 LL, its price, and
        self._pt = self.new_var(t1)                   # its candle's open time
        trash: list[LabelAbs] = []
        self._trash = self.new_var(trash)

    def _tf(self, s: str) -> TimeFrame:
        t = TimeFrame.from_str(s)
        if t.to_minutes() < self._chart_m:
            return self.time_frame                     # never below the chart; that slot is off
        return t

    def _vals(self, s: int) -> list[float]:
        if s == 0:
            return [self._a0[0], self._a1[0], self._a2[0], self._a3[0], self._a4[0], self._a5[0],
                    self._a6[0]]
        if s == 1:
            return [self._b0[0], self._b1[0], self._b2[0], self._b3[0], self._b4[0], self._b5[0],
                    self._b6[0]]
        if s == 2:
            return [self._c0[0], self._c1[0], self._c2[0], self._c3[0], self._c4[0], self._c5[0],
                    self._c6[0]]
        return [self._d0[0], self._d1[0], self._d2[0], self._d3[0], self._d4[0], self._d5[0],
                self._d6[0]]

    def calc(self):
        if not self.is_closed_bar:
            return
        a = self._atr()
        t = self.time[0]
        close_t = t + self._secs
        for s in range(1, 4):
            if self._on[s]:
                self._bucket(s, t)
        for s in range(4):
            if self._on[s]:
                v = self._vals(s)
                m = float(self._mins[s]) * 60.0
                # only a candle that has CLOSED by this chart candle's close is used
                if not isnan(v[0]) and not isnan(v[6]) and v[6] + m <= close_t:
                    if int(v[0]) > self._seen.get()[s]:
                        self._seen.get()[s] = int(v[0])
                        self._label(s, int(v[1]), v[2], v[3], a)
        self._sync()

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
        return w

    def _bucket(self, s: int, t: float) -> None:
        """Track which chart candle made each higher-timeframe candle's high and low (the last
        one on a tie, as the zigzag keeps the last equal extreme)."""
        m = float(self._mins[s]) * 60.0
        b = m * float(int(t / m))                      # the candle's UTC bucket
        k = len(self._bk_t.get()) - 1                 # this timeframe's latest candle
        while k >= 0 and self._bk_s.get()[k] != s:
            k -= 1
        if k < 0 or self._bk_t.get()[k] != b:          # a new one starts
            self._bk_t.get().append(b)
            self._bk_s.get().append(s)
            self._bk_ht.get().append(t)
            self._bk_hp.get().append(self.high[0])
            self._bk_lt.get().append(t)
            self._bk_lp.get().append(self.low[0])
            if len(self._bk_t.get()) > 3 * 400:        # ~400 candles of each timeframe
                self._bk_t.get().pop(0)
                self._bk_s.get().pop(0)
                self._bk_ht.get().pop(0)
                self._bk_hp.get().pop(0)
                self._bk_lt.get().pop(0)
                self._bk_lp.get().pop(0)
            return
        if self.high[0] >= self._bk_hp.get()[k]:
            self._bk_hp.get()[k] = self.high[0]
            self._bk_ht.get()[k] = t
        if self.low[0] <= self._bk_lp.get()[k]:
            self._bk_lp.get()[k] = self.low[0]
            self._bk_lt.get()[k] = t

    def _where(self, s: int, pt: float, high: bool) -> float:
        """The chart candle a swing of slot s is drawn on."""
        if s == 0:
            return pt
        for k in range(len(self._bk_t.get())):
            if self._bk_s.get()[k] == s and self._bk_t.get()[k] == pt:
                if high:
                    return self._bk_ht.get()[k]
                return self._bk_lt.get()[k]
        return pt                                      # older than what is kept: its open

    def _label(self, s: int, code: int, px: float, pt: float, a: float) -> None:
        high = code > 0
        word = 'LL'
        if code == 2:
            word = 'HH'
        if code == 1:
            word = 'LH'
        if code == -1:
            word = 'HL'
        col = self._down
        if code == 2 or code == -1:
            col = self._up
        off = 0.0
        if not isnan(a):
            off = a * (0.5 + 0.9 * float(s))           # higher timeframes further out
        y = px - off
        if high:
            y = px + off
        lab = LabelAbs(self._names[s] + word, AbsolutePosition(self._where(s, pt, high), y),
                       bg_color=col(0.15 + 0.1 * float(s)), text_color=col,
                       font_size=9 + 2 * s)
        self._labs.get().append(lab)
        self._drawn.get().append(False)
        self._dirty.get().append(True)
        self._slot.get().append(s)
        self._code.get().append(code)
        self._px.get().append(px)
        self._pt.get().append(pt)
        n = 0
        for k in range(len(self._slot.get())):
            if self._slot.get()[k] == s:
                n += 1
        if n > self._kept:                             # too many: this timeframe's oldest goes
            k = 0
            while self._slot.get()[k] != s:
                k += 1
            if self._drawn.get()[k]:
                self._trash.get().append(self._labs.get()[k])
            self._labs.get().pop(k)
            self._drawn.get().pop(k)
            self._dirty.get().pop(k)
            self._slot.get().pop(k)
            self._code.get().pop(k)
            self._px.get().pop(k)
            self._pt.get().pop(k)

    def _sync(self) -> None:
        """At most 90 drawing changes per update (Exness allows 100); the rest wait."""
        left = 90
        while left > 0 and len(self._trash.get()) > 0:
            self.chart.erase(self._trash.get()[0])
            self._trash.get().pop(0)
            left -= 1
        for k in range(len(self._labs.get())):
            if left > 0 and self._dirty.get()[k]:
                self.chart.draw(self._labs.get()[k])
                self._dirty.get()[k] = False
                self._drawn.get()[k] = True
                left -= 1
