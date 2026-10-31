# indie:lang_version = 5
# TRENDLINES, MULTI-TIMEFRAME -- the email alerts' strict trendlines for the chart's own timeframe
# AND 15m, 1h and 4h on one chart, with a mark where a chart candle first closes through a higher
# timeframe's line (the alerts' EARLY BREAK). For the Exness web terminal: Indicators -> Create ->
# paste -> Save -> Add to chart. Paste exness/probe_timeframes.py first: it says whether Exness lets
# an indicator read other timeframes. Without that, exness/trendlines.py still draws the chart's own.
#
# EACH TIMEFRAME'S LINES ARE THE RESEARCH CODE'S (quant.indicators.add_trendlines, fit="strict"), on
# that timeframe's own candles, with the alerts' own settings for it (`Settings: auto`: the 1m
# alerts' values on 1m, the 5m-4h ones on 5m and above). The fitting below is exness/trendlines.py's
# own code, copied unchanged; tests/test_exness_trendlines_mtf.py holds every timeframe's lines,
# breaks and early marks to the research code. See exness/trendlines.py for how a line is chosen.
#
# SAFE -- nothing is ever drawn from a candle that has not closed:
#   - a 1h line appears when the 1h candle that completes it CLOSES, and breaks on a 1h CLOSE
#     through it, never while the candle forms (each timeframe's fitting runs on closed candles
#     only, and the chart takes a higher timeframe's state only once its candle has closed by the
#     chart candle's close -- whatever the terminal shows while it forms);
#   - an EARLY mark is the first chart candle that closes through a live higher-timeframe line --
#     a 5m close below a 1h support, up to an hour before the 1h candle itself can close there.
#     Once per line. Between that timeframe's closes the line moves on in time; across a shut
#     market (a weekend) it holds, as in the alerts.
#   - a line drawn on the chart passes, at every chart candle, through the price that candle's
#     close is judged against: a candle closing beyond the drawn line is what makes the mark.
#
# DISTINGUISHABLE: red = resistance, green = support on every timeframe; THICKNESS is the timeframe
# (the chart's own thinnest, 4h thickest), and a higher timeframe's live line carries a tag at its
# tip ('1h R', '4h S'). Broken lines stay faded; dropped ones grey and dotted.
#
# Live, a higher timeframe's state can reach the chart one chart candle after its close (the
# terminal decides the order); the line then appears one candle later than on a reload, and no
# early mark is judged on a candle whose higher-timeframe state is late.
from math import isnan, nan
from indie import indicator, param, param_ref, color, line_style, MainContext, SecContext
from indie import TimeFrame
from indie.color import rgba
from indie.drawings import LineSegment, LabelAbs, AbsolutePosition


@param_ref('preset')
@param_ref('swing_size')
@param_ref('min_touches')
@param_ref('touch_tol')
@param_ref('wick_tol')
@param_ref('max_distance')
@param_ref('hold')
@param_ref('span')
@param_ref('prefer_newer')
class Lines(SecContext):
    """exness/trendlines.py's fitter on this context's own candles. After each closed candle it
    returns, per side (resistance, support): on, born, broke this candle, born candle's time, first
    touch, its time, slope, intercept -- and the candle's index, open time and close."""
    def __init__(self, preset, swing_size, min_touches, touch_tol, wick_tol, max_distance, hold,
                 span, prefer_newer):
        mode = preset
        if mode == 'auto':
            if self.time_frame.to_minutes() == 1:
                mode = '1m alerts'
            else:
                mode = '5m-4h alerts'
        self._swing = 3.5                            # the 5m-4h alerts' values (STRICT) ...
        self._min_touches = 3
        self._tol = 0.3
        self._wick = 0.15
        self._maxd = 8.0
        self._hold = 0
        self._span = 0.5
        self._newer = False
        if mode == '1m alerts':                     # ... and the 1m alerts' (STRICT_1M)
            self._tol = 0.4
            self._wick = 0.3
            self._maxd = 20.0
            self._hold = 1
            self._span = 1.0 / 3.0
            self._newer = True
        elif mode == 'custom':
            self._swing = swing_size
            self._min_touches = min_touches
            self._tol = touch_tol
            self._wick = wick_tol
            self._maxd = max_distance
            self._hold = hold
            self._span = span
            self._newer = prefer_newer
        self._lookback = 16
        self._min_span_bars = 40
        self._bar_secs = float(self.time_frame.to_minutes()) * 60.0
        self._max_age = 500
        self._atr_len = 14
        a0 = 1.0 / float(self._atr_len)               # pandas' ewm(alpha=1/14, adjust=False),
        com = (1.0 - a0) / a0                         # bit for bit
        self._alpha = 1.0 / (1.0 + com)
        self._owf = 1.0 - self._alpha
        self._atr_w = self.new_var(0.0)
        self._atr_n = self.new_var(0)
        self._prev_close = self.new_var(0.0)
        self._zz_dir = self.new_var(0)
        self._zz_px = self.new_var(0.0)
        self._zz_i = self.new_var(-1)
        self._zz_hi = self.new_var(nan)
        self._zz_lo = self.new_var(nan)
        pv_i: list[int] = []
        pv_px: list[float] = []
        pv_kind: list[int] = []
        self._pv_i = self.new_var(pv_i)
        self._pv_px = self.new_var(pv_px)
        self._pv_kind = self.new_var(pv_kind)
        rs: list[str] = []
        rf: list[int] = []
        rk: list[int] = []
        self._ret_sig = self.new_var(rs)
        self._ret_first = self.new_var(rf)
        self._ret_kind = self.new_var(rk)
        self._on = self.new_var([False, False])
        self._slope = self.new_var([0.0, 0.0])
        self._icpt = self.new_var([0.0, 0.0])
        self._touches = self.new_var([0, 0])
        self._start = self.new_var([0, 0])
        self._born = self.new_var([-1, -1])
        self._end = self.new_var([-1, -1])
        self._beyond = self.new_var([0, 0])
        self._last = self.new_var([0, 0])
        self._sig = self.new_var(['', ''])
        self._sfirst = self.new_var([0, 0])
        self._level = self.new_var([nan, nan])
        self._brk = self.new_var([False, False])
        # THE CANDLES THE FITTER READS, KEPT HERE. Exness lets a script read its own candles live
        # only as far back as it read while the chart was loading ("Series offset=510 is out of
        # size=506", 30 Oct 2026), and a line checked live can reach further back than any did
        # while loading. 2,000 candles: the deepest any line needed on six long histories (1m to
        # 4h, 6,000-60,000 candles each) was 769, and drawing one 979
        self._keep = 2000
        hc: list[float] = []
        hh: list[float] = []
        hl: list[float] = []
        ht: list[float] = []
        self._hc = self.new_var(hc)
        self._hh = self.new_var(hh)
        self._hl = self.new_var(hl)
        self._ht = self.new_var(ht)
        out: list[float] = []
        for k in range(19):
            out.append(nan)
        self._out = self.new_var(out)
        # the state before the latest candle, for when it is computed again
        self._k_atr_w = self.new_var(self._atr_w.get())
        self._k_atr_n = self.new_var(self._atr_n.get())
        self._k_prev_close = self.new_var(self._prev_close.get())
        self._k_zz_dir = self.new_var(self._zz_dir.get())
        self._k_zz_px = self.new_var(self._zz_px.get())
        self._k_zz_i = self.new_var(self._zz_i.get())
        self._k_zz_hi = self.new_var(self._zz_hi.get())
        self._k_zz_lo = self.new_var(self._zz_lo.get())
        self._k_pv_i = self.new_var(list(self._pv_i.get()))
        self._k_pv_px = self.new_var(list(self._pv_px.get()))
        self._k_pv_kind = self.new_var(list(self._pv_kind.get()))
        self._k_ret_sig = self.new_var(list(self._ret_sig.get()))
        self._k_ret_first = self.new_var(list(self._ret_first.get()))
        self._k_ret_kind = self.new_var(list(self._ret_kind.get()))
        self._k_on = self.new_var(list(self._on.get()))
        self._k_slope = self.new_var(list(self._slope.get()))
        self._k_icpt = self.new_var(list(self._icpt.get()))
        self._k_touches = self.new_var(list(self._touches.get()))
        self._k_start = self.new_var(list(self._start.get()))
        self._k_born = self.new_var(list(self._born.get()))
        self._k_end = self.new_var(list(self._end.get()))
        self._k_beyond = self.new_var(list(self._beyond.get()))
        self._k_last = self.new_var(list(self._last.get()))
        self._k_sig = self.new_var(list(self._sig.get()))
        self._k_sfirst = self.new_var(list(self._sfirst.get()))
        self._k_level = self.new_var(list(self._level.get()))
        self._k_brk = self.new_var(list(self._brk.get()))
        self._k_out = self.new_var(list(self._out.get()))
        self._done_t = self.new_var(nan)        # the latest candle computed

    def calc(self):
        if self.is_closed_bar:
            self._once()
            self._remember()
            i = self.bar_index
            a = self._atr()
            confirmed = self._zigzag(i, a)
            if confirmed and not isnan(a) and a > 0:
                self._fit_side(i, a, 1)
                self._fit_side(i, a, -1)
            self._publish(i, a)
            self._emit(i)
        o = self._out.get()
        return (o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11],
                o[12], o[13], o[14], o[15], o[16], o[17], o[18])

    def _emit(self, i: int) -> None:
        o = self._out.get()
        for s in range(2):
            b = s * 8
            on = self._on.get()[s]
            brk = self._brk.get()[s]
            born = self._born.get()[s]
            o[b] = 0.0
            if on:
                o[b] = 1.0
            o[b + 1] = float(born)
            o[b + 2] = 0.0
            if brk:
                o[b + 2] = 1.0
            for q in range(3, 8):
                o[b + q] = nan
            if (on or brk) and born >= 0:
                st = self._start.get()[s]
                o[b + 3] = self._t_ago(i - born)
                o[b + 4] = float(st)
                o[b + 5] = self._t_ago(i - st)
                o[b + 6] = self._slope.get()[s]
                o[b + 7] = self._icpt.get()[s]
        o[16] = float(i)
        o[17] = self.time[0]
        o[18] = self.close[0]

    def _save(self) -> None:
        self._k_atr_w.set(self._atr_w.get())
        self._k_atr_n.set(self._atr_n.get())
        self._k_prev_close.set(self._prev_close.get())
        self._k_zz_dir.set(self._zz_dir.get())
        self._k_zz_px.set(self._zz_px.get())
        self._k_zz_i.set(self._zz_i.get())
        self._k_zz_hi.set(self._zz_hi.get())
        self._k_zz_lo.set(self._zz_lo.get())
        self._k_pv_i.set(list(self._pv_i.get()))
        self._k_pv_px.set(list(self._pv_px.get()))
        self._k_pv_kind.set(list(self._pv_kind.get()))
        self._k_ret_sig.set(list(self._ret_sig.get()))
        self._k_ret_first.set(list(self._ret_first.get()))
        self._k_ret_kind.set(list(self._ret_kind.get()))
        self._k_on.set(list(self._on.get()))
        self._k_slope.set(list(self._slope.get()))
        self._k_icpt.set(list(self._icpt.get()))
        self._k_touches.set(list(self._touches.get()))
        self._k_start.set(list(self._start.get()))
        self._k_born.set(list(self._born.get()))
        self._k_end.set(list(self._end.get()))
        self._k_beyond.set(list(self._beyond.get()))
        self._k_last.set(list(self._last.get()))
        self._k_sig.set(list(self._sig.get()))
        self._k_sfirst.set(list(self._sfirst.get()))
        self._k_level.set(list(self._level.get()))
        self._k_brk.set(list(self._brk.get()))
        self._k_out.set(list(self._out.get()))

    def _load(self) -> None:
        self._atr_w.set(self._k_atr_w.get())
        self._atr_n.set(self._k_atr_n.get())
        self._prev_close.set(self._k_prev_close.get())
        self._zz_dir.set(self._k_zz_dir.get())
        self._zz_px.set(self._k_zz_px.get())
        self._zz_i.set(self._k_zz_i.get())
        self._zz_hi.set(self._k_zz_hi.get())
        self._zz_lo.set(self._k_zz_lo.get())
        self._pv_i.set(list(self._k_pv_i.get()))
        self._pv_px.set(list(self._k_pv_px.get()))
        self._pv_kind.set(list(self._k_pv_kind.get()))
        self._ret_sig.set(list(self._k_ret_sig.get()))
        self._ret_first.set(list(self._k_ret_first.get()))
        self._ret_kind.set(list(self._k_ret_kind.get()))
        self._on.set(list(self._k_on.get()))
        self._slope.set(list(self._k_slope.get()))
        self._icpt.set(list(self._k_icpt.get()))
        self._touches.set(list(self._k_touches.get()))
        self._start.set(list(self._k_start.get()))
        self._born.set(list(self._k_born.get()))
        self._end.set(list(self._k_end.get()))
        self._beyond.set(list(self._k_beyond.get()))
        self._last.set(list(self._k_last.get()))
        self._sig.set(list(self._k_sig.get()))
        self._sfirst.set(list(self._k_sfirst.get()))
        self._level.set(list(self._k_level.get()))
        self._brk.set(list(self._k_brk.get()))
        self._out.set(list(self._k_out.get()))

    def _once(self) -> None:
        """Exness computes a timeframe's still-forming candle as if closed when a chart
        loads. If that candle is computed again when it really closes, start it from the
        state before its first time, so it counts once, with its final values."""
        if self.time[0] == self._done_t.get():
            self._load()
        else:
            self._save()
        self._done_t.set(self.time[0])

    def _finish(self, s: int, broke: bool) -> None:
        self._on.get()[s] = False

    def _atr(self) -> float:
        h = self.high[0]
        lo = self.low[0]
        c = self.close[0]
        tr = h - lo
        n = self._atr_n.get()
        if n > 0:
            pc = self._prev_close.get()
            tr = max(tr, max(abs(h - pc), abs(lo - pc)))
        self._prev_close.set(c)
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

    def _zigzag(self, i: int, a: float) -> bool:
        if isnan(a):
            return False
        t = a * self._swing
        if t <= 0:
            return False
        d = self._zz_dir.get()
        if d == 0:
            if self._zz_i.get() < 0:
                self._zz_px.set(self.high[0])
                self._zz_i.set(i)
                self._zz_dir.set(1)
            return False
        confirmed = False
        px = self._zz_px.get()
        if d == 1:
            if self.high[0] >= px:
                self._zz_px.set(self.high[0])
                self._zz_i.set(i)
            elif px - self.low[0] >= t:
                if not isnan(self._zz_hi.get()):
                    self._add_pivot(self._zz_i.get(), px, 1)
                    confirmed = True
                self._zz_hi.set(px)
                self._zz_dir.set(-1)
                self._zz_px.set(self.low[0])
                self._zz_i.set(i)
        else:
            if self.low[0] <= px:
                self._zz_px.set(self.low[0])
                self._zz_i.set(i)
            elif self.high[0] - px >= t:
                if not isnan(self._zz_lo.get()):
                    self._add_pivot(self._zz_i.get(), px, -1)
                    confirmed = True
                self._zz_lo.set(px)
                self._zz_dir.set(1)
                self._zz_px.set(self.high[0])
                self._zz_i.set(i)
        return confirmed

    def _add_pivot(self, at: int, price: float, kind: int) -> None:
        self._pv_i.get().append(at)
        self._pv_px.get().append(price)
        self._pv_kind.get().append(kind)
        while len(self._pv_i.get()) > self._lookback:
            self._pv_i.get().pop(0)
            self._pv_px.get().pop(0)
            self._pv_kind.get().pop(0)
        # a retired line through a swing that has left the window can never be fitted again
        oldest = self._pv_i.get()[0]
        k = len(self._ret_first.get()) - 1
        while k >= 0:
            if self._ret_first.get()[k] < oldest:
                self._ret_sig.get().pop(k)
                self._ret_first.get().pop(k)
                self._ret_kind.get().pop(k)
            k -= 1

    def _price(self, b: int) -> float:
        hc = self._hc.get()
        return hc[len(hc) - 1 - (self.bar_index - b)]

    def _clear(self, slope: float, icpt: float, x0: int, kind: int, wick: float) -> bool:
        """No close beyond the line and no wick beyond it by more than `wick`, on every candle
        from x0 to the latest one."""
        i = self.bar_index
        hc = self._hc.get()
        hh = self._hh.get()
        hl = self._hl.get()
        n = len(hc)
        if i - x0 >= n:
            return False                       # older than the candles kept: cannot be proved
        # the research code's tolerance is 1e-9 x the largest |line price| on the span; a line's
        # largest is at one of its ends (rounding keeps slope x candle + intercept in order)
        big = max(1.0, max(abs(slope * float(x0) + icpt), abs(slope * float(i) + icpt)))
        eps = 1e-9 * big
        ok = True
        b = x0
        while ok and b <= i:
            y = slope * float(b) + icpt
            q = n - 1 - (i - b)
            if kind < 0:
                ok = hc[q] >= y - eps and hl[q] >= y - wick - eps
            else:
                ok = hc[q] <= y + eps and hh[q] <= y + wick + eps
            b += 1
        return ok

    def _rowsum(self, vals: list[float]) -> float:
        """numpy's sum of a row, in numpy's order: left to right below 8 values; from 8, eight
        running totals combined in pairs, then the rest."""
        n = len(vals)
        if n < 8:
            r = vals[0]
            j = 1
            while j < n:
                r = r + vals[j]
                j += 1
            return r
        acc: list[float] = []
        for j in range(8):
            acc.append(vals[j])
        j = 8
        while j + 8 <= n:
            for q in range(8):
                acc[q] = acc[q] + vals[j + q]
            j += 8
        res = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
        while j < n:
            res = res + vals[j]
            j += 1
        return res

    def _fit_side(self, i: int, a: float, kind: int) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for k in range(len(self._pv_i.get())):
            if self._pv_kind.get()[k] == kind:
                xs.append(float(self._pv_i.get()[k]))
                ys.append(self._pv_px.get()[k])
        n = len(xs)
        if n < self._min_touches or n < 2:
            return
        tol = self._tol * a
        lo_x = xs[0]
        hi_x = xs[0]
        for k in range(n):
            lo_x = min(lo_x, xs[k])
            hi_x = max(hi_x, xs[k])
        total_span = max(hi_x - lo_x, 1.0)
        # every pair of swings proposes a line; keep the well-supported ones
        c_slope: list[float] = []
        c_icpt: list[float] = []
        c_count: list[int] = []
        c_span: list[float] = []
        c_dev: list[float] = []
        for p in range(n):
            for q in range(p + 1, n):
                dx = xs[q] - xs[p]
                if dx > 0:
                    slope = (ys[q] - ys[p]) / dx
                    icpt = ys[p] - slope * xs[p]
                    count = 0
                    first = 0.0
                    last = 0.0
                    devs: list[float] = []
                    for r in range(n):
                        dev = abs(ys[r] - (slope * xs[r] + icpt))
                        if dev <= tol:
                            if count == 0:
                                first = xs[r]
                                last = xs[r]
                            first = min(first, xs[r])
                            last = max(last, xs[r])
                            count += 1
                            devs.append(dev)
                        else:
                            devs.append(0.0)
                    if count >= self._min_touches and last - first >= self._span * total_span:
                        c_slope.append(slope)
                        c_icpt.append(icpt)
                        c_count.append(count)
                        c_span.append(last - first)
                        c_dev.append(self._rowsum(devs) / float(max(count, 1)))
        # most touches first, then the widest, then the closest fit (a stable order, as numpy's)
        order: list[int] = []
        for c in range(len(c_count)):
            order.append(c)
            pos = len(order) - 1
            while pos > 0 and self._better(c, order[pos - 1], c_count, c_span, c_dev):
                order[pos] = order[pos - 1]
                pos -= 1
            order[pos] = c
        wick = self._wick * a
        found = -1
        f_first = 0
        f_last = 0
        f_sig = ''
        o = 0
        while found < 0 and o < len(order):
            c = order[o]
            sig = ''
            first_i = -1
            last_i = -1
            for r in range(n):
                if abs(ys[r] - (c_slope[c] * xs[r] + c_icpt[c])) <= tol:
                    xi = int(xs[r])
                    if first_i < 0:
                        first_i = xi
                    last_i = xi
                    sig = sig + str(xi) + ','
            if self._clear(c_slope[c], c_icpt[c], first_i, kind, wick):
                found = c
                f_first = first_i
                f_last = last_i
                f_sig = sig
            o += 1
        if found < 0 or f_last - f_first < self._min_span_bars:
            return
        self._take(i, a, kind, c_slope[found], c_icpt[found], c_count[found], f_first, f_last,
                   f_sig)

    def _better(self, c: int, d: int, cnt: list[int], spn: list[float], dev: list[float]) -> bool:
        if cnt[c] != cnt[d]:
            return cnt[c] > cnt[d]
        if spn[c] != spn[d]:
            return spn[c] > spn[d]
        return dev[c] < dev[d]

    def _take(self, i: int, a: float, kind: int, slope: float, icpt: float, touches: int,
              first: int, last: int, sig: str) -> None:
        for k in range(len(self._ret_sig.get())):
            if self._ret_kind.get()[k] == kind and self._ret_sig.get()[k] == sig:
                return                                # broken or dropped lines never come back
        if abs(self._price(i) - (slope * float(i) + icpt)) > self._maxd * a:
            return                                    # already far from price: stale on arrival
        s = 0 if kind > 0 else 1
        if self._on.get()[s]:
            if touches <= self._touches.get()[s]:
                newer = (self._newer and touches == self._touches.get()[s]
                         and last > self._last.get()[s])
                if not newer:
                    return
            # the SAME line refined: within tolerance of the live one across its span, and clean
            same = True
            x0 = self._start.get()[s]
            live_y0 = self._slope.get()[s] * float(x0) + self._icpt.get()[s]
            if abs(slope * float(x0) + icpt - live_y0) > self._tol * a:
                same = False
            live_y1 = self._slope.get()[s] * float(i) + self._icpt.get()[s]
            if abs(slope * float(i) + icpt - live_y1) > self._tol * a:
                same = False
            if same:
                same = self._clear(slope, icpt, min(self._start.get()[s], first), kind,
                                   self._wick * a)
            if same:
                self._slope.get()[s] = slope
                self._icpt.get()[s] = icpt
                self._touches.get()[s] = touches
                self._sig.get()[s] = sig
                self._sfirst.get()[s] = first
                self._last.get()[s] = last
                self._start.get()[s] = min(self._start.get()[s], first)
                return
            self._finish(s, False)                    # a better line takes the slot
        self._on.get()[s] = True
        self._slope.get()[s] = slope
        self._icpt.get()[s] = icpt
        self._touches.get()[s] = touches
        self._start.get()[s] = first
        self._born.get()[s] = i
        self._end.get()[s] = -1
        self._beyond.get()[s] = 0
        self._last.get()[s] = last
        self._sig.get()[s] = sig
        self._sfirst.get()[s] = first

    def _retire(self, s: int) -> None:
        self._ret_sig.get().append(self._sig.get()[s])
        self._ret_first.get().append(self._sfirst.get()[s])
        kind = -1
        if s == 0:
            kind = 1
        self._ret_kind.get().append(kind)

    def _publish(self, i: int, a: float) -> None:
        if isnan(a):
            a = 0.0
        on = self._on.get()
        if on[0] and on[1]:
            res_y = self._slope.get()[0] * float(i) + self._icpt.get()[0]
            sup_y = self._slope.get()[1] * float(i) + self._icpt.get()[1]
            if sup_y > res_y:                          # a support above a resistance: the weaker
                drop = 1                               # goes (fewer touches; the younger on a tie)
                t_sup = self._touches.get()[1]
                t_res = self._touches.get()[0]
                if t_res < t_sup or (t_res == t_sup and -self._born.get()[0] < -self._born.get()[1]):
                    drop = 0
                self._retire(drop)
                self._finish(drop, False)
        self._level.set([nan, nan])
        self._brk.set([False, False])
        for s in range(2):
            if self._on.get()[s]:
                y = self._slope.get()[s] * float(i) + self._icpt.get()[s]
                if i - self._born.get()[s] > self._max_age:
                    self._retire(s)
                    self._finish(s, False)
                elif a > 0 and abs(self._price(i) - y) > self._maxd * a:
                    self._retire(s)
                    self._finish(s, False)
                else:
                    self._end.get()[s] = i
                    self._level.get()[s] = y
                    beyond = self._price(i) > y
                    if s == 1:
                        beyond = self._price(i) < y
                    if beyond:
                        self._beyond.get()[s] = self._beyond.get()[s] + 1
                    else:
                        self._beyond.get()[s] = 0
                    if beyond and self._beyond.get()[s] > self._hold:
                        self._brk.get()[s] = True
                        self._retire(s)
                        self._finish(s, True)

    def _remember(self) -> None:
        """Keep this closed candle in the fitter's own history (the last `_keep` of them)."""
        ht = self._ht.get()
        n = len(ht)
        if n > 0 and ht[n - 1] == self.time[0]:      # the same candle again: its final values
            self._hc.get()[n - 1] = self.close[0]
            self._hh.get()[n - 1] = self.high[0]
            self._hl.get()[n - 1] = self.low[0]
            return
        ht.append(self.time[0])
        self._hc.get().append(self.close[0])
        self._hh.get().append(self.high[0])
        self._hl.get().append(self.low[0])
        if n + 1 > self._keep:
            ht.pop(0)
            self._hc.get().pop(0)
            self._hh.get().pop(0)
            self._hl.get().pop(0)

    def _t_ago(self, k: int) -> float:
        """The open time of the candle k back; one older than those kept is counted back in
        candles (used only to draw where a line starts)."""
        ht = self._ht.get()
        n = len(ht)
        if k < n:
            return ht[n - 1 - k]
        return ht[0] - float(k - (n - 1)) * self._bar_secs


@indicator('Trendlines MTF (quant)', overlay_main_pane=True)
@param.str('preset', default='auto', options=['auto', '1m alerts', '5m-4h alerts', 'custom'], title='Settings: auto = the email alerts own for each timeframe')
@param.float('swing_size', default=3.5, min=0.5, max=20.0, step=0.1, title='custom: Swing size (ATR)')
@param.int('min_touches', default=3, min=2, max=10, title='custom: Min touches')
@param.float('touch_tol', default=0.3, min=0.01, max=5.0, step=0.05, title='custom: Touch tolerance (ATR)')
@param.float('wick_tol', default=0.15, min=0.0, max=5.0, step=0.05, title='custom: Wick tolerance (ATR)')
@param.float('max_distance', default=8.0, min=0.5, max=100.0, step=0.5, title='custom: Max distance (ATR)')
@param.int('hold', default=0, min=0, max=10, title='custom: Hold (more closes beyond to break)')
@param.float('span', default=0.5, min=0.05, max=1.0, step=0.05, title='custom: Span (share of the swings stretch)')
@param.bool('prefer_newer', default=False, title='custom: Prefer newer line on a tie')
@param.bool('show_chart', default=True, title="This chart's timeframe")
@param.bool('show_15m', default=True, title='15m lines')
@param.bool('show_1h', default=True, title='1h lines')
@param.bool('show_4h', default=True, title='4h lines')
@param.bool('early', default=True, title='Mark early breaks of 15m / 1h / 4h lines')
@param.bool('keep_old', default=True, title='Keep broken and dropped lines')
@param.int('max_kept', default=30, min=1, max=500, title='Old lines kept (each timeframe)')
@param.int('marks_kept', default=60, min=1, max=500, title='Early marks kept')
@param.int('reach', default=20, min=2, max=500, title='Live lines reach ahead (candles)')
@param.color('res_color', default=color.RED, title='Resistance colour')
@param.color('sup_color', default=color.GREEN, title='Support colour')
@param.color('old_color', default=rgba(128, 128, 128, 1.0), title='Dropped line colour')
class Main(MainContext):
    def __init__(self, preset, swing_size, min_touches, touch_tol, wick_tol, max_distance, hold,
                 span, prefer_newer, show_chart, show_15m, show_1h, show_4h, early, keep_old,
                 max_kept, marks_kept, reach, res_color, sup_color, old_color):
        self._chart_m = self.time_frame.to_minutes()
        self._secs = float(self._chart_m) * 60.0
        self._mins = [self._chart_m, 15, 60, 240]       # slot 0 = the chart's own timeframe
        self._names = ['', '15m', '1h', '4h']
        self._on = [show_chart, show_15m and self._chart_m < 15, show_1h and self._chart_m < 60,
                    show_4h and self._chart_m < 240]
        self._early = early
        self._keep_old = keep_old
        self._max_kept = max_kept
        self._marks_kept = marks_kept
        self._reach = reach
        self._colors = [res_color, sup_color]          # side 0 = resistance, 1 = support
        self._old_color = old_color
        (self._a0, self._a1, self._a2, self._a3, self._a4, self._a5, self._a6, self._a7,
         self._a8, self._a9, self._a10, self._a11, self._a12, self._a13, self._a14, self._a15,
         self._a16, self._a17, self._a18) = self.calc_on(Lines, time_frame=self.time_frame)
        (self._b0, self._b1, self._b2, self._b3, self._b4, self._b5, self._b6, self._b7,
         self._b8, self._b9, self._b10, self._b11, self._b12, self._b13, self._b14, self._b15,
         self._b16, self._b17, self._b18) = self.calc_on(Lines, time_frame=self._tf('15m'))
        (self._c0, self._c1, self._c2, self._c3, self._c4, self._c5, self._c6, self._c7,
         self._c8, self._c9, self._c10, self._c11, self._c12, self._c13, self._c14, self._c15,
         self._c16, self._c17, self._c18) = self.calc_on(Lines, time_frame=self._tf('1h'))
        (self._d0, self._d1, self._d2, self._d3, self._d4, self._d5, self._d6, self._d7,
         self._d8, self._d9, self._d10, self._d11, self._d12, self._d13, self._d14, self._d15,
         self._d16, self._d17, self._d18) = self.calc_on(Lines, time_frame=self._tf('4h'))
        self._atr_len = 14
        a0 = 1.0 / float(self._atr_len)
        com = (1.0 - a0) / a0
        self._alpha = 1.0 / (1.0 + com)
        self._owf = 1.0 - self._alpha
        self._atr_w = self.new_var(0.0)
        self._atr_n = self.new_var(0)
        self._prev_close = self.new_var(0.0)
        self._prev_t = self.new_var(nan)                # the previous chart candle's open
        # per timeframe: the latest state taken (its candle's index and open time)
        self._lj = self.new_var([-1, -1, -1, -1])
        self._ltj = self.new_var([nan, nan, nan, nan])
        # per line slot k = timeframe * 2 + side: the line drawn there and its geometry
        self._cur = self.new_var([-1, -1, -1, -1, -1, -1, -1, -1])
        self._g_start = self.new_var([0, 0, 0, 0, 0, 0, 0, 0])
        self._g_tstart = self.new_var([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._g_slope = self.new_var([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._g_icpt = self.new_var([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._g_seen = self.new_var([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._done = self.new_var([False, False, False, False, False, False, False, False])
        segs: list[LineSegment] = []
        tags: list[LabelAbs] = []
        for k in range(8):
            segs.append(LineSegment(AbsolutePosition(0.0, 0.0), AbsolutePosition(0.0, 0.0)))
            tags.append(LabelAbs('', AbsolutePosition(0.0, 0.0), bg_color=res_color(0.2),
                                 text_color=res_color, font_size=10))
        self._seg = self.new_var(segs)
        self._seg_on = self.new_var([False, False, False, False, False, False, False, False])
        self._seg_drawn = self.new_var([False, False, False, False, False, False, False, False])
        self._seg_dirty = self.new_var([False, False, False, False, False, False, False, False])
        self._seg_tip = self.new_var([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._tag = self.new_var(tags)
        self._tag_drawn = self.new_var([False, False, False, False, False, False, False, False])
        self._tag_dirty = self.new_var([False, False, False, False, False, False, False, False])
        old: list[LineSegment] = []
        f1: list[bool] = []
        f2: list[bool] = []
        o1: list[int] = []
        o2: list[int] = []
        self._old = self.new_var(old)
        self._old_drawn = self.new_var(f1)
        self._old_dirty = self.new_var(f2)
        self._old_slot = self.new_var(o1)              # which timeframe
        self._old_broke = self.new_var(o2)             # 1 broken, 0 dropped
        marks: list[LabelAbs] = []
        f3: list[bool] = []
        f4: list[bool] = []
        m1: list[int] = []
        m2: list[float] = []
        self._marks = self.new_var(marks)
        self._mk_drawn = self.new_var(f3)
        self._mk_dirty = self.new_var(f4)
        self._mk_k = self.new_var(m1)                  # its line slot
        self._mk_t = self.new_var(m2)                  # the chart candle it marks
        trash: list[LineSegment] = []
        ltrash: list[LabelAbs] = []
        self._trash = self.new_var(trash)
        self._ltrash = self.new_var(ltrash)

    def _tf(self, s: str) -> TimeFrame:
        t = TimeFrame.from_str(s)
        if t.to_minutes() < self._chart_m:
            return self.time_frame                       # never below the chart; slot is off
        return t

    def _vals(self, s: int) -> list[float]:
        if s == 0:
            return [self._a0[0], self._a1[0], self._a2[0], self._a3[0], self._a4[0], self._a5[0],
                    self._a6[0], self._a7[0], self._a8[0], self._a9[0], self._a10[0],
                    self._a11[0], self._a12[0], self._a13[0], self._a14[0], self._a15[0],
                    self._a16[0], self._a17[0], self._a18[0]]
        if s == 1:
            return [self._b0[0], self._b1[0], self._b2[0], self._b3[0], self._b4[0], self._b5[0],
                    self._b6[0], self._b7[0], self._b8[0], self._b9[0], self._b10[0],
                    self._b11[0], self._b12[0], self._b13[0], self._b14[0], self._b15[0],
                    self._b16[0], self._b17[0], self._b18[0]]
        if s == 2:
            return [self._c0[0], self._c1[0], self._c2[0], self._c3[0], self._c4[0], self._c5[0],
                    self._c6[0], self._c7[0], self._c8[0], self._c9[0], self._c10[0],
                    self._c11[0], self._c12[0], self._c13[0], self._c14[0], self._c15[0],
                    self._c16[0], self._c17[0], self._c18[0]]
        return [self._d0[0], self._d1[0], self._d2[0], self._d3[0], self._d4[0], self._d5[0],
                self._d6[0], self._d7[0], self._d8[0], self._d9[0], self._d10[0], self._d11[0],
                self._d12[0], self._d13[0], self._d14[0], self._d15[0], self._d16[0],
                self._d17[0], self._d18[0]]

    def calc(self):
        if not self.is_closed_bar:
            return                                    # nothing moves while a candle forms
        a = self._chart_atr()
        t = self.time[0]
        close_t = t + self._secs
        for s in range(4):
            if self._on[s]:
                v = self._vals(s)
                m = float(self._mins[s]) * 60.0
                tj = v[17]
                # a state is taken once its candle has CLOSED by this candle's close -- and at
                # that very close only if its close is this candle's close (the same market at
                # the same instant): otherwise it may not be final yet, and it is taken a
                # chart candle later
                done = tj + m < close_t or (tj + m == close_t and v[18] == self.close[0])
                if not isnan(tj) and done and tj != self._ltj.get()[s]:
                    self._take_state(s, v)
                for side in range(2):
                    k = s * 2 + side
                    if self._cur.get()[k] >= 0:
                        if self._seg_tip.get()[k] < t + 2.0 * self._secs:
                            self._place(k, t + float(self._reach) * self._secs)
                        if self._early and s > 0 and not self._done.get()[k]:
                            self._check_early(s, side, t, close_t, a)
        self._prev_t.set(t)
        self._sync()

    def _chart_atr(self) -> float:
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

    # ── a timeframe's new closed candle: lines that ended, lines that began or were refitted ──
    def _take_state(self, s: int, v: list[float]) -> None:
        j = int(v[16])
        tj = v[17]
        m = float(self._mins[s]) * 60.0
        pj = self._lj.get()[s]                         # the candle taken before this one
        ptj = self._ltj.get()[s]
        self._lj.get()[s] = j
        self._ltj.get()[s] = tj
        for side in range(2):
            k = s * 2 + side
            b = side * 8
            on = v[b] == 1.0
            born = int(v[b + 1])
            broke = v[b + 2] == 1.0
            cur = self._cur.get()[k]
            if cur >= 0 and (not on or born != cur):
                if broke and born == cur:              # broken on this candle: ends here
                    self._end_line(k, j, tj, True)
                else:                                  # dropped: ends where it was last valid
                    self._end_line(k, pj, ptj, False)
            if on:
                st = int(v[b + 4])
                if born != self._cur.get()[k]:         # a new line
                    self._cur.get()[k] = born
                    self._g_start.get()[k] = st
                    self._g_tstart.get()[k] = v[b + 5]
                    self._g_slope.get()[k] = v[b + 6]
                    self._g_icpt.get()[k] = v[b + 7]
                    self._g_seen.get()[k] = v[b + 3] + m
                    self._done.get()[k] = False
                    w = s + 1
                    self._seg.get()[k] = LineSegment(AbsolutePosition(0.0, 0.0),
                                                     AbsolutePosition(0.0, 0.0),
                                                     color=self._colors[side],
                                                     line_style=line_style.SOLID, line_width=w)
                    self._seg_on.get()[k] = True
                    self._seg_drawn.get()[k] = False
                    self._place(k, self.time[0] + float(self._reach) * self._secs)
                elif (st != self._g_start.get()[k] or v[b + 6] != self._g_slope.get()[k]
                      or v[b + 7] != self._g_icpt.get()[k]):   # the same line, refitted
                    self._g_start.get()[k] = st
                    self._g_tstart.get()[k] = v[b + 5]
                    self._g_slope.get()[k] = v[b + 6]
                    self._g_icpt.get()[k] = v[b + 7]
                    self._place(k, self._seg_tip.get()[k])

    # ── where a line is on the chart ──
    def _y(self, k: int, x: float) -> float:
        return self._g_slope.get()[k] * x + self._g_icpt.get()[k]

    def _x(self, s: int, close_t: float) -> float:
        """The line coordinate a chart close at close_t is judged at: x = j at candle j's close;
        on through the next candle linearly in time; held across a shut market. NaN when this
        timeframe's latest state is late (live only: the terminal sent it after this candle)."""
        j = float(self._lj.get()[s])
        tj = self._ltj.get()[s]
        m = float(self._mins[s]) * 60.0
        t_open = close_t - self._secs
        b = tj + m * float(int((t_open - tj) / m))     # the candle this chart candle is in
        if b <= tj:
            return j - 1.0 + (close_t - tj) / m
        late = close_t >= b + m                        # its own candle closed, state not here
        prev = self._prev_t.get()
        if not isnan(prev) and prev >= tj + m and prev < b:
            late = True                                # a whole candle since tj, state not here
        if late:
            return nan
        return j + (close_t - b) / m

    def _place(self, k: int, tip: float) -> None:
        """Lay line slot k from its first touch to the chart candle opening at `tip`: at every
        chart candle, the price its close is judged against."""
        s = k // 2
        m = float(self._mins[s]) * 60.0
        seg = self._seg.get()[k]
        x0 = float(self._g_start.get()[k])
        seg.point_a = AbsolutePosition(self._g_tstart.get()[k] + m - self._secs, self._y(k, x0))
        now = self.time[0]
        x = self._x(s, now + self._secs)
        if isnan(x):
            x = float(self._lj.get()[s])
            now = self._ltj.get()[s] + m - self._secs
        xt = x + (tip - now) / m
        seg.point_b = AbsolutePosition(tip, self._y(k, xt))
        self._seg_tip.get()[k] = tip
        self._seg_dirty.get()[k] = True
        if s > 0:
            side = k % 2
            tag = self._tag.get()[k]
            word = ' R'
            if side == 1:
                word = ' S'
            tag.text = self._names[s] + word
            tag.position = AbsolutePosition(tip, self._y(k, xt))
            tag.bg_color = self._colors[side](0.2)
            tag.text_color = self._colors[side]
            self._tag_dirty.get()[k] = True

    def _end_line(self, k: int, e: int, te: float, broke: bool) -> None:
        s = k // 2
        side = k % 2
        m = float(self._mins[s]) * 60.0
        self._cur.get()[k] = -1
        if s > 0 and self._tag_drawn.get()[k]:
            self._ltrash.get().append(self._tag.get()[k])
            tags = self._tag.get()
            tags[k] = LabelAbs('', AbsolutePosition(0.0, 0.0), bg_color=self._colors[side](0.2),
                               text_color=self._colors[side], font_size=10)
            self._tag_drawn.get()[k] = False
        self._tag_dirty.get()[k] = False
        if not self._seg_on.get()[k]:
            return
        seg = self._seg.get()[k]
        seg.point_b = AbsolutePosition(te + m - self._secs, self._y(k, float(e)))
        seg.line_width = 1
        if broke:
            seg.color = self._colors[side](0.55)
        else:
            seg.color = self._old_color(0.6)
            seg.line_style = line_style.DOTTED
        if self._keep_old:
            self._old.get().append(seg)
            self._old_drawn.get().append(self._seg_drawn.get()[k])
            self._old_dirty.get().append(True)
            self._old_slot.get().append(s)
            b = 0
            if broke:
                b = 1
            self._old_broke.get().append(b)
            n = 0
            for q in range(len(self._old_slot.get())):
                if self._old_slot.get()[q] == s:
                    n += 1
            if n > self._max_kept:                     # too many of this timeframe: oldest goes
                q = 0
                while self._old_slot.get()[q] != s:
                    q += 1
                if self._old_drawn.get()[q]:
                    self._trash.get().append(self._old.get()[q])
                self._old.get().pop(q)
                self._old_drawn.get().pop(q)
                self._old_dirty.get().pop(q)
                self._old_slot.get().pop(q)
                self._old_broke.get().pop(q)
        elif self._seg_drawn.get()[k]:
            self._trash.get().append(seg)
        self._seg_on.get()[k] = False
        self._seg_drawn.get()[k] = False
        self._seg_dirty.get()[k] = False

    # ── early breaks: a chart candle closing through a higher timeframe's live line ──
    def _check_early(self, s: int, side: int, t: float, close_t: float, a: float) -> None:
        k = s * 2 + side
        if close_t <= self._g_seen.get()[k]:
            return                                     # not known before its own candle closed
        x = self._x(s, close_t)
        if isnan(x):
            return                                     # its state is late: judged next candle
        level = self._y(k, x)
        c = self.close[0]
        up = side == 0
        if (up and c > level) or ((not up) and c < level):
            self._done.get()[k] = True
            y = self.high[0] + 0.5 * a
            if not up:
                y = self.low[0] - 0.5 * a
            word = ' early up'
            if not up:
                word = ' early down'
            col = self._colors[1]
            if not up:
                col = self._colors[0]
            self._marks.get().append(LabelAbs(self._names[s] + word, AbsolutePosition(t, y),
                                              bg_color=col(0.3), text_color=col,
                                              font_size=9 + s))
            self._mk_drawn.get().append(False)
            self._mk_dirty.get().append(True)
            self._mk_k.get().append(k)
            self._mk_t.get().append(t)
            if len(self._marks.get()) > self._marks_kept:
                if self._mk_drawn.get()[0]:
                    self._ltrash.get().append(self._marks.get()[0])
                self._marks.get().pop(0)
                self._mk_drawn.get().pop(0)
                self._mk_dirty.get().pop(0)
                self._mk_k.get().pop(0)
                self._mk_t.get().pop(0)

    def _sync(self) -> None:
        """At most 90 drawing changes per update (Exness allows 100): erasures first, then old
        lines, marks, live lines and their tags. What does not fit waits for the next candle."""
        left = 90
        while left > 0 and len(self._trash.get()) > 0:
            self.chart.erase(self._trash.get()[0])
            self._trash.get().pop(0)
            left -= 1
        while left > 0 and len(self._ltrash.get()) > 0:
            self.chart.erase(self._ltrash.get()[0])
            self._ltrash.get().pop(0)
            left -= 1
        for q in range(len(self._old.get())):
            if left > 0 and self._old_dirty.get()[q]:
                self.chart.draw(self._old.get()[q])
                self._old_dirty.get()[q] = False
                self._old_drawn.get()[q] = True
                left -= 1
        for q in range(len(self._marks.get())):
            if left > 0 and self._mk_dirty.get()[q]:
                self.chart.draw(self._marks.get()[q])
                self._mk_dirty.get()[q] = False
                self._mk_drawn.get()[q] = True
                left -= 1
        for k in range(8):
            if left > 0 and self._seg_on.get()[k] and self._seg_dirty.get()[k]:
                self.chart.draw(self._seg.get()[k])
                self._seg_dirty.get()[k] = False
                self._seg_drawn.get()[k] = True
                left -= 1
            if left > 0 and self._cur.get()[k] >= 0 and self._tag_dirty.get()[k]:
                self.chart.draw(self._tag.get()[k])
                self._tag_dirty.get()[k] = False
                self._tag_drawn.get()[k] = True
                left -= 1
