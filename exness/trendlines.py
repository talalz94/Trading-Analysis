# indie:lang_version = 5
# TRENDLINES -- the quant project's strict trendlines (quant/indicators/trendlines.py, the lines
# behind the email alerts), ported to Indie for the Exness web terminal: Indicators -> Create (the
# code editor) -> paste this file -> Save -> Add to chart. It draws the lines of the chart's own
# timeframe, on Exness's own candles.
#
# THE RULES ARE THE RESEARCH CODE'S, value for value (tests/test_exness_trendlines.py runs this
# file candle by candle against quant.indicators.add_trendlines on real candles and requires the
# same lines, on the same candles, to the last digit):
#   swings   a swing high is confirmed once price falls `Swing size` x ATR below the highest high
#            since the last swing low (and the mirror for lows). ATR = Wilder's, 14, computed here
#            exactly as pandas does, so nothing depends on the terminal's own ATR
#   a line   runs EXACTLY through two swing highs (resistance) or two swing lows (support) among
#            the last 16 swings; at least `Min touches` swings sit within `Touch tolerance` x ATR
#            of it; its touches span 40+ candles and at least `Span` of the stretch those 16
#            swings cover; and NO candle from its first touch to now closes beyond it, nor pokes a
#            wick through by more than `Wick tolerance` x ATR. Of the lines that pass, the one with
#            the most touches wins, then the widest, then the closest fit
#   live     one support and one resistance line at a time. A new line takes the place of the live
#            one only with MORE touches (or as many and a later last touch, with `Prefer newer`);
#            a refit of the same line keeps it. A line is dropped when price closes `Max distance`
#            x ATR away from it, or 500 candles after it was drawn
#   broken   when a candle CLOSES beyond it -- or, with `Hold` = N, when N more closes in a row
#            confirm it (a close back restarts the count)
# `Settings: auto` uses the email alerts' own values: on a 1m chart the 1m alerts' (touch 0.4,
# wick 0.3, max distance 20, prefer newer, hold 1, span 1/3), on any other the 5m-4h alerts'
# (touch 0.3, wick 0.15, max distance 8, hold 0, span 1/2). `custom` uses the fields below.
#
# ON THE CHART: a live line is solid and runs `Reach ahead` candles past the latest one, so you see
# where price will meet it. A broken line stops on the candle that broke it, dimmed; a line dropped
# without a break stops where it was last valid, grey and dotted.
#
# WHAT EXNESS'S RUNTIME REQUIRES (learnt on the FVG indicator, exness/fair_value_gaps.py):
#   - ALL STATE IN Var: Exness rolls the chart back and re-runs a candle whenever its value is
#     updated, even after it closed; a Var rolls back with it, a plain list does not.
#   - Lines are fitted and judged only on CLOSED candles: nothing changes while one forms.
#   - At most 100 drawing changes per update: two lines are live at a time, each redrawn only when
#     it changes or the latest candle nears its tip; every update makes at most 90 changes.
#   - TimeFrame has to_minutes, not to_seconds.
# Two limits no test here can reach, since Exness does not publish them: how long a script may run
# (fitting reads every candle from a line's first touch, so the first load of a long history is the
# heaviest moment) and how far back it may read its own candles (up to 663 back on 12 days of
# 1m Gold, BTC and USDJPY).
# If Save or Add to chart fails, the terminal's message says which.
# Candles here are Exness's, so a line can differ from the email alerts', which read Dukascopy
# (gold, FX) and Binance (BTC) -- the same rules on slightly different candles.
from math import isnan, nan
from indie import indicator, param, color, line_style, MainContext
from indie.color import rgba
from indie.drawings import LineSegment, AbsolutePosition


@indicator('Trendlines (quant)', overlay_main_pane=True)
@param.str('preset', default='auto', options=['auto', '1m alerts', '5m-4h alerts', 'custom'],
           title='Settings: auto = the email alerts own for this timeframe')
@param.float('swing_size', default=3.5, min=0.5, max=20.0, step=0.1, title='custom: Swing size (ATR)')
@param.int('min_touches', default=3, min=2, max=10, title='custom: Min touches')
@param.float('touch_tol', default=0.3, min=0.01, max=5.0, step=0.05, title='custom: Touch tolerance (ATR)')
@param.float('wick_tol', default=0.15, min=0.0, max=5.0, step=0.05, title='custom: Wick tolerance (ATR)')
@param.float('max_distance', default=8.0, min=0.5, max=100.0, step=0.5, title='custom: Max distance (ATR)')
@param.int('hold', default=0, min=0, max=10, title='custom: Hold (more closes beyond to break)')
@param.float('span', default=0.5, min=0.05, max=1.0, step=0.05, title='custom: Span (share of the swings stretch)')
@param.bool('prefer_newer', default=False, title='custom: Prefer newer line on a tie')
@param.bool('show_res', default=True, title='Resistance lines')
@param.bool('show_sup', default=True, title='Support lines')
@param.bool('keep_old', default=True, title='Keep broken and dropped lines on the chart')
@param.int('max_kept', default=100, min=1, max=1000, title='Max old lines kept')
@param.int('reach', default=20, min=2, max=500, title='Live lines reach ahead (candles)')
@param.color('res_color', default=color.RED, title='Resistance colour')
@param.color('sup_color', default=color.GREEN, title='Support colour')
@param.color('old_color', default=rgba(128, 128, 128, 1.0), title='Dropped line colour')
class Main(MainContext):
    def __init__(self, preset, swing_size, min_touches, touch_tol, wick_tol, max_distance, hold,
                 span, prefer_newer, show_res, show_sup, keep_old, max_kept, reach, res_color,
                 sup_color, old_color):
        mode = preset
        if mode == 'auto':
            if self.time_frame.to_minutes() == 1:
                mode = '1m alerts'
            else:
                mode = '5m-4h alerts'
        # the 5m-4h alerts' values (alerts/alerts_config_trendlines.py, STRICT) ...
        self._swing = 3.5
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
        # fixed, as in the research code: the last 16 swings, touches 40+ candles apart, a line
        # retired 500 candles after it was drawn, a 14-candle ATR
        self._lookback = 16
        self._min_span_bars = 40
        self._max_age = 500
        self._atr_len = 14
        a0 = 1.0 / float(self._atr_len)               # pandas' ewm(alpha=1/14, adjust=False):
        com = (1.0 - a0) / a0                         # alpha -> centre of mass -> alpha, as it
        self._alpha = 1.0 / (1.0 + com)               # does, so every ATR is bit-identical
        self._owf = 1.0 - self._alpha
        self._show = [show_res, show_sup]             # index 0 = resistance (+1), 1 = support (-1)
        self._keep_old = keep_old
        self._max_kept = max_kept
        self._reach = reach
        self._colors = [res_color, sup_color]
        self._old_color = old_color
        self._bar_secs = float(self.time_frame.to_minutes()) * 60.0
        # ── state: ALL in Var (see the top of the file) ──
        self._atr_w = self.new_var(0.0)
        self._atr_n = self.new_var(0)
        self._prev_close = self.new_var(0.0)
        self._zz_dir = self.new_var(0)               # +1 seeking a high, -1 a low, 0 not started
        self._zz_px = self.new_var(0.0)
        self._zz_i = self.new_var(-1)
        self._zz_hi = self.new_var(nan)              # the last swing high / low ever seen
        self._zz_lo = self.new_var(nan)
        pv_i: list[int] = []
        pv_px: list[float] = []
        pv_kind: list[int] = []
        self._pv_i = self.new_var(pv_i)              # the last 16 swings: extreme's candle,
        self._pv_px = self.new_var(pv_px)            # price, +1 high / -1 low
        self._pv_kind = self.new_var(pv_kind)
        rs: list[str] = []
        rf: list[int] = []
        rk: list[int] = []
        self._ret_sig = self.new_var(rs)             # lines that broke or were dropped: never back
        self._ret_first = self.new_var(rf)
        self._ret_kind = self.new_var(rk)
        # the live line of each side
        self._on = self.new_var([False, False])
        self._slope = self.new_var([0.0, 0.0])
        self._icpt = self.new_var([0.0, 0.0])
        self._touches = self.new_var([0, 0])
        self._start = self.new_var([0, 0])
        self._born = self.new_var([0, 0])
        self._end = self.new_var([-1, -1])
        self._beyond = self.new_var([0, 0])
        self._last = self.new_var([0, 0])            # its latest touch
        self._sig = self.new_var(['', ''])
        self._sfirst = self.new_var([0, 0])          # its signature's first touch
        self._level = self.new_var([nan, nan])       # the line's price on the latest candle
        self._brk = self.new_var([False, False])     # it broke on the latest candle
        # its drawing
        blank = LineSegment(AbsolutePosition(0.0, 0.0), AbsolutePosition(0.0, 0.0))
        self._seg = self.new_var([blank, blank])
        self._seg_on = self.new_var([False, False])  # this side's live segment exists
        self._seg_drawn = self.new_var([False, False])
        self._seg_dirty = self.new_var([False, False])
        self._seg_tip = self.new_var([0, 0])         # the candle its far end reaches
        old: list[LineSegment] = []
        old_drawn: list[bool] = []
        old_dirty: list[bool] = []
        self._old = self.new_var(old)
        self._old_drawn = self.new_var(old_drawn)
        self._old_dirty = self.new_var(old_dirty)
        trash: list[LineSegment] = []
        self._trash = self.new_var(trash)
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

    def calc(self):
        if not self.is_closed_bar:
            return                                    # the live candle: nothing until it closes
        self._remember()
        i = self.bar_index
        a = self._atr()
        confirmed = self._zigzag(i, a)
        if confirmed and not isnan(a) and a > 0:
            self._fit_side(i, a, 1)
            self._fit_side(i, a, -1)
        self._publish(i, a)
        self._sync()

    # ── ATR: Wilder's, as pandas computes ewm(alpha=1/14, adjust=False, min_periods=14) ──
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

    # ── swings: the ATR-scaled zigzag (quant.indicators.structure.add_zigzag_structure) ──
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

    # ── the line fit (quant.indicators.chart_patterns._best_line, strict) ──
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

    # ── which line holds the slot (quant.indicators.trendlines.add_trendlines) ──
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
                if self._seg_on.get()[s]:
                    self._place(s, self._seg_tip.get()[s])    # its whole span moves with it
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
                    self._draw_live(s, i)
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

    # ── the drawing ──
    def _time_of(self, b: int) -> float:
        i = self.bar_index
        if b <= i:
            return self._t_ago(i - b)
        return self.time[0] + float(b - i) * self._bar_secs

    def _place(self, s: int, tip: int) -> None:
        """Lay this side's live segment from its first touch to candle `tip`."""
        seg = self._seg.get()[s]
        x0 = self._start.get()[s]
        seg.point_a = AbsolutePosition(self._time_of(x0),
                                       self._slope.get()[s] * float(x0) + self._icpt.get()[s])
        seg.point_b = AbsolutePosition(self._time_of(tip),
                                       self._slope.get()[s] * float(tip) + self._icpt.get()[s])
        self._seg_tip.get()[s] = tip
        self._seg_dirty.get()[s] = True

    def _draw_live(self, s: int, i: int) -> None:
        if not self._show[s]:
            return
        if not self._seg_on.get()[s]:                  # the line's first valid candle
            self._seg.get()[s] = LineSegment(AbsolutePosition(0.0, 0.0),
                                             AbsolutePosition(0.0, 0.0),
                                             color=self._colors[s], line_style=line_style.SOLID,
                                             line_width=2)
            self._seg_on.get()[s] = True
            self._seg_drawn.get()[s] = False
            self._place(s, i + self._reach)
        elif self._seg_tip.get()[s] < i + 2:           # the latest candle nears its tip
            self._place(s, i + self._reach)

    def _finish(self, s: int, broke: bool) -> None:
        """This side's line stops being live: end its drawing where it was last valid."""
        self._on.get()[s] = False
        if not self._seg_on.get()[s]:
            return
        seg = self._seg.get()[s]
        end = self._end.get()[s]
        seg.point_b = AbsolutePosition(self._time_of(end),
                                       self._slope.get()[s] * float(end) + self._icpt.get()[s])
        if broke:
            seg.color = self._colors[s](0.55)
            seg.line_width = 1
        else:
            seg.color = self._old_color(0.6)
            seg.line_width = 1
            seg.line_style = line_style.DOTTED
        if self._keep_old:
            self._old.get().append(seg)
            self._old_drawn.get().append(self._seg_drawn.get()[s])
            self._old_dirty.get().append(True)
            if len(self._old.get()) > self._max_kept:  # too many kept: the OLDEST goes
                if self._old_drawn.get()[0]:
                    self._trash.get().append(self._old.get()[0])
                self._old.get().pop(0)
                self._old_drawn.get().pop(0)
                self._old_dirty.get().pop(0)
        elif self._seg_drawn.get()[s]:
            self._trash.get().append(seg)
        self._seg_on.get()[s] = False
        self._seg_drawn.get()[s] = False
        self._seg_dirty.get()[s] = False

    def _sync(self) -> None:
        """Bring the chart in line with the state, at most 90 changes (Exness allows 100):
        erasures first, then the old lines, then the live ones. What does not fit waits."""
        left = 90
        while left > 0 and len(self._trash.get()) > 0:
            self.chart.erase(self._trash.get()[0])
            self._trash.get().pop(0)
            left -= 1
        for j in range(len(self._old.get())):
            if left > 0 and self._old_dirty.get()[j]:
                self.chart.draw(self._old.get()[j])
                self._old_dirty.get()[j] = False
                self._old_drawn.get()[j] = True
                left -= 1
        for s in range(2):
            if left > 0 and self._seg_on.get()[s] and self._seg_dirty.get()[s]:
                self.chart.draw(self._seg.get()[s])
                self._seg_dirty.get()[s] = False
                self._seg_drawn.get()[s] = True
                left -= 1
