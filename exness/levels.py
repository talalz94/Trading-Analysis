# indie:lang_version = 5
# LEVELS -- the previous day / week / month high and low, and each market session's high and low,
# as the research charts draw them (quant.indicators.levels: period_levels, session_windows), for
# the Exness web terminal: Indicators -> Create -> paste -> Save -> Add to chart. Any timeframe.
#
# PREVIOUS DAY / WEEK / MONTH: from the first moment of a day, its previous day's high and low,
# drawn across the whole day (week: Monday on; month: the 1st on). The previous period is the last
# one WITH CANDLES, so on a market shut at weekends Monday's previous day is Friday. A period the
# chart holds only part of (its history starts mid-way) is never used: its high would be the high
# of the part loaded. `Day starts` picks the clock: UTC midnight (as Exness's daily candles),
# New York midnight, or 17:00 New York (the FX day, so Sunday evening belongs to Monday).
#
# SESSIONS (charts of 1h and less): Sydney 07-16, Tokyo 09-18, London 08-17 and New York 08-17,
# each in ITS OWN clock with its own daylight saving (London opens at 08:00 London time all year:
# 07:00 or 08:00 UTC), on its own Monday to Friday. A session's high and low are drawn over the session (solid) and then
# carried on (dotted) until the same session opens again, so Tokyo's range is still there in
# London. While a session is open its lines show the high and low SO FAR.
#
# Indie has no time-zone library: the daylight-saving rules are written out below (US: 2nd Sunday
# of March to the 1st of November; UK/EU: last Sunday of March to the last of October; Sydney:
# 1st Sunday of October to the 1st of April) and tests/test_exness_levels.py holds them to Python's
# zoneinfo hour by hour over years.
#
# All state is in Var and drawings change only on closed candles (Exness rolls a candle back and
# re-runs it); every update makes at most 90 drawing changes (Exness refuses more than 100).
from math import isnan, nan
from indie import indicator, param, color, line_style, MainContext
from indie.color import rgba
from indie.drawings import LineSegment, LabelAbs, AbsolutePosition


@indicator('Levels (quant)', overlay_main_pane=True)
@param.str('day_start', default='UTC midnight', options=['UTC midnight', 'New York midnight', '17:00 New York (FX day)'], title='Day starts')
@param.bool('show_day', default=True, title='Previous day high / low')
@param.bool('show_week', default=True, title='Previous week high / low')
@param.bool('show_month', default=False, title='Previous month high / low')
@param.int('days_kept', default=10, min=1, max=200, title='Past days kept')
@param.bool('show_sydney', default=False, title='Sydney session')
@param.bool('show_tokyo', default=True, title='Tokyo session')
@param.bool('show_london', default=True, title='London session')
@param.bool('show_new_york', default=True, title='New York session')
@param.bool('extend', default=True, title='Carry a session range on until it opens again')
@param.int('sessions_kept', default=5, min=1, max=100, title='Past sessions kept (each)')
@param.color('day_color', default=rgba(66, 133, 244, 1.0), title='Day colour')
@param.color('week_color', default=rgba(171, 71, 188, 1.0), title='Week colour')
@param.color('month_color', default=rgba(0, 150, 136, 1.0), title='Month colour')
@param.color('sydney_color', default=rgba(158, 158, 158, 1.0), title='Sydney colour')
@param.color('tokyo_color', default=rgba(233, 30, 99, 1.0), title='Tokyo colour')
@param.color('london_color', default=rgba(255, 152, 0, 1.0), title='London colour')
@param.color('new_york_color', default=rgba(0, 188, 212, 1.0), title='New York colour')
class Main(MainContext):
    def __init__(self, day_start, show_day, show_week, show_month, days_kept, show_sydney,
                 show_tokyo, show_london, show_new_york, extend, sessions_kept, day_color,
                 week_color, month_color, sydney_color, tokyo_color, london_color,
                 new_york_color):
        cm = self.time_frame.to_minutes()
        self._secs = float(cm) * 60.0
        # the day's clock: zone (0 UTC, 1 New York) and a shift (+7 h: 17:00 New York -> midnight)
        self._dzone = 0
        self._dshift = 0.0
        if day_start == 'New York midnight':
            self._dzone = 1
        if day_start == '17:00 New York (FX day)':
            self._dzone = 1
            self._dshift = 7.0 * 3600.0
        # a period's levels on charts finer than the period only (on a daily chart the previous
        # day is just the candle before)
        self._pshow = [show_day and cm < 1440, show_week and cm < 10080,
                       show_month and cm < 43200]
        self._pkept = [days_kept, max(2, days_kept // 5), 3]
        # colours by index: 0-2 day / week / month, 3-6 the sessions
        self._cols = [day_color, week_color, month_color, sydney_color, tokyo_color, london_color,
                      new_york_color]
        self._pname = ['D', 'W', 'M']
        # sessions: zone, local open / close hour, colour. Zones: 1 New York, 2 London, 3 Tokyo,
        # 4 Sydney
        self._szone = [4, 3, 2, 1]
        self._sopen = [7.0, 9.0, 8.0, 8.0]
        self._sclose = [16.0, 18.0, 17.0, 17.0]
        self._sname = ['Sydney', 'Tokyo', 'London', 'New York']
        # sessions on charts whose candles fit inside the hour only: a 4h candle straddles a
        # session's open and close, so its high could be from outside the session
        fits = cm <= 60 and 60 % cm == 0
        self._sshow = [show_sydney and fits, show_tokyo and fits, show_london and fits,
                       show_new_york and fits]
        self._extend = extend
        self._skept = sessions_kept
        # period state: the current period's key, its high / low, whether the chart holds all of
        # it, and the levels in force (from the previous period with candles)
        self._pkey = self.new_var([-1, -1, -1])
        self._phi = self.new_var([nan, nan, nan])
        self._plo = self.new_var([nan, nan, nan])
        self._pwhole = self.new_var([False, False, False])
        self._first = self.new_var(True)
        # session state: the local day (days since 1970) of the instance running, its window,
        # its high / low so far
        self._sday = self.new_var([-1, -1, -1, -1])
        self._sopen_t = self.new_var([0.0, 0.0, 0.0, 0.0])
        self._sclose_t = self.new_var([0.0, 0.0, 0.0, 0.0])
        self._shi = self.new_var([nan, nan, nan, nan])
        self._slo = self.new_var([nan, nan, nan, nan])
        # drawings: every one lives in a Var; `_dirty` = redraw on the next sync
        segs: list[LineSegment] = []
        f1: list[bool] = []
        f2: list[bool] = []
        o1: list[int] = []
        self._segs = self.new_var(segs)              # every segment kept, oldest first
        self._seg_drawn = self.new_var(f1)
        self._seg_dirty = self.new_var(f2)
        self._seg_owner = self.new_var(o1)           # 0-2 period D/W/M, 10-13 session
        labs: list[LabelAbs] = []
        f3: list[bool] = []
        f4: list[bool] = []
        o2: list[int] = []
        self._labs = self.new_var(labs)              # one label per level in force
        self._lab_drawn = self.new_var(f3)
        self._lab_dirty = self.new_var(f4)
        self._lab_owner = self.new_var(o2)
        trash: list[LineSegment] = []
        self._trash = self.new_var(trash)
        ltrash: list[LabelAbs] = []
        self._ltrash = self.new_var(ltrash)
        # the live session's two segments + its two carried-on ones, by session: their index in
        # _segs (-1 = none)
        self._scur = self.new_var([-1, -1, -1, -1])

    def calc(self):
        if not self.is_closed_bar:
            return
        t = self.time[0]
        for p in range(3):
            if self._pshow[p]:
                self._period(p, t)
        self._first.set(False)
        for s in range(4):
            if self._sshow[s]:
                self._session(s, t)
        self._sync()

    # ── calendar arithmetic (Howard Hinnant's civil-from-days; days since 1970-01-01) ──
    def _days(self, y: int, m: int, d: int) -> int:
        yy = y
        if m <= 2:
            yy = y - 1
        era = yy // 400
        yoe = yy - era * 400
        mp = m + 9
        if m > 2:
            mp = m - 3
        doy = (153 * mp + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return era * 146097 + doe - 719468

    def _civil(self, z: int) -> list[int]:
        zz = z + 719468
        era = zz // 146097
        doe = zz - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + 3
        if mp >= 10:
            m = mp - 9
        y = yoe + era * 400
        if m <= 2:
            y += 1
        return [y, m, d]

    def _weekday(self, z: int) -> int:
        return (z + 3) % 7                           # 0 = Monday; 1970-01-01 was a Thursday

    def _sunday(self, y: int, m: int, n: int) -> int:
        """The n-th Sunday of a month (n = -1: the last), as days since 1970."""
        if n < 0:
            nm = m + 1
            ny = y
            if nm > 12:
                nm = 1
                ny = y + 1
            last = self._days(ny, nm, 1) - 1
            return last - (self._weekday(last) + 1) % 7
        first = self._days(y, m, 1)
        return first + (6 - self._weekday(first)) % 7 + 7 * (n - 1)

    def _dst(self, zone: int, t: float) -> bool:
        """Daylight saving in force at UTC second `t`. 1 New York, 2 London, 4 Sydney."""
        y = self._civil(int(t / 86400.0))[0]
        if zone == 1:                                # 2nd Sunday of March 07:00 UTC to the
            a = float(self._sunday(y, 3, 2)) * 86400.0 + 7.0 * 3600.0     # 1st of Nov 06:00
            b = float(self._sunday(y, 11, 1)) * 86400.0 + 6.0 * 3600.0
            return a <= t and t < b
        if zone == 2:                                # last Sunday of March to that of October,
            a = float(self._sunday(y, 3, -1)) * 86400.0 + 3600.0          # 01:00 UTC both
            b = float(self._sunday(y, 10, -1)) * 86400.0 + 3600.0
            return a <= t and t < b
        if zone == 4:                                # from the 1st Sunday of October to that of
            a = float(self._sunday(y, 4, 1)) * 86400.0 - 8.0 * 3600.0     # April (Saturday
            b = float(self._sunday(y, 10, 1)) * 86400.0 - 8.0 * 3600.0    # 16:00 UTC both)
            return t < a or b <= t
        return False

    def _std(self, zone: int) -> float:
        if zone == 1:
            return -5.0 * 3600.0
        if zone == 3:
            return 9.0 * 3600.0
        if zone == 4:
            return 10.0 * 3600.0
        return 0.0

    def _offset(self, zone: int, t: float) -> float:
        """The zone's clock minus UTC at UTC second t."""
        off = self._std(zone)
        if self._dst(zone, t):
            off += 3600.0
        return off

    def _utc(self, zone: int, local: float) -> float:
        """A wall-clock time in `zone` (seconds since 1970 as if it were UTC) -> UTC. Exact away
        from the small hours of a clock change, where no session or period here starts."""
        return local - self._offset(zone, local - self._std(zone))

    # ── previous day / week / month ──
    def _key(self, p: int, local: float) -> int:
        z = int(local / 86400.0)
        if p == 0:
            return z
        if p == 1:
            return z - self._weekday(z)                # its Monday
        c = self._civil(z)
        return c[0] * 12 + c[1] - 1

    def _start(self, p: int, key: int) -> float:
        """A period's first moment in UTC."""
        if p == 2:
            z = self._days(key // 12, key % 12 + 1, 1)
        else:
            z = key
        return self._utc(self._dzone, float(z) * 86400.0) - self._dshift

    def _end(self, p: int, key: int) -> float:
        if p == 0:
            return self._start(p, key + 1)
        if p == 1:
            return self._start(p, key + 7)
        return self._start(p, key + 1)

    def _period(self, p: int, t: float) -> None:
        local = t + self._dshift
        local = local + self._offset(self._dzone, local)
        key = self._key(p, local)
        cur = self._pkey.get()[p]
        if key == cur:
            self._phi.get()[p] = max(self._phi.get()[p], self.high[0])
            self._plo.get()[p] = min(self._plo.get()[p], self.low[0])
            return
        if cur >= 0 and self._pwhole.get()[p]:         # the period just finished is the source
            self._draw_period(p, key, self._phi.get()[p], self._plo.get()[p])
        self._pkey.get()[p] = key
        self._phi.get()[p] = self.high[0]
        self._plo.get()[p] = self.low[0]
        whole = True
        if self._first.get():                          # the chart starts within one candle of it?
            whole = t - self._start(p, key) <= self._secs
        self._pwhole.get()[p] = whole

    def _draw_period(self, p: int, key: int, hi: float, lo: float) -> None:
        a = self._start(p, key)
        b = self._end(p, key)
        self._add_seg(LineSegment(AbsolutePosition(a, hi), AbsolutePosition(b, hi),
                                  color=self._cols[p], line_style=line_style.DASHED,
                                  line_width=1), p)
        self._add_seg(LineSegment(AbsolutePosition(a, lo), AbsolutePosition(b, lo),
                                  color=self._cols[p], line_style=line_style.DASHED,
                                  line_width=1), p)
        self._trim(p, self._pkept[p] * 2)
        name = self._pname[p]
        self._set_label(p * 2, 'P' + name + 'H', b, hi, p)
        self._set_label(p * 2 + 1, 'P' + name + 'L', b, lo, p)

    # ── sessions ──
    def _window(self, s: int, day: int) -> list[float]:
        zone = self._szone[s]
        o = self._utc(zone, float(day) * 86400.0 + self._sopen[s] * 3600.0)
        cday = day
        if self._sclose[s] <= self._sopen[s]:
            cday = day + 1                             # runs past local midnight
        c = self._utc(zone, float(cday) * 86400.0 + self._sclose[s] * 3600.0)
        return [o, c]

    def _instance(self, s: int, t: float) -> int:
        """The local day whose session holds UTC second t, or -1."""
        zone = self._szone[s]
        ld = int((t + self._offset(zone, t)) / 86400.0)
        for back in range(2):
            d = ld - back
            if self._weekday(d) < 5:
                w = self._window(s, d)
                if w[0] <= t and t < w[1]:
                    return d
        return -1

    def _next_open(self, s: int, day: int) -> float:
        d = day + 1
        while self._weekday(d) >= 5:
            d += 1
        return self._window(s, d)[0]

    def _session(self, s: int, t: float) -> None:
        d = self._instance(s, t)
        cur = self._sday.get()[s]
        if d >= 0 and d == cur:
            hi = max(self._shi.get()[s], self.high[0])
            lo = min(self._slo.get()[s], self.low[0])
            if hi != self._shi.get()[s] or lo != self._slo.get()[s]:
                self._shi.get()[s] = hi
                self._slo.get()[s] = lo
                self._place_session(s)
            return
        if d < 0:
            return
        # a new session opens: the last one stays as it is, its carry-on ends at this open
        self._sday.get()[s] = d
        w = self._window(s, d)
        self._sopen_t.get()[s] = w[0]
        self._sclose_t.get()[s] = w[1]
        self._shi.get()[s] = self.high[0]
        self._slo.get()[s] = self.low[0]
        for k in range(2):
            self._add_seg(LineSegment(AbsolutePosition(0.0, 0.0), AbsolutePosition(0.0, 0.0),
                                      color=self._cols[3 + s], line_style=line_style.SOLID,
                                      line_width=1), 10 + s)
        if self._extend:
            for k in range(2):
                self._add_seg(LineSegment(AbsolutePosition(0.0, 0.0),
                                          AbsolutePosition(0.0, 0.0),
                                          color=self._cols[3 + s](0.7),
                                          line_style=line_style.DOTTED, line_width=1), 10 + s)
        per = 2
        if self._extend:
            per = 4
        self._trim(10 + s, self._skept * per)
        self._scur.get()[s] = len(self._segs.get()) - per
        self._place_session(s)

    def _place_session(self, s: int) -> None:
        i = self._scur.get()[s]
        if i < 0:
            return
        o = self._sopen_t.get()[s]
        c = self._sclose_t.get()[s]
        hi = self._shi.get()[s]
        lo = self._slo.get()[s]
        self._move(i, o, c, hi)
        self._move(i + 1, o, c, lo)
        if self._extend:
            nxt = self._next_open(s, self._sday.get()[s])
            self._move(i + 2, c, nxt, hi)
            self._move(i + 3, c, nxt, lo)
        name = self._sname[s]
        end = c
        if self._extend:
            end = self._next_open(s, self._sday.get()[s])
        self._set_label(10 + s * 2, name + ' H', end, hi, 3 + s)
        self._set_label(10 + s * 2 + 1, name + ' L', end, lo, 3 + s)

    # ── drawing bookkeeping ──
    def _add_seg(self, seg: LineSegment, owner: int) -> None:
        self._segs.get().append(seg)
        self._seg_drawn.get().append(False)
        self._seg_dirty.get().append(True)
        self._seg_owner.get().append(owner)

    def _move(self, i: int, a: float, b: float, y: float) -> None:
        seg = self._segs.get()[i]
        seg.point_a = AbsolutePosition(a, y)
        seg.point_b = AbsolutePosition(b, y)
        self._seg_dirty.get()[i] = True

    def _trim(self, owner: int, keep: int) -> None:
        """Keep the newest `keep` segments of this owner; the oldest go."""
        n = 0
        for i in range(len(self._segs.get())):
            if self._seg_owner.get()[i] == owner:
                n += 1
        i = 0
        while n > keep and i < len(self._segs.get()):
            if self._seg_owner.get()[i] == owner:
                if self._seg_drawn.get()[i]:
                    self._trash.get().append(self._segs.get()[i])
                self._segs.get().pop(i)
                self._seg_drawn.get().pop(i)
                self._seg_dirty.get().pop(i)
                self._seg_owner.get().pop(i)
                for q in range(4):
                    if self._scur.get()[q] > i:
                        self._scur.get()[q] = self._scur.get()[q] - 1
                n -= 1
            else:
                i += 1

    def _set_label(self, owner: int, text: str, t: float, y: float, ci: int) -> None:
        """One label per level in force, at its right end; moved, not recreated."""
        for i in range(len(self._labs.get())):
            if self._lab_owner.get()[i] == owner:
                lab = self._labs.get()[i]
                lab.position = AbsolutePosition(t, y)
                self._lab_dirty.get()[i] = True
                return
        self._labs.get().append(LabelAbs(text, AbsolutePosition(t, y),
                                         bg_color=self._cols[ci](0.25),
                                         text_color=self._cols[ci], font_size=10))
        self._lab_drawn.get().append(False)
        self._lab_dirty.get().append(True)
        self._lab_owner.get().append(owner)

    def _sync(self) -> None:
        """At most 90 drawing changes (Exness allows 100): erasures first, then labels, then
        segments, oldest first. What does not fit waits for the next candle."""
        left = 90
        while left > 0 and len(self._trash.get()) > 0:
            self.chart.erase(self._trash.get()[0])
            self._trash.get().pop(0)
            left -= 1
        for i in range(len(self._labs.get())):
            if left > 0 and self._lab_dirty.get()[i]:
                self.chart.draw(self._labs.get()[i])
                self._lab_dirty.get()[i] = False
                self._lab_drawn.get()[i] = True
                left -= 1
        for i in range(len(self._segs.get())):
            if left > 0 and self._seg_dirty.get()[i]:
                self.chart.draw(self._segs.get()[i])
                self._seg_dirty.get()[i] = False
                self._seg_drawn.get()[i] = True
                left -= 1
