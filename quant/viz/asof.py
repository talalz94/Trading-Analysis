"""
REWIND — put the chart back to what was knowable at one candle, by clicking it.

Every artefact this project draws already carries its own birth and death: a gap has `born_i` and
`dead_i`, an S/R revision a `start_i`/`end_i`, a trendline a `born_i`/`end_i`, a trade an entry and
an exit. So "as of bar k" is a FILTER over geometry that is already on the chart, not a
re-detection — which is why this costs one small script and no extra payload.

THE WHOLE TOOL IS ONE CLIP. Every rewindable trace is a sequence of `None`-separated segments, so
one generic operation does the entire job:

    drop any segment that starts after k, and clip the rest at x = k,
    interpolating y where a segment crosses it

Read what that gives you, shape by shape, and note that nothing is special-cased:

    gap box      [x0,x1,x1,x0,x0] -> CLAMPED, so it stays a box of the gap's real height and
                 only gets shorter. A gap MITIGATED AFTER k therefore still looks open, because at
                 k it was. That is the whole point of the tool. (Truncating it like a polyline
                 kept only the bottom edge and every gap rewound into a flat line.)
    trendline    two points -> clipped with y interpolated ALONG the line, so the slope stays
                 honest. A break that happened after k is simply not drawn.
    trade        entry/stop/target run from entry to exit -> clipped at k; a trade entered after k
                 disappears; one still open at k is drawn only as far as k.
    markers      single points -> dropped when they sit after k.

ONE INACCURACY IS HANDLED AT DRAW TIME, NOT HERE. `sr_zone_table` emits a row per price revision,
and the normal chart draws only the LAST one — today's level, painted across the whole window. For
studying history that is a quiet lookahead, so `pattern_figure(rewind=True)` draws every revision as
its own segment. The clip above then shows the revision that was live at k, with no extra logic.

WHAT IT READS. The trace SPEC, `gd.data` -- and it decodes plotly.py's base64 columns itself.
plotly only expands `_fullData` and only attaches `_inputArray` for traces it has DRAWN, so anything
built `visible:false` (every isolatable gap, every per-trade group) has no readable geometry in
either place. Snapshotting from `_fullData` captured empty arrays for exactly the traces the pickers
exist to show, and `live` then restored that emptiness permanently.

WHERE IT RUNS. In the browser tab, via `open_in_browser(fig, extra_js=AS_OF_JS)` — the same hook
`CROSSHAIR_JS` uses. It needs no kernel and no round trip. It does NOT work in an inline notebook
cell, where the script is never injected; for that, set `AS_OF` in the config and let the figure be
rebuilt on truncated data, which is exact and reproducible but costs a re-run.
"""
from __future__ import annotations

__all__ = ["AS_OF_JS"]


AS_OF_JS = r"""
(function () {
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd) return;

  function rewindable() {
    var m = (gd.layout && gd.layout.meta) || {};
    return m.rewind || [];
  }
  if (!rewindable().length) return;          // figure was not built with rewind=True

  // the untouched geometry, kept so "live" is an exact restore rather than a redraw
  var ORIG = null, armed = false, atX = null;

  // plotly.py 6 sends numeric columns base64-encoded ({dtype, bdata}), so gd.data[i].y is not
  // always an array -- three trade-marker traces arrived that way and .slice() threw.
  //
  // DECODE IT HERE rather than leaning on plotly's decoded copy. plotly only decodes a column when
  // it DRAWS the trace: `_inputArray` is attached during supply-defaults, which returns early for
  // anything `visible:false`. So for every isolatable gap and every per-trade group -- exactly the
  // traces the pickers exist to show -- neither _fullData[i].y nor gd.data[i]._inputArray exists,
  // and the only copy of the numbers in the page is the base64 string itself.
  var DTYPE = { f8: Float64Array, f4: Float32Array, i4: Int32Array, u4: Uint32Array,
                i2: Int16Array, u2: Uint16Array, i1: Int8Array, u1: Uint8Array };

  function fromB64(o) {
    var C = DTYPE[o.dtype];
    if (!C || typeof o.bdata !== 'string') { return null; }
    try {
      var bin = atob(o.bdata), n = bin.length, buf = new Uint8Array(n), i;
      if (n % C.BYTES_PER_ELEMENT) { return null; }
      for (i = 0; i < n; i++) { buf[i] = bin.charCodeAt(i); }
      return Array.from(new C(buf.buffer, 0, n / C.BYTES_PER_ELEMENT));
    } catch (e) { return null; }
  }

  function arr(v) {
    if (v === null || v === undefined) { return []; }
    if (Array.isArray(v)) { return v.slice(); }
    if (ArrayBuffer.isView(v)) { return Array.from(v); }
    if (typeof v === 'object') {
      if (v._inputArray) { return Array.from(v._inputArray); }
      if (v.bdata) { return fromB64(v) || []; }
    }
    return [];
  }

  // gd.data IS THE TRUTH; _fullData is only a fallback.
  //
  // plotly expands _fullData for traces it has actually DRAWN. Everything built `visible:false` --
  // every isolatable gap, every per-trade entry/stop/target group, every trendline behind the
  // picker -- has no x, no y and no fill there. Snapshotting from _fullData therefore captured
  // EMPTY arrays for precisely the traces the pickers exist to show, and because `live` restores
  // from the snapshot it wrote that emptiness straight back: after one rewind, picking a trade drew
  // nothing at all, for the rest of the session, and pressing live could not bring it back.
  function pick(spec, full, key) {
    var a = arr(spec && spec[key]);
    return a.length ? a : arr(full && full[key]);
  }

  function snapshot() {
    if (ORIG) { return; }
    ORIG = {};
    rewindable().forEach(function (i) {
      var full = (gd._fullData && gd._fullData[i]) || null;
      var spec = gd.data[i] || full;
      if (!spec) { return; }
      var x = pick(spec, full, 'x');
      // A trace we could not read is a trace we must not write: one unreadable column would
      // otherwise be blanked by the first clip and stay blank forever.
      if (!x.length) { return; }
      ORIG[i] = { x: x, y: pick(spec, full, 'y'),
                  cd: (spec.customdata || (full && full.customdata))
                      ? pick(spec, full, 'customdata') : null,
                  box: spec.fill === 'toself' || (full && full.fill === 'toself') };
    });
  }

  var ms = function (v) { return (v instanceof Date) ? v.getTime() : new Date(v).getTime(); };

  // TWO CLIPS, because a closed polygon is not a polyline.
  //
  //   a LINE is TRUNCATED: walk to the cut and interpolate y onto it, then stop. Moving x without
  //   recomputing y would tilt a trendline, which is worse than not clipping at all.
  //
  //   a BOX is CLAMPED: a gap is the polygon [x0,x1,x1,x0,x0] x [bot,bot,top,top,bot], and
  //   truncating it keeps only the first edge -- the bottom -- so every rewound gap collapsed into
  //   a horizontal line. Pulling each vertex back to the cut instead keeps it a box of the gap's
  //   real height, only shorter. Its height never depends on where you click; only its length.
  function clipBox(xs, ys, cd, cut) {
    var ox = [], oy = [], ocd = [], seg = [], i;
    function flush() {
      if (seg.length) {
        var lo = Infinity, j;
        for (j = 0; j < seg.length; j++) { if (seg[j].t < lo) { lo = seg[j].t; } }
        if (lo <= cut) {                       // born at or before the cut: keep it, clamped
          for (j = 0; j < seg.length; j++) {
            var p = seg[j];
            ox.push(p.t > cut ? new Date(cut) : p.x); oy.push(p.y);
            if (cd) { ocd.push(p.c); }
          }
          ox.push(null); oy.push(null); if (cd) { ocd.push(null); }
        }
      }
      seg = [];
    }
    for (i = 0; i < xs.length; i++) {
      if (xs[i] === null || xs[i] === undefined) { flush(); continue; }
      seg.push({ t: ms(xs[i]), x: xs[i], y: ys[i], c: cd ? cd[i] : null });
    }
    flush();
    while (ox.length && ox[ox.length - 1] === null) { ox.pop(); oy.pop(); if (cd) { ocd.pop(); } }
    return { x: ox, y: oy, cd: cd ? ocd : null };
  }

  function clip(xs, ys, cd, cut) {
    var ox = [], oy = [], ocd = [], seg = [], i;
    function flush() {
      if (!seg.length) { return; }
      var keep = [], j;
      for (j = 0; j < seg.length; j++) {
        var p = seg[j];
        if (p.t <= cut) { keep.push(p); continue; }
        var q = seg[j - 1];
        if (q && q.t <= cut) {                       // crosses the cut: interpolate onto it
          var f = (p.t === q.t) ? 0 : (cut - q.t) / (p.t - q.t);
          keep.push({ t: cut, x: new Date(cut), y: q.y + f * (p.y - q.y), c: q.c });
        }
        break;                                       // everything past the cut is future
      }
      if (keep.length) {
        for (j = 0; j < keep.length; j++) {
          ox.push(keep[j].x); oy.push(keep[j].y); if (cd) ocd.push(keep[j].c);
        }
        ox.push(null); oy.push(null); if (cd) ocd.push(null);
      }
      seg = [];
    }
    for (i = 0; i < xs.length; i++) {
      if (xs[i] === null || xs[i] === undefined) { flush(); continue; }
      seg.push({ t: ms(xs[i]), x: xs[i], y: ys[i], c: cd ? cd[i] : null });
    }
    flush();
    while (ox.length && ox[ox.length - 1] === null) { ox.pop(); oy.pop(); if (cd) ocd.pop(); }
    return { x: ox, y: oy, cd: cd ? ocd : null };
  }

  // How many trades exist at the cut. The trade picker and the rewind are two controls over the
  // same traces, and when a trade has not happened yet its row draws nothing -- which reads as a
  // broken dropdown rather than as "that trade is in your future". Counting them is the cheapest
  // honest answer.
  function tradeCount() {
    var total = 0, live = 0;
    rewindable().forEach(function (i) {
      var t = gd.data[i];
      // the ENTRY markers only. Counting every marker in the `trades` group counts exits too and
      // reports "16 trades" for eight.
      if (!t || !/^(long|short) entries/.test(String(t.name || ''))) { return; }
      // BOTH counts come off gd.data: the snapshot for the total, the CURRENT (already-clipped)
      // spec for the live one. Reading `live` from _fullData reported 0 of 8 whenever a single
      // trade was isolated, because the group trace it counts is hidden then.
      var keep = function (v) { return v !== null && v !== undefined && v === v; };
      total += ((ORIG && ORIG[i] ? ORIG[i].x : arr(t.x)) || []).filter(keep).length;
      live += arr(t.x).filter(keep).length;
    });
    return { total: total, live: live };
  }

  function apply(cut, rawX) {
    snapshot();
    var idx = rewindable().filter(function (i) { return !!ORIG[i]; });
    var X = [], Y = [], C = [], any = false;
    idx.forEach(function (i) {
      var o = ORIG[i];
      var r = (cut === null) ? { x: o.x, y: o.y, cd: o.cd }
            : (o.box ? clipBox(o.x, o.y, o.cd, cut) : clip(o.x, o.y, o.cd, cut));
      X.push(r.x); Y.push(r.y); C.push(r.cd); if (r.cd) any = true;
    });
    var up = { x: X, y: Y };
    if (any) { up.customdata = C; }
    Plotly.restyle(gd, up, idx);
    var shapes = (gd.layout.shapes || []).filter(function (s) { return s.name !== 'asof'; });
    if (cut !== null) {
      // the RAW value, not new Date(cut) -- plotly renders a Date in the browser's local zone,
      // which puts the marker hours away from the candle that was clicked
      shapes = shapes.concat([{ name: 'asof', type: 'line', xref: 'x', yref: 'paper',
                                x0: rawX, x1: rawX, y0: 0, y1: 1,
                                line: { color: '#ffd54f', width: 1, dash: 'dot' } }]);
    }
    Plotly.relayout(gd, { shapes: shapes });
  }

  // ---- the control ------------------------------------------------------------------------
  var bar = document.createElement('div');
  // BOTTOM-left: the chart's own control rows (FVGs / S-R / EMAs) own the strip above the plot,
  // and this sat straight on top of them.
  bar.style.cssText = 'position:absolute;left:10px;bottom:8px;z-index:20;font:11px system-ui,' +
    'sans-serif;display:flex;gap:4px;align-items:center;background:rgba(17,20,24,.85);' +
    'padding:3px 5px;border-radius:4px';
  function mk(txt) {
    var b = document.createElement('button');
    b.textContent = txt;
    b.style.cssText = 'background:rgba(28,32,38,.96);color:#dfe3e8;border:1px solid #4a5058;' +
      'border-radius:3px;padding:3px 8px;cursor:pointer;font:inherit';
    return b;
  }
  var bRewind = mk('⟲ Rewind: off'), bLive = mk('live'), note = document.createElement('span');
  note.style.cssText = 'color:#9aa4ae;padding-left:4px';
  bar.appendChild(bRewind); bar.appendChild(bLive); bar.appendChild(note);
  gd.style.position = 'relative';
  gd.appendChild(bar);

  function setArmed(on) {
    armed = on;
    bRewind.textContent = on ? '⟲ Rewind: click a candle' : '⟲ Rewind: off';
    bRewind.style.background = on ? '#3b4a5a' : 'rgba(28,32,38,.96)';
  }
  bRewind.onclick = function () { setArmed(!armed); };
  bLive.onclick = function () {
    setArmed(false); atX = null; note.textContent = ''; apply(null, null);
  };
  gd.on('plotly_click', function (ev) {
    if (!armed || !ev.points || !ev.points.length) { return; }
    var p = ev.points[0];
    // THE POINT'S OWN DATA VALUE, not ev.points[0].x. The click hands back a formatted,
    // TIMEZONE-NAIVE string ("2026-06-06 06:10"), which new Date() reads as LOCAL time -- on a
    // UTC+5 machine that cut the chart five hours early and silently dropped everything in
    // between. The stored value carries its offset ("...T06:10:00+00:00") and parses correctly.
    var fd = gd._fullData && gd._fullData[p.curveNumber];
    var raw = (fd && fd.x && p.pointIndex !== undefined && p.pointIndex !== null)
      ? fd.x[p.pointIndex] : p.x;
    atX = ms(raw);
    apply(atX, raw);
    var tc = tradeCount();
    note.textContent = 'as of ' + new Date(atX).toISOString().slice(0, 16).replace('T', ' ')
                     + ' UTC \u00b7 ' + tc.live + ' of ' + tc.total + ' trades so far'
                     + (tc.live < tc.total ? ' (the rest are hidden \u2014 press live)' : '');
  });
})();
"""
