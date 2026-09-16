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

    gap box      [x0,x1,x1,x0,x0] -> the box ends at k. A gap MITIGATED AFTER k therefore still
                 looks open, because at k it was. That is the whole point of the tool.
    trendline    two points -> clipped with y interpolated ALONG the line, so the slope stays
                 honest. A break that happened after k is simply not drawn.
    trade        entry/stop/target run from entry to exit -> clipped at k; a trade entered after k
                 disappears; one still open at k is drawn only as far as k.
    markers      single points -> dropped when they sit after k.

ONE INACCURACY IS HANDLED AT DRAW TIME, NOT HERE. `sr_zone_table` emits a row per price revision,
and the normal chart draws only the LAST one — today's level, painted across the whole window. For
studying history that is a quiet lookahead, so `pattern_figure(rewind=True)` draws every revision as
its own segment. The clip above then shows the revision that was live at k, with no extra logic.

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
  // always an array -- three trade-marker traces arrived that way and .slice() threw. _fullData
  // is plotly's decoded copy, where the same column is a Float64Array.
  function arr(v) {
    if (v === null || v === undefined) { return []; }
    if (Array.isArray(v)) { return v.slice(); }
    if (ArrayBuffer.isView(v)) { return Array.from(v); }
    if (typeof v === 'object' && v._inputArray) { return Array.from(v._inputArray); }
    return [];
  }

  function snapshot() {
    if (ORIG) { return; }
    ORIG = {};
    rewindable().forEach(function (i) {
      var t = (gd._fullData && gd._fullData[i]) || gd.data[i];
      ORIG[i] = { x: arr(t.x), y: arr(t.y), cd: t.customdata ? arr(t.customdata) : null };
    });
  }

  var ms = function (v) { return (v instanceof Date) ? v.getTime() : new Date(v).getTime(); };

  // ONE CLIP, applied to every shape. Segments are separated by nulls; a segment starting after
  // the cut is dropped, one crossing it is cut with y interpolated along the segment.
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

  function apply(cut) {
    snapshot();
    var idx = rewindable(), X = [], Y = [], C = [], any = false;
    idx.forEach(function (i) {
      var o = ORIG[i];
      var r = (cut === null) ? { x: o.x, y: o.y, cd: o.cd } : clip(o.x, o.y, o.cd, cut);
      X.push(r.x); Y.push(r.y); C.push(r.cd); if (r.cd) any = true;
    });
    var up = { x: X, y: Y };
    if (any) { up.customdata = C; }
    Plotly.restyle(gd, up, idx);
    var shapes = (gd.layout.shapes || []).filter(function (s) { return s.name !== 'asof'; });
    if (cut !== null) {
      shapes = shapes.concat([{ name: 'asof', type: 'line', xref: 'x', yref: 'paper',
                                x0: new Date(cut), x1: new Date(cut), y0: 0, y1: 1,
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
    setArmed(false); atX = null; note.textContent = ''; apply(null);
  };
  gd.on('plotly_click', function (ev) {
    if (!armed || !ev.points || !ev.points.length) return;
    atX = ms(ev.points[0].x);
    note.textContent = 'as of ' + new Date(atX).toISOString().slice(0, 16).replace('T', ' ');
    apply(atX);
  });
})();
"""
