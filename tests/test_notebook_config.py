"""Notebook 18's settings must live in a file the notebook cannot overwrite.

The notebook is regenerated from a script. That script used to hardcode every setting, so adding
one quietly re-imposed all the others — commit 8a8b8de changed TF, LINE_TFS and WARMUP_START that
way, none of them asked for. These tests make that class of mistake impossible to reintroduce
without failing the suite.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

# what an assignment looks like, in one place so every notebook is judged the same
SETTING_RE = r"\s*([A-Z][A-Z0-9_]*)\s*(?:,\s*[A-Z][A-Z0-9_]*\s*)*=[^=]"

NB = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "18_trendline_trading_sim.ipynb"
CFG = NB.parent / "nb18_config.py"


def _cells():
    return json.loads(NB.read_text(encoding="utf-8"))["cells"]


def test_the_settings_file_exists_and_defines_the_settings():
    assert CFG.is_file(), f"{CFG.name} is the file the user edits; it must be in the repo"
    ns = {}
    exec(compile(CFG.read_text(encoding="utf-8"), str(CFG), "exec"), ns)
    for k in ("SYMBOL", "TF", "LINE_TFS", "VIEW_START", "END", "WARMUP_START",
              "TL_CLOSE_N", "STOP_N", "BE_AT_RR", "TP_RR", "COSTS", "TRENDLINE", "ENTRY_RULES"):
        assert k in ns, f"{k} is missing from {CFG.name}"


def test_the_notebook_states_no_setting_of_its_own():
    """The one rule that keeps a rebuild harmless: the notebook may READ settings, never state them.

    An assignment here is a value the build script owns, and the build script overwrites the whole
    notebook — which is exactly how three settings got changed without being asked for.
    """
    owned = {"SYMBOL", "SOURCE", "TF", "LINE_TFS", "VIEW_START", "END", "WARMUP_START",
             "TF_WARMUP_BARS", "TL_CLOSE_N", "TL_FULL_M", "TL_TOUCH_K", "TL_DIR_N", "TL_DIR_SRC",
             "TL_DIR_FULL_N", "TL_EXTEND", "TL_COMBINE", "TL_TOL_ATR", "TRADE_LONG", "TRADE_SHORT",
             "STOP_N", "STOP_BUFFER_PCT", "BE_AT_RR", "TP_RR", "USE_RETEST_BAIL", "SEC_TOUCH_N",
             "SEC_WRONG_M", "SEC_OPP_N", "SEC_MIN_PCT", "SEC_REPLACE_TP", "ENTRY_RULES",
             "RULES_COMBINE", "EMAS", "MTF_EMAS", "COSTS", "TRENDLINE", "HEIGHT", "MAX_PICKER",
             "MAX_SHAPES_TF", "MAX_TRADE_PICKER", "SHOW_SR_ZONES", "ZONE_LOOKBACK",
             "EMAS_ON"}
    offenders = []
    for c in _cells():
        if c["cell_type"] != "code":
            continue
        for line in "".join(c["source"]).splitlines():
            m = re.match(SETTING_RE, line)
            if not m:
                continue
            names = set(re.findall(r"[A-Z][A-Z0-9_]*", line.split("=")[0]))
            if names & owned:
                offenders.append(line.strip())
    assert not offenders, ("the notebook assigns settings it does not own:\n  "
                           + "\n  ".join(offenders))


def test_the_notebook_reads_the_settings_file():
    src = "\n".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "code")
    assert "nb18_config" in src, "the notebook must import its settings"
    assert "importlib.reload" in src, ("reload, so editing the file lands without restarting the "
                                       "kernel — otherwise the next surprise is an edit that "
                                       "appears to do nothing")


@pytest.mark.parametrize("name", ["TF", "LINE_TFS", "WARMUP_START"])
def test_the_three_values_that_were_changed_are_settable_in_one_place(name):
    """Whatever they are set to, there is exactly ONE place that says so."""
    import ast

    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    n = 0
    for node in tree.body:                       # module level only — parsed, so prose cannot count
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for sub in ast.walk(t):
                    n += isinstance(sub, ast.Name) and sub.id == name
    assert n == 1, f"{name} is assigned {n} times in {CFG.name}; it must have one home"


def test_a_setting_added_later_does_not_break_an_older_config_file():
    """EMAS_ON was APPENDED to the config; an editor tab holding the older file then saved over
    it and the notebook died on a NameError two cells later. Appending is still the right way to
    add a setting -- it cannot touch a line the user edited -- but the notebook must then treat a
    setting the config predates as missing, not as fatal.
    """
    import ast

    src = chr(10).join(''.join(c['source']) for c in _cells() if c['cell_type'] == 'code')
    assert 'SETTING_DEFAULTS' in src, 'the notebook has no fallback for a setting the config lacks'

    names = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(x, 'id', '') == 'SETTING_DEFAULTS'
                                                for x in node.targets):
            names = [k.value for k in node.value.keys]
            break
    assert names, 'SETTING_DEFAULTS is not assigned a literal dict'

    ns = {}
    exec(compile(CFG.read_text(encoding='utf-8'), str(CFG), 'exec'), ns)
    body = src.replace('SETTING_DEFAULTS', '')
    for k in names:
        # a default for something the notebook never reads is dead weight, and one for a setting
        # the config states is a second home for a value that must have exactly one
        assert k in body, f'{k} has a default but the notebook never uses it'
        assert k in ns, f'{k} has a default but no line in the config file to edit'


# ---------------------------------------------------------------------------
# notebook 19 lives under the same rule, for the same reason
# ---------------------------------------------------------------------------
NB19 = NB.parent / '19_fvg_viewer.ipynb'
CFG19 = NB.parent / 'nb19_config.py'


def _cells19():
    return json.loads(NB19.read_text(encoding='utf-8'))['cells']


def test_notebook_19_keeps_its_settings_in_a_file_too():
    assert CFG19.is_file(), 'nb19_config.py is the file the user edits; it must be in the repo'
    ns = {}
    exec(compile(CFG19.read_text(encoding='utf-8'), str(CFG19), 'exec'), ns)
    for k in ('SYMBOL', 'TF', 'FVG_TFS', 'FVG_MITIGATION', 'FVG_MIN_SIZE_ATR',
              'USE_FVG_VETO', 'FVG_VETO_MODE', 'FVG_VETO_TFS', 'ENTRY_RULES'):
        assert k in ns, f'{k} is missing from {CFG19.name}'
    from quant.indicators import MITIGATION_MODES
    from quant.strategies import VETO_MODES
    assert ns['FVG_MITIGATION'] in MITIGATION_MODES
    assert ns['FVG_VETO_MODE'] in VETO_MODES


def test_notebook_19_states_no_setting_of_its_own():
    owned = {'SYMBOL', 'SOURCE', 'TF', 'LINE_TFS', 'VIEW_START', 'END', 'WARMUP_START',
             'TF_WARMUP_BARS', 'FVG_TFS', 'FVG_MITIGATION', 'FVG_MIN_SIZE_ATR',
             'FVG_MIN_DISP_ATR', 'FVG_MAX_AGE_BARS', 'FVG_ATR_PERIOD', 'FVG_DRAW',
             'FVG_EXTEND', 'FVG_SHOW', 'MAX_FVG_PICKER', 'USE_FVG_VETO', 'FVG_VETO_MODE',
             'FVG_VETO_TFS', 'FVG_VETO_MAX_AGE', 'FVG_VETO_MIN_SIZE', 'FVG_VETO_MAX_DIST',
             'ENTRY_RULES', 'RULES_COMBINE', 'EMAS', 'MTF_EMAS', 'EMAS_ON', 'COSTS',
             'TRENDLINE', 'HEIGHT', 'SHOW_SR_ZONES', 'SHOW_TRENDLINES', 'START_VISIBLE',
             'MAX_PICKER', 'MAX_SHAPES_TF', 'MAX_TRADE_PICKER', 'ZONE_LOOKBACK',
             'TL_CLOSE_N', 'TL_FULL_M', 'TL_TOUCH_K', 'TL_DIR_N', 'TL_DIR_SRC',
             'TL_DIR_FULL_N', 'TL_EXTEND', 'TL_COMBINE', 'TL_TOL_ATR', 'TRADE_LONG',
             'TRADE_SHORT', 'STOP_N', 'STOP_BUFFER_PCT', 'BE_AT_RR', 'TP_RR',
             'TL_CONFIRM_N', 'FVG_RULES', 'REWIND', 'AS_OF', 'TL_ONCE_PER_LINE',
             'USE_RETEST_BAIL', 'SEC_TOUCH_N', 'SEC_WRONG_M', 'SEC_OPP_N', 'SEC_MIN_PCT',
             'SEC_REPLACE_TP'}
    offenders = []
    for c in _cells19():
        if c['cell_type'] != 'code':
            continue
        for line in ''.join(c['source']).splitlines():
            m = re.match(r'\s*([A-Z][A-Z0-9_]*)\s*(?:,\s*[A-Z][A-Z0-9_]*\s*)*=[^=]', line)
            if not m:
                continue
            names = set(re.findall(r'[A-Z][A-Z0-9_]*', line.split('=')[0]))
            if names & owned:
                offenders.append(line.strip())
    assert not offenders, ('notebook 19 assigns settings it does not own:\n  '
                          + '\n  '.join(offenders))


def test_a_gap_only_timeframe_is_never_fitted_or_traded():
    """FVG_TFS is for DRAWING and vetoing; LINE_TFS is what gets fitted and TRADED. If the lab ever
    passed the gap frames where the line frames go, adding "4h" just to see its gaps would quietly
    start trading 4h trendlines -- a change to the strategy made by a setting whose name says
    drawing. A source check on the library, because the alternative needs a live data fetch."""
    import inspect

    # `quant.research` re-exports the FUNCTION `fvg_lab`, which shadows the submodule of the same
    # name -- so import the names, not the module path
    from quant.research import FvgLab, fvg_lab as build

    src = inspect.getsource(build)
    assert "mtf_trendlines(df, line_frames" in src, "lines must be fitted on LINE frames only"
    assert "mtf_fvgs(df, gap_frames" in src, "gaps come from the GAP frames"
    fig = inspect.getsource(FvgLab.figure)
    assert "tf_frames=self.line_frames" in fig and "fvg_frames=self.gap_frames" in fig

    ns = {}
    exec(compile(CFG19.read_text(encoding="utf-8"), str(CFG19), "exec"), ns)
    assert set(ns["FVG_TFS"]) - set(ns["LINE_TFS"]),         "the config no longer has a gap-only timeframe, so this guards nothing"


def test_no_notebook_19_cell_defines_a_function():
    """A helper defined in a cell is code you scroll past, re-read to check it still matches the
    library, and eventually edit by accident. The wiring lives in quant/research/fvg_lab.py."""
    import ast

    offenders = []
    for c in _cells19():
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                offenders.append(node.name)
    assert not offenders, ("notebook 19 defines these in cells; they belong in quant/research/: "
                           + ", ".join(offenders))


def test_notebook_19_stays_short_enough_to_read():
    """The complaint this encodes: "too much stuff I breeze past". A viewer notebook is settings,
    one call and a chart -- if it grows past this, machinery has leaked back into it."""
    code = [c for c in _cells19() if c["cell_type"] == "code"]
    lines = sum(len([x for x in c["source"] if x.strip()]) for c in code)
    assert len(code) <= 9, f"{len(code)} code cells"
    assert lines <= 60, f"{lines} lines of code in the notebook; move wiring into quant/research/"


def test_the_lab_loads_a_frame_for_every_timeframe_an_ema_names():
    """MTF_EMAS is its own setting, so it may name a timeframe that is neither a line, a gap nor an
    S/R layer. One FINER than the chart cannot be resampled into existence -- pattern_figure refuses
    it by name rather than handing back the chart's own EMA under another label -- so the lab has to
    load it. Without this the config raised `1m is FINER than the chart and no 1m frame was given`
    and the only fix looked like adding "1m" to a setting that has nothing to do with EMAs."""
    import inspect

    from quant.research import FvgLab, fvg_lab as build

    src = inspect.getsource(build)
    assert "mtf_ema_timeframes(c.MTF_EMAS" in src, "the lab does not ask what the EMAs need"
    assert "ema_frames.update(tf_frames(" in src, "it asks, but never loads them"
    assert "ema_frames=self.ema_frames" in inspect.getsource(FvgLab.figure)

    ns = {}
    exec(compile(CFG19.read_text(encoding="utf-8"), str(CFG19), "exec"), ns)
    from quant.viz import mtf_ema_timeframes
    named = set(mtf_ema_timeframes(ns["MTF_EMAS"], chart_tf=ns["TF"]))
    elsewhere = set(ns["LINE_TFS"]) | set(ns["FVG_TFS"]) | set(ns.get("SR_TFS", ()))
    assert named - elsewhere, ("nb19's MTF_EMAS no longer names a timeframe outside the other "
                               "settings, so this guards nothing")


# ---------------------------------------------------------------------------
# notebook 20 — the rewind copy. Same two rules; plus it must stay a COPY.
# ---------------------------------------------------------------------------
NB20 = NB.parent / "20_rewind.ipynb"
CFG20 = NB.parent / "nb20_config.py"


def _cells20():
    return json.loads(NB20.read_text(encoding="utf-8"))["cells"]


def test_notebook_20_states_no_setting_and_defines_no_function():
    import ast

    bad = []
    for c in _cells20():
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for line in src.splitlines():
            if re.match(SETTING_RE, line):
                bad.append("assigns: " + line.strip())
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        bad += ["defines: " + n.name for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not bad, bad


def test_notebook_20_config_is_a_copy_of_19s_plus_the_rewind_settings():
    """The point of notebook 20 is that ONLY the rewind differs. If the trading settings drift
    apart the two notebooks stop being comparable and the copy has no reason to exist."""
    a, b = {}, {}
    exec(compile(CFG19.read_text(encoding="utf-8"), str(CFG19), "exec"), a)
    exec(compile(CFG20.read_text(encoding="utf-8"), str(CFG20), "exec"), b)
    ka = {k: v for k, v in a.items() if k.isupper()}
    kb = {k: v for k, v in b.items() if k.isupper()}
    drifted = [k for k in ka if repr(ka[k]) != repr(kb.get(k))]
    assert not drifted, f"notebook 20 no longer matches 19 on: {drifted}"
    # notebook 20 owns the point-in-time tool AND the two entry conditions built alongside it;
    # everything else must still match 19 exactly
    assert set(kb) - set(ka) == {"REWIND", "AS_OF", "FVG_RULES",
                                 "TL_CONFIRM_N", "TL_CONFIRM_ALL", "TL_CONFIRM_MODE",
                                 "TL_CONFIRM_COLOUR"},         sorted(set(kb) - set(ka))


def test_notebook_20_opens_the_chart_with_the_rewind_script():
    src = chr(10).join("".join(c["source"]) for c in _cells20()
                       if c["cell_type"] == "code")
    assert "extra_js=AS_OF_JS" in src, (
        "the tool only exists in the browser tab, so the script has to be injected there")


# ---------------------------------------------------------------------------
# notebook 21 — the 1h break confirmation
# ---------------------------------------------------------------------------
NB21 = NB.parent / "21_break_confirmation.ipynb"
CFG21 = NB.parent / "nb21_config.py"


def _cells21():
    return json.loads(NB21.read_text(encoding="utf-8"))["cells"]


def test_notebook_21_exists_with_its_own_settings_file():
    assert NB21.is_file() and CFG21.is_file()


def test_notebook_21_states_no_setting_and_defines_no_function():
    import ast

    bad = []
    for c in _cells21():
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        bad += ["assigns: " + l.strip() for l in src.splitlines() if re.match(SETTING_RE, l)]
        bad += ["defines: " + n.name for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not bad, bad


def test_notebook_21_config_is_20s_plus_the_cancel_switch():
    """21 exists to isolate ONE rule. If the trading settings drift from 20 the comparison stops
    meaning anything, exactly as it would between 19 and 20."""
    a, b = {}, {}
    exec(compile(CFG20.read_text(encoding="utf-8"), str(CFG20), "exec"), a)
    exec(compile(CFG21.read_text(encoding="utf-8"), str(CFG21), "exec"), b)
    ka = {k: v for k, v in a.items() if k.isupper()}
    kb = {k: v for k, v in b.items() if k.isupper()}
    assert set(kb) - set(ka) == {"TL_CONFIRM_CANCEL", "TL_CONFIRM_FROM",
                                 "SHOW_REJECTED"}, sorted(set(kb) - set(ka))
    drifted = [k for k in ka if repr(ka[k]) != repr(kb.get(k))]
    assert drifted == ["TL_CONFIRM_N"], f"only the confirmation may differ, got {drifted}"
    assert kb["TL_CONFIRM_N"], "notebook 21 is pointless with the confirmation off"
    assert kb["TL_CONFIRM_CANCEL"] is True


def test_notebook_21_config_is_not_an_essay():
    """The settings file is a control panel, not documentation -- the rules themselves are
    documented beside the code that runs them. A config you have to READ is one you stop editing."""
    text = CFG21.read_text(encoding="utf-8")
    lines = text.splitlines()
    comment = sum(1 for l in lines if l.lstrip().startswith("#"))
    assert len(lines) < 160, f"{len(lines)} lines"
    assert comment / max(len(lines), 1) < 0.35, f"{comment} of {len(lines)} lines are comment"


def test_notebook_21_runs_the_study_both_ways():
    """A confirmation can only remove or delay trades, so a single run cannot say whether it
    removed losers or removed trades at random. Both columns or the notebook proves nothing."""
    src = chr(10).join("".join(c["source"]) for c in _cells21() if c["cell_type"] == "code")
    assert "variant(" in src and "TL_CONFIRM_N=0" in src, "the OFF run must be built from the ON one"
    assert "compare_runs(" in src and "trade_diff(" in src
