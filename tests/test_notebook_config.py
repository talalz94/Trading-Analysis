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
            m = re.match(r"\s*([A-Z][A-Z0-9_]*)\s*(?:,\s*[A-Z][A-Z0-9_]*\s*)*=[^=]", line)
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
