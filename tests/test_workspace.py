"""
STRATEGIES AS FILES, NOTEBOOKS AS VIEWS -- the layout `quant/research/workspace.py` describes.

What each test protects: a setting living in two files (the change that "didn't take"), a notebook
that states or defines something, a stale strategy file answering instead of the edited one, the
re-trade shortcut reusing frames a changed setting should have rebuilt, and the evaluation seeing
different trades from the lab it is supposed to judge.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant.research.workspace import (NOTEBOOKS, STRATEGIES, evaluate_config, lab_config,
                                      list_strategies, load_settings, load_strategy)
from test_notebook_config import SETTING_RE

ROOT = Path(__file__).resolve().parents[1]
CACHED = (ROOT / "data" / "binance_spot_BTCUSDT_5m.parquet").is_file()
VIEW_KEYS = set(load_settings(NOTEBOOKS / "lab_config.py")) - {"STRATEGY"}
EVAL_KEYS = set(load_settings(NOTEBOOKS / "evaluate_config.py")) - {"STRATEGY"}


# ---------------------------------------------------------------- one home per setting
@pytest.mark.parametrize("name", list_strategies())
def test_a_strategy_file_states_only_what_is_traded(name):
    s = load_strategy(name)
    for need in ("SYMBOL", "TF", "LINE_TFS", "TRENDLINE", "TP_RR", "COSTS", "COMPARE"):
        assert need in s, f"{name} has no {need}"
    assert not VIEW_KEYS & set(s), f"{name} states view settings {sorted(VIEW_KEYS & set(s))}"
    assert not EVAL_KEYS & set(s), f"{name} states evaluation settings"


def test_the_view_configs_name_a_real_strategy_and_merge():
    assert lab_config().STRATEGY in list_strategies()
    assert evaluate_config().STRATEGY in list_strategies()


def test_a_setting_in_both_files_is_refused(tmp_path):
    (tmp_path / "s.py").write_text("TF = '5m'\nTP_RR = 3.0\n", encoding="utf-8")
    view = tmp_path / "view.py"
    view.write_text("STRATEGY = 's'\nTP_RR = 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="TP_RR"):
        lab_config(view, folder=tmp_path)
    view.write_text("VIEW_START = '2026-01-01'\n", encoding="utf-8")
    with pytest.raises(KeyError, match="STRATEGY"):
        lab_config(view, folder=tmp_path)
    view.write_text("STRATEGY = 'nope'\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nope"):
        lab_config(view, folder=tmp_path)


def test_an_edited_strategy_is_read_fresh_without_a_reload(tmp_path):
    """runpy, not import: a stale module can never answer instead of the file on disk."""
    (tmp_path / "s.py").write_text("TP_RR = 3.0\n", encoding="utf-8")
    view = tmp_path / "view.py"
    view.write_text("STRATEGY = 's'\n", encoding="utf-8")
    assert lab_config(view, folder=tmp_path).TP_RR == 3.0
    (tmp_path / "s.py").write_text("TP_RR = 2.0\n", encoding="utf-8")
    assert lab_config(view, folder=tmp_path).TP_RR == 2.0


# ---------------------------------------------------------------- the notebooks state nothing
@pytest.mark.parametrize("nb", ["lab.ipynb", "evaluate.ipynb"])
def test_the_new_notebooks_state_no_setting_and_define_no_function(nb):
    bad = []
    for c in json.loads((NOTEBOOKS / nb).read_text(encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        bad += ["assigns: " + l.strip() for l in src.splitlines() if re.match(SETTING_RE, l)]
        bad += ["defines: " + n.name for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert not bad, bad


@pytest.mark.parametrize("nb", ["lab.ipynb", "evaluate.ipynb"])
def test_the_new_notebooks_read_their_config_through_the_workspace(nb):
    src = "".join("".join(c["source"]) for c in
                  json.loads((NOTEBOOKS / nb).read_text(encoding="utf-8"))["cells"])
    want = "lab_config()" if nb == "lab.ipynb" else "evaluate_config()"
    assert want in src
    assert "import nb" not in src, "a view notebook must not reach for a numbered notebook's config"


# ---------------------------------------------------------------- the re-trade shortcut
def test_the_loader_never_reads_a_setting_retrade_may_change():
    """RETRADE_KEYS are re-run WITHOUT reloading. If the loading half of fvg_lab ever read one, a
    re-trade would silently keep frames built from the old value -- so parse it and check."""
    from quant.research.fvg_lab import RETRADE_KEYS
    src = (ROOT / "quant" / "research" / "fvg_lab.py").read_text(encoding="utf-8")
    body = src[src.index("def fvg_lab("):src.index("return _trade(c, base)")]
    read = set(re.findall(r"\bc\.([A-Z][A-Z0-9_]*)", body))
    read |= set(re.findall(r'_get\(c, "([A-Z][A-Z0-9_]*)"', body))
    assert read, "the parse found nothing -- the check itself is broken"
    assert not read & RETRADE_KEYS, f"the loader reads {sorted(read & RETRADE_KEYS)}"


# ---------------------------------------------------------------- the evaluation
def test_quarters_and_windows():
    from experiments.evaluate import quarter_window, quarters
    assert quarters("2022Q3", "2023Q2") == ["2022Q3", "2022Q4", "2023Q1", "2023Q2"]
    s, e = quarter_window("2026Q3", "2026-08-31")
    assert (s, e) == (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-31"))
    assert quarter_window("2024Q1")[1] == pd.Timestamp("2024-04-01")


@pytest.mark.skipif(not CACHED, reason="needs the cached BTCUSDT data")
def test_the_evaluation_sees_the_trades_the_lab_sees():
    """Evaluation switches drawing off and builds 4h from 1h. Neither may change a single trade, or
    it would be judging a different strategy from the one the lab shows."""
    import quant.data as qdata
    from experiments.evaluate import _WINDOW_ONLY, frames_4h_from_1h
    from quant.data.timeframe import tf_frames
    from quant.research import fvg_lab
    cfg = lab_config()
    want = fvg_lab(cfg).trades
    slim = SimpleNamespace(**{**vars(cfg), **_WINDOW_ONLY})
    qdata.tf_frames = frames_4h_from_1h
    try:
        got = fvg_lab(slim).trades
    finally:
        qdata.tf_frames = tf_frames
    cols = ["entry_time", "side", "R", "close_reason"]
    pd.testing.assert_frame_equal(want[cols], got[cols])


@pytest.mark.skipif(not CACHED, reason="needs the cached BTCUSDT data")
def test_an_evaluation_runs_end_to_end_and_skips_an_arm_that_changes_nothing():
    from experiments.evaluate import evaluate
    cfg = evaluate_config()
    ev = evaluate(cfg, arms={"same": dict(TP_RR=cfg.TP_RR), "2R": dict(TP_RR=2.0)},
                  train=("2023Q1", "2023Q1"), forward=("2025Q1", "2025Q1"), progress=False)
    assert ev.same_as_set == ["same"]
    b = ev.baseline()
    assert list(b.run) == ["as set", "as set, zero cost"] and (b.trades > 0).all()
    a = ev.arms()
    assert list(a.arm) == ["2R"] and {"train dR", "forward dR", "t", "fwd trim3"} <= set(a)
    assert "not run: ['same']" in ev.verdict()
