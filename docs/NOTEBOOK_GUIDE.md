# Notebooks and their configs — where everything belongs

> For whoever (person or agent) touches a notebook next. Two rules and a table; the rest is why.

A notebook here reads as **settings → one call → a chart**. It is a control panel, not a program.
Everything else lives in the library, where it is imported, tested and versioned.

---

## Start here: strategies are files, notebooks are views (2026-09-27)

```
notebooks/strategies/<name>.py   WHAT is traded -- entries, exits, costs, the gaps the rules read,
                                 and COMPARE: the questions this strategy is asking
notebooks/lab_config.py          WHICH strategy, WHEN (the window), HOW it is drawn
notebooks/lab.ipynb              that strategy on that window: trades, its COMPARE arms, the chart
notebooks/evaluate_config.py     WHICH strategy, and which quarters train and judge it
notebooks/evaluate.ipynb         the strategy and its COMPARE arms over 19 quarters: the verdict
```

**A new idea is a toggle in a strategy file plus an arm in its `COMPARE`. A new strategy is a copied
strategy file. Neither is a new notebook.** The lab shows the idea on a window; evaluate says
whether it is real -- about a minute, and the answer to every idea so far has come from there, not
from the window.

Why: notebooks 18 -> 19 -> 20 -> 21 were each a copy with one change, each config a few hundred
lines, each held equal to its parent by a test -- and those tests broke whenever the owner edited
their own settings. The layout separates what those files mixed.

- **Each setting has exactly one home.** `quant.research.workspace` merges a strategy with a view
  config and REFUSES a setting found in both. `tests/test_workspace.py` holds strategy files to
  trading settings only and the view configs to none.
- **Files are executed fresh** (`runpy`), not imported: an edited strategy is read on the next call,
  with no kernel restart or `importlib.reload`.
- **Arms that change trading settings only re-trade** the lab already built
  (`FvgLab.retrade`, `RETRADE_KEYS`) -- confirmations, stops, targets, costs, exits. A test parses
  the loader and fails if it ever reads one of them.
- Notebooks 18-21 and their `nb<N>_config.py` are untouched and still work; they are the history.
  The first strategy file, `tl_break_confirm.py`, is notebook 21's settings value for value, plus
  a `no confirmation` arm so the lab asks the question 21 was built for.

---

## The two rules

**1. A notebook cell may never ASSIGN a setting.**
Settings live in `notebooks/nb<N>_config.py`. Notebooks are regenerated from build scripts, and a
build script that states values re-imposes them on every rebuild — which is exactly how `TF`,
`LINE_TFS` and `WARMUP_START` were silently changed in `8a8b8de`, none of them asked for.
`tests/test_notebook_config.py` fails if an assignment creeps back into a cell.

**2. A notebook cell may never DEFINE a function or class.**
A helper defined in a cell is code you scroll past, re-read to check it still matches the library,
and eventually edit by accident. Orchestration goes in `quant/research/`.
`test_no_notebook_19_cell_defines_a_function` enforces this, and
`test_notebook_19_stays_short_enough_to_read` caps the notebook at 9 code cells / 60 lines.

---

## Where a thing goes

| If it… | it belongs in |
|---|---|
| decides **when** to trade | `quant/scenarios/` (a `Scenario`) or `quant/strategies/` (a `Cond`) |
| computes a **number from bars** | `quant/indicators/` |
| **draws** | `quant/viz/` |
| **wires those together** for a notebook | `quant/research/` |
| **sweeps** settings, scores windows, counts survivors | `experiments/` — never in `quant/` |
| is a **value you might change** | `notebooks/nb<N>_config.py` |
| is **prose** explaining what to look at | a markdown cell |

**An A/B is a setting too.** Which comparisons a notebook shows is a `COMPARE` dict in its config
(`{"move target": dict(USE_TP_MOVE=True), ...}`); the cell is just `run_arms(CFG).table()`. Each arm
changes ONLY what it names. When every arm touches exit settings only, the lab is built once and
re-traded per arm (`FvgLab.retrade`), so a four-arm table costs one load, not four.

`quant/research/` is the only one of these that is new and the only one whose boundary is easy to
get wrong. It **calls** indicators, scenarios and viz; it never contains strategy logic, indicator
maths, drawing, or a default for any setting a config owns. A default stated in two places is the
bug rule 1 exists to prevent.

---

## Writing a config file

Sections, numbered, in the order someone would change them. Notebook 19's is the model:

```python
"""
NOTEBOOK 19 — FAIR VALUE GAPS.  THIS FILE IS YOURS.

    §1  WHAT AND WHEN      symbol, timeframe, date range
    §2  GAP DETECTION      what counts as a gap and when it dies
    ...
"""

# =====================================================================
# §2  GAP DETECTION — what counts as a gap, and when it stops counting
# =====================================================================
FVG_MIN_SIZE_ATR = 0.50   # the band must be at least this tall...
```

- The docstring lists the sections, so the file has a table of contents.
- Every setting carries a comment saying what it does **and why the default is that value**. A
  default chosen from a measurement says which measurement; one chosen by convention says so.
- Related settings sit together and are separated by blank lines, not crammed onto one line.
  `HEIGHT, MAX_PICKER, MAX_SHAPES_TF, MAX_TRADE_PICKER = 900, 24, 12, 40` saves three lines and
  costs you the ability to comment any of them.

### Changing an existing config

**APPEND. Never rewrite a line the user has edited.** Add new settings at the bottom under a new
section. If a genuine re-arrangement is asked for, re-arrange — then prove nothing moved:

```python
a, b = {}, {}
exec(compile(old_text, "old", "exec"), a)
exec(compile(new_text, "new", "exec"), b)
changed = [k for k in a if k.isupper() and k in b and repr(a[k]) != repr(b[k])]
assert not changed, changed
```

### Before a change that alters what an existing notebook shows

Copy every affected config into `notebooks/backups/<date>_<reason>/` with a README naming the
values in force, the code commit, and how to get the old behaviour back — and commit the backup
BEFORE the change. If the old behaviour is reachable by a setting (as `SR_SENSITIVITY = None` is),
prove it against the old code rather than asserting it: a temporary `git worktree` at the
pre-change commit makes that a two-minute check.

### The stale-buffer trap

A setting appended to a config is **invisible to an editor tab that was already open**. Saving that
tab writes the old tail back and the new line vanishes — this cost `EMAS_ON` once, and the notebook
died on a `NameError` two cells later.

Two defences, use both:

- The notebook carries `SETTING_DEFAULTS` for settings added after the fact, so an older config
  still runs and the notebook prints the line to paste in.
- `quant/research/` helpers read settings through a `_get(cfg, name, default)` shim rather than
  `cfg.NAME`, so a missing setting degrades instead of raising.

If Jupyter offers **"File Changed"** on a config, the answer is **Reload**, never Overwrite.

---

## Making a copy of a notebook

Notebook 20 is notebook 19 plus one feature. The copy is only worth having while it stays a copy, so
generate the config **from** the original rather than by hand, and let a test hold it there:

```python
drifted = [k for k in ka if repr(ka[k]) != repr(kb.get(k))]
assert not drifted
assert set(kb) - set(ka) == {"REWIND", "AS_OF"}     # exactly the new settings, nothing else
```

`test_notebook_20_config_is_a_copy_of_19s_plus_the_rewind_settings` is that test. Without it the two
drift apart over a few sessions and the comparison the copy exists for stops meaning anything.

## Building a notebook

Build scripts live in the scratchpad, not the repo — they are throwaway. The generated `.ipynb` is
what is committed. Regenerate and execute in one go:

```bash
python build_nb19.py && python -m jupyter nbconvert --to notebook --execute --inplace notebooks/19_fvg_viewer.ipynb
```

Executing before committing matters: an `.ipynb` whose outputs are from an older version of the
library is a screenshot of something that no longer happens.

---

## The notebooks

| notebook | config | what it is for |
|---|---|---|
| **`23_break_confirmation_news`** | `nb23_config.py` | notebook 21 + news, patterns, previous D/W/M/Y highs and lows, market sessions, and S/R as role-coloured bands (docs/CHART_GUIDE.md) |
| **`market_view`** | `market_view_config.py` | **no strategy**: any instrument's latest days on 1m, with lines, zones and gaps from 1m to 4h and the news |
| `22_other_markets` | `nb22_config.py` (+ `nb21_config.py`, read fresh) | notebook 21's strategy on gold, oil and FX beside BTC, against a metronome |
| **`lab`** | `lab_config.py` + `strategies/*.py` | **start here** -- any strategy on any window |
| **`evaluate`** | `evaluate_config.py` + `strategies/*.py` | **is it real** -- a strategy and its arms over 19 quarters |
| `21_break_confirmation` | `nb21_config.py` | the 1h break confirmation + the two exit rules' A/B (now `strategies/tl_break_confirm.py`) |
| `20_rewind` | `nb20_config.py` | 19's chart + the point-in-time tool; its config is a tested copy of 19's |
| `19_fvg_viewer` | `nb19_config.py` | fair value gaps from several timeframes + the FVG veto, run with and without |
| `18_trendline_trading_sim` | `nb18_config.py` | multi-timeframe trendline breaks, traded |
| `17_multi_timeframe_patterns` | — | chart patterns from several timeframes on one chart |
| `16_trendline_research` | — | reads `experiments/trendline_settings_search` |
| `15_tuned_pattern_viewer` | — | all 22 chart patterns at calibrated per-pattern settings |
| `14_pattern_viewer` | — | chart-pattern shapes + S/R zones |

Notebooks 14–17 predate these rules and still state settings in cells. That is fine — they are
viewers nobody is actively tuning. **Any notebook that gets a config file gets both rules**, and the
test file is where you add it.
