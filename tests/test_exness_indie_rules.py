"""
Exness's Indie compiles a narrower language than the CPython the stand-in runs, so an indicator
can pass every other test here and still fail to COMPILE on Exness. On 1 Nov 2026 one did:
trendlines_mtf.py copied its lists with `list(...)`, and Exness answered "class `list` does not
have a public constructor" (the current Indie docs list `list()` as a builtin; Exness runs an
older Indie). Three checks run on every file in exness/:

1. ONLY WHAT EXNESS HAS COMPILED. Every syntax node, call, method, module attribute, import,
   keyword argument and annotated type must be one that compiled on Exness in
   fair_value_gaps.py, trendlines.py or probe_timeframes.py (v1 and v2) -- or, marked below,
   one the Indie docs' compact API lists. Something new goes in the lists only after it has
   compiled on Exness.
2. BLOCK SCOPE. "In Indie, variable `res` will not exist after the if..else statements"
   (Indie docs, Language differences with Python): a variable lives only in the block that
   created it. CPython keeps it, so a read after the block passes every run here and is a
   compile error there. levels.py had one (`z` in `_start`), found by this check.
3. TYPES, when mypy is installed. Indie is statically typed: a variable keeps the type it was
   created with, None needs Optional, every function states its types. The Indie API is
   described in tests/indie_stubs/ and each @param gives its __init__ argument its type, the
   way Indie does. The files that compiled on Exness pass this check with no error.
"""
import ast
import builtins
import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "exness").glob("*.py"))
STUBS = ROOT / "tests" / "indie_stubs"

NODES = {  # every syntax node in the files that compiled on Exness
    "Add", "And", "AnnAssign", "Assign", "Attribute", "AugAssign", "BinOp", "BoolOp", "Call",
    "ClassDef", "Compare", "Constant", "Continue", "Div", "Eq", "Expr", "FloorDiv", "For",
    "FunctionDef", "Gt", "GtE", "If", "IfExp", "ImportFrom", "List", "Load", "Lt", "LtE", "Mod",
    "Module", "Mult", "Name", "Not", "NotEq", "Or", "Return", "Store", "Sub", "Subscript",
    "Tuple", "USub", "UnaryOp", "While", "alias", "arg", "arguments", "keyword"}
CALLS = {  # bare names called
    "abs", "float", "int", "isnan", "len", "max", "min", "range", "round", "str", "rgba",
    "indicator", "param_ref", "AbsolutePosition", "RelativePosition", "LabelAbs", "LabelRel",
    "LineSegment", "Rectangle"}
METHODS = {"append", "pop", "get", "set", "draw", "erase", "to_minutes"}  # on a value
FIELDS = {"point_a", "point_b", "time", "position", "text", "color", "bg_color",  # of a drawing
          "line_color", "line_style", "line_width"}
SELF_API = {"bar_index", "calc_on", "chart", "close", "high", "low", "open", "time",
            "time_frame", "is_closed_bar", "is_last_bar", "new_var"}
MODULE_ATTRS = {
    "color.GREEN", "color.RED", "color.BLUE", "color.TRANSPARENT",
    "color.NAVY", "color.YELLOW",                         # the docs' compact API
    "line_style.SOLID", "line_style.DOTTED",
    "line_style.DASHED",                                  # the docs' compact API
    "vertical_anchor.TOP", "horizontal_anchor.LEFT",
    "horizontal_anchor.RIGHT",                            # the docs' compact API
    "param.int", "param.float", "param.bool", "param.str", "param.color",
    "TimeFrame.from_str", "Atr.new"}
IMPORTS = {
    "math.isnan", "math.nan", "indie.indicator", "indie.param", "indie.param_ref", "indie.color",
    "indie.line_style", "indie.MainContext", "indie.SecContext", "indie.TimeFrame",
    "indie.color.rgba", "indie.algorithms.Atr", "indie.drawings.Rectangle",
    "indie.drawings.AbsolutePosition", "indie.drawings.RelativePosition",
    "indie.drawings.LineSegment", "indie.drawings.LabelAbs", "indie.drawings.LabelRel",
    "indie.drawings.vertical_anchor", "indie.drawings.horizontal_anchor"}
KEYWORDS = {
    "LabelAbs": {"bg_color", "text_color", "font_size"},
    "LabelRel": {"bg_color", "text_color", "font_size"},
    "LineSegment": {"color", "line_style", "line_width"},
    "Rectangle": {"bg_color", "line_color", "line_style", "line_width", "middle_line_color",
                  "middle_line_style", "middle_line_width"},
    "indicator": {"overlay_main_pane"},
    "param.int": {"default", "min", "max", "title"},
    "param.float": {"default", "min", "max", "step", "title"},
    "param.bool": {"default", "title"},
    "param.str": {"default", "options", "title"},
    "param.color": {"default", "title"},
    "self.calc_on": {"time_frame"}}
TYPES = {"int", "float", "bool", "str", "list", "None", "TimeFrame", "LineSegment", "Rectangle",
         "LabelRel",
         "LabelAbs"}  # the docs: LabelAbs and LabelRel are the same Label class


def _dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        d = _dotted(n.value)
        return f"{d}.{n.attr}" if d else None
    return None


def _unproven(src):
    """Everything in `src` that has not compiled on Exness, as (line, what)."""
    tree = ast.parse(src)
    bad = []
    alias = {}                                   # name in the file -> its imported name
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                alias[a.asname or a.name] = a.name
                if f"{n.module}.{a.name}" not in IMPORTS:
                    bad.append((n.lineno, f"import {n.module}.{a.name}"))
    own = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assigned_self = {m.attr for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign))
                     for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                     for m in ast.walk(t) if isinstance(m, ast.Attribute)
                     and isinstance(m.value, ast.Name) and m.value.id == "self"}

    def canon(d):
        head, _, rest = d.partition(".")
        return f"{alias.get(head, head)}.{rest}" if rest else alias.get(head, head)

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        local = {a.arg for a in fn.args.args} | {
            m.id for m in ast.walk(fn) if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store)}
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) \
                    and c.func.id in local and c.func.id not in own:
                c.func.id = "__colour_call__"    # a colour in a variable, called with an alpha
    for n in ast.walk(tree):
        kind = type(n).__name__
        line = getattr(n, "lineno", 0)
        if kind not in NODES:
            bad.append((line, f"syntax {kind}"))
        if isinstance(n, ast.Call):
            f = n.func
            name = _dotted(f) or "?"
            if isinstance(f, ast.Name):
                if f.id not in CALLS and f.id not in own and f.id != "__colour_call__":
                    bad.append((line, f"call {f.id}()"))
            elif isinstance(f, ast.Attribute):
                base = _dotted(f.value)
                if base == "self":
                    pass                         # own methods and SELF_API: checked below
                elif base and base.split(".")[0] in alias:
                    if canon(name) not in MODULE_ATTRS:
                        bad.append((line, f"call {canon(name)}()"))
                elif f.attr not in METHODS:
                    bad.append((line, f"method .{f.attr}()"))
            key = canon(name) if name.split(".")[0] in alias else name
            if key in KEYWORDS:
                for kw in n.keywords:
                    if kw.arg not in KEYWORDS[key]:
                        bad.append((line, f"keyword {key}({kw.arg}=)"))
            elif n.keywords and not (isinstance(f, ast.Name) and f.id in own) and not (
                    name.startswith("self.") and name[5:] in own):
                bad.append((line, f"keywords to {name}()"))
        if isinstance(n, ast.Attribute) and not (isinstance(n.value, ast.Name) and (
                n.value.id == "self" or n.value.id in alias)):
            if n.attr not in FIELDS | METHODS:
                bad.append((line, f"field .{n.attr}"))
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            if n.value.id == "self":
                if n.attr not in own and n.attr not in assigned_self and n.attr not in SELF_API:
                    bad.append((line, f"self.{n.attr}"))
            elif n.value.id in alias and canon(_dotted(n)) not in MODULE_ATTRS:
                bad.append((line, canon(_dotted(n))))
        for ann in [getattr(n, "annotation", None), getattr(n, "returns", None)]:
            if ann is not None:
                for m in ast.walk(ann):
                    t = m.id if isinstance(m, ast.Name) else (
                        "None" if isinstance(m, ast.Constant) and m.value is None else None)
                    if t is not None and t not in TYPES:
                        bad.append((line, f"type {t}"))
        if isinstance(n, ast.FunctionDef) and n.args.defaults:
            bad.append((line, f"default argument in {n.name}()"))
        if isinstance(n, ast.FunctionDef) and n.name not in ("__init__", "calc"):
            if n.returns is None or any(a.annotation is None for a in n.args.args[1:]):
                bad.append((line, f"{n.name}() does not state its types"))
    return sorted(set(bad))


class _Scope:
    """Reads of a name outside the block that created it (Indie scopes variables to blocks)."""

    def __init__(self, src):
        self.tree = ast.parse(src)
        self.module = set(dir(builtins))
        for n in self.tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                self.module.add(n.name)
            elif isinstance(n, ast.ImportFrom):
                self.module |= {a.asname or a.name for a in n.names}
        self.bad = []

    def run(self):
        for n in ast.walk(self.tree):
            if isinstance(n, ast.FunctionDef):
                self._block(n.body, [{a.arg for a in n.args.args}])
        return sorted(set(self.bad))

    def _known(self, name, scopes):
        return name in self.module or any(name in s for s in scopes)

    def _reads(self, node, scopes):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                    and not self._known(n.id, scopes):
                self.bad.append((n.lineno, n.id))

    def _bind(self, target, scopes):
        for n in ast.walk(target):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                if not self._known(n.id, scopes):
                    scopes[-1].add(n.id)
            elif isinstance(n, ast.Subscript):
                self._reads(n.value, scopes)
                self._reads(n.slice, scopes)
            elif isinstance(n, ast.Attribute):
                self._reads(n.value, scopes)

    def _block(self, stmts, scopes):
        scopes = scopes + [set()]
        for s in stmts:
            if isinstance(s, ast.Assign):
                self._reads(s.value, scopes)
                for t in s.targets:
                    self._bind(t, scopes)
            elif isinstance(s, ast.AnnAssign):
                if s.value is not None:
                    self._reads(s.value, scopes)
                self._bind(s.target, scopes)
            elif isinstance(s, ast.AugAssign):
                self._reads(s.value, scopes)
                self._reads(ast.Name(s.target.id, ast.Load(), lineno=s.lineno)
                            if isinstance(s.target, ast.Name) else s.target, scopes)
            elif isinstance(s, (ast.If, ast.While)):
                self._reads(s.test, scopes)
                self._block(s.body, scopes)
                self._block(s.orelse, scopes)
            elif isinstance(s, ast.For):
                self._reads(s.iter, scopes)
                inner = scopes + [set()]
                self._bind(s.target, inner)
                self._block(s.body, inner)
                self._block(s.orelse, scopes)
            elif isinstance(s, (ast.Return, ast.Expr)) and s.value is not None:
                self._reads(s.value, scopes)


def _with_param_types(src):
    """The source with each __init__ argument typed from its @param (or @param_ref), as Indie
    types it."""
    tree = ast.parse(src)
    kinds = {"int": "int", "float": "float", "bool": "bool", "str": "str", "color": "Color"}
    types = {}
    for c in tree.body:
        if isinstance(c, ast.ClassDef):
            for d in c.decorator_list:
                if isinstance(d, ast.Call) and _dotted(d.func) in {f"param.{k}" for k in kinds}:
                    types[d.args[0].value] = kinds[d.func.attr]
    for c in tree.body:
        if isinstance(c, ast.ClassDef):
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                    for a in f.args.args[1:]:
                        a.annotation = ast.Name(types[a.arg])
    return "from indie.color import Color\n" + ast.unparse(tree)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_only_constructs_exness_has_compiled(path):
    bad = _unproven(path.read_text(encoding="utf-8"))
    assert not bad, f"{path.name}: not seen compiling on Exness: {bad}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_no_variable_read_outside_its_block(path):
    bad = _Scope(path.read_text(encoding="utf-8")).run()
    assert not bad, f"{path.name}: read outside the block that created it: {bad}"


def test_the_checks_catch_what_exness_refuses():
    ok = ("from indie import MainContext\n"
          "class Main(MainContext):\n"
          "    def _f(self, a: list[float]) -> float:\n"
          "        z = 0.0\n"
          "        if len(a) > 0:\n"
          "            z = a[0]\n"
          "        return z\n")
    assert _unproven(ok) == [] and _Scope(ok).run() == []
    for bad in ["        b = list(a)\n", "        b = [x for x in a]\n", "        a.insert(0, 1.0)\n",
                "        b = f'{z}'\n", "        b = a[1:]\n", "        b = {1: 2}\n",
                "        b = color.WHITE\n", "        b = a.copy()\n",
                "        b = LabelAbs('', AbsolutePosition(0.0, 0.0)).font_size\n"]:
        src = ok.replace("        return z\n", bad + "        return z\n")
        if "color." in bad:
            src = "from indie import color\n" + src
        assert _unproven(src), bad
    scoped = ok.replace("        z = 0.0\n", "").replace(
        "            z = a[0]\n", "            z = a[0]\n        else:\n            z = 0.0\n")
    assert [n for _, n in _Scope(scoped).run()] == ["z"]   # the Indie docs' own example
    loop = ok.replace("        return z\n", "        for k in range(2):\n            z = 1.0\n"
                                             "        return z + float(k)\n")
    assert [n for _, n in _Scope(loop).run()] == ["k"]
    untyped = ok.replace("def _f(self, a: list[float]) -> float:", "def _f(self, a):")
    assert _unproven(untyped) == [(3, "_f() does not state its types")]


def test_types_as_indie_checks_them(tmp_path):
    api = pytest.importorskip("mypy.api", reason="the type check needs mypy")
    paths = []
    for p in FILES:
        out = tmp_path / f"x_{p.stem}.py"
        out.write_text(_with_param_types(p.read_text(encoding="utf-8")), encoding="utf-8")
        paths.append(str(out))
    old = os.environ.get("MYPYPATH")
    os.environ["MYPYPATH"] = str(STUBS)
    try:
        report, err, status = api.run([
            "--check-untyped-defs", "--enable-error-code", "possibly-undefined",
            "--python-version", "3.12", "--no-error-summary", "--show-error-codes",
            "--cache-dir", os.path.join(tempfile.gettempdir(), "quant_indie_mypy"), *paths])
    finally:
        if old is None:
            del os.environ["MYPYPATH"]
        else:
            os.environ["MYPYPATH"] = old
    assert status == 0, report + err
