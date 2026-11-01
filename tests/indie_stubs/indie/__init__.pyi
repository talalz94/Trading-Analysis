import builtins
from typing import Any, Callable, Generic, TypeVar
from indie import color as color
from indie import line_style as line_style
from indie.color import Color

_T = TypeVar("_T")
_C = TypeVar("_C")

class Series(Generic[_T]):
    def __getitem__(self, i: builtins.int) -> _T: ...

class Var(Generic[_T]):
    def get(self) -> _T: ...
    def set(self, value: _T) -> None: ...

class TimeFrame:
    @staticmethod
    def from_str(s: builtins.str) -> TimeFrame: ...
    def to_minutes(self) -> builtins.int: ...

class _Chart:
    def draw(self, drawing: Any) -> None: ...
    def erase(self, drawing: Any) -> None: ...

class _Context:
    open: Series[builtins.float]
    high: Series[builtins.float]
    low: Series[builtins.float]
    close: Series[builtins.float]
    volume: Series[builtins.float]
    time: Series[builtins.float]
    bar_index: builtins.int
    is_closed_bar: builtins.bool
    is_last_bar: builtins.bool
    time_frame: TimeFrame
    chart: _Chart
    def new_var(self, init: _T) -> Var[_T]: ...
    def calc_on(self, sec_context: type, time_frame: TimeFrame) -> tuple[Series[builtins.float], ...]: ...

class MainContext(_Context): ...
class SecContext(_Context): ...

def indicator(name: builtins.str, overlay_main_pane: builtins.bool = ...) -> Callable[[_C], _C]: ...
def param_ref(name: builtins.str) -> Callable[[_C], _C]: ...

class _Param:
    def int(self, name: builtins.str, default: builtins.int = ..., min: builtins.int = ...,
            max: builtins.int = ..., title: builtins.str = ...) -> Callable[[_C], _C]: ...
    def float(self, name: builtins.str, default: builtins.float = ..., min: builtins.float = ...,
              max: builtins.float = ..., step: builtins.float = ...,
              title: builtins.str = ...) -> Callable[[_C], _C]: ...
    def bool(self, name: builtins.str, default: builtins.bool = ...,
             title: builtins.str = ...) -> Callable[[_C], _C]: ...
    def str(self, name: builtins.str, default: builtins.str = ...,
            options: builtins.list[builtins.str] = ..., title: builtins.str = ...) -> Callable[[_C], _C]: ...
    def color(self, name: builtins.str, default: Color = ...,
              title: builtins.str = ...) -> Callable[[_C], _C]: ...

param: _Param
