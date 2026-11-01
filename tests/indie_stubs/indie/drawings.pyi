from indie.color import Color
from indie.line_style import LineStyle

class VerticalAnchor: ...
class HorizontalAnchor: ...

class vertical_anchor:
    TOP: VerticalAnchor
    MIDDLE: VerticalAnchor
    BOTTOM: VerticalAnchor

class horizontal_anchor:
    LEFT: HorizontalAnchor
    CENTER: HorizontalAnchor
    RIGHT: HorizontalAnchor

class AbsolutePosition:
    time: float
    price: float
    def __init__(self, time: float, price: float) -> None: ...

class RelativePosition:
    vertical_anchor: VerticalAnchor
    horizontal_anchor: HorizontalAnchor
    top_offset: float
    left_offset: float
    def __init__(self, vertical_anchor: VerticalAnchor, horizontal_anchor: HorizontalAnchor,
                 top_offset: float, left_offset: float) -> None: ...

class LineSegment:
    point_a: AbsolutePosition
    point_b: AbsolutePosition
    color: Color
    line_style: LineStyle
    line_width: int
    def __init__(self, point_a: AbsolutePosition, point_b: AbsolutePosition, color: Color = ...,
                 line_style: LineStyle = ..., line_width: int = ...) -> None: ...

class Rectangle:
    point_a: AbsolutePosition
    point_b: AbsolutePosition
    bg_color: Color
    line_color: Color
    line_style: LineStyle
    line_width: int
    middle_line_color: Color
    middle_line_style: LineStyle
    middle_line_width: int
    def __init__(self, point_a: AbsolutePosition, point_b: AbsolutePosition,
                 bg_color: Color = ..., line_color: Color = ..., line_style: LineStyle = ...,
                 line_width: int = ..., middle_line_color: Color = ...,
                 middle_line_style: LineStyle = ..., middle_line_width: int = ...) -> None: ...

class LabelAbs:
    text: str
    position: AbsolutePosition
    bg_color: Color
    text_color: Color
    font_size: int
    def __init__(self, text: str, position: AbsolutePosition, bg_color: Color = ...,
                 text_color: Color = ..., font_size: int = ...) -> None: ...

class LabelRel:
    text: str
    position: RelativePosition
    bg_color: Color
    text_color: Color
    font_size: int
    def __init__(self, text: str, position: RelativePosition, bg_color: Color = ...,
                 text_color: Color = ..., font_size: int = ...) -> None: ...
