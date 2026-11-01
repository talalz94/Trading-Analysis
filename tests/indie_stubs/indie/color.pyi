class Color:
    def __call__(self, alpha: float) -> Color: ...

def rgba(r: int, g: int, b: int, a: float) -> Color: ...

GREEN: Color
RED: Color
BLUE: Color
NAVY: Color
YELLOW: Color
TRANSPARENT: Color
