# What a module needs to describe a drawing without knowing what will draw it: rectangles, fonts,
# and the painter a chart function is handed. Nothing here needs a window, so modules that draw
# charts can still be loaded, run and tested with no screen.
#
# A chart is a function(painter, area). area is a Rect and painter has these methods, where a
# colour is (red, green, blue), a point is (x, y) and a width of 0 fills the shape:
#   painter.line(colour, start, end, width=1)
#   painter.lines(colour, points, width=1)             a line through each point in turn
#   painter.rect(colour, rect, width=0, radius=0)
#   painter.circle(colour, centre, radius, width=0)
#   painter.polygon(colour, points, width=0)
#   painter.text(font, text, colour, rotated=False, **anchor)
#       anchor places the text by one point of its box, e.g. midtop=(x, y) or topleft=(x, y):
#       any of topleft, midtop, topright, midleft, center, midright, bottomleft, midbottom,
#       bottomright. rotated turns the text to read upwards
#   painter.set_clip(rect)                             only draws inside rect until set_clip(None)
# The window's own painter is in ui/painter.py


class Rect:
    # A rectangle by its left, top, width and height, with the usual edges and middles to read
    def __init__(self, left=0, top=0, width=0, height=0):
        if isinstance(left, (tuple, list, Rect)):
            left, top, width, height = tuple(left)
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    def __iter__(self):
        return iter((self.left, self.top, self.width, self.height))

    def __repr__(self):
        return f"Rect({self.left}, {self.top}, {self.width}, {self.height})"

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def centerx(self):
        return self.left + self.width / 2

    @property
    def centery(self):
        return self.top + self.height / 2

    @property
    def center(self):
        return (self.centerx, self.centery)

    def copy(self):
        return Rect(self.left, self.top, self.width, self.height)

    # The rectangle grown by x and y in all, about its middle. Negative values shrink it
    def inflate(self, x, y):
        return Rect(self.left - x / 2, self.top - y / 2, self.width + x, self.height + y)

    def collidepoint(self, x, y=None):
        if y is None:
            x, y = x
        return self.left <= x < self.right and self.top <= y < self.bottom


# The points a box can be placed by, as the share of its width and height each sits at
ANCHORS = {
    "topleft": (0, 0), "midtop": (0.5, 0), "topright": (1, 0),
    "midleft": (0, 0.5), "center": (0.5, 0.5), "midright": (1, 0.5),
    "bottomleft": (0, 1), "midbottom": (0.5, 1), "bottomright": (1, 1),
}


# The top left corner of a box of the given size placed by an anchor, e.g. midtop=(x, y).
# With no anchor the box starts at (0, 0)
def anchored(width, height, anchor):
    for name, (share_x, share_y) in ANCHORS.items():
        if name in anchor:
            x, y = anchor[name]
            return x - width * share_x, y - height * share_y
    return 0, 0


# Works out how much room text takes. The window swaps in one that asks the real font (see
# ui/painter.py); without a window this estimate stands in, which is close enough to lay a chart out
def estimate(font, text):
    return round(len(text) * font.points * 0.55), round(font.points * 1.4)


measure = estimate


class Font:
    # A font by its size, written the way the app has always sized its text: 22 is the small text
    # in tables and charts, 26 the usual text in panels and 32 a heading. points is the size the
    # letters are drawn at
    def __init__(self, size, bold=False):
        self.size_name = size
        self.bold = bold
        self.points = max(6, round(size * 0.62))

    # The (width, height) of text in this font
    def size(self, text):
        return measure(self, str(text))

    def get_height(self):
        return self.size("Ag")[1]


# Fonts already made, by size, so asking for the same size twice gives the same font
fonts = {}


def font(size, bold=False):
    if (size, bold) not in fonts:
        fonts[(size, bold)] = Font(size, bold)
    return fonts[(size, bold)]


# Shortens text with "..." until it fits inside a width
def fit_text(font, text, width):
    if font.size(text)[0] <= width:
        return text
    while text and font.size(text + "...")[0] > width:
        text = text[:-1]
    return text + "..."


# Splits text into lines that each fit inside a width, breaking between words
def wrap_text(font, text, width):
    lines = []
    line = ""
    for word in text.split():
        attempt = f"{line} {word}".strip()
        if font.size(attempt)[0] <= width:
            line = attempt
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class Recorder:
    # A painter that only keeps a list of what it was asked to draw, as (method, arguments).
    # The tests use it to check a chart draws without a window to draw on
    def __init__(self):
        self.calls = []

    def line(self, colour, start, end, width=1):
        self.calls.append(("line", colour, start, end, width))

    def lines(self, colour, points, width=1):
        self.calls.append(("lines", colour, list(points), width))

    def rect(self, colour, rect, width=0, radius=0):
        self.calls.append(("rect", colour, tuple(rect), width))

    def circle(self, colour, centre, radius, width=0):
        self.calls.append(("circle", colour, centre, radius, width))

    def polygon(self, colour, points, width=0):
        self.calls.append(("polygon", colour, list(points), width))

    def text(self, font, text, colour, rotated=False, **anchor):
        self.calls.append(("text", str(text), colour, rotated, anchor))

    def set_clip(self, rect):
        self.calls.append(("clip", None if rect is None else tuple(rect)))
