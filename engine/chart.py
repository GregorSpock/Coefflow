import math

from engine import colours, draw
from engine.draw import Rect, fit_text


TICKS = 5
LEFT_MARGIN = 70
BOTTOM_MARGIN = 50
POINT_RADIUS = 3
LINE_WIDTH = 3
# A correlogram's lags come in twelves, so four ticks fall on whole lags
CORRELOGRAM_TICKS = 4
DASH = 8
# Bars fill this share of their slot, leaving a gap between neighbours
BAR_SHARE = 0.7
# Category names narrower than this can't be read, so names are skipped until each has this much room
MIN_LABEL_WIDTH = 60
# A line chart draws at most this many lines, one per colour in the theme's CHART_SERIES
MAX_SERIES = 6

# The size of the font panels hand to their charts. The margins and gaps above are for text this
# size, and grow or shrink with the text when a chart's style sets another size
TEXT_SIZE = 22
# How much bigger the chart's title is than the rest of its text
TITLE_EXTRA = 8
# The size in pixels of a chart saved as a PNG, unless its style sets another, and the space left round it
EXPORT_SIZE = (1200, 800)
EXPORT_MARGIN = 20
# The space left inside a background the style has coloured, so the chart doesn't touch its edge
BACKGROUND_MARGIN = 6
# A chart given less room than this, as in a short panel on a small screen, can't be drawn
MIN_SIZE = (160, 90)

# Every chart can be restyled from its panel's Settings button. The choices are kept in the node's
# settings under STYLE_KEY, so they are saved with the workflow, as a dict that only holds what
# has been changed from the automatic look:
#   title, x_label, y_label          text; without a label the chart's own is used
#   x_min, x_max, y_min, y_max       the ends of each axis
#   ticks, text_size, point_size, line_width
#   grid, key                        False hides the grid lines or the key
#   export_width, export_height      the size of the saved PNG
#   colours                          a dict of colour name -> [red, green, blue], the names being
#                                    those in THEME_COLOURS and series1 to series6
# render sets style to the chart's choices while it is drawn, and the helpers below read them, so
# a drawing function asks colour("point") rather than using colours.CHART_POINT directly.
# The drawing functions are handed a painter (see engine/draw.py) rather than a window, so a
# chart is drawn the same way on screen, into a saved picture, and in the tests
STYLE_KEY = "chart"
style = {}

# The theme colour each part of a chart has until the style gives it another
THEME_COLOURS = {
    "background": "TOOLBAR",
    "text": "TEXT",
    "grid": "GRID",
    "border": "TOOLBAR_LINE",
    "point": "CHART_POINT",
    "line": "CHART_LINE",
}
SERIES_NAMES = [f"series{i + 1}" for i in range(MAX_SERIES)]

# Colours that suit a white page, for charts going into a paper or a slide
PRINT_COLOURS = {
    "background": [255, 255, 255],
    "text": [0, 0, 0],
    "grid": [225, 225, 225],
    "border": [0, 0, 0],
    "point": [31, 119, 180],
    "line": [214, 39, 40],
    "series1": [31, 119, 180],
    "series2": [255, 127, 14],
    "series3": [44, 160, 44],
    "series4": [214, 39, 40],
    "series5": [148, 103, 189],
    "series6": [90, 90, 90],
}

# The colour of a part of the chart: the one the style gives it, or else the theme's
def colour(name):
    chosen = style.get("colours", {}).get(name)
    if chosen:
        return tuple(chosen)
    if name in SERIES_NAMES:
        return colours.CHART_SERIES[SERIES_NAMES.index(name) % len(colours.CHART_SERIES)]
    return getattr(colours, THEME_COLOURS[name])


# The colour a part of the chart has under a style, for showing it when no chart is being drawn
def colour_in(settings, name):
    global style
    style = settings or {}
    try:
        return colour(name)
    finally:
        style = {}


# The colour of the i-th series, counting from 0. A chart with colours of its own for its series
# gives them as default, which is used unless the style sets that series' colour
def series_colour(i, default=None):
    name = SERIES_NAMES[i % MAX_SERIES]
    if default is not None and (i >= MAX_SERIES or name not in style.get("colours", {})):
        return default
    return colour(name)


def sized_font(size):
    return draw.font(size)


# The font to write the chart's text in: the one it was given, or one of the style's text size
def text_font(font):
    size = style.get("text_size")
    return sized_font(int(size)) if size else font


# A length in pixels meant for text of the usual size, grown or shrunk to suit the font in use
def scaled(font, pixels):
    return round(pixels * font.get_height() / sized_font(TEXT_SIZE).get_height())


def point_radius():
    return int(style.get("point_size") or POINT_RADIUS)


def line_width():
    return int(style.get("line_width") or LINE_WIDTH)


# True unless the style hides this part of the chart ("grid" or "key")
def shows(part):
    return style.get(part, True)


# The ends of an axis ("x" or "y"): those the chart worked out, unless the style sets either.
# Ends that would leave the axis with no length are ignored
def axis_range(axis, low, high):
    chosen_low, chosen_high = style.get(f"{axis}_min"), style.get(f"{axis}_max")
    new_low = low if chosen_low is None else chosen_low
    new_high = high if chosen_high is None else chosen_high
    if new_low >= new_high:
        return low, high
    return new_low, new_high


# The part of area the data is drawn in, leaving room for the tick values and labels left and
# below, and on the right for half of the last tick value when the style makes the text bigger
def plot_rect(area, font):
    left, bottom = scaled(font, LEFT_MARGIN), scaled(font, BOTTOM_MARGIN)
    right = max(10, font.size("00000")[0] // 2) if style.get("text_size") else 10
    return Rect(area.left + left, area.top + 10, area.width - left - right, area.height - bottom - 10)


# True if an area is big enough to draw a chart in
def has_room(area):
    return area.width >= MIN_SIZE[0] and area.height >= MIN_SIZE[1]


# Draws a chart in its style. chart is the chart's own function(painter, area) and settings the
# style from its node, or None. The title, if it has one, goes across the top and the chart gets
# the space below. fill paints the background even when it is the theme's own, as a saved picture
# needs, and margin leaves that much space round the chart. Nothing is drawn in an area too small
def render(painter, area, chart, settings=None, fill=False, margin=0):
    global style
    if not has_room(area):
        return
    style = settings or {}
    try:
        if fill or "background" in style.get("colours", {}):
            painter.rect(colour("background"), area)
            margin = max(margin, BACKGROUND_MARGIN)
        area = area.inflate(-margin * 2, -margin * 2)

        title = style.get("title")
        if title:
            font = sized_font(int(style.get("text_size") or TEXT_SIZE) + TITLE_EXTRA)
            painter.text(font, fit_text(font, title, area.width), colour("text"), midtop=(area.centerx, area.top))
            used = font.get_height() + 6
            area = Rect(area.left, area.top + used, area.width, area.height - used)

        if has_room(area):
            chart(painter, area)
    finally:
        style = {}
        painter.set_clip(None)


# The size in pixels a chart is saved at as a picture: the style's, or else EXPORT_SIZE
def export_size(settings):
    settings = settings or {}
    return int(settings.get("export_width") or EXPORT_SIZE[0]), int(settings.get("export_height") or EXPORT_SIZE[1])


# Returns the lowest and highest value with a little space added either side
def padded_range(values):
    low, high = min(values), max(values)
    if low == high:
        return low - 1, high + 1
    space = (high - low) * 0.05
    return low - space, high + space


# Shows large numbers like years in full, and everything else to 3 significant figures
def tick_text(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.3g}"


# Draws a scatter chart of y against x inside area, with an optional fitted line
# given as (intercept, slope)
def draw_scatter(painter, area, font, x_values, y_values, x_label, y_label, line=None):
    font = text_font(font)
    plot = plot_rect(area, font)
    x_low, x_high = axis_range("x", *padded_range(x_values))
    y_low, y_high = axis_range("y", *padded_range(y_values))

    # Converts a data point to a position on screen
    def to_screen(x, y):
        screen_x = plot.left + (x - x_low) / (x_high - x_low) * plot.width
        screen_y = plot.bottom - (y - y_low) / (y_high - y_low) * plot.height
        return screen_x, screen_y

    draw_axes(painter, plot, font, x_low, x_high, y_low, y_high)

    painter.set_clip(plot)
    for x, y in zip(x_values, y_values):
        painter.circle(colour("point"), to_screen(x, y), point_radius())

    if line:
        intercept, slope = line
        start = to_screen(x_low, intercept + slope * x_low)
        end = to_screen(x_high, intercept + slope * x_high)
        painter.line(colour("line"), start, end, line_width())
    painter.set_clip(None)

    draw_labels(painter, plot, font, x_label, y_label)


# Draws one line per series inside area, each in its own colour, with a key in the top left.
# series is a list of (name, x values, y values), each sorted along x
def draw_lines(painter, area, font, series, x_label):
    font = text_font(font)
    plot = plot_rect(area, font)
    x_low, x_high = axis_range("x", *padded_range([x for name, xs, ys in series for x in xs]))
    y_low, y_high = axis_range("y", *padded_range([y for name, xs, ys in series for y in ys]))

    # Converts a data point to a position on screen
    def to_screen(x, y):
        screen_x = plot.left + (x - x_low) / (x_high - x_low) * plot.width
        screen_y = plot.bottom - (y - y_low) / (y_high - y_low) * plot.height
        return screen_x, screen_y

    draw_axes(painter, plot, font, x_low, x_high, y_low, y_high)

    painter.set_clip(plot)
    for i, (name, xs, ys) in enumerate(series):
        points = [to_screen(x, y) for x, y in zip(xs, ys)]
        if len(points) > 1:
            painter.lines(series_colour(i), points, line_width())
        else:
            painter.circle(series_colour(i), points[0], point_radius())
    draw_key(painter, plot, font, [(name, series_colour(i)) for i, (name, xs, ys) in enumerate(series)])
    painter.set_clip(None)

    draw_labels(painter, plot, font, x_label, "")


# Draws a key in the top left of the plot: a short line in each entry's colour beside its name.
# entries is a list of (name, colour). The style can hide the key
def draw_key(painter, plot, font, entries):
    if not shows("key"):
        return
    row = scaled(font, 20)
    for i, (name, entry_colour) in enumerate(entries):
        y = plot.top + 10 + i * row
        painter.line(entry_colour, (plot.left + 10, y), (plot.left + 30, y), line_width())
        painter.text(font, str(name), colour("text"), midleft=(plot.left + 36, y))


# Draws a correlogram: a bar from zero for each lag's correlation, on a scale from -1 to 1, with
# dashed lines at plus and minus bound, outside which a correlation is significant
def draw_correlogram(painter, area, font, values, bound, label):
    font = text_font(font)
    plot = plot_rect(area, font)
    lags = len(values)
    y_low, y_high = axis_range("y", -1, 1)

    # Converts a lag and a correlation to a position on screen
    def to_screen(lag, value):
        return plot.left + lag / lags * plot.width, plot.bottom - (value - y_low) / (y_high - y_low) * plot.height

    draw_axes(painter, plot, font, 0, lags, y_low, y_high, CORRELOGRAM_TICKS)

    painter.set_clip(plot)
    painter.line(colour("text"), to_screen(0, 0), to_screen(lags, 0))

    for level in [bound, -bound]:
        for start in range(0, int(plot.width), DASH * 2):
            y = to_screen(0, level)[1]
            painter.line(colour("line"), (plot.left + start, y), (plot.left + min(start + DASH, plot.width), y), 2)

    width = max(2, int(plot.width / lags * 0.4))
    for lag, value in enumerate(values, start=1):
        x, top = to_screen(lag, max(value, 0))
        bottom = to_screen(lag, min(value, 0))[1]
        painter.rect(colour("point"), (x - width / 2, top, width, max(1, bottom - top)))
    painter.set_clip(None)

    draw_labels(painter, plot, font, "Lag", label)


# Draws a bar for each category's value, from zero, with the category names under the bars.
# Names that don't fit their slot are shortened, and if the slots are too narrow to read
# only every few names are written
def draw_bars(painter, area, font, names, values, x_label, y_label):
    font = text_font(font)
    plot = plot_rect(area, font)
    # The axis always includes zero, with space added only on the side the bars grow towards
    low, high = padded_range(values)
    y_low = low if min(values) < 0 else 0
    y_high = high if max(values) > 0 else 0
    if y_low == y_high:
        y_high = 1
    y_low, y_high = axis_range("y", y_low, y_high)
    slot = plot.width / len(values)

    # Converts a value to a height on screen
    def to_y(value):
        return plot.bottom - (value - y_low) / (y_high - y_low) * plot.height

    draw_axes(painter, plot, font, 0, 1, y_low, y_high, x_ticks=False)

    painter.set_clip(plot)
    painter.line(colour("text"), (plot.left, to_y(0)), (plot.right, to_y(0)))
    width = max(1, slot * BAR_SHARE)
    for i, value in enumerate(values):
        centre = plot.left + (i + 0.5) * slot
        top, bottom = to_y(max(value, 0)), to_y(min(value, 0))
        painter.rect(colour("point"), (centre - width / 2, top, width, max(1, bottom - top)))
    painter.set_clip(None)

    draw_category_names(painter, plot, font, names)
    draw_labels(painter, plot, font, x_label, y_label)


# Writes a name under each of the equal slots across the bottom of the plot, shortening names
# that don't fit, and writing only every few names if the slots are too narrow to read
def draw_category_names(painter, plot, font, names):
    font = text_font(font)
    slot = plot.width / len(names)
    step = math.ceil(scaled(font, MIN_LABEL_WIDTH) / slot)
    for i, name in enumerate(names):
        if i % step == 0:
            painter.text(font, fit_text(font, str(name), slot * step - 4), colour("text"), midtop=(plot.left + (i + 0.5) * slot, plot.bottom + scaled(font, 12)))


# Draws the plot border, grid lines and evenly spaced tick values on both axes,
# or only up the side when the x axis holds categories rather than numbers
def draw_axes(painter, plot, font, x_low, x_high, y_low, y_high, ticks=TICKS, x_ticks=True):
    font = text_font(font)
    ticks = max(1, int(style.get("ticks") or ticks))
    for i in range(ticks + 1):
        share = i / ticks

        if x_ticks:
            x = plot.left + share * plot.width
            if shows("grid"):
                painter.line(colour("grid"), (x, plot.top), (x, plot.bottom))
            painter.text(font, tick_text(x_low + share * (x_high - x_low)), colour("text"), midtop=(x, plot.bottom + scaled(font, 12)))

        y = plot.bottom - share * plot.height
        if shows("grid"):
            painter.line(colour("grid"), (plot.left, y), (plot.right, y))
        painter.text(font, tick_text(y_low + share * (y_high - y_low)), colour("text"), midright=(plot.left - 6, y))

    painter.rect(colour("border"), plot, width=2)


# Writes the variable names under the x axis and up the side of the y axis, or the labels
# the style gives in their place
def draw_labels(painter, plot, font, x_label, y_label):
    font = text_font(font)
    x_label = style.get("x_label") or x_label or ""
    y_label = style.get("y_label") or y_label or ""

    painter.text(font, str(x_label), colour("text"), midtop=(plot.centerx, plot.bottom + scaled(font, 32)))

    painter.text(font, str(y_label), colour("text"), rotated=True, midright=(plot.left - scaled(font, 44), plot.centery))
