import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from engine import chart
from ui import dialogs


# The chart settings, opened by the Settings button beside any chart. They change how that
# node's chart looks: its title and axis labels, the ends of each axis, the number of ticks, the
# sizes of its text, points and lines, whether it has grid lines and a key, its colours, and the
# size it is saved at as a PNG. The choices make up the chart's style (see engine/chart.py), kept
# in the node's settings so they are saved with the workflow. A box left empty, and a colour not
# picked, stay automatic. The chart is redrawn as each choice is made

COLUMNS = 4
SWATCH_COLUMNS = 6

# Each box: the style key it sets, its label, what it holds ("text", "number" or "whole" for a
# whole number), its row and column in the grid of boxes, how many columns wide it is, and the
# axis the chart must have for it to apply ("any" for either axis, "" if it always applies)
FIELDS = [
    ("title", "Title", "text", 0, 0, 4, ""),
    ("x_label", "X label", "text", 1, 0, 2, "any"),
    ("y_label", "Y label", "text", 1, 2, 2, "any"),
    ("x_min", "X min", "number", 2, 0, 1, "x"),
    ("x_max", "X max", "number", 2, 1, 1, "x"),
    ("y_min", "Y min", "number", 2, 2, 1, "y"),
    ("y_max", "Y max", "number", 2, 3, 1, "y"),
    ("ticks", "Ticks", "whole", 3, 0, 1, "any"),
    ("text_size", "Text size", "whole", 3, 1, 1, ""),
    ("point_size", "Point size", "whole", 3, 2, 1, ""),
    ("line_width", "Line width", "whole", 3, 3, 1, ""),
    ("export_width", "PNG width", "whole", 4, 0, 1, ""),
    ("export_height", "PNG height", "whole", 4, 1, 1, ""),
]
# The smallest and largest value each whole number can be set to
LIMITS = {
    "ticks": (1, 20),
    "text_size": (8, 60),
    "point_size": (1, 20),
    "line_width": (1, 12),
    "export_width": (200, 6000),
    "export_height": (200, 6000),
}
# The parts of the chart that can be shown or hidden, as buttons after the last boxes
TOGGLES = [("grid", "Grid lines"), ("key", "Key")]
TOGGLE_ROW = 4
TOGGLE_COLUMN = 2

COLOUR_LABELS = [
    ("background", "Background"),
    ("text", "Text"),
    ("grid", "Grid"),
    ("border", "Border"),
    ("point", "Points/bars"),
    ("line", "Lines"),
] + [(name, f"Series {i + 1}") for i, name in enumerate(chart.SERIES_NAMES)]


# The text to show in a box for a value already in the style
def value_text(value):
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


# The value typed into a box, or None if it is empty or can't be used, which leaves it automatic.
# Whole numbers are kept within their limits
def typed_value(key, kind, text):
    text = text.strip()
    if kind == "text":
        return text or None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if kind == "number":
        return number
    low, high = LIMITS[key]
    return max(low, min(high, int(number)))


class Swatch(QPushButton):
    # A square of one of the chart's colours. A click picks another, a right-click goes back to the theme's
    reset = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.reset.emit()
        else:
            super().mousePressEvent(event)


class ChartSettings(QFrame):
    # changed is sent after every choice, so the chart can be drawn again
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("chartSettings")
        self.node = None
        self.boxes = {}
        self.toggles = {}
        self.swatches = {}
        self.rows = QVBoxLayout(self)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(6)

    # Shows the settings of a node's chart. axes names the axes the chart has: "xy", "y" or ""
    def show_for(self, node, axes):
        self.node = node
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout():
                clear_layout(item.layout())
        self.boxes, self.toggles, self.swatches = {}, {}, {}

        def applies(axis):
            return bool(axes) if axis == "any" else axis in axes

        buttons = QHBoxLayout()
        for label, action in [("Print colours", self.use_print_colours), ("Reset", self.reset_style)]:
            button = QPushButton(label)
            button.setObjectName("small")
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(action)
            buttons.addWidget(button)
        buttons.addStretch()
        self.rows.addLayout(buttons)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        rows = sorted({row for key, label, kind, row, column, span, axis in FIELDS if applies(axis)} | {TOGGLE_ROW})
        row_of = {row: i for i, row in enumerate(rows)}
        for key, label, kind, row, column, span, axis in FIELDS:
            if not applies(axis):
                continue
            name = QLabel(label)
            name.setObjectName("small")
            box = QLineEdit()
            box.setPlaceholderText("auto")
            box.textEdited.connect(lambda text, key=key, kind=kind: self.set_value(key, typed_value(key, kind, text)))
            grid.addWidget(name, row_of[row], column * 2)
            grid.addWidget(box, row_of[row], column * 2 + 1, 1, span * 2 - 1)
            self.boxes[key] = box
        for i, (key, label) in enumerate(TOGGLES):
            if key == "grid" and not applies("any"):
                continue
            button = QPushButton()
            button.setObjectName("small")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda checked, key=key: self.set_value(key, None if checked else False))
            grid.addWidget(button, row_of[TOGGLE_ROW], (TOGGLE_COLUMN + i) * 2, 1, 2)
            self.toggles[key] = (button, label)
        for column in range(COLUMNS):
            grid.setColumnStretch(column * 2 + 1, 1)
        self.rows.addLayout(grid)

        colours = QGridLayout()
        colours.setHorizontalSpacing(6)
        colours.setVerticalSpacing(4)
        for i, (name, label) in enumerate(COLOUR_LABELS):
            swatch = Swatch()
            swatch.setFixedSize(20, 20)
            swatch.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            swatch.clicked.connect(lambda checked=False, name=name: self.pick_colour(name))
            swatch.reset.connect(lambda name=name: self.set_colour(name, None))
            text = QLabel(label)
            text.setObjectName("small")
            colours.addWidget(swatch, i // SWATCH_COLUMNS, i % SWATCH_COLUMNS * 2)
            colours.addWidget(text, i // SWATCH_COLUMNS, i % SWATCH_COLUMNS * 2 + 1)
            self.swatches[name] = swatch
        for column in range(SWATCH_COLUMNS):
            colours.setColumnStretch(column * 2 + 1, 1)
        self.rows.addLayout(colours)

        hint = QLabel("Click a colour to change it, right-click it for the theme's. Empty boxes are automatic.")
        hint.setObjectName("hint")
        self.rows.addWidget(hint)
        self.fill_boxes()
        self.refresh()

    # The chart's style: the choices made so far
    def style(self):
        return self.node.settings.get(chart.STYLE_KEY, {})

    # Keeps a choice in the style, or takes it out if value is None. A style with nothing
    # in it is taken out of the node's settings, so an untouched chart leaves no trace
    def set_value(self, key, value):
        style = self.node.settings.setdefault(chart.STYLE_KEY, {})
        if value is None:
            style.pop(key, None)
        else:
            style[key] = value
        if not style:
            del self.node.settings[chart.STYLE_KEY]
        self.refresh()
        self.changed.emit()

    # Sets one colour, or puts it back to the theme's if colour is None
    def set_colour(self, name, colour):
        chosen = dict(self.style().get("colours", {}))
        if colour is None:
            chosen.pop(name, None)
        else:
            chosen[name] = colour
        self.set_value("colours", chosen or None)

    def pick_colour(self, name):
        picked = dialogs.choose_colour(chart.colour_in(self.style(), name))
        if picked:
            self.set_colour(name, picked)

    # Sets every colour to one that suits a white page
    def use_print_colours(self):
        self.set_value("colours", {name: list(colour) for name, colour in chart.PRINT_COLOURS.items()})

    # Forgets every choice, so the chart goes back to its automatic look
    def reset_style(self):
        self.node.settings.pop(chart.STYLE_KEY, None)
        self.fill_boxes()
        self.refresh()
        self.changed.emit()

    # Puts the style's values in the boxes
    def fill_boxes(self):
        style = self.style()
        for key, box in self.boxes.items():
            box.setText(value_text(style.get(key)))

    # Shows the toggles and colours as the style has them. A colour that has been picked
    # gets a bold border, to tell it from the theme's own
    def refresh(self):
        style = self.style()
        for key, (button, label) in self.toggles.items():
            shown = style.get(key, True)
            button.setChecked(shown)
            button.setText(f"{label}: {'On' if shown else 'Off'}")
        picked = style.get("colours", {})
        for name, swatch in self.swatches.items():
            red, green, blue = chart.colour_in(style, name)[:3]
            text = chart.colour_in(style, "text")
            border = "2px solid rgb({}, {}, {})".format(*text[:3]) if name in picked else "1px solid gray"
            swatch.setStyleSheet(f"background: rgb({red}, {green}, {blue}); border: {border}; border-radius: 0; padding: 0;")


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())
