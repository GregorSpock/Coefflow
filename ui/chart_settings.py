import math

import pygame

from ui import chart, colours, dialogs
from ui.button import Button
from ui.panel import PADDING, ROW_HEIGHT, Panel, TextField


# The Chart Settings panel, opened by the Settings button beside any chart. It changes how that
# node's chart looks: its title and axis labels, the ends of each axis, the number of ticks, the
# sizes of its text, points and lines, whether it has grid lines and a key, its colours, and the
# size it is saved at as a PNG. The choices make up the chart's style (see ui/chart.py), kept in
# the node's settings so they are saved with the workflow. A box left empty, and a colour not
# picked, stay automatic. The chart is redrawn underneath as each choice is made

ROW = 34
FIELD_HEIGHT = 28
COLUMNS = 4
LABEL_WIDTH = 88
SWATCH = 20
SWATCH_COLUMNS = 6
SWATCH_ROW = 28

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
# The parts of the chart that can be shown or hidden, drawn as buttons after the last boxes
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


# Goes back to the chart's own panel
def go_back(panel, button):
    panel.source.next_panel = None
    panel.next_panel = panel.source


# Sets every colour to one that suits a white page
def use_print_colours(panel, button):
    panel.set_value("colours", {name: list(colour) for name, colour in chart.PRINT_COLOURS.items()})


# Forgets every choice, so the chart goes back to its automatic look
def reset_style(panel, button):
    panel.node.settings.pop(chart.STYLE_KEY, None)
    for field in panel.fields:
        field.set_text("")


# Shows or hides the part of the chart the clicked button is for
def toggle_part(panel, button):
    shown = panel.style().get(button.value, True)
    panel.set_value(button.value, False if shown else None)


# Running the workflow redraws the chart from the new data, which is the chart panel's job
def refresh_source(panel):
    panel.source.refresh(panel.source)


class ChartSettingsPanel(Panel):
    # source is the chart's own panel, which this one replaces until Back is clicked.
    # A panel has one text field, the one being typed in, so text_field is set to whichever
    # box was clicked last
    def __init__(self, source):
        self.fields = []
        self.toggles = []
        self.swatches = []
        self.hint_top = 0
        self.row_of = {TOGGLE_ROW: 0}
        super().__init__(source.node, "Chart Settings")
        self.source = source
        self.add_button("Back", go_back)
        self.add_button("Print colours", use_print_colours)
        self.add_button("Reset", reset_style)
        if source.refresh:
            self.refresh = refresh_source

        style = self.style()
        for key, label, kind, row, column, span, axis in FIELDS:
            if not self.applies(axis):
                continue
            field = TextField(label, value_text(style.get(key)), self.changer(key, kind), blocked="")
            field.cell = (row, column, span)
            self.fields.append(field)

        for key, label in TOGGLES:
            if key == "grid" and not self.applies("any"):
                continue
            button = Button(0, 0, 0, FIELD_HEIGHT, label, self.table_font, "TAB", "TAB_HOVER", "TAB_SELECTED")
            button.value = key
            button.label = label
            self.toggles.append(button)
            self.actions[button] = toggle_part
        self.visible_buttons = self.toggles

        # Rows whose boxes don't apply to this chart are closed up
        rows = sorted({field.cell[0] for field in self.fields} | {TOGGLE_ROW})
        self.row_of = {row: i for i, row in enumerate(rows)}
        self.layout()

    # True if a setting for the given axis applies to this chart
    def applies(self, axis):
        axes = self.source.chart_axes
        if axis == "any":
            return bool(axes)
        return axis in axes

    # The function a box calls with its text after every edit
    def changer(self, key, kind):
        def change(panel, text):
            panel.set_value(key, typed_value(key, kind, text))
        return change

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

    # Sets one colour, or puts it back to the theme's if colour is None
    def set_colour(self, name, colour):
        chosen = dict(self.style().get("colours", {}))
        if colour is None:
            chosen.pop(name, None)
        else:
            chosen[name] = colour
        self.set_value("colours", chosen or None)

    # Places the boxes in their grid, then the show/hide buttons and the colours below them
    def layout(self):
        super().layout()
        left = self.rect.left + PADDING
        top = self.rect.top + ROW_HEIGHT + PADDING * 3
        cell = (self.rect.width - PADDING * 2) / COLUMNS

        for field in self.fields:
            row, column, span = field.cell
            x = round(left + column * cell)
            y = top + self.row_of.get(row, 0) * ROW
            field.rect = pygame.Rect(x + LABEL_WIDTH, y, round(span * cell) - LABEL_WIDTH - 8, FIELD_HEIGHT)
            field.label_at = (x, field.rect.centery)

        for i, button in enumerate(self.toggles):
            x = round(left + (TOGGLE_COLUMN + i) * cell)
            button.rect = pygame.Rect(x, top + self.row_of[TOGGLE_ROW] * ROW, round(cell) - 8, FIELD_HEIGHT)

        swatch_top = top + len(self.row_of) * ROW + 4
        cell = (self.rect.width - PADDING * 2) / SWATCH_COLUMNS
        self.swatches = []
        for i, (name, label) in enumerate(COLOUR_LABELS):
            x = round(left + i % SWATCH_COLUMNS * cell)
            y = swatch_top + i // SWATCH_COLUMNS * SWATCH_ROW
            self.swatches.append((name, label, pygame.Rect(x, y, SWATCH, SWATCH)))
        rows = (len(COLOUR_LABELS) + SWATCH_COLUMNS - 1) // SWATCH_COLUMNS
        self.hint_top = swatch_top + rows * SWATCH_ROW

    # The name of the colour whose swatch is at a position, or None
    def swatch_at(self, position):
        for name, label, rect in self.swatches:
            if rect.collidepoint(position):
                return name
        return None

    # A click on a box makes it the one typed in. A click on a colour opens the colour picker,
    # and a right-click puts that colour back to the theme's. The rest is handled as in any panel
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            name = self.swatch_at(event.pos)
            if name and event.button == 3:
                self.set_colour(name, None)
                return True
            if name and event.button == 1:
                picked = dialogs.choose_colour(chart.colour_in(self.style(), name))
                if picked:
                    self.set_colour(name, picked)
                return True
            if event.button == 1:
                for field in self.fields:
                    field.focused = False
                    if field.rect.collidepoint(event.pos):
                        self.text_field = field
        return super().handle_event(event)

    # Draws the boxes, buttons and colours, then the chart in its style in the space left below
    def draw(self, screen):
        self.draw_frame(screen)
        style = self.style()

        for field in self.fields:
            label = self.table_font.render(field.label, True, colours.TEXT)
            screen.blit(label, label.get_rect(midleft=field.label_at))
            self.draw_text_box(screen, field)
            if not field.text and not field.focused:
                hint = self.table_font.render("auto", True, colours.SUBTITLE)
                screen.blit(hint, hint.get_rect(midleft=(field.rect.left + 8, field.rect.centery)))

        for button in self.toggles:
            shown = style.get(button.value, True)
            button.text = f"{button.label}: {'On' if shown else 'Off'}"
            button.selected = shown
            button.draw(screen)

        # A colour that has been picked gets a bold border, to tell it from the theme's own
        picked = style.get("colours", {})
        for name, label, rect in self.swatches:
            pygame.draw.rect(screen, chart.colour_in(style, name), rect)
            if name in picked:
                pygame.draw.rect(screen, colours.TEXT, rect, width=2)
            else:
                pygame.draw.rect(screen, colours.TOOLBAR_LINE, rect, width=1)
            text = self.table_font.render(label, True, colours.TEXT)
            screen.blit(text, text.get_rect(midleft=(rect.right + 6, rect.centery)))

        hint = "Click a colour to change it, right-click it for the theme's. Empty boxes are automatic."
        screen.blit(self.table_font.render(hint, True, colours.SUBTITLE), (self.rect.left + PADDING, self.hint_top))

        heading_top = self.hint_top + 20 + PADDING
        y = self.draw_heading(screen, "Preview", heading_top)
        self.chart = self.source.chart
        if self.chart:
            area = pygame.Rect(self.rect.left + PADDING, y, self.rect.width - PADDING * 2, self.rect.bottom - PADDING - y)
            chart.render(screen, area, self.chart, style)
            self.draw_chart_buttons(screen, self.chart_buttons[1:], heading_top)
            if not chart.has_room(area):
                self.draw_no_room(screen, y)
        else:
            self.shown_chart_buttons = []
            message = self.source.message or "Go back and choose what to draw"
            screen.blit(self.font.render(message, True, colours.TEXT), (self.rect.left + PADDING, y))
