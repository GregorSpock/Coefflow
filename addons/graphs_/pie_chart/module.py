import math

import pandas as pd
import pygame

from ui import colours
from ui.panel import Panel, fit_text


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with a category column, and optionally a numeric column to add up"),
]
OUTPUT_SPECS = [
    ("Shares", "each category with its total and its share of the whole, largest first"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["category", "value"]

# The biggest slices drawn on their own; any more categories are joined into one "Other" slice
MAX_SLICES = 8
LEGEND_WIDTH = 260
SWATCH = 14
LEGEND_ROW = 24


# Opens a panel to pick the category and, optionally, a column to add up, showing the pie
def on_click(node):
    panel = Panel(node, "Pie Chart")
    panel.preview_title = "Pie Chart"
    panel.refresh = show_pie
    show_pie(panel)
    return panel


# Sets the clicked column as the category or the value to add up, or clears it if it already was.
# A column can't be both, so it leaves the other role if needed
def choose_role(panel, button):
    settings = panel.node.settings
    role = "category" if button.text == "Cat" else "value"
    other = "value" if role == "category" else "category"

    if settings.get(role) == button.value:
        settings[role] = None
    else:
        settings[role] = button.value
        if settings.get(other) == button.value:
            settings[other] = None

    show_pie(panel)


# The total for each category, largest first: the sum of value, or the number of rows if there
# is no value column. Rows with no category are left out. Raises ValueError if there's nothing to draw
def category_shares(data, category, value=None):
    if value:
        totals = data.dropna(subset=[category]).groupby(category)[value].sum()
    else:
        totals = data[category].value_counts()

    if (totals < 0).any():
        raise ValueError(f"A pie chart can't show negative totals of {value}")
    totals = totals[totals > 0].sort_values(ascending=False)
    if totals.empty:
        raise ValueError("There is nothing to show: every total is zero")

    return pd.DataFrame({
        category: totals.index.astype(str),
        "total": totals.values,
        "share": (totals.values / totals.sum()).round(4),
    })


# Keeps the biggest slices and joins the rest into one "Other" slice, so the pie stays readable.
# Returns a list of (label, share)
def slices(shares, category):
    rows = list(zip(shares[category], shares["share"]))
    if len(rows) <= MAX_SLICES:
        return rows
    kept = rows[:MAX_SLICES - 1]
    rest = sum(share for label, share in rows[MAX_SLICES - 1:])
    return kept + [(f"Other ({len(rows) - len(kept)})", rest)]


# Distinct colours from the current theme for the slices, used in turn
def slice_colours():
    names = ["CHART_POINT", "CHART_LINE", "NODE_DONE", "NODE_ERROR", "TAB_SELECTED", "NODE_BORDER", "CONNECTION", "SUBTITLE"]
    found = []
    for name in names:
        colour = getattr(colours, name)
        if colour not in found and colour != colours.TOOLBAR:
            found.append(colour)
    return found


# Draws the pie on the left of area and a legend of labels and percentages on the right.
# Slices start at the top and go clockwise, largest first
def draw_pie(screen, area, font, pie_slices):
    legend_width = min(LEGEND_WIDTH, area.width // 2)
    pie_area = pygame.Rect(area.left, area.top, area.width - legend_width, area.height)
    radius = max(10, min(pie_area.width, pie_area.height) // 2 - 10)
    centre = pie_area.center
    palette = slice_colours()

    angle = -math.pi / 2
    for i, (label, share) in enumerate(pie_slices):
        sweep = share * 2 * math.pi
        steps = max(2, int(math.degrees(sweep)))
        points = [centre]
        for step in range(steps + 1):
            a = angle + sweep * step / steps
            points.append((centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a)))
        pygame.draw.polygon(screen, palette[i % len(palette)], points)
        if len(pie_slices) > 1:
            pygame.draw.polygon(screen, colours.TOOLBAR, points, width=2)
        angle += sweep
    pygame.draw.circle(screen, colours.TOOLBAR_LINE, centre, radius, width=2)

    left = pie_area.right + 10
    top = area.centery - len(pie_slices) * LEGEND_ROW // 2
    for i, (label, share) in enumerate(pie_slices):
        y = top + i * LEGEND_ROW
        pygame.draw.rect(screen, palette[i % len(palette)], (left, y, SWATCH, SWATCH))
        percent = f"{share * 100:.1f}%"
        percent_width = font.size(percent)[0]
        text = fit_text(font, label, area.right - left - SWATCH - percent_width - 24)
        screen.blit(font.render(text, True, colours.TEXT), (left + SWATCH + 8, y))
        screen.blit(font.render(percent, True, colours.TEXT), (area.right - percent_width, y))


# Lists the columns with buttons to choose the category and the value, then draws the pie
def show_pie(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.chart = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    category = settings.get("category")
    value = settings.get("value")
    panel.lines = [f"Category: {category or 'none'}    Adds up: {value or 'row count'}"]

    data = node.last_inputs[0]
    numeric = list(data.select_dtypes("number").columns)
    for column in data.columns:
        choices = [("Cat", choose_role, column == category)]
        if column in numeric:
            choices.append(("Sum", choose_role, column == value))
        panel.add_option(column, choices)

    if not category or category not in data:
        panel.message = "Choose a category column (Cat), and optionally a column to add up (Sum)"
        return
    if value and value not in data:
        value = None

    try:
        shares = category_shares(data, category, value)
    except ValueError as error:
        panel.message = str(error)
        return

    pie_slices = slices(shares, category)
    measure = f"total {value}" if value else "number of rows"
    panel.note = f"Each slice is a {category}'s share of the {measure}. {len(shares)} categories"
    if len(shares) > MAX_SLICES:
        panel.note += f"; the smallest {len(shares) - MAX_SLICES + 1} are joined into Other"
    panel.message = ""

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_pie(screen, area, panel.table_font, pie_slices)

    panel.chart = draw


# Gives out the total and share of each category
def run(inputs, settings):
    data = inputs[0]
    category = settings.get("category")
    value = settings.get("value")
    if not category:
        raise ValueError("Choose a category column")
    for column in [category, value]:
        if column and column not in data.columns:
            raise ValueError(f"Column not in the data: {column}")
    return [category_shares(data, category, value)]
