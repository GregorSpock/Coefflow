import numpy as np
import pandas as pd

from ui.chart import MAX_SERIES, draw_lines
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 0
INPUT_SPECS = [
    ("Data", "a table with a numeric field for the X axis, such as year, and one or more numeric fields to draw as lines"),
]
OUTPUT_SPECS = []

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["x", "y"]


# Opens a panel to pick the X field and the fields to draw as lines, showing the chart
def on_click(node):
    panel = Panel(node, "Line Chart")
    panel.preview_title = "Line Chart"
    panel.refresh = show_chart
    show_chart(panel)
    return panel


# Sets the clicked field as X, or clears it if it already was. The X field can't also be a line
def choose_x(panel, button):
    settings = panel.node.settings
    lines = settings.setdefault("y", [])

    if settings.get("x") == button.value:
        settings["x"] = None
    else:
        settings["x"] = button.value
        if button.value in lines:
            lines.remove(button.value)

    show_chart(panel)


# Adds the clicked field as a line, or takes it off if it already was one
def choose_line(panel, button):
    settings = panel.node.settings
    lines = settings.setdefault("y", [])

    if button.value in lines:
        lines.remove(button.value)
    elif len(lines) < MAX_SERIES:
        lines.append(button.value)
        if settings.get("x") == button.value:
            settings["x"] = None
    else:
        panel.message = f"At most {MAX_SERIES} lines can be drawn"
        return

    show_chart(panel)


# Lists the numeric fields with X and Line buttons, then draws one line per chosen field
def show_chart(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.chart = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0].replace([np.inf, -np.inf], np.nan)
    x_name = settings.get("x")
    lines = settings.get("y", [])
    panel.lines = [f"X axis: {x_name or 'none'}    Lines: {', '.join(lines) or 'none'}"]

    for column in data.select_dtypes("number").columns:
        panel.add_option(column, [
            ("X", choose_x, column == x_name),
            ("Line", choose_line, column in lines),
        ])

    try:
        check_fields(data, settings)
    except ValueError as error:
        panel.message = str(error)
        return

    series = chart_series(data, settings)
    if not series:
        panel.message = "None of the chosen lines have any values"
        return

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_lines(screen, area, panel.table_font, series, x_name)

    panel.message = ""
    panel.chart = draw


# The points of each line as (name, x values, y values), sorted along X, leaving out rows
# where either value is missing. A line with no points is left out
def chart_series(data, settings):
    x_name = settings["x"]
    series = []
    for name in settings["y"]:
        pair = data[[x_name, name]].dropna().sort_values(x_name)
        if not pair.empty:
            series.append((name, pair[x_name].to_numpy(float), pair[name].to_numpy(float)))
    return series


# Checks an X field and at least one line are chosen, and that they are numeric fields in the data
def check_fields(data, settings):
    x_name = settings.get("x")
    lines = settings.get("y") or []
    if not x_name or not lines:
        raise ValueError("Choose an X field and at least one Line")
    for column in [x_name] + lines:
        if column not in data.columns:
            raise ValueError(f"Column not in the data: {column}")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} is not numeric")


# Nothing to calculate for the workflow itself; just checks the fields are chosen
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    check_fields(data, settings)
    return []
