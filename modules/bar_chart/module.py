import numpy as np
import pandas as pd

from ui.chart import draw_bars
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 0
INPUT_SPECS = [
    ("Data", "a table with a category column, and a numeric column to sum or average (or none to count the rows)"),
]
OUTPUT_SPECS = []

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["category", "value"]
STATS = ["Mean", "Sum", "Median", "Count"]
ORDERS = ["A-Z", "Largest", "Smallest"]
# More bars than this are too thin to read, so only this many are drawn
MAX_BARS = 30


# Opens a panel to choose the category and value fields, how to summarise each category
# and what order to put the bars in, showing the bar chart
def on_click(node):
    panel = Panel(node, "Bar Chart")
    panel.add_button(stat_label(node.settings), cycle_stat)
    panel.add_button(order_label(node.settings), cycle_order)
    panel.preview_title = "Bar Chart"
    panel.refresh = show_chart
    show_chart(panel)
    return panel


def stat_label(settings):
    return f"Bars: {settings.get('stat', STATS[0])}"


def order_label(settings):
    return f"Order: {settings.get('order', ORDERS[0])}"


# Moves on to the next way of summarising each category, going back to the first after the last
def cycle_stat(panel, button):
    settings = panel.node.settings
    current = STATS.index(settings.get("stat", STATS[0]))
    settings["stat"] = STATS[(current + 1) % len(STATS)]
    button.text = stat_label(settings)
    show_chart(panel)


def cycle_order(panel, button):
    settings = panel.node.settings
    current = ORDERS.index(settings.get("order", ORDERS[0]))
    settings["order"] = ORDERS[(current + 1) % len(ORDERS)]
    button.text = order_label(settings)
    show_chart(panel)


# Sets the clicked field as the category or the value, or clears it if it already was.
# A field can't be both, so it leaves the other role if needed
def choose_role(panel, button):
    settings = panel.node.settings
    key = "category" if button.text == "Cat" else "value"
    other = "value" if key == "category" else "category"

    if settings.get(key) == button.value:
        settings[key] = None
    else:
        settings[key] = button.value
        if settings.get(other) == button.value:
            settings[other] = None

    show_chart(panel)


# Summarises the value field for each category, in the chosen order. Count ignores the value field
# and counts the rows in each category. Missing categories are left out, and a category with no
# values gets no bar. Returns the summaries as a Series indexed by category
def summarise(data, settings):
    category, value = settings.get("category"), settings.get("value")
    stat = settings.get("stat", STATS[0])
    rows = data.dropna(subset=[category])
    groups = rows.groupby(rows[category].astype(str), sort=True)

    if stat == "Count":
        bars = groups.size()
    elif stat == "Sum":
        # A plain sum of no values is 0, which would draw a bar for a category with no data
        bars = groups[value].sum(min_count=1).dropna()
    else:
        bars = groups[value].agg(stat.lower()).dropna()

    order = settings.get("order", ORDERS[0])
    if order == "Largest":
        bars = bars.sort_values(ascending=False, kind="stable")
    elif order == "Smallest":
        bars = bars.sort_values(kind="stable")
    return bars


# Lists every field with Cat and Value buttons (only numeric ones can be the value), then draws the chart
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
    category, value = settings.get("category"), settings.get("value")
    stat = settings.get("stat", STATS[0])
    panel.lines = [f"Category: {category or 'none'}    Value: {value or 'none'}"]

    numeric = data.select_dtypes("number").columns
    for column in data.columns:
        choices = [("Cat", choose_role, column == category)]
        if column in numeric:
            choices.append(("Value", choose_role, column == value))
        panel.add_option(column, choices)

    try:
        check_fields(data, settings)
    except ValueError as error:
        panel.message = str(error)
        return

    bars = summarise(data, settings)
    if bars.empty:
        panel.message = f"{value} has no values in any category"
        return
    if len(bars) > MAX_BARS:
        panel.note = f"{category} has {len(bars)} categories, so only the first {MAX_BARS} in this order are drawn."
        bars = bars.iloc[:MAX_BARS]

    y_label = "Rows" if stat == "Count" else f"{stat} of {value}"
    names, heights = list(bars.index), bars.to_numpy(float)

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_bars(screen, area, panel.table_font, names, heights, category, y_label)

    panel.chart = draw


# Checks the category is chosen, and the value too unless the bars count rows
def check_fields(data, settings):
    category, value = settings.get("category"), settings.get("value")
    needs_value = settings.get("stat", STATS[0]) != "Count"
    if not category or (needs_value and not value):
        raise ValueError("Choose a category field" + (" and a value field, or Bars: Count" if needs_value else ""))
    for column in [category, value if needs_value else None]:
        if column and column not in data.columns:
            raise ValueError(f"Column not in the data: {column}")
    if needs_value and not pd.api.types.is_numeric_dtype(data[value]):
        raise ValueError(f"{value} is not numeric")
    if data[category].isna().all():
        raise ValueError(f"{category} has no values")


# Nothing to calculate for the workflow itself; just checks the fields are chosen
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    check_fields(data, settings)
    return []
