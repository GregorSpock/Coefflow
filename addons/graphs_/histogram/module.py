import math

import numpy as np
import pandas as pd
import pygame
from statsmodels.stats.stattools import jarque_bera

from ui import colours
from ui.chart import BOTTOM_MARGIN, LEFT_MARGIN, draw_axes, draw_labels
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with a numeric column"),
]
OUTPUT_SPECS = [
    ("Bins", "the start, end, count and share of rows of each bin"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["column"]

MIN_BINS = 1
MAX_BINS = 100
CURVE_STEPS = 120


# Opens a panel to pick the variable and number of bins, showing the histogram and a normality note
def on_click(node):
    panel = Panel(node, "Histogram")
    panel.preview_title = "Histogram"
    panel.add_button("Fewer bins", change_bins)
    panel.add_button("More bins", change_bins)
    panel.add_button("Auto bins", change_bins)
    curve = panel.add_button("Normal", toggle_curve)
    curve.selected_colour = "TAB_SELECTED"
    panel.refresh = show_histogram
    show_histogram(panel)
    return panel


# Sets the clicked variable as the one to draw, or clears it if it already was
def choose_column(panel, button):
    settings = panel.node.settings
    settings["column"] = None if settings.get("column") == button.value else button.value
    show_histogram(panel)


# Steps the number of bins down or up from what is drawn now, or goes back to choosing it automatically
def change_bins(panel, button):
    settings = panel.node.settings
    if button.text == "Auto bins":
        settings["bins"] = None
    else:
        current = panel.bins_shown or 10
        step = -1 if button.text == "Fewer bins" else 1
        settings["bins"] = max(MIN_BINS, min(MAX_BINS, current + step))
    show_histogram(panel)


# Turns the fitted normal curve on or off
def toggle_curve(panel, button):
    settings = panel.node.settings
    settings["normal_curve"] = not settings.get("normal_curve", True)
    show_histogram(panel)


# The values of a column with gaps left out. Raises ValueError if there are none
def column_values(data, column):
    values = data[column].dropna().astype(float)
    if values.empty:
        raise ValueError(f"{column} has no values")
    return values


# The bin edges: the number of bins asked for, or numpy's automatic choice if bins is None
def bin_edges(values, bins=None):
    if values.min() == values.max():
        return np.array([values.min() - 0.5, values.max() + 0.5])
    edges = np.histogram_bin_edges(values, bins=bins or "auto")
    if len(edges) - 1 > MAX_BINS:
        edges = np.histogram_bin_edges(values, bins=MAX_BINS)
    return edges


# One row per bin with its edges, how many rows fall in it and what share of rows that is
def bin_table(values, edges):
    counts, edges = np.histogram(values, bins=edges)
    return pd.DataFrame({
        "bin_start": edges[:-1].round(6),
        "bin_end": edges[1:].round(6),
        "count": counts,
        "share": (counts / counts.sum()).round(4),
    })


# Writes the mean, spread, shape and a plain-English Jarque-Bera normality result
def describe_values(name, values):
    text = f"n = {len(values)}, mean = {values.mean():.4g}, sd = {values.std():.4g}. "
    if len(values) < 3:
        return text
    statistic, p_value, skew, kurtosis = jarque_bera(values)
    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    text += f"Skew = {skew:.2f}, kurtosis = {kurtosis:.2f}. Jarque-Bera = {statistic:.1f}, p {p_text}: "
    if p_value < 0.05:
        side = "a long right tail" if skew > 0.5 else "a long left tail" if skew < -0.5 else "heavy or light tails"
        return text + f"{name} is not normally distributed ({side})."
    return text + f"no significant evidence that {name} is not normal at the 5% level."


# Draws the bars inside area, with the normal curve for the values' mean and sd if curve is on
def draw_histogram(screen, area, font, values, table, name, curve):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    x_low, x_high = table["bin_start"].iloc[0], table["bin_end"].iloc[-1]
    width = (x_high - x_low) / len(table)
    mean, sd = values.mean(), values.std()

    # Expected count in a bin of this width under a normal distribution, at x
    def normal_count(x):
        return len(values) * width * math.exp(-((x - mean) / sd) ** 2 / 2) / (sd * math.sqrt(2 * math.pi))

    y_high = table["count"].max()
    show_curve = curve and sd > 0 and len(values) > 1
    if show_curve:
        y_high = max(y_high, normal_count(mean))
    y_high *= 1.08

    # Converts a data point to a position on screen
    def to_screen(x, y):
        return plot.left + (x - x_low) / (x_high - x_low) * plot.width, plot.bottom - y / y_high * plot.height

    draw_axes(screen, plot, font, x_low, x_high, 0, y_high)

    screen.set_clip(plot)
    for start, end, count in zip(table["bin_start"], table["bin_end"], table["count"]):
        left, top = to_screen(start, count)
        right = to_screen(end, 0)[0]
        bar = pygame.Rect(round(left), round(top), max(1, round(right) - round(left)), plot.bottom - round(top))
        pygame.draw.rect(screen, colours.CHART_POINT, bar)
        if bar.width > 3:
            pygame.draw.rect(screen, colours.TOOLBAR, bar, width=1)

    if show_curve:
        points = []
        for step in range(CURVE_STEPS + 1):
            x = x_low + (x_high - x_low) * step / CURVE_STEPS
            points.append(to_screen(x, normal_count(x)))
        pygame.draw.lines(screen, colours.CHART_LINE, False, points, 3)
    screen.set_clip(None)

    draw_labels(screen, plot, font, name, "Count")


# Lists the numeric variables with a button each, then draws the histogram and the normality note
def show_histogram(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.chart = None
    panel.note = ""
    panel.bins_shown = None

    curve = settings.get("normal_curve", True)
    for button in panel.buttons:
        if button.text == "Normal":
            button.selected = curve

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    name = settings.get("column")
    for column in data.select_dtypes("number").columns:
        panel.add_option(column, [("Use", choose_column, column == name)])

    if not name or name not in data:
        panel.lines = ["Variable: none"]
        panel.message = "Choose a numeric variable"
        return

    try:
        values = column_values(data, name)
    except ValueError as error:
        panel.lines = [f"Variable: {name}"]
        panel.message = str(error)
        return

    table = bin_table(values, bin_edges(values, settings.get("bins")))
    panel.bins_shown = len(table)
    chosen = "auto" if not settings.get("bins") else "set"
    panel.lines = [f"Variable: {name}    Bins: {len(table)} ({chosen})    Normal curve: {'on' if curve else 'off'}"]
    panel.note = describe_values(name, values)
    panel.message = ""

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_histogram(screen, area, panel.table_font, values, table, name, curve)

    panel.chart = draw


# Gives out the bin counts for the chosen variable
def run(inputs, settings):
    data = inputs[0]
    name = settings.get("column")
    if not name:
        raise ValueError("Choose a numeric variable")
    if name not in data.columns:
        raise ValueError(f"Column not in the data: {name}")
    values = column_values(data, name)
    return [bin_table(values, bin_edges(values, settings.get("bins")))]
