import numpy as np
import pandas as pd
from scipy import stats

from engine import chart
from engine.chart import MAX_SERIES, draw_axes, draw_category_names, draw_labels, padded_range
from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with a numeric outcome (Y), a column naming groups for the horizontal axis (G) and optionally a second one to split each group by (By)"),
]
OUTPUT_SPECS = [
    ("Means", "the count, mean, standard deviation, standard error and confidence interval of Y in each group"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "group", "by"]
ROLES = [("Y", "dependent", False, True), ("G", "group", False, False), ("By", "by", False, False)]
STYLES = ["Bars", "Lines"]
# What the error bars reach out to either side of each mean
ERRORS = ["95% CI", "SE", "SD"]
# More groups along the axis than this are too narrow to read
MAX_GROUPS = 12
BAR_SHARE = 0.7
CAP = 5


# Opens a panel to choose the outcome and groups, showing the means with their error bars
def on_click(node):
    panel = common.open_panel(node, "Means Plot", show_plot, "Means Plot")
    panel.chart_axes = "y"
    common.add_choice(panel, "Style", "style", STYLES)
    common.add_choice(panel, "Error", "error", ERRORS)
    show_plot(panel)
    return panel


# Checks the fields are chosen and in the data
def check_columns(data, settings):
    dependent, group, by = settings.get("dependent"), settings.get("group"), settings.get("by")
    if not dependent or not group:
        raise ValueError("Choose a numeric outcome (Y) and a column naming the groups (G)")
    common.check_fields(data, [dependent, group] + ([by] if by else []), numeric=[dependent])
    return dependent, group, by


# The count, mean, spread and both ends of the error bar of Y in each group, or in each
# combination of group and By. Groups with no values are left out
def cell_means(data, dependent, group, by, error):
    keys = [group, by] if by else [group]
    rows = data[[dependent] + keys].dropna()
    cells = []
    for labels, cell in rows.groupby(keys, sort=True):
        values = cell[dependent].to_numpy(float)
        mean, low, high = common.mean_interval(values)
        sd = values.std(ddof=1) if len(values) > 1 else np.nan
        se = sd / np.sqrt(len(values))
        reach = {"95% CI": mean - low, "SE": se, "SD": sd}[error]
        row = {key: common.group_text(label) for key, label in zip(keys, labels)}
        row.update({"n": len(values), "mean": mean, "sd": sd, "se": se, "ci_low": low, "ci_high": high, "error_low": mean - reach, "error_high": mean + reach})
        cells.append(row)
    if not cells:
        raise ValueError(f"No rows have both {dependent} and {group}")

    table = pd.DataFrame(cells)
    if table[group].nunique() > MAX_GROUPS:
        raise ValueError(f"{group} has {table[group].nunique()} groups; at most {MAX_GROUPS} can be drawn")
    if by and table[by].nunique() > MAX_SERIES:
        raise ValueError(f"{by} has {table[by].nunique()} groups; at most {MAX_SERIES} can be drawn")
    return table


# Draws each group's mean inside area, as a bar or as a point joined to its neighbours by a line,
# with an error bar through it. With By there is a bar or line for each of its groups, in
# their own colours with a key
def draw_means(painter, area, font, table, group, by, style, y_label):
    font = chart.text_font(font)
    plot = chart.plot_rect(area, font)
    groups = list(dict.fromkeys(table[group]))
    series = list(dict.fromkeys(table[by])) if by else [None]

    ends = pd.concat([table["mean"], table["error_low"], table["error_high"]]).dropna()
    low, high = padded_range(ends)
    if style == "Bars":
        # Bars start from zero, so the axis includes it
        low, high = min(low, 0) if ends.min() < 0 else 0, max(high, 0) if ends.max() > 0 else 0
    if high <= low:
        high = low + 1
    if by and chart.shows("key"):
        # Room above the tallest mean for the key, so it doesn't sit on top of the bars or lines
        room = min(plot.height / 2, len(series) * chart.scaled(font, 20) + 10)
        high = low + (high - low) * plot.height / (plot.height - room)
    low, high = chart.axis_range("y", low, high)
    slot = plot.width / len(groups)

    def to_y(value):
        return plot.bottom - (value - low) / (high - low) * plot.height

    draw_axes(painter, plot, font, 0, 1, low, high, x_ticks=False)
    painter.set_clip(plot)
    if style == "Bars":
        painter.line(chart.colour("text"), (plot.left, to_y(0)), (plot.right, to_y(0)))

    for s, name in enumerate(series):
        colour = chart.series_colour(s) if by else chart.colour("point")
        rows = table[table[by] == name] if by else table
        points = []
        for index, row in rows.iterrows():
            position = groups.index(row[group])
            if style == "Bars":
                width = slot * BAR_SHARE / len(series)
                centre = plot.left + (position + 0.5) * slot + (s - (len(series) - 1) / 2) * width
                top, bottom = to_y(max(row["mean"], 0)), to_y(min(row["mean"], 0))
                painter.rect(colour, (centre - width / 2 + 1, top, max(1, width - 2), max(1, bottom - top)))
                bar_colour = chart.colour("text")
            else:
                # Lines for different By groups are nudged apart so their error bars don't hide each other
                centre = plot.left + (position + 0.5) * slot + (s - (len(series) - 1) / 2) * 6
                points.append((centre, to_y(row["mean"])))
                bar_colour = colour

            if pd.notna(row["error_low"]):
                top, bottom = to_y(row["error_high"]), to_y(row["error_low"])
                painter.line(bar_colour, (centre, top), (centre, bottom), 2)
                for end in [top, bottom]:
                    painter.line(bar_colour, (centre - CAP, end), (centre + CAP, end), 2)

        if len(points) > 1:
            painter.lines(colour, points, chart.line_width())
        for point in points:
            painter.circle(colour, point, chart.point_radius() + 2)

    if by:
        chart.draw_key(painter, plot, font, [(name, chart.series_colour(s)) for s, name in enumerate(series)])
    painter.set_clip(None)

    draw_category_names(painter, plot, font, groups)
    draw_labels(painter, plot, font, group, y_label)


# Lists the fields with Y, G and By buttons, then draws the means
def show_plot(panel):
    settings = panel.node.settings
    data = common.list_fields(panel, ROLES)
    if data is None:
        return
    panel.lines = [
        f"Outcome (Y): {common.names(settings, 'dependent')}    Groups (G): {common.names(settings, 'group')}    "
        f"Split by: {common.names(settings, 'by', 'nothing')}",
    ]
    error = settings.get("error", ERRORS[0])
    style = settings.get("style", STYLES[0])
    try:
        dependent, group, by = check_columns(data, settings)
        table = cell_means(common.clean(data), dependent, group, by, error)
    except ValueError as problem:
        panel.message = str(problem)
        return

    panel.note = f"Each {'bar' if style == 'Bars' else 'point'} is the mean of {dependent}; the error bars show {describe_error(error)}."
    if not by:
        panel.note += " " + difference_text(common.clean(data), dependent, group)

    # Draws the chart into the space the panel gives it
    def draw(painter, area):
        draw_means(painter, area, panel.table_font, table, group, by, style, f"Mean of {dependent}")

    panel.chart = draw


def describe_error(error):
    return {"95% CI": "its 95% confidence interval", "SE": "one standard error either side", "SD": "one standard deviation either side"}[error]


# A one-way ANOVA across the groups (Welch's t-test for two), as a sentence for the note
def difference_text(data, dependent, group):
    groups = [values for label, values in common.grouped(data, dependent, group) if len(values) > 1]
    if len(groups) < 2:
        return ""
    if len(groups) == 2:
        result = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        return f"Welch's t = {result.statistic:.2f}, {common.p_text(result.pvalue)}."
    result = stats.f_oneway(*groups)
    df = sum(len(values) for values in groups) - len(groups)
    return f"One-way ANOVA: F({len(groups) - 1}, {df}) = {result.statistic:.2f}, {common.p_text(result.pvalue)}."


# Gives out the means the chart shows
def run(inputs, settings):
    data = common.clean(inputs[0])
    dependent, group, by = check_columns(data, settings)
    table = cell_means(data, dependent, group, by, settings.get("error", ERRORS[0]))
    return [table.drop(columns=["error_low", "error_high"])]
