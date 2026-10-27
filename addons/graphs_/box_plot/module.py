import pandas as pd
import pygame
from scipy.stats import kruskal

from ui import colours
from ui.chart import BOTTOM_MARGIN, LEFT_MARGIN, TICKS, draw_labels, padded_range, tick_text
from ui.panel import Panel, fit_text


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with a numeric column, and optionally a column to group the rows by"),
]
OUTPUT_SPECS = [
    ("Summary", "the count, five-number summary, mean and number of outliers of each group"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["value", "group"]

# Only the groups with the most rows get a box, so each box stays wide enough to read
MAX_BOXES = 12
# Whiskers reach the furthest value within this many interquartile ranges of the box; beyond is an outlier
WHISKER_IQR = 1.5
POINT_RADIUS = 3
MEAN_SIZE = 5


# Opens a panel to pick the variable and an optional grouping, showing the boxes and a test note
def on_click(node):
    panel = Panel(node, "Box Plot")
    panel.preview_title = "Box Plot"
    panel.refresh = show_boxes
    show_boxes(panel)
    return panel


# Sets the clicked column as the variable (Y) or the grouping (By), or clears it if it already was.
# A column can't be both, so it leaves the other role if needed
def choose_role(panel, button):
    settings = panel.node.settings
    role = "value" if button.text == "Y" else "group"
    other = "group" if role == "value" else "value"

    if settings.get(role) == button.value:
        settings[role] = None
    else:
        settings[role] = button.value
        if settings.get(other) == button.value:
            settings[other] = None

    show_boxes(panel)


# The values of each group as a list of (label, values), in the groups' own order. Without a grouping
# there is one group of every value. Only the MAX_BOXES biggest groups are kept.
# Returns the groups and how many were left out
def grouped_values(data, value, group=None):
    if not group:
        values = data[value].dropna().astype(float)
        if values.empty:
            raise ValueError(f"{value} has no values")
        return [(value, values)], 0

    pairs = data[[group, value]].dropna()
    if pairs.empty:
        raise ValueError(f"No rows have both {value} and {group}")
    # Biggest groups first, and groups of the same size in their own order, so the same ones are always kept
    sizes = pairs[group].value_counts().sort_index().sort_values(ascending=False, kind="stable")
    kept = sorted(sizes.index[:MAX_BOXES])
    groups = [(str(label), pairs.loc[pairs[group] == label, value].astype(float)) for label in kept]
    return groups, len(sizes) - len(kept)


# The box for one group's values: quartiles, whisker ends, mean and the outliers beyond the whiskers
def box_stats(values):
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    reach = WHISKER_IQR * (q3 - q1)
    inside = values[(values >= q1 - reach) & (values <= q3 + reach)]
    return {
        "count": len(values),
        "min": values.min(),
        "low_whisker": inside.min(),
        "q1": q1,
        "median": median,
        "q3": q3,
        "high_whisker": inside.max(),
        "max": values.max(),
        "mean": values.mean(),
        "outliers": values[(values < q1 - reach) | (values > q3 + reach)],
    }


# One row per group with its count, five-number summary, mean and number of outliers
def summary_table(groups, group_name):
    rows = []
    for label, values in groups:
        stats = box_stats(values)
        rows.append({
            group_name: label,
            "count": stats["count"],
            "min": stats["min"],
            "q1": stats["q1"],
            "median": stats["median"],
            "q3": stats["q3"],
            "max": stats["max"],
            "mean": stats["mean"],
            "outliers": len(stats["outliers"]),
        })
    return pd.DataFrame(rows).round(4)


# Runs a Kruskal-Wallis test of whether the groups come from the same distribution, and writes
# a short plain-English note about it. Groups of one value can't be compared, so they are skipped
def describe_test(value, group, groups):
    tested = [values for label, values in groups if len(values) > 1]
    if len(tested) < 2:
        return "Choose a grouping (By) with at least two groups to compare them."
    try:
        statistic, p_value = kruskal(*tested)
    except ValueError as error:
        return f"Could not compare the groups: {error}"

    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    result = f"Kruskal-Wallis test across {len(tested)} groups: H = {statistic:.1f}, p {p_text}. "
    if p_value < 0.05:
        return result + f"{value} differs significantly between at least two {group} groups at the 5% level."
    return result + f"There is no significant evidence that {value} differs between {group} groups at the 5% level."


# Draws horizontal grid lines and evenly spaced value ticks up the side, and the plot border
def draw_value_axis(screen, plot, font, low, high):
    for i in range(TICKS + 1):
        share = i / TICKS
        y = plot.bottom - share * plot.height
        pygame.draw.line(screen, colours.GRID, (plot.left, y), (plot.right, y))
        text = font.render(tick_text(low + share * (high - low)), True, colours.TEXT)
        screen.blit(text, text.get_rect(midright=(plot.left - 6, y)))
    pygame.draw.rect(screen, colours.TOOLBAR_LINE, plot, width=2)


# Draws one box per group side by side inside area: a box from the lower to the upper quartile with
# a line at the median, whiskers, a diamond at the mean and a circle for each outlier
def draw_boxes(screen, area, font, groups, value, group):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    every_value = pd.concat([values for label, values in groups])
    low, high = padded_range(every_value)
    slot = plot.width / len(groups)
    box_width = max(6, min(80, slot * 0.6))

    def to_y(v):
        return plot.bottom - (v - low) / (high - low) * plot.height

    draw_value_axis(screen, plot, font, low, high)

    for i, (label, values) in enumerate(groups):
        stats = box_stats(values)
        centre = plot.left + slot * (i + 0.5)
        left, right = centre - box_width / 2, centre + box_width / 2

        pygame.draw.line(screen, colours.TEXT, (centre, to_y(stats["low_whisker"])), (centre, to_y(stats["q1"])), 2)
        pygame.draw.line(screen, colours.TEXT, (centre, to_y(stats["q3"])), (centre, to_y(stats["high_whisker"])), 2)
        for end in [stats["low_whisker"], stats["high_whisker"]]:
            pygame.draw.line(screen, colours.TEXT, (centre - box_width / 4, to_y(end)), (centre + box_width / 4, to_y(end)), 2)

        box = pygame.Rect(round(left), round(to_y(stats["q3"])), round(box_width), max(2, round(to_y(stats["q1"]) - to_y(stats["q3"]))))
        pygame.draw.rect(screen, colours.CHART_POINT, box)
        pygame.draw.rect(screen, colours.TEXT, box, width=2)
        pygame.draw.line(screen, colours.TOOLBAR, (left + 2, to_y(stats["median"])), (right - 2, to_y(stats["median"])), 3)

        y = to_y(stats["mean"])
        diamond = [(centre, y - MEAN_SIZE), (centre + MEAN_SIZE, y), (centre, y + MEAN_SIZE), (centre - MEAN_SIZE, y)]
        pygame.draw.polygon(screen, colours.CHART_LINE, diamond)

        for outlier in stats["outliers"]:
            pygame.draw.circle(screen, colours.CHART_LINE, (centre, to_y(outlier)), POINT_RADIUS, width=1)

        text = font.render(fit_text(font, label, slot - 4), True, colours.TEXT)
        screen.blit(text, text.get_rect(midtop=(centre, plot.bottom + 12)))

    draw_labels(screen, plot, font, group or "", value)


# Lists the columns with buttons to choose the variable and grouping, then draws the boxes and test note
def show_boxes(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.chart = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    value = settings.get("value")
    group = settings.get("group")
    panel.lines = [f"Variable: {value or 'none'}    Grouped by: {group or 'nothing'}"]

    data = node.last_inputs[0]
    numeric = list(data.select_dtypes("number").columns)
    for column in data.columns:
        choices = [("Y", choose_role, column == value)] if column in numeric else []
        choices.append(("By", choose_role, column == group))
        panel.add_option(column, choices)

    if not value or value not in data:
        panel.message = "Choose a numeric variable (Y), and optionally a column to group by (By)"
        return
    if group and group not in data:
        group = None

    try:
        groups, left_out = grouped_values(data, value, group)
    except ValueError as error:
        panel.message = str(error)
        return

    panel.message = ""
    if group:
        panel.note = describe_test(value, group, groups)
        if left_out:
            panel.note += f" Showing {MAX_BOXES} of the {MAX_BOXES + left_out} groups, those with the most rows."
    else:
        stats = box_stats(groups[0][1])
        panel.note = (
            f"Median {stats['median']:.4g}, interquartile range {stats['q3'] - stats['q1']:.4g}, "
            f"{len(stats['outliers'])} outliers. Choose a grouping (By) to compare groups."
        )

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_boxes(screen, area, panel.table_font, groups, value, group)

    panel.chart = draw


# Gives out the summary of each group the boxes show
def run(inputs, settings):
    data = inputs[0]
    value = settings.get("value")
    group = settings.get("group")
    if not value:
        raise ValueError("Choose a numeric variable")
    for column in [value, group]:
        if column and column not in data.columns:
            raise ValueError(f"Column not in the data: {column}")
    groups, left_out = grouped_values(data, value, group)
    return [summary_table(groups, group or "variable")]
