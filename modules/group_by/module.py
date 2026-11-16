import pandas as pd

from engine.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table, with one or more fields to group the rows by (G) and fields to summarise"),
]
OUTPUT_SPECS = [
    ("Data", "one row per group with the group fields, a rows column counting the rows in each and a column per statistic, "
             "named like gdp_mean. With Every row on, the statistics are added to every row of the input instead"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["groups", "statistics"]

# The statistics: the button's text, what the column is called after the field, and how to work it out
STATISTICS = {
    "mean": ("mean", "mean"),
    "sum": ("sum", "sum"),
    "median": ("median", "median"),
    "sd": ("sd", "std"),
    "min": ("min", "min"),
    "max": ("max", "max"),
    "n": ("n", "count"),
    "distinct": ("distinct", "nunique"),
    "first": ("first", "first"),
    "last": ("last", "last"),
}
# Statistics that need numbers; the rest work on text too
NUMERIC_ONLY = ["mean", "sum", "median", "sd"]
ROWS = "rows"


# Opens a panel to choose the group fields (G) and the statistics of each other field
def on_click(node):
    panel = Panel(node, "Group By")
    panel.add_button(shape_label(node.settings), toggle_shape)
    panel.add_button("Clear", clear_all)
    panel.options_title = "Group fields (G) and statistics"
    panel.option_columns = 1
    panel.refresh = show_groups
    show_groups(panel)
    return panel


def shape_label(settings):
    return "Every row: On" if settings.get("every_row") else "Every row: Off"


# Switches between one row per group and the statistics added to every row
def toggle_shape(panel, button):
    settings = panel.node.settings
    settings["every_row"] = not settings.get("every_row", False)
    button.text = shape_label(settings)
    show_groups(panel)


def clear_all(panel, button):
    panel.node.settings["groups"] = []
    panel.node.settings["statistics"] = {}
    show_groups(panel)


# Makes the clicked field a group field, or stops it being one
def toggle_group(panel, button):
    groups = panel.node.settings.setdefault("groups", [])
    column = button.value
    if column in groups:
        groups.remove(column)
    else:
        groups.append(column)
        panel.node.settings.get("statistics", {}).pop(column, None)
    show_groups(panel)


# Adds the clicked statistic for the field, or takes it away
def toggle_statistic(panel, button):
    settings = panel.node.settings
    statistics = settings.setdefault("statistics", {})
    column = button.value
    chosen = statistics.setdefault(column, [])
    if button.text in chosen:
        chosen.remove(button.text)
    else:
        chosen.append(button.text)
        settings["groups"] = [c for c in settings.get("groups", []) if c != column]
    if not chosen:
        del statistics[column]
    show_groups(panel)


def show_groups(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    groups = settings.get("groups", [])
    statistics = settings.get("statistics", {})
    panel.lines = [
        f"Group by: {', '.join(groups) or 'nothing (one row for the whole table)'}",
        f"Statistics: {sum(len(v) for v in statistics.values())} chosen",
    ]

    for column in data.columns:
        numeric = pd.api.types.is_numeric_dtype(data[column]) and not pd.api.types.is_bool_dtype(data[column])
        choices = [("G", toggle_group, column in groups)]
        for name in STATISTICS:
            if numeric or name not in NUMERIC_ONLY:
                choices.append((name, toggle_statistic, name in statistics.get(column, [])))
        panel.add_option(column, choices)
        for button in panel.options[-1][1]:
            button.value = column

    try:
        panel.preview = run([data], settings)[0]
        panel.message = ""
    except Exception as error:
        panel.message = f"Could not group: {error}"


# The column a statistic of a field goes in, e.g. gdp_mean
def column_name(column, statistic):
    return f"{column}_{STATISTICS[statistic][0]}"


# Groups the rows by the group fields and works out each chosen statistic of each field, with a
# rows column counting the rows in each group. Missing values in a group field form a group of
# their own, so no rows are lost. With no group fields the whole table is one group. With
# settings["every_row"] the statistics are added as new columns on every row of the input, as
# Stata's egen does, rather than giving one row per group
def run(inputs, settings):
    data = inputs[0]
    groups = [c for c in settings.get("groups", []) if c in data.columns]
    statistics = {
        column: [s for s in chosen if s in STATISTICS]
        for column, chosen in settings.get("statistics", {}).items()
        if column in data.columns and column not in groups
    }

    for column, chosen in statistics.items():
        if not pd.api.types.is_numeric_dtype(data[column]) and any(s in NUMERIC_ONLY for s in chosen):
            raise ValueError(f"{column} is not numeric, so it can't have a mean, sum, median or sd")

    every_row = settings.get("every_row", False)
    if groups:
        grouped = data.groupby(groups, dropna=False, sort=True)
    else:
        grouped = data.groupby(lambda index: 0)

    # The new columns are made first and joined on at once, which keeps a wide result quick
    if every_row:
        added = {ROWS: grouped[data.columns[0]].transform("size")}
        for column, chosen in statistics.items():
            for statistic in chosen:
                added[column_name(column, statistic)] = grouped[column].transform(STATISTICS[statistic][1])
        kept = data.drop(columns=[c for c in added if c in data.columns])
        return [pd.concat([kept, pd.DataFrame(added, index=data.index)], axis=1)]

    added = {ROWS: grouped.size()}
    for column, chosen in statistics.items():
        for statistic in chosen:
            added[column_name(column, statistic)] = grouped[column].agg(STATISTICS[statistic][1])
    result = pd.concat(added, axis=1).reset_index(drop=not groups)
    return [result]


# Only the group fields keep their names on the way out, unless every row is kept
def output_name(settings, name):
    if settings.get("every_row") or name in settings.get("groups", []):
        return name
    return None
