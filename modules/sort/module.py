import pandas as pd

from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the same rows, sorted by the chosen fields in the order they were chosen"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["sort"]

# The ways a field can be sorted: alphabetic (A-Z, Z-A) or numerical (0-9, 9-0)
ALPHABETIC = ["A-Z", "Z-A"]
NUMERICAL = ["0-9", "9-0"]
ASCENDING = ["A-Z", "0-9"]


# Opens a panel to choose which fields to sort by and how, with a preview of the sorted data
def on_click(node):
    panel = Panel(node, "Sort")
    panel.add_button("Clear", clear_sort)
    panel.refresh = show_sort
    show_sort(panel)
    return panel


# Sorts by the clicked field in the clicked direction, or stops sorting by it if it already was.
# A field chosen for the first time goes to the end of the order, so it only breaks ties.
# Switching the direction of a field keeps its place in the order
def choose_order(panel, button):
    sort = panel.node.settings.setdefault("sort", {})
    column = button.value
    if sort.get(column) == button.text:
        del sort[column]
    else:
        sort[column] = button.text
    show_sort(panel)


def clear_sort(panel, button):
    panel.node.settings["sort"] = {}
    show_sort(panel)


# Lists the fields from the last run with a button for each way of sorting,
# numbering the chosen ones in the order they are sorted by, and previews the sorted data
def show_sort(panel):
    node = panel.node
    sort = node.settings.get("sort", {})
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    order = ", ".join(f"{column} ({direction})" for column, direction in sort.items())
    panel.lines = [f"Sorting by: {order or 'nothing'}"]

    data = node.last_inputs[0]
    priority = list(sort)
    for column in data.columns:
        label = f"{priority.index(column) + 1}. {column}" if column in sort else column
        panel.add_option(label, [
            (direction, choose_order, sort.get(column) == direction)
            for direction in ALPHABETIC + NUMERICAL
        ])
        for button in panel.options[-1][1]:
            button.value = column

    try:
        panel.preview = run([data], node.settings)[0]
        panel.message = ""
    except Exception as error:
        panel.message = f"Could not sort: {error}"


# Alphabetic sorting compares values as text, ignoring case. Numerical sorting compares them as numbers,
# with values that aren't numbers going last
def sort_key(sort):
    def key(values):
        if sort[values.name] in NUMERICAL:
            return pd.to_numeric(values, errors="coerce")
        return values.astype(str).str.lower().where(values.notna())
    return key


# Sorts the rows by each field in settings["sort"], a dict of field name to direction,
# with later fields breaking ties in earlier ones. Missing values go last.
# Fields not in the data are skipped, so a changed file or input still runs
def run(inputs, settings):
    data = inputs[0]
    sort = {c: d for c, d in settings.get("sort", {}).items() if c in data.columns}

    if not sort:
        return [data]
    return [data.sort_values(
        by=list(sort),
        ascending=[d in ASCENDING for d in sort.values()],
        key=sort_key(sort),
        kind="stable",
        na_position="last",
    )]
