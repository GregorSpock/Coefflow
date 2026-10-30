from engine.fields import drop_downstream
from ui.fields import connections
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the same rows with only the chosen columns, in the order they were chosen"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["columns"]


# Opens a panel to choose which fields to keep and in what order, with a preview of the result
def on_click(node):
    panel = Panel(node, "Select")
    panel.add_button("Keep all", keep_all)
    panel.add_button("Clear", clear_columns)
    panel.refresh = show_select
    show_select(panel)
    return panel


# Keeps the clicked field, putting it after the ones already chosen, or takes it out if it was chosen.
# A field taken out is also taken out of the nodes below, e.g. out of a regression's variables
def toggle_column(panel, button):
    columns = panel.node.settings.setdefault("columns", [])
    column = button.value
    if column in columns:
        columns.remove(column)
        if columns:
            drop_downstream(panel.node, connections(panel), column)
    else:
        columns.append(column)
    show_select(panel)


# Chooses every incoming field, in the order they arrive
def keep_all(panel, button):
    data = incoming(panel)
    if data is not None:
        panel.node.settings["columns"] = list(data.columns)
    show_select(panel)


# Chooses no fields, so the data passes through unchanged
def clear_columns(panel, button):
    panel.node.settings["columns"] = []
    show_select(panel)


def incoming(panel):
    inputs = panel.node.last_inputs
    return inputs[0] if inputs else None


# Lists the fields from the last run with a Keep button for each, numbering the chosen ones
# in the order they come out, and previews the data as it will come out
def show_select(panel):
    columns = panel.node.settings.get("columns", [])
    panel.clear_options()
    panel.preview = None

    data = incoming(panel)
    if data is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    if columns:
        panel.lines = [f"Keeping {len(columns)} of {len(data.columns)} fields, in the order chosen"]
    else:
        panel.lines = ["No fields chosen, so every field passes through. Click Keep to choose fields"]

    for column in data.columns:
        label = f"{columns.index(column) + 1}. {column}" if column in columns else column
        panel.add_option(label, [("Keep", toggle_column, column in columns)])
        panel.options[-1][1][0].value = column

    try:
        panel.preview = run([data], panel.node.settings)[0]
        panel.message = ""
    except ValueError as error:
        panel.message = str(error)


# Keeps only the columns listed in settings["columns"], in that order.
# With no columns set, the data passes through unchanged
def run(inputs, settings):
    data = inputs[0]
    columns = settings.get("columns")

    if not columns:
        return [data]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Column not in the data: {missing[0]}")
    return [data[columns]]


# Fields not in the chosen columns don't come out. With no columns chosen, every field does
def output_name(settings, name):
    columns = settings.get("columns")
    if columns and name not in columns:
        return None
    return name
