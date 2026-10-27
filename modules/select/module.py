from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the same rows with only the chosen columns"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["columns"]


# Opens a panel showing what the module takes in and gives out. It has no settings yet
def on_click(node):
    panel = Panel(node, "Select")
    panel.preview_title = "Settings"
    panel.message = "This module has no settings yet"
    return panel


# Keeps only the columns listed in settings["columns"].
# With no columns set, the data passes through unchanged
def run(inputs, settings):
    data = inputs[0]
    columns = settings.get("columns")

    if not columns:
        return [data]
    return [data[columns]]


# Fields not in the chosen columns don't come out. With no columns chosen, every field does
def output_name(settings, name):
    columns = settings.get("columns")
    if columns and name not in columns:
        return None
    return name
