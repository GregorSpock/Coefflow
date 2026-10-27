import re

from engine import conditions
from ui.fields import add_field_buttons, apply_fields, output_name, show_fields
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the rows that match the condition, without the dropped fields and with any renamed fields given their new names"),
]

# Settings that hold incoming field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dropped", "renamed"]


# Opens a panel to choose which of the incoming fields to keep and what to call them, with a preview
def on_click(node):
    panel = Panel(node, "Filter")
    add_field_buttons(panel)
    panel.show_fields = show_filter
    panel.refresh = show_filter
    show_filter(panel)
    return panel


# Lists the fields that arrived on the last run with Keep and Rename buttons,
# and previews the data as it will come out
def show_filter(panel):
    node = panel.node
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    panel.lines = []
    data = node.last_inputs[0]
    condition = node.settings.get("condition")
    if condition:
        panel.lines.append(f"Condition: {condition}")
        try:
            data = conditions.query(data, condition)
        except Exception as error:
            panel.message = f"Could not apply the condition: {error}"
            return

    show_fields(panel, data)


# Keeps the rows matching settings["condition"], e.g. "year >= 2021", if one is set,
# then drops and renames fields. The condition uses the incoming field names
def run(inputs, settings):
    data = inputs[0]
    condition = settings.get("condition")

    if condition:
        data = conditions.query(data, condition)
    return [apply_fields(data, settings)]


# Renames a field in the condition, where it may be written bare (year) or in backticks (`year`).
# A new name that isn't a plain word goes in backticks so the condition still reads it as a field
def rename_field(settings, old, new):
    condition = settings.get("condition")
    if not condition:
        return
    pattern = f"`{re.escape(old)}`"
    if old.isidentifier():
        pattern += rf"|(?<![\w`.]){re.escape(old)}(?![\w`])"
    replacement = new if new.isidentifier() else f"`{new}`"
    settings["condition"] = re.sub(pattern, lambda match: replacement, condition)
