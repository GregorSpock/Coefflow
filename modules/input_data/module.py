from pathlib import Path

import pandas as pd

from ui.dialogs import choose_file
from ui.fields import add_field_buttons, apply_fields, show_fields, stop_renaming
from ui.panel import Panel


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Data", "the chosen CSV or Excel file (or the sample data) without the dropped fields, and with any renamed fields given their new names"),
]
SAMPLE_DATA = Path(__file__).parent / "sample_data.csv"


# Reads a CSV or Excel file, or only its first few rows if rows is set
def read_data(path, rows=None):
    if str(path).lower().endswith(".xlsx"):
        return pd.read_excel(path, nrows=rows)
    return pd.read_csv(path, nrows=rows)


# Opens a panel to select a file and choose which fields to keep and what to call them, with a preview
def on_click(node):
    panel = Panel(node, "Input Data")
    panel.add_button("Select", select_file)
    add_field_buttons(panel)
    panel.show_fields = show_file
    show_file(panel)
    return panel


# Asks for a file and remembers it in the node's settings. A new file starts with every field kept and named as it is
def select_file(panel, button):
    path = choose_file()
    if path:
        panel.node.settings["path"] = path
        panel.node.settings["dropped"] = []
        panel.node.settings["renamed"] = {}
        stop_renaming(panel)
        show_file(panel)


# Shows the chosen file, Keep and Rename buttons for every field, and the first 10 rows as they will come out
def show_file(panel):
    path = panel.node.settings.get("path")
    panel.clear_options()

    if path:
        panel.lines = [f"File: {path}"]
    else:
        path = SAMPLE_DATA
        panel.lines = ["No file selected, using the sample data"]

    try:
        data = read_data(path, rows=10)
    except Exception as error:
        panel.preview = None
        panel.message = f"Could not read the file: {error}"
        return

    show_fields(panel, data)


# Reads the chosen file (or the sample data) without the dropped fields and with the new names
def run(inputs, settings):
    return [apply_fields(read_data(settings.get("path", SAMPLE_DATA)), settings)]
