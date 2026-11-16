from pathlib import Path

from engine import file_formats
from engine.dialogs import choose_file
from engine.field_panel import add_field_buttons, apply_fields, show_fields, stop_renaming
from engine.panel import Panel


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Data", "the chosen file (or the sample data) without the dropped fields, and with any renamed fields given their new names. "
             "Reads CSV and text, Excel and OpenDocument, Stata, SPSS, SAS, Parquet, Feather, JSON, gretl and SQLite files"),
]
SAMPLE_DATA = Path(__file__).parent / "sample_data.csv"


# Reads a data file of any kind Coefflow knows (see engine/file_formats.py), or only its first
# few rows if rows is set. table is the sheet of a workbook or the table of a database to read
def read_data(path, rows=None, table=None):
    return file_formats.read_table(path, rows=rows, table=table)


# Opens a panel to select a file and choose which fields to keep and what to call them, with a preview
def on_click(node):
    panel = Panel(node, "Input Data")
    panel.add_button("Select", select_file)
    add_field_buttons(panel)
    panel.show_fields = show_file
    show_file(panel)
    return panel


# Asks for a file and remembers it in the node's settings. A new file starts with every field kept
# and named as it is, reading its first sheet or table
def select_file(panel, button):
    path = choose_file()
    if path:
        settings = panel.node.settings
        settings["path"] = path
        settings["dropped"] = []
        settings["renamed"] = {}
        settings.pop("table", None)
        stop_renaming(panel)
        # Opened again, so the Sheet or Table button matches the new file
        panel.next_panel = on_click(panel.node)


# Moves on to the next sheet of a workbook or table of a database
def next_table(panel, button):
    settings = panel.node.settings
    names = panel.tables
    current = settings.get("table")
    index = names.index(current) if current in names else 0
    settings["table"] = names[(index + 1) % len(names)]
    settings["dropped"] = []
    settings["renamed"] = {}
    stop_renaming(panel)
    button.text = table_label(panel, settings["table"])
    show_file(panel)


def table_label(panel, name):
    kind = "Table" if file_formats.ending_of(panel.node.settings.get("path", "")) in file_formats.SQLITE_ENDINGS else "Sheet"
    return f"{kind}: {name}"


# Adds the Sheet or Table button when the file holds more than one
def add_table_button(panel, names):
    panel.tables = names
    if len(names) > 1 and not any(button.text.startswith(("Sheet:", "Table:")) for button in panel.buttons):
        current = panel.node.settings.get("table")
        panel.add_button(table_label(panel, current if current in names else names[0]), next_table)


# Shows the chosen file, Keep and Rename buttons for every field, and the first 10 rows as they will come out
def show_file(panel):
    settings = panel.node.settings
    path = settings.get("path")
    panel.clear_options()

    if path:
        panel.lines = [f"File: {path}"]
    else:
        path = SAMPLE_DATA
        panel.lines = ["No file selected, using the sample data"]

    try:
        names = file_formats.tables_in(path)
        add_table_button(panel, names)
        if len(names) > 1:
            panel.lines.append(f"{len(names)} sheets or tables: click the Sheet or Table button to move to the next")
        data = read_data(path, rows=10, table=settings.get("table"))
    except Exception as error:
        panel.preview = None
        panel.message = f"Could not read the file: {error}"
        return

    show_fields(panel, data)


# Reads the chosen file (or the sample data) without the dropped fields and with the new names
def run(inputs, settings):
    return [apply_fields(read_data(settings.get("path", SAMPLE_DATA), table=settings.get("table")), settings)]
