import numpy as np

from engine import conditions
from modules.filter.module import rename_field
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the chosen rows, or the chosen fields and the fields kept in both"),
    ("Data", "the other rows, or the other fields and the fields kept in both"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream.
# Field names inside settings["condition"] are renamed by rename_field, shared with Filter
FIELD_SETTINGS = ["fields", "keys"]

# Splitting by Rows sends some rows to Out 1 and the rest to Out 2.
# Splitting by Fields sends the chosen fields to Out 1 and the rest to Out 2
SPLIT_BY = ["Rows", "Fields"]
DEFAULT_SPLIT_BY = "Rows"

# How the rows for Out 1 are picked: a percentage of them, a number of them, or those matching a condition.
# Share and Count take rows from the top unless Shuffle is on
ROW_MODES = ["Share", "Count", "Condition"]
DEFAULT_ROW_MODE = "Share"
DEFAULT_AMOUNT = {"Share": "80", "Count": "100"}

# Shuffle always picks rows the same way, so a workflow gives the same split every time it runs
SHUFFLE_SEED = 1


# Opens a panel to choose how to split the data, with a preview of either output
def on_click(node):
    panel = Panel(node, "Split")
    panel.shown_output = 0
    panel.refresh = show_split
    show_split(panel)
    return panel


def choose_split_by(panel, button):
    panel.node.settings["by"] = button.text
    show_split(panel)


def choose_row_mode(panel, button):
    panel.node.settings["rows"] = button.text
    show_split(panel)


def toggle_shuffle(panel, button):
    settings = panel.node.settings
    settings["shuffle"] = not settings.get("shuffle", False)
    show_split(panel)


def choose_preview(panel, button):
    panel.shown_output = int(button.text[-1]) - 1
    show_split(panel)


# Sends the clicked field to Out 1 or keeps it in both, or puts it back to Out 2 if it already was
def choose_field(panel, button):
    settings = panel.node.settings
    column = button.value
    key = "fields" if button.text == "Out 1" else "keys"
    other = "keys" if key == "fields" else "fields"
    chosen = settings.setdefault(key, [])
    if column in chosen:
        chosen.remove(column)
    else:
        chosen.append(column)
        if column in settings.get(other, []):
            settings[other].remove(column)
    show_split(panel)


# Keeps the typed amount or condition up to date as the user types
def change_text(panel, text):
    settings = panel.node.settings
    if row_mode(settings) == "Condition":
        settings["condition"] = text
    else:
        settings["amount"] = text
    show_split(panel)


def split_by(settings):
    return settings.get("by", DEFAULT_SPLIT_BY)


def row_mode(settings):
    return settings.get("rows", DEFAULT_ROW_MODE)


# Shows the box for the amount or condition while splitting by rows, keeping it as is
# if it is already showing for the same thing, so typing isn't interrupted
def show_text_field(panel):
    settings = panel.node.settings
    if split_by(settings) != "Rows":
        panel.text_field = None
        return

    mode = row_mode(settings)
    label = {"Share": "Percent to Out 1:", "Count": "Rows to Out 1:", "Condition": "Condition:"}[mode]
    if panel.text_field and panel.text_field.label == label:
        return
    if mode == "Condition":
        text = settings.get("condition", "")
    else:
        text = settings.get("amount", DEFAULT_AMOUNT[mode])
    panel.add_text_field(label, text, change_text, blocked="", fill=True)


# Shows the choices for the chosen kind of split, with a preview of Out 1 or Out 2
def show_split(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    show_text_field(panel)

    by = split_by(settings)
    panel.add_option("Split by", [(b, choose_split_by, b == by) for b in SPLIT_BY])
    if by == "Rows":
        mode = row_mode(settings)
        panel.add_option("Rows to Out 1", [(m, choose_row_mode, m == mode) for m in ROW_MODES])
        if mode != "Condition":
            shuffle = settings.get("shuffle", False)
            panel.add_option("Shuffle", [("On" if shuffle else "Off", toggle_shuffle, shuffle)])

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    shown = panel.shown_output
    panel.add_option("Preview", [(f"Out {i + 1}", choose_preview, i == shown) for i in range(OUTPUTS)])
    if by == "Fields":
        fields, keys = settings.get("fields", []), settings.get("keys", [])
        for column in data.columns:
            panel.add_option(column, [("Out 1", choose_field, column in fields), ("Both", choose_field, column in keys)])

    try:
        outputs = run([data], settings)
    except Exception as error:
        panel.lines = []
        panel.message = f"Could not split: {error}"
        return

    if by == "Rows":
        panel.lines = [f"Rows in: {len(data)}    Out 1: {len(outputs[0])}    Out 2: {len(outputs[1])}"]
    else:
        panel.lines = [f"Fields in: {len(data.columns)}    Out 1: {len(outputs[0].columns)}    Out 2: {len(outputs[1].columns)}"]
    panel.preview = outputs[shown]
    panel.message = ""


# Reads the amount typed for Share (a percentage) or Count (a number of rows), as a number of rows
def rows_wanted(settings, total):
    mode = row_mode(settings)
    text = str(settings.get("amount", DEFAULT_AMOUNT[mode])).strip().rstrip("%")
    try:
        amount = float(text)
    except ValueError:
        raise ValueError(f"{mode} needs a number, not '{text}'")
    if amount < 0 or (mode == "Share" and amount > 100):
        raise ValueError("Share needs a percentage from 0 to 100" if mode == "Share" else "Count can't be negative")
    if mode == "Share":
        return round(total * amount / 100)
    if amount != int(amount):
        raise ValueError("Count needs a whole number of rows")
    return min(int(amount), total)


# Rows: Out 1 gets the rows matching settings["condition"], or settings["amount"] of them taken from
# the top (or at random with settings["shuffle"]), and Out 2 gets the rest. Both keep the original row order.
# Fields: Out 1 gets settings["fields"], Out 2 gets every other field, and settings["keys"] go to both.
# Chosen fields missing from the data are skipped, so a changed file or input still runs
def run(inputs, settings):
    data = inputs[0]

    if split_by(settings) == "Fields":
        keys = [column for column in data.columns if column in settings.get("keys", [])]
        fields = [column for column in data.columns if column in settings.get("fields", []) and column not in keys]
        rest = [column for column in data.columns if column not in fields and column not in keys]
        first = [column for column in data.columns if column in keys or column in fields]
        others = [column for column in data.columns if column in keys or column in rest]
        return [data[first], data[others]]

    if row_mode(settings) == "Condition":
        condition = settings.get("condition", "").strip()
        if not condition:
            raise ValueError("Type a condition, e.g. year >= 2021")
        chosen = conditions.evaluate(data, condition)
        if getattr(chosen, "dtype", None) != bool:
            raise ValueError("The condition needs to be true or false for each row, e.g. year >= 2021")
        chosen = chosen.to_numpy()
    else:
        count = rows_wanted(settings, len(data))
        order = np.arange(len(data))
        if settings.get("shuffle"):
            order = np.random.default_rng(SHUFFLE_SEED).permutation(len(data))
        chosen = np.zeros(len(data), dtype=bool)
        chosen[order[:count]] = True

    return [data[chosen], data[~chosen]]
