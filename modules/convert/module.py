import pandas as pd

from engine.fields import drop_downstream, rename_downstream
from ui.fields import connections
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the same rows, with each chosen text field replaced by numbers, and its original text next to it where Label is on"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["convert"]

# The ways a text field can be turned into numbers:
#   Codes   each value gets a whole number by its place in an order the user sets, lowest first,
#           e.g. GCSE 1, A level 2, Degree 3. Suits categories with a natural order
#   Number  the text is read as a number, ignoring commas, spaces, currency signs and %,
#           e.g. "1,250" or "£3.50". Suits numbers that arrived as text
# settings["convert"] maps a field to {"mode": "Codes", "order": [values]} or {"mode": "Number"},
# plus "label": a name if the original text is kept as a new field next to it, e.g. education_label.
# settings["start"] is the code given to the first value in an order (1 unless changed)
MODES = ["Codes", "Number"]
STARTS = [1, 0]
STRIP = r"[,\s£$€%]"


# True if a field holds text, so it can be converted
def is_text(values):
    return pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values) \
        or isinstance(values.dtype, pd.CategoricalDtype)


# The text fields of a table
def text_columns(data):
    return [c for c in data.columns if is_text(data[c])]


# A field's distinct values, as text, in the order they first appear
def values_of(values):
    return list(dict.fromkeys(values.dropna().astype(str)))


# The order a field's values are coded in: the saved order, with any values it doesn't have yet
# (e.g. after the input changed) added at the end in A-Z order. Saved values no longer in the data
# stay in the settings, so they keep their place if they come back, but get no code
def full_order(saved, values):
    missing = sorted(v for v in values if v not in saved)
    return list(saved) + missing


# Reads text as numbers. Returns the numbers and how many non-missing values could not be read
def to_number(values):
    text = values.astype(str).str.replace(STRIP, "", regex=True).where(values.notna())
    numbers = pd.to_numeric(text, errors="coerce")
    return numbers, int((numbers.isna() & values.notna()).sum())


def default_label(column):
    return f"{column}_label"


# Replaces each chosen text field with numbers, adding a field holding its original text straight
# after it where Label is on. Fields not in the data are skipped, so a changed file or input still runs.
# Returns the data and notes on anything worth checking
def convert(data, settings):
    start = settings.get("start", 1)
    original = data
    data = data.copy()
    notes = []
    for column, choice in settings.get("convert", {}).items():
        if column not in data.columns:
            continue
        values = original[column]
        if choice.get("mode") == "Number":
            data[column], failed = to_number(values)
            if failed:
                notes.append(f"{failed} values of {column} are not numbers, so they became missing.")
        else:
            present = values_of(values)
            order = [v for v in full_order(choice.get("order", []), present) if v in present]
            added = [v for v in order if v not in choice.get("order", [])]
            if added:
                notes.append(f"{column} has values with no place in its order yet, put last: {', '.join(added)}.")
            codes = {value: start + i for i, value in enumerate(order)}
            data[column] = values.astype(str).map(codes).where(values.notna()).astype(float)

        label = choice.get("label")
        if label:
            if label in data.columns:
                raise ValueError(f"There is already a field called {label}, so {column}'s label needs another name")
            data.insert(data.columns.get_loc(column) + 1, label, values)
    return data, notes


def run(inputs, settings):
    return [convert(inputs[0], settings)[0]]


# Opens a panel listing the text fields, each with a button for each way of converting it,
# a Label button to keep its original text, and an Edit button to set its order and label name
def on_click(node):
    panel = Panel(node, "Convert")
    panel.add_button(start_label(node.settings), toggle_start)
    panel.add_button("Clear", clear_all)
    panel.refresh = show_convert
    show_convert(panel)
    return panel


def start_label(settings):
    return f"Start at: {settings.get('start', 1)}"


# Switches whether codes start at 1 or 0
def toggle_start(panel, button):
    settings = panel.node.settings
    settings["start"] = STARTS[(STARTS.index(settings.get("start", 1)) + 1) % len(STARTS)]
    button.text = start_label(settings)
    panel.refresh(panel)


# Stops converting every field, taking their labels out of the nodes below
def clear_all(panel, button):
    for choice in panel.node.settings.get("convert", {}).values():
        if choice.get("label"):
            drop_downstream(panel.node, connections(panel), choice["label"])
    panel.node.settings["convert"] = {}
    panel.refresh(panel)


# Starts converting a field to codes, in A-Z order unless it already has an order. Returns its settings
def start_codes(panel, column):
    chosen = panel.node.settings.setdefault("convert", {})
    values = values_of(panel.node.last_inputs[0][column])
    choice = chosen.setdefault(column, {})
    choice["mode"] = "Codes"
    choice["order"] = full_order(choice.get("order", []), values)
    return choice


# Converts the clicked field the clicked way, or stops converting it if it already was.
# A field no longer converted loses its label too, which is taken out of the nodes below
def choose_mode(panel, button):
    chosen = panel.node.settings.setdefault("convert", {})
    column = button.value
    if chosen.get(column, {}).get("mode") == button.text:
        label = chosen.pop(column).get("label")
        if label:
            drop_downstream(panel.node, connections(panel), label)
    elif button.text == "Codes":
        start_codes(panel, column)
    else:
        chosen.setdefault(column, {})["mode"] = "Number"
    panel.refresh(panel)


# Keeps or stops keeping a field's original text as a new field, named like education_label.
# A field that isn't converted yet is converted to codes. Works from the field list, where the
# button holds the field, and from a field's edit panel
def toggle_label(panel, button):
    column = getattr(panel, "column", None) or button.value
    chosen = panel.node.settings.setdefault("convert", {})
    choice = chosen.get(column) or start_codes(panel, column)
    if choice.get("label"):
        drop_downstream(panel.node, connections(panel), choice.pop("label"))
    else:
        choice["label"] = default_label(column)
    panel.refresh(panel)


# Opens the edit panel for the clicked field, converting it to codes if it wasn't converted yet
def open_edit(panel, button):
    column = button.value
    if column not in panel.node.settings.get("convert", {}):
        start_codes(panel, column)
    panel.next_panel = edit_panel(panel.node, column)


# How a field is shown in the list: its name, how many values it has and what it becomes
def field_label(column, values, choice):
    count = len(values_of(values))
    if not choice:
        return f"{column} ({count} values)"
    label = f" + {choice['label']}" if choice.get("label") else ""
    return f"{column} ({count} values) -> {choice['mode'].lower()}{label}"


# Lists the text fields from the last run with Codes, Number, Label and Edit buttons,
# and previews the data as it will come out
def show_convert(panel):
    node = panel.node
    chosen = node.settings.get("convert", {})
    panel.clear_options()
    panel.preview = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    columns = text_columns(data)
    converting = [c for c in chosen if c in data.columns]
    panel.lines = [
        f"Converting: {', '.join(converting) or 'nothing'}",
        "Codes numbers values by an order you set. Number reads text like 1,250 or 45%",
        "Label keeps the original text as a new field. Edit sets the order and the label's name",
    ]

    for column in columns:
        choice = chosen.get(column, {})
        panel.add_option(field_label(column, data[column], chosen.get(column)), [
            ("Codes", choose_mode, choice.get("mode") == "Codes"),
            ("Number", choose_mode, choice.get("mode") == "Number"),
            ("Label", toggle_label, bool(choice.get("label"))),
            ("Edit", open_edit, False),
        ])
        for button in panel.options[-1][1]:
            button.value = column

    if not columns and not converting:
        panel.message = "The data has no text fields to convert"
        return
    try:
        panel.preview, notes = convert(data, node.settings)
        panel.note = " ".join(notes)
        panel.message = ""
    except Exception as error:
        panel.message = f"Could not convert: {error}"


# The panel for one field: Label switches keeping its original text on and off, with a box to name
# that field, and for codes each value has buttons to move it up or down one place, or to either end.
# Back returns to the field list
def edit_panel(node, column):
    panel = Panel(node, f"Convert: {column}")
    panel.column = column
    panel.add_button("Back", go_back)
    panel.label_button = panel.add_button("", toggle_label)
    if node.settings["convert"][column]["mode"] == "Codes":
        panel.add_button("A-Z", sort_order)
        panel.add_button("Reverse", reverse_order)
    panel.preview_title = "Codes"
    panel.refresh = show_edit
    show_edit(panel)
    return panel


def go_back(panel, button):
    panel.next_panel = panel.node.on_click()


def label_button_text(choice):
    return "Label: On" if choice.get("label") else "Label: Off"


# Shows the label name box while Label is on, keeping what is typed if it is already showing
def show_label_field(panel, choice):
    label = choice.get("label")
    if not label:
        panel.text_field = None
        return
    if panel.text_field is None:
        panel.downstream_name = label
        panel.add_text_field("Label field name:", label, rename_label, blocked="")


# Keeps the label's name up to date as the user types, and the nodes below with it.
# An empty name falls back to the default. While the name matches another field's, the nodes
# below are left alone so the two fields don't get mixed up
def rename_label(panel, text):
    settings = panel.node.settings
    choice = settings["convert"][panel.column]
    choice["label"] = text.strip() or default_label(panel.column)
    others = set(panel.node.last_inputs[0].columns) | {
        c.get("label") for field, c in settings["convert"].items() if field != panel.column
    }
    if choice["label"] != panel.downstream_name and choice["label"] not in others:
        rename_downstream(panel.node, connections(panel), {panel.downstream_name: choice["label"]})
        panel.downstream_name = choice["label"]
    panel.refresh(panel)


def order_of(panel):
    return panel.node.settings["convert"][panel.column]["order"]


def sort_order(panel, button):
    order_of(panel).sort(key=str.lower)
    panel.refresh(panel)


def reverse_order(panel, button):
    order_of(panel).reverse()
    panel.refresh(panel)


# Moves the clicked value: "^" and "v" one place up or down, "Top" and "End" to either end
def move_value(panel, button):
    order = order_of(panel)
    position = order.index(button.value)
    value = order.pop(position)
    targets = {"^": max(0, position - 1), "v": position + 1, "Top": 0, "End": len(order)}
    order.insert(targets[button.text], value)
    panel.refresh(panel)


# Shows the field's label setting, then for codes lists its values in code order with buttons to
# move them and a table of each value's code and how many rows have it (saved values not in the
# current data are left out). For numbers, a table of each value and the number it becomes
def show_edit(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.note = ""
    panel.message = ""

    choice = settings.get("convert", {}).get(panel.column)
    if not node.last_inputs or node.last_inputs[0] is None or panel.column not in node.last_inputs[0].columns or not choice:
        panel.lines = [f"{panel.column} is not converted or not in the data. Go back and run the workflow"]
        panel.preview = None
        panel.text_field = None
        return

    panel.label_button.text = label_button_text(choice)
    show_label_field(panel, choice)
    label = choice.get("label")
    label_line = f"Original text kept as {label}" if label else "Original text not kept. Label: On keeps it as a new field"
    if label and label in node.last_inputs[0].columns:
        panel.note = f"There is already a field called {label}, so choose another name."

    values = node.last_inputs[0][panel.column]
    if choice["mode"] == "Number":
        numbers, failed = to_number(values)
        panel.lines = [f"{panel.column}: read as numbers", label_line]
        if failed:
            panel.lines.append(f"{failed} values are not numbers, so they become missing")
        panel.preview_title = "Numbers"
        panel.preview = pd.DataFrame({"value": values, "number": numbers}).drop_duplicates()
        return

    present = values_of(values)
    # Saved values not in the data go to the end, so moving a value never jumps over a hidden one
    saved = full_order(choice.get("order", []), present)
    order = [v for v in saved if v in present]
    choice["order"] = order + [v for v in saved if v not in present]

    start = settings.get("start", 1)
    panel.lines = [
        f"{panel.column}: lowest first, so the first value is coded {start} and the last {start + len(order) - 1}",
        label_line,
        "Use ^ and v to move a value one place, Top and End to move it to either end",
    ]
    for i, value in enumerate(order):
        panel.add_option(f"{start + i}. {value}", [
            ("^", move_value, False),
            ("v", move_value, False),
            ("Top", move_value, False),
            ("End", move_value, False),
        ])
        for button in panel.options[-1][1]:
            button.value = value

    counts = values.astype(str).where(values.notna()).value_counts()
    panel.preview = pd.DataFrame({
        "code": [start + i for i in range(len(order))],
        "value": order,
        "rows": [int(counts.get(v, 0)) for v in order],
    })
