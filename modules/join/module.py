import pandas as pd

from ui.panel import Panel


INPUTS = 2
OUTPUTS = 1
INPUT_SPECS = [
    ("Left", "a table sharing one or more key columns with Right"),
    ("Right", "a table sharing one or more key columns with Left"),
]
OUTPUT_SPECS = [
    ("Data", "the rows of both tables side by side, matched on the key columns"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["on"]

# How many rows each key can match on each side. "1" means a key may appear only once in that table:
# M:1 is many Left rows to one Right row, e.g. firm-year data joined to one row of firm details
RELATIONSHIPS = ["1:1", "M:1", "1:M", "M:M"]
DEFAULT_RELATIONSHIP = "M:1"

# Which rows come out: only those matched in both tables, every Left or Right row, or every row of either.
# Rows with no match get empty values in the other table's fields
KEEP = {"Both": "inner", "Left": "left", "Right": "right", "All": "outer"}
DEFAULT_KEEP = "Both"

# Added to fields both tables have that aren't keys, so both versions come out
SUFFIXES = ("_left", "_right")


# Opens a panel to choose the key fields, the relationship and which rows to keep, with a preview
def on_click(node):
    panel = Panel(node, "Join")
    panel.add_button("Clear", clear_keys)
    panel.refresh = show_join
    show_join(panel)
    return panel


# Adds the clicked field to the keys, or takes it out if it already was one
def choose_key(panel, button):
    on = panel.node.settings.setdefault("on", [])
    column = button.value
    if column in on:
        on.remove(column)
    else:
        on.append(column)
    show_join(panel)


def choose_relationship(panel, button):
    panel.node.settings["relationship"] = button.text
    show_join(panel)


def choose_keep(panel, button):
    panel.node.settings["keep"] = button.text
    show_join(panel)


def clear_keys(panel, button):
    panel.node.settings["on"] = []
    show_join(panel)


# Shows the relationship and rows kept, then lists the fields both inputs share with a Key button,
# and previews the joined data
def show_join(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or any(data is None for data in node.last_inputs):
        panel.lines = ["Run the workflow with both inputs connected to load the fields"]
        panel.message = ""
        return

    left, right = node.last_inputs
    keys = key_columns(left, right, settings)
    chosen = ", ".join(keys) if settings.get("on") else f"every shared field ({', '.join(keys) or 'none'})"
    panel.lines = [f"Joining on: {chosen}"]

    relationship = settings.get("relationship", DEFAULT_RELATIONSHIP)
    panel.add_option("Relationship", [(r, choose_relationship, r == relationship) for r in RELATIONSHIPS])
    keep = settings.get("keep", DEFAULT_KEEP)
    panel.add_option("Rows kept", [(k, choose_keep, k == keep) for k in KEEP])

    on = settings.get("on", [])
    for column in shared_columns(left, right):
        panel.add_option(column, [("Key", choose_key, column in on)])

    try:
        panel.preview = run([left, right], settings)[0]
        panel.lines.append(f"Rows in: {len(left)} Left, {len(right)} Right    Rows out: {len(panel.preview)}")
        panel.message = ""
    except Exception as error:
        panel.message = f"Could not join: {error}"


def shared_columns(left, right):
    return [column for column in left.columns if column in right.columns]


# The fields in settings["on"] that both tables have, or every field they share if none are chosen.
# Chosen fields missing from either table are skipped, so a changed file or input still runs
def key_columns(left, right, settings):
    shared = shared_columns(left, right)
    on = [column for column in settings.get("on", []) if column in shared]
    return on or shared


# Raises an error if a side that should have each key only once has some keys more than once
def check_unique(data, keys, side, relationship):
    repeats = data.duplicated(subset=keys, keep=False)
    if repeats.any():
        example = ", ".join(str(v) for v in data.loc[repeats, keys].iloc[0])
        raise ValueError(f"{relationship} needs each key once in {side}, but some repeat there, e.g. ({example})")


# Joins Left and Right on the key columns. settings["relationship"] checks each key appears once
# on the "1" sides, and settings["keep"] picks which rows come out
def run(inputs, settings):
    left, right = inputs
    keys = key_columns(left, right, settings)
    if not keys:
        raise ValueError("Left and Right share no fields to join on")

    relationship = settings.get("relationship", DEFAULT_RELATIONSHIP)
    left_side, right_side = relationship.split(":")
    if left_side == "1":
        check_unique(left, keys, "Left", relationship)
    if right_side == "1":
        check_unique(right, keys, "Right", relationship)

    how = KEEP[settings.get("keep", DEFAULT_KEEP)]
    return [pd.merge(left, right, on=keys, how=how, suffixes=SUFFIXES)]
