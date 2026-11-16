import numpy as np
import pandas as pd

from engine.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table with missing values, and optionally a field naming each group (G) for panel data"),
]
OUTPUT_SPECS = [
    ("Data", "the same table with each chosen field's missing values dropped or filled in, plus a field_missing flag "
             "for each filled field when Flag is on. A note says how many values were changed"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["methods", "group"]

# How a field's missing values are dealt with: the button's text and what it does
METHODS = {
    "drop": "drop the rows where the field is missing",
    "mean": "fill with the field's mean",
    "median": "fill with the field's median",
    "mode": "fill with the most common value",
    "value": "fill with the value typed in the box",
    "prev": "carry the last value forward",
    "next": "carry the next value back",
    "interp": "draw a straight line between the values either side",
}
NUMERIC_ONLY = ["mean", "median", "interp"]
FLAG_SUFFIX = "_missing"


def set_defaults(settings):
    settings.setdefault("methods", {})
    settings.setdefault("group", None)
    settings.setdefault("flag", False)
    settings.setdefault("value", "0")


# Opens a panel to choose how each field's missing values are dealt with
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Missing Values")
    panel.add_button(flag_label(node.settings), toggle_flag)
    panel.add_button("Clear", clear_methods)
    panel.add_text_field("Fill value", node.settings["value"], type_value, blocked="")
    panel.options_title = "Fields: group (G) and how to fill"
    panel.option_columns = 1
    panel.refresh = show_missing
    show_missing(panel)
    return panel


def flag_label(settings):
    return "Flag: On" if settings.get("flag") else "Flag: Off"


def toggle_flag(panel, button):
    settings = panel.node.settings
    settings["flag"] = not settings["flag"]
    button.text = flag_label(settings)
    show_missing(panel)


def clear_methods(panel, button):
    panel.node.settings["methods"] = {}
    panel.node.settings["group"] = None
    show_missing(panel)


def type_value(panel, text):
    panel.node.settings["value"] = text
    show_missing(panel)


def choose(panel, button):
    settings = panel.node.settings
    column = button.value
    if button.text == "G":
        settings["group"] = None if settings["group"] == column else column
        settings["methods"].pop(column, None)
    elif settings["methods"].get(column) == button.text:
        del settings["methods"][column]
    else:
        settings["methods"][column] = button.text
        if settings["group"] == column:
            settings["group"] = None
    show_missing(panel)


def show_missing(panel):
    node = panel.node
    settings = node.settings
    set_defaults(settings)
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    counts = data.isna().sum()
    panel.lines = [
        f"{int(counts.sum()):,} missing values in {int((counts > 0).sum())} of {len(data.columns)} fields    "
        f"Group (G): {settings['group'] or 'none'}",
    ]
    for column in data.columns:
        numeric = pd.api.types.is_numeric_dtype(data[column])
        choices = [("G", choose, settings["group"] == column)]
        for name in METHODS:
            if numeric or name not in NUMERIC_ONLY:
                choices.append((name, choose, settings["methods"].get(column) == name))
        panel.add_option(f"{column} ({int(counts[column])} missing)", choices)
        for button in panel.options[-1][1]:
            button.value = column

    try:
        result = run([data], settings)[0]
        panel.preview = result
        panel.message = ""
        panel.note = result.attrs.get("summary", "")
    except Exception as error:
        panel.message = f"Could not fill the missing values: {error}"
        panel.note = ""


# The typed fill value as a number when the field is numeric, else as text
def fill_value(text, values):
    if pd.api.types.is_numeric_dtype(values):
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"The fill value {text!r} is not a number, but {values.name} is numeric")
    return text


def most_common(values):
    modes = values.mode(dropna=True)
    return modes.iloc[0] if len(modes) else np.nan


# The field with its missing values filled by the method, within each group if one is given.
# prev, next and interp work in row order, so sort the rows by group and time first
def filled(data, column, method, group, value):
    values = data[column]
    grouped = values.groupby(data[group], sort=False) if group else None

    def by_group(statistic):
        return values.fillna(grouped.transform(statistic) if grouped is not None else getattr(values, statistic)())

    if method == "mean":
        return by_group("mean")
    if method == "median":
        return by_group("median")
    if method == "mode":
        if grouped is not None:
            return values.fillna(grouped.transform(most_common))
        return values.fillna(most_common(values))
    if method == "value":
        return values.fillna(fill_value(value, values))
    if method == "prev":
        return grouped.ffill() if grouped is not None else values.ffill()
    if method == "next":
        return grouped.bfill() if grouped is not None else values.bfill()
    if method == "interp":
        if grouped is not None:
            return grouped.transform(lambda part: part.interpolate(limit_area="inside"))
        return values.interpolate(limit_area="inside")
    raise ValueError(f"Unknown way to fill: {method}")


# Deals with the missing values of each field in settings["methods"]: drops those rows or fills
# them in. Fills are worked out from the data as it arrives, before any rows are dropped, and
# within each group if a group field is chosen. Flag adds a 0/1 field_missing column for each
# filled field so a model can control for which values were made up
def run(inputs, settings):
    set_defaults(settings)
    data = inputs[0]
    group = settings["group"] if settings["group"] in data.columns else None
    methods = {c: m for c, m in settings["methods"].items() if c in data.columns and m in METHODS}

    for column, method in methods.items():
        if method in NUMERIC_ONLY and not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} is not numeric, so it can't be filled with its {method}")

    result = data.copy()
    changed = {}
    drop = [c for c, m in methods.items() if m == "drop"]
    for column, method in methods.items():
        if method == "drop":
            continue
        before = result[column].isna()
        result[column] = filled(data, column, method, group, settings["value"])
        changed[column] = int((before & result[column].notna()).sum())
        if settings["flag"]:
            result[f"{column}{FLAG_SUFFIX}"] = before.astype(int)

    rows = len(result)
    if drop:
        result = result.dropna(subset=drop)
    dropped = rows - len(result)

    left = {c: int(result[c].isna().sum()) for c in changed if result[c].isna().any()}
    parts = [f"Filled {n:,} in {c}" for c, n in changed.items()]
    if drop:
        parts.append(f"dropped {dropped:,} rows missing {', '.join(drop)}")
    if left:
        parts.append("still missing (no value to fill from): " + ", ".join(f"{c} {n}" for c, n in left.items()))
    result.attrs = {"summary": "; ".join(parts) + "." if parts else "Choose how to fill each field."}
    return [result]
