import re

import pandas as pd

from engine.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table in long form (one row per id and period) to spread wide, or in wide form (a column per period) to stack long"),
]
OUTPUT_SPECS = [
    ("Data", "the same data reshaped: wide with a column for each value of the column field (C), or long with one row per id and stacked field"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["ids", "column", "values"]

DIRECTIONS = ["Long to wide", "Wide to long"]
# What to do when an id has more than one row for the same column value, as it would when spreading wide
COMBINE = ["first", "mean", "sum", "count", "last", "min", "max"]
DEFAULT_NAMES = "variable, value"


def set_defaults(settings):
    settings.setdefault("direction", DIRECTIONS[0])
    settings.setdefault("ids", [])
    settings.setdefault("column", None)
    settings.setdefault("values", [])
    settings.setdefault("combine", COMBINE[0])
    settings.setdefault("names", DEFAULT_NAMES)
    settings.setdefault("stubs", False)


# Opens a panel to choose the direction and give fields their roles: I (identifies a row),
# C (the column whose values become new columns, long to wide only) and V (the values)
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Pivot")
    settings = node.settings
    panel.add_button(settings["direction"], switch_direction)
    if settings["direction"] == DIRECTIONS[0]:
        panel.add_button(f"Duplicates: {settings['combine']}", switch_combine)
    else:
        panel.add_button(stubs_label(settings), toggle_stubs)
        if not settings["stubs"]:
            panel.add_text_field("New field names (name, value)", settings["names"], type_names, blocked="")
    panel.add_button("Clear", clear_roles)
    panel.options_title = "Fields"
    panel.refresh = show_pivot
    show_pivot(panel)
    return panel


def stubs_label(settings):
    return "Stata style: On" if settings.get("stubs") else "Stata style: Off"


def switch_direction(panel, button):
    settings = panel.node.settings
    settings["direction"] = DIRECTIONS[(DIRECTIONS.index(settings["direction"]) + 1) % len(DIRECTIONS)]
    settings["column"] = None
    panel.next_panel = on_click(panel.node)


def switch_combine(panel, button):
    settings = panel.node.settings
    settings["combine"] = COMBINE[(COMBINE.index(settings["combine"]) + 1) % len(COMBINE)]
    button.text = f"Duplicates: {settings['combine']}"
    show_pivot(panel)


def toggle_stubs(panel, button):
    settings = panel.node.settings
    settings["stubs"] = not settings["stubs"]
    panel.next_panel = on_click(panel.node)


def type_names(panel, text):
    panel.node.settings["names"] = text
    show_pivot(panel)


def clear_roles(panel, button):
    settings = panel.node.settings
    settings["ids"], settings["column"], settings["values"] = [], None, []
    show_pivot(panel)


def role_of(settings, column):
    if column in settings["ids"]:
        return "I"
    if settings["column"] == column:
        return "C"
    if column in settings["values"]:
        return "V"
    return None


def choose_role(panel, button):
    settings = panel.node.settings
    column, role = button.value, button.text
    current = role_of(settings, column)
    settings["ids"] = [c for c in settings["ids"] if c != column]
    settings["values"] = [c for c in settings["values"] if c != column]
    if settings["column"] == column:
        settings["column"] = None
    if current != role:
        if role == "I":
            settings["ids"].append(column)
        elif role == "C":
            settings["column"] = column
        else:
            settings["values"].append(column)
    show_pivot(panel)


def show_pivot(panel):
    node = panel.node
    settings = node.settings
    set_defaults(settings)
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    wide = settings["direction"] == DIRECTIONS[0]
    panel.lines = [f"Ids (I): {', '.join(settings['ids']) or 'none'}"]
    if wide:
        panel.lines.append(f"Column field (C): {settings['column'] or 'none'}    Values (V): {', '.join(settings['values']) or 'none'}")
        panel.note = "Each value of C becomes a column for each V field, e.g. year spread into gdp_2019, gdp_2020."
    else:
        panel.lines.append(f"Fields to stack (V): {', '.join(settings['values']) or 'none'}")
        panel.note = (
            "Stata style splits names like gdp_2019, gdp_2020, pop_2019 into gdp and pop columns with a period column, "
            "as Stata's reshape long does." if settings["stubs"] else
            "Every V field is stacked into one value column, with a column saying which field each row came from."
        )

    roles = ["I", "C", "V"] if wide else ["I", "V"]
    data = node.last_inputs[0]
    for column in data.columns:
        role = role_of(settings, column)
        panel.add_option(column, [(r, choose_role, role == r) for r in roles])

    try:
        panel.preview = run([data], settings)[0]
        panel.message = ""
    except Exception as error:
        panel.message = f"Could not reshape: {error}"


def check_fields(data, names):
    missing = [c for c in names if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")


# Spreads a long table wide: one row per combination of the id fields, and a column for each
# value field and each value of the column field, named like gdp_2019. With a single value field
# the new columns are named by the column field's values alone, prefixed with the field's name
def long_to_wide(data, settings):
    ids, column, values = settings["ids"], settings["column"], settings["values"]
    if not column:
        raise ValueError("Choose the column field (C) whose values become new columns")
    if not values:
        values = [c for c in data.columns if c not in ids and c != column]
    check_fields(data, ids + [column] + values)

    combine = settings.get("combine", "first")
    keys = ids or None
    if keys:
        duplicated = data.duplicated(subset=ids + [column]).any()
    else:
        duplicated = data.duplicated(subset=[column]).any()
    table = pd.pivot_table(
        data.assign(_row=0) if not keys else data,
        index=keys or ["_row"],
        columns=column,
        values=values,
        aggfunc=combine,
        dropna=False,
        sort=True,
        observed=True,
    )
    table.columns = [f"{value}_{level}" for value, level in table.columns]
    table = table.dropna(axis=1, how="all").reset_index(drop=not keys)
    if duplicated:
        table.attrs["note"] = f"Some ids had more than one row for a value of {column}; they were combined with {combine}."
    return table


# The two names for the stacked columns, from the text "name, value"
def stack_names(settings):
    parts = [part.strip() for part in str(settings.get("names") or DEFAULT_NAMES).split(",")]
    name = parts[0] or "variable"
    value = parts[1] if len(parts) > 1 and parts[1] else "value"
    if name == value:
        raise ValueError("The two new fields need different names")
    return name, value


# Splits a field name into its stub and suffix, at its last underscore or before trailing digits:
# gdp_2019 -> (gdp, 2019), pop1990 -> (pop, 1990). Returns None if it can't be split
def split_name(name):
    name = str(name)
    if "_" in name.strip("_"):
        stub, suffix = name.rsplit("_", 1)
        if stub and suffix:
            return stub, suffix
    match = re.fullmatch(r"(.*?\D)(\d+)", name)
    return (match.group(1), match.group(2)) if match else None


# Stacks a wide table long. Plainly, every value field goes into one value column beside a column
# naming the field. Stata style, fields like gdp_2019 and pop_2019 become gdp and pop columns with
# a period column holding 2019, one row per id and period
def wide_to_long(data, settings):
    ids, values = settings["ids"], settings["values"]
    if not values:
        values = [c for c in data.columns if c not in ids]
    check_fields(data, ids + values)
    if not values:
        raise ValueError("Choose the fields to stack (V)")

    if not settings.get("stubs"):
        name, value = stack_names(settings)
        clash = [n for n in (name, value) if n in ids]
        if clash:
            raise ValueError(f"An id field is already called {clash[0]}")
        return data.melt(id_vars=ids, value_vars=values, var_name=name, value_name=value)

    parts = {}
    for column in values:
        split = split_name(column)
        if split is None:
            raise ValueError(f"{column} doesn't end in a period such as _2019 or 2019, so Stata style can't split it")
        parts[column] = split
    period = "period"
    while period in ids or period in {stub for stub, suffix in parts.values()}:
        period = f"_{period}"

    frame = data[ids + values].copy()
    frame["_id"] = range(len(frame))
    stacked = frame.melt(id_vars=["_id"] + ids, value_vars=values, var_name="_field", value_name="_value")
    stacked["_stub"] = stacked["_field"].map(lambda c: parts[c][0])
    stacked[period] = stacked["_field"].map(lambda c: parts[c][1])
    if stacked[period].str.fullmatch(r"\d+").all():
        stacked[period] = stacked[period].astype(int)
    result = stacked.pivot_table(index=["_id", period], columns="_stub", values="_value", aggfunc="first", dropna=False, sort=True)
    result.columns = [str(c) for c in result.columns]
    result = result.reset_index()
    result = frame[["_id"] + ids].merge(result, on="_id").drop(columns="_id")
    return result.infer_objects()


# Reshapes the table long to wide or wide to long, as settings["direction"] says
def run(inputs, settings):
    set_defaults(settings)
    data = inputs[0]
    if settings["direction"] == DIRECTIONS[0]:
        return [long_to_wide(data, settings)]
    return [wide_to_long(data, settings)]


# Fields named as ids keep their names; the rest are reshaped into new ones
def output_name(settings, name):
    return name if name in settings.get("ids", []) else None
