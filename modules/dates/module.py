import pandas as pd

from engine import dates
from engine.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with fields holding dates, as dates or as text such as 2020-03-31, 31/03/2020, 2020Q1 or 1990:1"),
]
OUTPUT_SPECS = [
    ("Data", "the same table with the chosen fields turned into dates, plus a new field for each part asked for (e.g. date_year, date_quarter), "
             "the time between the From and To dates, and a time trend counting periods from the earliest T date"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["convert", "parts", "start", "end", "trend"]
UNITS = ["days", "weeks", "months", "years"]
ROLES = ["From", "To", "T"]
ROLE_KEYS = {"From": "start", "To": "end", "T": "trend"}


def set_defaults(settings):
    settings.setdefault("convert", [])
    settings.setdefault("parts", {})
    settings.setdefault("start", None)
    settings.setdefault("end", None)
    settings.setdefault("trend", None)
    settings.setdefault("unit", UNITS[0])
    settings.setdefault("format", "")
    settings.setdefault("dayfirst", False)


# Opens a panel to choose, field by field, what to do with dates: turn the field into dates (Date),
# split off parts of it, measure From one date To another, or count periods from the earliest (T)
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Dates")
    panel.add_button(f"Unit: {node.settings['unit']}", switch_unit)
    panel.add_button(dayfirst_label(node.settings), toggle_dayfirst)
    panel.add_button("Clear", clear_all)
    panel.add_text_field("Format (blank to work it out, e.g. %d/%m/%Y)", node.settings["format"], type_format, blocked="")
    panel.options_title = "Fields"
    panel.option_columns = 1
    panel.refresh = show_dates
    show_dates(panel)
    return panel


def dayfirst_label(settings):
    return "Day first: On" if settings.get("dayfirst") else "Day first: Off"


def switch_unit(panel, button):
    settings = panel.node.settings
    settings["unit"] = UNITS[(UNITS.index(settings["unit"]) + 1) % len(UNITS)]
    button.text = f"Unit: {settings['unit']}"
    show_dates(panel)


def toggle_dayfirst(panel, button):
    settings = panel.node.settings
    settings["dayfirst"] = not settings["dayfirst"]
    button.text = dayfirst_label(settings)
    show_dates(panel)


def type_format(panel, text):
    panel.node.settings["format"] = text
    show_dates(panel)


def clear_all(panel, button):
    settings = panel.node.settings
    settings.update(convert=[], parts={}, start=None, end=None, trend=None)
    show_dates(panel)


def choose(panel, button):
    settings = panel.node.settings
    column, text = button.value, button.text
    if text == "Date":
        convert = settings["convert"]
        if column in convert:
            convert.remove(column)
        else:
            convert.append(column)
    elif text in ROLE_KEYS:
        key = ROLE_KEYS[text]
        settings[key] = None if settings[key] == column else column
    else:
        chosen = settings["parts"].setdefault(column, [])
        if text in chosen:
            chosen.remove(text)
        else:
            chosen.append(text)
        if not chosen:
            del settings["parts"][column]
    show_dates(panel)


def show_dates(panel):
    node = panel.node
    settings = node.settings
    set_defaults(settings)
    panel.clear_options()
    panel.preview = None

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    panel.lines = [
        f"Dates: {', '.join(settings['convert']) or 'none'}    From: {settings['start'] or 'none'}    "
        f"To: {settings['end'] or 'none'}    T: {settings['trend'] or 'none'}",
    ]
    panel.note = (
        "Date turns a field into dates in place. Year, Qtr, Month, Day, Wkday (1 = Monday), Week, YQ (2020Q1) and YM (2020-03) "
        "add a new field each. From and To give the time between two dates, and T a trend counting periods from the earliest date."
    )
    data = node.last_inputs[0]
    for column in data.columns:
        choices = [("Date", choose, column in settings["convert"])]
        choices += [(name, choose, name in settings["parts"].get(column, [])) for name in dates.PARTS]
        choices += [(role, choose, settings[ROLE_KEYS[role]] == column) for role in ROLES]
        panel.add_option(column, choices)
        for button in panel.options[-1][1]:
            button.value = column

    try:
        result = run([data], settings)[0]
        panel.preview = result
        panel.message = ""
        if result.attrs.get("unread"):
            panel.note = result.attrs["unread"] + " " + panel.note
    except Exception as error:
        panel.message = f"Could not work with the dates: {error}"


# Turns the chosen fields into dates and adds the parts, the time between dates and the trend.
# Every field used for a part, From, To or T is read as dates, whether or not it is also turned
# into dates in place. A note counts values that couldn't be read as dates
def run(inputs, settings):
    set_defaults(settings)
    data = inputs[0]
    used = list(dict.fromkeys(
        settings["convert"] + list(settings["parts"]) + [settings[k] for k in ("start", "end", "trend") if settings[k]]
    ))
    missing = [c for c in used if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")

    parsed, unread = {}, []
    for column in used:
        values = dates.parse(data[column], settings["format"], settings["dayfirst"])
        failed = int((values.isna() & data[column].notna()).sum())
        if failed:
            unread.append(f"{failed} values of {column}")
        parsed[column] = values

    result = data.copy()
    for column in settings["convert"]:
        result[column] = parsed[column]
    for column, chosen in settings["parts"].items():
        for name in chosen:
            if name in dates.PARTS:
                result[f"{column}_{dates.PARTS[name][0]}"] = dates.part(parsed[column], name)

    unit = settings["unit"]
    if settings["start"] and settings["end"]:
        result[f"{unit}_between"] = dates.between(parsed[settings["start"]], parsed[settings["end"]], unit)
    elif settings["start"] or settings["end"]:
        raise ValueError("Choose both a From and a To date to measure the time between them")
    if settings["trend"]:
        result[f"trend_{unit}"] = dates.trend(parsed[settings["trend"]], unit)

    if unread:
        result.attrs = {"unread": f"Couldn't read {', '.join(unread)} as dates, so they are missing. Give the format if they are written another way."}
    else:
        result.attrs = {}
    return [result]
