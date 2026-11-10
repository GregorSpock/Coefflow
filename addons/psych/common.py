import numpy as np
import pandas as pd
from scipy import stats

from ui.panel import Panel


# Shared by the Psych pack's modules: the panel where fields are given roles, buttons that step
# through a setting's choices, the checks on the chosen fields, and the wording of results.
#
# A module describes its roles as a list of (button text, settings key, many, numeric only).
# A role with many holds a list of fields, in the order they were clicked, and one without holds
# a single field. A field has at most one role, so giving it another takes it out of the first

LEVEL = 0.05
# A grouping field with more groups than this is almost certainly not a grouping at all
MAX_GROUPS = 30


# Makes the module's panel. show(panel) lists the fields and the results, and is called again
# whenever a role or setting changes or the workflow runs
def open_panel(node, title, show, preview_title="Results"):
    panel = Panel(node, title)
    panel.preview_title = preview_title
    panel.refresh = show
    return panel


# Adds a button that steps through a setting's choices, e.g. "Test: Paired". The first choice is
# the one used until another is chosen
def add_choice(panel, label, key, choices):
    def text():
        return f"{label}: {panel.node.settings.get(key, choices[0])}"

    def step(panel, button):
        settings = panel.node.settings
        current = choices.index(settings.get(key, choices[0])) if settings.get(key, choices[0]) in choices else 0
        settings[key] = choices[(current + 1) % len(choices)]
        button.text = text()
        panel.refresh(panel)

    return panel.add_button(text(), step)


# Adds a button that switches a setting on or off, e.g. "Welch: On"
def add_switch(panel, label, key, default=True):
    def text():
        return f"{label}: {'On' if panel.node.settings.get(key, default) else 'Off'}"

    def switch(panel, button):
        settings = panel.node.settings
        settings[key] = not settings.get(key, default)
        button.text = text()
        panel.refresh(panel)

    return panel.add_button(text(), switch)


# True if a field has the role kept under key
def has_role(settings, key, many, column):
    held = settings.get(key)
    return column in (held or []) if many else held == column


# Gives a field the clicked role, or takes it away if it already had it. The field leaves any
# other role, and a role that holds one field swaps to the new one
def choose_role(panel, button):
    settings = panel.node.settings
    column = button.value
    for text, key, many, numeric in panel.roles:
        held = has_role(settings, key, many, column)
        if many:
            names = [name for name in settings.get(key) or [] if name != column]
            if text == button.text and not held:
                names.append(column)
            settings[key] = names
        elif text == button.text:
            settings[key] = None if held else column
        elif held:
            settings[key] = None
    panel.refresh(panel)


# Clears the panel and lists the fields of the last run with a button for each role they can
# take. Returns the data, or None if the workflow hasn't run yet
def list_fields(panel, roles, action=choose_role):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.roles = roles
    panel.preview = None
    panel.chart = None
    panel.note = ""
    panel.message = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        return None

    data = node.last_inputs[0]
    numeric = set(data.select_dtypes("number").columns)
    for column in data.columns:
        choices = [
            (text, action, has_role(settings, key, many, column))
            for text, key, many, numeric_only in roles
            if column in numeric or not numeric_only
        ]
        if choices:
            panel.add_option(column, choices)
    return data


# The fields held under a setting, as text for a panel's info lines
def names(settings, key, none="none"):
    held = settings.get(key)
    if isinstance(held, list):
        return ", ".join(held) or none
    return held or none


# Shows the first output of the last run with its note, or a hint on what to choose if there is none
def show_results(panel, hint, output=0):
    outputs = panel.node.last_outputs
    if not outputs:
        panel.message = hint
        return
    panel.preview = outputs[output]
    panel.note = outputs[0].attrs.get("note", "")


# The data with infinite values treated as missing
def clean(data):
    return data.replace([np.inf, -np.inf], np.nan)


# Checks every chosen field is in the data, and that those in numeric are numbers
def check_fields(data, columns, numeric=()):
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(data[column]) or pd.api.types.is_bool_dtype(data[column]):
            raise ValueError(f"{column} is not numeric")


# The values of a numeric field in each group of another, as a list of (group name, values) in
# the groups' own order. Rows missing either field are left out, as are groups left with no values
def grouped(data, value, group):
    pairs = data[[value, group]].dropna()
    groups = []
    for label, rows in pairs.groupby(group, sort=True):
        groups.append((group_text(label), rows[value].to_numpy(float)))
    if len(groups) > MAX_GROUPS:
        raise ValueError(f"{group} has {len(groups)} groups, which is too many to compare. Is it the right field?")
    return groups


# Short text for a group's value, so 3.0 shows as 3
def group_text(value):
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


# A p value written the way journals ask for: "p < .001" or "p = .023"
def p_text(p_value):
    if pd.isna(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".", 1)


# A number that can't be above 1, such as a correlation, written without its leading zero
def short(value, digits=2):
    if pd.isna(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return text.replace("0.", ".", 1) if abs(value) < 1 else text


# Degrees of freedom as text: whole numbers plainly, others to one decimal place
def df_text(df):
    return f"{df:.0f}" if float(df).is_integer() else f"{df:.1f}"


# How big Cohen's d is in words, by Cohen's (1988) rules of thumb
def d_size(d):
    size = abs(d)
    if size < 0.2:
        return "negligible"
    if size < 0.5:
        return "small"
    if size < 0.8:
        return "medium"
    return "large"


# The standard deviation of two groups pooled, which Cohen's d divides the difference in means by
def pooled_sd(a, b):
    return np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))


# The mean of some values with the ends of its 95% confidence interval
def mean_interval(values, level=1 - LEVEL):
    mean = values.mean()
    if len(values) < 2:
        return mean, np.nan, np.nan
    reach = stats.t.ppf((1 + level) / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - reach, mean + reach


# Questionnaire items, for the modules that work on a scale. Item marks a field as one of the
# scale's items and Rev as an item scored the other way round, which is reversed before use
ITEM_ROLES = [("Item", "items", True, True), ("Rev", "reversed", True, True)]


# Marks a field as an item or as a reversed item, or clears the mark if it already had it.
# A reversed item is still an item, so Rev adds it to both and taking Item away clears both
def choose_item(panel, button):
    settings = panel.node.settings
    column = button.value
    items = [name for name in settings.get("items") or []]
    reversed_items = [name for name in settings.get("reversed") or []]

    if button.text == "Item":
        if column in items:
            items.remove(column)
            reversed_items = [name for name in reversed_items if name != column]
        else:
            items.append(column)
    elif column in reversed_items:
        reversed_items.remove(column)
    else:
        reversed_items.append(column)
        if column not in items:
            items.append(column)

    settings["items"] = items
    settings["reversed"] = reversed_items
    panel.refresh(panel)


# The chosen items as a table, with the reversed ones turned round: a reversed answer becomes the
# lowest plus the highest answer given to any of the items, minus itself, so on a 1 to 5 scale 2
# becomes 4. Also returns the lowest and highest answer, so a module can say what it did
def item_scores(data, settings, least=2):
    items = settings.get("items") or []
    reversed_items = [name for name in settings.get("reversed") or [] if name in items]
    if len(items) < least:
        raise ValueError(f"Choose at least {least} items")
    check_fields(data, items, numeric=items)

    scores = data[items].astype(float)
    low, high = scores.min().min(), scores.max().max()
    for name in reversed_items:
        scores[name] = low + high - scores[name]
    return scores, low, high


# Says how the reversed items were turned round, for a module's note
def reversal_text(settings, low, high):
    reversed_items = [name for name in settings.get("reversed") or [] if name in (settings.get("items") or [])]
    if not reversed_items:
        return ""
    return f"Reversed {', '.join(reversed_items)} as {low + high:g} minus the answer (answers run from {low:g} to {high:g})."
