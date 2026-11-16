import pandas as pd

from engine import dummies
from engine.panel import Panel


# A settings panel described rather than written out, shared by the modules that give each field a
# role (Y, X, T...) and have a few switches and orders. A module describes its panel with a Spec:
#   roles    the role letters, each a Role saying which setting holds it and which fields can take it
#   choices  buttons on the top row that step through a few values, e.g. "Model: Poisson"
#   numbers  rows of number buttons, e.g. the lag order 1 to 8
#   text     an optional text box, as (setting, label, default)
#   buttons  extra top-row buttons, as (label function(settings), action(panel, button))
#   chart    an optional function(node) giving a chart function(painter, area), or None. With one,
#            a View button switches the panel between the results table and the chart
# Its run puts the lines to show above the results in attrs["summary"] of the output it previews,
# and any warnings in attrs["notes"]. Nothing here needs a window


class Role:
    # letter     the button's text, e.g. "Y"
    # key        the setting that holds it: one field name, or a list of them if many
    # kinds      which fields can take it: "numeric", "model" (numeric or text, which enters as
    #            dummies) or "any" (dates too, as for a time or group field)
    def __init__(self, letter, key, many=False, kinds="numeric"):
        self.letter = letter
        self.key = key
        self.many = many
        self.kinds = kinds


class Choice:
    # options is a list of (value, text); the first is the default unless default says otherwise
    def __init__(self, key, label, options, default=None):
        self.key = key
        self.label = label
        self.options = options
        self.default = options[0][0] if default is None else default


class Number:
    def __init__(self, key, label, values, default):
        self.key = key
        self.label = label
        self.values = list(values)
        self.default = default


class Spec:
    def __init__(self, title, roles, choices=(), numbers=(), text=None, hint="", chart=None, preview=0, preview_title="Results",
                 series=False, buttons=(), data_input=0):
        self.title = title
        self.roles = list(roles)
        self.choices = list(choices)
        self.numbers = list(numbers)
        self.text = text
        self.hint = hint or "Choose the fields' roles, then run the workflow"
        self.chart = chart
        self.preview = preview
        self.preview_title = preview_title
        # A G role picks one series at a time, with a Next series button
        self.series = series
        # Extra top-row buttons as (label function(settings), action(panel, button))
        self.buttons = list(buttons)
        # Which input's fields get the role buttons
        self.data_input = data_input

    # The settings that hold field names, for the module's FIELD_SETTINGS
    def field_settings(self):
        return [role.key for role in self.roles]

    def defaults(self, settings):
        for role in self.roles:
            settings.setdefault(role.key, [] if role.many else None)
        for choice in self.choices:
            settings.setdefault(choice.key, choice.default)
        for number in self.numbers:
            settings.setdefault(number.key, number.default)
        if self.text:
            settings.setdefault(self.text[0], self.text[2])


VIEW_KEY = "view"


def open_panel(node, spec):
    spec.defaults(node.settings)
    panel = Panel(node, spec.title)
    panel.spec = spec
    if spec.series:
        panel.add_button("Next series", next_series)
    for choice in spec.choices:
        button = panel.add_button(choice_label(node.settings, choice), cycle_choice)
        button.value = choice
    for label, action in spec.buttons:
        panel.add_button(label(node.settings), action)
    if spec.chart:
        node.settings.setdefault(VIEW_KEY, "table")
        panel.add_button(view_label(node.settings), toggle_view)
    if spec.text:
        key, label, default = spec.text
        panel.add_text_field(label, str(node.settings.get(key) or ""), type_text, blocked="", fill=True)
    panel.preview_title = spec.preview_title
    panel.refresh = show
    show(panel)
    return panel


def choice_label(settings, choice):
    value = settings.get(choice.key, choice.default)
    text = next((t for v, t in choice.options if v == value), str(value))
    return f"{choice.label}: {text}"


def cycle_choice(panel, button):
    choice = button.value
    values = [v for v, t in choice.options]
    current = panel.node.settings.get(choice.key, choice.default)
    index = values.index(current) if current in values else -1
    panel.node.settings[choice.key] = values[(index + 1) % len(values)]
    button.text = choice_label(panel.node.settings, choice)
    show(panel)


def view_label(settings):
    return "View: Chart" if settings.get(VIEW_KEY) == "chart" else "View: Table"


def toggle_view(panel, button):
    settings = panel.node.settings
    settings[VIEW_KEY] = "table" if settings.get(VIEW_KEY) == "chart" else "chart"
    button.text = view_label(settings)
    show(panel)


def type_text(panel, text):
    panel.node.settings[panel.spec.text[0]] = text
    show(panel)


def choose_number(panel, button):
    number = next(n for n in panel.spec.numbers if n.label == button.value)
    panel.node.settings[number.key] = int(button.text)
    show(panel)


# The values of the group field, sorted as text, and the one chosen (the first if none is)
def series_names(data, settings):
    group = settings.get("group")
    if not group or group not in data.columns:
        return []
    return sorted(data[group].dropna().astype(str).unique())


def chosen_series(data, settings):
    names = series_names(data, settings)
    if not names:
        return None
    return settings["series"] if settings.get("series") in names else names[0]


def next_series(panel, button):
    node = panel.node
    if node.last_inputs and node.last_inputs[0] is not None:
        names = series_names(node.last_inputs[0], node.settings)
        if names:
            current = chosen_series(node.last_inputs[0], node.settings)
            node.settings["series"] = names[(names.index(current) + 1) % len(names)]
    show(panel)


# The role letter a field has, or None
def role_of(settings, column, roles):
    for role in roles:
        value = settings.get(role.key)
        if (role.many and column in (value or [])) or (not role.many and value == column):
            return role.letter
    return None


def clear_role(settings, column, roles):
    for role in roles:
        if role.many:
            settings[role.key] = [c for c in settings.get(role.key) or [] if c != column]
        elif settings.get(role.key) == column:
            settings[role.key] = None


# Gives a field the clicked role, or takes it away if it already had it. A role held by one
# field moves to the newly clicked one
def choose_role(panel, button):
    roles = panel.spec.roles
    settings = panel.node.settings
    column = button.value
    role = next(r for r in roles if r.letter == button.text)
    current = role_of(settings, column, roles)
    clear_role(settings, column, roles)
    if current != role.letter:
        if role.many:
            settings.setdefault(role.key, []).append(column)
        else:
            settings[role.key] = column
            if role.key == "group":
                settings["series"] = None
    show(panel)


# Which role letters a field can take
def letters_for(data, column, roles):
    numeric = dummies.is_numeric(data[column])
    usable = column in dummies.model_columns(data)
    letters = []
    for role in roles:
        if role.kinds == "any" or (role.kinds == "numeric" and numeric) or (role.kinds == "model" and usable):
            letters.append(role.letter)
    return letters


def role_lines(settings, roles):
    parts = []
    for role in roles:
        value = settings.get(role.key)
        shown = ", ".join(value) if role.many else value
        parts.append(f"{role.letter}: {shown or 'none'}")
    lines, line = [], ""
    for part in parts:
        if line and len(line) + len(part) > 90:
            lines.append(line)
            line = ""
        line = f"{line}    {part}" if line else part
    return lines + ([line] if line else [])


# Shows the roles, number rows and each field with its role buttons, then the last results
def show(panel):
    node = panel.node
    spec = panel.spec
    settings = node.settings
    spec.defaults(settings)
    panel.clear_options()
    panel.preview = None
    panel.chart = None
    panel.note = ""
    panel.message = ""
    panel.lines = role_lines(settings, spec.roles)

    for number in spec.numbers:
        current = settings.get(number.key, number.default)
        panel.add_option(number.label, [(str(n), choose_number, n == current) for n in number.values])

    data = node.last_inputs[spec.data_input] if len(node.last_inputs or []) > spec.data_input else None
    if not isinstance(data, pd.DataFrame):
        panel.message = "Run the workflow to load the fields"
        return
    if spec.series and settings.get("group"):
        panel.lines.append(f"Series: {settings['group']} = {chosen_series(data, settings)}")
    for column in data.columns:
        letters = letters_for(data, column, spec.roles)
        if letters:
            current = role_of(settings, column, spec.roles)
            panel.add_option(column, [(letter, choose_role, current == letter) for letter in letters])

    outputs = node.last_outputs
    if not outputs or len(outputs) <= spec.preview or not isinstance(outputs[spec.preview], pd.DataFrame):
        panel.message = node.error or spec.hint
        return
    results = outputs[spec.preview]
    panel.lines += list(results.attrs.get("summary", []))
    panel.note = " ".join(results.attrs.get("notes", []))
    if spec.chart and settings.get(VIEW_KEY) == "chart":
        panel.chart = spec.chart(node)
        if panel.chart is None:
            panel.message = "Nothing to draw yet"
    else:
        panel.preview = results


# Checks the fields named in the settings for each role exist and, where the role needs it, are
# numeric. required lists the role letters that must be filled. Returns nothing; raises ValueError
def check(data, settings, roles, required=()):
    for role in roles:
        value = settings.get(role.key)
        names = (value or []) if role.many else ([value] if value else [])
        if role.letter in required and not names:
            raise ValueError(f"Choose the {role.letter} field" + (" (or several)" if role.many else ""))
        missing = [n for n in names if n not in data.columns]
        if missing:
            raise ValueError(f"Columns not in the data: {', '.join(missing)}")
        if role.kinds == "numeric":
            text = [n for n in names if not dummies.is_numeric(data[n])]
            if text:
                raise ValueError(f"These fields need to be numeric for {role.letter}: {', '.join(text)}")


# Steps a setting through the given names, e.g. which shock a chart of responses shows, and
# shows the panel again
def step(panel, key, names):
    if not names:
        return
    current = panel.node.settings.get(key)
    index = names.index(current) if current in names else -1
    panel.node.settings[key] = names[(index + 1) % len(names)]
    show(panel)
