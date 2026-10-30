import keyword
import re

import pandas as pd

from engine import generate
from engine.fields import drop_downstream, rename_downstream
from ui.fields import connections
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Data", "the same rows with each new field added at the end, or in place of a field it shares a name with"),
]

# settings["lines"] holds one line of Python per new field, such as "gdp_pc = rgdpe / pop",
# worked out top to bottom so a line can use the fields made above it (see engine/generate.py).
# The main panel lists the lines with a preview of the data, and + opens a panel to write a new one,
# by typing or with buttons that write operators, functions and field names in at the cursor

# Rows of buttons on the edit panel. Those ending in () are functions, which wrap the field or
# bracket just before the cursor, or else go in empty with the cursor inside
TOOLS = [
    ("Maths", ["+", "-", "*", "/", "^2", "^", "(", ")"]),
    ("More", ["//", "%", "log()", "exp()", "sqrt()", "abs()"]),
    ("Round", [",", "int()", "round()", "floor()", "ceil()"]),
    ("Compare", ["==", "!=", ">", "<", ">=", "<="]),
    ("Logic", ["and", "or", "not", "where()", "missing()"]),
    ("Panel", ["lag()", "lead()", "diff()", "growth()", "by="]),
    ("Group", ["mean()", "sum()", "std()", "min()", "max()"]),
]
SPACED = ["+", "-", "*", "/", "//", "%", "==", "!=", ">", "<", ">=", "<=", "and", "or"]
EDIT_ROWS = 12
# How many lines the main panel shows at once; the rest are reached by scrolling
LINE_ROWS = 5


def lines_of(settings):
    return settings.setdefault("lines", [])


def targets(settings):
    return [generate.target_name(line) for line in settings.get("lines", [])]


def run(inputs, settings):
    return [generate.generate(inputs[0], settings.get("lines", []))]


# A field that arrives under a name a line makes is replaced, so it doesn't come out
def output_name(settings, name):
    return None if name in targets(settings) else name


# Renames a field arriving from upstream in each line that uses it, down to a line that makes a
# field of the same name, as the lines below that one mean the new field
def rename_field(settings, old, new):
    lines = lines_of(settings)
    for i, line in enumerate(lines):
        lines[i] = generate.rename_line(line, old, new)
        if generate.target_name(line) == old:
            break


# The data as it comes out, or None and the error if a line fails or nothing has run yet
def generated(node):
    if not node.last_inputs or node.last_inputs[0] is None:
        return None, None
    try:
        return generate.generate(node.last_inputs[0], node.settings.get("lines", [])), None
    except ValueError as error:
        return None, str(error)


# Opens the main panel: the lines, each with Edit, ^ and v to move it, and X to remove it,
# then a preview of the data with the new fields first
def on_click(node):
    panel = Panel(node, "Gen")
    panel.option_columns = 1
    panel.options_title = "Lines"
    panel.max_option_rows = LINE_ROWS
    panel.add_button("+", add_line)
    panel.add_button("Clear", clear_all)
    panel.refresh = show_gen
    show_gen(panel)
    return panel


def show_gen(panel):
    node = panel.node
    lines = lines_of(node.settings)
    panel.clear_options()
    panel.preview = None
    panel.note = ""
    panel.lines = [
        f"New fields: {', '.join(n for n in targets(node.settings) if n) or 'none yet'}. Click + to add one",
        "Lines run top to bottom, so a line can use the fields made above it",
    ]

    for i, line in enumerate(lines):
        panel.add_option(line, [
            ("Edit", open_edit, False),
            ("^", move_line, False),
            ("v", move_line, False),
            ("X", delete_line, False),
        ])
        for button in panel.options[-1][1]:
            button.value = i

    data, error = generated(node)
    if error:
        panel.note = error
    if data is None:
        panel.message = "" if error else "Run the workflow to preview the new fields"
        return

    new = [n for n in dict.fromkeys(targets(node.settings)) if n in data.columns]
    panel.preview_title = "Data Preview (new fields first)" if new else "Data Preview"
    panel.preview = data[new + [c for c in data.columns if c not in new]]
    panel.message = ""


# A name for a new field that isn't taken yet: new_var_1, new_var_2 and so on
def unused_name(node):
    taken = set(targets(node.settings))
    if node.last_inputs and node.last_inputs[0] is not None:
        taken |= set(node.last_inputs[0].columns)
    number = 1
    while f"new_var_{number}" in taken:
        number += 1
    return f"new_var_{number}"


def add_line(panel, button):
    lines = lines_of(panel.node.settings)
    lines.append(f"{unused_name(panel.node)} = ")
    panel.next_panel = edit_panel(panel.node, len(lines) - 1)


def open_edit(panel, button):
    panel.next_panel = edit_panel(panel.node, button.value)


# Takes a line's field out of the nodes below, unless it replaced a field that still arrives from upstream
def forget_field(panel, index):
    node = panel.node
    name = generate.target_name(lines_of(node.settings)[index])
    arriving = node.last_inputs[0].columns if node.last_inputs and node.last_inputs[0] is not None else []
    if name and name not in arriving and targets(node.settings).count(name) == 1:
        drop_downstream(node, connections(panel), name)


def delete_line(panel, button):
    forget_field(panel, button.value)
    lines_of(panel.node.settings).pop(button.value)
    panel.refresh(panel)


def clear_all(panel, button):
    lines = lines_of(panel.node.settings)
    for index in reversed(range(len(lines))):
        forget_field(panel, index)
        lines.pop(index)
    panel.refresh(panel)


# Moves a line up (^) or down (v) one place
def move_line(panel, button):
    lines = lines_of(panel.node.settings)
    index = button.value
    other = index - 1 if button.text == "^" else index + 1
    if 0 <= other < len(lines):
        lines[index], lines[other] = lines[other], lines[index]
    panel.refresh(panel)


# The panel for writing one line: a box holding the line's Python, then rows of operator and function
# buttons and a button for each field, which all write into the box at the cursor.
# The preview shows the new field next to the fields it uses. Back returns to the list of lines
def edit_panel(node, index):
    line = lines_of(node.settings)[index]
    panel = Panel(node, "Gen: new field")
    panel.index = index
    panel.add_button("Back", go_back)
    panel.add_button("Clear", clear_line)
    panel.add_button("Delete", delete_from_edit)
    panel.add_text_field("Code:", line, change_line, blocked="", fill=True)
    panel.text_field.focused = True
    panel.downstream_name = generate.target_name(line)
    panel.max_option_rows = EDIT_ROWS
    panel.preview_title = "Preview"
    panel.refresh = reload_edit
    reload_edit(panel)
    return panel


def go_back(panel, button):
    panel.next_panel = panel.node.on_click()


# Empties the expression, keeping the new field's name
def clear_line(panel, button):
    field = panel.text_field
    name, expression = generate.split_line(field.text)
    field.set_text(f"{name} = " if name else "")
    field.focused = True
    change_line(panel, field.text)


def delete_from_edit(panel, button):
    forget_field(panel, panel.index)
    lines_of(panel.node.settings).pop(panel.index)
    panel.next_panel = panel.node.on_click()


# Works out the data as it reaches this line, from the input and the lines above, then redraws
def reload_edit(panel):
    node = panel.node
    panel.before, panel.before_error = None, None
    if node.last_inputs and node.last_inputs[0] is not None:
        try:
            panel.before = generate.generate(node.last_inputs[0], lines_of(node.settings)[:panel.index])
        except ValueError as error:
            panel.before_error = f"A line above this one has a problem. {error}"
    show_edit(panel)


# Saves the line as it is typed, and when the new field's name changes, renames it in the lines
# below and the nodes below. While the name matches a field that already exists they are left alone,
# so the two fields don't get mixed up
def change_line(panel, text):
    node = panel.node
    lines = lines_of(node.settings)
    lines[panel.index] = text

    new = generate.target_name(text)
    old = panel.downstream_name
    if new and old and new != old:
        others = {n for i, n in enumerate(targets(node.settings)) if i != panel.index}
        arriving = set(panel.before.columns) if panel.before is not None else set()
        if new not in others | arriving:
            if old not in arriving:
                for i in range(panel.index + 1, len(lines)):
                    renamed = generate.rename_line(lines[i], old, new)
                    stop = generate.target_name(lines[i]) == old
                    lines[i] = renamed
                    if stop:
                        break
                rename_downstream(node, connections(panel), {old: new})
            panel.downstream_name = new
    elif new and not old:
        panel.downstream_name = new
    show_edit(panel)


def type_of(values):
    if pd.api.types.is_bool_dtype(values):
        return "0/1"
    if pd.api.types.is_numeric_dtype(values):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "date"
    return "text"


def show_edit(panel):
    panel.clear_options()
    panel.preview = None
    panel.note = ""
    panel.message = ""
    panel.lines = ["Type Python, or click the buttons and fields below to write it in at the cursor"]

    for label, tools in TOOLS:
        panel.add_option(label, [(tool, use_tool, False) for tool in tools])

    data = panel.before
    if data is None:
        panel.lines.append(panel.before_error or "Run the workflow to load the fields")
        return

    for column in data.columns:
        panel.add_option(type_of(data[column]), [(generate.written(str(column)), insert_field, False)])
        panel.options[-1][1][0].value = column

    line = panel.text_field.text
    used = generate.fields_used(generate.split_line(line)[1], data)
    try:
        name, values = generate.evaluate(line, data)
    except ValueError as error:
        panel.lines.append(f"Not ready: {error}")
        panel.preview = data[used] if used else None
        panel.message = "" if used else "The new field shows here once the line works"
        return

    count = int(values.notna().sum())
    replaces = " (replaces the field of that name)" if name in data.columns else ""
    panel.lines.append(f"Makes {name}{replaces}: {type_of(values)}, {count} values, {len(values) - count} missing")
    panel.preview = pd.concat([values.rename(name), data[[c for c in used if c != name]]], axis=1)
    if count == 0:
        panel.note = f"Every value of {name} is missing. Check the fields it uses have values in the same rows."


# Where the field or bracketed part ending at end starts, so a function can wrap it,
# or None if there isn't one ending exactly there
def operand_start(text, end):
    start = end
    if start and text[start - 1] == ")":
        depth = 0
        while start > 0:
            start -= 1
            depth += {")": 1, "(": -1}.get(text[start], 0)
            if depth == 0:
                break
        if depth:
            return None
    elif start and text[start - 1] == "`":
        start = text.rfind("`", 0, start - 1)
        return start if start >= 0 else None
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_."):
        start -= 1
    if start == end or keyword.iskeyword(text[start:end]):
        return None
    return start


# Puts a piece of code in at the cursor and tidies the spaces around it:
#   spaced   operators get one space either side
#   glued    goes straight after the code before it, like ) or **2
#   inside   puts the cursor this many characters back from the end of the piece, e.g. inside log()
def put(field, piece, spaced=False, glued=False, inside=0):
    before, after = field.text[:field.cursor], field.text[field.cursor:]
    if spaced:
        before = before.rstrip()
        before += " " if before else ""
        piece += " "
        after = after.lstrip()
    elif glued:
        before = before.rstrip()
    elif before and not before.endswith((" ", "(")) and not re.search(r"\w=$", before):
        before += " "
    field.set_text(before + piece + after, len(before) + len(piece) - inside)


# Writes the clicked operator or function in at the cursor (see TOOLS)
def use_tool(panel, button):
    field = panel.text_field
    tool = button.text
    if tool == "^2":
        put(field, "**2", glued=True)
    elif tool == "^":
        put(field, "**", spaced=True)
    elif tool in SPACED:
        put(field, tool, spaced=True)
    elif tool == ")":
        put(field, ")", glued=True)
    elif tool == ",":
        put(field, ", ", glued=True)
    elif tool == "by=":
        # Straight after an opening bracket or comma it starts the argument, otherwise it adds one
        previous = field.text[:field.cursor].rstrip()
        put(field, "by=" if previous.endswith(("(", ",")) else ", by=", glued=True)
    elif tool == "not":
        put(field, "not ")
    elif tool.endswith("()"):
        wrap(field, tool[:-2])
    else:
        put(field, tool)
    field.focused = True
    change_line(panel, field.text)


# Wraps the field or bracket just before the cursor in a function, e.g. rgdpe becomes log(rgdpe).
# With nothing to wrap, or only the new field's own name, the function goes in empty with the cursor inside
def wrap(field, function):
    text, cursor = field.text, field.cursor
    start = operand_start(text, cursor)
    name, expression = generate.split_line(text)
    if start is None or start < len(text) - len(expression):
        put(field, f"{function}()", inside=1)
        return
    operand = text[start:cursor]
    wrapped = f"{function}{operand}" if operand.startswith("(") else f"{function}({operand})"
    field.set_text(text[:start] + wrapped + text[cursor:], start + len(wrapped))


def insert_field(panel, button):
    field = panel.text_field
    put(field, generate.written(str(button.value)))
    field.focused = True
    change_line(panel, field.text)
