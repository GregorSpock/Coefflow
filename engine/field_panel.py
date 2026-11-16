from engine.fields import drop_downstream, rename_downstream


# Keep and rename controls for a module that lets the user drop and rename fields.
# Settings: "dropped" is a list of original field names, "renamed" maps an original name to its new one.
# The module sets panel.show_fields to its own function that redraws the panel.
# Renaming or dropping a field also updates the nodes further down the workflow that use it


# Adds the Keep all, Drop all and Reset names buttons to the top row
def add_field_buttons(panel):
    panel.renaming = None
    panel.add_button("Keep all", keep_all)
    panel.add_button("Drop all", drop_all)
    panel.add_button("Reset names", reset_names)


# What a field arriving under name is called on the way out, or None if it is dropped
def output_name(settings, name):
    if name in settings.get("dropped", []):
        return None
    return settings.get("renamed", {}).get(name, name)


def connections(panel):
    return panel.canvas.connections if panel.canvas else []


# Drops the dropped fields, then gives the rest their new names.
# Fields in the settings that are not in the data are ignored, so a changed file or input still runs
def apply_fields(data, settings):
    dropped = [c for c in settings.get("dropped", []) if c in data.columns]
    if len(dropped) == len(data.columns):
        raise ValueError("Every field is dropped")

    kept = data.drop(columns=dropped)
    renamed = {old: new for old, new in settings.get("renamed", {}).items() if old in kept.columns}
    kept = kept.rename(columns=renamed)

    duplicates = kept.columns[kept.columns.duplicated()]
    if len(duplicates):
        raise ValueError(f"More than one field is called {duplicates[0]}")
    return kept


# Adds a Keep and Rename button for every field, a line saying how many are kept,
# and a preview of the data with the fields dropped and renamed
def show_fields(panel, data):
    settings = panel.node.settings
    dropped = settings.get("dropped", [])
    renamed = settings.get("renamed", {})
    panel.columns = list(data.columns)

    kept = [c for c in data.columns if c not in dropped]
    panel.lines.append(f"Keeping {len(kept)} of {len(data.columns)} fields. Click Keep to drop a field, Rename to rename it")

    for column in data.columns:
        label = f"{column} -> {renamed[column]}" if column in renamed else column
        panel.add_option(label, [
            ("Keep", toggle_field, column not in dropped),
            ("Rename", toggle_rename, panel.renaming == column),
        ])
        for button in panel.options[-1][1]:
            button.value = column

    try:
        panel.preview = apply_fields(data, settings)
        panel.message = ""
    except ValueError as error:
        panel.preview = None
        panel.message = str(error)


# Keeps or drops the field that was clicked. A dropped field is taken out of the nodes below,
# e.g. out of a regression's variables. Keeping it again doesn't put it back there
def toggle_field(panel, button):
    settings = panel.node.settings
    dropped = settings.setdefault("dropped", [])
    column = button.value
    if column in dropped:
        dropped.remove(column)
    else:
        drop_downstream(panel.node, connections(panel), output_name(settings, column))
        dropped.append(column)
    if panel.renaming == column:
        panel.downstream_name = output_name(settings, column)
    panel.show_fields(panel)


# Opens the rename box for the field that was clicked, or closes it if that field was already being renamed
def toggle_rename(panel, button):
    column = button.value
    if panel.renaming == column:
        stop_renaming(panel)
    else:
        panel.renaming = column
        panel.downstream_name = output_name(panel.node.settings, column)
        current = panel.node.settings.get("renamed", {}).get(column, column)
        panel.add_text_field(f"Rename {column} to", current, rename_field, blocked="")
        panel.text_field.focused = True
    panel.show_fields(panel)


def stop_renaming(panel):
    panel.renaming = None
    panel.text_field = None


# Keeps the new name in the settings up to date as the user types, and the nodes below with it.
# An empty name or the original name means the field is not renamed.
# panel.downstream_name is what the nodes below call the field. While the name being typed
# matches another field's, they are left alone so the two fields don't get mixed up
def rename_field(panel, text):
    settings = panel.node.settings
    renamed = settings.setdefault("renamed", {})
    column = panel.renaming
    if text.strip() and text != column:
        renamed[column] = text
    else:
        renamed.pop(column, None)

    name = output_name(settings, column)
    others = {output_name(settings, c) for c in getattr(panel, "columns", []) if c != column}
    if name and panel.downstream_name and name != panel.downstream_name and name not in others:
        rename_downstream(panel.node, connections(panel), {panel.downstream_name: name})
        panel.downstream_name = name
    panel.show_fields(panel)


def keep_all(panel, button):
    panel.node.settings["dropped"] = []
    panel.show_fields(panel)


def drop_all(panel, button):
    settings = panel.node.settings
    for column in getattr(panel, "columns", []):
        if column not in settings.get("dropped", []):
            drop_downstream(panel.node, connections(panel), output_name(settings, column))
    settings["dropped"] = list(getattr(panel, "columns", []))
    panel.show_fields(panel)


# Gives every field its original name back, here and in the nodes below
def reset_names(panel, button):
    settings = panel.node.settings
    renames = {output_name(settings, c): c for c in settings.get("renamed", {}) if c not in settings.get("dropped", [])}
    rename_downstream(panel.node, connections(panel), renames)
    settings["renamed"] = {}
    stop_renaming(panel)
    panel.show_fields(panel)
