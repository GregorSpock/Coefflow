# Keeps the settings of downstream nodes in step when a field is renamed or dropped upstream.
# A module lists the settings that hold field names in FIELD_SETTINGS. Each can hold one name,
# a list of names, or a dict keyed by name (e.g. a lag per field).
# A module can also have:
#   rename_field(settings, old, new) for names held some other way, e.g. inside a condition
#   output_name(settings, name) giving what a field arriving under name is called on the way out,
#   or None if the module drops it. Without one, fields go out under the name they came in with


# What a node's module calls an incoming field on the way out
def output_name(node, name):
    if hasattr(node.module, "output_name"):
        return node.module.output_name(node.settings, name)
    return name


# Swaps a name for another in one setting value. new=None takes the name out
def replace_name(value, old, new):
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, list):
        names = [new if v == old else v for v in value]
        return list(dict.fromkeys(n for n in names if n is not None))
    if isinstance(value, dict):
        return {(new if k == old else k): v for k, v in value.items() if not (k == old and new is None)}
    return value


def update_settings(node, old, new):
    for key in getattr(node.module, "FIELD_SETTINGS", []):
        if key in node.settings:
            node.settings[key] = replace_name(node.settings[key], old, new)
    if new is not None and hasattr(node.module, "rename_field"):
        node.module.rename_field(node.settings, old, new)


# Follows the connections out of node, giving every node that uses the field old the name new instead.
# A node passes the change on under whatever it calls the field, and stops it if it renames or drops the field itself
def rename_one(node, connections, old, new, seen):
    for connection in connections:
        target = connection.target
        if connection.source is not node or (target, old) in seen or target.module is None:
            continue
        seen.add((target, old))

        before = output_name(target, old)
        update_settings(target, old, new)
        after = output_name(target, new)
        if before and after and before != after:
            rename_one(target, connections, before, after, seen)


# Renames fields below node from each old name to its new one, e.g. {"gdp": "output"}.
# Goes through stand-in names first so swapping two names doesn't merge them
def rename_downstream(node, connections, renames):
    renames = {old: new for old, new in renames.items() if old != new}
    stand_ins = {old: f"\0renaming{i}" for i, old in enumerate(renames)}
    for old, stand_in in stand_ins.items():
        rename_one(node, connections, old, stand_in, set())
    for old, stand_in in stand_ins.items():
        rename_one(node, connections, stand_in, renames[old], set())


# Takes a field node no longer gives out out of every node below it
def drop_downstream(node, connections, name, seen=None):
    seen = set() if seen is None else seen
    for connection in connections:
        target = connection.target
        if connection.source is not node or (target, name) in seen or target.module is None:
            continue
        seen.add((target, name))

        onward = output_name(target, name)
        update_settings(target, name, None)
        if onward:
            drop_downstream(target, connections, onward, seen)
