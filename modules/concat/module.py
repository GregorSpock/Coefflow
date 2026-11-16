import pandas as pd

from engine.panel import Panel


# Takes as many tables as are connected: a new empty input appears whenever every input is used
GROWING_INPUTS = True
REQUIRED_INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Tables", "any tables to stack on top of each other, one per input. In 1's rows come first, then In 2's and so on"),
]
OUTPUT_SPECS = [
    ("Data", "the rows of every input, one table after another, with the fields matched by name"),
]

# Which fields come out: every field any table has, the missing ones left empty, or only those all have
FIELDS = ["All fields", "Shared fields"]
SOURCE = "source"


def set_defaults(settings):
    settings.setdefault("fields", FIELDS[0])
    settings.setdefault("source", False)
    settings.setdefault("dedupe", False)


# Opens a panel to choose which fields to keep, whether to note where each row came from and
# whether to drop repeated rows
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Concat")
    panel.add_button(node.settings["fields"], switch_fields)
    panel.add_button(source_label(node.settings), toggle_source)
    panel.add_button(dedupe_label(node.settings), toggle_dedupe)
    panel.refresh = show_concat
    show_concat(panel)
    return panel


def source_label(settings):
    return "Source column: On" if settings.get("source") else "Source column: Off"


def dedupe_label(settings):
    return "Drop repeats: On" if settings.get("dedupe") else "Drop repeats: Off"


def switch_fields(panel, button):
    settings = panel.node.settings
    settings["fields"] = FIELDS[(FIELDS.index(settings["fields"]) + 1) % len(FIELDS)]
    button.text = settings["fields"]
    show_concat(panel)


def toggle_source(panel, button):
    settings = panel.node.settings
    settings["source"] = not settings["source"]
    button.text = source_label(settings)
    show_concat(panel)


def toggle_dedupe(panel, button):
    settings = panel.node.settings
    settings["dedupe"] = not settings["dedupe"]
    button.text = dedupe_label(settings)
    show_concat(panel)


# Says how many rows and fields each input brings, and previews the result
def show_concat(panel):
    node = panel.node
    panel.preview = None
    tables = [table for table in node.last_inputs or [] if table is not None]
    if not tables:
        panel.lines = ["Connect tables and run the workflow to stack them"]
        panel.message = ""
        return
    panel.lines = [
        "    ".join(f"In {i + 1}: {len(t):,} rows, {len(t.columns)} fields" for i, t in enumerate(tables)),
    ]
    try:
        panel.preview = run(node.last_inputs, node.settings)[0]
        panel.message = ""
    except Exception as error:
        panel.message = f"Could not stack the tables: {error}"
    panel.note = ""
    fields = [set(map(str, t.columns)) for t in tables]
    unshared = sorted(set.union(*fields) - set.intersection(*fields))
    if unshared:
        panel.note = f"Not in every table: {', '.join(unshared[:12])}{'...' if len(unshared) > 12 else ''}"


# Stacks the connected tables in input order, matching fields by name. With Shared fields only
# the fields every table has are kept. A source column can say which input each row came from
# (1, 2, ...), and repeated rows can be dropped, keeping the first
def run(inputs, settings):
    set_defaults(settings)
    tables = [(number, table) for number, table in enumerate(inputs, start=1) if table is not None]
    if not tables:
        raise ValueError("Connect at least one table")

    join = "inner" if settings["fields"] == FIELDS[1] else "outer"
    frames = []
    for number, table in tables:
        if table.columns.duplicated().any():
            raise ValueError(f"In {number} has more than one field with the same name")
        frame = table.reset_index(drop=True)
        if settings["source"]:
            frame = frame.assign(**{SOURCE: number})
        frames.append(frame)

    order = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    result = pd.concat(frames, join=join, ignore_index=True, sort=False)
    result = result[[column for column in order if column in result.columns]]
    if settings["source"]:
        result = result[[SOURCE] + [c for c in result.columns if c != SOURCE]]
    if settings["dedupe"]:
        subset = [c for c in result.columns if c != SOURCE]
        result = result.drop_duplicates(subset=subset).reset_index(drop=True)
    return [result]
