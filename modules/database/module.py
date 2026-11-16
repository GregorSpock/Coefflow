from engine import file_formats
from engine.dialogs import choose_file
from engine.panel import Panel


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Data", "the rows the SQL query gives back, with a column for each field it selects"),
]

# The text box edits either the query or the server's address, switched with the Edit button
EDITING = ["Query", "Server URL"]
PREVIEW_ROWS = 10


# A database on a server can change between runs, so its query is always run again. A SQLite
# file is run again only when the file or the query changes
def ALWAYS_RUN(settings):
    return bool(settings.get("url")) and not settings.get("path")


# A server's password is never kept in the settings, as they are saved in the workflow file. One
# found in the address, as in a workflow saved before this was so, is moved to this session's memory
def set_defaults(settings):
    settings.setdefault("query", "")
    settings.setdefault("url", "")
    settings.setdefault("editing", EDITING[0])
    settings["url"] = file_formats.remember_password(settings["url"])


# Opens a panel to choose a SQLite file or type a server's address, and type the query
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Database")
    panel.add_button("SQLite file", select_file)
    panel.add_button(f"Edit: {node.settings['editing']}", switch_editing)
    panel.add_button("Tables", list_tables)
    panel.add_button("Preview", preview)
    add_box(panel)
    panel.refresh = show_query
    show_query(panel)
    return panel


def add_box(panel):
    settings = panel.node.settings
    if settings["editing"] == "Query":
        panel.add_text_field("SQL", settings["query"], type_query, blocked="", fill=True)
    else:
        panel.add_text_field("URL", settings["url"], type_url, blocked="", fill=True)


def select_file(panel, button):
    path = choose_file()
    if path:
        panel.node.settings["path"] = path
        panel.node.settings["url"] = ""
        show_query(panel)


def switch_editing(panel, button):
    settings = panel.node.settings
    settings["editing"] = EDITING[(EDITING.index(settings["editing"]) + 1) % len(EDITING)]
    button.text = f"Edit: {settings['editing']}"
    add_box(panel)
    show_query(panel)


# Typing only keeps the query. It is not run until Preview is clicked or the workflow runs, so a
# half-typed query is never sent to the database and a large table is not read at every keystroke
def type_query(panel, text):
    panel.node.settings["query"] = text
    show_query(panel)


# A server's address replaces the SQLite file. A password typed in it is remembered for this
# session only: the settings, and so the workflow file, keep the address without it
def type_url(panel, text):
    settings = panel.node.settings
    settings["url"] = file_formats.remember_password(text)
    if settings["url"]:
        settings.pop("path", None)
    show_query(panel)


# Runs the query for its first few rows
def preview(panel, button):
    show_query(panel, run_query=True)


# Fills the query with a SELECT of the first table, as a start, and lists the tables
def list_tables(panel, button):
    settings = panel.node.settings
    try:
        names = tables(settings)
    except Exception as error:
        panel.message = f"Could not list the tables: {error}"
        panel.preview = None
        return
    panel.note = f"Tables: {', '.join(names) or 'none'}"
    if names and not settings["query"].strip():
        settings["query"] = f'SELECT * FROM "{names[0]}"'
        add_box(panel)
    show_query(panel, keep_note=True, run_query=True)


def tables(settings):
    if settings.get("path"):
        return file_formats.tables_in(settings["path"])
    if settings.get("url"):
        import sqlalchemy
        engine = sqlalchemy.create_engine(file_formats.connection_url(settings["url"]))
        try:
            return sqlalchemy.inspect(engine).get_table_names()
        finally:
            engine.dispose()
    raise ValueError("Choose a SQLite file or type a server URL first")


def source(settings):
    return settings.get("path") or settings.get("url") or ""


# Where the data comes from, as shown in the panel. A server address never shows a password
def shown_source(settings):
    if settings.get("path"):
        return settings["path"]
    url, password = file_formats.split_password(settings.get("url", ""))
    return url or "none chosen"


# Says whether a password is known for the server, and how to give one
def password_note(settings):
    url = file_formats.split_password(settings.get("url", ""))[0]
    if file_formats.connection_url(url) != url:
        known = "A password is set for this session."
    else:
        known = "No password is set."
    return (
        f"{known} A password typed in the address (postgresql://user:password@host/dbname) is used for this session "
        f"and never saved in the workflow; or put it in the {file_formats.PASSWORD_VARIABLE} environment variable. "
        "Servers need the sqlalchemy package and the database's driver."
    )


# Shows where the data comes from and whether the query can be run. With run_query, as when
# Preview is clicked, the query is run for its first rows; otherwise the rows of the last run
# are shown, if there are any
def show_query(panel, keep_note=False, run_query=False):
    node = panel.node
    settings = node.settings
    panel.lines = [f"Database: {shown_source(settings)}"]
    if not keep_note:
        panel.note = password_note(settings) if settings.get("url") else ""
    panel.preview = None
    if not source(settings):
        panel.message = "Click SQLite file to choose a database, or Edit: Server URL to type a server's address"
        return
    if not settings["query"].strip():
        panel.message = "Type a query, e.g. SELECT * FROM table_name, or click Tables to start from the first table"
        return
    try:
        file_formats.check_query(settings["query"])
    except ValueError as error:
        panel.message = str(error)
        return

    if run_query:
        try:
            panel.preview = file_formats.read_sql(source(settings), settings["query"], rows=PREVIEW_ROWS)
            panel.preview_title = f"Data Preview (first {PREVIEW_ROWS} rows)"
            panel.message = ""
        except Exception as error:
            panel.message = f"The query failed: {error}"
        return

    ran = getattr(node, "last_outputs", None)
    if ran and getattr(node, "status", None) == "done":
        panel.preview = ran[0]
        panel.preview_title = "Data"
        panel.message = ""
    else:
        panel.message = "Click Preview to see the first rows, or run the workflow"


# Runs the query against the SQLite file (opened read-only) or the server, and gives its rows.
# Only a query that reads is run (see file_formats.check_query)
def run(inputs, settings):
    set_defaults(settings)
    if not source(settings):
        raise ValueError("Choose a SQLite file or type a server URL")
    return [file_formats.read_sql(source(settings), settings["query"])]
