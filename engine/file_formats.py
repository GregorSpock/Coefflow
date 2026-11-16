import csv
import gzip
import json
import os
import re
import shlex
import sqlite3
import urllib.parse
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pandas as pd

from engine import latex


# Reading and writing tables in every file format Coefflow knows, kept in one place so Input Data,
# Output Data, the Database module and the command line all agree on what can be opened.
#
# A format that needs a package beyond the ones Coefflow always installs (SPSS needs pyreadstat,
# Parquet and Feather need pyarrow) says which one is missing when a file of that kind is opened,
# rather than stopping the app from starting.
#
# Some files hold more than one table: an Excel workbook has sheets and a SQLite database has
# tables. tables_in(path) lists them, and read_table takes the one to read as table=, the first
# being read when none is given

# Each kind of file: its name in file choosers, and the endings it is opened by
FORMATS = [
    ("CSV", [".csv"]),
    ("Text, tab or semicolon separated", [".tsv", ".tab", ".txt"]),
    ("Excel", [".xlsx", ".xlsm", ".xls"]),
    ("OpenDocument spreadsheet", [".ods"]),
    ("Stata", [".dta"]),
    ("SPSS", [".sav", ".zsav", ".por"]),
    ("SAS", [".sas7bdat", ".xpt"]),
    ("Parquet", [".parquet", ".pq"]),
    ("Feather", [".feather", ".arrow"]),
    ("JSON", [".json", ".jsonl", ".ndjson"]),
    ("gretl", [".gdt"]),
    ("SQLite database", [".sqlite", ".sqlite3", ".db"]),
]
READ_ENDINGS = [ending for name, endings in FORMATS for ending in endings]

# The kinds Output Data can save as: the setting's value, its ending and what it is called
WRITE_FORMATS = {
    "csv": (".csv", "CSV"),
    "xlsx": (".xlsx", "Excel"),
    "dta": (".dta", "Stata"),
    "parquet": (".parquet", "Parquet"),
    "json": (".json", "JSON"),
    "tex": (".tex", "LaTeX"),
}

# The packages a format needs, by ending, and what to tell the user to install
NEEDS = {
    ".sav": "pyreadstat", ".zsav": "pyreadstat", ".por": "pyreadstat",
    ".parquet": "pyarrow", ".pq": "pyarrow", ".feather": "pyarrow", ".arrow": "pyarrow",
    ".xls": "xlrd", ".ods": "odfpy",
}
SQLITE_ENDINGS = [".sqlite", ".sqlite3", ".db"]
EXCEL_ENDINGS = [".xlsx", ".xlsm", ".xls", ".ods"]


# The file chooser's list of types, e.g. "Data files (*.csv *.xlsx ...);;CSV (*.csv);;..."
def file_filter():
    every = " ".join(f"*{ending}" for ending in READ_ENDINGS)
    kinds = [f"{name} (" + " ".join(f"*{e}" for e in endings) + ")" for name, endings in FORMATS]
    return ";;".join([f"Data files ({every})"] + kinds + ["All files (*)"])


def ending_of(path):
    name = str(path).lower()
    if name.endswith(".gz") and not name.endswith(".tar.gz"):
        name = name[:-3]
    return Path(name).suffix


# Imports a package a format needs, or raises a ValueError saying how to install it
def need(package, ending):
    try:
        return __import__(package)
    except ImportError:
        raise ValueError(
            f"Opening {ending} files needs the {package} package. Install it with: pip install {package}"
        ) from None


# The sheets of a workbook or the tables of a database, in order, or [] for a file holding one table
def tables_in(path):
    ending = ending_of(path)
    if ending in SQLITE_ENDINGS:
        with closing_sqlite(path) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [row[0] for row in rows]
    if ending in EXCEL_ENDINGS:
        if ending in NEEDS:
            need(NEEDS[ending], ending)
        with pd.ExcelFile(path) as workbook:
            return [str(name) for name in workbook.sheet_names]
    return []


class closing_sqlite:
    # Opens a SQLite file read-only, so reading a database can never change it
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"No such file: {self.path}")
        self.connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        return self.connection

    def __exit__(self, *details):
        self.connection.close()


# Reads a table from a file of any known kind. rows reads only the first few, for a preview.
# table names the sheet or database table to read, the first being used when it is None
def read_table(path, rows=None, table=None):
    path = Path(path)
    ending = ending_of(path)
    if ending in NEEDS:
        need(NEEDS[ending], ending)

    if ending == ".csv":
        return pd.read_csv(path, nrows=rows)
    if ending in (".tsv", ".tab", ".txt"):
        return pd.read_csv(path, sep=sniff_separator(path), nrows=rows, engine="python")
    if ending in EXCEL_ENDINGS:
        return pd.read_excel(path, sheet_name=table if table is not None else 0, nrows=rows)
    if ending == ".dta":
        return read_stata(path, rows)
    if ending in (".sav", ".zsav"):
        return pd.read_spss(path, convert_categoricals=True).head(rows) if rows else pd.read_spss(path)
    if ending == ".por":
        import pyreadstat
        data, meta = pyreadstat.read_por(str(path), row_limit=rows or 0)
        return data
    if ending in (".sas7bdat", ".xpt"):
        return read_sas(path, ending, rows)
    if ending in (".parquet", ".pq"):
        data = pd.read_parquet(path)
        return data.head(rows) if rows else data
    if ending in (".feather", ".arrow"):
        data = pd.read_feather(path)
        return data.head(rows) if rows else data
    if ending in (".json", ".jsonl", ".ndjson"):
        return read_json(path, rows, lines=ending != ".json")
    if ending == ".gdt":
        data = read_gdt(path)
        return data.head(rows) if rows else data
    if ending in SQLITE_ENDINGS:
        names = tables_in(path)
        if not names:
            raise ValueError("The database has no tables")
        name = table if table in names else names[0]
        limit = f" LIMIT {int(rows)}" if rows else ""
        return read_sql(path, f'SELECT * FROM "{name.replace(chr(34), chr(34) * 2)}"{limit}')
    raise ValueError(
        f"Coefflow can't open {ending or 'files with no ending'} files. "
        f"It opens: {', '.join(READ_ENDINGS)}"
    )


# SAS files are read with pyreadstat when it is installed, as it reads both versions of the
# transport format (5 and 8) and keeps the field types; pandas, which reads only version 5, is
# used otherwise
def read_sas(path, ending, rows):
    try:
        import pyreadstat
    except ImportError:
        data = pd.read_sas(path, format="xport" if ending == ".xpt" else "sas7bdat", encoding="latin-1")
        return data.head(rows) if rows else data
    reader = pyreadstat.read_xport if ending == ".xpt" else pyreadstat.read_sas7bdat
    data, meta = reader(str(path), row_limit=rows or 0)
    return data


# Works out whether a text file is separated by tabs, semicolons, commas or pipes
def sniff_separator(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as file:
        sample = file.read(20000)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t;,|").delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


# Stata files keep value labels, which become the text categories they stand for, as they would
# in Stata's browser. A file whose labels don't cover every value is read with the plain numbers
def read_stata(path, rows):
    try:
        with pd.read_stata(path, iterator=True, convert_categoricals=True) as reader:
            return reader.read(rows) if rows else reader.read()
    except ValueError:
        with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
            return reader.read(rows) if rows else reader.read()


# A JSON file is either a list of records ([{"a": 1}, ...]), a table pandas wrote in any of its
# orients, or an object holding such a list under one key (e.g. {"data": [...]}). Nested objects
# are spread into columns such as address.city. JSON Lines files hold one record per line
def read_json(path, rows, lines):
    if lines:
        return pd.read_json(path, lines=True, nrows=rows)
    with open(path, encoding="utf-8") as file:
        contents = json.load(file)

    if isinstance(contents, dict):
        lists = [value for value in contents.values() if isinstance(value, list)]
        if "columns" in contents and "data" in contents:
            data = pd.DataFrame(contents["data"], columns=contents["columns"])
            return data.head(rows) if rows else data
        if len(lists) == 1 and all(isinstance(item, dict) for item in lists[0]):
            contents = lists[0]
        elif all(isinstance(value, dict) for value in contents.values()):
            data = pd.DataFrame(contents)
            return data.head(rows) if rows else data
        elif all(isinstance(value, list) for value in contents.values()):
            data = pd.DataFrame(contents)
            return data.head(rows) if rows else data
        else:
            contents = [contents]

    if not isinstance(contents, list):
        raise ValueError("The JSON file doesn't hold a table")
    if rows:
        contents = contents[:rows]
    if contents and all(isinstance(item, dict) for item in contents):
        return pd.json_normalize(contents)
    return pd.DataFrame(contents)


# Reads a gretl data file (.gdt), which is XML, optionally gzipped. Each <variable> is a column
# and each <obs> a row of values separated by spaces, NA meaning missing. A series of text
# values is stored as codes 1, 2, ... with the text in a <valueset>, and is read as the text.
# Observation labels, such as 1990:1 for time series, come out as an "obs" column first
def read_gdt(path):
    with open(path, "rb") as file:
        start = file.read(2)
    opener = gzip.open if start == b"\x1f\x8b" else open
    with opener(path, "rb") as file:
        root = ElementTree.parse(file).getroot()
    if root.tag != "gretldata":
        raise ValueError("This is not a gretl data file")

    variables = root.find("variables")
    names = [variable.get("name") for variable in variables.findall("variable")] if variables is not None else []
    if not names:
        raise ValueError("The gretl file has no variables")

    labels, values = [], []
    observations = root.find("observations")
    for obs in observations.findall("obs") if observations is not None else []:
        labels.append(obs.get("label"))
        cells = shlex.split(obs.text or "")
        if len(cells) != len(names):
            raise ValueError(f"An observation in the gretl file has {len(cells)} values for {len(names)} variables")
        values.append(cells)

    data = pd.DataFrame(values, columns=names, dtype=object)
    for name in names:
        numbers = pd.to_numeric(data[name].replace({"NA": np.nan, "na": np.nan, "nan": np.nan, ".": np.nan}), errors="coerce")
        text_cells = data[name].notna() & ~data[name].isin(["NA", "na", "nan", "."])
        data[name] = numbers if numbers[text_cells].notna().all() else data[name]

    strings = root.find("string-tables")
    for valueset in strings.findall("valueset") if strings is not None else []:
        name = valueset.get("name")
        table = valueset.find("strings")
        if name not in data.columns or table is None:
            continue
        words = shlex.split(table.text or "")
        codes = data[name]
        if pd.api.types.is_numeric_dtype(codes):
            data[name] = codes.map(lambda code: words[int(code) - 1] if pd.notna(code) and 1 <= code <= len(words) else np.nan)

    if any(label is not None for label in labels):
        data.insert(0, "obs", labels)
    elif root.get("type") == "time-series" and root.get("startobs"):
        data.insert(0, "obs", time_labels(root.get("startobs"), int(root.get("frequency", "1")), len(data)))
    return data


# The labels gretl gives a time series' rows: years, or year:period for quarterly and monthly data
def time_labels(start, frequency, count):
    if ":" not in start or frequency <= 1:
        try:
            first = int(start)
        except ValueError:
            return [f"{start}+{i}" for i in range(count)]
        return [str(first + i) for i in range(count)]
    year, period = (int(part) for part in start.split(":"))
    width = 2 if frequency >= 10 else 1
    labels = []
    for _ in range(count):
        labels.append(f"{year}:{period:0{width}d}")
        period += 1
        if period > frequency:
            year, period = year + 1, 1
    return labels


# The only statements a query may be: ones that read. A WITH query is allowed as it usually
# reads too, but not if it holds any of the words that change a database
READING = ("select", "with", "values", "table")
CHANGING = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "merge", "grant",
    "revoke", "exec", "execute", "attach", "detach", "pragma", "vacuum", "into",
]
# The environment variable a server's password is read from when it hasn't been typed this session
PASSWORD_VARIABLE = "COEFFLOW_DB_PASSWORD"
# The passwords typed this session, by server address without its password. They are kept here,
# in memory only, and never in a node's settings, so they are never saved in a workflow file
passwords = {}


# The query with its comments and the text inside its quotes taken out, in lower case, so
# its words can be checked without being fooled by a word inside a string
def bare_query(query):
    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", query, flags=re.DOTALL)
    text = re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]", " ", text, flags=re.DOTALL)
    return text.lower()


# Checks a query only reads: one statement, starting with SELECT (or WITH, VALUES or TABLE), with
# none of the words that add, change or remove anything. Raises ValueError saying what is wrong.
# A workflow can come from someone else, so a query in it must never be able to change a database
def check_query(query):
    query = query.strip().rstrip(";").strip()
    if not query:
        raise ValueError("Type a query, e.g. SELECT * FROM table_name")
    bare = bare_query(query)
    if ";" in bare:
        raise ValueError("Only one statement can be run: take out everything after the first ;")
    words = re.findall(r"[a-z_]+", bare)
    if not words or words[0] not in READING:
        raise ValueError("Only queries that read data can be run: start the query with SELECT")
    for word in CHANGING:
        if word in words:
            raise ValueError(f"Only queries that read data can be run, so {word.upper()} can't be used")
    return query


# Takes the password out of a server address. Returns the address without it, and the password,
# or None if it had none: postgresql://me:secret@host/db gives postgresql://me@host/db and "secret"
def split_password(url):
    url = str(url or "").strip()
    if "://" not in url or "@" not in url:
        return url, None
    scheme, rest = url.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    if ":" not in credentials or "/" in credentials:
        return url, None
    user, password = credentials.split(":", 1)
    return f"{scheme}://{user}@{host}", password


# Remembers the password in a server address for this session and returns the address without
# it, which is the only form a node's settings should ever hold
def remember_password(url):
    plain, password = split_password(url)
    if password:
        passwords[plain] = password
    return plain


# The address to connect with: the one given, with the password typed this session put back,
# or else the one in the COEFFLOW_DB_PASSWORD environment variable
def connection_url(url):
    plain = remember_password(url)
    password = passwords.get(plain)
    if password is None and os.environ.get(PASSWORD_VARIABLE):
        password = urllib.parse.quote_plus(os.environ[PASSWORD_VARIABLE])
    if password is None or "://" not in plain or "@" not in plain:
        return plain
    scheme, rest = plain.split("://", 1)
    user, host = rest.rsplit("@", 1)
    return f"{scheme}://{user}:{password}@{host}"


# Runs a query against a database and returns its result. Only a query that reads is run (see
# check_query). A path to a SQLite file is opened read-only. Anything else is taken as a
# SQLAlchemy connection URL, e.g. postgresql://user@host/dbname, which needs the sqlalchemy
# package and the database's driver; whatever the query does there is undone when it ends.
# rows fetches only that many rows from the database, for a preview, rather than every row
def read_sql(source, query, rows=None):
    query = check_query(query)
    if is_url(source):
        try:
            import sqlalchemy
        except ImportError:
            raise ValueError(
                "Connecting to a database server needs the sqlalchemy package and the database's own driver. "
                "Install them with, e.g.: pip install sqlalchemy psycopg2-binary"
            ) from None
        engine = sqlalchemy.create_engine(connection_url(source))
        try:
            with engine.connect() as connection:
                try:
                    result = connection.execution_options(stream_results=bool(rows)).execute(sqlalchemy.text(query))
                    names = list(result.keys())
                    records = result.fetchmany(rows) if rows else result.fetchall()
                    result.close()
                finally:
                    connection.rollback()
        except sqlalchemy.exc.SQLAlchemyError as error:
            raise ValueError(without_passwords(str(getattr(error, "orig", None) or error))) from None
        finally:
            engine.dispose()
    else:
        with closing_sqlite(source) as connection:
            cursor = connection.execute(query)
            names = [column[0] for column in cursor.description or []]
            records = cursor.fetchmany(rows) if rows else cursor.fetchall()
    return pd.DataFrame.from_records([tuple(record) for record in records], columns=names)


# A message with any password known this session blanked out, as a driver's error can quote
# the address it failed to reach
def without_passwords(text):
    hidden = list(passwords.values())
    if os.environ.get(PASSWORD_VARIABLE):
        hidden += [os.environ[PASSWORD_VARIABLE], urllib.parse.quote_plus(os.environ[PASSWORD_VARIABLE])]
    for password in hidden:
        if password:
            text = text.replace(password, "***")
    return text


def is_url(source):
    return "://" in str(source)


# Writes a table to a file of the given kind, one of WRITE_FORMATS. Stata needs short plain
# field names and text columns of no more than 2045 characters, so names are tidied to suit
def write_table(data, path, kind, title=""):
    path = Path(path)
    if kind == "xlsx":
        data.to_excel(path, index=False)
    elif kind == "dta":
        stata_ready(data).to_stata(path, write_index=False, version=118)
    elif kind == "parquet":
        need("pyarrow", ".parquet")
        tidy_columns(data).to_parquet(path, index=False)
    elif kind == "json":
        data.to_json(path, orient="records", indent=1, date_format="iso", force_ascii=False)
    elif kind == "tex":
        path.write_text(latex.table(data, caption=title), encoding="utf-8")
    else:
        data.to_csv(path, index=False)


# Column names as text, as Parquet and Stata need
def tidy_columns(data):
    data = data.copy()
    data.columns = [str(column) for column in data.columns]
    return data


# A copy of the table Stata can store: names of letters, digits and underscores up to 32
# characters, not starting with a digit and not repeated; text in object columns; dates as dates
def stata_ready(data):
    data = tidy_columns(data)
    names, seen = [], set()
    for column in data.columns:
        name = "".join(c if c.isalnum() or c == "_" else "_" for c in column)[:32] or "v"
        if name[0].isdigit():
            name = f"_{name}"[:32]
        base, number = name, 2
        while name.lower() in seen:
            suffix = f"_{number}"
            name = base[:32 - len(suffix)] + suffix
            number += 1
        seen.add(name.lower())
        names.append(name)
    data.columns = names
    for column in data.columns:
        values = data[column]
        if values.dtype == bool:
            data[column] = values.astype(int)
        elif values.dtype == object or isinstance(values.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(values):
            data[column] = values.astype(object).where(values.notna(), "").astype(str)
    return data
