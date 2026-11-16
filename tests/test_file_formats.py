import gzip
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from engine import file_formats, latex
from engine.node import Node
from tests.support import SAMPLE_DATA


# Every file format Coefflow reads and writes, checked by writing a table and reading it back,
# and the LaTeX it writes for papers

TABLE = pd.DataFrame({
    "region": ["North", "South", "East", None],
    "year": [2020, 2021, 2022, 2023],
    "gdp_growth": [1.5, -0.25, 3.0, np.nan],
})


@pytest.mark.parametrize("kind", ["csv", "xlsx", "dta", "parquet", "json"])
def test_written_tables_read_back_the_same(kind, tmp_path):
    path = tmp_path / f"table{file_formats.WRITE_FORMATS[kind][0]}"
    file_formats.write_table(TABLE, path, kind)
    back = file_formats.read_table(path)
    assert list(back.columns) == list(TABLE.columns)
    assert back["year"].tolist() == TABLE["year"].tolist()
    assert np.allclose(back["gdp_growth"].to_numpy(float), TABLE["gdp_growth"].to_numpy(float), equal_nan=True)
    assert back["region"].iloc[:3].tolist() == ["North", "South", "East"]


def test_a_preview_reads_only_the_first_rows(tmp_path):
    for kind in ["csv", "dta", "parquet", "json"]:
        path = tmp_path / f"t{file_formats.WRITE_FORMATS[kind][0]}"
        file_formats.write_table(TABLE, path, kind)
        assert len(file_formats.read_table(path, rows=2)) == 2


def test_stata_names_are_made_safe(tmp_path):
    awkward = pd.DataFrame({"GDP per head (£)": [1.0], "1st": [2.0], "gdp per head (£)": [3.0]})
    path = tmp_path / "awkward.dta"
    file_formats.write_table(awkward, path, "dta")
    names = list(file_formats.read_table(path).columns)
    assert len(set(n.lower() for n in names)) == 3
    assert all(n[0].isalpha() or n[0] == "_" for n in names)


def test_stata_value_labels_become_text(tmp_path):
    path = tmp_path / "labelled.dta"
    data = pd.DataFrame({"sex": pd.Categorical(["female", "male", "female"])})
    data.to_stata(path, write_index=False)
    assert file_formats.read_table(path)["sex"].astype(str).tolist() == ["female", "male", "female"]


def test_spss_files_are_read(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    path = tmp_path / "survey.sav"
    pyreadstat.write_sav(TABLE.fillna({"region": ""}), str(path))
    back = file_formats.read_table(path)
    assert back["year"].tolist() == [2020, 2021, 2022, 2023]


def test_sas_transport_files_are_read(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    path = tmp_path / "study.xpt"
    pyreadstat.write_xport(TABLE[["year", "gdp_growth"]], str(path))
    assert file_formats.read_table(path)["year"].tolist() == [2020, 2021, 2022, 2023]


@pytest.mark.parametrize("separator, ending", [("\t", ".tsv"), (";", ".txt"), ("|", ".txt")])
def test_text_files_find_their_separator(separator, ending, tmp_path):
    path = tmp_path / f"data{ending}"
    TABLE.to_csv(path, sep=separator, index=False)
    assert list(file_formats.read_table(path).columns) == list(TABLE.columns)


def test_json_in_its_usual_shapes(tmp_path):
    records = [{"id": 1, "place": {"city": "Leeds"}}, {"id": 2, "place": {"city": "York"}}]
    shapes = {
        "records.json": records,
        "wrapped.json": {"data": records},
        "split.json": {"columns": ["id", "x"], "data": [[1, 2.0], [2, 3.0]]},
        "columns.json": {"id": [1, 2], "x": [2.0, 3.0]},
    }
    for name, contents in shapes.items():
        (tmp_path / name).write_text(json.dumps(contents), encoding="utf-8")
        data = file_formats.read_table(tmp_path / name)
        assert data["id"].tolist() == [1, 2], name
    assert file_formats.read_table(tmp_path / "records.json")["place.city"].tolist() == ["Leeds", "York"]
    (tmp_path / "lines.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert len(file_formats.read_table(tmp_path / "lines.jsonl")) == 2


GRETL = """<?xml version="1.0" encoding="UTF-8"?>
<gretldata version="1.4" name="demo" frequency="4" startobs="1990:1" endobs="1990:4" type="time-series">
<variables count="3">
<variable name="gdp" label="real output"/>
<variable name="unemp"/>
<variable name="region" discrete="true"/>
</variables>
<observations count="4" labels="false">
<obs>1.5 7.0 1 </obs>
<obs>NA 7.2 2 </obs>
<obs>2.5 NA 1 </obs>
<obs>3.0 6.9 2 </obs>
</observations>
<string-tables count="1">
<valueset name="region">
<strings count="2">"North" "South West"</strings>
</valueset>
</string-tables>
</gretldata>
"""


def test_gretl_files_are_read_with_their_text_values_and_dates(tmp_path):
    path = tmp_path / "demo.gdt"
    path.write_text(GRETL, encoding="utf-8")
    data = file_formats.read_table(path)
    assert data.columns.tolist() == ["obs", "gdp", "unemp", "region"]
    assert data["obs"].tolist() == ["1990:1", "1990:2", "1990:3", "1990:4"]
    assert np.isnan(data["gdp"].iloc[1]) and data["gdp"].iloc[3] == 3.0
    assert data["region"].tolist() == ["North", "South West", "North", "South West"]

    zipped = tmp_path / "zipped.gdt"
    zipped.write_bytes(gzip.compress(GRETL.encode("utf-8")))
    assert file_formats.read_table(zipped).equals(data)


def make_database(path):
    with sqlite3.connect(path) as connection:
        TABLE.to_sql("growth", connection, index=False)
        pd.DataFrame({"region": ["North", "South"], "capital": ["Leeds", "Exeter"]}).to_sql("places", connection, index=False)
    return path


def test_sqlite_tables_and_queries(tmp_path):
    path = make_database(tmp_path / "economy.sqlite")
    assert file_formats.tables_in(path) == ["growth", "places"]
    assert len(file_formats.read_table(path)) == 4
    assert file_formats.read_table(path, table="places")["capital"].tolist() == ["Leeds", "Exeter"]
    joined = file_formats.read_sql(path, "SELECT g.year, p.capital FROM growth g JOIN places p USING (region) ORDER BY year")
    assert joined["capital"].tolist() == ["Leeds", "Exeter"]


# A query can't change the database, as it is opened read-only
def test_sqlite_is_opened_read_only(tmp_path):
    path = make_database(tmp_path / "economy.db")
    with pytest.raises(Exception):
        file_formats.read_sql(path, "DROP TABLE growth")
    assert file_formats.tables_in(path) == ["growth", "places"]


def test_server_urls_explain_what_they_need(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_sqlalchemy(name, *args, **kwargs):
        if name == "sqlalchemy":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sqlalchemy)
    with pytest.raises(ValueError, match="sqlalchemy"):
        file_formats.read_sql("postgresql://me@host/db", "SELECT 1")


def test_unknown_files_say_what_can_be_opened(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_text("x")
    with pytest.raises(ValueError, match=".dta"):
        file_formats.read_table(path)


# Input Data reads the new formats and steps through a workbook's sheets
def test_input_data_reads_other_formats_and_sheets(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        TABLE.to_excel(writer, sheet_name="first", index=False)
        TABLE.head(2).to_excel(writer, sheet_name="second", index=False)
    node = Node("Input Data", 0, 0, 1)
    node.settings = {"path": str(path)}
    assert len(node.module.run([], node.settings)[0]) == 4

    panel = node.on_click()
    sheet = panel.button("Sheet: first")
    assert sheet is not None
    panel.click(sheet)
    assert node.settings["table"] == "second"
    assert len(node.module.run([], node.settings)[0]) == 2


def test_database_module_runs_its_query(tmp_path):
    path = make_database(tmp_path / "economy.sqlite")
    node = Node("Database", 0, 0, 1)
    node.settings = {"path": str(path)}
    panel = node.on_click()
    panel.click(panel.button("Tables"))
    assert node.settings["query"] == 'SELECT * FROM "growth"'
    assert panel.preview is not None and len(panel.preview) == 4
    assert len(node.module.run([], node.settings)[0]) == 4
    assert not node.module.ALWAYS_RUN(node.settings)
    url_settings = {"url": "postgresql://me:secret@host/db"}
    assert node.module.ALWAYS_RUN(url_settings)
    assert "secret" not in node.module.shown_source(url_settings)


# Only a query that reads is ever run, whatever database it is for, as a workflow may come
# from someone else
@pytest.mark.parametrize("query", [
    "SELECT * FROM growth",
    "select region, year from growth where gdp > 2;",
    "WITH recent AS (SELECT * FROM growth WHERE year > 2000) SELECT * FROM recent",
    "SELECT 'delete; drop table' AS words, \"update\" FROM growth -- insert a comment",
    "SELECT REPLACE(region, 'North', 'N') AS region, updated_at, created FROM growth /* drop */",
])
def test_reading_queries_are_allowed(query):
    assert file_formats.check_query(query) == query.strip().rstrip(";")


@pytest.mark.parametrize("query, message", [
    ("", "Type a query"),
    ("DELETE FROM growth", "start the query with SELECT"),
    ("DROP TABLE growth", "start the query with SELECT"),
    ("UPDATE growth SET gdp = 0", "start the query with SELECT"),
    ("SELECT 1; DROP TABLE growth", "Only one statement"),
    ("WITH gone AS (DELETE FROM growth RETURNING *) SELECT * FROM gone", "DELETE can't be used"),
    ("SELECT * INTO backup FROM growth", "INTO can't be used"),
    ("-- SELECT\nTRUNCATE growth", "start the query with SELECT"),
])
def test_queries_that_change_a_database_are_refused(query, message):
    with pytest.raises(ValueError, match=message):
        file_formats.check_query(query)


# A preview asks the database for its first rows only, rather than reading every row and
# keeping a few
def test_a_preview_fetches_only_its_rows(tmp_path):
    path = tmp_path / "big.sqlite"
    with sqlite3.connect(path) as connection:
        pd.DataFrame({"n": range(5000)}).to_sql("numbers", connection, index=False)
    preview = file_formats.read_sql(path, "SELECT n FROM numbers ORDER BY n", rows=10)
    assert list(preview["n"]) == list(range(10))
    assert len(file_formats.read_sql(path, "SELECT n FROM numbers")) == 5000
    empty = file_formats.read_sql(path, "SELECT n FROM numbers WHERE n < 0")
    assert list(empty.columns) == ["n"] and empty.empty


# Typing in the query box never runs the query: only Preview, Tables and a run do
def test_typing_a_query_does_not_run_it(tmp_path, monkeypatch):
    path = make_database(tmp_path / "economy.sqlite")
    node = Node("Database", 0, 0, 1)
    node.settings = {"path": str(path)}
    panel = node.on_click()
    calls = []
    real = file_formats.read_sql
    monkeypatch.setattr(file_formats, "read_sql", lambda *a, **k: calls.append(a) or real(*a, **k))

    for text in ["S", "SELECT", "SELECT * FROM growth", "DELETE FROM growth"]:
        panel.text_field.on_change(panel, text)
    assert calls == [] and panel.preview is None
    assert "start the query with SELECT" in panel.message

    panel.text_field.on_change(panel, "SELECT * FROM growth WHERE year > 2000")
    assert panel.message == "Click Preview to see the first rows, or run the workflow"
    panel.click(panel.button("Preview"))
    assert len(calls) == 1 and panel.preview is not None

    with pytest.raises(ValueError, match="start the query with SELECT"):
        node.module.run([], {"path": str(path), "query": "DROP TABLE growth"})
    assert file_formats.tables_in(path) == ["growth", "places"]


# A server's password is used for the session but never kept in the settings, so it can't reach
# a workflow file, the undo history or the clipboard
def test_a_password_is_never_saved(tmp_path, monkeypatch):
    from engine import workspace
    from engine.workflow import Workflow

    monkeypatch.setattr(file_formats, "passwords", {})
    monkeypatch.delenv(file_formats.PASSWORD_VARIABLE, raising=False)
    workflow = Workflow()
    node = workflow.add_node("Database", 0, 0)
    panel = node.on_click()
    panel.click(panel.button("Edit: Query"))
    panel.text_field.on_change(panel, "postgresql://me:s3cret@host:5432/db")

    assert node.settings["url"] == "postgresql://me@host:5432/db"
    assert "s3cret" not in str(node.settings) and "s3cret" not in " ".join(panel.lines)
    assert file_formats.connection_url(node.settings["url"]) == "postgresql://me:s3cret@host:5432/db"
    assert panel.note.startswith("A password is set for this session.")

    saved = tmp_path / "saved.confw"
    workspace.save(workflow, saved)
    assert "s3cret" not in saved.read_text(encoding="utf-8")
    assert "s3cret" not in workspace.snapshot(workflow) and "s3cret" not in str(workspace.state(workflow))

    # A workflow saved by an earlier version, with the password in it, loses it on the way in
    old = tmp_path / "old.confw"
    old.write_text(saved.read_text(encoding="utf-8").replace("me@host", "me:0ldpass@host"), encoding="utf-8")
    monkeypatch.setattr(file_formats, "passwords", {})
    opened = workspace.load(old, stamp=False)[0]
    assert opened.nodes[0].settings["url"] == "postgresql://me@host:5432/db"
    assert file_formats.connection_url("postgresql://me@host:5432/db") == "postgresql://me:0ldpass@host:5432/db"
    workspace.save(opened, old)
    assert "0ldpass" not in old.read_text(encoding="utf-8")


# With no password typed, the one in the environment variable is used, as on a server or in a script
def test_the_password_can_come_from_the_environment(monkeypatch):
    monkeypatch.setattr(file_formats, "passwords", {})
    monkeypatch.delenv(file_formats.PASSWORD_VARIABLE, raising=False)
    assert file_formats.connection_url("postgresql://me@host/db") == "postgresql://me@host/db"
    monkeypatch.setenv(file_formats.PASSWORD_VARIABLE, "p@ss word")
    assert file_formats.connection_url("postgresql://me@host/db") == "postgresql://me:p%40ss+word@host/db"
    assert file_formats.without_passwords("could not reach me:p%40ss+word@host") == "could not reach me:***@host"
    assert file_formats.connection_url("sqlite:///file.db") == "sqlite:///file.db"


# A server is reached through SQLAlchemy. A SQLite address stands in for one here, which takes
# the same path through the code as PostgreSQL or MySQL would
def test_a_server_query_reads_and_cannot_change_anything(tmp_path):
    pytest.importorskip("sqlalchemy")
    path = make_database(tmp_path / "economy.sqlite")
    url = f"sqlite:///{path.as_posix()}"
    assert len(file_formats.read_sql(url, "SELECT * FROM growth")) == 4
    assert len(file_formats.read_sql(url, "SELECT * FROM growth ORDER BY year", rows=2)) == 2
    with pytest.raises(ValueError, match="start the query with SELECT"):
        file_formats.read_sql(url, "DELETE FROM growth")
    with pytest.raises(ValueError, match="no such table"):
        file_formats.read_sql(url, "SELECT * FROM nowhere")

    node = Node("Database", 0, 0, 1)
    node.settings = {"url": url, "query": "SELECT region FROM growth"}
    assert list(node.module.run([], node.settings)[0].columns) == ["region"]
    assert node.module.tables(node.settings) == ["growth", "places"]


def test_output_data_saves_each_kind(tmp_path):
    node = Node("Output Data", 0, 0, 1)
    data = pd.read_csv(SAMPLE_DATA).head(5)
    for kind in file_formats.WRITE_FORMATS:
        node.settings = {"folder": str(tmp_path), "name": "saved", "file_type": kind}
        node.module.run([data], node.settings)
        assert (tmp_path / f"saved.{kind}").exists()
    assert "\\toprule" in (tmp_path / "saved.tex").read_text(encoding="utf-8")


# ----- LaTeX -----

def test_latex_escapes_special_characters():
    assert latex.escape("gdp_growth 50% & $5 #1") == r"gdp\_growth 50\% \& \$5 \#1"
    assert latex.cell("0.123**") == "0.123$^{**}$"
    assert latex.cell("(0.045)") == "(0.045)"
    assert latex.cell(1995) == "1995"
    assert latex.cell(np.nan) == ""
    assert latex.cell(0.00001, 3) == "$1.000 \\times 10^{-5}$"


def test_latex_table_is_a_booktabs_table():
    text = latex.table(TABLE, caption="Growth", label="tab:growth", note="Source: ONS.")
    lines = text.splitlines()
    assert lines[0] == "\\begin{table}[htbp]" and lines[-1] == "\\end{table}"
    assert "\\begin{tabular}{lrr}" in lines
    assert lines.count("\\toprule") == 1 and lines.count("\\bottomrule") == 1
    assert "region & year & gdp\\_growth \\\\" in lines
    assert "\\item Source: ONS." in lines
    assert sum(1 for line in lines if line.endswith("\\\\")) == len(TABLE) + 1
