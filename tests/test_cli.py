import json
import shutil
import subprocess
import sys

import pandas as pd

from engine import cli, table_view
from tests.support import ROOT, SAMPLE_DATA


# The command line runs workflows with no window, reports each module, and says by its exit code
# whether everything worked. The table viewer's searching, filtering and summaries work the same way

SAMPLES = ROOT / "sample_workflows"


# A copy of a sample workflow in a folder of its own, its Output Data saving beside it
def copy_sample(name, folder):
    contents = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    for node in contents["nodes"]:
        if "folder" in node["settings"]:
            node["settings"]["folder"] = "saved"
        if "path" in node["settings"]:
            node["settings"]["path"] = str((SAMPLES / node["settings"]["path"]).resolve())
    path = folder / name
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def coefflow(*arguments):
    return subprocess.run([sys.executable, "main.py", *map(str, arguments)], cwd=ROOT, capture_output=True, text=True)


def test_a_workflow_runs_from_the_command_line(tmp_path):
    path = copy_sample("01_first_regression.confw", tmp_path)
    result = coefflow("run", path, "--export", tmp_path / "exported")
    assert result.returncode == 0, result.stderr
    assert "4 of 4 modules worked, 0 failed" in result.stdout
    assert (tmp_path / "saved" / "01_ols_coefficients.csv").exists()
    exported = sorted(p.name for p in (tmp_path / "exported").iterdir())
    assert "2_OLS_out1.csv" in exported and "1_Input_Data_out1.csv" in exported


def test_the_command_line_never_loads_the_window(tmp_path):
    path = copy_sample("01_first_regression.confw", tmp_path)
    code = (
        "import sys; from engine import cli; cli.main(['run', sys.argv[1], '--quiet']);"
        "print([m for m in ('PySide6', 'ui') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT, capture_output=True, text=True)
    assert result.stdout.strip().endswith("[]"), result.stderr


def test_set_changes_a_setting_for_one_run(tmp_path, capsys):
    path = copy_sample("01_first_regression.confw", tmp_path)
    assert cli.main(["run", str(path), "--set", "2.hac=true", "--set", "2.hac_lags=2", "--show", "2", "--quiet"]) == 0
    assert "Out 1" in capsys.readouterr().out
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "hac" not in saved["nodes"][1]["settings"]


def test_failures_give_exit_code_1_and_say_why(tmp_path, capsys):
    path = copy_sample("01_first_regression.confw", tmp_path)
    assert cli.main(["run", str(path), "--set", '2.dependent="no_such_field"']) == 1
    assert "FAILED #2 OLS" in capsys.readouterr().err


def test_unusable_workflows_give_exit_code_2(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path / "missing.confw")]) == 2
    broken = tmp_path / "broken.confw"
    broken.write_text("{not json", encoding="utf-8")
    assert cli.main(["run", str(broken)]) == 2
    path = copy_sample("01_first_regression.confw", tmp_path)
    assert cli.main(["run", str(path), "--set", "99.x=1"]) == 2
    assert cli.main(["run", str(path), "--set", "nonsense"]) == 2


def test_only_runs_a_module_and_what_it_needs(tmp_path, capsys):
    path = copy_sample("01_first_regression.confw", tmp_path)
    assert cli.main(["run", str(path), "--only", "2"]) == 0
    assert "2 of 2 modules worked" in capsys.readouterr().out
    assert not (tmp_path / "saved").exists()


def test_check_and_modules(tmp_path, capsys):
    path = copy_sample("09_reshape_and_diagnostics.confw", tmp_path)
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "#8 Diagnostics  <- #7 Out 1, #7 Out 2" in out
    assert cli.main(["modules"]) == 0
    assert "Group By" in capsys.readouterr().out


# ----- The table viewer's workings -----

DATA = pd.read_csv(SAMPLE_DATA)


def test_the_viewer_narrows_rows_by_search_and_condition():
    assert len(table_view.narrow(DATA, "cornwall")) == (DATA["region"] == "Cornwall").sum()
    both = table_view.narrow(DATA, "cornwall", "year >= 2020")
    assert len(both) == ((DATA["region"] == "Cornwall") & (DATA["year"] >= 2020)).sum()
    assert len(table_view.narrow(DATA)) == len(DATA)


def test_the_viewer_summarises_every_column():
    data = DATA.assign(when=pd.to_datetime(DATA["year"].astype(str)), flag=DATA["urban"] == 1)
    data.loc[0, "inflation"] = None
    summary = table_view.column_summary(data).set_index("field")
    assert summary.loc["inflation", "missing"] == 1
    assert summary.loc["unemployment", "mean"] == DATA["unemployment"].mean()
    assert summary.loc["region", "kind"] == "text" and summary.loc["region", "distinct"] == DATA["region"].nunique()
    assert summary.loc["when", "kind"] == "date" and summary.loc["when", "min"] == "1995-01-01"
    assert summary.loc["flag", "kind"] == "yes/no"
    frequencies = table_view.frequencies(data["region"])
    assert frequencies["count"].iloc[0] == data["region"].value_counts().iloc[0]


def test_the_viewer_lists_a_nodes_outputs_then_inputs():
    from engine.node import Node
    node = Node("OLS", 0, 0, 1)
    node.settings = {"dependent": "unemployment", "independent": ["gdp_growth"]}
    node.last_inputs = [DATA]
    node.last_outputs = node.module.run([DATA], node.settings)
    labels = [label for label, table in table_view.tables_of(node)]
    assert labels == ["Out 1: Coefficients", "Out 2: Fitted data", "In 1: Data"]
