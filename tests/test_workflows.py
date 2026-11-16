import shutil

import pandas as pd
import pytest

from engine import addons, runner, workspace
from engine.workflow import Workflow
from tests.support import ROOT, read_golden, same, summarise_outputs


SAMPLES = sorted((ROOT / "sample_workflows").glob("*.confw"))


@pytest.fixture(scope="module", autouse=True)
def loaded_addons():
    addons.load_all()


# Opens a sample workflow with its Output Data nodes pointed at a folder the test can write to
def open_sample(path, folder):
    workflow, view, problems = workspace.load(path, stamp=False)
    for node in workflow.nodes:
        if "folder" in node.settings:
            node.settings["folder"] = str(folder)
    return workflow, problems


# Every sample workflow still opens, runs without errors and gives the results it gave
# when the golden summaries were made
@pytest.mark.parametrize("path", SAMPLES, ids=lambda path: path.stem)
def test_sample_workflow_matches_golden(path, tmp_path):
    workflow, problems = open_sample(path, tmp_path)
    results, errors = runner.run_workflow(workflow)
    golden = read_golden(path.stem)

    assert problems == golden["problems"]
    assert errors == {}
    for node in workflow.nodes:
        expected = golden["nodes"][str(node.id)]
        assert node.name == expected["name"]
        assert same(summarise_outputs(results.get(node.id, [])), expected["outputs"])


def statuses(run):
    return {outcome.node.name: outcome.reused for outcome in run.outcomes.values()}


# A second run with nothing changed keeps every result, apart from Output Data, which always
# writes its file again
def test_unchanged_nodes_are_not_run_again(tmp_path):
    workflow, _ = open_sample(SAMPLES[0], tmp_path)
    runner.run_workflow(workflow)

    again = runner.Run(workflow)
    again.execute()
    assert statuses(again) == {"Input Data": True, "OLS": True, "Scatter": True, "Output Data": False}
    assert again.counts() == (1, 3)


# Changing one node's settings runs it and the nodes below it again, and nothing above it
def test_a_change_only_reruns_what_is_downstream(tmp_path):
    workflow, _ = open_sample(SAMPLES[0], tmp_path)
    runner.run_workflow(workflow)
    ols = next(node for node in workflow.nodes if node.name == "OLS")
    before = ols.last_outputs[0]

    ols.settings["robust"] = not ols.settings.get("robust", False)
    assert runner.mark_stale(workflow)
    assert ols.status is None
    assert next(node for node in workflow.nodes if node.name == "Input Data").status == "done"

    again = runner.Run(workflow)
    again.execute()
    again.apply()
    assert statuses(again) == {"Input Data": True, "OLS": False, "Scatter": False, "Output Data": False}
    assert not ols.last_outputs[0]["std_error"].equals(before["std_error"])


# Editing the data file a node reads counts as a change, though its settings are the same
def test_a_changed_data_file_is_read_again(tmp_path):
    data = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(data, index=False)
    workflow = Workflow()
    node = workflow.add_node("Input Data", 0, 0)
    node.settings["path"] = str(data)
    runner.run_workflow(workflow)
    assert len(node.last_outputs[0]) == 3

    pd.DataFrame({"x": [1, 2, 3, 4, 5]}).to_csv(data, index=False)
    assert runner.mark_stale(workflow)
    runner.run_workflow(workflow)
    assert len(node.last_outputs[0]) == 5


# force works everything out again, and only runs just the nodes a node depends on
def test_force_and_run_to_here(tmp_path):
    workflow, _ = open_sample(SAMPLES[0], tmp_path)
    ols = next(node for node in workflow.nodes if node.name == "OLS")

    data = next(node for node in workflow.nodes if node.name == "Input Data")
    first = runner.Run(workflow, only=[data])
    first.execute()
    assert sorted(statuses(first)) == ["Input Data"]

    to_here = runner.Run(workflow, only=[ols])
    to_here.execute()
    to_here.apply()
    assert sorted(statuses(to_here)) == ["Input Data", "OLS"]
    assert next(node for node in workflow.nodes if node.name == "Scatter").status is None

    forced = runner.Run(workflow, force=True)
    forced.execute()
    assert not any(statuses(forced).values())


# A cancelled run leaves the nodes it never reached as they were
def test_cancel_stops_before_the_next_node(tmp_path):
    workflow, _ = open_sample(SAMPLES[0], tmp_path)
    run = runner.Run(workflow)
    run.execute(on_status=lambda node, status: run.cancel())
    run.apply()
    done = [node.name for node in workflow.nodes if node.status == "done"]
    assert done == ["Input Data"]


# A node that fails reports why, and the nodes below it say their input is missing
def test_errors_are_reported_per_node():
    workflow = Workflow()
    data = workflow.add_node("Input Data", 0, 0)
    ols = workflow.add_node("OLS", 200, 0)
    scatter = workflow.add_node("Scatter", 400, 0)
    workflow.connect(data, 0, ols, 0)
    workflow.connect(ols, 1, scatter, 0)

    results, errors = runner.run_workflow(workflow)
    assert errors == {ols.id: "Choose a dependent variable (Y)", scatter.id: "Missing input"}
    assert (data.status, ols.status, scatter.status) == ("done", "error", "error")


def test_connections_keep_to_the_rules():
    workflow = Workflow()
    data = workflow.add_node("Input Data", 0, 0)
    ols = workflow.add_node("OLS", 200, 0)
    scatter = workflow.add_node("Scatter", 400, 0)

    assert workflow.connect(data, 0, ols, 0)
    assert workflow.connect(ols, 1, scatter, 0)
    assert not workflow.connect(ols, 0, ols, 0)
    assert workflow.refusal(ols, 0, ols, 0) == "A module can't be connected to itself"

    # A second line into the same input replaces the first
    other = workflow.add_node("Input Data", 0, 200)
    assert workflow.connect(other, 0, ols, 0)
    assert [c.source for c in workflow.connections if c.target is ols] == [other]

    workflow.remove_node(ols)
    assert workflow.connections == []


# A plain table can't be plugged into an input that needs a fitted model, but a model can go
# anywhere a table can
def test_typed_ports():
    workflow = Workflow()
    data = workflow.add_node("Input Data", 0, 0)
    fixed = workflow.add_node("Fixed Effects", 200, 0)
    hausman = workflow.add_node("Hausman", 400, 0)
    output = workflow.add_node("Output Data", 400, 200)
    forecast = workflow.add_node("Forecast", 400, 400)

    assert not workflow.connect(data, 0, hausman, 0)
    assert "takes a fitted model" in workflow.refusal(data, 0, hausman, 0)
    assert not workflow.connect(fixed, 1, hausman, 0)
    assert workflow.connect(fixed, 0, hausman, 0)
    assert workflow.connect(fixed, 0, output, 0)
    assert workflow.connect(data, 0, forecast, 0)


# A model input given a table without a fit says so when the workflow runs
def test_a_model_input_checks_what_arrives():
    from engine import ports
    workflow = Workflow()
    hausman = workflow.add_node("Hausman", 0, 0)
    with pytest.raises(ValueError, match="needs a fitted model"):
        ports.check_inputs(hausman, [pd.DataFrame({"a": [1]}), None])


def test_save_and_load_round_trip(tmp_path):
    copy = tmp_path / "copy.confw"
    shutil.copy(SAMPLES[7], copy)
    workflow, view, problems = workspace.load(copy, stamp=False)
    saved = tmp_path / "saved.confw"
    workspace.save(workflow, saved, view)
    again, view_again, _ = workspace.load(saved)

    assert workspace.snapshot(again) == workspace.snapshot(workflow)
    assert view_again == view


def test_a_broken_file_is_refused(tmp_path):
    broken = tmp_path / "broken.confw"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid workflow file"):
        workspace.load(broken)


# Undo puts back the nodes, settings and connections, keeping the results of nodes it didn't change
def test_state_and_restore(tmp_path):
    workflow, _ = open_sample(SAMPLES[0], tmp_path)
    runner.run_workflow(workflow)
    before = workspace.state(workflow)
    text = workspace.snapshot(workflow)
    ols = next(node for node in workflow.nodes if node.name == "OLS")
    data = next(node for node in workflow.nodes if node.name == "Input Data")

    ols.settings["controls"] = []
    workflow.remove_node(next(node for node in workflow.nodes if node.name == "Scatter"))
    workflow.add_node("Filter", 0, 0)
    assert workspace.snapshot(workflow) != text

    workspace.restore(workflow, before)
    assert workspace.snapshot(workflow) == text
    assert next(node for node in workflow.nodes if node.name == "Input Data") is data
    assert data.status == "done" and data.last_outputs
    assert ols.status is None
