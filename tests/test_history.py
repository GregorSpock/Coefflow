from engine import addons, workspace
from engine.history import History
from engine.workflow import Workflow


def test_history_steps_back_and_forward():
    addons.load_all()
    workflow = Workflow()
    history = History(workflow)
    assert not history.can_undo() and not history.undo()

    data = workflow.add_node("Input Data", 0, 0)
    assert history.record() and not history.record()
    ols = workflow.add_node("OLS", 200, 0)
    workflow.connect(data, 0, ols, 0)
    history.record()
    ols.settings["dependent"] = "unemployment"

    # Undo keeps the change that had not been recorded yet, so redo can bring it back
    assert history.undo()
    assert workflow.node_by_id(ols.id).settings == {}
    assert history.undo()
    assert [node.name for node in workflow.nodes] == ["Input Data"] and workflow.connections == []
    assert history.undo() and workflow.nodes == []
    assert not history.undo()

    assert history.redo() and history.redo() and history.redo()
    assert workflow.node_by_id(ols.id).settings == {"dependent": "unemployment"}
    assert len(workflow.connections) == 1 and not history.redo()


# A new change after an undo drops what had been undone
def test_a_change_after_undo_clears_redo():
    workflow = Workflow()
    history = History(workflow)
    workflow.add_node("Input Data", 0, 0)
    history.record()
    workflow.add_node("OLS", 200, 0)
    history.record()

    history.undo()
    workflow.add_node("Filter", 200, 0)
    assert history.record() and not history.can_redo()
    assert [node.name for node in workflow.nodes] == ["Input Data", "Filter"]


# Paths in the settings come back from undo as they were, not relative to some folder
def test_undo_keeps_paths_whole(tmp_path):
    workflow = Workflow()
    node = workflow.add_node("Input Data", 0, 0)
    node.settings["path"] = str(tmp_path / "data.csv")
    history = History(workflow)
    node.settings["dropped"] = ["x"]
    history.record()
    history.undo()
    assert workflow.nodes[0].settings == {"path": str(tmp_path / "data.csv")}
    assert workspace.snapshot(workflow) == history.texts[0]


def test_old_import_paths_still_work():
    from ui import chart, colours
    from ui.clusters import add_cluster_button
    from ui.fields import apply_fields
    from ui.panel import Panel, fit_text
    from ui.panel_models import FIELD_SETTINGS
    from engine import colours as engine_colours
    from engine.panel import Panel as EnginePanel

    assert Panel is EnginePanel and colours is engine_colours
    assert callable(chart.draw_scatter) and callable(add_cluster_button) and callable(apply_fields)
    assert callable(fit_text) and "entity" in FIELD_SETTINGS
