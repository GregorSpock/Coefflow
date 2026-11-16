import time

import pytest
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QImage, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from engine import chart, colours
from tests.support import ROOT
from ui import app as app_module
from ui import canvas as canvas_module
from ui import dialogs, painter, theme


# Drives the real window, with Qt drawing to nothing instead of the screen: modules are placed,
# lines are drawn with the mouse, workflows run on their thread, and changes are undone

SAMPLES = ROOT / "sample_workflows"


@pytest.fixture(scope="session")
def application():
    qt = QApplication.instance() or QApplication([])
    qt.setStyleSheet(theme.style_sheet())
    return qt


# A window past the opening screen, with every question answered without a dialog: messages are
# kept in window.messages and unsaved changes are never saved
@pytest.fixture
def window(application, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "TEMP_FOLDER", tmp_path / "temp")
    made = app_module.MainWindow()
    made.messages = []
    monkeypatch.setattr(dialogs, "show_message", lambda title, text: made.messages.append((title, text)))
    monkeypatch.setattr(dialogs, "ask_save_changes", lambda name=None: False)
    made.resize(1500, 900)
    made.show()
    made.start(False)
    application.processEvents()
    yield made
    wait_for_run(made)
    made.close()
    application.processEvents()


def wait_for_run(window, timeout=60):
    end = time.time() + timeout
    while window.running() and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    assert not window.running()


def open_sample(window, name, tmp_path):
    window.open_workflow(str(SAMPLES / name))
    for node in window.active.workflow.nodes:
        if "folder" in node.settings:
            node.settings["folder"] = str(tmp_path)
    window.active.history.record()
    window.active.saved_state = app_module.workspace.snapshot(window.active.workflow)
    return window.active.workflow


def node_named(workflow, name):
    return next(node for node in workflow.nodes if node.name == name)


# Sends one mouse event to the canvas at a canvas position
def mouse(canvas, kind, scene_pos, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    position = QPointF(canvas.mapFromScene(scene_pos))
    held = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else button
    pressed = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else button
    event = QMouseEvent(kind, position, QPointF(canvas.viewport().mapToGlobal(position.toPoint())), pressed, held, modifiers)
    QApplication.sendEvent(canvas.viewport(), event)


def drag(canvas, start, end, modifiers=Qt.KeyboardModifier.NoModifier):
    mouse(canvas, QEvent.Type.MouseButtonPress, start, modifiers=modifiers)
    middle = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
    mouse(canvas, QEvent.Type.MouseMove, middle, modifiers=modifiers)
    mouse(canvas, QEvent.Type.MouseMove, end, modifiers=modifiers)
    mouse(canvas, QEvent.Type.MouseButtonRelease, end, modifiers=modifiers)
    QApplication.processEvents()


def out_port(node, index=0):
    return canvas_module.port_positions(node, "output")[index]


def in_port(node, index=0):
    return canvas_module.port_positions(node, "input")[index]


def panel_button(window, text):
    QApplication.processEvents()
    for button in window.panel_view.findChildren(QPushButton):
        if button.text() == text and button.isVisible():
            return button
    raise AssertionError(f"No {text} button on the panel")


# ----- The canvas -----

def test_starts_with_a_blank_workflow(window):
    assert window.pages.currentIndex() == 1
    assert len(window.workflows) == 1 and window.active.is_blank()
    assert window.windowTitle() == "Coefflow - UntitledWorkflow_1"
    assert not window.panel_view.isVisible()


def test_a_line_is_drawn_between_ports_with_the_mouse(window):
    canvas = window.canvas
    data = canvas.add_node("Input Data", QPointF(-200, 0))
    ols = canvas.add_node("OLS", QPointF(100, 0))

    drag(canvas, out_port(data), in_port(ols))
    connections = window.active.workflow.connections
    assert [(c.source, c.output_index, c.target, c.input_index) for c in connections] == [(data, 0, ols, 0)]
    assert len(canvas.connection_items) == 1

    # Grabbing the connected input and letting go over nothing unplugs the line
    drag(canvas, in_port(ols), QPointF(100, 300))
    assert window.active.workflow.connections == []


def test_a_refused_connection_says_why(window):
    canvas = window.canvas
    data = canvas.add_node("Input Data", QPointF(-200, 0))
    hausman = canvas.add_node("Hausman", QPointF(100, 0))

    drag(canvas, out_port(data), in_port(hausman))
    assert window.active.workflow.connections == []
    assert "takes a fitted model" in window.status_label.text()


def test_a_click_opens_the_settings_and_a_drag_moves_the_node(window):
    canvas = window.canvas
    ols = canvas.add_node("OLS", QPointF(0, 0))
    window.watch()

    drag(canvas, QPointF(0, 10), QPointF(0, 10))
    assert window.panel_view.isVisible()
    assert window.panel_view.title.text() == "OLS  #1"

    window.close_panel()
    drag(canvas, QPointF(0, 10), QPointF(160, 90))
    assert (round(ols.x), round(ols.y)) == (160, 80)
    assert not window.panel_view.isVisible()

    window.undo()
    moved_back = window.active.workflow.node_by_id(1)
    assert (round(moved_back.x), round(moved_back.y)) == (0, 0)


def test_a_module_dropped_from_the_toolbar_is_placed(window):
    canvas = window.canvas
    data = QMimeData()
    data.setData(canvas_module.MODULE_MIME, b"Filter")
    position = QPointF(canvas.mapFromScene(QPointF(80, 40)))
    event = QDropEvent(position, Qt.DropAction.CopyAction, data, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.dropEvent(event)

    nodes = window.active.workflow.nodes
    assert [node.name for node in nodes] == ["Filter"]
    assert abs(nodes[0].x - 80) < 2 and abs(nodes[0].y - 40) < 2


# The buttons on the toolbar's second row, leaving out each module's ? and any being removed
def module_row(window):
    QApplication.processEvents()
    buttons = window.toolbar.modules.findChildren(QPushButton)
    return [button.text() for button in buttons if button.text() != "?" and not button.isHidden()]


# A wheel turn or trackpad swipe over a toolbar button, as the pointer sits on one nearly always
def swipe(target, x, y):
    centre = QPointF(target.rect().center())
    event = QWheelEvent(centre, QPointF(target.mapToGlobal(centre.toPoint())), QPoint(), QPoint(x, y),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(target, event)


def test_both_toolbar_rows_scroll_sideways_from_over_a_button(window):
    window.resize(700, 600)
    window.toolbar.select("Clean")
    for _ in range(3):
        QApplication.processEvents()
    for row in [window.toolbar.tabs, window.toolbar.modules]:
        bar = row.horizontalScrollBar()
        assert bar.maximum() > 0
        button = row.widget().findChildren(QPushButton)[0]
        swipe(button, 0, -60)
        assert bar.value() == 60
        swipe(button, -40, 5)
        assert bar.value() == 100


def test_the_toolbar_lists_every_module(window):
    window.toolbar.select("Base Reg.")
    assert module_row(window) == ["OLS", "Auto OLS", "Logit", "Probit"]
    window.toolbar.select("File")
    assert module_row(window) == ["New", "Open", "Save", "Save As"]
    window.toolbar.select(None)
    assert window.toolbar.modules_bar.isHidden()


def test_zoom_keeps_within_its_limits(window):
    canvas = window.canvas
    for _ in range(100):
        canvas.zoom_in()
    assert canvas.zoom == canvas_module.MAX_ZOOM
    for _ in range(200):
        canvas.zoom_out()
    assert canvas.zoom == canvas_module.MIN_ZOOM
    assert window.zoom_label.text() == "25%"


# ----- Undo, redo, copy and paste -----

def test_undo_and_redo(window):
    canvas = window.canvas
    workflow = window.active.workflow
    data = canvas.add_node("Input Data", QPointF(-200, 0))
    ols = canvas.add_node("OLS", QPointF(100, 0))
    drag(canvas, out_port(data), in_port(ols))
    window.watch()
    assert not window.toolbar.redo_button.isEnabled() and window.toolbar.undo_button.isEnabled()

    canvas.select_only([ols])
    canvas.delete_selected()
    assert [node.name for node in workflow.nodes] == ["Input Data"] and workflow.connections == []

    window.undo()
    assert [node.name for node in workflow.nodes] == ["Input Data", "OLS"] and len(workflow.connections) == 1
    assert len(canvas.node_items) == 2 and len(canvas.connection_items) == 1

    window.undo()
    assert workflow.connections == []
    window.redo()
    window.redo()
    assert [node.name for node in workflow.nodes] == ["Input Data"]
    window.redo()
    assert window.status_label.text() == "Nothing to redo"


def test_a_setting_changed_in_a_panel_can_be_undone(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    ols = node_named(workflow, "OLS")
    window.node_clicked(ols)
    assert ols.settings["robust"] is True

    QTest.mouseClick(panel_button(window, "Robust: On"), Qt.MouseButton.LeftButton)
    assert ols.settings["robust"] is False
    panel_button(window, "Robust: Off")
    window.watch()

    window.undo()
    assert node_named(workflow, "OLS").settings["robust"] is True
    panel_button(window, "Robust: On")


def test_copy_and_paste(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    canvas = window.canvas
    data, ols = node_named(workflow, "Input Data"), node_named(workflow, "OLS")
    canvas.select_only([data, ols])
    canvas.copy()
    canvas.paste(QApplication.clipboard().text())

    assert len(workflow.nodes) == 6 and len(workflow.connections) == 4
    pasted = canvas.selected_nodes()
    copy_of_ols = next(node for node in pasted if node.name == "OLS")
    assert sorted(node.name for node in pasted) == ["Input Data", "OLS"]
    assert copy_of_ols.id not in (data.id, ols.id)
    assert copy_of_ols.settings == ols.settings and copy_of_ols.settings is not ols.settings
    assert (copy_of_ols.x, copy_of_ols.y) == (ols.x + canvas_module.PASTE_OFFSET, ols.y + canvas_module.PASTE_OFFSET)
    # The line between the two copies came with them, and the lines to the nodes left behind did not
    assert [c.source.name for c in workflow.connections if c.target is copy_of_ols] == ["Input Data"]
    assert not any(c.source is copy_of_ols for c in workflow.connections)

    window.undo()
    assert len(workflow.nodes) == 4


def test_several_nodes_are_selected_moved_and_deleted_together(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    canvas = window.canvas
    canvas.set_view({"x": 200, "y": 120, "zoom": 1.0})
    before = {node.id: (node.x, node.y) for node in workflow.nodes}

    # Shift + dragging the background selects every node in the box
    drag(canvas, QPointF(-100, -100), QPointF(300, 400), Qt.KeyboardModifier.ShiftModifier)
    assert sorted(node.name for node in canvas.selected_nodes()) == ["Input Data", "OLS"]

    ols = node_named(workflow, "OLS")
    drag(canvas, QPointF(ols.x, ols.y + 10), QPointF(ols.x + 50, ols.y + 60))
    moved = [node for node in workflow.nodes if (node.x, node.y) != before[node.id]]
    assert sorted(node.name for node in moved) == ["Input Data", "OLS"]

    canvas.select_all()
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert workflow.nodes == [] and workflow.connections == []
    window.undo()
    assert len(workflow.nodes) == 4 and len(workflow.connections) == 3


# ----- Running -----

def test_a_run_happens_away_from_the_window(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    window.run_workflow()
    assert window.running() and window.canvas.locked
    assert window.toolbar.run_button.text() == "Stop"

    wait_for_run(window)
    assert {node.name: node.status for node in workflow.nodes} == {
        "Input Data": "done", "OLS": "done", "Output Data": "done", "Scatter": "done",
    }
    assert window.status_label.text() == "Ran 4 modules"
    assert window.toolbar.run_button.text() == "Run" and not window.canvas.locked
    assert (tmp_path / "01_ols_coefficients.csv").exists()

    # Nothing has changed, so only Output Data, which always writes its file, runs again
    window.run_workflow()
    wait_for_run(window)
    assert window.status_label.text() == "Ran 1 module, kept the results of 3 that had not changed"


def test_a_changed_node_goes_back_to_not_run(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    window.run_workflow()
    wait_for_run(window)

    ols = node_named(workflow, "OLS")
    window.node_clicked(ols)
    QTest.mouseClick(panel_button(window, "Robust: On"), Qt.MouseButton.LeftButton)
    window.watch()
    assert {node.name: node.status for node in workflow.nodes} == {
        "Input Data": "done", "OLS": None, "Output Data": None, "Scatter": None,
    }

    window.run_workflow()
    wait_for_run(window)
    assert window.status_label.text() == "Ran 3 modules, kept the results of 1 that had not changed"
    assert window.panel_view.panel.preview is ols.last_outputs[0]


def test_run_to_here_and_errors(window):
    canvas = window.canvas
    data = canvas.add_node("Input Data", QPointF(-200, 0))
    ols = canvas.add_node("OLS", QPointF(100, 0))
    scatter = canvas.add_node("Scatter", QPointF(400, 0))
    drag(canvas, out_port(data), in_port(ols))
    drag(canvas, out_port(ols, 1), in_port(scatter))

    window.run_workflow(only=[data])
    wait_for_run(window)
    assert (data.status, ols.status, scatter.status) == ("done", None, None)

    window.run_workflow()
    wait_for_run(window)
    assert (data.status, ols.status, scatter.status) == ("done", "error", "error")
    assert ols.error == "Choose a dependent variable (Y)"
    assert window.status_label.text() == "Ran 2 modules, kept the results of 1 that had not changed, 2 failed"
    assert canvas.tooltip_at(QPointF(ols.x, ols.y + 10)) == "<b>OLS #2</b><br>Choose a dependent variable (Y)"


def test_a_run_can_be_stopped(window, tmp_path, monkeypatch):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    data = node_named(workflow, "Input Data")
    run = data.module.run

    def slow(inputs, settings):
        time.sleep(0.3)
        return run(inputs, settings)

    monkeypatch.setattr(data.module, "run", slow)
    window.run_workflow()
    while data.status != "running":
        QApplication.processEvents()
    window.run_clicked()
    wait_for_run(window)
    assert data.status == "done"
    assert node_named(workflow, "OLS").status is None
    assert window.status_label.text() == "Ran 1 module, stopped early"


# ----- Panels -----

def test_typing_in_a_panel_reaches_the_settings(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    output = node_named(workflow, "Output Data")
    window.node_clicked(output)
    box = window.panel_view.field_box
    assert box.isVisible() and box.text() == "01_ols_coefficients"

    box.setFocus()
    box.end(False)
    QTest.keyClicks(box, "_v2")
    assert output.settings["name"] == "01_ols_coefficients_v2"
    # Characters that can't be in a file name are not typed
    QTest.keyClicks(box, "?*")
    assert output.settings["name"] == "01_ols_coefficients_v2" and box.text() == "01_ols_coefficients_v2"


def test_the_panel_shows_results_as_a_table(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    window.run_workflow()
    wait_for_run(window)
    ols = node_named(workflow, "OLS")
    window.node_clicked(ols)

    view = window.panel_view
    model = view.table.model()
    assert view.stack.currentWidget() is view.table
    assert model.rowCount() == 4 and model.columnCount() == len(ols.last_outputs[0].columns)
    assert model.data(model.index(0, 1)) == "Intercept"
    assert view.legend.text().startswith("Stars: * p < ")

    # Sorting the view by a column leaves the node's own results alone
    column = list(ols.last_outputs[0].columns).index("coefficient")
    model.sort(column, Qt.SortOrder.DescendingOrder)
    shown = [float(model.frame.iat[row, column]) for row in range(4)]
    assert shown == sorted(shown, reverse=True)
    assert list(ols.last_outputs[0]["variable"]) == ["Intercept", "gdp_growth", "inflation", "bank_rate"]


def test_chart_settings_and_export(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    window.run_workflow()
    wait_for_run(window)
    scatter = node_named(workflow, "Scatter")
    window.node_clicked(scatter)
    view = window.panel_view
    assert view.stack.currentWidget() is view.chart and view.settings_button.isVisible()

    QTest.mouseClick(view.settings_button, Qt.MouseButton.LeftButton)
    assert view.chart_settings.isVisible()
    QTest.keyClicks(view.chart_settings.boxes["title"], "Residuals")
    QTest.keyClicks(view.chart_settings.boxes["ticks"], "99")
    assert scatter.settings[chart.STYLE_KEY] == {"title": "Residuals", "ticks": 20}
    view.chart_settings.use_print_colours()
    assert scatter.settings[chart.STYLE_KEY]["colours"]["background"] == [255, 255, 255]

    path = tmp_path / "chart.png"
    painter.export_png(view.panel.chart, scatter.settings[chart.STYLE_KEY], str(path))
    picture = QImage(str(path))
    assert (picture.width(), picture.height()) == chart.EXPORT_SIZE
    colours_used = {picture.pixel(x, y) for x in range(0, 1200, 7) for y in range(0, 800, 7)}
    assert len(colours_used) > 3

    view.chart_settings.reset_style()
    assert chart.STYLE_KEY not in scatter.settings


def test_every_sample_workflow_runs_and_shows_in_the_window(window, tmp_path):
    for path in sorted(SAMPLES.glob("*.confw")):
        workflow = open_sample(window, path.name, tmp_path)
        window.run_workflow()
        wait_for_run(window)
        assert [node.name for node in workflow.nodes if node.status != "done"] == []
        for node in workflow.nodes:
            window.node_clicked(node)
            QApplication.processEvents()
            assert window.panel_view.title.text() == f"{window.panel_view.panel.title}  #{node.id}"
        window.grab()
    assert window.messages == []


# ----- Themes and files -----

def test_every_theme_can_be_shown(window, application, tmp_path):
    open_sample(window, "01_first_regression.confw", tmp_path)
    first = colours.current_theme
    seen = []
    for _ in range(len(colours.THEMES)):
        window.next_theme()
        application.processEvents()
        seen.append(colours.current_theme)
        assert not window.grab().isNull()
    assert sorted(seen) == sorted(colours.THEMES) and colours.current_theme == first


def test_save_and_open_through_the_window(window, tmp_path, monkeypatch):
    canvas = window.canvas
    data = canvas.add_node("Input Data", QPointF(-200, 0))
    ols = canvas.add_node("OLS", QPointF(100, 0))
    drag(canvas, out_port(data), in_port(ols))
    ols.settings["dependent"] = "unemployment"
    window.watch()
    assert window.tabs.tabText(0) == "UntitledWorkflow_1*"

    path = tmp_path / "mine.confw"
    monkeypatch.setattr(dialogs, "choose_save_path", lambda: str(path))
    assert window.save_workflow()
    assert path.exists() and window.tabs.tabText(0) == "mine" and window.windowTitle() == "Coefflow - mine"

    window.new_workflow()
    assert len(window.workflows) == 2 and window.active.is_blank()
    window.close_workflow(window.workflows[0])
    window.open_workflow(str(path))
    assert len(window.workflows) == 1
    reopened = window.active.workflow
    assert [node.name for node in reopened.nodes] == ["Input Data", "OLS"]
    assert node_named(reopened, "OLS").settings == {"dependent": "unemployment"}
    assert len(reopened.connections) == 1 and window.messages == []


# ----- The table viewer -----

def test_the_table_viewer_shows_searches_and_filters_a_modules_tables(window, tmp_path):
    workflow = open_sample(window, "01_first_regression.confw", tmp_path)
    window.run_workflow()
    wait_for_run(window)
    ols = node_named(workflow, "OLS")

    window.node_clicked(ols)
    QApplication.processEvents()
    assert window.panel_view.view_button.isVisible()

    viewer = window.view_data(ols)
    QApplication.processEvents()
    assert viewer.picker.count() == 3
    viewer.picker.setCurrentIndex(1)
    rows = len(ols.last_outputs[1])
    assert viewer.table.model().rowCount() == rows
    assert viewer.summary.model().rowCount() == len(ols.last_outputs[1].columns)

    viewer.condition.setText("year >= 2020")
    viewer.update_view()
    expected = int((ols.last_outputs[1]["year"] >= 2020).sum())
    assert viewer.table.model().rowCount() == expected
    assert f"Showing {expected:,} of {rows:,}" in viewer.size_label.text()

    viewer.condition.setText("year >= ")
    viewer.update_view()
    assert "Condition not used" in viewer.condition_error.text()
    assert viewer.table.model().rowCount() == rows

    viewer.copy_rows()
    assert QApplication.clipboard().text().splitlines()[0].startswith("region\tyear")
    viewer.close()
