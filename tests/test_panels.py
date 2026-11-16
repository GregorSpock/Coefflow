import subprocess
import sys

import pandas as pd
import pytest

from engine import addons, catalogue, chart, runner, workspace
from engine.draw import Recorder, Rect
from engine.node import Node
from engine.panel import Panel
from tests.addon_cases import CASES
from tests.support import ROOT


# Every module's settings panel can be opened, its buttons pressed and its chart drawn without
# a window, which is what keeps the modules apart from the app that shows them

AREA = Rect(0, 0, 700, 400)
SAMPLES = sorted((ROOT / "sample_workflows").glob("*.confw"))


@pytest.fixture(scope="module", autouse=True)
def loaded_addons():
    addons.load_all()


def all_buttons(panel):
    return list(panel.buttons) + [button for label, buttons in panel.options for button in buttons]


# Draws the panel's chart, if it has one, and returns what was drawn
def draw_chart(panel):
    if not panel.chart:
        return []
    recorder = Recorder()
    chart.render(recorder, AREA, panel.chart, panel.node.settings.get(chart.STYLE_KEY))
    return recorder.calls


# Opens the node's panel and clicks every button on it once, following any panel a click opens.
# Each click is on a freshly opened panel, as a click may remake the buttons
def exercise(node):
    panel = node.on_click()
    assert isinstance(panel, Panel)
    draw_chart(panel)

    labels = [(button.text, button.value) for button in all_buttons(panel)]
    for index in range(len(labels)):
        panel = node.on_click()
        buttons = all_buttons(panel)
        if index >= len(buttons):
            break
        panel.click(buttons[index])
        shown = panel.next_panel or panel
        if shown.text_field:
            shown.text_field.on_change(shown, shown.text_field.text)
        if shown.refresh:
            shown.refresh(shown)
        draw_chart(shown)
    return panel


# Loading every module pulls in neither the window toolkit nor anything else that needs a screen
def test_modules_load_without_a_window():
    code = (
        "import sys; from engine import addons, catalogue; from engine.node import load_module; addons.load_all();"
        "[load_module(name) for name in catalogue.all_modules()];"
        "print([name for name in ('PySide6', 'pygame', 'tkinter', 'ui') if name in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.stdout.strip() == "[]", result.stderr


@pytest.mark.parametrize("name", catalogue.all_modules() or ["Input Data"])
def test_every_module_keeps_to_the_contract(name):
    node = Node(name, 0, 0, 1)
    module = node.module
    assert module is not None, f"{name} has no module folder"
    assert callable(module.run) and callable(module.on_click)
    if not node.growing:
        assert len(module.INPUT_SPECS) == module.INPUTS
    assert len(module.OUTPUT_SPECS) == module.OUTPUTS
    assert len(getattr(module, "OUTPUT_TYPES", [])) <= module.OUTPUTS


# A node that has never run still opens its panel
@pytest.mark.parametrize("name", catalogue.all_modules() or ["Input Data"])
def test_panel_opens_before_a_run(name):
    panel = Node(name, 0, 0, 1).on_click()
    assert isinstance(panel, Panel)
    draw_chart(panel)


@pytest.mark.parametrize("path", SAMPLES, ids=lambda path: path.stem)
def test_panels_of_a_run_workflow(path, tmp_path):
    workflow, view, problems = workspace.load(path, stamp=False)
    for node in workflow.nodes:
        if "folder" in node.settings:
            node.settings["folder"] = str(tmp_path)
    runner.run_workflow(workflow)
    for node in workflow.nodes:
        panel = node.on_click()
        panel.canvas = workflow
        draw_chart(panel)
        exercise(node)


@pytest.mark.parametrize("name", sorted(CASES))
def test_addon_panels_after_a_run(name):
    pack, folder, data, settings = CASES[name]
    module = addons_module(pack, folder)
    node = Node("Input Data", 0, 0, 1)
    node.module = module
    node.name = folder
    node.inputs, node.outputs = module.INPUTS, module.OUTPUTS
    node.settings = dict(settings)
    node.last_inputs = [pd.read_csv(data)] if data else []
    node.last_outputs = module.run(node.last_inputs, node.settings)
    node.status = "done"
    exercise(node)


def addons_module(pack, folder):
    import importlib
    return importlib.import_module(f"addons.{pack}.{folder}.module")


# The charts draw something: the scatter of the first sample workflow has its points and its axes
def test_a_chart_draws_its_data(tmp_path):
    workflow, view, problems = workspace.load(SAMPLES[0], stamp=False)
    for node in workflow.nodes:
        if "folder" in node.settings:
            node.settings["folder"] = str(tmp_path)
    runner.run_workflow(workflow)
    scatter = next(node for node in workflow.nodes if node.name == "Scatter")
    calls = draw_chart(scatter.on_click())
    points = [call for call in calls if call[0] == "circle"]
    assert len(points) == len(scatter.last_inputs[0])
    assert any(call[0] == "text" and call[1] == "fitted" for call in calls)
