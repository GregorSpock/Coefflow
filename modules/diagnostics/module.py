import numpy as np

from engine import diagnostics
from engine.chart import draw_lines, draw_scatter
from engine.panel import Panel


INPUTS = 2
OUTPUTS = 3
INPUT_TYPES = ["model", "table"]
OUTPUT_TYPES = ["table", "table", "table"]
INPUT_SPECS = [
    ("Model", "the coefficient table (Out 1) of OLS or Auto OLS"),
    ("Fitted data", "the same model's fitted data (Out 2), with the rows it was fitted on in time order for the time series tests"),
]
OUTPUT_SPECS = [
    ("Tests", "one row per test with what it assumes, the statistic, its degrees of freedom, the p value and what the result means"),
    ("VIF", "the variance inflation factor and tolerance of each regressor, flagged above 5 and 10"),
    ("Influence", "the fitted rows with each one's leverage, studentized residual, Cook's distance and DFFITS, flagged against the usual cut-offs"),
]

# The plots the panel can show in place of the tests table
VIEWS = ["Tests", "Residuals vs fitted", "Normal Q-Q", "CUSUM", "Leverage"]
LAGS = [1, 2, 4, 8, 12]
# Chow needs a break, so it starts off; the rest start on
DEFAULT_OFF = ["Chow"]


def set_defaults(settings):
    settings.setdefault("tests", [t for t in diagnostics.TESTS if t not in DEFAULT_OFF])
    settings.setdefault("lags", 1)
    settings.setdefault("break", "")
    settings.setdefault("view", VIEWS[0])


# Opens a panel to choose the tests, the lags for the autocorrelation tests and the break for the
# Chow test, showing the results or one of the residual plots
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Diagnostics")
    panel.add_button(f"Show: {node.settings['view']}", switch_view)
    panel.add_button(f"Lags: {node.settings['lags']}", switch_lags)
    panel.add_text_field("Chow break (rows after it, e.g. year >= 2008)", node.settings["break"], type_break, blocked="")
    panel.preview_title = "Results"
    panel.refresh = show_diagnostics
    show_diagnostics(panel)
    return panel


def switch_view(panel, button):
    settings = panel.node.settings
    settings["view"] = VIEWS[(VIEWS.index(settings["view"]) + 1) % len(VIEWS)] if settings["view"] in VIEWS else VIEWS[0]
    button.text = f"Show: {settings['view']}"
    show_diagnostics(panel)


def switch_lags(panel, button):
    settings = panel.node.settings
    current = settings["lags"] if settings["lags"] in LAGS else LAGS[0]
    settings["lags"] = LAGS[(LAGS.index(current) + 1) % len(LAGS)]
    button.text = f"Lags: {settings['lags']}"
    show_diagnostics(panel)


def type_break(panel, text):
    settings = panel.node.settings
    settings["break"] = text
    if text.strip() and "Chow" not in settings["tests"]:
        settings["tests"].append("Chow")
    show_diagnostics(panel)


def toggle_test(panel, button):
    tests = panel.node.settings["tests"]
    if button.value in tests:
        tests.remove(button.value)
    else:
        tests.append(button.value)
    show_diagnostics(panel)


# The problems the tests found, in a sentence, for the note under the results. What each means
# and what to do is in the result column, so the note only names them and leaves the table room
def verdict(tests):
    failed = tests[tests["p_value"] < diagnostics.LEVEL]
    if tests.empty:
        return ""
    if failed.empty:
        return "No test found a problem at the 5% level."
    if len(failed) == 1:
        return f"Problem found at the 5% level: {failed['result'].iloc[0]}"
    return (
        f"Problems found at the 5% level by {', '.join(failed['test'])}. "
        "The result column says what each means and what to do; View all shows it in full."
    )


def show_diagnostics(panel):
    node = panel.node
    settings = node.settings
    set_defaults(settings)
    panel.clear_options()
    panel.preview = None
    panel.chart = None
    panel.note = ""
    panel.lines = [f"Lags for Breusch-Godfrey and ARCH-LM: {settings['lags']}    Chow break: {settings['break'] or 'none'}"]

    for name in diagnostics.TESTS:
        panel.add_option(name, [("On", toggle_test, name in settings["tests"])])
        panel.options[-1][1][0].value = name

    if not node.last_outputs:
        panel.message = node.error or "Connect a model (Out 1 of OLS) and its fitted data (Out 2), then run the workflow"
        return

    tests, vif, influence = node.last_outputs
    notes = tests.attrs.get("notes", [])
    panel.note = " ".join([verdict(tests)] + notes).strip()
    high = vif[vif["vif"] > 10]["variable"].tolist() if not vif.empty else []
    if high:
        panel.note += f" High collinearity (VIF > 10): {', '.join(high)}."
    flagged = int((influence["flag"] != "").sum())
    panel.lines.append(f"N: {len(influence)}    Rows flagged as high leverage, outliers or influential: {flagged}")

    view = settings["view"]
    if view == "Tests":
        panel.preview = tests
        return
    font = panel.table_font
    if view == "Residuals vs fitted":
        x, y = influence["fitted"].to_numpy(float), influence["residual"].to_numpy(float)
        panel.chart = lambda painter, area: draw_scatter(painter, area, font, x, y, "Fitted", "Residual", (0.0, 0.0))
    elif view == "Normal Q-Q":
        x, y = diagnostics.qq_points(influence["residual"])
        panel.chart = lambda painter, area: draw_scatter(painter, area, font, x, y, "Normal quantile", "Standardized residual", (0.0, 1.0))
    elif view == "CUSUM":
        path = tests.attrs.get("cusum")
        if not path:
            panel.message = "The CUSUM path could not be worked out for this model"
            return
        positions, values, lower, upper = (np.asarray(part, dtype=float) for part in path)
        series = [("CUSUM", positions, values), ("5% bounds", positions, upper), ("", positions, lower)]
        panel.chart = lambda painter, area: draw_lines(painter, area, font, series, "Row")
    else:
        x, y = influence["leverage"].to_numpy(float), influence["studentized"].to_numpy(float)
        panel.chart = lambda painter, area: draw_scatter(painter, area, font, x, y, "Leverage", "Studentized residual")


# Rebuilds the model from its fitted data and runs the chosen tests, the VIFs and the influence
# measures. The tests are of the model's errors, so they are the same whatever standard errors
# the model was fitted with
def run(inputs, settings):
    set_defaults(settings)
    model_table, data = inputs
    fit = model_table.attrs.get("fit") or {}
    if fit.get("kind") != "linear":
        raise ValueError("Diagnostics works on linear models: connect Out 1 of OLS or Auto OLS")

    y, x, used = diagnostics.design(data, fit)
    if len(y) <= x.shape[1] + 1:
        raise ValueError(f"{len(y)} rows is too few to test a model with {x.shape[1]} coefficients")
    lags = int(settings.get("lags") or 1)
    tests, notes, model = diagnostics.run_tests(y, x, used, settings["tests"], lags=lags, condition=settings["break"])
    attrs = {"notes": notes}
    try:
        attrs["cusum"] = [part.tolist() for part in diagnostics.cusum_path(model)]
    except Exception:
        pass
    tests.attrs = attrs
    return [tests, diagnostics.vif(x), diagnostics.influence(model, used)]
