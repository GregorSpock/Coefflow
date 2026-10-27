import numpy as np

from engine import clusters, dummies
from ui.clusters import add_cluster_button
from ui.panel import Panel


# The panel shared by the Fixed Effects and Random Effects modules, where each field gets a role:
# Y (dependent), X (independent), C (control), E (the field naming each entity) or T (the time field).
# Buttons switch time effects and errors clustered by entity on and off, and choose another field to
# cluster by. The settings are described in engine/panel_data.py

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "independent", "controls", "entity", "time", "cluster"]


def open_panel(node, title):
    panel = Panel(node, title)
    panel.add_button(robust_label(node.settings), toggle_robust)
    add_cluster_button(panel)
    panel.add_button(time_effects_label(node.settings), toggle_time_effects)
    panel.preview_title = "Results"
    panel.refresh = show_model
    show_model(panel)
    return panel


def robust_label(settings):
    return "Robust: On" if settings.get("robust") else "Robust: Off"


def time_effects_label(settings):
    return "Time effects: On" if settings.get("time_effects") else "Time effects: Off"


# Switches errors clustered by entity on or off
def toggle_robust(panel, button):
    settings = panel.node.settings
    settings["robust"] = not settings.get("robust", False)
    button.text = robust_label(settings)
    show_model(panel)


# Switches a dummy for each period on or off, which needs the time field (T)
def toggle_time_effects(panel, button):
    settings = panel.node.settings
    settings["time_effects"] = not settings.get("time_effects", False)
    button.text = time_effects_label(settings)
    show_model(panel)


# Returns "Y", "X", "C", "E", "T" or None for a field
def role_of(settings, column):
    for role, key in [("Y", "dependent"), ("E", "entity"), ("T", "time")]:
        if settings.get(key) == column:
            return role
    if column in settings.get("independent", []):
        return "X"
    if column in settings.get("controls", []):
        return "C"
    return None


def clear_role(settings, column):
    for key in ["dependent", "entity", "time"]:
        if settings.get(key) == column:
            settings[key] = None
    settings["independent"] = [c for c in settings.get("independent", []) if c != column]
    settings["controls"] = [c for c in settings.get("controls", []) if c != column]


# Gives a field the clicked role, or clears it if it already had that role.
# There is only one Y, E and T, so choosing a new one replaces the old one
def choose_role(panel, button):
    settings = panel.node.settings
    column = button.value
    role = button.text
    current = role_of(settings, column)

    clear_role(settings, column)
    if current != role:
        if role == "Y":
            settings["dependent"] = column
        elif role == "E":
            settings["entity"] = column
        elif role == "T":
            settings["time"] = column
        elif role == "X":
            settings.setdefault("independent", []).append(column)
        else:
            settings.setdefault("controls", []).append(column)

    show_model(panel)


def number_text(value):
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.3g}"


def periods_text(fit):
    if fit["balanced"]:
        return f"{fit['periods_min']} periods each"
    return f"{fit['periods_min']} to {fit['periods_max']} periods, {fit['periods_mean']:.1f} on average"


# The fit statistics shown above the results
def fit_lines(fit):
    lines = [
        f"N: {fit['observations']}    Entities: {fit['entities']} ({periods_text(fit)})",
        f"R squared within: {number_text(fit['r_squared_within'])}    "
        f"Between: {number_text(fit['r_squared_between'])}    Overall: {number_text(fit['r_squared_overall'])}",
        f"sigma_u: {number_text(fit['sigma_u'])}    sigma_e: {number_text(fit['sigma_e'])}    "
        f"rho: {number_text(fit['rho'])}    Standard errors: {clusters.errors_text(fit)}",
    ]
    if fit["model"] == "Fixed Effects":
        lines.append(f"F test that all entity effects are 0: p = {number_text(fit['f_effects_p'])}")
    else:
        theta = number_text(fit["theta_median"])
        if fit["theta_min"] != fit["theta_max"]:
            theta = f"{number_text(fit['theta_min'])} to {number_text(fit['theta_max'])}"
        lines.append(f"theta: {theta}    Breusch-Pagan LM test for entity effects: p = {number_text(fit['lm_effects_p'])}")
    return lines


# Lists the fields from the last run with role buttons (text and date fields can only be E or T,
# though text ones can also be X or C as dummies), then the last results
def show_model(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    panel.lines = [
        f"Y: {settings.get('dependent') or 'none'}    Entity (E): {settings.get('entity') or 'none'}    "
        f"Time (T): {settings.get('time') or 'none'}",
        f"X: {', '.join(settings.get('independent', [])) or 'none'}    "
        f"C: {', '.join(settings.get('controls', [])) or 'none'}",
    ]

    data = node.last_inputs[0]
    roles = dummies.model_columns(data)
    for column in data.columns:
        role = role_of(settings, column)
        if column in roles:
            names = ["Y", "X", "C", "E", "T"]
        else:
            names = ["E", "T"]
        panel.add_option(column, [(r, choose_role, role == r) for r in names])

    if node.last_outputs:
        results = node.last_outputs[0]
        fit = results.attrs
        panel.lines += fit_lines(fit)
        warnings = list(fit["notes"])
        if fit.get("cluster"):
            warnings += fit["cluster"]["warnings"]
        panel.note = " ".join(warnings)
        panel.preview = results
    else:
        panel.message = "Choose Y, X and the entity field (E), then run the workflow to fit the model"
