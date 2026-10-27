import numpy as np
import pandas as pd
from scipy import stats

from modules.arimax import module as arimax
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Fitted data", "a table with a residual column (R), e.g. Out 2 of OLS, with a time column (T) if the rows aren't in time order and a group column (G) for panel data"),
]
OUTPUT_SPECS = [
    ("Test", "one row per series with the Durbin-Watson statistic, the autocorrelation it implies, and a p value. With G there is a pooled row for the whole panel first"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["field", "time", "group"]
ROLE_KEYS = {"R": "field", "T": "time", "G": "group"}
# The residual column picked when none is chosen, as OLS, Fixed Effects and ARIMAX all call it this
DEFAULT_FIELD = "residual"
LEVEL = 0.05
MIN_VALUES = 3
POOLED = "All (pooled)"


# Opens a panel to choose the residual column, the time column and the column naming each series,
# showing the Durbin-Watson statistic of the last run
def on_click(node):
    panel = Panel(node, "Durbin-Watson")
    panel.preview_title = "Test"
    panel.refresh = show_test
    show_test(panel)
    return panel


# The residual column: the one chosen, or the residual column if there is one
def residual_field(data, settings):
    if settings.get("field"):
        return settings["field"]
    if DEFAULT_FIELD in data.columns:
        return DEFAULT_FIELD
    return None


# Returns "R", "T", "G" or None for a field
def role_of(data, settings, column):
    if residual_field(data, settings) == column:
        return "R"
    for role, key in ROLE_KEYS.items():
        if settings.get(key) == column:
            return role
    return None


# Gives a field the clicked role, or clears it if it already had that role.
# Each role belongs to one field, so choosing a new one replaces the old
def choose_role(panel, button):
    settings = panel.node.settings
    data = panel.node.last_inputs[0]
    current = role_of(data, settings, button.value)
    for key in ROLE_KEYS.values():
        if settings.get(key) == button.value:
            settings[key] = None
    if current != button.text:
        settings[ROLE_KEYS[button.text]] = button.value
    show_test(panel)


# Lists every field with its role buttons, then the results of the last run
def show_test(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    panel.lines = [
        f"Residual (R): {residual_field(data, settings) or 'none'}    "
        f"Time: {settings.get('time') or 'row order'}    Series: {settings.get('group') or 'none'}",
    ]

    numeric = data.select_dtypes("number").columns
    for column in data.columns:
        role = role_of(data, settings, column)
        roles = ["R", "T", "G"] if column in numeric else ["T", "G"]
        panel.add_option(column, [(r, choose_role, role == r) for r in roles])

    if not node.last_outputs:
        panel.message = "Choose the residual column (R), and T if the rows aren't in time order, then run the workflow"
        return

    table = node.last_outputs[0]
    test = table.attrs
    first = table.iloc[0]
    panel.message = ""
    panel.lines.append(
        f"Durbin-Watson: {first['durbin_watson']:.3f}    Implied rho: {first['rho']:.3f}    "
        f"p: {arimax.number_text(first['p_value'])}    N: {first['observations']}"
    )
    # The verdict goes in the note, which wraps, as it is too long for one line
    panel.note = " ".join([test["verdict"]] + test["notes"])
    panel.preview = table


# The Durbin-Watson statistic from residuals in time order, one array per series: the sum of
# squared changes between neighbouring residuals within each series over the sum of squared
# residuals. With several series this is the panel version of Bhargava, Franzini and Narendranathan
# (1982). Also returns the number of neighbouring pairs it used
def durbin_watson(series):
    changes = sum(float(np.sum(np.diff(values) ** 2)) for values in series)
    squares = sum(float(np.sum(values ** 2)) for values in series)
    pairs = sum(len(values) - 1 for values in series)
    return changes / squares, pairs


# One row of the results. As d is about 2(1 - rho), the first-order autocorrelation of the residuals
# is about 1 - d/2, and with no autocorrelation rho times the square root of the number of pairs is
# roughly standard normal. That gives a large-sample p value; the exact one depends on the regressors
def test_row(name, series):
    d, pairs = durbin_watson(series)
    rho = 1 - d / 2
    z = rho * np.sqrt(pairs)
    return {
        "series": name,
        "observations": sum(len(values) for values in series),
        "durbin_watson": d,
        "rho": rho,
        "z_stat": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


# Says in plain English what the statistic means for the regression
def describe(row):
    d = row["durbin_watson"]
    if row["p_value"] >= LEVEL:
        return f"d = {d:.2f}, close enough to 2 that there is no evidence of autocorrelation at the {LEVEL:.0%} level."
    if d < 2:
        return (
            f"d = {d:.2f} is well below 2, so the residuals are positively autocorrelated and the standard errors are too small. "
            "Cluster them, add lags, or model the errors with ARIMAX."
        )
    return (
        f"d = {d:.2f} is well above 2, so the residuals are negatively autocorrelated, "
        "which often means a field has been differenced too many times."
    )


# Runs the Durbin-Watson test on the residuals of each series in time order. Without G the whole
# table is one series; with G each series is tested on its own, and the pooled panel statistic
# comes first as it is the one to judge the model by
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    field = residual_field(data, settings)
    if not field:
        raise ValueError("Choose the residual column (R)")
    group = settings.get("group")
    missing = [c for c in [field, settings.get("time"), group] if c and c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    if not pd.api.types.is_numeric_dtype(data[field]):
        raise ValueError(f"{field} is not numeric")

    notes = []
    if group:
        dropped = data[group].isna().sum()
        if dropped:
            notes.append(f"Dropped {dropped} rows with no {group} value.")
        parts = [(str(name), rows) for name, rows in data.dropna(subset=[group]).groupby(group, sort=True)]
    else:
        parts = [("All", data)]

    series = {}
    for name, rows in parts:
        rows, times = arimax.sort_by_time(rows, settings, notes)
        values = rows[field].dropna().to_numpy(float)
        if len(values) >= MIN_VALUES:
            series[name] = values

    if not series:
        raise ValueError(f"No series has at least {MIN_VALUES} residuals")
    skipped = len(parts) - len(series)
    if skipped:
        notes.append(f"Left out {skipped} series with fewer than {MIN_VALUES} residuals.")
    blanks = data[field].isna().sum()
    if blanks:
        notes.append(f"Skipped {blanks} blank residuals, so the values either side of each are treated as neighbours.")

    rows = [test_row(name, [values]) for name, values in series.items()]
    if group:
        rows.insert(0, test_row(POOLED, list(series.values())))

    table = pd.DataFrame(rows)
    # sort_by_time adds the same note for every series, so each one is kept only once
    table.attrs = {"verdict": describe(rows[0]), "notes": list(dict.fromkeys(notes))}
    return [table]
