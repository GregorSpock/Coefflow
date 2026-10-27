import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from modules.arimax import module as arimax
from modules.arimax.module import MAX_DIFF, chosen_series, next_series
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a time series table: numeric fields to test, a time column (T), and a group column (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Tests", "one row per field with the ADF and KPSS p values at each level of differencing and the d each test suggests"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["fields", "time", "group"]
ROLE_KEYS = {"T": "time", "G": "group"}
# The significance level both tests are judged at
LEVEL = 0.05
MIN_VALUES = 10


# Opens a panel to choose the fields to test, the time column and the series, showing for each
# field how many times it needs differencing before it is stationary
def on_click(node):
    panel = Panel(node, "ADF")
    panel.add_button("Next series", next_series)
    panel.add_button(trend_label(node.settings), toggle_trend)
    panel.preview_title = "Stationarity"
    panel.refresh = show_tests
    show_tests(panel)
    return panel


def trend_label(settings):
    return "Trend: On" if settings.get("trend") else "Trend: Off"


# Switches between testing for stationarity around a constant, or around a straight-line trend
def toggle_trend(panel, button):
    settings = panel.node.settings
    settings["trend"] = not settings.get("trend", False)
    button.text = trend_label(settings)
    show_tests(panel)


# Returns "Test", "T", "G" or None for a field
def role_of(settings, column):
    for role, key in ROLE_KEYS.items():
        if settings.get(key) == column:
            return role
    if column in settings.get("fields", []):
        return "Test"
    return None


# Gives a field the clicked role, or clears it if it already had that role.
# T and G belong to one field each, so choosing a new one replaces the old
def choose_role(panel, button):
    settings = panel.node.settings
    column, role = button.value, button.text
    current = role_of(settings, column)

    for key in ROLE_KEYS.values():
        if settings.get(key) == column:
            settings[key] = None
    settings["fields"] = [c for c in settings.get("fields", []) if c != column]
    if current != role:
        if role == "Test":
            settings["fields"].append(column)
        else:
            settings[ROLE_KEYS[role]] = column
    if role == "G":
        settings["series"] = None
    show_tests(panel)


# The fields to test: those chosen, or every numeric field other than T and G if none are
def tested_fields(data, settings):
    chosen = [c for c in settings.get("fields", []) if c in data.columns]
    if chosen:
        return chosen
    others = [settings.get("time"), settings.get("group")]
    return [c for c in data.select_dtypes("number").columns if c not in others]


# Lists every field with its role buttons, then the results of the last run
def show_tests(panel):
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
    series = chosen_series(data, settings)
    group = f"{settings.get('group')} = {series}" if series is not None else "none"
    fields = tested_fields(data, settings)
    panel.lines = [
        f"Testing: {', '.join(fields) if settings.get('fields') else 'every numeric field'}    "
        f"Time: {settings.get('time') or 'row order'}    Series: {group}",
    ]

    numeric = data.select_dtypes("number").columns
    for column in data.columns:
        role = role_of(settings, column)
        roles = ["Test", "T", "G"] if column in numeric else ["T", "G"]
        panel.add_option(column, [(r, choose_role, role == r) for r in roles])

    if node.last_outputs:
        table = node.last_outputs[0]
        panel.lines.append(
            f"Stationary around: {'a trend' if settings.get('trend') else 'a constant'}    "
            f"Judged at the {LEVEL:.0%} level    Suggested d: "
            + ", ".join(f"{row.field} {row.suggested_d}" for row in table.itertuples())
        )
        panel.note = " ".join(table.attrs["notes"])
        panel.preview = table
    else:
        panel.message = "Choose the fields to test (or none for all) and T, then run the workflow"


# The p values of the ADF test (a small p means no unit root, so stationary) and the KPSS test
# (a small p means not stationary) for a series, or NaN for a test that can't be run on it.
# KPSS only gives p values from 0.01 to 0.1, so p values at those ends are as far as it can say
def test_p_values(values, regression):
    values = values.dropna()
    if len(values) < MIN_VALUES or values.std() == 0:
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            adf_p = adfuller(values, regression=regression, autolag="AIC")[1]
        except (ValueError, np.linalg.LinAlgError):
            adf_p = np.nan
        try:
            kpss_p = kpss(values, regression=regression, nlags="auto")[1]
        except (ValueError, OverflowError):
            kpss_p = np.nan
    return adf_p, kpss_p


# The fewest differences at which a test says the series is stationary, or None if it never does
def first_stationary(p_values, stationary):
    for d, p_value in enumerate(p_values):
        if not np.isnan(p_value) and stationary(p_value):
            return d
    return None


# Describes the result for one field, saying what to do when the two tests disagree
def describe(field, adf_d, kpss_d, suggested):
    if adf_d is None and kpss_d is None:
        return f"{field} isn't stationary even after {MAX_DIFF} differences, so check it for breaks or outliers."
    if adf_d != kpss_d:
        return (
            f"{field}: ADF says d = {'over ' + str(MAX_DIFF) if adf_d is None else adf_d} but KPSS says "
            f"d = {'over ' + str(MAX_DIFF) if kpss_d is None else kpss_d}, so d = {suggested} is the safer choice."
        )
    return ""


# Runs the ADF and KPSS tests on each field in time order, as it is and after each difference
# up to MAX_DIFF. ADF tests for a unit root and KPSS tests for stationarity, so they check each
# other: the suggested d is the fewest differences at which both say the field is stationary
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    missing = [c for c in [settings.get("time"), settings.get("group")] if c and c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    fields = tested_fields(data, settings)
    if not fields:
        raise ValueError("There are no numeric fields to test")
    text = [c for c in fields if not pd.api.types.is_numeric_dtype(data[c])]
    if text:
        raise ValueError(f"These fields are not numeric: {', '.join(text)}")

    data = arimax.pick_series(data, settings)
    data, times = arimax.sort_by_time(data, settings, [])
    regression = "ct" if settings.get("trend") else "c"
    rows, notes = [], []

    for field in fields:
        values = data[field].astype(float)
        row = {"field": field, "observations": int(values.notna().sum())}
        adf_ps, kpss_ps = [], []
        for d in range(MAX_DIFF + 1):
            adf_p, kpss_p = test_p_values(values, regression)
            row[f"adf_p_d{d}"], row[f"kpss_p_d{d}"] = adf_p, kpss_p
            adf_ps.append(adf_p)
            kpss_ps.append(kpss_p)
            values = values.diff()

        adf_d = first_stationary(adf_ps, lambda p: p < LEVEL)
        kpss_d = first_stationary(kpss_ps, lambda p: p > LEVEL)
        both = [d for d in range(MAX_DIFF + 1) if adf_ps[d] < LEVEL and kpss_ps[d] > LEVEL]
        suggested = both[0] if both else max(d if d is not None else MAX_DIFF for d in [adf_d, kpss_d])
        row.update({"adf_d": adf_d, "kpss_d": kpss_d, "suggested_d": suggested})
        rows.append(row)
        if row["observations"] < MIN_VALUES:
            notes.append(f"{field} has too few values to test.")
        elif describe(field, adf_d, kpss_d, suggested):
            notes.append(describe(field, adf_d, kpss_d, suggested))

    # The suggested d first, as it is what the table is for; a test that never says stationary has no d
    table = pd.DataFrame(rows)
    first = ["field", "suggested_d", "adf_d", "kpss_d", "observations"]
    table = table[first + [c for c in table.columns if c not in first]]
    table[["adf_d", "kpss_d"]] = table[["adf_d", "kpss_d"]].astype("Int64")
    table.attrs = {"notes": notes}
    return [table]
