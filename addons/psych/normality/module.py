import numpy as np
import pandas as pd
from scipy import stats

from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with the numeric variables to check (V), and optionally a column naming groups to check them within (G)"),
]
OUTPUT_SPECS = [
    ("Checks", "one row per variable, or per variable and group, with its skewness and kurtosis, their z scores, and the Shapiro-Wilk test"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["variables", "group"]
ROLES = [("V", "variables", True, True), ("G", "group", False, False)]
# Shapiro-Wilk needs at least this many values, and its p value is only approximate beyond the most
MIN_VALUES = 3
MOST_VALUES = 5000
# Skewness or kurtosis this many standard errors from zero is a clear departure from normal
Z_LIMIT = 1.96


# Opens a panel to choose the variables and an optional grouping, showing the checks of the last run
def on_click(node):
    panel = common.open_panel(node, "Normality", show_checks, "Checks")
    show_checks(panel)
    return panel


def show_checks(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [f"Variables (V): {common.names(settings, 'variables')}    Within groups (G): {common.names(settings, 'group', 'no')}"]
    common.show_results(panel, "Click V on the variables to check, then run the workflow")


# Skewness and excess kurtosis the way SPSS reports them (corrected for sample size), each with
# its z score, and the Shapiro-Wilk test, for one set of values
def check(values):
    count = len(values)
    row = {"n": count, "mean": values.mean(), "sd": values.std(ddof=1) if count > 1 else np.nan}
    series = pd.Series(values)
    skew_error = np.sqrt(6 * count * (count - 1) / ((count - 2) * (count + 1) * (count + 3))) if count > 2 else np.nan
    kurtosis_error = 2 * skew_error * np.sqrt((count ** 2 - 1) / ((count - 3) * (count + 5))) if count > 3 else np.nan
    row["skew"] = series.skew() if count > 2 else np.nan
    row["skew_z"] = row["skew"] / skew_error
    row["kurtosis"] = series.kurt() if count > 3 else np.nan
    row["kurtosis_z"] = row["kurtosis"] / kurtosis_error

    row["shapiro_w"], row["p_value"] = np.nan, np.nan
    if count >= MIN_VALUES and values.min() != values.max():
        result = stats.shapiro(values)
        row["shapiro_w"], row["p_value"] = result.statistic, result.pvalue
    row["normal"] = "" if pd.isna(row["p_value"]) else ("no" if row["p_value"] < common.LEVEL else "yes")
    return row


# Checks each variable, within each group if a grouping is chosen, against a normal distribution
def run(inputs, settings):
    data = common.clean(inputs[0])
    variables = settings.get("variables") or []
    group = settings.get("group")
    if not variables:
        raise ValueError("Choose at least one variable (V)")
    common.check_fields(data, variables + ([group] if group else []), numeric=variables)

    rows = []
    notes = []
    for name in variables:
        if group:
            groups = common.grouped(data, name, group)
            for label, values in groups:
                rows.append({"variable": name, group: label, **check(values)})
            testable = [values for label, values in groups if len(values) > 1]
            if len(testable) > 1:
                levene = stats.levene(*testable, center="median")
                verdict = "the groups vary by different amounts" if levene.pvalue < common.LEVEL else "the groups' variances look equal"
                notes.append(f"Levene's test for {name}: F = {levene.statistic:.2f}, {common.p_text(levene.pvalue)}, so {verdict}.")
        else:
            rows.append({"variable": name, **check(data[name].dropna().to_numpy(float))})

    table = pd.DataFrame(rows)
    failed = table[table["normal"] == "no"]
    summary = f"Shapiro-Wilk: {len(failed)} of {len(table)} sets of values depart from normal at the 5% level"
    if len(failed):
        summary += f" ({', '.join(dict.fromkeys(failed['variable']))})"
    notes.insert(0, summary + f". A skew or kurtosis z beyond {Z_LIMIT} either way shows what kind of departure it is.")
    if table["n"].max() > 200:
        notes.append("With a few hundred rows or more the test flags even tiny departures, so look at a Histogram too.")
    if table["n"].max() > MOST_VALUES:
        notes.append(f"Shapiro-Wilk p values are only approximate beyond {MOST_VALUES} values.")

    table.attrs = {"note": " ".join(notes)}
    return [table]
