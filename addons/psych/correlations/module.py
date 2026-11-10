from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with two or more numeric variables (V), and optionally numeric variables to control for (C)"),
]
OUTPUT_SPECS = [
    ("Pairs", "one row per pair of variables with the correlation r, its p_value and confidence interval, and the number of rows used"),
    ("Matrix", "the correlations as a square table"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["variables", "controls"]
ROLES = [("V", "variables", True, True), ("C", "controls", True, True)]
METHODS = ["Pearson", "Spearman", "Kendall"]
SYMBOLS = {"Pearson": "r", "Spearman": "rho", "Kendall": "tau"}
MIN_ROWS = 4


# Opens a panel to choose the variables, controls and kind of correlation, showing those of the last run
def on_click(node):
    panel = common.open_panel(node, "Correlations", show_correlations, "Correlations")
    common.add_choice(panel, "Type", "method", METHODS)
    common.add_choice(panel, "Show", "show", ["Pairs", "Matrix"])
    show_correlations(panel)
    return panel


def show_correlations(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [f"Variables (V): {common.names(settings, 'variables')}    Controlling for (C): {common.names(settings, 'controls', 'nothing')}"]
    matrix = settings.get("show") == "Matrix"
    panel.preview_title = "Correlation Matrix" if matrix else "Correlations"
    common.show_results(panel, "Click V on two or more variables, then run the workflow", output=1 if matrix else 0)


# What is left of each column once the controls' straight-line effect on it is taken away
def residuals(values, controls):
    design = np.column_stack([np.ones(len(controls)), controls])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


# The correlation of two variables over the rows that have both, and any controls, with its
# p value and 95% confidence interval. With controls it is the partial correlation: that of what
# is left of each variable after the controls are taken out (of their ranks, for Spearman).
# The interval comes from Fisher's z, widened for Spearman as Fieller suggested; Kendall has none
def correlate(data, first, second, controls, method):
    rows = data[[first, second] + controls].dropna().astype(float)
    count = len(rows)
    if count < MIN_ROWS + len(controls):
        return np.nan, np.nan, np.nan, np.nan, count
    if method == "Spearman":
        rows = rows.rank()
    x, y = rows[first].to_numpy(), rows[second].to_numpy()
    if x.std() == 0 or y.std() == 0:
        return np.nan, np.nan, np.nan, np.nan, count

    if method == "Kendall":
        result = stats.kendalltau(x, y)
        return result.statistic, result.pvalue, np.nan, np.nan, count

    if controls:
        held = rows[controls].to_numpy()
        x, y = residuals(x, held), residuals(y, held)
        if x.std() < 1e-12 or y.std() < 1e-12:
            return np.nan, np.nan, np.nan, np.nan, count
    r = float(np.corrcoef(x, y)[0, 1])
    df = count - 2 - len(controls)
    if abs(r) >= 1:
        return r, 0.0, r, r, count
    t = r * np.sqrt(df / (1 - r ** 2))
    p_value = float(2 * stats.t.sf(abs(t), df))

    error = np.sqrt((1.06 if method == "Spearman" else 1) / (count - 3 - len(controls)))
    reach = stats.norm.ppf(1 - common.LEVEL / 2) * error
    return r, p_value, np.tanh(np.arctanh(r) - reach), np.tanh(np.arctanh(r) + reach), count


# Correlates every pair of the chosen variables
def run(inputs, settings):
    data = common.clean(inputs[0])
    variables = settings.get("variables") or []
    controls = settings.get("controls") or []
    method = settings.get("method", METHODS[0])
    if len(variables) < 2:
        raise ValueError("Choose at least 2 variables (V)")
    common.check_fields(data, variables + controls, numeric=variables + controls)
    if controls and method == "Kendall":
        raise ValueError("Controls work with Pearson and Spearman correlations, not Kendall's tau")

    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    pairs = []
    for first, second in combinations(variables, 2):
        r, p_value, low, high, count = correlate(data, first, second, controls, method)
        matrix.loc[first, second] = matrix.loc[second, first] = r
        pairs.append({"variable_1": first, "variable_2": second, "r": r, "p_value": p_value, "ci_low": low, "ci_high": high, "n": count})
    table = significance.add_stars(pd.DataFrame(pairs), after="r")

    symbol = SYMBOLS[method]
    kind = f"Partial {method} correlations controlling for {', '.join(controls)}" if controls else f"{method} correlations"
    notes = [f"{kind} ({symbol}), each over the rows that have both variables."]
    strongest = table.dropna(subset=["r"]).sort_values("r", key=abs, ascending=False).head(1)
    for index, row in strongest.iterrows():
        df = row["n"] - 2 - len(controls)
        shown = f"{symbol}({df}) = " if method != "Kendall" else f"{symbol} = "
        notes.append(f"Strongest: {row['variable_1']} and {row['variable_2']}, {shown}{common.short(row['r'])}, {common.p_text(row['p_value'])}.")
    if len(table) > 1:
        notes.append(f"{len(table)} pairs are tested, so some may pass by chance; the p values are not corrected for that.")
    if table["r"].isna().any():
        notes.append("Pairs with too few rows, or a variable that doesn't vary, are left blank.")

    table.attrs = {"note": " ".join(notes)}
    matrix = matrix.round(4).reset_index().rename(columns={"index": "variable"})
    return [table, matrix]
