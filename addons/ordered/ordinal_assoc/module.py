import numpy as np
import pandas as pd
from scipy import stats

from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with an ordered numeric Y column and one or more numeric X columns"),
]
OUTPUT_SPECS = [
    ("Association", "one row per X with Goodman-Kruskal gamma, Kendall's tau-b, Somers' D and Spearman's rho, each with a p_value"),
]

FIELD_SETTINGS = ["dependent", "independent"]


# Measures and tests the association between an ordered Y and each X without assuming a model.
# All four measures run from -1 to 1 and only use the order of the values, not the gaps between
# them. Gamma ignores tied pairs, tau-b and Somers' D count them, and Somers' D treats Y as the
# outcome. Spearman's rho is the correlation of the ranks
def on_click(node):
    return common.open_panel(node, "Ordinal Assoc", show_association)


def show_association(panel):
    if not common.show_roles(panel, controls=False, text_x=False):
        return
    node = panel.node
    if not node.last_outputs:
        panel.message = "Choose an ordered Y and one or more numeric X variables, then run the workflow"
        panel.lines.append(common.Y_HINT)
        return
    panel.note = " ".join(node.last_outputs[0].attrs["warnings"])
    panel.preview = node.last_outputs[0]


# For each cell of a cross table, the number of rows concordant with it (above and to the left,
# or below and to the right) and discordant with it (the other two corners)
def concordance(table):
    def below_right(t):
        # Sum of the cells strictly below and strictly to the right of each cell
        totals = t[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
        shifted = np.zeros_like(totals)
        shifted[:-1, :-1] = totals[1:, 1:]
        return shifted

    concordant = below_right(table) + below_right(table[::-1, ::-1])[::-1, ::-1]
    discordant = below_right(table[:, ::-1])[:, ::-1] + below_right(table[::-1, :])[::-1, :]
    return concordant, discordant


# Goodman-Kruskal gamma with its p value, using the standard error under no association
def gamma(y, x):
    table = pd.crosstab(y, x).to_numpy(dtype=float)
    concordant, discordant = concordance(table)
    p = (table * concordant).sum()
    q = (table * discordant).sum()
    if p + q == 0:
        return np.nan, np.nan
    value = (p - q) / (p + q)
    variance = (table * (concordant - discordant) ** 2).sum() - (p - q) ** 2 / table.sum()
    error = 2 / (p + q) * np.sqrt(max(variance, 0.0))
    if error == 0:
        return value, np.nan
    return value, 2 * stats.norm.sf(abs(value / error))


def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    common.check_columns(data, settings)
    dependent = settings["dependent"]
    notes = []

    rows = []
    for column in settings["independent"]:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{column} must be numeric, so its values have an order")
        pair = data[[dependent, column]].dropna()
        if len(pair) < len(data):
            notes.append(f"{len(data) - len(pair)} rows missing {dependent} or {column} were left out of its row.")
        y, x = pair[dependent].to_numpy(dtype=float), pair[column].to_numpy(dtype=float)
        if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
            raise ValueError(f"{dependent} and {column} must each take at least 2 values")

        gamma_value, gamma_p = gamma(y, x)
        tau = stats.kendalltau(x, y, variant="b")
        somers = stats.somersd(x, y)
        spearman = stats.spearmanr(x, y)
        rows.append({
            "variable": column,
            "gamma": gamma_value,
            "gamma_p": gamma_p,
            "tau_b": tau.statistic,
            "tau_b_p": tau.pvalue,
            "somers_d": somers.statistic,
            "somers_d_p": somers.pvalue,
            "spearman": spearman.statistic,
            "spearman_p": spearman.pvalue,
            "observations": len(pair),
        })

    results = pd.DataFrame(rows)
    results.attrs = {"warnings": notes}
    return [results]
