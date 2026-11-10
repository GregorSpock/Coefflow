from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with a numeric outcome (Y) and a column naming three or more groups (G)"),
]
OUTPUT_SPECS = [
    ("Comparisons", "one row per pair of groups with the difference in their means, t, the p_value corrected for the number of pairs, its confidence interval and Cohen's d"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "group"]
ROLES = [("Y", "dependent", False, True), ("G", "group", False, False)]
METHODS = ["Tukey", "G-Howell", "Bonferroni", "Holm"]
FULL_NAMES = {"Tukey": "Tukey's HSD", "G-Howell": "Games-Howell", "Bonferroni": "Bonferroni", "Holm": "Holm"}
# The studentised range is slow to work out, and more groups than this make too many pairs to read
MAX_COMPARED = 12


# Opens a panel to choose the outcome, the groups and the correction, showing the comparisons of the last run
def on_click(node):
    panel = common.open_panel(node, "Post Hoc", show_comparisons, "Comparisons")
    common.add_choice(panel, "Adj", "method", METHODS)
    show_comparisons(panel)
    return panel


def show_comparisons(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    method = FULL_NAMES.get(settings.get("method"), FULL_NAMES[METHODS[0]])
    panel.lines = [f"Outcome (Y): {common.names(settings, 'dependent')}    Groups (G): {common.names(settings, 'group')}    Correction: {method}"]
    common.show_results(panel, "Choose an outcome (Y) and a column naming the groups (G), then run the workflow")


# Holm's step-down correction: the smallest p value is multiplied by the number of tests, the
# next by one fewer and so on, never letting a corrected value fall below the one before it
def holm(p_values):
    order = np.argsort(p_values)
    corrected = np.empty(len(p_values))
    highest = 0.0
    for rank, index in enumerate(order):
        highest = max(highest, min(1.0, p_values[index] * (len(p_values) - rank)))
        corrected[index] = highest
    return corrected


# Compares every pair of groups. Tukey, Bonferroni and Holm pool the spread of all the groups,
# as the ANOVA does, while Games-Howell uses each pair's own spreads, so it is the one to use
# when the groups vary by different amounts. Tukey and Games-Howell correct through the
# studentised range; Bonferroni and Holm correct the p values of ordinary t-tests
def compare(groups, method):
    count = len(groups)
    total = sum(len(values) for label, values in groups)
    df_error = total - count
    mean_square_error = sum(((values - values.mean()) ** 2).sum() for label, values in groups) / df_error
    pairs = list(combinations(range(count), 2))
    level = 1 - common.LEVEL

    rows = []
    for i, j in pairs:
        (label_1, a), (label_2, b) = groups[i], groups[j]
        difference = a.mean() - b.mean()
        if method == "G-Howell":
            error = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            df = error ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)) if error else df_error
        else:
            error = np.sqrt(mean_square_error * (1 / len(a) + 1 / len(b)))
            df = df_error
        if error == 0:
            raise ValueError("The groups don't vary, so there is nothing to test")
        t = difference / error

        if method in ("Tukey", "G-Howell"):
            p_value = float(stats.studentized_range.sf(abs(t) * np.sqrt(2), count, df))
            reach = stats.studentized_range.ppf(level, count, df) / np.sqrt(2) * error
        else:
            p_value = float(2 * stats.t.sf(abs(t), df))
            reach = stats.t.ppf(1 - common.LEVEL / (2 * len(pairs)), df) * error

        rows.append({
            "group_1": label_1, "group_2": label_2, "mean_1": a.mean(), "mean_2": b.mean(), "mean_diff": difference,
            "se": error, "t": t, "df": df, "p_value": p_value, "ci_low": difference - reach, "ci_high": difference + reach,
            "cohen_d": difference / common.pooled_sd(a, b) if common.pooled_sd(a, b) else np.nan,
        })

    table = pd.DataFrame(rows)
    if method == "Bonferroni":
        table["p_value"] = (table["p_value"] * len(pairs)).clip(upper=1.0)
    elif method == "Holm":
        table["p_value"] = holm(table["p_value"].to_numpy())
    return table


# Compares the mean of Y between every pair of groups, correcting for the number of pairs
def run(inputs, settings):
    data = common.clean(inputs[0])
    dependent, group = settings.get("dependent"), settings.get("group")
    if not dependent or not group:
        raise ValueError("Choose an outcome (Y) and a column naming the groups (G)")
    common.check_fields(data, [dependent, group], numeric=[dependent])
    method = settings.get("method", METHODS[0])

    groups = common.grouped(data, dependent, group)
    small = [label for label, values in groups if len(values) < 2]
    groups = [(label, values) for label, values in groups if len(values) >= 2]
    if len(groups) < 2:
        raise ValueError(f"{group} needs at least 2 groups with 2 or more values of {dependent}")
    if len(groups) > MAX_COMPARED:
        raise ValueError(f"{group} has {len(groups)} groups; at most {MAX_COMPARED} can be compared")

    table = significance.add_stars(compare(groups, method), after="t")
    different = table[table["p_value"] < common.LEVEL]
    notes = [f"{FULL_NAMES[method]} comparisons: {len(different)} of {len(table)} pairs differ at the 5% level."]
    if len(different):
        pairs = [f"{row['group_1']} vs {row['group_2']} ({common.p_text(row['p_value'])})" for index, row in different.head(4).iterrows()]
        notes.append("Differing: " + ", ".join(pairs) + ("..." if len(different) > 4 else "") + ".")
    notes.append("p values and confidence intervals are corrected for the number of pairs.")
    if method == "Holm":
        notes.append("Holm has no intervals of its own, so the Bonferroni ones are shown.")
    if small:
        notes.append(f"Left out groups with a single value: {', '.join(small)}.")

    table.attrs = {"note": " ".join(notes)}
    return [table]
