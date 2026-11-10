import numpy as np
import pandas as pd
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with a numeric or ordinal outcome (Y) and a column naming groups (G), or a column for each of two or more repeated measures (W)"),
]
OUTPUT_SPECS = [
    ("Test", "the test used with its statistic, df, z, p_value and an effect size"),
    ("Ranks", "the count, median and mean rank of each group or measure"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "group", "within"]

# Groups compares separate groups and Repeated compares measures of the same people. The test
# follows from how many there are: two groups get Mann-Whitney and more get Kruskal-Wallis;
# two measures get the Wilcoxon signed-rank test and more get Friedman
DESIGNS = ["Groups", "Repeated"]
ROLES = {
    "Groups": [("Y", "dependent", False, True), ("G", "group", False, False)],
    "Repeated": [("W", "within", True, True)],
}


# Opens a panel to choose the design and its variables, showing the test of the last run
def on_click(node):
    panel = common.open_panel(node, "Rank Tests", show_test, "Test")
    common.add_choice(panel, "Design", "design", DESIGNS)
    common.add_choice(panel, "Show", "show", ["Test", "Ranks"])
    show_test(panel)
    return panel


def design_of(settings):
    design = settings.get("design", DESIGNS[0])
    return design if design in DESIGNS else DESIGNS[0]


def show_test(panel):
    settings = panel.node.settings
    design = design_of(settings)
    if common.list_fields(panel, ROLES[design]) is None:
        return
    if design == "Groups":
        panel.lines = [f"Outcome (Y): {common.names(settings, 'dependent')}    Groups (G): {common.names(settings, 'group')}"]
        hint = "Choose an outcome (Y) and a column naming the groups (G), then run the workflow"
    else:
        panel.lines = [f"Measures (W): {common.names(settings, 'within')}"]
        hint = "Click W on two or more measures of the same people, then run the workflow"
    ranks = settings.get("show") == "Ranks"
    panel.preview_title = "Ranks" if ranks else "Test"
    common.show_results(panel, hint, output=1 if ranks else 0)


# The sum of t cubed minus t over each set of t tied values, which the tests' variances are cut by
def tie_term(values):
    counts = np.unique(values, return_counts=True)[1]
    return float(np.sum(counts ** 3 - counts))


# Mann-Whitney U for two separate groups. U counts the pairs in which the first group's value
# is the higher. The rank-biserial correlation is the share of pairs favouring the first group
# minus the share favouring the second, from -1 to 1
def mann_whitney(groups):
    (label_1, a), (label_2, b) = groups
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    n_1, n_2 = len(a), len(b)
    total = n_1 + n_2
    spread = np.sqrt(n_1 * n_2 / 12 * (total + 1 - tie_term(np.concatenate([a, b])) / (total * (total - 1))))
    return {
        "test": "Mann-Whitney U", "statistic": result.statistic, "df": np.nan,
        "z": (result.statistic - n_1 * n_2 / 2) / spread if spread else np.nan,
        "p_value": float(result.pvalue), "effect_size": 2 * result.statistic / (n_1 * n_2) - 1,
        "effect": "rank-biserial r", "n": total,
    }


# Kruskal-Wallis H for three or more separate groups, with epsilon squared: the share of the
# variation in ranks that the groups account for
def kruskal_wallis(groups):
    result = stats.kruskal(*[values for label, values in groups])
    total = sum(len(values) for label, values in groups)
    return {
        "test": "Kruskal-Wallis H", "statistic": result.statistic, "df": len(groups) - 1, "z": np.nan,
        "p_value": float(result.pvalue), "effect_size": result.statistic / (total - 1),
        "effect": "epsilon squared", "n": total,
    }


# Wilcoxon signed-rank test for two measures of the same people. People who don't change are
# left out. The statistic is the sum of the ranks of the positive changes (first minus second),
# and the matched-pairs rank-biserial correlation is the positive share of the rank sum minus
# the negative share
def wilcoxon(scores):
    changes = scores[:, 0] - scores[:, 1]
    changes = changes[changes != 0]
    if len(changes) < 1:
        raise ValueError("No one's score changes between the two measures")
    ranks = stats.rankdata(np.abs(changes))
    positive, negative = ranks[changes > 0].sum(), ranks[changes < 0].sum()
    count = len(changes)
    spread = np.sqrt(count * (count + 1) * (2 * count + 1) / 24 - tie_term(np.abs(changes)) / 48)
    return {
        "test": "Wilcoxon signed-rank", "statistic": positive, "df": np.nan,
        "z": (positive - count * (count + 1) / 4) / spread if spread else np.nan,
        "p_value": float(stats.wilcoxon(changes).pvalue), "effect_size": (positive - negative) / (positive + negative),
        "effect": "rank-biserial r", "n": count,
    }


# Friedman's test for three or more measures of the same people, with Kendall's W: how far the
# people agree on the order of the measures, from 0 to 1
def friedman(scores):
    people, measures = scores.shape
    result = stats.friedmanchisquare(*scores.T)
    return {
        "test": "Friedman chi-square", "statistic": result.statistic, "df": measures - 1, "z": np.nan,
        "p_value": float(result.pvalue), "effect_size": result.statistic / (people * (measures - 1)),
        "effect": "Kendall's W", "n": people,
    }


# One result written the way a results section would
def describe(row):
    if row["test"] == "Mann-Whitney U":
        text = f"U = {row['statistic']:.1f}, z = {row['z']:.2f}"
    elif row["test"] == "Wilcoxon signed-rank":
        text = f"W = {row['statistic']:.1f}, z = {row['z']:.2f}"
    elif row["test"] == "Kruskal-Wallis H":
        text = f"H({row['df']:.0f}) = {row['statistic']:.2f}"
    else:
        text = f"chi-sq({row['df']:.0f}) = {row['statistic']:.2f}"
    verdict = "a significant difference" if row["p_value"] < common.LEVEL else "no significant difference"
    return f"{row['test']}: {text}, {common.p_text(row['p_value'])}, {row['effect']} = {common.short(row['effect_size'])}: {verdict} at the 5% level."


# Runs the rank test that suits the design and the number of groups or measures
def run(inputs, settings):
    data = common.clean(inputs[0])
    notes = []

    if design_of(settings) == "Groups":
        dependent, group = settings.get("dependent"), settings.get("group")
        if not dependent or not group:
            raise ValueError("Choose an outcome (Y) and a column naming the groups (G)")
        common.check_fields(data, [dependent, group], numeric=[dependent])
        groups = common.grouped(data, dependent, group)
        if len(groups) < 2:
            raise ValueError(f"{group} needs at least 2 groups with values of {dependent}")
        row = mann_whitney(groups) if len(groups) == 2 else kruskal_wallis(groups)

        # Mean ranks come from ranking every value together, whichever group it is in
        ranks = stats.rankdata(np.concatenate([values for label, values in groups]))
        ends = np.cumsum([len(values) for label, values in groups])
        summary = [
            {"group": label, "n": len(values), "median": np.median(values), "mean_rank": ranks[end - len(values):end].mean()}
            for (label, values), end in zip(groups, ends)
        ]
        if len(groups) > 2 and row["p_value"] < common.LEVEL:
            notes.append("To see which groups differ, compare pairs with Filter and this module, correcting for the number of pairs.")
    else:
        within = settings.get("within") or []
        if len(within) < 2:
            raise ValueError("Choose at least 2 measures (W)")
        common.check_fields(data, within, numeric=within)
        rows = data[within].dropna().astype(float)
        if len(rows) < 2:
            raise ValueError("Fewer than 2 rows have every measure")
        scores = rows.to_numpy()
        row = wilcoxon(scores) if len(within) == 2 else friedman(scores)

        # Mean ranks come from ranking each person's own measures
        ranks = stats.rankdata(scores, axis=1)
        summary = [
            {"measure": name, "n": len(rows), "median": rows[name].median(), "mean_rank": ranks[:, i].mean()}
            for i, name in enumerate(within)
        ]
        if len(rows) < len(data):
            notes.append(f"Left out {len(data) - len(rows)} rows missing a measure.")

    table = significance.add_stars(pd.DataFrame([row]), after="z")
    table.attrs = {"note": " ".join([describe(row)] + notes)}
    return [table, pd.DataFrame(summary)]
