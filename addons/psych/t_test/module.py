import numpy as np
import pandas as pd
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with the numeric variables to test, and a column naming two groups for a Groups test"),
]
OUTPUT_SPECS = [
    ("Test", "one row per test with the means, their difference and its confidence interval, t, df, p_value and Cohen's d"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["y", "group", "first", "second"]

# Groups compares two separate groups, Paired two measures of the same people, and 1 sample
# a mean with a set value
TESTS = ["Groups", "Paired", "1 sample"]
ROLES = {
    "Groups": [("Y", "y", True, True), ("G", "group", False, False)],
    "Paired": [("1st", "first", False, True), ("2nd", "second", False, True)],
    "1 sample": [("Y", "y", True, True)],
}
# How many tests the note describes in words; the table has them all
MAX_DESCRIBED = 3


# Opens a panel to choose the test and its variables, showing the results of the last run
def on_click(node):
    panel = common.open_panel(node, "T-Test", show_test, "Test")
    common.add_choice(panel, "Test", "test", TESTS)
    common.add_switch(panel, "Welch", "welch")
    show_test(panel)
    return panel


def test_of(settings):
    test = settings.get("test", TESTS[0])
    return test if test in TESTS else TESTS[0]


# Keeps the value a 1 sample test compares the mean with, as it is typed
def change_value(panel, text):
    panel.node.settings["value"] = text.strip()


# Lists the fields with the roles the chosen test uses, then the results. The box for the
# test value is only there for a 1 sample test
def show_test(panel):
    settings = panel.node.settings
    test = test_of(settings)
    if test == "1 sample" and panel.text_field is None:
        panel.add_text_field("Test value:", str(settings.get("value", "0")), change_value, blocked="")
    elif test != "1 sample":
        panel.text_field = None

    if common.list_fields(panel, ROLES[test]) is None:
        return
    if test == "Groups":
        panel.lines = [f"Variables (Y): {common.names(settings, 'y')}    Groups (G): {common.names(settings, 'group')}"]
        hint = "Choose one or more variables (Y) and a column naming two groups (G), then run the workflow"
    elif test == "Paired":
        panel.lines = [f"First measure: {common.names(settings, 'first')}    Second measure: {common.names(settings, 'second')}"]
        hint = "Choose the first and second measure of the same people, then run the workflow"
    else:
        panel.lines = [f"Variables (Y): {common.names(settings, 'y')}"]
        hint = "Choose one or more variables (Y) and type the value to test their mean against, then run the workflow"
    common.show_results(panel, hint)


# The t statistic, two-sided p value and 95% confidence interval for a difference with a standard error
def t_result(difference, error, df):
    if error == 0:
        raise ValueError("The values don't vary, so there is nothing to test")
    t = difference / error
    reach = stats.t.ppf(1 - common.LEVEL / 2, df) * error
    return {"t": t, "df": df, "p_value": float(2 * stats.t.sf(abs(t), df)), "ci_low": difference - reach, "ci_high": difference + reach}


# Compares the means of two separate groups. Welch's test doesn't assume the groups vary by the
# same amount, so it is the safer default; without it this is Student's test. Cohen's d divides
# the difference by the pooled standard deviation, and Hedges' g corrects it for small samples
def groups_row(name, groups, welch):
    (label_1, a), (label_2, b) = groups
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"Each group needs at least 2 values of {name}")
    var_1, var_2 = a.var(ddof=1), b.var(ddof=1)
    if welch:
        error = np.sqrt(var_1 / len(a) + var_2 / len(b))
        df = error ** 4 / ((var_1 / len(a)) ** 2 / (len(a) - 1) + (var_2 / len(b)) ** 2 / (len(b) - 1)) if error else 1
    else:
        error = common.pooled_sd(a, b) * np.sqrt(1 / len(a) + 1 / len(b))
        df = len(a) + len(b) - 2

    difference = a.mean() - b.mean()
    d = difference / common.pooled_sd(a, b) if common.pooled_sd(a, b) else np.nan
    row = {
        "variable": name, "group_1": label_1, "group_2": label_2, "n_1": len(a), "n_2": len(b),
        "mean_1": a.mean(), "mean_2": b.mean(), "sd_1": np.sqrt(var_1), "sd_2": np.sqrt(var_2),
        "mean_diff": difference,
    }
    row.update(t_result(difference, error, df))
    row["cohen_d"] = d
    row["hedges_g"] = d * (1 - 3 / (4 * (len(a) + len(b)) - 9))
    row["levene_p"] = float(stats.levene(a, b, center="median").pvalue)
    return row


# Compares two measures of the same people by testing whether their differences average zero.
# Cohen's d here is the mean difference over the standard deviation of the differences (d_z)
def paired_row(data, first, second):
    pairs = data[[first, second]].dropna().astype(float)
    if len(pairs) < 2:
        raise ValueError(f"Fewer than 2 rows have both {first} and {second}")
    differences = (pairs[first] - pairs[second]).to_numpy()
    spread = differences.std(ddof=1)
    row = {
        "variable": f"{first} - {second}", "n": len(pairs),
        "mean_1": pairs[first].mean(), "mean_2": pairs[second].mean(),
        "sd_1": pairs[first].std(), "sd_2": pairs[second].std(),
        "mean_diff": differences.mean(), "sd_diff": spread,
    }
    row.update(t_result(differences.mean(), spread / np.sqrt(len(pairs)), len(pairs) - 1))
    row["cohen_d"] = differences.mean() / spread
    row["correlation"] = pairs[first].corr(pairs[second])
    return row


# Compares the mean of a variable with a set value
def one_sample_row(data, name, value):
    values = data[name].dropna().to_numpy(float)
    if len(values) < 2:
        raise ValueError(f"{name} has fewer than 2 values")
    spread = values.std(ddof=1)
    difference = values.mean() - value
    row = {"variable": name, "n": len(values), "mean": values.mean(), "sd": spread, "test_value": value, "mean_diff": difference}
    row.update(t_result(difference, spread / np.sqrt(len(values)), len(values) - 1))
    row["cohen_d"] = difference / spread
    return row


# One test written the way a results section would
def describe(row, test):
    result = f"t({common.df_text(row['df'])}) = {row['t']:.2f}, {common.p_text(row['p_value'])}, d = {row['cohen_d']:.2f} ({common.d_size(row['cohen_d'])})"
    if test == "Groups":
        return (
            f"{row['variable']}: {row['group_1']} (M = {row['mean_1']:.2f}, SD = {row['sd_1']:.2f}) vs "
            f"{row['group_2']} (M = {row['mean_2']:.2f}, SD = {row['sd_2']:.2f}), {result}."
        )
    if test == "Paired":
        return f"{row['variable']}: mean difference = {row['mean_diff']:.2f}, {result}."
    return f"{row['variable']}: M = {row['mean']:.2f}, SD = {row['sd']:.2f} against {row['test_value']:g}, {result}."


# Runs the chosen t-test: on each Y between the two groups of G, on the difference between two
# measures, or on each Y against the test value
def run(inputs, settings):
    data = common.clean(inputs[0])
    test = test_of(settings)
    notes = []

    if test == "Paired":
        first, second = settings.get("first"), settings.get("second")
        if not first or not second:
            raise ValueError("Choose the first (1st) and second (2nd) measure")
        common.check_fields(data, [first, second], numeric=[first, second])
        rows = [paired_row(data, first, second)]
    else:
        variables = settings.get("y") or []
        if not variables:
            raise ValueError("Choose at least one variable (Y)")
        common.check_fields(data, variables, numeric=variables)

        if test == "1 sample":
            try:
                value = float(settings.get("value", 0) or 0)
            except ValueError:
                raise ValueError(f"The test value must be a number, not {settings.get('value')}")
            rows = [one_sample_row(data, name, value) for name in variables]
        else:
            group = settings.get("group")
            if not group:
                raise ValueError("Choose the column naming the two groups (G)")
            common.check_fields(data, [group])
            welch = settings.get("welch", True)
            rows = []
            for name in variables:
                groups = common.grouped(data, name, group)
                if len(groups) != 2:
                    raise ValueError(f"{group} must name exactly 2 groups, but has {len(groups)}. Use Filter to keep two, or ANOVA to compare them all")
                rows.append(groups_row(name, groups, welch))
            unequal = [row["variable"] for row in rows if row["levene_p"] < common.LEVEL]
            if unequal and not welch:
                notes.append(f"Levene's test says the groups vary by different amounts for {', '.join(unequal)}: turn Welch on.")
            notes.append("Welch's t-test (equal variances not assumed)." if welch else "Student's t-test (equal variances assumed).")

    results = significance.add_stars(pd.DataFrame(rows), after="t")
    described = [describe(row, test) for row in rows[:MAX_DESCRIBED]]
    results.attrs = {"note": " ".join(described + notes)}
    return [results]
