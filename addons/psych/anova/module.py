import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with a numeric outcome (Y), one to three columns naming groups (F) and, for an ANCOVA, numeric covariates (C)"),
]
OUTPUT_SPECS = [
    ("ANOVA", "one row per effect with its sum of squares, df, F, p_value, eta squared, partial eta squared and omega squared"),
    ("Means", "the count, mean, standard deviation and confidence interval of Y in each group or combination of groups"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "factors", "covariates"]
ROLES = [("Y", "dependent", False, True), ("F", "factors", True, False), ("C", "covariates", True, True)]
SS_TYPES = ["Type 3", "Type 2"]
MAX_FACTORS = 3


# Opens a panel to choose the outcome, factors and covariates, showing the ANOVA table of the last run
def on_click(node):
    panel = common.open_panel(node, "ANOVA", show_anova, "ANOVA")
    common.add_switch(panel, "Interact", "interact")
    common.add_choice(panel, "SS", "ss", SS_TYPES)
    common.add_choice(panel, "Show", "show", ["ANOVA", "Means"])
    show_anova(panel)
    return panel


# Lists the fields with Y, F and C buttons, then the ANOVA table or the group means of the last run
def show_anova(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [
        f"Outcome (Y): {common.names(settings, 'dependent')}    Factors (F): {common.names(settings, 'factors')}    "
        f"Covariates (C): {common.names(settings, 'covariates')}",
    ]
    means = settings.get("show") == "Means"
    panel.preview_title = "Group Means" if means else "ANOVA"
    common.show_results(panel, "Choose an outcome (Y) and at least one factor (F), then run the workflow", output=1 if means else 0)


# Checks the roles are filled with fields that are in the data
def check_columns(data, settings):
    dependent = settings.get("dependent")
    factors = settings.get("factors") or []
    covariates = settings.get("covariates") or []
    if not dependent:
        raise ValueError("Choose a numeric outcome (Y)")
    if not factors:
        raise ValueError("Choose at least one factor (F), a column naming the groups to compare")
    if len(factors) > MAX_FACTORS:
        raise ValueError(f"At most {MAX_FACTORS} factors can be crossed")
    common.check_fields(data, [dependent] + factors + covariates, numeric=[dependent] + covariates)
    return dependent, factors, covariates


# The rows with every chosen field present, under plain names (y, f0, f1.., c0, c1..) so that
# field names with spaces or symbols can't upset the model formula. Factors become text, so
# numbered groups such as 1, 2, 3 are treated as groups rather than as amounts
def model_frame(data, dependent, factors, covariates):
    rows = data[[dependent] + factors + covariates].dropna()
    frame = pd.DataFrame({"y": rows[dependent].astype(float)})
    for i, factor in enumerate(factors):
        frame[f"f{i}"] = rows[factor].map(common.group_text)
        levels = frame[f"f{i}"].nunique()
        if levels < 2:
            raise ValueError(f"{factor} has only one group in the rows used")
        if levels > common.MAX_GROUPS:
            raise ValueError(f"{factor} has {levels} groups, which is too many. Is it the right field?")
    for i, covariate in enumerate(covariates):
        frame[f"c{i}"] = rows[covariate].astype(float)
    return frame


# The name of an effect as the model writes it, e.g. "C(f0, Sum):C(f1, Sum)", in the fields' own names
def effect_name(term, factors, covariates):
    known = {f"C(f{i}, Sum)": name for i, name in enumerate(factors)}
    known.update({f"c{i}": name for i, name in enumerate(covariates)})
    return " * ".join(known.get(part, part) for part in term.split(":"))


# Fits the model and returns its ANOVA table. Factors use sum-to-zero coding, which Type 3 sums
# of squares need to test each main effect averaged over the others, as SPSS does. Type 2 tests
# each effect after the others of its own order, and is the better choice without an interaction
def anova_table(frame, factors, covariates, interact, ss_type):
    coded = [f"C(f{i}, Sum)" for i in range(len(factors))]
    formula = "y ~ " + (" * " if interact else " + ").join(coded)
    formula += "".join(f" + c{i}" for i in range(len(covariates)))
    model = smf.ols(formula, frame).fit()
    if model.df_resid < 1:
        raise ValueError("There are too few rows for this many groups")
    if np.linalg.matrix_rank(model.model.exog) < model.model.exog.shape[1]:
        raise ValueError("Some combinations of the factors have no rows, so the model can't be fitted. Turn Interact off or drop a factor")

    table = anova_lm(model, typ=3 if ss_type == "Type 3" else 2).drop(index="Intercept", errors="ignore")
    total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
    residual = table.loc["Residual"]
    mean_square_error = residual["sum_sq"] / residual["df"]

    rows = []
    for term, row in table.iterrows():
        if term == "Residual":
            continue
        rows.append({
            "effect": effect_name(term, factors, covariates),
            "sum_sq": row["sum_sq"],
            "df": row["df"],
            "mean_sq": row["sum_sq"] / row["df"],
            "f": row["F"],
            "p_value": row["PR(>F)"],
            "eta_sq": row["sum_sq"] / total,
            "partial_eta_sq": row["sum_sq"] / (row["sum_sq"] + residual["sum_sq"]),
            "omega_sq": (row["sum_sq"] - row["df"] * mean_square_error) / (total + mean_square_error),
        })
    rows.append({"effect": "Residual", "sum_sq": residual["sum_sq"], "df": residual["df"], "mean_sq": mean_square_error})
    rows.append({"effect": "Total", "sum_sq": total, "df": len(frame) - 1})
    return significance.add_stars(pd.DataFrame(rows), after="f")


# The count, mean, spread and 95% confidence interval of Y in each combination of the factors' groups
def group_means(frame, factors):
    keys = [f"f{i}" for i in range(len(factors))]
    rows = []
    for labels, group in frame.groupby(keys, sort=True):
        values = group["y"].to_numpy()
        mean, low, high = common.mean_interval(values)
        row = dict(zip(factors, labels))
        row.update({
            "n": len(values), "mean": mean, "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
            "se": values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan,
            "ci_low": low, "ci_high": high,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# Levene's test of whether the groups vary by the same amount, as a sentence. Groups of one
# value can't be tested, so they are skipped
def levene_text(frame, factors):
    keys = [f"f{i}" for i in range(len(factors))]
    cells = [group["y"].to_numpy() for labels, group in frame.groupby(keys) if len(group) > 1]
    if len(cells) < 2:
        return ""
    result = stats.levene(*cells, center="median")
    text = f"Levene's test: F({len(cells) - 1}, {sum(len(c) for c in cells) - len(cells)}) = {result.statistic:.2f}, {common.p_text(result.pvalue)}"
    if result.pvalue < common.LEVEL:
        return text + ", so the groups vary by different amounts; check the result with Games-Howell in Post Hoc."
    return text + ", so the groups' variances look equal."


# Fits a between-subjects ANOVA of Y on the factors, with their interactions if Interact is on,
# and any covariates (an ANCOVA)
def run(inputs, settings):
    data = common.clean(inputs[0])
    dependent, factors, covariates = check_columns(data, settings)
    frame = model_frame(data, dependent, factors, covariates)
    interact = settings.get("interact", True) and len(factors) > 1
    ss_type = settings.get("ss", SS_TYPES[0])

    table = anova_table(frame, factors, covariates, interact, ss_type)
    residual_df = table.loc[table["effect"] == "Residual", "df"].iloc[0]
    notes = []
    for index, row in table.iterrows():
        if row["effect"] in ("Residual", "Total"):
            continue
        notes.append(
            f"{row['effect']}: F({common.df_text(row['df'])}, {common.df_text(residual_df)}) = {row['f']:.2f}, "
            f"{common.p_text(row['p_value'])}, partial eta sq = {common.short(row['partial_eta_sq'])}."
        )
    notes.append(levene_text(frame, factors))
    if covariates:
        notes.append("The group means are not adjusted for the covariates.")
    dropped = len(data) - len(frame)
    if dropped:
        notes.append(f"Left out {dropped} rows with missing values.")
    notes.append(f"{ss_type} sums of squares.")

    table.attrs = {"note": " ".join(note for note in notes if note)}
    return [table, group_means(frame, factors)]
