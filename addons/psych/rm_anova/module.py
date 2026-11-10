import numpy as np
import pandas as pd
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with one row per person and a numeric column for each condition or time point (W)"),
]
OUTPUT_SPECS = [
    ("ANOVA", "the effect of the conditions with F, df and p_value as they are and with the Greenhouse-Geisser and Huynh-Feldt corrections, and its effect sizes"),
    ("Means", "the count, mean, standard deviation and confidence interval of each condition"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["within"]
ROLES = [("W", "within", True, True)]
# Above this Greenhouse-Geisser epsilon, Huynh-Feldt is the correction usually recommended
HUYNH_FELDT_FROM = 0.75


# Opens a panel to choose the columns holding each condition, showing the ANOVA of the last run
def on_click(node):
    panel = common.open_panel(node, "RM ANOVA", show_anova, "ANOVA")
    common.add_choice(panel, "Show", "show", ["ANOVA", "Means"])
    show_anova(panel)
    return panel


def show_anova(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [f"Conditions (W), in order: {common.names(settings, 'within')}"]
    means = settings.get("show") == "Means"
    panel.preview_title = "Condition Means" if means else "ANOVA"
    common.show_results(panel, "Click W on the column of each condition, in order, then run the workflow", output=1 if means else 0)


# Sphericity means the differences between every pair of conditions vary by the same amount.
# Returns Mauchly's W with its chi-square, df and p value, and the Greenhouse-Geisser and
# Huynh-Feldt epsilons, which shrink the degrees of freedom when sphericity fails (1 is none).
# All come from the conditions' covariances taken through orthonormal contrasts
def sphericity(scores):
    people, conditions = scores.shape
    d = conditions - 1
    # The first column of the Q of a QR decomposition of a matrix led by ones is constant, so the
    # rest are orthonormal contrasts between the conditions
    q = np.linalg.qr(np.column_stack([np.ones(conditions), np.eye(conditions)[:, :d]]))[0]
    contrasts = q[:, 1:]
    covariance = contrasts.T @ np.cov(scores, rowvar=False) @ contrasts

    greenhouse = np.trace(covariance) ** 2 / (d * np.sum(covariance ** 2))
    huynh = (people * d * greenhouse - 2) / (d * (people - 1 - d * greenhouse))
    result = {"greenhouse_geisser": min(1.0, greenhouse), "huynh_feldt": min(1.0, max(greenhouse, huynh))}

    if conditions > 2:
        w = np.linalg.det(covariance) / (np.trace(covariance) / d) ** d
        shrink = 1 - (2 * d ** 2 + d + 2) / (6 * d * (people - 1))
        df = conditions * (conditions - 1) / 2 - 1
        chi_square = -(people - 1) * shrink * np.log(w) if w > 0 else np.inf
        # The p value has a second term that matters in small samples, as SPSS and R include
        extra = (d + 2) * (d - 1) * (d - 2) * (2 * d ** 3 + 6 * d ** 2 + 3 * d + 2) / (288 * ((people - 1) * d * shrink) ** 2)
        p_value = stats.chi2.sf(chi_square, df)
        p_value += extra * (stats.chi2.sf(chi_square, df + 4) - p_value)
        result.update({"w": w, "chi_square": chi_square, "df": df, "p_value": float(p_value)})
    return result


# A one-way repeated measures ANOVA on a table of people by conditions. The variation between
# people is taken out, so the conditions are tested against what is left over.
# Returns the F test's parts and the sums of squares
def repeated_anova(scores):
    people, conditions = scores.shape
    grand = scores.mean()
    ss_conditions = people * np.sum((scores.mean(axis=0) - grand) ** 2)
    ss_people = conditions * np.sum((scores.mean(axis=1) - grand) ** 2)
    ss_error = np.sum((scores - grand) ** 2) - ss_conditions - ss_people
    df_1, df_2 = conditions - 1, (conditions - 1) * (people - 1)
    if ss_error <= 0:
        raise ValueError("Every person changes by exactly the same amount, so there is no error to test against")
    return {
        "f": (ss_conditions / df_1) / (ss_error / df_2), "df_1": df_1, "df_2": df_2,
        "ss_conditions": ss_conditions, "ss_people": ss_people, "ss_error": ss_error,
    }


# Degrees of freedom as text: whole numbers plainly, corrected ones to two decimal places
def df_text(df):
    return f"{df:.0f}" if abs(df - round(df)) < 1e-9 else f"{df:.2f}"


# Tests whether the mean differs between conditions measured on the same people
def run(inputs, settings):
    data = common.clean(inputs[0])
    within = settings.get("within") or []
    if len(within) < 2:
        raise ValueError("Choose at least 2 conditions (W), one column each")
    common.check_fields(data, within, numeric=within)

    rows = data[within].dropna().astype(float)
    scores = rows.to_numpy()
    people, conditions = scores.shape
    if people <= conditions:
        raise ValueError(f"Only {people} rows have every condition, which is too few for {conditions} conditions")

    anova = repeated_anova(scores)
    sphere = sphericity(scores)
    partial = anova["ss_conditions"] / (anova["ss_conditions"] + anova["ss_error"])
    general = anova["ss_conditions"] / (anova["ss_conditions"] + anova["ss_people"] + anova["ss_error"])

    table = []
    for label, epsilon in [("Sphericity assumed", 1.0), ("Greenhouse-Geisser", sphere["greenhouse_geisser"]), ("Huynh-Feldt", sphere["huynh_feldt"])]:
        df_1, df_2 = anova["df_1"] * epsilon, anova["df_2"] * epsilon
        table.append({
            "correction": label, "epsilon": epsilon, "f": anova["f"], "df_1": df_1, "df_2": df_2,
            "p_value": float(stats.f.sf(anova["f"], df_1, df_2)),
            "partial_eta_sq": partial, "general_eta_sq": general,
        })
    table = significance.add_stars(pd.DataFrame(table), after="f")

    # The row to report: uncorrected if sphericity holds, else the correction that suits its epsilon
    use = 0
    if conditions == 2:
        notes = ["With two conditions sphericity always holds, and this is the same test as a paired t-test."]
    else:
        notes = [f"Mauchly's test: W = {common.short(sphere['w'], 3)}, chi-sq({sphere['df']:.0f}) = {sphere['chi_square']:.2f}, {common.p_text(sphere['p_value'])}"]
        if sphere["p_value"] < common.LEVEL:
            use = 1 if sphere["greenhouse_geisser"] < HUYNH_FELDT_FROM else 2
            notes[0] += f", so sphericity is violated: report the {table['correction'][use]} row."
        else:
            notes[0] += ", so sphericity holds: report the first row."
    row = table.iloc[use]
    notes.insert(0, (
        f"Conditions: F({df_text(row['df_1'])}, {df_text(row['df_2'])}) = {row['f']:.2f}, "
        f"{common.p_text(row['p_value'])}, partial eta sq = {common.short(partial)}."
    ))
    dropped = len(data) - people
    if dropped:
        notes.append(f"Left out {dropped} rows missing a condition.")

    means = []
    for name in within:
        mean, low, high = common.mean_interval(rows[name].to_numpy())
        means.append({"condition": name, "n": people, "mean": mean, "sd": rows[name].std(), "se": rows[name].sem(), "ci_low": low, "ci_high": high})

    table.attrs = {"note": " ".join(notes)}
    return [table, pd.DataFrame(means)]
