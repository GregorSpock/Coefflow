import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with a numeric predictor (X), mediator (M) and outcome (Y), and optionally numeric covariates (C)"),
]
OUTPUT_SPECS = [
    ("Paths", "one row for each path (a, b, the total effect c and the direct effect c') and for the indirect effect a x b, with its estimate, standard error, p_value and confidence interval"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["predictor", "mediator", "outcome", "covariates"]
ROLES = [
    ("X", "predictor", False, True), ("M", "mediator", False, True), ("Y", "outcome", False, True),
    ("C", "covariates", True, True),
]
# The indirect effect's interval comes from refitting the model on this many resamples of the
# rows. The seed keeps the interval the same every run
RESAMPLES = 5000
SEED = 1
# How many numbers a batch of resamples may hold at once, to keep memory use modest
BATCH_VALUES = 2_000_000


# Opens a panel to choose the predictor, mediator, outcome and covariates, showing the paths of the last run
def on_click(node):
    panel = common.open_panel(node, "Mediation", show_paths, "Paths")
    show_paths(panel)
    return panel


def show_paths(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [
        f"Predictor (X): {common.names(settings, 'predictor')}    Mediator (M): {common.names(settings, 'mediator')}    "
        f"Outcome (Y): {common.names(settings, 'outcome')}",
        f"Covariates (C): {common.names(settings, 'covariates')}",
    ]
    common.show_results(panel, "Choose a predictor (X), a mediator (M) and an outcome (Y), then run the workflow")


# One row of the results from a fitted regression's coefficient
def path_row(label, fit, position):
    low, high = fit.conf_int()[position]
    return {
        "path": label, "estimate": fit.params[position], "se": fit.bse[position], "statistic": fit.tvalues[position],
        "p_value": fit.pvalues[position], "ci_low": low, "ci_high": high,
    }


# The indirect effect a x b in each of many resamples of the rows, drawn with replacement.
# Each resample's two regressions are solved together, a batch of resamples at a time
def resampled_indirect(x, m, y, covariates):
    count = len(x)
    ones = np.ones(count)
    mediator_design = np.column_stack([ones, x, covariates])
    outcome_design = np.column_stack([ones, x, m, covariates])
    rng = np.random.default_rng(SEED)
    batch = max(1, BATCH_VALUES // (count * outcome_design.shape[1]))

    # Solves a regression for every resample in the batch at once, returning each one's coefficients
    def solve(design, target):
        cross = np.einsum("rni,rnj->rij", design, design)
        moment = np.einsum("rni,rn->ri", design, target)
        return (np.linalg.pinv(cross) @ moment[..., None])[..., 0]

    effects = []
    for start in range(0, RESAMPLES, batch):
        rows = rng.integers(0, count, size=(min(batch, RESAMPLES - start), count))
        a = solve(mediator_design[rows], m[rows])[:, 1]
        b = solve(outcome_design[rows], y[rows])[:, 2]
        effects.append(a * b)
    return np.concatenate(effects)


# Tests whether X affects Y through M. Path a is X's effect on M, and b is M's effect on Y with X
# held constant. Their product is the indirect effect, the part of X's total effect (c) that runs
# through M, leaving the direct effect (c'). The indirect effect's interval is a percentile
# bootstrap, as its distribution isn't normal, and its p value is Sobel's test
def run(inputs, settings):
    data = common.clean(inputs[0])
    predictor, mediator, outcome = settings.get("predictor"), settings.get("mediator"), settings.get("outcome")
    covariates = settings.get("covariates") or []
    if not predictor or not mediator or not outcome:
        raise ValueError("Choose a predictor (X), a mediator (M) and an outcome (Y)")
    fields = [predictor, mediator, outcome] + covariates
    common.check_fields(data, fields, numeric=fields)

    rows = data[fields].dropna().astype(float)
    if len(rows) < len(fields) + 3:
        raise ValueError(f"Only {len(rows)} rows have every variable, which is too few")
    still = [name for name in fields if rows[name].std() == 0]
    if still:
        raise ValueError(f"These variables don't vary: {', '.join(still)}")
    x, m, y = (rows[name].to_numpy() for name in [predictor, mediator, outcome])
    held = rows[covariates].to_numpy()
    ones = np.ones(len(rows))

    mediator_fit = sm.OLS(m, np.column_stack([ones, x, held])).fit()
    outcome_fit = sm.OLS(y, np.column_stack([ones, x, m, held])).fit()
    total_fit = sm.OLS(y, np.column_stack([ones, x, held])).fit()
    a, b = mediator_fit.params[1], outcome_fit.params[2]
    total, direct = total_fit.params[1], outcome_fit.params[1]

    indirect = a * b
    effects = resampled_indirect(x, m, y, held)
    low, high = np.percentile(effects, [100 * common.LEVEL / 2, 100 * (1 - common.LEVEL / 2)])
    sobel_error = np.sqrt(a ** 2 * outcome_fit.bse[2] ** 2 + b ** 2 * mediator_fit.bse[1] ** 2)
    sobel = indirect / sobel_error

    paths = [
        path_row(f"a: {predictor} -> {mediator}", mediator_fit, 1),
        path_row(f"b: {mediator} -> {outcome}", outcome_fit, 2),
        path_row(f"c (total): {predictor} -> {outcome}", total_fit, 1),
        path_row(f"c' (direct): {predictor} -> {outcome}", outcome_fit, 1),
        {
            "path": "indirect (a x b)", "estimate": indirect, "se": effects.std(ddof=1), "statistic": sobel,
            "p_value": float(2 * stats.norm.sf(abs(sobel))), "ci_low": low, "ci_high": high,
        },
    ]
    table = significance.add_stars(pd.DataFrame(paths), after="estimate")

    mediated = low > 0 or high < 0
    notes = [
        f"Indirect effect = {indirect:.3f}, 95% bootstrap CI [{low:.3f}, {high:.3f}]: "
        + (f"the interval excludes zero, so {mediator} mediates the effect of {predictor} on {outcome}." if mediated
           else f"the interval includes zero, so there is no significant evidence that {mediator} mediates the effect.")
    ]
    if mediated and total != 0 and np.sign(indirect) == np.sign(total):
        kind = "partial" if outcome_fit.pvalues[1] < common.LEVEL else "full"
        notes.append(f"It carries {100 * indirect / total:.0f}% of the total effect ({kind} mediation: the direct effect is {'still' if kind == 'partial' else 'no longer'} significant).")
    notes.append(f"The interval is from {RESAMPLES} resamples; the indirect effect's p value is Sobel's test, which is less reliable than the interval.")
    notes.append("Mediation shows a pattern in the data, not that X causes M or M causes Y.")
    if len(rows) < len(data):
        notes.append(f"Left out {len(data) - len(rows)} rows with missing values.")

    table.attrs = {"note": " ".join(notes)}
    return [table]
