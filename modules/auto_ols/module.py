from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from engine import clusters, dummies, fits, significance
from ui.clusters import add_cluster_button
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with a numeric column for Y, numeric or text columns for X, and optionally a field to cluster by. Other columns are candidate controls unless skipped. Text columns enter as dummies"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, coefficient, std_error, t_stat, p_value and stars for the best set of controls, with robust (HC1) or clustered errors"),
    ("Fitted data", "the rows used, plus fitted and residual columns"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "independent", "skipped", "cluster"]
MAX_CANDIDATES = 12


# Opens a panel to choose Y and the X variables. Every other variable is a candidate
# control unless it is skipped, and the run keeps whichever controls give the best adjusted R squared.
# The cluster field, if one is chosen, is never a candidate control
def on_click(node):
    panel = Panel(node, "Auto OLS")
    panel.add_button("Skip all", skip_all)
    panel.add_button("Skip none", skip_none)
    add_cluster_button(panel)
    panel.preview_title = "Best Model"
    panel.refresh = show_model
    show_model(panel)
    return panel


# Returns "Y", "X", "Skip" or None for a variable. None means it is a candidate control
def role_of(settings, column):
    if settings.get("dependent") == column:
        return "Y"
    if column in settings.get("independent", []):
        return "X"
    if column in settings.get("skipped", []):
        return "Skip"
    return None


# Takes a variable out of whatever role it has, making it a candidate control again
def clear_role(settings, column):
    if settings.get("dependent") == column:
        settings["dependent"] = None
    settings["independent"] = [c for c in settings.get("independent", []) if c != column]
    settings["skipped"] = [c for c in settings.get("skipped", []) if c != column]


# Gives a variable the clicked role, or clears it if it already had that role.
# There is only one Y, so choosing a new Y replaces the old one
def choose_role(panel, button):
    settings = panel.node.settings
    column = button.value
    role = button.text
    current = role_of(settings, column)

    clear_role(settings, column)
    if current != role:
        if role == "Y":
            settings["dependent"] = column
        elif role == "X":
            settings["independent"].append(column)
        else:
            settings["skipped"].append(column)

    show_model(panel)


# Skips every candidate control, so the few wanted ones can be switched back on
def skip_all(panel, button):
    settings = panel.node.settings
    if panel.node.last_inputs and panel.node.last_inputs[0] is not None:
        settings["skipped"] = settings.get("skipped", []) + candidates_for(panel.node.last_inputs[0], settings)
    show_model(panel)


def skip_none(panel, button):
    panel.node.settings["skipped"] = []
    show_model(panel)


# The variables that are neither Y, X, skipped nor the cluster field
def candidates_for(data, settings):
    columns = dummies.model_columns(data)
    return [c for c in columns if role_of(settings, c) is None and c != settings.get("cluster")]


# Lists the variables with Y, X and Skip buttons, then the best model from the last run.
# Text variables enter as dummies
def show_model(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    candidates = candidates_for(data, settings)
    panel.lines = [
        f"Dependent (Y): {settings.get('dependent') or 'none'}",
        f"Independent (X): {', '.join(settings.get('independent', [])) or 'none'}",
        f"Candidate controls: {len(candidates)}, so {2 ** len(candidates):,} models to try",
    ]

    for column in dummies.model_columns(data):
        if column == settings.get("cluster"):
            continue
        role = role_of(settings, column)
        panel.add_option(column, [
            ("Y", choose_role, role == "Y"),
            ("X", choose_role, role == "X"),
            ("Skip", choose_role, role == "Skip"),
        ])

    if node.last_outputs:
        results = node.last_outputs[0]
        fit = results.attrs
        panel.lines.append(
            f"Adjusted R squared: {fit['adj_r_squared']:.3f}    N: {fit['observations']}    "
            f"Models tried: {fit['models_tried']:,}"
        )
        panel.lines.append(
            f"Standard errors: {clusters.errors_text(fit)}    Breusch-Pagan p: {fit['breusch_pagan_p']:.3g}"
        )
        panel.note = (
            f"Kept controls: {', '.join(fit['kept']) or 'none'}. "
            f"Dropped: {', '.join(fit['dropped']) or 'none'}."
        )
        if fit.get("cluster"):
            panel.note += " " + " ".join(fit["cluster"]["warnings"])
        panel.preview = results
    else:
        panel.message = "Choose Y and X, then run the workflow to search the controls"


# Adjusted R squared from a least squares fit. x already includes the intercept column
def adjusted_r_squared(y, x):
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    deviations = y - y.mean()
    r_squared = 1 - (residuals @ residuals) / (deviations @ deviations)
    observations, parameters = x.shape
    return 1 - (1 - r_squared) * (observations - 1) / (observations - parameters)


# Fits Y on the X variables plus every combination of the candidates, smallest sets first.
# Returns the controls with the highest adjusted R squared and how many models were tried.
# A larger set only wins if it strictly beats a smaller one. A text candidate's dummies
# go in or out together
def best_controls(data, dependent, independent, candidates):
    y = data[dependent].to_numpy(float)
    intercept = np.ones((len(data), 1))
    base = np.hstack([intercept, dummies.design_matrix(data, [(independent, "X")])[0].to_numpy()])
    blocks = [dummies.terms(data[c], c).to_numpy() for c in candidates]
    best = []
    best_score = -np.inf
    tried = 0

    for size in range(len(candidates) + 1):
        for chosen in combinations(range(len(candidates)), size):
            x = np.hstack([base] + [blocks[i] for i in chosen])
            if x.shape[0] <= x.shape[1]:
                continue
            score = adjusted_r_squared(y, x)
            tried += 1
            if score > best_score:
                best = [candidates[i] for i in chosen]
                best_score = score

    return best, tried


# Searches every combination of candidate controls, keeps the set with the highest adjusted
# R squared, and refits that model with robust (HC1) standard errors, or clustered ones if a
# cluster field is chosen. All models use the same rows (those complete for Y, X, every candidate
# and the cluster field) so their adjusted R squared can be compared.
# Returns the coefficient table and those rows with each one's fitted value and residual
def run(inputs, settings):
    data = inputs[0]
    dependent = settings.get("dependent")
    independent = settings.get("independent", [])

    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")
    if not independent:
        raise ValueError("Choose at least one independent variable (X)")

    missing = [c for c in [dependent] + independent if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    if not dummies.is_numeric(data[dependent]):
        raise ValueError(f"{dependent} is not numeric, so it cannot be the dependent variable (Y)")

    candidates = candidates_for(data, settings)
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(
            f"{len(candidates)} candidate controls means {2 ** len(candidates):,} models. "
            f"Skip some so there are at most {MAX_CANDIDATES}"
        )

    sample = data[[dependent] + independent + candidates + clusters.cluster_columns(data, settings)].dropna()
    kept, tried = best_controls(sample, dependent, independent, candidates)
    if tried == 0:
        raise ValueError("Not enough complete rows to fit the model")

    terms, roles = dummies.design_matrix(sample, [(independent, "X"), (kept, "C")])
    x = sm.add_constant(terms, has_constant="add")
    dummies.check_full_rank(x)
    cov_type, cov_kwds, cluster = clusters.covariance(sample, dict(settings, robust=True), x.shape[1])
    model = sm.OLS(sample[dependent], x).fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=True)
    breusch_pagan_p = het_breuschpagan(model.resid, model.model.exog)[1]

    roles = ["Intercept"] + roles
    results = pd.DataFrame({
        "role": roles,
        "variable": ["Intercept" if name == "const" else name for name in model.params.index],
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    significance.add_stars(results)
    results.attrs = {
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "observations": int(model.nobs),
        "robust": True,
        "cluster": cluster,
        "breusch_pagan_p": breusch_pagan_p,
        "models_tried": tried,
        "kept": kept,
        "dropped": [c for c in candidates if c not in kept],
    }
    fits.record_linear(results, "Auto OLS", model, dependent, sample, independent + kept)

    fitted = data.loc[sample.index].copy()
    fitted["fitted"] = model.fittedvalues
    fitted["residual"] = model.resid
    return [results, fitted]
