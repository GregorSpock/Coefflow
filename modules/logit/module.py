import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

from engine import clusters, dummies, fits, significance
from ui.clusters import add_cluster_button
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with a 0/1 Y column, numeric or text X and C columns (text ones enter as dummies), and optionally a field to cluster by"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with coefficient, std_error, z_stat, p_value, stars, odds_ratio, marginal_effect and marginal_p"),
    ("Fitted data", "the rows used, plus probability (that Y is 1) and predicted (0/1) columns"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "independent", "controls", "cluster"]
SPARE_ROWS = 10


# Opens a panel to give each variable a role: Y (a 0/1 outcome), X (independent) or C (control),
# with a button to switch heteroscedasticity-robust standard errors on and off and one to cluster them
def on_click(node):
    return open_panel(node, "Logit")


# Builds the panel for logit or probit, which share everything but the link function
def open_panel(node, title):
    panel = Panel(node, title)
    panel.add_button(robust_label(node.settings), toggle_robust)
    add_cluster_button(panel)
    panel.preview_title = "Results"
    panel.refresh = show_model
    show_model(panel)
    return panel


def robust_label(settings):
    return "Robust: On" if settings.get("robust") else "Robust: Off"


# Switches robust (HC1) standard errors on or off
def toggle_robust(panel, button):
    settings = panel.node.settings
    settings["robust"] = not settings.get("robust", False)
    button.text = robust_label(settings)
    show_model(panel)


# Returns "Y", "X", "C" or None for a variable
def role_of(settings, column):
    if settings.get("dependent") == column:
        return "Y"
    if column in settings.get("independent", []):
        return "X"
    if column in settings.get("controls", []):
        return "C"
    return None


# Takes a variable out of whatever role it has
def clear_role(settings, column):
    if settings.get("dependent") == column:
        settings["dependent"] = None
    settings["independent"] = [c for c in settings.get("independent", []) if c != column]
    settings["controls"] = [c for c in settings.get("controls", []) if c != column]


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
            settings.setdefault("independent", []).append(column)
        else:
            settings.setdefault("controls", []).append(column)

    show_model(panel)


# True if a column only holds 0 and 1 (ignoring missing values)
def is_binary(values):
    return set(values.dropna().unique()) <= {0, 1}


# Lists the variables from the last run with Y, X and C buttons, then the last results.
# Only 0/1 variables get a Y button, and text ones enter as dummies
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

    panel.lines = [
        f"Dependent (Y, 0/1): {settings.get('dependent') or 'none'}",
        f"Independent (X): {', '.join(settings.get('independent', [])) or 'none'}",
        f"Controls (C): {', '.join(settings.get('controls', [])) or 'none'}",
    ]

    data = node.last_inputs[0]
    for column in dummies.model_columns(data):
        if column == settings.get("cluster"):
            continue
        role = role_of(settings, column)
        binary = dummies.is_numeric(data[column]) and is_binary(data[column])
        choices = [("Y", choose_role, role == "Y")] if binary else []
        choices += [("X", choose_role, role == "X"), ("C", choose_role, role == "C")]
        panel.add_option(column, choices)

    if node.last_outputs:
        results = node.last_outputs[0]
        fit = results.attrs
        errors = clusters.errors_text(fit)
        panel.lines.append(
            f"McFadden R squared: {fit['pseudo_r_squared']:.3f}    Log likelihood: {fit['log_likelihood']:.1f}    "
            f"N: {fit['observations']} ({fit['ones']} ones)"
        )
        panel.lines.append(
            f"LR test p: {fit['lr_p']:.3g}    AIC: {fit['aic']:.1f}    "
            f"Correctly classified: {fit['correct']:.1%}    Standard errors: {errors}"
        )
        panel.note = " ".join(fit["warnings"])
        panel.preview = results
    else:
        panel.message = "Choose a 0/1 Y and the X variables, then run the workflow to fit the model"


# Checks the variables are chosen, exist and that Y is numeric and only holds 0 and 1
def check_columns(data, settings):
    dependent = settings.get("dependent")
    independent = settings.get("independent", [])
    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")
    if not independent:
        raise ValueError("Choose at least one independent variable (X)")

    variables = [dependent] + independent + settings.get("controls", [])
    missing = [c for c in variables if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")

    if not dummies.is_numeric(data[dependent]) or not is_binary(data[dependent]):
        raise ValueError(f"{dependent} must only hold 0 and 1 to be used as Y")


# Fits the model, collecting warnings from statsmodels as notes instead of printing them.
# Perfect separation means some X predicts Y exactly, so the estimates run off to infinity
def fit_model(model_class, y, x, cov_type, cov_kwds, notes):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = model_class(y, x).fit(disp=0, cov_type=cov_type, cov_kwds=cov_kwds, maxiter=200)
        except np.linalg.LinAlgError:
            raise ValueError("The model could not be fitted; an X or C variable may be a combination of the others")

    if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
        notes.append("Some variables predict Y perfectly (perfect separation), so the estimates are unreliable.")
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or not model.mle_retvals.get("converged", True):
        notes.append("The optimiser did not converge, so treat the estimates with caution.")
    if np.isnan(model.bse).any():
        notes.append("Some standard errors could not be computed, which usually means collinear variables.")
    return model


# Fits a binary choice model (logit or probit) of Y on the X and C variables with an intercept,
# with ordinary, robust (HC1) or clustered standard errors.
# Returns the coefficient table, with average marginal effects and fit statistics in its attrs,
# and the data rows used with each one's predicted probability and predicted class
def fit_binary(model_class, inputs, settings, odds_ratios):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    check_columns(data, settings)

    dependent = settings["dependent"]
    independent = settings.get("independent", [])
    controls = settings.get("controls", [])
    robust = settings.get("robust", False)
    notes = []

    used = data.dropna(subset=[dependent] + independent + controls + clusters.cluster_columns(data, settings))
    if len(used) < len(data):
        notes.append(f"Dropped {len(data) - len(used)} rows with missing values.")

    y = used[dependent].astype(float)
    terms, roles = dummies.design_matrix(used, [(independent, "X"), (controls, "C")])
    x = sm.add_constant(terms, has_constant="add")
    dummies.check_full_rank(x)
    if y.nunique() < 2:
        raise ValueError(f"{dependent} only takes one value in the rows used, so there is nothing to explain")
    if len(used) < x.shape[1] + SPARE_ROWS:
        raise ValueError(f"{len(used)} rows for {x.shape[1]} parameters. Use fewer variables or more rows")

    cov_type, cov_kwds, cluster = clusters.covariance(used, settings, x.shape[1])
    if cluster:
        notes += cluster["warnings"]
    model = fit_model(model_class, y, x, cov_type, cov_kwds, notes)

    # Average marginal effect of each variable on the probability that Y is 1
    effects = model.get_margeff(at="overall").summary_frame()
    marginal = [np.nan] + list(effects["dy/dx"].reindex(x.columns[1:]))
    marginal_p = [np.nan] + list(effects["Pr(>|z|)"].reindex(x.columns[1:]))

    roles = ["Intercept"] + roles
    results = pd.DataFrame({
        "role": roles,
        "variable": ["Intercept" if name == "const" else name for name in model.params.index],
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "z_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    significance.add_stars(results)
    if odds_ratios:
        results["odds_ratio"] = np.exp(model.params.values)
    results["marginal_effect"] = marginal
    results["marginal_p"] = marginal_p

    probability = model.predict(x)
    predicted = (probability >= 0.5).astype(int)
    results.attrs = {
        "pseudo_r_squared": model.prsquared,
        "log_likelihood": model.llf,
        "lr_p": model.llr_pvalue,
        "aic": model.aic,
        "observations": int(model.nobs),
        "ones": int(y.sum()),
        "correct": float((predicted == y).mean()),
        "robust": robust,
        "cluster": cluster,
        "warnings": notes,
    }
    kind = "logit" if odds_ratios else "probit"
    fits.record(results, kind, kind.capitalize(), dependent, used, independent + controls)

    fitted = used.copy()
    fitted["probability"] = probability
    fitted["predicted"] = predicted
    return [results, fitted]


def run(inputs, settings):
    return fit_binary(sm.Logit, inputs, settings, odds_ratios=True)
