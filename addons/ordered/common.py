import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from engine import clusters, dummies, fits, significance
from modules.logit.module import clear_role, role_of
from ui.clusters import add_cluster_button
from ui.panel import Panel


# Shared by the Ordered pack's modules: the panel where variables get roles, the checks on an
# ordered Y, and the fit of an ordered logit or probit. An ordered Y is a numeric field with a
# handful of values whose order means something but whose gaps don't, e.g. a 1-5 rating.
# Its categories are its distinct values in increasing order

# Fields with more values than this are treated as continuous, so they get no Y button.
# Whole numbers, such as the codes Convert gives, can have more, as they look like categories
MAX_CATEGORIES = 12
MAX_CODES = 30
SPARE_ROWS = 10
FIELD_SETTINGS = ["dependent", "independent", "controls", "cluster"]

# The distribution behind each link: P(Y <= category j) = F(cut j - Xb)
# Shown in the panels, since a field with too many values gets no Y button
Y_HINT = f"Y needs a numeric field with at most {MAX_CODES} whole number values ({MAX_CATEGORIES} if not whole numbers)"

DISTRIBUTIONS = {"logit": stats.logistic, "probit": stats.norm}


# The most values an ordered Y can have: MAX_CODES if they are all whole numbers, else MAX_CATEGORIES
def category_limit(values):
    values = values.dropna()
    return MAX_CODES if np.allclose(values, values.round()) else MAX_CATEGORIES


# True if a field could be an ordered Y: numeric with between 2 and category_limit values
def is_ordered(values):
    return dummies.is_numeric(values) and 2 <= values.nunique() <= category_limit(values)


# Short text for a category value, so 3.0 shows as 3
def category_text(value):
    return f"{value:g}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)


# Builds a panel with Y, X and C roles. show(panel) lists the variables and the module's results,
# and is called again whenever a role or setting changes. Robust and Cluster buttons are added
# for the modules whose standard errors can use them
def open_panel(node, title, show, robust=False):
    panel = Panel(node, title)
    if robust:
        panel.add_button(robust_label(node.settings), toggle_robust)
        add_cluster_button(panel)
    panel.preview_title = "Results"
    panel.refresh = show
    show(panel)
    return panel


def robust_label(settings):
    return "Robust: On" if settings.get("robust") else "Robust: Off"


# Switches robust (HC1) standard errors on or off
def toggle_robust(panel, button):
    settings = panel.node.settings
    settings["robust"] = not settings.get("robust", False)
    button.text = robust_label(settings)
    panel.refresh(panel)


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

    panel.refresh(panel)


# Lists the variables from the last run with Y, X and C buttons and the chosen roles in the info
# lines. Only ordered fields get a Y button. Without controls there is no C button, and without
# text_x only numeric fields can be X. Returns False if the workflow hasn't run yet
def show_roles(panel, controls=True, text_x=True):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""
    panel.message = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        return False

    panel.lines = [
        f"Dependent (Y, ordered): {settings.get('dependent') or 'none'}",
        f"Independent (X): {', '.join(settings.get('independent', [])) or 'none'}",
    ]
    if controls:
        panel.lines.append(f"Controls (C): {', '.join(settings.get('controls', [])) or 'none'}")

    data = node.last_inputs[0]
    for column in dummies.model_columns(data):
        if column == settings.get("cluster"):
            continue
        numeric = dummies.is_numeric(data[column])
        if not numeric and not text_x:
            continue
        role = role_of(settings, column)
        choices = [("Y", choose_role, role == "Y")] if numeric and is_ordered(data[column]) else []
        choices.append(("X", choose_role, role == "X"))
        if controls:
            choices.append(("C", choose_role, role == "C"))
        panel.add_option(column, choices)
    return True


# Checks the variables are chosen and exist, and that Y is numeric with few enough values to be ordered
def check_columns(data, settings):
    dependent = settings.get("dependent")
    independent = settings.get("independent", [])
    if not dependent:
        raise ValueError("Choose an ordered dependent variable (Y)")
    if not independent:
        raise ValueError("Choose at least one independent variable (X)")

    variables = [dependent] + independent + settings.get("controls", [])
    missing = [c for c in variables if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")

    values = data[dependent]
    if not dummies.is_numeric(values):
        raise ValueError(f"{dependent} must be numeric to be used as Y, with higher values for higher categories")
    limit = category_limit(values)
    if values.nunique() > limit:
        kind = "whole numbers" if limit == MAX_CODES else "values that aren't all whole numbers"
        raise ValueError(f"{dependent} has {values.nunique()} values; an ordered Y with {kind} can have at most {limit}")


# The rows a model uses, its Y, its X and C terms (text fields as dummies, no intercept, since the
# cut points take its place) and each term's role. Rows missing a variable are dropped with a note
def prepare(inputs, settings, notes, min_categories=2, use_clusters=False):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    check_columns(data, settings)

    dependent = settings["dependent"]
    independent = settings.get("independent", [])
    controls = settings.get("controls", [])
    extra = clusters.cluster_columns(data, settings) if use_clusters else []

    used = data.dropna(subset=[dependent] + independent + controls + extra)
    if len(used) < len(data):
        notes.append(f"Dropped {len(data) - len(used)} rows with missing values.")

    y = used[dependent].astype(float)
    categories = np.sort(y.unique())
    if len(categories) < min_categories:
        raise ValueError(f"{dependent} takes {len(categories)} values in the rows used; this needs at least {min_categories}")

    x, roles = dummies.design_matrix(used, [(independent, "X"), (controls, "C")])
    constant = [c for c in x.columns if x[c].nunique() < 2]
    if constant:
        raise ValueError(f"{', '.join(constant)} takes one value in the rows used, so it adds nothing to the model")
    parameters = x.shape[1] + len(categories) - 1
    if len(used) < parameters + SPARE_ROWS:
        raise ValueError(f"{len(used)} rows for {parameters} parameters. Use fewer variables or more rows")

    small = [category_text(c) for c in categories if (y == c).sum() < 5]
    if small:
        notes.append(f"Categories {', '.join(small)} of {dependent} have fewer than 5 rows, so their cut points are imprecise.")
    return used, y, x, roles, categories


# Fits an ordered logit or probit of y on x, collecting warnings from statsmodels as notes
def fit(y, x, link, notes, cov_type="nonrobust", cov_kwds=None):
    options = {"cov_type": cov_type}
    if cov_kwds:
        options["cov_kwds"] = cov_kwds
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = OrderedModel(y, x, distr=link).fit(method="bfgs", disp=0, maxiter=2000, **options)
        except (np.linalg.LinAlgError, ValueError):
            raise ValueError("The model could not be fitted; an X or C variable may be a combination of the others")

    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or not model.mle_retvals.get("converged", True):
        notes.append("The optimiser did not converge, so treat the estimates with caution.")
    if np.isnan(model.bse).any():
        notes.append("Some standard errors could not be computed, which usually means collinear variables.")
    return model


# statsmodels keeps the first cut point and then the log of each gap to the next, so the cuts
# always rise. Turns those into the cut points themselves
def cuts_from(thresholds):
    return np.cumsum(np.concatenate([thresholds[:1], np.exp(thresholds[1:])]))


# The cut points and their standard errors, by the delta method from the fitted parameters
def cut_points(model, terms):
    thresholds = np.asarray(model.params)[terms:]
    jacobian = np.tril(np.ones((len(thresholds), len(thresholds))))
    jacobian[:, 1:] *= np.exp(thresholds[1:])
    covariance = np.asarray(model.cov_params())[terms:, terms:]
    errors = np.sqrt(np.diag(jacobian @ covariance @ jacobian.T))
    return cuts_from(thresholds), errors


# The average effect of each term on the probability of each category, given a parameter vector.
# P(Y = j) = F(cut j - Xb) - F(cut j-1 - Xb), so its derivative by a term is the term's coefficient
# times f(cut j-1 - Xb) - f(cut j - Xb). Returns a (terms x categories) array
def average_effects(params, x, link):
    terms = x.shape[1]
    beta = params[:terms]
    edges = np.concatenate([[-np.inf], cuts_from(params[terms:]), [np.inf]])
    density = DISTRIBUTIONS[link].pdf(edges[None, :] - (x @ beta)[:, None])
    slopes = (density[:, :-1] - density[:, 1:]).mean(axis=0)
    return np.outer(beta, slopes)


# Average marginal effects on each category's probability, with delta method standard errors
# from a numerical derivative of the effects by the parameters
def marginal_effects(model, x, roles, categories, link):
    params = np.asarray(model.params, dtype=float)
    values = x.to_numpy(dtype=float)
    effects = average_effects(params, values, link)

    jacobian = np.empty((effects.size, len(params)))
    for i in range(len(params)):
        step = 1e-6 * max(1.0, abs(params[i]))
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        jacobian[:, i] = (average_effects(up, values, link) - average_effects(down, values, link)).ravel() / (2 * step)
    errors = np.sqrt(np.diag(jacobian @ np.asarray(model.cov_params()) @ jacobian.T)).reshape(effects.shape)

    rows = []
    for t, term in enumerate(x.columns):
        for c, category in enumerate(categories):
            z = effects[t, c] / errors[t, c] if errors[t, c] > 0 else np.nan
            rows.append({
                "role": roles[t],
                "variable": term,
                "category": category,
                "effect": effects[t, c],
                "std_error": errors[t, c],
                "z_stat": z,
                "p_value": 2 * stats.norm.sf(abs(z)),
            })
    return significance.add_stars(pd.DataFrame(rows), after="effect")


# The probability of each category for each row, as columns named like prob_3
def probability_columns(model, x, categories):
    probabilities = np.asarray(model.predict(x))
    return pd.DataFrame(
        probabilities, index=x.index, columns=[f"prob_{category_text(c)}" for c in categories]
    )


# Fits an ordered logit or probit of Y on the X and C variables, with ordinary, robust (HC1) or
# clustered standard errors. Returns the coefficient table (terms then cut points, with fit
# statistics in its attrs), the rows used with each category's probability and the predicted
# category, and the average marginal effects on each category's probability
def fit_ordered(inputs, settings, link):
    notes = []
    used, y, x, roles, categories = prepare(inputs, settings, notes, use_clusters=True)

    terms = x.shape[1]
    cov_type, cov_kwds, cluster = clusters.covariance(used, settings, terms + len(categories) - 1)
    if cluster:
        notes += cluster["warnings"]
    model = fit(y, x, link, notes, cov_type, cov_kwds)

    cuts, cut_errors = cut_points(model, terms)
    coefficients = np.asarray(model.params)[:terms]
    errors = np.asarray(model.bse)[:terms]
    estimates = np.concatenate([coefficients, cuts])
    std_errors = np.concatenate([errors, cut_errors])
    z = estimates / std_errors

    results = pd.DataFrame({
        "role": roles + ["Cut"] * len(cuts),
        "variable": list(x.columns) + [
            f"cut {category_text(low)}|{category_text(high)}" for low, high in zip(categories[:-1], categories[1:])
        ],
        "coefficient": estimates,
        "std_error": std_errors,
        "z_stat": z,
        "p_value": 2 * stats.norm.sf(np.abs(z)),
    })
    significance.add_stars(results)
    if link == "logit":
        results["odds_ratio"] = np.concatenate([np.exp(coefficients), np.full(len(cuts), np.nan)])

    probabilities = probability_columns(model, x, categories)
    predicted = categories[probabilities.to_numpy().argmax(axis=1)]
    results.attrs = {
        "pseudo_r_squared": model.prsquared,
        "log_likelihood": model.llf,
        "lr_p": model.llr_pvalue,
        "aic": model.aic,
        "observations": int(model.nobs),
        "categories": len(categories),
        "correct": float((predicted == y.to_numpy()).mean()),
        "robust": settings.get("robust", False),
        "cluster": cluster,
        "warnings": notes,
    }
    fits.record(
        results, f"o{link}", f"O{link}", settings["dependent"], used,
        settings.get("independent", []) + settings.get("controls", []),
    )

    fitted = pd.concat([used, probabilities], axis=1)
    fitted["predicted"] = predicted
    effects = marginal_effects(model, x, roles, categories, link)
    return [results, fitted, effects]


# The panel for Ologit or Oprobit: roles, then fit statistics and the coefficient table
def show_model(panel):
    if not show_roles(panel):
        return
    node = panel.node
    if not node.last_outputs:
        panel.message = "Choose an ordered Y and the X variables, then run the workflow to fit the model"
        panel.lines.append(Y_HINT)
        return

    results = node.last_outputs[0]
    fit = results.attrs
    panel.lines.append(
        f"McFadden R squared: {fit['pseudo_r_squared']:.3f}    Log likelihood: {fit['log_likelihood']:.1f}    "
        f"N: {fit['observations']} ({fit['categories']} categories)"
    )
    panel.lines.append(
        f"LR test p: {fit['lr_p']:.3g}    AIC: {fit['aic']:.1f}    "
        f"Correctly classified: {fit['correct']:.1%}    Standard errors: {clusters.errors_text(fit)}"
    )
    panel.note = " ".join(fit["warnings"])
    panel.preview = results
