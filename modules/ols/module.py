import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from engine import clusters, dummies, fits, significance
from ui.clusters import add_cluster_button
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with a numeric column for Y, numeric or text columns for X and any controls (C), and optionally a field to cluster by. Text columns enter as dummies. Rows missing any of them are dropped"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, coefficient, std_error, t_stat, p_value and stars (see Settings > Configure). R squared, N and Breusch-Pagan p are shown above"),
    ("Fitted data", "the rows used, plus fitted and residual columns"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["dependent", "independent", "controls", "cluster"]


# Opens a panel to give each variable a role: Y (dependent), X (independent) or C (control),
# with a button to switch heteroscedasticity-robust standard errors on and off and one to cluster them
def on_click(node):
    panel = Panel(node, "OLS")
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
            settings["independent"].append(column)
        else:
            settings["controls"].append(column)

    show_model(panel)


# Lists the variables from the last run with Y, X and C buttons (text ones enter as dummies), then the last results
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
        f"Dependent (Y): {settings.get('dependent') or 'none'}",
        f"Independent (X): {', '.join(settings.get('independent', [])) or 'none'}",
        f"Controls (C): {', '.join(settings.get('controls', [])) or 'none'}",
    ]

    data = node.last_inputs[0]
    for column in dummies.model_columns(data):
        if column == settings.get("cluster"):
            continue
        role = role_of(settings, column)
        panel.add_option(column, [
            ("Y", choose_role, role == "Y"),
            ("X", choose_role, role == "X"),
            ("C", choose_role, role == "C"),
        ])

    if node.last_outputs:
        results = node.last_outputs[0]
        fit = results.attrs
        errors = clusters.errors_text(fit)
        panel.lines.append(
            f"R squared: {fit['r_squared']:.3f}    Adjusted: {fit['adj_r_squared']:.3f}    N: {fit['observations']}"
        )
        panel.lines.append(
            f"Standard errors: {errors}    Breusch-Pagan p: {fit['breusch_pagan_p']:.3g}"
        )
        if fit.get("cluster"):
            panel.note = " ".join(fit["cluster"]["warnings"])
        panel.preview = results
    else:
        panel.message = "Choose the variables, then run the workflow to fit the model"


# Fits OLS of Y on the X and C variables with an intercept, and returns the coefficient table
# and the data rows used with each one's fitted value and residual.
# Robust (HC1) standard errors are used if switched on, or clustered ones if a cluster field is
# chosen, which are robust too. Rows with no cluster value are dropped. Fit statistics and the
# Breusch-Pagan test for heteroscedasticity are kept in the table's attrs
def run(inputs, settings):
    data = inputs[0]
    dependent = settings.get("dependent")
    independent = settings.get("independent", [])
    controls = settings.get("controls", [])
    robust = settings.get("robust", False)

    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")
    if not independent:
        raise ValueError("Choose at least one independent variable (X)")

    missing = [c for c in [dependent] + independent + controls if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")

    if not pd.api.types.is_numeric_dtype(data[dependent]):
        raise ValueError(f"{dependent} is not numeric, so it cannot be the dependent variable (Y)")

    used = data.dropna(subset=[dependent] + independent + controls + clusters.cluster_columns(data, settings))
    terms, roles = dummies.design_matrix(used, [(independent, "X"), (controls, "C")])
    x = sm.add_constant(terms, has_constant="add")
    dummies.check_full_rank(x)
    cov_type, cov_kwds, cluster = clusters.covariance(used, settings, x.shape[1])
    model = sm.OLS(used[dependent], x).fit(cov_type=cov_type, cov_kwds=cov_kwds, use_t=True)
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
        "robust": robust,
        "cluster": cluster,
        "breusch_pagan_p": breusch_pagan_p,
    }
    fits.record_linear(results, "OLS", model, dependent, used, independent + controls)

    fitted = used.copy()
    fitted["fitted"] = model.fittedvalues
    fitted["residual"] = model.resid
    return [results, fitted]

