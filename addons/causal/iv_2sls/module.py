import numpy as np
import pandas as pd
from scipy import stats

from engine import clusters, dummies, fits, panel_data
from .. import common


INPUTS = 1
OUTPUTS = 3
INPUT_SPECS = [
    ("Data", "a table with a numeric Y, the endogenous variables (En), the instruments (Z) and any exogenous controls (C). Text fields enter as dummies. Rows missing any of them are dropped"),
]
OUTPUT_SPECS = [
    ("Coefficients", "the second stage: one row per term with role, variable, coefficient, std_error, t_stat, p_value and stars"),
    ("First stage", "one row per term of each first stage regression of an endogenous variable on the instruments and controls"),
    ("Tests", "one row per test: first stage F and partial R squared, Sanderson-Windmeijer F, Cragg-Donald F with Stock-Yogo values, Anderson-Rubin, Sargan or Hansen J, and Durbin-Wu-Hausman, each with its p value and verdict"),
]

# The endogenous variables are kept under "independent" so the Cluster panel treats them as the model's X
ROLES = [("Y", "dependent", False), ("En", "independent", True), ("Z", "instruments", True), ("C", "controls", True)]
FIELD_SETTINGS = ["dependent", "independent", "instruments", "controls", "cluster"]
SHOW = ["Coefficients", "First stage", "Tests"]

# Stock & Yogo (2005) critical values for the Cragg-Donald F, by the number of endogenous variables
# then the number of instruments. Above the size value, a nominal 5% Wald test on the 2SLS estimates
# rejects at most 10% of the time. Above the bias value, 2SLS has at most 10% of the bias of OLS,
# which is only defined with at least 2 more instruments than endogenous variables
STOCK_YOGO_SIZE = {
    1: {1: 16.38, 2: 19.93, 3: 22.30, 4: 24.58, 5: 26.87, 6: 29.18, 7: 31.50, 8: 33.84, 9: 36.19, 10: 38.54},
    2: {2: 7.03, 3: 13.43, 4: 16.87, 5: 19.45, 6: 21.68, 7: 23.72, 8: 25.64, 9: 27.51, 10: 29.32},
}
STOCK_YOGO_BIAS = {
    1: {3: 9.08, 4: 10.27, 5: 10.83, 6: 11.12, 7: 11.29, 8: 11.39, 9: 11.46, 10: 11.49},
    2: {4: 7.56, 5: 8.78, 6: 9.48, 7: 9.92, 8: 10.22, 9: 10.43, 10: 10.58},
}


# Opens a panel to give each variable a role: Y (dependent), En (endogenous), Z (instrument) or
# C (exogenous control), with buttons for robust and clustered standard errors
def on_click(node):
    return common.open_panel(node, "IV 2SLS", ROLES, show_model)


def show_model(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""
    panel.message = ""
    panel.lines = [
        f"Y: {common.names_text(settings, 'dependent')}    Endogenous (En): {common.names_text(settings, 'independent')}",
        f"Instruments (Z): {common.names_text(settings, 'instruments')}    Controls (C): {common.names_text(settings, 'controls')}",
    ]
    common.add_choice(panel, "Show", "show", SHOW, SHOW[0])
    if not common.add_role_options(panel, ROLES, common.default_allowed(["En", "Z", "C"])):
        return

    if not node.last_outputs:
        panel.message = "Choose Y, the endogenous variables and at least as many instruments, then run the workflow"
        return
    results, first, tests = node.last_outputs
    fit = results.attrs
    panel.lines.append(
        f"N: {fit['observations']}    R squared: {common.number_text(fit['r_squared'])}    "
        f"{fit['endogenous']} endogenous, {fit['instruments']} instruments    Standard errors: {clusters.errors_text(fit)}"
    )
    panel.lines += fit["summary"]
    panel.note = " ".join(fit["warnings"] + (fit["cluster"]["warnings"] if fit.get("cluster") else []))
    panel.preview = {"Coefficients": results, "First stage": first, "Tests": tests}[settings.get("show", SHOW[0])]


# A row of the tests table
def test_row(test, applies_to, statistic, df1=np.nan, df2=np.nan, p_value=np.nan, verdict=""):
    return {"test": test, "applies_to": applies_to, "statistic": statistic, "df1": df1, "df2": df2, "p_value": p_value, "verdict": verdict}


def strength_verdict(f):
    if f >= common.STRONG_F:
        return f"Strong: F is {common.STRONG_F} or more"
    return f"Weak: F below {common.STRONG_F}, so 2SLS may be biased towards OLS and its tests unreliable"


# Stock-Yogo values for the Cragg-Donald F with k endogenous variables and l instruments, as text
def stock_yogo_verdict(f, k, l):
    parts = []
    for name, table in [("10% maximal size", STOCK_YOGO_SIZE), ("10% maximal bias", STOCK_YOGO_BIAS)]:
        value = table.get(k, {}).get(l)
        if value is not None:
            parts.append(f"{'above' if f > value else 'below'} the Stock-Yogo {name} value {value}")
    if not parts:
        return strength_verdict(f)
    return ("Strong: " if all(p.startswith("above") for p in parts) else "Weak: ") + "; ".join(parts)


# Each first stage: the regression of one endogenous term on the instruments and controls, the F test
# of the excluded instruments, the partial R squared (the share of the term's variation left after
# the controls that the instruments explain) and, with several endogenous terms, Sanderson and
# Windmeijer's F, which asks whether the instruments explain this term apart from the others
def first_stages(endog, excluded, w, z, z_names, z_roles, robust, groups, df_t):
    n, k = endog.shape
    l = excluded.shape[1]
    df_first = n - z.shape[1]
    positions = list(range(l))
    excluded_left = excluded - w @ np.linalg.lstsq(w, excluded, rcond=None)[0]
    tables, rows = [], []
    for j in range(k):
        target = endog[:, j]
        beta, cov, resid = common.least_squares(z, target, df_first, robust, groups)
        table = panel_data.coefficient_table(z_names, z_roles, beta, cov, df_t if groups is not None else df_first)
        tables.append(table)
        f, p = common.wald(beta, cov, positions, df_t if groups is not None else df_first)

        target_left = target - w @ np.linalg.lstsq(w, target, rcond=None)[0]
        explained = excluded_left @ np.linalg.lstsq(excluded_left, target_left, rcond=None)[0]
        partial = float(explained @ explained / (target_left @ target_left))
        rows.append((f, p, partial))
    return tables, rows


# Sanderson and Windmeijer's conditional F for endogenous term j: the term's 2SLS residual after the
# other endogenous terms and the controls, tested on the excluded instruments, with L - K + 1
# numerator degrees of freedom
def sanderson_windmeijer(endog, j, w, z, l, robust, groups, df_t):
    n, k = endog.shape
    others = np.column_stack([np.delete(endog, j, axis=1), w])
    beta, resid, xhat = common.two_stage(endog[:, j], others, z)
    df_resid = n - z.shape[1]
    b, cov, r = common.least_squares(z, resid, df_resid, robust, groups)
    f, p = common.wald(b, cov, list(range(l)), df_resid)
    numerator = l - k + 1
    f = f * l / numerator
    df2 = df_t if groups is not None else df_resid
    return f, numerator, df2, float(stats.f.sf(f, numerator, df2))


# The Cragg-Donald F: the smallest eigenvalue of the matrix version of the first stage F, which is
# the first stage F itself with one endogenous variable. It assumes errors that are independent and
# have the same variance
def cragg_donald(endog, excluded, w):
    n, l = excluded.shape
    x = endog - w @ np.linalg.lstsq(w, endog, rcond=None)[0]
    zt = excluded - w @ np.linalg.lstsq(w, excluded, rcond=None)[0]
    projected = zt @ np.linalg.lstsq(zt, x, rcond=None)[0]
    explained = projected.T @ x
    left = x.T @ x - explained
    value = float(np.min(np.real(np.linalg.eigvals(np.linalg.solve(left, explained)))))
    return value * (n - l - w.shape[1]) / l


# Hansen's J test of the overidentifying restrictions, robust to the kind of errors chosen: the
# moments Z'u at the two-step efficient GMM estimate, weighted by the inverse of their covariance
def hansen_j(y, x, z, resid, groups):
    n = len(y)
    scores = z * resid[:, None]
    if groups is not None:
        summed = np.zeros((groups.max() + 1, z.shape[1]))
        np.add.at(summed, groups, scores)
        scores = summed
    weight = np.linalg.pinv(scores.T @ scores / n)
    zx, zy = z.T @ x / n, z.T @ y / n
    beta = np.linalg.solve(zx.T @ weight @ zx, zx.T @ weight @ zy)
    moments = z.T @ (y - x @ beta) / n
    return float(n * moments @ weight @ moments)


# Two stage least squares of Y on the endogenous variables and controls, using the instruments for
# the endogenous ones. Returns the second stage coefficients (with fit statistics and notes in its
# attrs), the first stage coefficients and the table of tests
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    dependent = settings.get("dependent")
    endogenous = settings.get("independent", [])
    instruments = settings.get("instruments", [])
    controls = settings.get("controls", [])
    robust = settings.get("robust", False)
    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")
    if not endogenous:
        raise ValueError("Choose at least one endogenous variable (En)")
    if not instruments:
        raise ValueError("Choose at least one instrument (Z): a variable that moves the endogenous ones but affects Y only through them")
    common.check_fields(data, settings, ["dependent", "independent", "instruments", "controls"])

    fields = [dependent] + endogenous + instruments + controls
    used = data.dropna(subset=fields + clusters.cluster_columns(data, settings))
    notes = []
    if len(used) < len(data):
        notes.append(f"Dropped {len(data) - len(used)} rows with missing values.")

    endog_frame, endog_roles = dummies.design_matrix(used, [(endogenous, "En")])
    excluded_frame, excluded_roles = dummies.design_matrix(used, [(instruments, "Z")])
    controls_frame, control_roles = dummies.design_matrix(used, [(controls, "C")])
    n, k, l = len(used), endog_frame.shape[1], excluded_frame.shape[1]
    if l < k:
        raise ValueError(f"{k} endogenous terms need at least {k} instruments, but there are {l}. Add instruments or drop endogenous variables")

    y = used[dependent].to_numpy(float)
    endog = endog_frame.to_numpy(float)
    excluded = excluded_frame.to_numpy(float)
    w = np.column_stack([np.ones(n), controls_frame.to_numpy(float)])
    x = np.column_stack([w[:, :1], endog, w[:, 1:]])
    z = np.column_stack([excluded, w])
    if n < z.shape[1] + 10:
        raise ValueError(f"{n} rows for {z.shape[1]} instruments and controls. Use fewer variables or more rows")
    common.check_rank(z, "instruments and controls")
    common.check_rank(x, "endogenous variables and controls")

    parameters = x.shape[1]
    df_resid = n - parameters
    groups, cluster, df_t = common.error_groups(used, settings.get("cluster"), parameters, df_resid)
    beta, resid, xhat = common.two_stage(y, x, z)
    cov = common.covariance(xhat, resid, df_resid, robust, groups)
    names = ["Intercept"] + list(endog_frame.columns) + list(controls_frame.columns)
    results = panel_data.coefficient_table(names, ["Intercept"] + endog_roles + control_roles, beta, cov, df_t)

    # First stages
    z_names = list(excluded_frame.columns) + ["Intercept"] + list(controls_frame.columns)
    z_roles = excluded_roles + ["Intercept"] + control_roles
    first_tables, strengths = first_stages(endog, excluded, w, z, z_names, z_roles, robust, groups, df_t)
    first = pd.concat(
        [table.assign(endogenous=name) for table, name in zip(first_tables, endog_frame.columns)], ignore_index=True
    )
    first.insert(0, "endogenous", first.pop("endogenous"))

    df_first = df_t if groups is not None else n - z.shape[1]
    rows = []
    for name, (f, p, partial) in zip(endog_frame.columns, strengths):
        rows.append(test_row("First stage F", name, f, l, df_first, p, strength_verdict(f)))
        rows.append(test_row("Partial R squared", name, partial, verdict="Share of its variation, after the controls, the instruments explain"))
    if k > 1:
        for j, name in enumerate(endog_frame.columns):
            f, df1, df2, p = sanderson_windmeijer(endog, j, w, z, l, robust, groups, df_t)
            rows.append(test_row("Sanderson-Windmeijer F", name, f, df1, df2, p, strength_verdict(f)))
    cd = cragg_donald(endog, excluded, w)
    rows.append(test_row("Cragg-Donald F", "all", cd, verdict=stock_yogo_verdict(cd, k, l)))

    # Anderson-Rubin: Y on the instruments and controls. The instruments matter only through the
    # endogenous variables, so their joint test is a test of the endogenous coefficients being 0
    # that stays valid however weak the instruments are
    b_ar, cov_ar, r_ar = common.least_squares(z, y, n - z.shape[1], robust, groups)
    f_ar, p_ar = common.wald(b_ar, cov_ar, list(range(l)), df_first)
    rows.append(test_row(
        "Anderson-Rubin", "all", f_ar, l, df_first, p_ar,
        "The endogenous terms affect Y (robust to weak instruments)" if p_ar < common.LEVEL
        else "No evidence the endogenous terms affect Y, robust to weak instruments",
    ))

    # Overidentification: with more instruments than endogenous terms, the extra ones can check the
    # rest. Sargan's n R squared assumes ordinary errors, Hansen's J allows robust or clustered ones
    if l > k and groups is not None and groups.max() + 1 <= z.shape[1]:
        notes.append(f"Hansen J needs more clusters than instruments and controls ({z.shape[1]}), so the instruments' validity isn't tested.")
    elif l > k:
        if robust or groups is not None:
            over_name, statistic = "Hansen J", hansen_j(y, x, z, resid, groups)
        else:
            fitted = z @ np.linalg.lstsq(z, resid, rcond=None)[0]
            centred = resid - resid.mean()
            over_name, statistic = "Sargan", float(n * (1 - ((resid - fitted) @ (resid - fitted)) / (centred @ centred)))
        p_over = float(stats.chi2.sf(statistic, l - k))
        rows.append(test_row(
            over_name, "all", statistic, l - k, np.nan, p_over,
            "Rejected: at least one instrument may be invalid (correlated with the error)" if p_over < common.LEVEL
            else "Not rejected: no evidence against the instruments' validity",
        ))
    else:
        notes.append("Exactly identified (as many instruments as endogenous terms), so the instruments' validity can't be tested.")

    # Durbin-Wu-Hausman by control function: OLS with the first stage residuals added. If they matter,
    # the endogenous variables are correlated with the error and OLS is inconsistent
    first_resid = endog - z @ np.linalg.lstsq(z, endog, rcond=None)[0]
    augmented = np.column_stack([x, first_resid])
    df_dwh = n - augmented.shape[1]
    b_dwh, cov_dwh, r_dwh = common.least_squares(augmented, y, df_dwh, robust, groups)
    f_dwh, p_dwh = common.wald(b_dwh, cov_dwh, list(range(x.shape[1], augmented.shape[1])), df_t if groups is not None else df_dwh)
    rows.append(test_row(
        "Durbin-Wu-Hausman", "all", f_dwh, k, df_t if groups is not None else df_dwh, p_dwh,
        "Endogenous: OLS is inconsistent, so use 2SLS" if p_dwh < common.LEVEL
        else "Not rejected: OLS may be consistent, and is more precise",
    ))
    tests = pd.DataFrame(rows)

    weakest = min(f for f, p, partial in strengths)
    summary = [f"Weakest first stage F: {weakest:.3g}    Cragg-Donald F: {cd:.3g}    Anderson-Rubin p: {p_ar:.3g}",
               f"Endogeneity (Durbin-Wu-Hausman) p: {p_dwh:.3g}"]
    if any(row["test"] in ("Sargan", "Hansen J") for row in rows):
        summary[1] += f"    {over_name} p: {p_over:.3g}"
    if weakest < common.STRONG_F:
        notes.append("The instruments are weak, so the 2SLS estimates may be biased and their p values too small; use the Anderson-Rubin test.")
    if (robust or groups is not None) and k > 1:
        notes.append("The Cragg-Donald F assumes ordinary errors; with robust or clustered errors rely on the Sanderson-Windmeijer F.")

    centred = y - y.mean()
    results.attrs = {
        "r_squared": float(1 - resid @ resid / (centred @ centred)),
        "observations": n,
        "endogenous": k,
        "instruments": l,
        "robust": robust,
        "cluster": cluster,
        "summary": summary,
        "warnings": notes,
    }
    fits.record(
        results, "linear", "IV 2SLS", dependent, used, endogenous + controls,
        covariance=cov, residual_variance=float(resid @ resid / df_resid), degrees_of_freedom=float(df_resid),
    )
    return [results, first, tests]
