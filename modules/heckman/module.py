import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.special import log_ndtr

from engine import estimates, roles


INPUTS = 1
OUTPUTS = 2
OUTPUT_TYPES = ["model", "table"]
INPUT_SPECS = [
    ("Data", "a table with every row, selected or not: a numeric Y (missing where it isn't seen, e.g. wages of those not working), "
             "the X fields of the outcome equation, the Z fields that predict selection, and optionally a 0/1 field (S) saying "
             "who is selected"),
]
OUTPUT_SPECS = [
    ("Coefficients", "the outcome equation's terms (role X), then the selection equation's (role Selection), then lambda "
                     "(the inverse Mills ratio's coefficient), rho and sigma, each with std_error, z_stat, p_value and stars"),
    ("Fitted data", "every row used, plus selection_prob, mills (the inverse Mills ratio) and, for the selected rows, "
                    "fitted and residual"),
]

METHODS = [("two_step", "Two-step"), ("ml", "Maximum likelihood")]
SPEC = roles.Spec(
    "Heckman",
    [
        roles.Role("Y", "dependent"),
        roles.Role("X", "independent", many=True, kinds="model"),
        roles.Role("Z", "selection", many=True, kinds="model"),
        roles.Role("S", "selected"),
    ],
    choices=[roles.Choice("method", "Method", METHODS)],
    hint="Choose Y, the outcome's X fields and the selection's Z fields, then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings()


def on_click(node):
    return roles.open_panel(node, SPEC)


def mills(index):
    return np.exp(stats.norm.logpdf(index) - log_ndtr(index))


# Each row's log-likelihood in Heckman's model: unselected rows add the chance of not being
# selected, selected rows the density of Y times the chance of selection given Y.
# params holds beta, gamma, atanh(rho) and log(sigma)
def loglike(params, y, x, z, chosen):
    k, m = x.shape[1], z.shape[1]
    beta, gamma = params[:k], params[k:k + m]
    rho, sigma = np.tanh(params[-2]), np.exp(params[-1])
    zg = z @ gamma
    values = log_ndtr(-zg)
    e = (y[chosen] - x[chosen] @ beta) / sigma
    inner = (zg[chosen] + rho * e) / np.sqrt(1 - rho ** 2)
    values[chosen] = log_ndtr(inner) + stats.norm.logpdf(e) - np.log(sigma)
    return values


# Heckman's two-step estimator: a probit for selection, then OLS of Y on X and the inverse Mills
# ratio over the selected rows, with the standard errors corrected for the estimated ratio
# (Heckman 1979; Greene, Econometric Analysis, 19.5)
def two_step(y, x, z, chosen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probit = sm.Probit(chosen.astype(float), z).fit(disp=0)
    zg = z @ probit.params
    ratio = mills(zg)
    xs = np.column_stack([x[chosen], ratio[chosen]])
    ys = y[chosen]
    coefficients = np.linalg.lstsq(xs, ys, rcond=None)[0]
    residuals = ys - xs @ coefficients
    delta = ratio[chosen] * (ratio[chosen] + zg[chosen])
    beta_lambda = coefficients[-1]
    sigma2 = residuals @ residuals / len(ys) + beta_lambda ** 2 * delta.mean()
    sigma = np.sqrt(sigma2)
    rho = float(np.clip(beta_lambda / sigma, -0.999, 0.999))

    inverse = np.linalg.inv(xs.T @ xs)
    middle = xs.T @ (xs * (1 - rho ** 2 * delta)[:, None])
    q = rho ** 2 * (xs.T @ (z[chosen] * delta[:, None])) @ np.asarray(probit.cov_params()) @ ((z[chosen] * delta[:, None]).T @ xs)
    covariance = sigma2 * inverse @ (middle + q) @ inverse
    return coefficients, covariance, probit, rho, sigma


# Fits Heckman's sample selection model. The outcome Y = X b + u is seen only for selected rows,
# and a row is selected when Z g + v > 0, with u and v correlated (rho). Ignoring that, OLS on the
# selected rows is biased; lambda's coefficient (rho times sigma) measures the bias
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y", "X", "Z"))
    dependent = settings["dependent"]
    outcome_fields = settings["independent"]
    selection_fields = settings["selection"]
    flag = settings.get("selected")
    notes = []

    if flag:
        used = data.dropna(subset=list(dict.fromkeys(selection_fields + [flag])))
        values = used[flag]
        if not values.isin([0, 1]).all():
            raise ValueError(f"The selection field {flag} must be 0 or 1")
        chosen = values.to_numpy(int) == 1
    else:
        used = data.dropna(subset=selection_fields)
        chosen = used[dependent].notna().to_numpy()
    # Selected rows need their outcome fields; those missing them are dropped
    complete = used[[dependent] + outcome_fields].notna().all(axis=1).to_numpy()
    lost = chosen & ~complete
    if lost.any():
        notes.append(f"{int(lost.sum())} selected rows are missing Y or an X field, so they are left out.")
        used, chosen = used[~lost], chosen[~lost]
    if chosen.all() or not chosen.any():
        raise ValueError("Every row is selected or none is; the selection equation needs both")

    x_frame, x_roles = estimates.design(used.fillna({c: 0 for c in outcome_fields}), {"independent": outcome_fields})
    z_frame, _ = estimates.design(used, {"independent": selection_fields})
    x = x_frame.to_numpy(float)
    z = z_frame.to_numpy(float)
    y = used[dependent].fillna(0).to_numpy(float)
    if not set(selection_fields) - set(outcome_fields):
        notes.append("Every Z field is also an X field, so the model rests only on the normal distribution's shape. "
                     "Add a Z field that affects selection but not Y (an exclusion restriction).")

    coefficients, two_step_covariance, probit, rho, sigma = two_step(y, x, z, chosen)
    k = x.shape[1]
    if settings["method"] == "ml":
        start = np.concatenate([coefficients[:k], probit.params, [np.arctanh(rho)], [np.log(sigma)]])
        params, llf, covariance, converged = estimates.maximise(lambda p: loglike(p, y, x, z, chosen), start)
        if not converged:
            notes.append("The optimiser did not fully converge, so treat the estimates with caution.")
        rho, sigma = float(np.tanh(params[-2])), float(np.exp(params[-1]))
        # Turn atanh(rho) and log(sigma) back into rho, sigma and lambda = rho sigma by the delta method
        jacobian = np.zeros((len(params) + 1, len(params)))
        jacobian[:len(params) - 2, :len(params) - 2] = np.eye(len(params) - 2)
        jacobian[-3, -2] = sigma * (1 - rho ** 2)
        jacobian[-3, -1] = rho * sigma
        jacobian[-2, -2] = 1 - rho ** 2
        jacobian[-1, -1] = sigma
        covariance = jacobian @ covariance @ jacobian.T
        values = np.concatenate([params[:-2], [rho * sigma, rho, sigma]])
        errors = estimates.standard_errors(covariance)
        gamma = params[k:-2]
    else:
        llf = float(np.sum(loglike(np.concatenate([coefficients[:k], probit.params, [np.arctanh(rho)], [np.log(sigma)]]), y, x, z, chosen)))
        m = z.shape[1]
        size = k + m + 3
        covariance = np.full((size, size), np.nan)
        covariance[:k, :k] = two_step_covariance[:k, :k]
        covariance[:k, k + m] = covariance[k + m, :k] = two_step_covariance[:k, -1]
        covariance[k + m, k + m] = two_step_covariance[-1, -1]
        covariance[k:k + m, k:k + m] = np.asarray(probit.cov_params())
        values = np.concatenate([coefficients[:k], probit.params, [coefficients[-1], rho, sigma]])
        errors = np.sqrt(np.diag(np.nan_to_num(covariance, nan=-1.0)).clip(min=0))
        errors[k + m + 1:] = np.nan
        gamma = np.asarray(probit.params)

    names = list(x_frame.columns) + [f"select: {c}" for c in z_frame.columns] + ["lambda", "rho", "sigma"]
    term_roles = x_roles + ["Selection"] * z.shape[1] + ["Lambda", "Rho", "Sigma"]
    table = estimates.coefficient_table(names, term_roles, values, errors)

    lambda_p = float(table["p_value"].iloc[k + z.shape[1]])
    summary = [
        f"{dict(METHODS)[settings['method']]}    N: {len(used)} ({int(chosen.sum())} selected, {int((~chosen).sum())} not)",
        f"rho: {rho:.3f}    sigma: {sigma:.4g}    lambda: {values[k + z.shape[1]]:.4g} (p = {estimates.number_text(lambda_p)})",
    ]
    if lambda_p < 0.05:
        summary.append("lambda is significant: selection biases OLS on the selected rows, which this model corrects")
    else:
        summary.append("lambda is not significant: no evidence that selection biases OLS on the selected rows")
    if settings["method"] == "ml":
        summary.append(f"Log-likelihood: {llf:.2f}")
    table.attrs = {"observations": len(used), "selected": int(chosen.sum()), "rho": rho, "sigma": sigma,
                   "log_likelihood": llf, "summary": summary, "notes": notes}
    usable = np.isfinite(covariance)
    estimates.record(table, "heckman", "Heckman", dependent, used, outcome_fields, np.where(usable, covariance, 0.0), llf)

    fitted = used.copy()
    index = z @ gamma
    fitted["selection_prob"] = stats.norm.cdf(index)
    fitted["mills"] = mills(index)
    prediction = x @ values[:k]
    fitted["fitted"] = np.where(chosen, prediction, np.nan)
    fitted["residual"] = np.where(chosen, y - prediction, np.nan)
    return [table, fitted]
