import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime, approx_hess

from engine import dummies, fits, significance


# Pieces shared by the estimators added for gretl parity (count, censored, selection, quantile,
# GARCH, VAR, VECM and dynamic panel models): the coefficient table every regression gives, the
# record a fit keeps so Wald and LR tests and Bootstrap can use it, maximum likelihood with
# numerical derivatives, and getting a time series into time order


# A coefficient table in the shape OLS gives: role, variable, coefficient, std_error, a test
# statistic, p_value and stars. With df the statistic is t with df degrees of freedom, else z.
# extra columns (a dict of name -> values) go before the stars
def coefficient_table(names, roles, params, std_errors, df=None, extra=None, leading=None):
    params = np.asarray(params, dtype=float)
    std_errors = np.asarray(std_errors, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = params / std_errors
    if df:
        p_values = 2 * stats.t.sf(np.abs(statistic), df)
        statistic_name = "t_stat"
    else:
        p_values = 2 * stats.norm.sf(np.abs(statistic))
        statistic_name = "z_stat"
    columns = dict(leading or {})
    columns.update({
        "role": list(roles),
        "variable": [str(n) for n in names],
        "coefficient": params,
        "std_error": std_errors,
        statistic_name: statistic,
        "p_value": p_values,
    })
    table = pd.DataFrame(columns)
    for name, values in (extra or {}).items():
        table[name] = values
    significance.add_stars(table)
    return table


# Records a fit made by maximum likelihood (or any fit with a covariance), so Wald Test, LR tests
# and Bootstrap can use it
def record(results, kind, model_name, dependent, used, columns, covariance, log_likelihood=None, parameters=None,
           observations=None, **extra):
    fits.record(
        results, kind, model_name, dependent, used, columns,
        covariance=np.asarray(covariance, dtype=float),
        fields=list(columns),
        log_likelihood=None if log_likelihood is None else float(log_likelihood),
        parameters=int(parameters if parameters is not None else len(results)),
        observations=int(observations if observations is not None else len(used)),
        **extra,
    )


# The design matrix from the X and C roles, text fields as dummies, with a constant first unless
# constant is False. Returns the matrix and each column's role
def design(used, settings, constant=True, groups=(("independent", "X"), ("controls", "C"))):
    terms, roles = dummies.design_matrix(used, [(settings.get(key) or [], role) for key, role in groups])
    if constant:
        terms.insert(0, "Intercept", 1.0)
        roles = ["Intercept"] + roles
    if terms.shape[1]:
        dummies.check_full_rank(terms.rename(columns={"Intercept": "const"}))
    return terms.astype(float), roles


# Maximises a log-likelihood. loglike_obs(params) gives each row's contribution. Returns the
# estimates, the total log-likelihood and the covariance from the inverse of the numerical
# Hessian, or the sandwich (robust) one built with the outer product of the rows' scores
def maximise(loglike_obs, start, robust=False, bounds=None, method=None):
    def negative(params):
        value = -np.sum(loglike_obs(params))
        return value if np.isfinite(value) else 1e300

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize.minimize(negative, np.asarray(start, dtype=float), method=method or ("L-BFGS-B" if bounds else "BFGS"),
                                   bounds=bounds, options={"maxiter": 5000})
        if method is None and not bounds:
            # Starting BFGS again from where it stopped, with a tighter tolerance, gets it off flat spots
            again = optimize.minimize(negative, result.x, method="BFGS", options={"maxiter": 5000, "gtol": 1e-8})
            if again.fun <= result.fun:
                result = again
        params = result.x
        covariance = covariance_of(loglike_obs, params, robust)
        # BFGS often stops saying it lost precision when it is at the top already; a gradient
        # near zero (relative to the size of the log-likelihood) is taken as converged
        gradient = approx_fprime(params, lambda p: np.sum(loglike_obs(p)), centered=True)
        converged = bool(result.success) or float(np.max(np.abs(gradient))) < 1e-3 * max(1.0, abs(result.fun)) ** 0.5
    return params, -float(result.fun), covariance, converged


def covariance_of(loglike_obs, params, robust=False):
    hessian = approx_hess(params, lambda p: np.sum(loglike_obs(p)))
    try:
        inverse = np.linalg.inv(-hessian)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(-hessian)
    if not robust:
        return inverse
    scores = approx_fprime(params, loglike_obs, centered=True)
    meat = scores.T @ scores
    return inverse @ meat @ inverse


def standard_errors(covariance):
    diagonal = np.diag(np.asarray(covariance, dtype=float)).copy()
    diagonal[diagonal < 0] = np.nan
    return np.sqrt(diagonal)


# Gets one time series ready: keeps the rows of the chosen series when a group field is set, then
# sorts them by the time field. Returns the rows and their times
def time_order(data, settings, notes):
    from engine.roles import chosen_series
    from modules.arimax.module import sort_by_time
    series = chosen_series(data, settings)
    if series is not None:
        data = data[data[settings["group"]].astype(str) == series]
    return sort_by_time(data, settings, notes)


# Shows dates without a time of day
def time_text(value):
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def number_text(value, digits=3):
    try:
        if value is None or np.isnan(value):
            return "n/a"
    except TypeError:
        return str(value)
    return f"{value:.{digits}g}"


# The information criteria for a log-likelihood with k parameters and n rows
def information_criteria(log_likelihood, k, n):
    return {
        "aic": -2 * log_likelihood + 2 * k,
        "bic": -2 * log_likelihood + k * np.log(n),
        "hqc": -2 * log_likelihood + 2 * k * np.log(np.log(n)),
    }
