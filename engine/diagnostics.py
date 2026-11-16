import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg, stats
from statsmodels.stats import diagnostic, outliers_influence, stattools

from engine import conditions, dummies


# The tests a linear regression is checked with after it is fitted, worked out from the model's
# fitted data so they can be run on any OLS-type fit. Each test gives a row of the results:
#   test        its name
#   null        what it assumes, which a small p value rejects
#   statistic, df, p_value
#   result      what the p value means for the model, in plain words, with what to do about it
# The level a p value is judged at
LEVEL = 0.05

TESTS = [
    "Jarque-Bera", "Breusch-Pagan", "White", "RESET", "Breusch-Godfrey", "ARCH-LM", "Durbin-Watson",
    "Chow", "CUSUM",
]


# Rebuilds what a linear model was fitted on from its fitted data: the response, the regressors
# with an intercept (text fields as dummies, as when it was fitted) and the rows used.
# fit is the "fit" record of the model's coefficient table (see engine/fits.py)
def design(data, fit):
    dependent = fit["dependent"]
    fields = fit.get("fields")
    if fields is None:
        raise ValueError("The model was fitted by an older Coefflow and doesn't say which fields it used. Run its workflow again")
    missing = [c for c in [dependent] + fields if c not in data.columns]
    if missing:
        raise ValueError(f"The fitted data is missing {', '.join(missing)}. Connect the model's own fitted data (Out 2)")
    used = data.dropna(subset=[dependent] + fields)
    if fields:
        terms, roles = dummies.design_matrix(used, [(fields, "X")])
        x = sm.add_constant(terms, has_constant="add")
    else:
        x = pd.DataFrame({"const": 1.0}, index=used.index)
    return used[dependent].astype(float), x.astype(float), used


def fit_ols(y, x):
    return sm.OLS(y, x).fit()


def row(test, null, statistic, df, p_value, good, bad):
    passed = bool(np.isnan(p_value) or p_value >= LEVEL)
    return {
        "test": test,
        "null": null,
        "statistic": float(statistic),
        "df": df,
        "p_value": float(p_value),
        "result": good if passed else bad,
    }


def jarque_bera(model):
    statistic, p_value, skew, kurtosis = stattools.jarque_bera(model.resid)
    return row(
        "Jarque-Bera", "the errors are normally distributed", statistic, "2", p_value,
        "No evidence against normal errors.",
        f"The errors aren't normal (skew {skew:.2f}, kurtosis {kurtosis:.2f}). Tests are still fine in large samples; "
        "in small ones look for outliers or transform Y.",
    )


def breusch_pagan(model):
    statistic, p_value, f, f_p = diagnostic.het_breuschpagan(model.resid, model.model.exog)
    return row(
        "Breusch-Pagan", "the errors have the same variance throughout", statistic, str(int(model.df_model)), p_value,
        "No evidence of heteroscedasticity.",
        "Heteroscedasticity detected: the spread of the errors changes with X. Use robust (HC1) standard errors.",
    )


# White's test regresses the squared residuals on the regressors, their squares and their cross
# products, and n R squared is chi-squared. Dummies make many of those columns repeat (a dummy's
# square is itself), so only the columns that add something are kept and counted
def white(model):
    exog = np.asarray(model.model.exog, dtype=float)
    columns = [exog[:, i] for i in range(exog.shape[1])]
    for i in range(exog.shape[1]):
        for j in range(i, exog.shape[1]):
            columns.append(exog[:, i] * exog[:, j])
    candidates = np.column_stack(columns)
    kept = independent_columns(candidates)
    auxiliary = sm.OLS(np.asarray(model.resid, dtype=float) ** 2, kept).fit()
    df = kept.shape[1] - 1
    statistic = len(model.resid) * auxiliary.rsquared
    return row(
        "White", "the errors have the same variance throughout (against any form)", statistic, str(df),
        float(stats.chi2.sf(statistic, df)),
        "No evidence of heteroscedasticity of any form.",
        "Heteroscedasticity detected, possibly of a non-linear form. Use robust (HC1) standard errors.",
    )


# The columns of a matrix that aren't linear combinations of the others, found with a pivoted QR
# decomposition and kept in their original order
def independent_columns(matrix):
    r, pivots = linalg.qr(matrix, mode="r", pivoting=True)
    diagonal = np.abs(np.diag(r))
    tolerance = diagonal.max() * max(matrix.shape) * np.finfo(float).eps if len(diagonal) else 0
    rank = int((diagonal > tolerance).sum())
    return matrix[:, sorted(pivots[:rank])]


def reset(model, power=3):
    result = diagnostic.linear_reset(model, power=power, test_type="fitted", use_f=True)
    return row(
        "RESET", f"the model is correctly specified (no missing powers of the fitted values up to {power})",
        result.fvalue, f"{int(result.df_num)}, {int(result.df_denom)}", result.pvalue,
        "No evidence the functional form is wrong.",
        "The functional form looks wrong: try logs, squares or interactions, or a variable may be missing.",
    )


def breusch_godfrey(model, lags):
    statistic, p_value, f, f_p = diagnostic.acorr_breusch_godfrey(model, nlags=lags)
    return row(
        "Breusch-Godfrey", f"no autocorrelation in the errors up to lag {lags}", statistic, str(lags), p_value,
        "No evidence of autocorrelation.",
        "Autocorrelation detected, so the standard errors are wrong. Use HAC (Newey-West) errors, add lags, or use ARIMAX.",
    )


def arch_lm(model, lags):
    statistic, p_value, f, f_p = diagnostic.het_arch(model.resid, nlags=lags)
    return row(
        "ARCH-LM", f"no ARCH effects (volatility clustering) up to lag {lags}", statistic, str(lags), p_value,
        "No evidence the variance of the errors clusters over time.",
        "The variance of the errors clusters over time (ARCH effects). Use HAC errors, or a GARCH model for the variance.",
    )


def durbin_watson(model):
    d = float(stattools.durbin_watson(model.resid))
    pairs = len(model.resid) - 1
    rho = 1 - d / 2
    p_value = float(2 * stats.norm.sf(abs(rho * np.sqrt(pairs))))
    return row(
        "Durbin-Watson", "no first-order autocorrelation (d near 2)", d, "", p_value,
        f"d = {d:.2f}, close to 2: no evidence of first-order autocorrelation.",
        f"d = {d:.2f}, implying autocorrelation of about {rho:.2f}. Use HAC errors or model the errors with ARIMAX.",
    )


# The Chow test of whether the coefficients are the same before and after a break. The break is
# a condition such as year >= 2008 picking the rows after it. F compares the residual sum of
# squares of the pooled fit with the two fits on each side
def chow(y, x, used, condition):
    after = conditions.evaluate(used, condition).fillna(False).astype(bool).to_numpy()
    k = x.shape[1]
    first, second = ~after, after
    if first.sum() <= k or second.sum() <= k:
        raise ValueError(f"Each side of the break needs more than {k} rows; {int(first.sum())} and {int(second.sum())} fall either side of {condition}")
    pooled = fit_ols(y, x).ssr
    parts = fit_ols(y[first], x[first]).ssr + fit_ols(y[second], x[second]).ssr
    df_denom = len(y) - 2 * k
    f = ((pooled - parts) / k) / (parts / df_denom)
    p_value = float(stats.f.sf(f, k, df_denom))
    return row(
        "Chow", f"the coefficients are the same either side of {condition}", f, f"{k}, {df_denom}", p_value,
        "No evidence of a structural break there.",
        f"Structural break: the coefficients differ either side of {condition}. Fit the two periods apart, or interact X with a dummy for the break.",
    )


# The CUSUM test on the OLS residuals (Ploberger and Kramer 1992): their running sum, scaled,
# stays inside its bounds if the coefficients are stable over the rows in order
def cusum(model):
    statistic, p_value, critical = diagnostic.breaks_cusumolsresid(np.asarray(model.resid), ddof=int(model.df_model) + 1)
    return row(
        "CUSUM", "the coefficients are stable over the rows in order", statistic, "", p_value,
        "No evidence the coefficients drift over the sample.",
        "The coefficients aren't stable over the sample: look for a structural break, e.g. with the Chow test.",
    )


# The running sum of the recursive residuals and its 5% bounds, for the CUSUM chart, as
# (positions, cusum, lower bound, upper bound)
def cusum_path(model):
    results = diagnostic.recursive_olsresiduals(model, skip=None, alpha=0.95)
    path = np.asarray(results[5], dtype=float)
    lower, upper = (np.asarray(bound, dtype=float) for bound in results[6])
    positions = np.arange(len(path)) + (len(model.resid) - len(path))
    return positions, path, lower, upper


# Runs every chosen test, skipping any that can't be worked out for this model with the reason.
# Returns the results table and a list of notes
def run_tests(y, x, used, chosen, lags=1, power=3, condition=""):
    model = fit_ols(y, x)
    rows, notes = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return finish_tests(model, y, x, used, chosen, lags, power, condition, rows, notes)


def finish_tests(model, y, x, used, chosen, lags, power, condition, rows, notes):
    actions = {
        "Jarque-Bera": lambda: jarque_bera(model),
        "Breusch-Pagan": lambda: breusch_pagan(model),
        "White": lambda: white(model),
        "RESET": lambda: reset(model, power),
        "Breusch-Godfrey": lambda: breusch_godfrey(model, lags),
        "ARCH-LM": lambda: arch_lm(model, lags),
        "Durbin-Watson": lambda: durbin_watson(model),
        "Chow": lambda: chow(y, x, used, condition),
        "CUSUM": lambda: cusum(model),
    }
    for name in TESTS:
        if name not in chosen:
            continue
        if name == "Chow" and not condition.strip():
            notes.append("Type a break condition, such as year >= 2008, to run the Chow test.")
            continue
        if name in ("Breusch-Pagan", "White") and x.shape[1] < 2:
            continue
        try:
            rows.append(actions[name]())
        except Exception as error:
            notes.append(f"{name} could not be worked out: {error}")
    table = pd.DataFrame(rows, columns=["test", "null", "statistic", "df", "p_value", "result"])
    return table, notes, model


# The variance inflation factor of each regressor: how much its coefficient's variance is blown up
# by its correlation with the others. Above 10 is the usual sign of troubling collinearity
def vif(x):
    rows = []
    values = x.to_numpy(float)
    for index, name in enumerate(x.columns):
        if name == "const":
            continue
        factor = float(outliers_influence.variance_inflation_factor(values, index)) if x.shape[1] > 1 else np.nan
        rows.append({
            "variable": name,
            "vif": factor,
            "tolerance": 1 / factor if factor else np.nan,
            "flag": "high" if factor > 10 else ("moderate" if factor > 5 else ""),
        })
    return pd.DataFrame(rows, columns=["variable", "vif", "tolerance", "flag"])


# For each row: its leverage (how unusual its X values are), studentized residual, Cook's distance
# (how much it moves the coefficients) and DFFITS, flagged against the usual cut-offs:
# leverage above 2k/n, |studentized residual| above 3, Cook's distance above 4/n
def influence(model, used):
    # The closed forms (Belsley, Kuh and Welsch 1980), rather than refitting without each row
    exog = np.asarray(model.model.exog, dtype=float)
    n, k = exog.shape
    q, r = np.linalg.qr(exog)
    leverage = np.clip((q ** 2).sum(axis=1), 0, 1 - 1e-12)
    resid = np.asarray(model.resid, dtype=float)
    ssr = float(resid @ resid)
    s2 = ssr / (n - k)
    s2_without = np.maximum((ssr - resid ** 2 / (1 - leverage)) / (n - k - 1), 1e-300)
    studentized = resid / np.sqrt(s2_without * (1 - leverage))
    table = used.copy()
    table["fitted"] = model.fittedvalues
    table["residual"] = model.resid
    table["leverage"] = leverage
    table["studentized"] = studentized
    table["cooks_d"] = resid ** 2 / (k * s2) * leverage / (1 - leverage) ** 2
    table["dffits"] = studentized * np.sqrt(leverage / (1 - leverage))
    flags = []
    for leverage, student, cook in zip(table["leverage"], table["studentized"], table["cooks_d"]):
        found = []
        if leverage > 2 * k / n:
            found.append("leverage")
        if abs(student) > 3:
            found.append("outlier")
        if cook > 4 / n:
            found.append("influential")
        flags.append(", ".join(found))
    table["flag"] = flags
    return table


# Theoretical normal quantiles against the sorted standardized residuals, for a Q-Q plot
def qq_points(residuals):
    values = np.sort(np.asarray(residuals, dtype=float))
    values = (values - values.mean()) / values.std(ddof=1) if values.std(ddof=1) > 0 else values
    probabilities = (np.arange(1, len(values) + 1) - 0.5) / len(values)
    return stats.norm.ppf(probabilities), values
