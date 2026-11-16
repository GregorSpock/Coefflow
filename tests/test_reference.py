import importlib

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from tests.addon_cases import CASES, run_case
from tests.support import SAMPLE_DATA


# Checks the estimators against values worked out another way: the certified results NIST
# publishes, or the textbook formula written out in numpy. They guard the numbers themselves,
# not just that a module runs


def core(name):
    return importlib.import_module(f"modules.{name}.module")


def coefficient(table, variable, column="coefficient"):
    return float(table.loc[table["variable"] == variable, column].iloc[0])


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(SAMPLE_DATA)


# Least squares coefficients, their usual standard errors and their HC1 robust ones
def least_squares(y, x):
    x = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residuals = y - x @ beta
    n, k = x.shape
    classic = np.sqrt(np.diag(inverse) * (residuals @ residuals) / (n - k))
    meat = (x * residuals[:, None] ** 2).T @ x
    robust = np.sqrt(np.diag(inverse @ meat @ inverse) * n / (n - k))
    return beta, classic, robust, residuals


# The Longley data is badly collinear, which makes it the hard test for least squares in the
# NIST Statistical Reference Datasets. The certified values are from there
def test_ols_matches_nist_longley():
    longley = sm.datasets.longley.load_pandas().data
    fields = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
    results = core("ols").run([longley], {"dependent": "TOTEMP", "independent": fields})[0]
    certified = {
        "Intercept": -3482258.63459582,
        "GNPDEFL": 15.0618722713733,
        "GNP": -0.358191792925910e-01,
        "UNEMP": -0.202022980381683e-01,
        "ARMED": -0.103322686717359e-01,
        "POP": -0.511041056535807e-01,
        "YEAR": 1829.15146461355,
    }
    # The copy of the data in statsmodels measures these two fields in units a hundredth the size
    # of those in the NIST file, so their coefficients come out a hundred times larger
    units = {"UNEMP": 100, "ARMED": 100}
    for variable, value in certified.items():
        assert coefficient(results, variable) == pytest.approx(value * units.get(variable, 1), rel=1e-7)
    assert results.attrs["r_squared"] == pytest.approx(0.995479004577296, rel=1e-9)


def test_ols_standard_errors(data):
    fields = ["gdp_growth", "inflation", "bank_rate"]
    beta, classic, robust, _ = least_squares(data["unemployment"].to_numpy(), data[fields].to_numpy())

    plain = core("ols").run([data], {"dependent": "unemployment", "independent": fields[:1], "controls": fields[1:]})[0]
    assert plain["coefficient"].to_numpy() == pytest.approx(beta, rel=1e-8)
    assert plain["std_error"].to_numpy() == pytest.approx(classic, rel=1e-8)

    hc1 = core("ols").run([data], {"dependent": "unemployment", "independent": fields, "robust": True})[0]
    assert hc1["std_error"].to_numpy() == pytest.approx(robust, rel=1e-8)


# A text field enters as dummies, so each category's residuals average zero
def test_ols_text_field_enters_as_dummies(data):
    settings = {"dependent": "unemployment", "independent": ["gdp_growth"], "controls": ["region"]}
    results, fitted = core("ols").run([data], settings)
    assert len(results) == 2 + data["region"].nunique() - 1
    assert np.abs(fitted.groupby("region")["residual"].mean()).max() < 1e-8


# Logit and probit by Newton's method on the log likelihood
def maximum_likelihood(y, x, link):
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        index = x @ beta
        if link == "logit":
            p = 1 / (1 + np.exp(-index))
            gradient = x.T @ (y - p)
            hessian = -(x * (p * (1 - p))[:, None]).T @ x
        else:
            p, density = stats.norm.cdf(index), stats.norm.pdf(index)
            score = density * (y - p) / (p * (1 - p))
            gradient = x.T @ score
            hessian = -(x * (score * (score + index))[:, None]).T @ x
        step = np.linalg.solve(hessian, gradient)
        beta = beta - step
        if np.abs(step).max() < 1e-12:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(-hessian)))


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_binary_models(data, link):
    data = data.assign(high=(data["unemployment"] > data["unemployment"].median()).astype(int))
    fields = ["gdp_growth", "inflation"]
    beta, errors = maximum_likelihood(data["high"].to_numpy(float), data[fields].to_numpy(), link)
    results = core(link).run([data], {"dependent": "high", "independent": fields})[0]
    assert results["coefficient"].to_numpy() == pytest.approx(beta, rel=1e-5)
    assert results["std_error"].to_numpy() == pytest.approx(errors, rel=1e-4)


# Fixed effects is least squares on each entity's deviations from its own mean, with the
# degrees of freedom reduced by the number of entities
def test_fixed_effects_is_the_within_estimator(data):
    fields = ["gdp_growth", "inflation"]
    settings = {"dependent": "unemployment", "independent": fields, "entity": "region", "time": "year"}
    results = core("fixed_effects").run([data], settings)[0]

    columns = ["unemployment"] + fields
    demeaned = data[columns] - data.groupby("region")[columns].transform("mean")
    x, y = demeaned[fields].to_numpy(), demeaned["unemployment"].to_numpy()
    beta = np.linalg.solve(x.T @ x, x.T @ y)
    residuals = y - x @ beta
    freedom = len(data) - data["region"].nunique() - len(fields)
    errors = np.sqrt(np.diag(np.linalg.inv(x.T @ x)) * (residuals @ residuals) / freedom)

    for i, field in enumerate(fields):
        assert coefficient(results, field) == pytest.approx(beta[i], rel=1e-8)
        assert coefficient(results, field, "std_error") == pytest.approx(errors[i], rel=1e-8)
    assert results.attrs["r_squared_within"] == pytest.approx(1 - residuals @ residuals / (y @ y), rel=1e-8)


# Random effects reports the share of variance from the entities, and Hausman compares the two models
def test_random_effects_and_hausman(data):
    settings = {"dependent": "unemployment", "independent": ["gdp_growth", "inflation"], "entity": "region", "time": "year"}
    fixed = core("fixed_effects").run([data], dict(settings))[0]
    random = core("random_effects").run([data], dict(settings))[0]
    assert 0 < random.attrs["theta_median"] < 1
    sigma_u, sigma_e = random.attrs["sigma_u"], random.attrs["sigma_e"]
    assert random.attrs["rho"] == pytest.approx(sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2), rel=1e-8)
    hausman = core("hausman").run([fixed, random], {})
    assert not hausman[0].empty


def test_durbin_watson(data):
    fitted = core("ols").run([data], {"dependent": "unemployment", "independent": ["gdp_growth"]})[1]
    table = core("durbin_watson").run([fitted], {})[0]
    residuals = fitted["residual"].to_numpy()
    expected = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    assert table["durbin_watson"].iloc[0] == pytest.approx(expected, rel=1e-10)


# Two stage least squares: regress Y on the first stage's fitted values
def test_iv_2sls():
    results = run_case("iv_2sls")[0]
    data = pd.read_csv(CASES["iv_2sls"][2])
    ones = np.ones(len(data))
    instruments = np.column_stack([ones, data[["distance", "q1_birth", "female"]]])
    regressors = np.column_stack([ones, data[["educ", "female"]]])
    fitted = instruments @ np.linalg.lstsq(instruments, regressors, rcond=None)[0]
    beta = np.linalg.lstsq(fitted, data["log_wage"].to_numpy(), rcond=None)[0]
    for variable, value in zip(["Intercept", "educ", "female"], beta):
        assert coefficient(results, variable) == pytest.approx(value, rel=1e-8)


# The 2x2 difference in differences is the change in the treated minus the change in the controls
def test_did_2x2():
    results = run_case("did_2x2")[0]
    data = pd.read_csv(CASES["did_2x2"][2])
    means = data.groupby(["treated_state", "post"])["employment"].mean()
    expected = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
    effect = float(results.loc[results["role"] == "Effect", "coefficient"].iloc[0])
    assert effect == pytest.approx(expected, rel=1e-9)


def test_welch_t_test():
    table = run_case("t_test")[0]
    data = pd.read_csv(CASES["t_test"][2])
    female = data.loc[data["gender"] == "female", "wellbeing"].dropna()
    male = data.loc[data["gender"] == "male", "wellbeing"].dropna()
    expected = stats.ttest_ind(female, male, equal_var=False)
    assert table["t"].iloc[0] == pytest.approx(expected.statistic, rel=1e-9)
    assert table["p_value"].iloc[0] == pytest.approx(expected.pvalue, rel=1e-9)


def test_pearson(data):
    table = run_case("pearson")[0].set_index("field")
    expected = np.corrcoef(data["unemployment"], data["inflation"])[0, 1]
    assert table.loc["unemployment", "inflation"] == pytest.approx(expected, abs=1e-4)


# Cronbach's alpha from its definition: k / (k - 1) * (1 - sum of item variances / variance of the total)
def test_cronbach_alpha():
    table = run_case("cronbach_alpha")[0]
    items = pd.read_csv(CASES["cronbach_alpha"][2])[CASES["cronbach_alpha"][3]["items"]].dropna()
    k = items.shape[1]
    expected = k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))
    assert table["alpha"].iloc[0] == pytest.approx(expected, rel=1e-9)
