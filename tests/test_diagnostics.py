import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from engine import clusters, diagnostics, restrictions
from engine.node import load_module
from tests.support import SAMPLE_DATA


# HAC standard errors, the regression diagnostics and the Wald and LR tests, each checked against
# the formula written out in numpy, as the textbooks give it, rather than against the code they use

DATA = pd.read_csv(SAMPLE_DATA)
ols = load_module("OLS")
diagnostics_module = load_module("Diagnostics")
wald_module = load_module("Wald Test")
SETTINGS = {"dependent": "unemployment", "independent": ["gdp_growth", "inflation"], "controls": ["bank_rate"]}
COLUMNS = ["gdp_growth", "inflation", "bank_rate"]


def design():
    x = np.column_stack([np.ones(len(DATA)), DATA[COLUMNS].to_numpy(float)])
    return DATA["unemployment"].to_numpy(float), x


def least_squares(y, x):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta, y - x @ beta


def r_squared(y, x):
    beta, resid = least_squares(y, x)
    return 1 - resid @ resid / np.sum((y - y.mean()) ** 2)


# ----- HAC -----

# Newey-West: the sandwich with Bartlett weights 1 - l / (L + 1), scaled by n / (n - k)
def newey_west(x, resid, lags):
    n, k = x.shape
    scores = x * resid[:, None]
    middle = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        cross = scores[lag:].T @ scores[:-lag]
        middle += weight * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    return bread @ middle @ bread * n / (n - k)


@pytest.mark.parametrize("lags", [None, 0, 3])
def test_hac_errors_match_newey_west(lags):
    results = ols.run([DATA], dict(SETTINGS, hac=True, hac_lags=lags))[0]
    y, x = design()
    beta, resid = least_squares(y, x)
    used = clusters.automatic_lags(len(y)) if lags is None else lags
    expected = np.sqrt(np.diag(newey_west(x, resid, used)))
    assert results.attrs["hac_lags"] == used
    assert np.allclose(results["std_error"].to_numpy(), expected, rtol=1e-10)
    assert clusters.errors_text(results.attrs) == f"Newey-West HAC ({used} lags)"


def test_automatic_lags_follow_newey_west_1994():
    assert clusters.automatic_lags(100) == 4
    assert clusters.automatic_lags(600) == int(4 * 6 ** (2 / 9))
    assert clusters.automatic_lags(5) == 2
    assert clusters.automatic_lags(1) == 1


def test_hac_and_clustering_cannot_both_be_chosen():
    with pytest.raises(ValueError, match="either"):
        ols.run([DATA], dict(SETTINGS, hac=True, cluster="region"))
    with pytest.raises(ValueError, match="between"):
        ols.run([DATA], dict(SETTINGS, hac=True, hac_lags=10_000))


def test_the_ols_panel_switches_hac_on_with_a_lags_box():
    from engine.node import Node
    node = Node("OLS", 0, 0, 1)
    node.settings = dict(SETTINGS)
    panel = node.on_click()
    panel.click(panel.button("HAC: Off"))
    assert node.settings["hac"] and panel.text_field is not None
    panel.text_field.on_change(panel, "4x")
    assert node.settings["hac_lags"] == 4


# ----- Diagnostics -----

def run_diagnostics(settings=None, model_settings=SETTINGS):
    model, fitted = ols.run([DATA], dict(model_settings))
    return diagnostics_module.run([model, fitted], dict(settings or {}))


def row_of(tests, name):
    return tests.set_index("test").loc[name]


def test_diagnostics_match_their_auxiliary_regressions():
    tests, vif, influence = run_diagnostics({"lags": 2, "break": "year >= 2008", "tests": diagnostics.TESTS})
    y, x = design()
    n = len(y)
    beta, resid = least_squares(y, x)
    fitted = x @ beta

    # Breusch-Pagan (Koenker's studentized form): n R squared of e^2 on X
    assert row_of(tests, "Breusch-Pagan")["statistic"] == pytest.approx(n * r_squared(resid ** 2, x))

    # White: n R squared of e^2 on X, its squares and cross products
    extra = [x[:, i] * x[:, j] for i in range(1, x.shape[1]) for j in range(i, x.shape[1])]
    white_x = np.column_stack([x] + extra)
    assert row_of(tests, "White")["statistic"] == pytest.approx(n * r_squared(resid ** 2, white_x))
    assert row_of(tests, "White")["df"] == str(white_x.shape[1] - 1)

    # RESET: F test that the squared and cubed fitted values add nothing
    restricted = resid @ resid
    unrestricted = least_squares(y, np.column_stack([x, fitted ** 2, fitted ** 3]))[1]
    df = n - x.shape[1] - 2
    f = ((restricted - unrestricted @ unrestricted) / 2) / (unrestricted @ unrestricted / df)
    assert row_of(tests, "RESET")["statistic"] == pytest.approx(f)

    # Breusch-Godfrey: n R squared of e on X and two lags of e, missing lags set to zero
    lagged = np.column_stack([np.r_[np.zeros(lag), resid[:-lag]] for lag in (1, 2)])
    assert row_of(tests, "Breusch-Godfrey")["statistic"] == pytest.approx(n * r_squared(resid, np.column_stack([x, lagged])))

    # Jarque-Bera from the skewness and kurtosis of the residuals
    skew, kurtosis = stats.skew(resid), stats.kurtosis(resid, fisher=False)
    assert row_of(tests, "Jarque-Bera")["statistic"] == pytest.approx(n / 6 * (skew ** 2 + (kurtosis - 3) ** 2 / 4))

    # Durbin-Watson
    assert row_of(tests, "Durbin-Watson")["statistic"] == pytest.approx(np.sum(np.diff(resid) ** 2) / (resid @ resid))

    # ARCH-LM: n' R squared of e^2 on two of its own lags
    squared = resid ** 2
    arch_y = squared[2:]
    arch_x = np.column_stack([np.ones(n - 2), squared[1:-1], squared[:-2]])
    assert row_of(tests, "ARCH-LM")["statistic"] == pytest.approx((n - 2) * r_squared(arch_y, arch_x))

    # Chow: pooled against separate fits either side of 2008
    after = DATA["year"].to_numpy() >= 2008
    parts = sum(least_squares(y[side], x[side])[1] @ least_squares(y[side], x[side])[1] for side in (after, ~after))
    k = x.shape[1]
    chow = ((restricted - parts) / k) / (parts / (n - 2 * k))
    assert row_of(tests, "Chow")["statistic"] == pytest.approx(chow)
    assert tests["result"].str.len().min() > 10


def test_vif_and_influence_match_their_formulas():
    tests, vif, influence = run_diagnostics()
    y, x = design()
    for index, name in enumerate(COLUMNS, start=1):
        others = np.delete(x, index, axis=1)
        expected = 1 / (1 - r_squared(x[:, index], others))
        assert vif.set_index("variable").loc[name, "vif"] == pytest.approx(expected)

    hat = x @ np.linalg.inv(x.T @ x) @ x.T
    assert np.allclose(influence["leverage"], np.diag(hat))
    beta, resid = least_squares(y, x)
    n, k = x.shape
    # The studentized residual of row 0 from refitting without it
    without = least_squares(y[1:], x[1:])
    s = np.sqrt(without[1] @ without[1] / (n - 1 - k))
    assert influence["studentized"].iloc[0] == pytest.approx(resid[0] / (s * np.sqrt(1 - hat[0, 0])))
    cooks = resid ** 2 / (k * resid @ resid / (n - k)) * np.diag(hat) / (1 - np.diag(hat)) ** 2
    assert np.allclose(influence["cooks_d"], cooks)


def test_diagnostics_skip_chow_without_a_break_and_explain():
    tests, vif, influence = run_diagnostics({"tests": ["Chow", "Jarque-Bera"]})
    assert tests["test"].tolist() == ["Jarque-Bera"]
    assert "Chow" in " ".join(tests.attrs["notes"])


def test_diagnostics_handle_text_regressors():
    settings = dict(SETTINGS, controls=["region"])
    tests, vif, influence = run_diagnostics(model_settings=settings)
    assert len(vif) == 2 + DATA["region"].nunique() - 1
    assert int(row_of(tests, "White")["df"]) < 100


def test_diagnostics_need_a_linear_model_and_its_own_data():
    logit = load_module("Logit")
    data = DATA.assign(high=(DATA["unemployment"] > 7.5).astype(int))
    model, fitted = logit.run([data], {"dependent": "high", "independent": ["gdp_growth"]})
    with pytest.raises(ValueError, match="linear"):
        diagnostics_module.run([model, fitted], {})
    model, fitted = ols.run([DATA], dict(SETTINGS))
    with pytest.raises(ValueError, match="missing"):
        diagnostics_module.run([model, fitted.drop(columns="bank_rate")], {})


def test_the_diagnostics_panel_draws_each_view():
    from engine import chart
    from engine.draw import Recorder, Rect
    from engine.node import Node
    node = Node("Diagnostics", 0, 0, 1)
    model, fitted = ols.run([DATA], dict(SETTINGS))
    node.last_inputs = [model, fitted]
    node.last_outputs = diagnostics_module.run(node.last_inputs, node.settings)
    panel = node.on_click()
    assert panel.preview is not None
    for _ in diagnostics_module.VIEWS[1:]:
        panel.click(panel.buttons[0])
        assert panel.chart is not None, node.settings["view"]
        recorder = Recorder()
        chart.render(recorder, Rect(0, 0, 700, 400), panel.chart)
        assert recorder.calls


# ----- Wald and LR tests -----

def test_wald_matches_statsmodels_with_robust_errors():
    model = ols.run([DATA], dict(SETTINGS, robust=True))[0]
    table = wald_module.run([model, None], {"restrictions": "gdp_growth = 0; inflation = 2 * bank_rate - 1"})[0]
    x = sm.add_constant(DATA[COLUMNS])
    reference = sm.OLS(DATA["unemployment"], x).fit(cov_type="HC1", use_t=True)
    expected = reference.f_test("gdp_growth = 0, inflation - 2 * bank_rate = -1")
    joint = table.iloc[0]
    assert joint["statistic"] == pytest.approx(float(np.squeeze(expected.fvalue)))
    assert joint["p_value"] == pytest.approx(float(expected.pvalue))
    assert joint["distribution"] == f"F(2, {len(DATA) - 4})"
    single = table.iloc[2]
    assert single["estimate"] == pytest.approx(reference.params["inflation"] - 2 * reference.params["bank_rate"] + 1)


def test_wald_on_a_logit_uses_chi_squared():
    logit = load_module("Logit")
    data = DATA.assign(high=(DATA["unemployment"] > 7.5).astype(int))
    model = logit.run([data], {"dependent": "high", "independent": ["gdp_growth", "inflation"]})[0]
    table = wald_module.run([model, None], {"restrictions": "gdp_growth = 0"})[0]
    z = model.set_index("variable").loc["gdp_growth", "z_stat"]
    assert table.iloc[0]["distribution"] == "chi2(1)"
    assert table.iloc[0]["statistic"] == pytest.approx(z ** 2)


def test_lr_compares_nested_models():
    full = ols.run([DATA], dict(SETTINGS))[0]
    smaller = ols.run([DATA], dict(SETTINGS, controls=[]))[0]
    table = wald_module.run([full, smaller], {"restrictions": ""})[0]
    llf = lambda columns: sm.OLS(DATA["unemployment"], sm.add_constant(DATA[columns])).fit().llf
    expected = 2 * (llf(COLUMNS) - llf(COLUMNS[:2]))
    assert table.iloc[0]["test"] == "LR"
    assert table.iloc[0]["statistic"] == pytest.approx(expected)
    assert table.iloc[0]["p_value"] == pytest.approx(stats.chi2.sf(expected, 1))
    with pytest.raises(ValueError, match="fewer"):
        wald_module.run([smaller, full], {"restrictions": ""})


@pytest.mark.parametrize("text, message", [
    ("gdp_growth * inflation = 0", "multiply"),
    ("unknown = 0", "no coefficient"),
    ("gdp_growth = 0, gdp_growth = 0", "repeat"),
    ("3 = 3", "doesn't involve"),
    ("gdp_growth = = 1", "more than one"),
])
def test_bad_restrictions_are_explained(text, message):
    known = ["Intercept", "gdp_growth", "inflation", "region=North West"]
    with pytest.raises(ValueError, match=message):
        restrictions.parse(text, known)


def test_restrictions_read_backticks_and_the_constant():
    known = ["Intercept", "gdp_growth", "region=North West"]
    R, q, shown = restrictions.parse("`region=North West` + const = 1, -gdp_growth/2", known)
    assert R.tolist() == [[1, 0, 1], [0, -0.5, 0]]
    assert q.tolist() == [1, 0]
    assert shown[1].endswith("= 0")
