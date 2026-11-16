import importlib
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

from engine import runner, workspace
from engine.node import Node
from tests.support import ROOT, SAMPLE_DATA
from tests.test_panels import exercise


# Checks the estimators added for gretl parity against values worked out another way: published
# results, statsmodels' own estimators, formulas written out in numpy, or simulated data whose
# true values are known


def core(name):
    return importlib.import_module(f"modules.{name}.module")


def coefficient(table, variable, column="coefficient"):
    return float(table.loc[table["variable"] == variable, column].iloc[0])


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE_DATA)


@pytest.fixture(scope="module")
def macro():
    d = sm.datasets.macrodata.load_pandas().data
    return pd.DataFrame({
        "t": d.year + (d.quarter - 1) / 4,
        "gdp": d.realgdp.pct_change() * 100, "cons": d.realcons.pct_change() * 100, "infl": d.infl,
        "lgdp": np.log(d.realgdp), "lcons": np.log(d.realcons), "linv": np.log(d.realinv),
    }).iloc[1:].reset_index(drop=True)


# Count models

def count_data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(list("abc"), n), "exposure": rng.uniform(1, 3, n)})
    mu = np.exp(0.3 + 0.5 * data["x"]) * data["exposure"]
    data["y"] = rng.poisson(mu)
    return data


def test_poisson_matches_a_glm():
    data = count_data()
    table = core("count").run([data], {"dependent": "y", "independent": ["x", "g"], "exposure": "exposure", "model": "poisson"})[0]
    x = sm.add_constant(pd.get_dummies(data[["x", "g"]], columns=["g"], drop_first=True, dtype=float))
    glm = sm.GLM(data["y"], x, family=sm.families.Poisson(), offset=np.log(data["exposure"])).fit()
    assert table["coefficient"].to_numpy() == pytest.approx(glm.params.to_numpy(), rel=1e-5)
    assert table["std_error"].to_numpy() == pytest.approx(glm.bse.to_numpy(), rel=1e-4)
    assert table["irr"].iloc[1] == pytest.approx(np.exp(glm.params["x"]), rel=1e-5)
    assert table.attrs["fit"]["kind"] == "count"


def test_negative_binomial_finds_overdispersion():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    mu = np.exp(0.5 + 0.4 * x)
    data = pd.DataFrame({"x": x, "y": rng.negative_binomial(2, 2 / (2 + mu))})
    poisson = core("count").run([data], {"dependent": "y", "independent": ["x"], "model": "poisson"})[0]
    negbin = core("count").run([data], {"dependent": "y", "independent": ["x"], "model": "negbin"})[0]
    assert poisson.attrs["overdispersion_p"] < 0.01
    assert negbin.attrs["alpha_lr_p"] < 0.01
    assert coefficient(negbin, "alpha") == pytest.approx(0.5, abs=0.1)
    assert coefficient(negbin, "x") == pytest.approx(0.4, abs=0.05)


def test_zero_inflated_poisson_runs_with_inflate_fields():
    data = count_data(2)
    data.loc[data.index % 4 == 0, "y"] = 0
    table, fitted = core("count").run([data], {"dependent": "y", "independent": ["x"], "inflate": ["g"], "model": "zip"})
    assert (table["role"] == "Inflate").sum() == 3
    assert fitted["predicted"].notna().all()


def test_count_needs_counts(sample):
    with pytest.raises(ValueError, match="count"):
        core("count").run([sample], {"dependent": "inflation", "independent": ["gdp_growth"]})


# Tobit

def test_tobit_with_nothing_censored_is_ols(sample):
    table = core("tobit").run([sample], {"dependent": "unemployment", "independent": ["gdp_growth", "inflation"], "limits": "-100,"})[0]
    ols = sm.OLS(sample["unemployment"], sm.add_constant(sample[["gdp_growth", "inflation"]])).fit()
    assert table["coefficient"].iloc[:3].to_numpy() == pytest.approx(ols.params.to_numpy(), rel=1e-4)
    assert coefficient(table, "sigma") == pytest.approx(np.sqrt(ols.ssr / ols.nobs), rel=1e-4)


def test_tobit_recovers_a_censored_model():
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    latent = 0.5 + 2 * x + rng.normal(scale=1.5, size=4000)
    data = pd.DataFrame({"x": x, "y": np.clip(latent, 0, 4)})
    table, fitted = core("tobit").run([data], {"dependent": "y", "independent": ["x"], "limits": "0, 4"})
    assert coefficient(table, "x") == pytest.approx(2, abs=0.08)
    assert coefficient(table, "sigma") == pytest.approx(1.5, abs=0.06)
    assert table.attrs["censored_below"] > 0 and table.attrs["censored_above"] > 0
    assert fitted["expected"].between(0, 4).all()


# Heckman

def heckman_data(seed=4, n=3000):
    rng = np.random.default_rng(seed)
    x, w = rng.normal(size=n), rng.normal(size=n)
    u, v = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n).T
    selected = 0.5 + w + 0.5 * x + v > 0
    return pd.DataFrame({"x": x, "w": w, "y": np.where(selected, 1 + 2 * x + u, np.nan)})


def test_heckman_two_step_matches_probit_then_ols():
    data = heckman_data()
    table = core("heckman").run([data], {"dependent": "y", "independent": ["x"], "selection": ["x", "w"], "method": "two_step"})[0]
    z = sm.add_constant(data[["x", "w"]])
    chosen = data["y"].notna()
    probit = sm.Probit(chosen.astype(float), z).fit(disp=0)
    index = z @ probit.params
    mills = stats.norm.pdf(index) / stats.norm.cdf(index)
    second = sm.OLS(data["y"][chosen], np.column_stack([np.ones(chosen.sum()), data["x"][chosen], mills[chosen]])).fit()
    assert coefficient(table, "x") == pytest.approx(second.params.iloc[1], rel=1e-8)
    assert coefficient(table, "lambda") == pytest.approx(second.params.iloc[2], rel=1e-8)
    assert coefficient(table, "select: w") == pytest.approx(probit.params["w"], rel=1e-6)


def test_heckman_ml_recovers_rho():
    table = core("heckman").run([heckman_data()], {"dependent": "y", "independent": ["x"], "selection": ["x", "w"], "method": "ml"})[0]
    assert coefficient(table, "x") == pytest.approx(2, abs=0.05)
    assert coefficient(table, "rho") == pytest.approx(0.6, abs=0.1)
    assert table.attrs["notes"] == []


# Quantile regression: Koenker's median regression of food spending on income (Engel data), as
# given by R's quantreg (rq(foodexp ~ income, tau = 0.5))
def test_quantile_matches_koenker_engel():
    engel = sm.datasets.engel.load_pandas().data
    table, process, fitted = core("quantile_reg").run([engel], {"dependent": "foodexp", "independent": ["income"], "quantiles": "0.5, 0.25"})
    assert coefficient(table, "Intercept") == pytest.approx(81.48225, rel=1e-5)
    assert coefficient(table, "income") == pytest.approx(0.5601806, rel=1e-5)
    assert sorted(process["quantile"].unique()) == [0.25, 0.5]
    assert "fitted_q0.25" in fitted


def test_quantile_text_is_checked():
    with pytest.raises(ValueError):
        core("quantile_reg").parse_quantiles("0.5, half")
    assert core("quantile_reg").parse_quantiles("25%, 0.5") == [0.25, 0.5]


# VAR, cointegration and VECM

def test_var_is_ols_equation_by_equation(macro):
    table, impulses, decomposition, tests = core("var").run(
        [macro], {"dependent": ["gdp", "cons", "infl"], "time": "t", "selection": "fixed", "lags": 2})
    y = macro[["gdp", "cons", "infl"]].to_numpy()
    x = np.column_stack([np.ones(len(y) - 2), y[1:-1], y[:-2]])
    beta = np.linalg.lstsq(x, y[2:], rcond=None)[0]
    gdp = table[table["equation"] == "gdp"]["coefficient"].to_numpy()
    assert gdp == pytest.approx(beta[:, 0], rel=1e-8)
    # At horizon 0 orthogonal responses are the Cholesky factor of the residuals' covariance
    residuals = y[2:] - x @ beta
    sigma = residuals.T @ residuals / (len(residuals) - x.shape[1])
    first = impulses[impulses["horizon"] == 0]
    response = first.loc[(first["impulse"] == "gdp") & (first["response"] == "cons"), "value"].iloc[0]
    assert response == pytest.approx(np.linalg.cholesky(sigma)[1, 0], rel=1e-6)
    assert decomposition.filter(like="share_").sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert (tests["test"] == "Granger causality").sum() == 6


def cointegrated(seed=6, n=400):
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(size=n))
    return pd.DataFrame({"t": np.arange(n), "a": trend + rng.normal(scale=0.5, size=n), "b": trend + rng.normal(scale=0.5, size=n),
                         "c": np.cumsum(rng.normal(size=n))})


def test_johansen_and_engle_granger_find_the_relation():
    data = cointegrated()
    tests, vectors = core("cointegration").run([data], {"dependent": ["a", "b", "c"], "time": "t", "method": "johansen"})
    assert tests.attrs["rank"] == 1
    assert vectors["vector_1"].to_numpy()[:2] == pytest.approx([1, -1], abs=0.1)
    tests, _ = core("cointegration").run([data], {"dependent": ["a", "b"], "time": "t", "method": "engle_granger"})
    assert tests["p_value"].iloc[0] < 0.01


def test_vecm_recovers_the_long_run_relation():
    data = cointegrated()
    table, vectors, rank_tests, impulses = core("vecm").run([data], {"dependent": ["a", "b", "c"], "time": "t", "lags": 1})
    assert table.attrs["rank"] == 1
    b = vectors.loc[vectors["variable"] == "b", "coefficient"].iloc[0]
    assert b == pytest.approx(-1, abs=0.05)
    with pytest.raises(ValueError, match="no cointegration"):
        core("vecm").run([data], {"dependent": ["a", "c"], "time": "t", "lags": 1})


# GARCH: the log-likelihood reported is the one a plain loop over the periods gives at the estimates

def garch_data(seed=7, n=1500):
    rng = np.random.default_rng(seed)
    e, h = np.zeros(n), np.ones(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.normal()
    return pd.DataFrame({"t": np.arange(n), "r": 0.05 + e})


def test_garch_recovers_and_its_likelihood_checks_out():
    data = garch_data()
    table, fitted, forecast = core("garch").run([data], {"dependent": "r", "time": "t"})
    mu, omega, alpha, beta = table["coefficient"].to_numpy()
    assert alpha == pytest.approx(0.1, abs=0.04)
    assert beta == pytest.approx(0.85, abs=0.06)
    e = data["r"].to_numpy() - mu
    # The presample variance is fixed before the fit, from the residuals around Y's mean
    backcast = core("garch").backcast_of(data["r"].to_numpy() - data["r"].mean())
    h, total = np.empty(len(e)), 0.0
    for t in range(len(e)):
        h[t] = omega + alpha * (e[t - 1] ** 2 if t else backcast) + beta * (h[t - 1] if t else backcast)
        total += stats.norm.logpdf(e[t], scale=np.sqrt(h[t]))
    assert table.attrs["log_likelihood"] == pytest.approx(total, rel=1e-8)
    assert fitted["volatility"].to_numpy() == pytest.approx(np.sqrt(h), rel=1e-8)
    long_run = omega / (1 - alpha - beta)
    assert abs(forecast["variance"].iloc[-1] - long_run) < abs(forecast["variance"].iloc[0] - long_run) + 1e-9


@pytest.mark.parametrize("model", ["gjr", "egarch"])
def test_garch_variants_run(model):
    table, fitted, forecast = core("garch").run([garch_data()], {"dependent": "r", "time": "t", "model": model, "distribution": "t"})
    assert "gamma1" in list(table["variable"]) and "nu" in list(table["variable"])
    assert np.isfinite(table["std_error"]).all()


# SARIMA through ARIMAX, and Forecast carrying it on

def test_seasonal_arimax_matches_sarimax():
    co2 = sm.datasets.co2.load_pandas().data.resample("MS").mean().ffill().iloc[-180:]
    data = pd.DataFrame({"date": co2.index, "co2": co2["co2"].to_numpy()})
    settings = {"dependent": "co2", "time": "date", "p": 1, "d": 1, "q": 0, "s": 12, "P": 0, "D": 1, "Q": 1, "constant": False}
    table, fitted = core("arimax").run([data.iloc[:-12]], settings)
    direct = SARIMAX(data["co2"].iloc[:-12].to_numpy(), order=(1, 1, 0), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    assert table["coefficient"].to_numpy() == pytest.approx(direct.params, rel=1e-3)
    assert list(table["role"]) == ["AR", "SMA", "Variance"]
    output, kind, fit, notes = core("forecast").forecast([table, data.iloc[-12:]], {})
    expected = direct.get_forecast(12).predicted_mean
    assert output["forecast"].to_numpy() == pytest.approx(expected, rel=1e-4)


# Filters

def test_filters_match_their_definitions(macro):
    from statsmodels.tsa.filters.hp_filter import hpfilter
    hp = core("ts_filter").run([macro], {"dependent": ["lgdp"], "time": "t", "method": "hp"})[0]
    assert hp["lgdp_trend"].to_numpy() == pytest.approx(hpfilter(macro["lgdp"], 1600)[1].to_numpy())
    hamilton = core("ts_filter").run([macro], {"dependent": ["lgdp"], "time": "t", "method": "hamilton"})[0]
    y = macro["lgdp"].to_numpy()
    x = np.column_stack([np.ones(len(y) - 11)] + [y[3 - j:len(y) - 8 - j] for j in range(4)])
    beta = np.linalg.lstsq(x, y[11:], rcond=None)[0]
    assert hamilton["lgdp_trend"].to_numpy()[11:] == pytest.approx(x @ beta)
    assert hamilton["lgdp_trend"].iloc[:11].isna().all()
    bk = core("ts_filter").run([macro], {"dependent": ["lgdp"], "time": "t", "method": "bk"})[0]
    assert bk["lgdp_cycle"].iloc[:12].isna().all() and bk["lgdp_cycle"].iloc[12:-12].notna().all()
    stl = core("ts_filter").run([macro], {"dependent": ["lgdp"], "time": "t", "method": "stl"})[0]
    assert (stl["lgdp_adjusted"] + stl["lgdp_seasonal"]).to_numpy() == pytest.approx(y)


def test_filters_run_within_each_group(sample):
    output = core("ts_filter").run([sample], {"dependent": ["unemployment"], "time": "year", "group": "region",
                                              "method": "hp", "frequency": "annual", "parameters": "lambda=100"})[0]
    one = output[output["region"] == "London"]
    from statsmodels.tsa.filters.hp_filter import hpfilter
    alone = hpfilter(sample[sample["region"] == "London"].sort_values("year")["unemployment"], 100)[1]
    assert one["unemployment_trend"].to_numpy() == pytest.approx(alone.to_numpy())


# Dynamic panel GMM

def dynamic_panel(seed=8, n=600, periods=7, rho=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        eta = rng.normal()
        y = eta / (1 - rho) + rng.normal()
        x = rng.normal()
        for t in range(periods):
            x = 0.5 * x + rng.normal()
            y = rho * y + x + eta + rng.normal()
            rows.append((i, t, y, x))
    return pd.DataFrame(rows, columns=["id", "year", "y", "x"])


# With three periods there is one differenced equation, instrumented by Y two periods back, so
# difference GMM is the Anderson-Hsiao IV estimator, which has a closed form
def test_difference_gmm_is_anderson_hsiao_when_just_identified():
    data = dynamic_panel(periods=3)
    table, tests = core("dynamic_panel").run([data], {"dependent": "y", "entity": "id", "time": "year", "estimator": "difference",
                                                      "steps": "one", "time_effects": False})
    wide = data.pivot(index="id", columns="year", values="y").to_numpy()
    rho = np.sum((wide[:, 2] - wide[:, 1]) * wide[:, 0]) / np.sum((wide[:, 1] - wide[:, 0]) * wide[:, 0])
    assert coefficient(table, "L1.y") == pytest.approx(rho, rel=1e-8)


@pytest.mark.parametrize("estimator", ["difference", "system"])
def test_dynamic_panel_recovers_rho(estimator):
    table, tests = core("dynamic_panel").run([dynamic_panel()], {"dependent": "y", "independent": ["x"], "entity": "id", "time": "year",
                                                                 "estimator": estimator, "steps": "two", "time_effects": False})
    assert coefficient(table, "L1.y") == pytest.approx(0.5, abs=0.06)
    assert coefficient(table, "x") == pytest.approx(1.0, abs=0.05)
    ar = tests.set_index("test")["p_value"]
    assert ar["AR(1)"] < 0.01 and ar["AR(2)"] > 0.01 and ar["Hansen J"] > 0.01


# Bootstrap refits the upstream model, so it is run through a workflow, which records the source

def test_bootstrap_through_a_workflow(tmp_path):
    workflow, view, problems = workspace.load(ROOT / "sample_workflows" / "10_time_series_and_panel_gmm.confw", stamp=False)
    results, errors = runner.run_workflow(workflow)
    assert errors == {}
    boot = results[10][0]
    assert list(boot["variable"]) == ["Intercept", "gdp_growth", "inflation"]
    assert (boot["boot_std_error"] > 0).all() and (boot["lower"] < boot["coefficient"]).all()
    again, _ = runner.run_workflow(workflow, force=True)
    assert again[10][0]["boot_std_error"].to_numpy() == pytest.approx(boot["boot_std_error"].to_numpy())


def test_bootstrap_refuses_a_model_without_its_source(sample):
    model = core("ols").run([sample], {"dependent": "unemployment", "independent": ["gdp_growth"]})[0]
    with pytest.raises(ValueError, match="which module"):
        core("bootstrap").run([model, sample], {"method": "pairs"})
    model.attrs["source"] = {"module": "OLS", "settings": {"dependent": "unemployment", "independent": ["gdp_growth"]}}
    with pytest.raises(ValueError, match="same data"):
        core("bootstrap").run([model, sample.iloc[:300]], {"method": "pairs"})
    permutation = core("bootstrap").run([model, sample], {"method": "permutation", "replications": 199})[0]
    assert permutation["p_value"].between(1 / 200, 1).all()


# Restriction tests on the new models

def test_wald_names_equations_and_refuses_joint_vecm_tests(macro):
    var = core("var").run([macro], {"dependent": ["gdp", "cons"], "time": "t", "selection": "fixed", "lags": 1})[0]
    names = core("wald_test").names_of(var)
    assert "gdp: L1.cons" in names
    wald = core("wald_test").run([var], {"restrictions": "`gdp: L1.cons` = 0"})[0]
    t = var.loc[(var["equation"] == "gdp") & (var["variable"] == "L1.cons"), "t_stat"].iloc[0]
    assert wald["statistic"].iloc[0] == pytest.approx(t ** 2, rel=1e-8)
    vecm = core("vecm").run([cointegrated()], {"dependent": ["a", "b", "c"], "time": "t", "lags": 1})[0]
    core("wald_test").run([vecm], {"restrictions": "`a: ec1` = 0"})
    with pytest.raises(ValueError, match="one coefficient at a time"):
        core("wald_test").run([vecm], {"restrictions": "`a: ec1` = `b: ec1`"})


# Forecast with the new models

def test_forecast_with_count_tobit_and_quantile_models():
    data = count_data()
    count = core("count").run([data], {"dependent": "y", "independent": ["x", "g"], "exposure": "exposure"})[0]
    output, kind, fit, notes = core("forecast").forecast([count, data], {})
    assert kind == "count"
    fitted = core("count").run([data], {"dependent": "y", "independent": ["x", "g"], "exposure": "exposure"})[1]
    assert output["forecast"].to_numpy() == pytest.approx(fitted["predicted"].to_numpy(), rel=1e-8)

    rng = np.random.default_rng(9)
    censored = pd.DataFrame({"x": rng.normal(size=500)})
    censored["y"] = np.maximum(1 + censored["x"] + rng.normal(size=500), 0)
    tobit, tobit_fitted = core("tobit").run([censored], {"dependent": "y", "independent": ["x"], "limits": "0,"})
    output = core("forecast").forecast([tobit, censored], {})[0]
    assert output["forecast"].to_numpy() == pytest.approx(tobit_fitted["expected"].to_numpy())

    engel = sm.datasets.engel.load_pandas().data
    quantile, _, fitted = core("quantile_reg").run([engel], {"dependent": "foodexp", "independent": ["income"], "quantiles": "0.9"})
    output = core("forecast").forecast([quantile, engel], {})[0]
    assert output["forecast"].to_numpy() == pytest.approx(fitted["fitted_q0.9"].to_numpy())


def test_forecast_refuses_garch():
    table = core("garch").run([garch_data()], {"dependent": "r", "time": "t"})[0]
    with pytest.raises(ValueError, match="GARCH"):
        core("forecast").forecast([table, garch_data()], {})


# Online data: each source's answer is read correctly, with the download itself replaced

FRED_CSV = "observation_date,UNRATE\n2020-01-01,3.6\n2020-02-01,.\n2020-03-01,4.4\n"
WORLD_BANK = json.dumps([{"page": 1}, [
    {"indicator": {"id": "X", "value": "GDP growth"}, "country": {"id": "GB", "value": "United Kingdom"}, "countryiso3code": "GBR",
     "date": "2020", "value": -10.0},
    {"indicator": {"id": "X", "value": "GDP growth"}, "country": {"id": "GB", "value": "United Kingdom"}, "countryiso3code": "GBR",
     "date": "2019", "value": 1.3},
    {"indicator": {"id": "X", "value": "GDP growth"}, "country": {"id": "US", "value": "United States"}, "countryiso3code": "USA",
     "date": "2019", "value": None},
]])
OECD_CSV = ("STRUCTURE,STRUCTURE_ID,STRUCTURE_NAME,ACTION,REF_AREA,Reference area,FREQ,Frequency,TIME_PERIOD,Time period,OBS_VALUE,Observation value\n"
            "DATAFLOW,X,Leading indicators,I,GBR,United Kingdom,M,Monthly,2023-01,,96.1,\n"
            "DATAFLOW,X,Leading indicators,I,USA,United States,M,Monthly,2023-01,,98.7,\n")
DBNOMICS = json.dumps({"series": {"docs": [{"provider_code": "AMECO", "dataset_code": "ZUTN", "series_code": "EA19.1",
                                            "series_name": "Unemployment", "period": ["2019", "2020"], "value": [7.6, "NA"]}]}})


@pytest.mark.parametrize("source, query, answer, rows", [
    ("fred", "UNRATE; from=2020-01-01", FRED_CSV, 3),
    ("worldbank", "X; countries=GBR,USA", WORLD_BANK, 3),
    ("oecd", "FLOW/GBR+USA.M; from=2023-01", OECD_CSV, 2),
    ("dbnomics", "AMECO/ZUTN/EA19.1", DBNOMICS, 2),
])
def test_online_sources_are_read(monkeypatch, source, query, answer, rows):
    module = core("online_data")
    asked = []
    monkeypatch.setattr(module, "fetch", lambda url: asked.append(url) or answer)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    table = module.run([], {"source": source, "query": query})[0]
    assert len(table) == rows
    assert table["value"].isna().sum() == 1 if source in ("fred", "worldbank", "dbnomics") else table["value"].notna().all()
    assert len(asked) == 1
    wide = module.run([], {"source": source, "query": query, "shape": "wide"})[0]
    assert "period" in wide


def test_online_query_is_checked():
    module = core("online_data")
    assert module.parse_query("A, B; from=2000; to=2010") == (["A", "B"], {"from": "2000", "to": "2010"})
    assert module.parse_query("FLOW,X,/KEY OTHER/KEY", spaces=True)[0] == ["FLOW,X,/KEY", "OTHER/KEY"]
    with pytest.raises(ValueError):
        module.parse_query("A; 2000")


# Every new module's panel opens after a run and every button works

PANEL_CASES = {
    "Count": (count_data, {"dependent": "y", "independent": ["x"], "inflate": ["g"]}),
    "Tobit": (lambda: heckman_data().fillna(0), {"dependent": "y", "independent": ["x"], "limits": "0,"}),
    "Heckman": (heckman_data, {"dependent": "y", "independent": ["x"], "selection": ["x", "w"]}),
    "Quantile Reg": (lambda: sm.datasets.engel.load_pandas().data, {"dependent": "foodexp", "independent": ["income"]}),
    "GARCH": (garch_data, {"dependent": "r", "time": "t"}),
    "VECM": (cointegrated, {"dependent": ["a", "b", "c"], "time": "t"}),
    "Cointegration": (cointegrated, {"dependent": ["a", "b"], "time": "t"}),
    "Dynamic Panel": (lambda: dynamic_panel(n=60), {"dependent": "y", "independent": ["x"], "entity": "id", "time": "year", "collapse": True}),
}


@pytest.mark.parametrize("name", sorted(PANEL_CASES))
def test_new_panels_after_a_run(name):
    make, settings = PANEL_CASES[name]
    node = Node(name, 0, 0, 1)
    node.settings = dict(settings)
    node.last_inputs = [make()]
    node.last_outputs = node.module.run(node.last_inputs, node.settings)
    node.status = "done"
    exercise(node)
