import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from engine import estimates, roles
from modules.var.module import series_data


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a time series table with two or more numeric Y fields in levels (not differenced), such as log prices or "
             "log output and consumption, a time field (T), and a group field (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Tests", "Johansen: one row per rank r with the trace and maximum eigenvalue statistics, their 90%, 95% and 99% critical "
              "values and whether rank r is rejected. Engle-Granger: the ADF test on the residuals of the cointegrating regression"),
    ("Cointegrating vectors", "Johansen: one column per vector (normalised so the first Y has coefficient 1), one row per Y. "
                              "Engle-Granger: the cointegrating regression's coefficients"),
]

METHODS = [("johansen", "Johansen"), ("engle_granger", "Engle-Granger")]
# Johansen's det_order: -1 no deterministic terms, 0 a constant, 1 a linear trend
DETERMINISTIC = [("0", "Constant"), ("1", "Trend"), ("-1", "None")]
EG_TRENDS = {"0": "c", "1": "ct", "-1": "n"}
SPEC = roles.Spec(
    "Cointegration",
    [
        roles.Role("Y", "dependent", many=True),
        roles.Role("T", "time", kinds="any"),
        roles.Role("G", "group", kinds="any"),
    ],
    choices=[roles.Choice("method", "Test", METHODS), roles.Choice("deterministic", "Terms", DETERMINISTIC)],
    numbers=[roles.Number("lags", "Lags (in differences)", range(0, 9), 1)],
    series=True,
    hint="Choose two or more Y fields in levels and the time field (T), then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings()


def on_click(node):
    return roles.open_panel(node, SPEC)


# Johansen's trace and maximum eigenvalue tests of the cointegrating rank, one row per rank r
# tested (the null being at most r vectors), and the rank both tests settle on at 5%: the first
# r whose null isn't rejected
def johansen_table(y, det_order, lags):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = coint_johansen(y.to_numpy(float), det_order, lags)
    k = y.shape[1]
    rows = []
    for r in range(k):
        trace, trace_cv = result.lr1[r], result.cvt[r]
        eigen, eigen_cv = result.lr2[r], result.cvm[r]
        rows.append({
            "rank": r, "null": f"at most {r} cointegrating vectors" if r else "no cointegration",
            "eigenvalue": result.eig[r],
            "trace": trace, "trace_cv90": trace_cv[0], "trace_cv95": trace_cv[1], "trace_cv99": trace_cv[2],
            "max_eigen": eigen, "max_cv90": eigen_cv[0], "max_cv95": eigen_cv[1], "max_cv99": eigen_cv[2],
            "trace_rejects_5%": bool(trace > trace_cv[1]), "max_rejects_5%": bool(eigen > eigen_cv[1]),
        })
    table = pd.DataFrame(rows)
    rank = next((r for r in range(k) if not table["trace_rejects_5%"].iloc[r]), k)
    vectors = result.evec[:, :max(rank, 1)]
    vectors = vectors / vectors[0]
    names = [f"vector_{i + 1}" for i in range(vectors.shape[1])]
    vector_table = pd.DataFrame(vectors, columns=names)
    vector_table.insert(0, "variable", list(y.columns))
    return table, rank, vector_table


def rank_text(rank, k):
    if rank == 0:
        return "No cointegration at 5% (trace test): the fields drift apart, so model their differences with a VAR"
    if rank == k:
        return f"Rank {k} = the number of fields: they all look stationary, so a VAR in levels will do"
    return f"{rank} cointegrating vector{'s' if rank > 1 else ''} at 5% (trace test): the fields share a long-run relationship, so use a VECM"


# Engle and Granger's two-step test: regresses the first Y on the others, then tests the residuals
# for a unit root with MacKinnon's critical values. Cointegrated fields have stationary residuals
def engle_granger(y, det_order, lags):
    trend = EG_TRENDS[str(det_order)]
    first, others = y.columns[0], list(y.columns[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value, critical = coint(y[first], y[others], trend=trend, maxlag=lags, autolag=None if lags else "aic")
    x = y[others]
    if trend in ("c", "ct"):
        x = sm.add_constant(x, has_constant="add")
    if trend == "ct":
        x["trend"] = np.arange(1, len(y) + 1)
    regression = sm.OLS(y[first], x).fit()
    tests = pd.DataFrame([{
        "test": "Engle-Granger", "null": "no cointegration (the residuals have a unit root)", "statistic": statistic,
        "cv1": critical[0], "cv5": critical[1], "cv10": critical[2], "p_value": p_value,
        "result": "Cointegrated at 5%" if p_value < 0.05 else "No evidence of cointegration at 5%",
    }])
    vectors = estimates.coefficient_table(
        ["Intercept" if n == "const" else n for n in regression.params.index], ["Long run"] * len(regression.params),
        regression.params.values, regression.bse.values, df=regression.df_resid,
    )
    vectors.attrs["note"] = "Standard errors of a cointegrating regression aren't valid for t tests"
    return tests, vectors, p_value


# Tests whether trending (non-stationary) fields share long-run relationships that keep them
# from drifting apart, as prices of the same good in two markets do
def run(inputs, settings):
    SPEC.defaults(settings)
    notes = []
    y, x, times, notes = series_data(inputs, {**settings, "independent": []}, notes)
    k = y.shape[1]
    det_order = int(settings["deterministic"])
    lags = settings["lags"]
    if len(y) < 4 * (k * (lags + 1) + 2):
        notes.append("The series is short for this many fields and lags, so the tests have little power.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stationary = [c for c in y.columns if adfuller(y[c], autolag="AIC")[1] < 0.05]
    if stationary:
        notes.append(f"Already stationary in levels by ADF at 5%: {', '.join(stationary)}. Cointegration is about fields that are not.")

    if settings["method"] == "johansen":
        tests, rank, vectors = johansen_table(y, det_order, lags)
        summary = [f"Johansen    {k} fields    N: {len(y)}    Lags in differences: {lags}", rank_text(rank, k)]
        attrs = {"rank": rank}
    else:
        tests, vectors, p_value = engle_granger(y, det_order, lags)
        summary = [f"Engle-Granger    {y.columns[0]} on {', '.join(y.columns[1:])}    N: {len(y)}",
                   f"ADF on the residuals: p = {p_value:.3g}: " + ("cointegrated at 5%" if p_value < 0.05 else "no evidence of cointegration at 5%")]
        attrs = {"p_value": float(p_value)}
        notes.append(vectors.attrs.pop("note"))
    tests.attrs = {**attrs, "observations": len(y), "summary": summary, "notes": notes}
    vectors.attrs.update(summary=summary, notes=notes)
    return [tests, vectors]
