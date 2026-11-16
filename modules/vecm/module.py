import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import VECM

from engine import draw, estimates, roles
from engine.chart import draw_lines
from modules.cointegration.module import johansen_table, rank_text
from modules.var.module import series_data


INPUTS = 1
OUTPUTS = 4
OUTPUT_TYPES = ["model", "table", "table", "table"]
INPUT_SPECS = [
    ("Data", "a time series table with two or more numeric Y fields in levels that are cointegrated (see Cointegration), "
             "a time field (T), and a group field (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per equation and term: the adjustment (alpha) to each cointegrating relation ec1, ec2..., the "
                     "short-run terms on lagged differences (e.g. L1.d_inflation) and any deterministic terms, with std_error, "
                     "z_stat, p_value and stars"),
    ("Cointegrating vectors", "the long-run relations (beta), one row per Y and relation, normalised so the first fields have 1, "
                              "with std_error, z_stat and p_value"),
    ("Rank test", "Johansen's trace and maximum eigenvalue tests of the number of cointegrating relations"),
    ("Impulse responses", "one row per horizon, shock and response with the response of each Y's level to a one standard "
                          "deviation orthogonal shock"),
]

# statsmodels' deterministic terms: ci a constant inside the cointegrating relation, co one outside
# it (a drift in the levels), li and lo the same for a linear trend, n none
DETERMINISTIC = [("ci", "Constant in relation"), ("co", "Constant (drift)"), ("cili", "Constant + trend in relation"), ("n", "None")]
JOHANSEN_ORDER = {"ci": 0, "co": 0, "cili": 1, "n": -1}
RANKS = [("auto", "Johansen"), (1, "1"), (2, "2"), (3, "3")]


def shock_label(settings):
    return f"Shock: {settings.get('shock') or 'first Y'}"


def next_shock(panel, button):
    roles.step(panel, "shock", panel.node.settings.get("dependent") or [])
    button.text = shock_label(panel.node.settings)


def response_chart(node):
    if len(node.last_outputs) < 4:
        return None
    table = node.last_outputs[3]
    shocks = list(dict.fromkeys(table["impulse"]))
    if not shocks:
        return None
    shock = node.settings.get("shock") if node.settings.get("shock") in shocks else shocks[0]
    rows = table[table["impulse"] == shock]
    series = [(f"{name} to {shock}", part["horizon"].to_numpy(float), part["value"].to_numpy(float))
              for name, part in rows.groupby("response", sort=False)]

    def chart(painter, area):
        draw_lines(painter, area, draw.font(22), series, "periods after the shock")
    return chart


SPEC = roles.Spec(
    "VECM",
    [
        roles.Role("Y", "dependent", many=True),
        roles.Role("T", "time", kinds="any"),
        roles.Role("G", "group", kinds="any"),
    ],
    choices=[roles.Choice("rank", "Rank", RANKS), roles.Choice("deterministic", "Terms", DETERMINISTIC)],
    numbers=[roles.Number("lags", "Lags (in differences)", range(0, 9), 1), roles.Number("horizon", "Horizon", [4, 8, 10, 12, 20, 24], 10)],
    buttons=[(shock_label, next_shock)],
    chart=response_chart,
    series=True,
    hint="Choose two or more Y fields in levels and the time field (T), then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings() + ["shock"]


def on_click(node):
    return roles.open_panel(node, SPEC)


# Fits a vector error correction model: the differences of the Y fields on lagged differences and
# on how far the fields were from their long-run relations last period (the error correction
# terms). alpha says how fast each field moves back towards the relations; beta are the relations
def run(inputs, settings):
    SPEC.defaults(settings)
    notes = []
    y, x, times, notes = series_data(inputs, {**settings, "independent": []}, notes)
    endogenous = list(y.columns)
    k = len(endogenous)
    lags = settings["lags"]
    deterministic = settings["deterministic"]
    rank_tests, johansen_rank, _ = johansen_table(y, JOHANSEN_ORDER[deterministic], lags)
    rank_tests.attrs = {"rank": johansen_rank}
    if settings["rank"] == "auto":
        rank = johansen_rank
        if rank == 0:
            raise ValueError("Johansen's trace test finds no cointegration at 5%, so there is no error correction to model. "
                             "Use a VAR in differences, or set the rank by hand")
        if rank == k:
            raise ValueError("Johansen's test finds full rank, so the fields look stationary already. Use a VAR in levels, or set the rank by hand")
    else:
        rank = int(settings["rank"])
        if rank >= k:
            raise ValueError(f"The rank must be below the number of Y fields ({k})")
        if rank != johansen_rank:
            notes.append(f"Johansen's trace test suggests rank {johansen_rank}, not {rank}.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = VECM(y, k_ar_diff=lags, coint_rank=rank, deterministic=deterministic).fit()

    equations, term_roles, names, values, errors = [], [], [], [], []
    for i, equation in enumerate(endogenous):
        for r in range(rank):
            equations.append(equation); term_roles.append("Adjustment"); names.append(f"ec{r + 1}")
            values.append(fit.alpha[i, r]); errors.append(fit.stderr_alpha[i, r])
        for lag in range(lags):
            for j, source in enumerate(endogenous):
                column = lag * k + j
                equations.append(equation); term_roles.append("Short run"); names.append(f"L{lag + 1}.d_{source}")
                values.append(fit.gamma[i, column]); errors.append(fit.stderr_gamma[i, column])
        if "co" in deterministic and getattr(fit, "det_coef", None) is not None and fit.det_coef.size:
            for d in range(fit.det_coef.shape[1]):
                equations.append(equation); term_roles.append("Intercept"); names.append("Intercept" if d == 0 else f"det{d + 1}")
                values.append(fit.det_coef[i, d]); errors.append(fit.stderr_det_coef[i, d])
    table = estimates.coefficient_table(names, term_roles, values, errors, leading={"equation": equations})

    vector_rows = []
    for r in range(rank):
        for j, name in enumerate(endogenous):
            vector_rows.append({"relation": f"ec{r + 1}", "variable": name, "coefficient": fit.beta[j, r],
                                "std_error": fit.stderr_beta[j, r] if j >= rank else np.nan})
        if "ci" in deterministic and fit.det_coef_coint.size:
            vector_rows.append({"relation": f"ec{r + 1}", "variable": "Intercept", "coefficient": fit.det_coef_coint[0, r],
                                "std_error": fit.stderr_det_coef_coint[0, r]})
        if "li" in deterministic and fit.det_coef_coint.shape[0] > 1:
            vector_rows.append({"relation": f"ec{r + 1}", "variable": "Trend", "coefficient": fit.det_coef_coint[1, r],
                                "std_error": fit.stderr_det_coef_coint[1, r]})
    vectors = pd.DataFrame(vector_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        vectors["z_stat"] = vectors["coefficient"] / vectors["std_error"]
    vectors["p_value"] = 2 * stats.norm.sf(np.abs(vectors["z_stat"]))

    horizon = settings["horizon"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        responses = fit.irf(horizon).orth_irfs
    impulse_rows = [{"horizon": h, "impulse": shock, "response": response, "value": responses[h, i, j]}
                    for h in range(horizon + 1) for j, shock in enumerate(endogenous) for i, response in enumerate(endogenous)]
    impulses = pd.DataFrame(impulse_rows)

    slow = [endogenous[i] for i in range(k) if all(abs(fit.alpha[i, r]) / max(fit.stderr_alpha[i, r], 1e-12) < 1.96 for r in range(rank))]
    summary = [
        f"VECM    rank {rank}    lags in differences: {lags}    N: {len(y) - lags - 1}    Log-likelihood: {fit.llf:.2f}",
        rank_text(johansen_rank, k),
    ]
    if slow:
        summary.append(f"Not adjusting to the long-run relations (no significant alpha): {', '.join(slow)}; these drive the system")
    covariance = np.diag(np.square(np.nan_to_num(np.asarray(errors, dtype=float))))
    table.attrs = {"observations": int(len(y) - lags - 1), "rank": int(rank), "log_likelihood": float(fit.llf),
                   "summary": summary, "notes": notes}
    estimates.record(table, "vecm", "VECM", ", ".join(endogenous), y, endogenous, covariance, fit.llf, rank=int(rank),
                     equations=endogenous, diagonal_covariance=True)
    for output in (vectors, rank_tests, impulses):
        output.attrs.update(summary=summary, notes=notes)
    return [table, vectors, rank_tests, impulses]
