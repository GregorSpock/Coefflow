import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from engine import draw, estimates, roles
from engine.chart import draw_lines


INPUTS = 1
OUTPUTS = 4
OUTPUT_TYPES = ["model", "table", "table", "table"]
INPUT_SPECS = [
    ("Data", "a time series table: two or more numeric Y fields that move together (ideally stationary, e.g. growth rates), "
             "optionally exogenous X fields, a time field (T), and a group field (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per equation and term with equation, role, variable (e.g. L1.inflation for last period's inflation), "
                     "coefficient, std_error, t_stat, p_value and stars"),
    ("Impulse responses", "one row per horizon, shock and response with the response, its standard error and a 95% band "
                          "(lower, upper). Orthogonal shocks are one standard deviation, in the order the Y fields were chosen"),
    ("Variance decomposition", "one row per horizon and response, with the share of its forecast error variance due to each shock"),
    ("Tests", "Granger causality between each pair of Y fields, the lag order criteria for each lag, stability, "
              "and the Portmanteau and normality tests of the residuals, each with a plain-words result"),
]

TRENDS = [("c", "Constant"), ("ct", "Constant + trend"), ("n", "None")]
SELECTION = [("fixed", "Fixed"), ("aic", "AIC"), ("bic", "BIC"), ("hqic", "HQ")]
SHOCKS = [("orth", "Orthogonal"), ("plain", "Unit")]
LEVEL = 0.05


def shock_label(settings):
    return f"Shock: {settings.get('shock') or 'first Y'}"


def next_shock(panel, button):
    roles.step(panel, "shock", panel.node.settings.get("dependent") or [])
    button.text = shock_label(panel.node.settings)


# Draws the response of every Y to the chosen shock over the horizon
def response_chart(node):
    if len(node.last_outputs) < 2:
        return None
    table = node.last_outputs[1]
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
    "VAR",
    [
        roles.Role("Y", "dependent", many=True),
        roles.Role("X", "independent", many=True),
        roles.Role("T", "time", kinds="any"),
        roles.Role("G", "group", kinds="any"),
    ],
    choices=[roles.Choice("selection", "Lags", SELECTION), roles.Choice("trend", "Trend", TRENDS),
             roles.Choice("shocks", "Shocks", SHOCKS)],
    numbers=[roles.Number("lags", "Lag order (most tried)", range(1, 9), 2), roles.Number("horizon", "Horizon", [4, 8, 10, 12, 20, 24], 10)],
    buttons=[(shock_label, next_shock)],
    chart=response_chart,
    series=True,
    hint="Choose two or more Y fields and the time field (T), then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings() + ["shock"]


def on_click(node):
    return roles.open_panel(node, SPEC)


def verdict(p_value, rejected, kept):
    return f"{rejected} (p = {p_value:.3g})" if p_value < LEVEL else f"{kept} (p = {p_value:.3g})"


# Gets the Y and X fields of one series in time order, with no gaps. Returns the Y table, the X
# table or None, the times and notes
def series_data(inputs, settings, notes, least=2):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles)
    endogenous = settings.get("dependent") or []
    if len(endogenous) < least:
        raise ValueError(f"Choose at least {least} Y fields")
    exogenous = [c for c in settings.get("independent") or [] if c not in endogenous]
    data, times = estimates.time_order(data, settings, notes)
    complete = data[endogenous + exogenous].notna().all(axis=1)
    if not complete.any():
        raise ValueError("No rows have every Y and X field filled in")
    positions = np.flatnonzero(complete.to_numpy())
    data, times = data.iloc[positions[0]:positions[-1] + 1], times.iloc[positions[0]:positions[-1] + 1]
    gaps = data[endogenous + exogenous].isna().any(axis=1)
    if gaps.any():
        raise ValueError(f"{int(gaps.sum())} rows in the middle of the series are missing a Y or X value. Fill or filter them first")
    y = data[endogenous].astype(float).reset_index(drop=True)
    x = data[exogenous].astype(float).reset_index(drop=True) if exogenous else None
    return y, x, times.reset_index(drop=True), notes


# The coefficient table: each equation's terms in turn, with the matching block of the covariance
def coefficient_rows(fit, endogenous):
    names, equations, term_roles, values, errors, order = [], [], [], [], [], []
    labels = list(fit.params.index)
    covariance = fit.cov_params()
    for equation in endogenous:
        for label in labels:
            names.append("Intercept" if label == "const" else ("Trend" if label == "trend" else label))
            equations.append(equation)
            term_roles.append("Intercept" if label == "const" else ("Trend" if label == "trend" else
                              ("X" if not label.startswith("L") or "." not in label else "Lag")))
            values.append(fit.params.loc[label, equation])
            errors.append(fit.stderr.loc[label, equation])
            order.append((label, equation))
    df = fit.df_resid
    table = estimates.coefficient_table(names, term_roles, values, errors, df=df, leading={"equation": equations})
    matrix = covariance.loc[order, order].to_numpy(float)
    return table, matrix


def impulse_table(fit, endogenous, horizon, orthogonal):
    irf = fit.irf(horizon)
    values = irf.orth_irfs if orthogonal else irf.irfs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            errors = irf.stderr(orth=orthogonal)
        except (np.linalg.LinAlgError, ValueError):
            errors = np.full_like(values, np.nan)
    rows = []
    for h in range(horizon + 1):
        for j, shock in enumerate(endogenous):
            for i, response in enumerate(endogenous):
                value, error = values[h, i, j], errors[h, i, j]
                rows.append({"horizon": h, "impulse": shock, "response": response, "value": value,
                             "std_error": error, "lower": value - 1.96 * error, "upper": value + 1.96 * error})
    return pd.DataFrame(rows)


def decomposition_table(fit, endogenous, horizon):
    shares = fit.fevd(horizon + 1).decomp  # response, horizon, shock
    rows = []
    for h in range(horizon + 1):
        for i, response in enumerate(endogenous):
            row = {"horizon": h + 1, "response": response}
            for j, shock in enumerate(endogenous):
                row[f"share_{shock}"] = shares[i, h, j]
            rows.append(row)
    return pd.DataFrame(rows)


def test_rows(fit, endogenous, y, x, settings, chosen_lag):
    rows = []
    for caused in endogenous:
        for causing in endogenous:
            if caused == causing:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fit.test_causality(caused, [causing], kind="f")
            rows.append({
                "test": "Granger causality", "null": f"{causing} does not Granger-cause {caused}",
                "statistic": float(result.test_statistic), "df": f"F({result.df[0]}, {result.df[1]})", "p_value": float(result.pvalue),
                "result": verdict(result.pvalue, f"Past {causing} helps predict {caused}", f"No evidence past {causing} helps predict {caused}"),
            })
    roots = np.abs(fit.roots) if fit.k_ar else np.array([np.inf])
    smallest = float(roots.min())
    rows.append({
        "test": "Stability", "null": "", "statistic": 1 / smallest if smallest > 0 else np.inf, "df": "largest root modulus",
        "p_value": np.nan,
        "result": "Stable: every root is inside the unit circle" if fit.is_stable() else
        "Not stable: a root is on or outside the unit circle, so shocks don't die out; difference the Y fields",
    })
    lags = max(fit.k_ar + 1, min(10, fit.nobs // 5))
    if lags * len(endogenous) ** 2 > fit.k_ar * len(endogenous) ** 2 and lags < fit.nobs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            white = fit.test_whiteness(nlags=lags, adjusted=True)
        rows.append({
            "test": "Portmanteau", "null": f"no residual autocorrelation up to lag {lags}",
            "statistic": float(white.test_statistic), "df": f"chi2({white.df})", "p_value": float(white.pvalue),
            "result": verdict(white.pvalue, "Residuals are autocorrelated: add lags", "No sign of residual autocorrelation"),
        })
    normal = fit.test_normality()
    rows.append({
        "test": "Normality (Jarque-Bera)", "null": "residuals are normal", "statistic": float(normal.test_statistic),
        "df": f"chi2({normal.df})", "p_value": float(normal.pvalue),
        "result": verdict(normal.pvalue, "Residuals are not normal; the bands rest on large samples", "Residuals look normal"),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(y, exog=x)
        try:
            order = model.select_order(settings["lags"], trend=settings["trend"])
            for lag, values in enumerate(zip(*[order.ics[name] for name in ("aic", "bic", "hqic", "fpe")])):
                rows.append({
                    "test": "Lag order", "null": f"{lag} lags", "statistic": values[0],
                    "df": f"AIC; BIC {values[1]:.4g}, HQ {values[2]:.4g}",
                    "p_value": np.nan, "result": "Chosen" if lag == chosen_lag else "",
                })
        except (ValueError, np.linalg.LinAlgError):
            pass
    table = pd.DataFrame(rows, columns=["test", "null", "statistic", "df", "p_value", "result"])
    return table


# Fits a vector autoregression: each Y on its own past values and those of the other Y fields
# (and any X fields), equation by equation with OLS. The lag order is fixed or the one with the
# best information criterion up to the number set. Impulse responses trace how a shock to one Y
# feeds through all of them over the horizon
def run(inputs, settings):
    SPEC.defaults(settings)
    notes = []
    y, x, times, notes = series_data(inputs, settings, notes)
    endogenous = list(y.columns)
    most = settings["lags"]
    if len(y) <= most * len(endogenous) + 5:
        raise ValueError(f"{len(y)} periods is too few for {most} lags of {len(endogenous)} fields")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(y, exog=x)
        if settings["selection"] == "fixed":
            lags = most
        else:
            selected = model.select_order(most, trend=settings["trend"]).selected_orders[settings["selection"]]
            lags = max(1, int(selected))
            if int(selected) == 0:
                notes.append(f"{settings['selection'].upper()} picks 0 lags; 1 lag is used so the model has dynamics.")
        fit = model.fit(lags, trend=settings["trend"])

    table, covariance = coefficient_rows(fit, endogenous)
    horizon = settings["horizon"]
    orthogonal = settings["shocks"] == "orth"
    impulses = impulse_table(fit, endogenous, horizon, orthogonal)
    decomposition = decomposition_table(fit, endogenous, horizon)
    tests = test_rows(fit, endogenous, y, x, settings, lags)

    criteria = {"aic": float(fit.aic), "bic": float(fit.bic), "hqic": float(fit.hqic)}
    summary = [
        f"VAR({lags})    {len(endogenous)} equations    N: {fit.nobs} periods ({estimates.time_text(times.iloc[lags])} to "
        f"{estimates.time_text(times.iloc[-1])})",
        f"AIC: {criteria['aic']:.3f}    BIC: {criteria['bic']:.3f}    HQ: {criteria['hqic']:.3f}    "
        f"Log-likelihood: {fit.llf:.2f}    {'Stable' if fit.is_stable() else 'Not stable'}",
    ]
    granger = tests[(tests["test"] == "Granger causality") & (tests["p_value"] < LEVEL)]
    if len(granger):
        summary.append("Granger causality at 5%: " + "; ".join(n.replace(" does not Granger-cause ", " -> ") for n in granger["null"]))
    if not fit.is_stable():
        notes.append("The VAR is not stable, so impulse responses don't die away. Use growth rates or differences, or a VECM if the fields are cointegrated.")
    if orthogonal and len(endogenous) > 1:
        notes.append(f"Orthogonal shocks depend on the order of the Y fields ({', '.join(endogenous)}): earlier ones can move later ones in the same period, not the other way round.")
    table.attrs = {"observations": int(fit.nobs), "lags": int(lags), "log_likelihood": float(fit.llf), **criteria,
                   "summary": summary, "notes": notes}
    estimates.record(table, "var", "VAR", ", ".join(endogenous), y, endogenous, covariance, fit.llf,
                     observations=int(fit.nobs), lags=int(lags), equations=endogenous)
    for output in (impulses, decomposition, tests):
        output.attrs = {"summary": summary, "notes": notes}
    return [table, impulses, decomposition, tests]
