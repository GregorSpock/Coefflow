import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from engine import estimates, roles
from engine.chart import draw_lines


INPUTS = 1
OUTPUTS = 3
OUTPUT_TYPES = ["model", "table", "table"]
INPUT_SPECS = [
    ("Data", "a table with a numeric Y and numeric or text X and C fields"),
]
OUTPUT_SPECS = [
    ("Coefficients", "the fit at the first quantile typed: one row per term with role, variable, coefficient, std_error, t_stat, "
                     "p_value and stars, ready for Forecast, Wald Test or Bootstrap"),
    ("All quantiles", "one row per quantile and term with quantile, variable, coefficient, std_error, lower and upper (95%), "
                      "and the OLS coefficient for comparison"),
    ("Fitted data", "the rows used, plus fitted_q<quantile> for each quantile"),
]

ERRORS = [("robust", "Robust"), ("iid", "Ordinary")]
DEFAULT_QUANTILES = "0.25, 0.5, 0.75"


def term_label(settings):
    return f"Chart: {settings.get('chart_term') or 'first X'}"


def next_term(panel, button):
    model = panel.node.last_outputs[1] if len(panel.node.last_outputs) > 1 else None
    names = [] if model is None else [v for v in dict.fromkeys(model["variable"]) if v != "Intercept"]
    roles.step(panel, "chart_term", names)
    button.text = term_label(panel.node.settings)


# Draws each quantile's coefficient on the chosen term with its 95% band, against OLS's
def coefficient_chart(node):
    if len(node.last_outputs) < 2:
        return None
    table = node.last_outputs[1]
    terms = [v for v in dict.fromkeys(table["variable"]) if v != "Intercept"]
    if not terms:
        return None
    term = node.settings.get("chart_term")
    term = term if term in terms else terms[0]
    rows = table[table["variable"] == term].sort_values("quantile")
    q = rows["quantile"].to_numpy(float)
    series = [
        (f"{term} by quantile", q, rows["coefficient"].to_numpy(float)),
        ("lower 95%", q, rows["lower"].to_numpy(float)),
        ("upper 95%", q, rows["upper"].to_numpy(float)),
        ("OLS", q, rows["ols"].to_numpy(float)),
    ]

    def draw(painter, area):
        from engine import draw as drawing
        draw_lines(painter, area, drawing.font(22), series, "quantile")
    return draw


SPEC = roles.Spec(
    "Quantile Reg",
    [
        roles.Role("Y", "dependent"),
        roles.Role("X", "independent", many=True, kinds="model"),
        roles.Role("C", "controls", many=True, kinds="model"),
    ],
    choices=[roles.Choice("errors", "Errors", ERRORS)],
    text=("quantiles", "Quantiles, e.g. 0.25, 0.5, 0.75", DEFAULT_QUANTILES),
    buttons=[(term_label, next_term)],
    chart=coefficient_chart,
    hint="Choose Y and at least one X, type the quantiles, then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings() + ["chart_term"]


def on_click(node):
    return roles.open_panel(node, SPEC)


def parse_quantiles(text):
    quantiles = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip().rstrip("%")
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"{part} isn't a quantile. Type numbers between 0 and 1, e.g. 0.25, 0.5")
        if value >= 1:
            value /= 100
        if not 0 < value < 1:
            raise ValueError(f"Quantiles must be between 0 and 1, not {part}")
        quantiles.append(round(value, 6))
    if not quantiles:
        raise ValueError("Type at least one quantile, e.g. 0.5 for median regression")
    return list(dict.fromkeys(quantiles))


# Fits quantile regressions of Y on X and C: the line through each chosen quantile of Y given the
# fields, rather than through its mean, so effects on the low or high end of Y can differ from
# the effect on the middle. Median regression (0.5) is robust to outliers in Y
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y", "X"))
    dependent = settings["dependent"]
    fields = settings["independent"] + settings["controls"]
    quantiles = parse_quantiles(settings.get("quantiles"))
    used = data.dropna(subset=list(dict.fromkeys([dependent] + fields)))
    x, term_roles = estimates.design(used, settings)
    y = used[dependent].astype(float)
    if len(used) < x.shape[1] + 5:
        raise ValueError(f"{len(used)} rows is too few for {x.shape[1]} coefficients")

    ols = sm.OLS(y, x).fit()
    ols_ci = ols.conf_int()
    rows, fits_by_q, notes = [], {}, []
    fitted = used.copy()
    for q in quantiles:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = QuantReg(y, x).fit(q=q, vcov=settings["errors"], max_iter=5000)
        if any("iteration limit" in str(w.message).lower() or "convergence" in str(w.message).lower() for w in caught):
            notes.append(f"The fit at quantile {q:g} did not fully converge.")
        fits_by_q[q] = fit
        ci = fit.conf_int()
        for name in x.columns:
            rows.append({
                "quantile": q, "variable": name, "coefficient": fit.params[name], "std_error": fit.bse[name],
                "lower": ci.loc[name, 0], "upper": ci.loc[name, 1], "ols": ols.params[name],
                "ols_lower": ols_ci.loc[name, 0], "ols_upper": ols_ci.loc[name, 1],
            })
        fitted[f"fitted_q{q:g}"] = fit.fittedvalues

    first = fits_by_q[quantiles[0]]
    table = estimates.coefficient_table(x.columns, term_roles, first.params.values, first.bse.values, df=first.df_resid)
    process = pd.DataFrame(rows)
    pseudo = {q: float(fit.prsquared) for q, fit in fits_by_q.items()}
    summary = [
        f"N: {len(used)}    Out 1 holds quantile {quantiles[0]:g}    Pseudo R squared: "
        + ", ".join(f"q{q:g} {r:.3f}" for q, r in pseudo.items()),
        f"Errors: {dict(ERRORS)[settings['errors']]}    OLS R squared for comparison: {ols.rsquared:.3f}",
    ]
    if len(quantiles) > 1:
        moved = [name for name in x.columns if name != "Intercept" and differs(process, name)]
        if moved:
            summary.append(f"Effects that change across the quantiles (outside each other's 95% bands): {', '.join(moved)}")
    table.attrs = {"observations": len(used), "quantile": quantiles[0], "pseudo_r_squared": pseudo[quantiles[0]],
                   "summary": summary, "notes": notes}
    estimates.record(table, "quantile", "Quantile Reg", dependent, used, fields, first.cov_params(),
                     quantile=quantiles[0], degrees_of_freedom=float(first.df_resid))
    process.attrs = {"summary": summary, "notes": notes}
    return [table, process, fitted]


# True if some two quantiles' 95% bands for a term don't overlap
def differs(process, name):
    rows = process[process["variable"] == name]
    return rows["lower"].max() > rows["upper"].min()
