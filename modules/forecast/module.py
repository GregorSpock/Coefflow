import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from engine import fits
from modules.arimax.module import read_time
from ui.panel import Panel


INPUTS = 2
OUTPUTS = 1
INPUT_SPECS = [
    ("Coefficients", "Out 1 of a regression: OLS, Auto OLS, Logit, Probit, ARIMAX, or Ologit and Oprobit from the Ordered add-on. A coefficient table that was saved and loaded again works too, except for ARIMAX"),
    ("New data", "a table with the fields the model was fitted on, other than Y; for ARIMAX, one row per future period. If it has the Y field too, the forecasts are scored against it"),
]
OUTPUT_SPECS = [
    ("Forecast", "the new data plus forecast, lower and upper (linear models and ARIMAX), probability and predicted (logit and probit), or prob_<category> and predicted (ordered models), with error if Y is in the new data"),
]

# The widths of the prediction intervals the Interval button steps through
LEVELS = [0.8, 0.9, 0.95, 0.99]
# The Model button's choices: auto uses what the coefficient table says about itself
MODEL_CHOICES = ["auto"] + list(fits.KINDS)
SHORT_NAMES = {"auto": "Auto-detect", "linear": "OLS", "logit": "Logit", "probit": "Probit", "ologit": "Ologit", "oprobit": "Oprobit", "arimax": "ARIMAX"}
LINKS = {"logit": stats.logistic, "probit": stats.norm, "ologit": stats.logistic, "oprobit": stats.norm}
INTERCEPT = ["Intercept", "const"]


# Opens a panel showing which model the coefficients came from and the forecasts for the new data,
# with buttons to choose the model by hand and the width of the prediction intervals
def on_click(node):
    panel = Panel(node, "Forecast")
    panel.add_button(model_label(node.settings), cycle_model)
    panel.add_button(level_label(node.settings), cycle_level)
    panel.preview_title = "Forecast"
    panel.refresh = show_forecast
    show_forecast(panel)
    return panel


# The Model button shows the model chosen by hand, or Auto-detect; the panel's first line says which it found
def model_label(settings):
    return SHORT_NAMES[settings.get("model", "auto")]


def level_label(settings):
    return f"Interval: {settings.get('level', 0.95):.0%}"


# Steps through the models, for a coefficient table that has lost its record of which model made it
def cycle_model(panel, button):
    settings = panel.node.settings
    current = MODEL_CHOICES.index(settings.get("model", "auto"))
    settings["model"] = MODEL_CHOICES[(current + 1) % len(MODEL_CHOICES)]
    button.text = model_label(settings)
    show_forecast(panel)


def cycle_level(panel, button):
    settings = panel.node.settings
    current = LEVELS.index(settings.get("level", 0.95))
    settings["level"] = LEVELS[(current + 1) % len(LEVELS)]
    button.text = level_label(settings)
    show_forecast(panel)


# The kind of model behind a coefficient table (one of fits.KINDS) and what it recorded about its fit.
# The Model button can pick the kind by hand; otherwise the table's record says, and a table without
# one, such as one saved and loaded again, is recognised from its rows and columns
def model_of(coefficients, settings):
    for column in ["variable", "coefficient"]:
        if column not in coefficients.columns:
            raise ValueError("In 1 needs a coefficient table with variable and coefficient columns, like Out 1 of OLS")

    fit = coefficients.attrs.get("fit", {})
    chosen = settings.get("model", "auto")
    if chosen != "auto":
        return chosen, fit if fit.get("kind") == chosen else {}
    if fit.get("kind") in fits.KINDS:
        return fit["kind"], fit

    roles = set(coefficients["role"].astype(str)) if "role" in coefficients else set()
    if "lag" in coefficients or roles & {"AR", "MA", "Variance"}:
        return "arimax", {}
    if "Cut" in roles or coefficients["variable"].astype(str).str.startswith("cut ").any():
        return ("ologit" if "odds_ratio" in coefficients else "oprobit"), {}
    if "marginal_effect" in coefficients:
        return ("logit" if "odds_ratio" in coefficients else "probit"), {}
    return "linear", {}


# The value of each term for each new row: 1 for the intercept, the field itself for a numeric term,
# and 1 or 0 for a category's dummy, named like region=London. A row whose category of a text field
# wasn't seen in the fit, so isn't the baseline either, gets missing values, as the model can't say
# anything about it. Raises ValueError naming any fields the new data is missing
def term_values(data, names, levels, notes):
    columns, missing = {}, []
    for name in names:
        if name in INTERCEPT:
            columns[name] = pd.Series(1.0, index=data.index)
        elif name in data.columns:
            columns[name] = pd.to_numeric(data[name], errors="coerce")
        elif "=" in name and name.split("=", 1)[0] in data.columns:
            field, level = name.split("=", 1)
            values = data[field]
            columns[name] = (values.astype(str) == level).astype(float).where(values.notna())
        else:
            missing.append(name.split("=", 1)[0])
    if missing:
        raise ValueError(f"The new data is missing fields the model uses: {', '.join(dict.fromkeys(missing))}")

    terms = pd.DataFrame(columns, index=data.index)
    for field, known in levels.items():
        if field not in data:
            continue
        values = data[field]
        unseen = values.notna() & ~values.astype(str).isin(known)
        if unseen.any():
            notes.append(
                f"{unseen.sum()} rows have {field} categories the model never saw (e.g. {values[unseen].iloc[0]}), so they get no forecast."
            )
            terms.loc[unseen] = np.nan
    return terms


# The terms and coefficients of a coefficient table's rows, or of only some of them, such as an
# ordered model's rows other than its cut points. Raises ValueError if a coefficient is missing,
# as every forecast would be too
def coefficients_of(coefficients, rows=None):
    used = coefficients if rows is None else coefficients[rows]
    blank = used["variable"][used["coefficient"].isna()]
    if len(blank):
        raise ValueError(f"The coefficient of {blank.iloc[0]} is missing")
    return list(used["variable"].astype(str)), used["coefficient"].to_numpy(float)


# Says how many rows got no forecast because they are missing a value the model needs
def note_gaps(forecast, notes, before):
    gaps = int(np.isnan(forecast).sum()) - before
    if gaps > 0:
        notes.append(f"{gaps} rows have missing or non-numeric values in the model's fields, so they get no forecast.")


# Rows with a category the model never saw already have their own note
def unseen_rows(terms):
    return int(terms.isna().all(axis=1).sum())


# Linear models: the forecast is the intercept plus each coefficient times its term, with a
# prediction interval for a new observation (the coefficients' uncertainty plus the residual variance)
# when the table still has its record of the fit's covariance
def forecast_linear(data, coefficients, fit, level, notes):
    names, beta = coefficients_of(coefficients)
    terms = term_values(data, names, fit.get("levels", {}), notes)
    values = terms.to_numpy(float)
    forecast = values @ beta
    note_gaps(forecast, notes, unseen_rows(terms))

    output = data.copy()
    output["forecast"] = forecast
    covariance = fit.get("covariance")
    if covariance is not None and covariance.shape == (len(beta), len(beta)):
        spread = np.sqrt(np.einsum("ij,jk,ik->i", values, covariance, values) + fit["residual_variance"])
        critical = stats.t.ppf((1 + level) / 2, fit["degrees_of_freedom"])
        output["lower"] = forecast - critical * spread
        output["upper"] = forecast + critical * spread
    else:
        notes.append("No prediction intervals: the coefficient table has lost its record of the fit, e.g. by being saved and loaded again or having rows removed.")
    return output


# Logit and probit: the probability that Y is 1 is F(Xb), predicted as 1 when it is at least 0.5
def forecast_binary(data, coefficients, fit, kind, notes):
    names, beta = coefficients_of(coefficients)
    terms = term_values(data, names, fit.get("levels", {}), notes)
    index = terms.to_numpy(float) @ beta
    note_gaps(index, notes, unseen_rows(terms))

    output = data.copy()
    output["probability"] = LINKS[kind].cdf(index)
    output["predicted"] = pd.Series((output["probability"] >= 0.5).astype(int), index=data.index).astype("Int64")
    output.loc[np.isnan(index), "predicted"] = pd.NA
    return output


# A category from a cut point's name: "cut 2|3" has 2 below it and 3 above. Numbers are kept as numbers
def category_value(text):
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


# Ordered logit and probit: P(Y <= category j) = F(cut j - Xb), so the chance of each category is the
# gap between the cut points either side of it. The prediction is the most likely category
def forecast_ordered(data, coefficients, fit, kind, notes):
    variables = coefficients["variable"].astype(str)
    is_cut = variables.str.startswith("cut ")
    if "role" in coefficients:
        is_cut |= coefficients["role"].astype(str) == "Cut"
    if not is_cut.any():
        raise ValueError("An ordered model's coefficient table needs its cut point rows (named like cut 1|2)")

    names, beta = coefficients_of(coefficients, ~is_cut)
    cut_names, cuts = coefficients_of(coefficients, is_cut)
    bounds = [name[4:].split("|") for name in cut_names]
    categories = [category_value(bounds[0][0])] + [category_value(high) for low, high in bounds]

    terms = term_values(data, names, fit.get("levels", {}), notes)
    index = terms.to_numpy(float) @ beta if names else np.zeros(len(data))
    note_gaps(index, notes, unseen_rows(terms))

    below = LINKS[kind].cdf(cuts[None, :] - index[:, None])
    edges = np.hstack([np.zeros((len(data), 1)), below, np.ones((len(data), 1))])
    probabilities = np.diff(edges, axis=1)

    output = data.copy()
    for category, column in zip(categories, probabilities.T):
        output[f"prob_{category}"] = column
    predicted = pd.Series([categories[i] for i in probabilities.argmax(axis=1)], index=data.index, dtype=object)
    output["predicted"] = predicted.where(~np.isnan(index))
    return output


# The new rows for the series ARIMAX was fitted on, when it picked one series out by a group field
def series_rows(data, fit, notes):
    group = fit.get("group")
    if not group or group not in data or fit.get("series") is None:
        return data
    rows = data[data[group].astype(str) == fit["series"]]
    if len(rows) < len(data):
        notes.append(f"Kept the {len(rows)} rows for {group} {fit['series']}, the series the model was fitted on.")
    return rows


# The new rows sorted by the model's time field, which must not repeat
def in_time_order(data, fit, notes):
    time = fit.get("time")
    if not time or time not in data:
        notes.append("The new data has no time field, so its rows are taken as the next periods in their current order.")
        return data
    times = read_time(data[time])
    if times.isna().any() or times.duplicated().any():
        raise ValueError(f"Each new row needs its own {time}, with none missing or repeated")
    return data.loc[times.sort_values().index]


# The X and C columns for the new periods, lagged as in the fit. The first periods' lags reach
# back into the last periods of the fit, which the model's record keeps
def future_exog(data, fit):
    labels = fit["exog_labels"]
    if not labels:
        return None
    fields = list(dict.fromkeys(column for role, column, lag in labels.values()))
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"The new data is missing fields the model uses: {', '.join(missing)}")

    columns = {}
    for name, (role, column, lag) in labels.items():
        new = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        values = pd.Series(np.concatenate([fit["recent"][column], new]))
        columns[name] = values.shift(lag).to_numpy()[-len(data):]
    future = pd.DataFrame(columns)

    gaps = future.columns[future.isna().any()].tolist()
    if gaps:
        raise ValueError(f"ARIMAX needs every X and C value for every new period, but some are missing for: {', '.join(gaps)}")
    return future


# ARIMAX: carries the fitted series on into the new periods with the table's coefficients,
# using the new rows' X and C values, with forecast intervals that widen the further ahead they go
def forecast_arimax(data, coefficients, fit, level, notes):
    if not fit:
        raise ValueError("ARIMAX forecasts need the series the model was fitted on, which only comes with ARIMAX's own Out 1. Connect it straight to In 1")
    params = coefficients["coefficient"].to_numpy(float)
    if len(params) != len(fit["parameters"]) or np.isnan(params).any():
        raise ValueError("The ARIMAX coefficient table has had rows removed or changed. Connect ARIMAX's Out 1 straight to In 1")

    data = in_time_order(series_rows(data, fit, notes), fit, notes)
    if data.empty:
        raise ValueError("The new data has no rows to forecast")
    future = future_exog(data, fit)

    exog = pd.DataFrame(fit["exog"], columns=list(fit["exog_labels"])) if fit["exog_labels"] else None
    model = ARIMA(pd.Series(fit["y"]), exog=exog, order=fit["order"], trend=fit["trend"])
    result = model.filter(params)
    frame = result.get_forecast(len(data), exog=future).summary_frame(alpha=1 - level)

    output = data.copy()
    output["forecast"] = frame["mean"].to_numpy()
    output["lower"] = frame["mean_ci_lower"].to_numpy()
    output["upper"] = frame["mean_ci_upper"].to_numpy()
    return output


# When the new data has the Y field, adds each row's error and a note on how close the forecasts came
def score(output, kind, dependent, notes):
    if not dependent or dependent not in output:
        return
    actual = pd.to_numeric(output[dependent], errors="coerce")

    if kind in ("linear", "arimax"):
        output["error"] = actual - output["forecast"]
        errors = output["error"].dropna()
        if len(errors):
            notes.append(
                f"Against the {dependent} values in the new data ({len(errors)} rows): "
                f"RMSE = {np.sqrt((errors ** 2).mean()):.4g}, mean absolute error = {errors.abs().mean():.4g}."
            )
        return

    both = actual.notna() & output["predicted"].notna()
    if both.any():
        correct = (pd.to_numeric(output["predicted"][both]) == actual[both]).mean()
        notes.append(f"Against the {dependent} values in the new data ({both.sum()} rows): {correct:.1%} predicted correctly.")


# Applies the coefficients to the new data, whichever kind of model they came from.
# Returns the forecast table, the kind of model, its record and notes on the forecasts
def forecast(inputs, settings):
    coefficients, data = inputs
    kind, fit = model_of(coefficients, settings)
    level = settings.get("level", 0.95)
    notes = []

    if kind == "linear":
        output = forecast_linear(data, coefficients, fit, level, notes)
    elif kind in ("logit", "probit"):
        output = forecast_binary(data, coefficients, fit, kind, notes)
    elif kind in ("ologit", "oprobit"):
        output = forecast_ordered(data, coefficients, fit, kind, notes)
    else:
        output = forecast_arimax(data, coefficients, fit, level, notes)

    score(output, kind, fit.get("dependent"), notes)
    return output, kind, fit, notes


# Says which model the coefficients came from, how it was recognised, and what was forecast
def describe(kind, fit, settings, output):
    if settings.get("model", "auto") != "auto":
        source = "chosen by hand"
    elif fit:
        source = f"from {fit['model']}"
    else:
        source = "recognised from the table's columns"
    line = f"Model: {fits.KINDS[kind]}, {source}"
    if fit.get("dependent"):
        line += f"    Y: {fit['dependent']}"
    return [line, f"Forecast rows: {len(output)}"]


# Shows the model behind the coefficients, then the forecasts with any notes, or what went wrong
def show_forecast(panel):
    node = panel.node
    settings = node.settings
    panel.preview = None
    panel.note = ""
    panel.message = ""

    if not node.last_inputs or len(node.last_inputs) < 2 or any(table is None for table in node.last_inputs):
        panel.lines = ["Connect a model's coefficients to In 1 and the new data to In 2, then run the workflow"]
        return

    try:
        output, kind, fit, notes = forecast(node.last_inputs, settings)
    except ValueError as error:
        panel.lines = ["Could not forecast"]
        panel.note = str(error)
        return

    panel.lines = describe(kind, fit, settings, output)
    panel.note = " ".join(notes)
    # The forecast columns go first in the preview, so they show without scrolling across
    added = [column for column in output.columns if column not in node.last_inputs[1].columns]
    panel.preview = output[added + [column for column in output.columns if column not in added]]


def run(inputs, settings):
    return [forecast(inputs, settings)[0]]
