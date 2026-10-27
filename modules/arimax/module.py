import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from engine import fits, significance
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a time series table: numeric Y, X and C columns, a time column (T), and a group column (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, lag, coefficient, std_error, z_stat, p_value and stars"),
    ("Fitted data", "one row per period with the time, actual, fitted and residual values"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream.
# lags is keyed by field name
FIELD_SETTINGS = ["dependent", "independent", "controls", "time", "group", "lags"]
MAX_AR = 4
MAX_DIFF = 2
MAX_MA = 4
MAX_LAG = 4
SPARE_ROWS = 10
ORDER_ROWS = {"AR order (p)": ("p", MAX_AR), "Differencing (d)": ("d", MAX_DIFF), "MA order (q)": ("q", MAX_MA)}


# Opens a panel to set the ARIMA orders and give each variable a role: Y (dependent),
# X (independent), C (control), T (time) or G (the column naming each series in panel data).
# X and C variables get a lag button, and the top row picks the series and switches options.
# The buttons redraw through panel.refresh, so Auto ARIMAX can share them
def on_click(node):
    panel = Panel(node, "ARIMAX")
    panel.add_button("Next series", next_series)
    panel.add_button(lag_zero_label(node.settings), toggle_lag_zero)
    panel.add_button(constant_label(node.settings), toggle_constant)
    panel.preview_title = "Results"
    panel.refresh = show_model
    show_model(panel)
    return panel


def lag_zero_label(settings):
    return "Lag 0: On" if settings.get("lag_zero", True) else "Lag 0: Off"


def constant_label(settings):
    return "Constant: On" if settings.get("constant", True) else "Constant: Off"


# Switches whether X and C enter in the same period as Y, or only as past values
def toggle_lag_zero(panel, button):
    settings = panel.node.settings
    settings["lag_zero"] = not settings.get("lag_zero", True)
    button.text = lag_zero_label(settings)
    panel.refresh(panel)


# Switches the constant (or drift when d is 1) on or off
def toggle_constant(panel, button):
    settings = panel.node.settings
    settings["constant"] = not settings.get("constant", True)
    button.text = constant_label(settings)
    panel.refresh(panel)


# The values of the group column in sorted order, as text
def series_names(data, group):
    if not group or group not in data.columns:
        return []
    return sorted(data[group].dropna().astype(str).unique())


# The chosen series, or the first one if none is chosen or it no longer exists
def chosen_series(data, settings):
    names = series_names(data, settings.get("group"))
    if not names:
        return None
    if settings.get("series") in names:
        return settings["series"]
    return names[0]


# Moves on to the next series in the group column, going back to the first after the last
def next_series(panel, button):
    node = panel.node
    if not node.last_inputs or node.last_inputs[0] is None:
        return

    data = node.last_inputs[0]
    names = series_names(data, node.settings.get("group"))
    if names:
        current = chosen_series(data, node.settings)
        node.settings["series"] = names[(names.index(current) + 1) % len(names)]
    panel.refresh(panel)


# Returns "Y", "X", "C", "T", "G" or None for a variable
def role_of(settings, column):
    if settings.get("dependent") == column:
        return "Y"
    if column in settings.get("independent", []):
        return "X"
    if column in settings.get("controls", []):
        return "C"
    if settings.get("time") == column:
        return "T"
    if settings.get("group") == column:
        return "G"
    return None


# Takes a variable out of whatever role it has
def clear_role(settings, column):
    for key in ["dependent", "time", "group"]:
        if settings.get(key) == column:
            settings[key] = None
    settings["independent"] = [c for c in settings.get("independent", []) if c != column]
    settings["controls"] = [c for c in settings.get("controls", []) if c != column]


# Gives a variable the clicked role, or clears it if it already had that role.
# Y, T and G can only belong to one variable each, so choosing a new one replaces the old
def choose_role(panel, button):
    settings = panel.node.settings
    column = button.value
    role = button.text
    current = role_of(settings, column)

    clear_role(settings, column)
    if current != role:
        if role == "Y":
            settings["dependent"] = column
        elif role == "X":
            settings["independent"].append(column)
        elif role == "C":
            settings["controls"].append(column)
        elif role == "T":
            settings["time"] = column
        else:
            settings["group"] = column
            settings["series"] = None

    panel.refresh(panel)


def lag_of(settings, column):
    return settings.get("lags", {}).get(column, 0)


# Raises a variable's highest lag by one, going back to 0 after MAX_LAG
def cycle_lag(panel, button):
    lags = panel.node.settings.setdefault("lags", {})
    lags[button.value] = (lags.get(button.value, 0) + 1) % (MAX_LAG + 1)
    panel.refresh(panel)


def order_of(settings, key):
    defaults = {"p": 1, "d": 0, "q": 0}
    return settings.get(key, defaults[key])


# Sets p, d or q to the number clicked
def choose_order(panel, button):
    key = ORDER_ROWS[button.value][0]
    panel.node.settings[key] = int(button.text)
    panel.refresh(panel)


# Describes a variable's lags as text, e.g. "inflation (0-2)"
def lag_text(settings, column):
    first = 0 if settings.get("lag_zero", True) else 1
    last = lag_of(settings, column)
    if last < first:
        return f"{column} (none)"
    if first == last:
        return f"{column} ({first})"
    return f"{column} ({first}-{last})"


# Lists the orders and every variable with its role buttons, then the last results
def show_model(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    p, d, q = order_of(settings, "p"), order_of(settings, "d"), order_of(settings, "q")
    series = chosen_series(data, settings)
    group = f"{settings.get('group')} = {series}" if series is not None else "none"
    panel.lines = [
        f"ARIMA({p}, {d}, {q})    Y: {settings.get('dependent') or 'none'}    "
        f"Time: {settings.get('time') or 'row order'}    Series: {group}",
        f"X: {', '.join(lag_text(settings, c) for c in settings.get('independent', [])) or 'none'}    "
        f"C: {', '.join(lag_text(settings, c) for c in settings.get('controls', [])) or 'none'}",
    ]

    for label, (key, highest) in ORDER_ROWS.items():
        current = order_of(settings, key)
        panel.add_option(label, [(str(n), choose_order, n == current) for n in range(highest + 1)])

    numeric = data.select_dtypes("number").columns
    for column in data.columns:
        role = role_of(settings, column)
        if column in numeric:
            choices = [(r, choose_role, role == r) for r in ["Y", "X", "C", "T", "G"]]
        else:
            choices = [(r, choose_role, role == r) for r in ["T", "G"]]
        if role in ("X", "C"):
            choices.append((f"L{lag_of(settings, column)}", cycle_lag, False))
        panel.add_option(column, choices)

    if node.last_outputs:
        results = node.last_outputs[0]
        fit = results.attrs
        panel.lines.append(
            f"N: {fit['observations']} ({fit['start']} to {fit['end']})    "
            f"AIC: {fit['aic']:.1f}    BIC: {fit['bic']:.1f}"
        )
        panel.lines.append(
            f"ADF p (Y with d = {d}): {number_text(fit['adf_p'])}    "
            f"Ljung-Box p (lag {fit['ljung_box_lag']}): {number_text(fit['ljung_box_p'])}"
        )
        panel.note = " ".join(fit["warnings"])
        panel.preview = results
    else:
        panel.message = "Choose Y, T and the orders, then run the workflow to fit the model"


def number_text(value):
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.3g}"


# Checks Y, X and C are chosen, exist and are numeric
def check_columns(data, settings):
    dependent = settings.get("dependent")
    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")

    variables = [dependent] + settings.get("independent", []) + settings.get("controls", [])
    others = [c for c in [settings.get("time"), settings.get("group")] if c]
    missing = [c for c in variables + others if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")

    text = [c for c in variables if not pd.api.types.is_numeric_dtype(data[c])]
    if text:
        raise ValueError(f"These variables are not numeric: {', '.join(text)}")


# Keeps only the rows of the chosen series when a group column (G) is set
def pick_series(data, settings):
    series = chosen_series(data, settings)
    if series is None:
        return data
    return data[data[settings["group"]].astype(str) == series]


# Turns the time column into numbers or dates that can be sorted.
# Text that cannot be read as dates is an error rather than being silently dropped
def read_time(values):
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(values, errors="coerce")
    unreadable = dates.isna() & values.notna()
    if unreadable.any():
        raise ValueError(
            f"{unreadable.sum()} time values could not be read as dates, e.g. '{values[unreadable].iloc[0]}'"
        )
    return dates


# Sorts the rows by the time column (T). Repeated times mean several series are mixed together,
# which is an error unless a group column picks out one of them. Uneven steps only give a warning
def sort_by_time(data, settings, notes):
    time = settings.get("time")
    if not time:
        notes.append("No time column (T) is chosen, so the rows are used in their current order.")
        return data, pd.Series(range(1, len(data) + 1), index=data.index)

    times = read_time(data[time])
    if times.isna().any():
        notes.append(f"Dropped {times.isna().sum()} rows with no time value.")
        data, times = data[times.notna()], times[times.notna()]

    repeated = times[times.duplicated()]
    if len(repeated):
        example = repeated.iloc[0]
        count = (times == example).sum()
        if settings.get("group"):
            raise ValueError(f"{time} repeats within one series ({example} appears {count} times)")
        raise ValueError(
            f"{time} repeats ({example} appears {count} times). "
            f"Give the column naming each series the G role, or filter to one series first"
        )

    order = times.sort_values().index
    data, times = data.loc[order], times.loc[order]
    if not even_steps(times):
        notes.append(f"The steps in {time} are uneven, so lags and AR terms jump across the gaps.")
    return data, times


# True if the time values are equally spaced (or follow a regular calendar frequency)
def even_steps(times):
    if len(times) < 3:
        return True
    if pd.api.types.is_datetime64_any_dtype(times):
        return pd.infer_freq(pd.DatetimeIndex(times)) is not None
    steps = np.diff(times.to_numpy(float))
    return np.allclose(steps, steps[0])


# Builds the exogenous columns: each X and C variable shifted by every lag it is given.
# Returns the table and, for each column, its role, variable and lag
def lagged_exog(data, settings):
    first = 0 if settings.get("lag_zero", True) else 1
    columns = {}
    labels = {}

    for role, key in [("X", "independent"), ("C", "controls")]:
        for column in settings.get(key, []):
            last = lag_of(settings, column)
            if last < first:
                raise ValueError(f"{column} has no lags left with lag 0 off. Raise its lag to at least L1")
            for lag in range(first, last + 1):
                name = column if lag == 0 else f"{column} (t-{lag})"
                columns[name] = data[column].shift(lag)
                labels[name] = (role, column, lag)

    return pd.DataFrame(columns, index=data.index), labels


# Cuts the rows at the start and end where any value is missing, which is where lags run out.
# Missing X or C values in the middle would break the time spacing, so they are an error.
# Missing Y values in the middle are left in, as the model's Kalman filter can skip them
def trim_missing(y, exog, notes):
    complete = y.notna() & exog.notna().all(axis=1)
    if not complete.any():
        raise ValueError("No rows have every variable filled in once the lags are applied")

    positions = np.flatnonzero(complete.to_numpy())
    keep = slice(positions[0], positions[-1] + 1)
    y, exog = y.iloc[keep], exog.iloc[keep]

    gaps = exog.columns[exog.isna().any()].tolist()
    if gaps:
        raise ValueError(
            f"Missing or infinite values in the middle of: {', '.join(gaps)}. Fill or filter them first"
        )
    if y.isna().any():
        notes.append(f"Y is missing in {y.isna().sum()} periods inside the sample; the model skips over them.")
    return y, exog


# The trend term statsmodels allows: a constant for d = 0, a drift for d = 1, and none for d = 2,
# since differencing removes any trend of lower order than d
def trend_for(settings, d):
    if not settings.get("constant", True) or d > 1:
        return "n"
    return "c" if d == 0 else "t"


# Finds exogenous columns that are constant or exact combinations of earlier ones (including
# the constant or drift), since the model cannot separate their effects
def check_collinear(exog, trend):
    base = []
    if trend == "c":
        base.append(np.ones(len(exog)))
    elif trend == "t":
        base.append(np.arange(1, len(exog) + 1, dtype=float))

    matrix = np.column_stack(base) if base else np.empty((len(exog), 0))
    for name in exog.columns:
        attempt = np.column_stack([matrix, exog[name].to_numpy(float)])
        if np.linalg.matrix_rank(attempt) <= matrix.shape[1]:
            raise ValueError(
                f"{name} is constant or a combination of the other variables (or the {trend_name(trend)})"
            )
        matrix = attempt


def trend_name(trend):
    return {"c": "constant", "t": "drift", "n": "other terms"}[trend]


# Fits the model, collecting any warnings from statsmodels instead of printing them
def fit_model(y, exog, order, trend, notes):
    exog = exog if len(exog.columns) else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = ARIMA(y, exog=exog, order=order, trend=trend).fit()
        except (np.linalg.LinAlgError, ValueError) as error:
            raise ValueError(f"The model could not be fitted: {error}")

    messages = {str(w.message).split(".")[0] for w in caught}
    if not model.mle_retvals.get("converged", True):
        notes.append("The optimiser did not converge, so treat the estimates with caution.")
    elif any("Non-stationary" in m or "Non-invertible" in m for m in messages):
        notes.append("statsmodels found non-stationary or non-invertible starting values; try other orders.")
    if np.isnan(model.bse).any():
        notes.append("Some standard errors could not be computed, which usually means too many terms.")
    return model


# Augmented Dickey-Fuller p value for a series, or NaN if there are too few values
def adf_p(values):
    values = values.dropna()
    if len(values) < 10 or values.std() == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return adfuller(values, autolag="AIC")[1]
        except (ValueError, np.linalg.LinAlgError):
            return np.nan


# Ljung-Box test on the residuals at about a fifth of the sample, but at least one lag past
# the ARMA terms so the test has degrees of freedom left. Skips the first d residuals
def ljung_box(model, p, d, q):
    residuals = model.resid.iloc[d:].dropna()
    lag = max(p + q + 1, min(10, len(residuals) // 5))
    if lag >= len(residuals):
        return lag, np.nan
    result = acorr_ljungbox(residuals, lags=[lag], model_df=p + q)
    return lag, float(result["lb_pvalue"].iloc[0])


# Adds hints from the stationarity and residual tests to the notes
def add_hints(notes, level_p, differenced_p, ljung_box_p, d):
    if differenced_p > 0.05:
        notes.append(f"Y may still be non-stationary after {d} differences (ADF p = {differenced_p:.2f}); try a higher d.")
    if d > 0 and level_p < 0.05:
        notes.append("Y already looks stationary in levels, so it may be over-differenced; try a lower d.")
    if ljung_box_p < 0.05:
        notes.append("The residuals are still autocorrelated (Ljung-Box p < 0.05); try a higher p or q.")


# The role, variable and lag shown in the coefficient table for one of statsmodels' parameters
def describe_parameter(name, exog_labels):
    if name in exog_labels:
        return exog_labels[name]
    if name.startswith("ar.L"):
        return "AR", "Y (AR)", int(name[4:])
    if name.startswith("ma.L"):
        return "MA", "Error (MA)", int(name[4:])
    names = {"const": "Constant", "drift": "Drift", "x1": "Drift", "sigma2": "Variance"}
    return names.get(name, ""), names.get(name, name), ""


# Shows dates without a time of day, and anything else as it is
def time_text(value):
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


# Records what Forecast needs to carry the model on into new periods (see engine/fits.py): the
# orders and trend, the names of the parameters in the coefficient table's order, the Y and
# exogenous columns it was fitted on, how each exogenous column was made, and the last MAX_LAG
# rows of the X and C fields, which the lags of the first new periods reach back into
def record_fit(results, settings, model, y, exog, exog_labels, data, order, trend, series):
    columns = list(dict.fromkeys(column for role, column, lag in exog_labels.values()))
    recent = data.loc[:y.index[-1]].tail(MAX_LAG)
    fits.record(
        results, "arimax", "ARIMAX", settings["dependent"], data, [],
        order=order,
        trend=trend,
        parameters=list(model.params.index),
        y=y.to_numpy(float),
        exog=exog.to_numpy(float),
        exog_labels={name: exog_labels[name] for name in exog.columns},
        recent={column: recent[column].to_numpy(float) for column in columns},
        time=settings.get("time"),
        group=settings.get("group"),
        series=series,
    )


# Fits a regression with ARIMA(p, d, q) errors for one time series: Y on the lagged X and C
# variables. Returns the coefficient table (with fit statistics and test results in its attrs)
# and a table of the actual values, fitted values and residuals for each period
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    p, d, q = order_of(settings, "p"), order_of(settings, "d"), order_of(settings, "q")
    notes = []

    check_columns(data, settings)
    data = pick_series(data, settings)
    data, times = sort_by_time(data, settings, notes)

    exog, exog_labels = lagged_exog(data, settings)
    y, exog = trim_missing(data[settings["dependent"]].astype(float), exog, notes)
    times = times.loc[y.index]

    trend = trend_for(settings, d)
    parameters = p + q + len(exog.columns) + (trend != "n") + 1
    usable = y.notna().sum() - d
    if usable < parameters + SPARE_ROWS:
        raise ValueError(
            f"{usable} usable rows for {parameters} parameters. "
            f"Use lower orders or fewer lags, or at least {parameters + SPARE_ROWS + d} rows"
        )

    check_collinear(exog, trend)
    if exog.empty:
        notes.append("No X or C variables are chosen, so this is a plain ARIMA model.")

    model = fit_model(y.reset_index(drop=True), exog.reset_index(drop=True), (p, d, q), trend, notes)

    differenced = y
    for _ in range(d):
        differenced = differenced.diff()
    level_p, differenced_p = adf_p(y), adf_p(differenced)
    lag, ljung_box_p = ljung_box(model, p, d, q)
    add_hints(notes, level_p, differenced_p, ljung_box_p, d)

    described = [describe_parameter(name, exog_labels) for name in model.params.index]
    results = pd.DataFrame({
        "role": [role for role, variable, lag in described],
        "variable": [variable for role, variable, lag in described],
        "lag": [lag for role, variable, lag in described],
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "z_stat": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    significance.add_stars(results)
    results.attrs = {
        "observations": int(model.nobs),
        "start": time_text(times.iloc[0]),
        "end": time_text(times.iloc[-1]),
        "aic": model.aic,
        "bic": model.bic,
        "adf_p": differenced_p,
        "ljung_box_lag": lag,
        "ljung_box_p": ljung_box_p,
        "warnings": notes,
    }
    record_fit(results, settings, model, y, exog, exog_labels, data, (p, d, q), trend, chosen_series(inputs[0], settings))

    # The first d fitted values are only the model starting up, with huge residuals, so they are left blank
    fitted = pd.DataFrame({
        settings.get("time") or "row": times.to_numpy(),
        "actual": y.to_numpy(),
        "fitted": model.fittedvalues.to_numpy(),
        "residual": model.resid.to_numpy(),
    })
    fitted.loc[:d - 1, ["fitted", "residual"]] = np.nan
    if settings.get("group"):
        fitted.insert(0, settings["group"], chosen_series(inputs[0], settings))
    return [results, fitted]
