import numpy as np
import pandas as pd

from modules.arimax import module as arimax
from modules.arimax.module import (
    MAX_AR, MAX_DIFF, MAX_MA, SPARE_ROWS, chosen_series, choose_role, cycle_lag, lag_of, lag_text,
    next_series, number_text, role_of, toggle_constant, toggle_lag_zero,
)
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 3
INPUT_SPECS = [
    ("Data", "a time series table: numeric Y, X and C columns, a time column (T), and a group column (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Coefficients", "the best model's terms, as ARIMAX's Out 1, so it can go straight into Forecast"),
    ("Fitted data", "one row per period with the time, actual, fitted and residual values of the best model"),
    ("Search", "one row per model tried with p, d, q, AIC, BIC and whether it converged, best first"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream.
# lags is keyed by field name
FIELD_SETTINGS = ["dependent", "independent", "controls", "time", "group", "lags"]
DEFAULTS = {"max_p": 3, "d": "auto", "max_q": 3, "criterion": "AIC"}
CRITERIA = ["AIC", "BIC"]
SEARCH_ROWS = {
    "Max AR order (p)": ("max_p", [str(n) for n in range(MAX_AR + 1)]),
    "Differencing (d)": ("d", ["Auto"] + [str(n) for n in range(MAX_DIFF + 1)]),
    "Max MA order (q)": ("max_q", [str(n) for n in range(MAX_MA + 1)]),
}
# How sure the ADF test must be that a series is stationary before Auto stops differencing it
ADF_LEVEL = 0.05


# Opens a panel like ARIMAX's, but with the highest p and q to try instead of fixed orders,
# and d either fixed or chosen by Auto from ADF tests on Y
def on_click(node):
    panel = Panel(node, "Auto ARIMAX")
    panel.add_button("Next series", next_series)
    panel.add_button(arimax.lag_zero_label(node.settings), toggle_lag_zero)
    panel.add_button(arimax.constant_label(node.settings), toggle_constant)
    panel.add_button(criterion_label(node.settings), cycle_criterion)
    panel.preview_title = "Best Model"
    panel.refresh = show_model
    show_model(panel)
    return panel


def setting(settings, key):
    return settings.get(key, DEFAULTS[key])


def criterion_label(settings):
    return f"Pick by: {setting(settings, 'criterion')}"


def cycle_criterion(panel, button):
    settings = panel.node.settings
    current = CRITERIA.index(setting(settings, "criterion"))
    settings["criterion"] = CRITERIA[(current + 1) % len(CRITERIA)]
    button.text = criterion_label(settings)
    show_model(panel)


# Sets the highest p or q to try, or d to a number or Auto
def choose_search(panel, button):
    key = SEARCH_ROWS[button.value][0]
    panel.node.settings[key] = "auto" if button.text == "Auto" else int(button.text)
    show_model(panel)


# The search rows: the highest p, the d, and the highest q, each with its choices
def search_options(panel, settings):
    for label, (key, choices) in SEARCH_ROWS.items():
        current = str(setting(settings, key)).capitalize()
        panel.add_option(label, [(choice, choose_search, choice == current) for choice in choices])


# Lists the search settings and every variable with its role buttons, then the best model found
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
    series = chosen_series(data, settings)
    group = f"{settings.get('group')} = {series}" if series is not None else "none"
    d = setting(settings, "d")
    panel.lines = [
        f"Trying p 0-{setting(settings, 'max_p')}, d {'Auto' if d == 'auto' else d}, q 0-{setting(settings, 'max_q')}    "
        f"Y: {settings.get('dependent') or 'none'}    Time: {settings.get('time') or 'row order'}    Series: {group}",
        f"X: {', '.join(lag_text(settings, c) for c in settings.get('independent', [])) or 'none'}    "
        f"C: {', '.join(lag_text(settings, c) for c in settings.get('controls', [])) or 'none'}",
    ]

    search_options(panel, settings)

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

    if len(node.last_outputs) == OUTPUTS:
        results, fitted, search = node.last_outputs
        fit = results.attrs
        p, d, q = fit["order"]
        criterion = fit["criterion"]
        panel.lines.append(
            f"Best: ARIMA({p}, {d}, {q}) by {criterion} out of {fit['models_tried']} models    "
            f"N: {fit['observations']} ({fit['start']} to {fit['end']})    AIC: {fit['aic']:.1f}    BIC: {fit['bic']:.1f}"
        )
        panel.lines.append(
            f"ADF p (Y with d = {d}): {number_text(fit['adf_p'])}    "
            f"Ljung-Box p (lag {fit['ljung_box_lag']}): {number_text(fit['ljung_box_p'])}"
        )
        runners_up = [
            f"({row.p}, {row.d}, {row.q}) {getattr(row, criterion.lower()):.1f}"
            for row in search.iloc[1:4].itertuples()
        ]
        panel.note = " ".join([f"Next best by {criterion}: {', '.join(runners_up) or 'none'}. Out 3 lists every model tried."] + fit["warnings"])
        panel.preview = results
    else:
        panel.message = "Choose Y, T and the orders to try, then run the workflow to search"


# Differences Y until the ADF test says it is stationary, up to MAX_DIFF times.
# Returns d and the ADF p value it stopped at
def auto_d(y):
    values = y
    for d in range(MAX_DIFF + 1):
        p_value = arimax.adf_p(values)
        if np.isnan(p_value) or p_value < ADF_LEVEL or d == MAX_DIFF:
            return d, p_value
        values = values.diff()


# Y and the lagged X and C columns, prepared once the same way ARIMAX prepares them
def prepared(data, settings):
    arimax.check_columns(data, settings)
    data = arimax.pick_series(data, settings)
    data, times = arimax.sort_by_time(data, settings, [])
    exog, exog_labels = arimax.lagged_exog(data, settings)
    return arimax.trim_missing(data[settings["dependent"]].astype(float), exog, [])


# Fits ARIMA(p, d, q) with the X and C variables for every p and q up to the highest chosen.
# Orders with too few rows for their parameters, or that cannot be fitted, are left out
def search_orders(y, exog, settings, d):
    trend = arimax.trend_for(settings, d)
    usable = y.notna().sum() - d
    y, exog = y.reset_index(drop=True), exog.reset_index(drop=True)
    rows = []

    for p in range(setting(settings, "max_p") + 1):
        for q in range(setting(settings, "max_q") + 1):
            parameters = p + q + len(exog.columns) + (trend != "n") + 1
            if usable < parameters + SPARE_ROWS:
                continue
            try:
                model = arimax.fit_model(y, exog, (p, d, q), trend, [])
            except ValueError:
                continue
            rows.append({
                "p": p, "d": d, "q": q,
                "aic": model.aic,
                "bic": model.bic,
                "converged": bool(model.mle_retvals.get("converged", True)),
            })

    return pd.DataFrame(rows, columns=["p", "d", "q", "aic", "bic", "converged"])


# Searches the ARIMA orders for the one with the lowest AIC or BIC, preferring models whose
# optimiser converged, then fits it with ARIMAX so the outputs match ARIMAX's and Forecast can
# use them. Every model shares the same d and rows, so their AIC and BIC can be compared.
# Returns the best model's coefficients and fitted data, and the table of every model tried
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    y, exog = prepared(data, settings)

    d = setting(settings, "d")
    notes = []
    if d == "auto":
        d, p_value = auto_d(y)
        notes.append(f"Auto chose d = {d}, where the ADF test on Y gives p = {number_text(p_value)}.")

    search = search_orders(y, exog, settings, d)
    if search.empty:
        raise ValueError("None of the orders could be fitted. Lower the highest p and q, or use fewer lags")

    criterion = setting(settings, "criterion")
    search = search.sort_values(["converged", criterion.lower()], ascending=[False, True]).reset_index(drop=True)
    best = search.iloc[0]

    results, fitted = arimax.run(inputs, dict(settings, p=int(best["p"]), d=d, q=int(best["q"])))
    results.attrs["order"] = (int(best["p"]), d, int(best["q"]))
    results.attrs["criterion"] = criterion
    results.attrs["models_tried"] = len(search)
    results.attrs["warnings"] = notes + results.attrs["warnings"]
    return [results, fitted, search]
