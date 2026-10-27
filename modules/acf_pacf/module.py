import warnings

import numpy as np
import pandas as pd
import pygame
from statsmodels.tsa.stattools import acf, pacf

from modules.arimax import module as arimax
from modules.arimax.module import MAX_DIFF, chosen_series, next_series
from ui.chart import draw_correlogram
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a time series table: a numeric column to study (Y), a time column (T), and a group column (G) if it holds several series"),
]
OUTPUT_SPECS = [
    ("Correlations", "one row per lag with acf, pacf, the significance bound and the Ljung-Box p value up to that lag"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["field", "time", "group"]
ROLE_KEYS = {"Y": "field", "T": "time", "G": "group"}
LAG_CHOICES = [12, 24, 36, 48]
DEFAULT_LAGS = 24
# How many standard errors a correlation must be from zero to count as significant (5% level)
Z = 1.96
# An ACF still above this after several lags is decaying slowly, a sign the series needs differencing
SLOW_DECAY = 0.5
SLOW_DECAY_LAG = 6


# Opens a panel to choose the series and how many times to difference it, showing its
# autocorrelations (ACF) and partial autocorrelations (PACF) as bar charts
def on_click(node):
    panel = Panel(node, "ACF/PACF")
    panel.add_button("Next series", next_series)
    panel.add_button(lags_label(node.settings), cycle_lags)
    panel.preview_title = "Correlogram"
    panel.refresh = show_correlogram
    show_correlogram(panel)
    return panel


def lags_label(settings):
    return f"Lags: {settings.get('lags', DEFAULT_LAGS)}"


def cycle_lags(panel, button):
    settings = panel.node.settings
    current = LAG_CHOICES.index(settings.get("lags", DEFAULT_LAGS))
    settings["lags"] = LAG_CHOICES[(current + 1) % len(LAG_CHOICES)]
    button.text = lags_label(settings)
    show_correlogram(panel)


def choose_d(panel, button):
    panel.node.settings["d"] = int(button.text)
    show_correlogram(panel)


# Returns "Y", "T", "G" or None for a field
def role_of(settings, column):
    for role, key in ROLE_KEYS.items():
        if settings.get(key) == column:
            return role
    return None


# Gives a field the clicked role, or clears it if it already had that role.
# Each role belongs to one field, so choosing a new one replaces the old
def choose_role(panel, button):
    settings = panel.node.settings
    current = role_of(settings, button.value)
    for key in ROLE_KEYS.values():
        if settings.get(key) == button.value:
            settings[key] = None
    if current != button.text:
        settings[ROLE_KEYS[button.text]] = button.value
    if button.text == "G":
        settings["series"] = None
    show_correlogram(panel)


# Lists the differencing choices and every field with its role buttons, then draws the
# ACF above the PACF of the last run, with hints on the orders they suggest
def show_correlogram(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.chart = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = node.last_inputs[0]
    series = chosen_series(data, settings)
    group = f"{settings.get('group')} = {series}" if series is not None else "none"
    d = settings.get("d", 0)
    panel.lines = [
        f"Y: {settings.get('field') or 'none'}    Differenced: {d} times    "
        f"Time: {settings.get('time') or 'row order'}    Series: {group}",
    ]

    panel.add_option("Differencing (d)", [(str(n), choose_d, n == d) for n in range(MAX_DIFF + 1)])
    numeric = data.select_dtypes("number").columns
    for column in data.columns:
        role = role_of(settings, column)
        roles = ["Y", "T", "G"] if column in numeric else ["T", "G"]
        panel.add_option(column, [(r, choose_role, role == r) for r in roles])

    if not node.last_outputs:
        panel.message = "Choose Y, and T if the rows aren't in time order, then run the workflow"
        return

    table = node.last_outputs[0]
    fit = table.attrs
    panel.lines.append(
        f"N: {fit['observations']}    Significance bound: +/-{fit['bound']:.3f}    "
        f"Ljung-Box p (lag {len(table)}): {arimax.number_text(table['ljung_box_p'].iloc[-1])}"
    )
    panel.note = " ".join(fit["hints"])

    # Draws the ACF in the top half of the space the panel gives it and the PACF in the bottom half
    def draw(screen, area):
        half = area.height // 2
        draw_correlogram(screen, pygame.Rect(area.left, area.top, area.width, half), panel.table_font, table["acf"], fit["bound"], "ACF")
        draw_correlogram(screen, pygame.Rect(area.left, area.top + half, area.width, half), panel.table_font, table["pacf"], fit["bound"], "PACF")

    panel.chart = draw


# The number of lags in a row, from lag 1, whose correlation is outside the bound
def leading_spikes(values, bound):
    count = 0
    for value in values:
        if abs(value) <= bound:
            break
        count += 1
    return count


# Plain-English hints on the orders the correlations point to. A PACF that cuts off after
# k lags suggests AR(k), an ACF that cuts off after k lags suggests MA(k), and an ACF that
# dies away slowly suggests the series needs differencing first
def order_hints(table, bound, d):
    acf_spikes = leading_spikes(table["acf"], bound)
    pacf_spikes = leading_spikes(table["pacf"], bound)
    hints = []

    if len(table) >= SLOW_DECAY_LAG and table["acf"].iloc[SLOW_DECAY_LAG - 1] > SLOW_DECAY:
        hints.append(
            f"The ACF is still {table['acf'].iloc[SLOW_DECAY_LAG - 1]:.2f} at lag {SLOW_DECAY_LAG}, dying away slowly, "
            f"so the series may need differencing: try d = {d + 1}."
        )
    elif acf_spikes == 0 and pacf_spikes == 0:
        later = table["lag"][(table["acf"].abs() > bound) | (table["pacf"].abs() > bound)].tolist()
        if later:
            hints.append(
                f"Lag 1 isn't significant, but lag {', '.join(map(str, later))} {'is' if len(later) == 1 else 'are'}, "
                f"so there is a pattern further back: try a higher p or q, or that lag of an X variable."
            )
        else:
            hints.append("No lags are significant, so the series looks close to white noise at this d.")
    else:
        hints.append(
            f"The PACF has {pacf_spikes} significant lag{'s' if pacf_spikes != 1 else ''} before it cuts off, "
            f"suggesting p = {pacf_spikes}; the ACF has {acf_spikes}, suggesting q = {acf_spikes}. "
            f"Whichever cuts off more sharply is usually the better guide, and Auto ARIMAX can check both."
        )

    seasonal = [lag for lag in (12, 24) if lag <= len(table) and abs(table["acf"].iloc[lag - 1]) > bound]
    if seasonal and acf_spikes < 12:
        hints.append(f"There are spikes at lag {' and '.join(map(str, seasonal))}, a sign of a seasonal pattern.")
    return hints


# Works out the autocorrelation and partial autocorrelation of the chosen field, in time order,
# after differencing it d times. Returns one row per lag with the Ljung-Box p value for all the lags up to it
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    field = settings.get("field")
    if not field:
        raise ValueError("Choose the field to study (Y)")
    missing = [c for c in [field, settings.get("time"), settings.get("group")] if c and c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    if not pd.api.types.is_numeric_dtype(data[field]):
        raise ValueError(f"{field} is not numeric")

    data = arimax.pick_series(data, settings)
    data, times = arimax.sort_by_time(data, settings, [])
    values = data[field].astype(float)
    d = settings.get("d", 0)
    for _ in range(d):
        values = values.diff()
    values = values.dropna()

    lags = min(settings.get("lags", DEFAULT_LAGS), len(values) // 2 - 1)
    if lags < 1:
        raise ValueError(f"{len(values)} values is too few to work out correlations")
    if values.std() == 0:
        raise ValueError(f"{field} doesn't change, so it has no correlations")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autocorrelations, q_stats, q_p_values = acf(values, nlags=lags, qstat=True)
        partials = pacf(values, nlags=lags, method="ywm")

    bound = Z / np.sqrt(len(values))
    table = pd.DataFrame({
        "lag": range(1, lags + 1),
        "acf": autocorrelations[1:],
        "pacf": partials[1:],
        "bound": bound,
        "ljung_box_p": q_p_values,
    })
    table.attrs = {
        "observations": len(values),
        "bound": bound,
        "hints": order_hints(table, bound, d),
    }
    return [table]
