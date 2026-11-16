import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL

from engine import draw, estimates, roles
from engine.chart import draw_lines
from modules.arimax.module import read_time


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a time series table with numeric Y fields to filter, a time field (T), and a group field (G) if it holds several "
             "series, each of which is filtered on its own"),
]
OUTPUT_SPECS = [
    ("Filtered data", "the rows, in time order, plus <field>_trend and <field>_cycle for each Y (Baxter-King leaves the first and "
                      "last K periods blank). STL gives <field>_trend, <field>_seasonal, <field>_adjusted (seasonally adjusted) "
                      "and <field>_remainder instead"),
]

METHODS = [("hp", "Hodrick-Prescott"), ("bk", "Baxter-King"), ("cf", "Christiano-Fitzgerald"), ("hamilton", "Hamilton"), ("stl", "STL seasonal")]
FREQUENCIES = [("quarterly", "Quarterly"), ("annual", "Annual"), ("monthly", "Monthly")]
# Each frequency's usual settings: HP's lambda (Ravn and Uhlig 2002), the business cycle band in
# periods (6 to 32 quarters) with Baxter-King's K, Hamilton's horizon h and lags p, and the season
DEFAULTS = {
    "annual": {"lambda": 6.25, "low": 2, "high": 8, "K": 3, "h": 2, "p": 2, "period": None},
    "quarterly": {"lambda": 1600, "low": 6, "high": 32, "K": 12, "h": 8, "p": 4, "period": 4},
    "monthly": {"lambda": 129600, "low": 18, "high": 96, "K": 36, "h": 24, "p": 12, "period": 12},
}
NAMES = {"lambda": "lambda", "low": "low", "high": "high", "k": "K", "h": "h", "p": "p", "period": "period"}


def chart_label(settings):
    return f"Chart: {settings.get('chart_field') or 'first Y'}"


def next_field(panel, button):
    roles.step(panel, "chart_field", panel.node.settings.get("dependent") or [])
    button.text = chart_label(panel.node.settings)


# Draws the chosen Y with its trend (or seasonally adjusted series) for the first series
def filter_chart(node):
    if not node.last_outputs:
        return None
    table = node.last_outputs[0]
    # Only the fields filtered in the last run have their trend in the table
    fields = [f for f in node.settings.get("dependent") or [] if f in table and f"{f}_trend" in table]
    if not fields:
        return None
    field = node.settings.get("chart_field") if node.settings.get("chart_field") in fields else fields[0]
    group = node.settings.get("group")
    if group and group in table:
        table = table[table[group].astype(str) == str(table[group].iloc[0])]
    time = node.settings.get("time")
    x = np.arange(len(table), dtype=float)
    if time and time in table and pd.api.types.is_numeric_dtype(table[time]):
        x = table[time].to_numpy(float)
    second = f"{field}_adjusted" if f"{field}_adjusted" in table else f"{field}_trend"
    series = []
    for name in (field, second):
        if name not in table:
            continue
        values = table[name].to_numpy(float)
        keep = ~np.isnan(values)
        if keep.any():
            series.append((name, x[keep], values[keep]))
    if not series:
        return None

    def chart(painter, area):
        draw_lines(painter, area, draw.font(22), series, time or "row")
    return chart


SPEC = roles.Spec(
    "TS Filter",
    [
        roles.Role("Y", "dependent", many=True),
        roles.Role("T", "time", kinds="any"),
        roles.Role("G", "group", kinds="any"),
    ],
    choices=[roles.Choice("method", "Filter", METHODS), roles.Choice("frequency", "Data", FREQUENCIES)],
    text=("parameters", "Settings (blank for the usual ones), e.g. lambda=100 or low=6, high=32, K=12", ""),
    buttons=[(chart_label, next_field)],
    chart=filter_chart,
    hint="Choose the Y fields and the time field (T), then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings() + ["chart_field"]


def on_click(node):
    return roles.open_panel(node, SPEC)


# The frequency's usual settings with any typed ones on top, e.g. "lambda=100, K=8"
def parameters_of(settings):
    values = dict(DEFAULTS[settings["frequency"]])
    for part in str(settings.get("parameters") or "").replace(";", ",").split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Write settings as name=value, e.g. lambda=1600, not {part.strip()}")
        name, value = [p.strip() for p in part.split("=", 1)]
        key = NAMES.get(name.lower())
        if key is None:
            raise ValueError(f"{name} isn't a setting. Use lambda, low, high, K, h, p or period")
        try:
            values[key] = float(value)
        except ValueError:
            raise ValueError(f"{name} must be a number, not {value}")
    for key in ("K", "h", "p", "period"):
        if values.get(key) is not None:
            values[key] = int(values[key])
    return values


# Hamilton's (2018) alternative to the HP filter: regress y at t + h on y at t, t - 1 ... t - p + 1.
# The fitted value is the trend and the residual the cycle, which avoids HP's spurious cycles
def hamilton(values, h, p):
    n = len(values)
    trend = np.full(n, np.nan)
    rows = range(p - 1, n - h)
    if len(rows) < p + 5:
        raise ValueError(f"The series is too short for Hamilton's filter with h = {h} and p = {p}")
    x = np.column_stack([np.ones(len(rows))] + [values[[t - j for t in rows]] for j in range(p)])
    y = values[[t + h for t in rows]]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    trend[[t + h for t in rows]] = x @ beta
    return trend, values - trend


# Filters one series. Returns a dict of new column suffix -> values
def filter_one(values, method, parameters):
    n = len(values)
    if method == "hp":
        cycle, trend = hpfilter(values, lamb=parameters["lambda"])
        return {"trend": trend, "cycle": cycle}
    if method == "bk":
        K = parameters["K"]
        if n <= 2 * K + 2:
            raise ValueError(f"Baxter-King needs more than {2 * K + 2} periods with K = {K}")
        cycle = np.full(n, np.nan)
        cycle[K:n - K] = bkfilter(values, low=parameters["low"], high=parameters["high"], K=K)
        return {"trend": values - cycle, "cycle": cycle}
    if method == "cf":
        cycle, trend = cffilter(values, low=parameters["low"], high=parameters["high"], drift=True)
        return {"trend": trend, "cycle": cycle}
    if method == "hamilton":
        trend, cycle = hamilton(values, parameters["h"], parameters["p"])
        return {"trend": trend, "cycle": cycle}
    period = parameters.get("period")
    if not period or period < 2:
        raise ValueError("STL needs a season: choose quarterly or monthly data, or type period=... (e.g. 7 for daily data)")
    if n < 2 * period + 1:
        raise ValueError(f"STL needs at least two full seasons ({2 * period + 1} periods)")
    result = STL(values, period=period, robust=True).fit()
    return {"trend": result.trend, "seasonal": result.seasonal, "adjusted": values - result.seasonal, "remainder": result.resid}


# Splits each Y into a smooth trend and a cycle, or for STL into trend, seasonal and remainder
# parts (seasonal adjustment). Each series of a group field is filtered on its own
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y",))
    fields = settings["dependent"]
    method = settings["method"]
    parameters = parameters_of(settings)
    group = settings.get("group")
    time = settings.get("time")
    notes = []
    if not time:
        notes.append("No time field (T) is chosen, so the rows are taken in their current order.")

    parts = []
    groups = data.groupby(data[group].astype(str), sort=True) if group else [(None, data)]
    for name, part in groups:
        if time:
            order = read_time(part[time]).sort_values(kind="stable").index
            part = part.loc[order]
        part = part.copy()
        for field in fields:
            values = part[field].astype(float)
            present = values.notna().to_numpy()
            if not present.any():
                continue
            positions = np.flatnonzero(present)
            inside = slice(positions[0], positions[-1] + 1)
            run_values = values.iloc[inside]
            if run_values.isna().any():
                where = f" in {group} = {name}" if group else ""
                raise ValueError(f"{field} has missing values inside the series{where}. Fill them first (Missing Values can interpolate)")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pieces = filter_one(run_values.to_numpy(float), method, parameters)
            for suffix, piece in pieces.items():
                column = np.full(len(part), np.nan)
                column[inside] = piece
                part[f"{field}_{suffix}"] = column
        parts.append(part)
    output = pd.concat(parts) if parts else data.copy()

    used = {k: v for k, v in parameters.items() if v is not None and k in {
        "hp": ["lambda"], "bk": ["low", "high", "K"], "cf": ["low", "high"], "hamilton": ["h", "p"], "stl": ["period"]}[method]}
    summary = [f"{dict(METHODS)[method]}    " + ", ".join(f"{k} = {v:g}" for k, v in used.items())
               + (f"    {output[group].nunique()} series" if group else "")]
    if method == "hp" and settings["frequency"] != "annual" and parameters["lambda"] < 100:
        notes.append("A small lambda for this frequency gives a trend that follows the data closely.")
    if method in ("hp", "cf"):
        notes.append("Two-sided filters use future values, so the last few periods' trend can change as data arrives.")
    for field in fields:
        cycle = f"{field}_cycle"
        if cycle in output:
            spread = output[cycle].std()
            summary.append(f"{field}: cycle standard deviation {estimates.number_text(spread)}")
    output.attrs = {"summary": summary, "notes": notes}
    return [output]
