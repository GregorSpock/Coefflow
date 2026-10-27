import numpy as np
import pandas as pd
import pygame
from scipy import stats

from engine import clusters, dummies, panel_data
from ui import colours
from ui.chart import BOTTOM_MARGIN, LEFT_MARGIN, draw_axes, draw_labels, padded_range
from ui.clusters import add_cluster_button
from ui.panel import Panel


# Shared by the Causal pack's modules: the panel where fields get roles, least squares and two stage
# least squares with ordinary, robust (HC1) or clustered errors, Wald tests and a line chart.
# Each module lists its roles as (letter, settings key, many) triples, where many is True for a role
# any number of fields can have, kept as a list, and False for one kept as a single field name

LEVEL = 0.05
# The usual rule of thumb for a first stage strong enough that 2SLS isn't badly biased (Staiger & Stock 1997)
STRONG_F = 10
DASH = 8


# Returns the letter of the role a field has, or None
def role_of(roles, settings, column):
    for letter, key, many in roles:
        if (column in settings.get(key, [])) if many else settings.get(key) == column:
            return letter
    return None


def clear_role(roles, settings, column):
    for letter, key, many in roles:
        if many:
            settings[key] = [c for c in settings.get(key, []) if c != column]
        elif settings.get(key) == column:
            settings[key] = None


# Gives a field the clicked role, or clears it if it already had that role. A role kept as a single
# field replaces the field it had before. The panel keeps its module's roles in panel.roles
def choose_role(panel, button):
    settings = panel.node.settings
    column = button.value
    current = role_of(panel.roles, settings, column)
    clear_role(panel.roles, settings, column)
    if current != button.text:
        for letter, key, many in panel.roles:
            if letter == button.text:
                if many:
                    settings.setdefault(key, []).append(column)
                else:
                    settings[key] = column
    panel.refresh(panel)


# Adds a row of role buttons for each field of the last run's data. allowed(column, letter) says
# which roles a field can take. Returns False if the workflow hasn't run yet
def add_role_options(panel, roles, allowed):
    node = panel.node
    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        return False
    data = node.last_inputs[0]
    for column in data.columns:
        if column == node.settings.get("cluster"):
            continue
        role = role_of(roles, node.settings, column)
        choices = [(letter, choose_role, role == letter) for letter, key, many in roles if allowed(data, column, letter)]
        if choices:
            panel.add_option(column, choices)
    return True


# The usual rule for which roles a field can take: dates none, text fields only the roles in text_roles
def default_allowed(text_roles):
    def allowed(data, column, letter):
        if column not in dummies.model_columns(data):
            return letter in ("E", "T")
        return dummies.is_numeric(data[column]) or letter in text_roles
    return allowed


# A short list of the fields in a role for the panel's info lines
def names_text(settings, key):
    value = settings.get(key)
    if isinstance(value, list):
        return ", ".join(value) or "none"
    return value or "none"


def robust_label(settings, default=False):
    return "Robust: On" if settings.get("robust", default) else "Robust: Off"


# Opens a panel with Robust and Cluster buttons. show(panel) fills it and runs again on every change
def open_panel(node, title, roles, show, robust_default=False):
    panel = Panel(node, title)
    panel.roles = roles
    panel.robust_default = robust_default
    panel.add_button(robust_label(node.settings, robust_default), toggle_robust)
    add_cluster_button(panel)
    panel.preview_title = "Results"
    panel.refresh = show
    show(panel)
    return panel


def toggle_robust(panel, button):
    settings = panel.node.settings
    settings["robust"] = not settings.get("robust", panel.robust_default)
    button.text = robust_label(settings, panel.robust_default)
    panel.refresh(panel)


# Adds an option row of choices kept in settings[key], one of which is selected
def add_choice(panel, label, key, choices, default):
    current = panel.node.settings.get(key, default)

    def choose(panel, button):
        panel.node.settings[key] = button.text
        panel.refresh(panel)

    panel.add_option(label, [(choice, choose, choice == current) for choice in choices])


# Checks the fields named in settings are in the data, that Y is numeric, and that no field has two roles
def check_fields(data, settings, keys, dependent="dependent"):
    fields = []
    for key in keys:
        value = settings.get(key)
        fields += value if isinstance(value, list) else [value] if value else []
    missing = [c for c in fields if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    repeated = sorted({c for c in fields if fields.count(c) > 1})
    if repeated:
        raise ValueError(f"{', '.join(repeated)} has more than one role")
    if settings.get(dependent) and not dummies.is_numeric(data[settings[dependent]]):
        raise ValueError(f"{settings[dependent]} is not numeric, so it cannot be the dependent variable (Y)")
    return fields


# Reads a field that must be 0/1, such as a treatment, as floats, or raises ValueError
def binary(values, name, role):
    if not dummies.is_numeric(values) or not values.isin([0, 1]).all():
        raise ValueError(f"{name} must hold only 0 and 1 to be the {role}")
    return values.astype(float)


# The groups to cluster by (or None), the cluster summary shown in the panel and the degrees of
# freedom for the t tests, which are G - 1 with clustered errors, as in Stata
def error_groups(used, column, parameters, df_resid):
    if not column:
        return None, None, df_resid
    summary = clusters.summarise(used[column])
    if summary["clusters"] < 2:
        raise ValueError(f"{column} has fewer than 2 values in the rows used, so it cannot be clustered by")
    info = dict(summary, column=column, warnings=clusters.warnings_for(summary, column, parameters))
    return pd.factorize(used[column])[0], info, summary["clusters"] - 1


# The covariance of coefficients fitted by least squares on x, from the residuals. For 2SLS x is the
# first stage fitted values and the residuals are those of the structural equation. Ordinary errors
# use df_resid, robust ones HC1 and clustered ones Stata's small-sample correction
def covariance(x, resid, df_resid, robust=False, groups=None):
    if groups is not None or not robust:
        return panel_data.covariance(x, resid, df_resid, groups)
    bread = np.linalg.inv(x.T @ x)
    meat = (x * resid[:, None] ** 2).T @ x
    return bread @ meat @ bread * len(resid) / df_resid


# Least squares of y on x. Returns the coefficients, their covariance and the residuals
def least_squares(x, y, df_resid, robust=False, groups=None):
    beta, resid = panel_data.ols(x, y)
    return beta, covariance(x, resid, df_resid, robust, groups), resid


# Two stage least squares of y on x, where z holds every instrument: the excluded instruments and
# the columns of x that are their own instruments. Returns the coefficients, the residuals y - x b
# and the first stage fitted values of x, which the covariance is built from
def two_stage(y, x, z):
    xhat = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    beta = np.linalg.lstsq(xhat, y, rcond=None)[0]
    return beta, y - x @ beta, xhat


# Wald test that the coefficients at positions are all 0. Returns the F statistic (the Wald statistic
# over the number of coefficients tested) and its p value with df_resid denominator degrees of freedom
def wald(beta, cov, positions, df_resid):
    b = beta[positions]
    statistic = float(b @ np.linalg.pinv(cov[np.ix_(positions, positions)]) @ b)
    f = statistic / len(positions)
    return f, float(stats.f.sf(f, len(positions), df_resid))


# Checks a matrix's columns are not combinations of each other, naming what to fix if they are
def check_rank(matrix, what):
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError(f"The {what} are collinear: one is a combination of the others, or doesn't vary in the rows used")


# Takes the average of each column within each group away, as fixed effects do
def demean(matrix, codes):
    return matrix - panel_data.entity_means(matrix, codes)


def number_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.3g}"


# Draws lines through points inside area. series is a list of (x values, y values, colour, label),
# drawn with a dot at each point. intervals gives (x values, lows, highs) drawn as bars around the
# first series. vertical is an x value marked with a dashed line, e.g. a cutoff, and horizontal a y value
def draw_lines(screen, area, font, series, x_label, y_label, intervals=None, vertical=None, horizontal=None, points=None):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    xs = [x for s in series for x in s[0]] + ([x for x in points[0]] if points else [])
    ys = [y for s in series for y in s[1]] + ([y for y in points[1]] if points else [])
    if intervals:
        ys += list(intervals[1]) + list(intervals[2])
    if horizontal is not None:
        ys.append(horizontal)
    ys = [y for y in ys if np.isfinite(y)]
    if not xs or not ys:
        return
    x_low, x_high = padded_range(xs)
    y_low, y_high = padded_range(ys)

    def to_screen(x, y):
        return (plot.left + (x - x_low) / (x_high - x_low) * plot.width,
                plot.bottom - (y - y_low) / (y_high - y_low) * plot.height)

    def dashed(start, end):
        length = max(1, int(np.hypot(end[0] - start[0], end[1] - start[1])))
        for step in range(0, length, DASH * 2):
            a = step / length
            b = min(step + DASH, length) / length
            pygame.draw.line(screen, colours.TEXT, (start[0] + a * (end[0] - start[0]), start[1] + a * (end[1] - start[1])),
                             (start[0] + b * (end[0] - start[0]), start[1] + b * (end[1] - start[1])), 1)

    draw_axes(screen, plot, font, x_low, x_high, y_low, y_high)
    screen.set_clip(plot)
    if vertical is not None:
        dashed(to_screen(vertical, y_low), to_screen(vertical, y_high))
    if horizontal is not None:
        dashed(to_screen(x_low, horizontal), to_screen(x_high, horizontal))
    if points:
        for x, y in zip(*points):
            if np.isfinite(y):
                pygame.draw.circle(screen, colours.CHART_POINT, to_screen(x, y), 3)
    if intervals:
        for x, low, high in zip(*intervals):
            if np.isfinite(low) and np.isfinite(high):
                pygame.draw.line(screen, colours.CHART_POINT, to_screen(x, low), to_screen(x, high), 2)
    for x_values, y_values, colour, label in series:
        screen_points = [to_screen(x, y) for x, y in zip(x_values, y_values) if np.isfinite(y)]
        if len(screen_points) > 1:
            pygame.draw.lines(screen, colour, False, screen_points, 3)
        if label is not None:
            for point in screen_points:
                pygame.draw.circle(screen, colour, point, 4)
    screen.set_clip(None)

    # A legend in the top left corner for the series that have a label
    y = plot.top + 6
    for x_values, y_values, colour, label in series:
        if label:
            pygame.draw.line(screen, colour, (plot.left + 8, y + 8), (plot.left + 28, y + 8), 3)
            text = font.render(label, True, colours.TEXT)
            screen.blit(text, (plot.left + 34, y))
            y += text.get_height() + 4
    draw_labels(screen, plot, font, x_label, y_label)
