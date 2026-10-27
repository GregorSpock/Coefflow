import math
import re

import numpy as np
import pandas as pd
import pygame

from ui import colours
from ui.chart import BOTTOM_MARGIN, LEFT_MARGIN, draw_axes, draw_labels
from ui.panel import Panel


# Shared by the Distributions pack's modules. Each module describes its distribution with a
# Distribution, and its panel has one line of text setting the parameters, e.g. "mean = 100, sd = 15".
# Besides the parameters the line can hold:
#   x     a value to give the probabilities of, e.g. P(X <= x)
#   q     a probability to give the value of, e.g. q = 0.975 gives the x with P(X <= x) = 0.975
#   size  how many values Sample draws, DEFAULT_SIZE if not given
#   seed  a whole number that makes Sample draw the same values every run
# Anything not typed takes its default. Settings are separated by commas or semicolons
EXTRAS = ["x", "q", "size", "seed"]
MODES = ["Table", "Sample"]
VIEWS = ["Chart", "Data"]

DEFAULT_SIZE = 100
MAX_SIZE = 1_000_000
# The table covers the values between the TAIL and 1 - TAIL quantiles, leaving out the far tails
TAIL = 0.0001
CURVE_POINTS = 201
# A continuous table reaches at most this many interquartile ranges either side of the median
MAX_SPREAD = 10
MAX_BINS = 60
# The chart's y axis reaches at most this many times the 95th percentile of the heights it draws,
# so a density shooting up at one end, like the chi-square with 1 df, doesn't flatten the rest
MAX_PEAK = 1.5
BAR_WIDTH = 0.8


class Distribution:
    # name is shown on the panel and names the column Sample makes, e.g. "Student t" gives "student_t".
    # discrete is True for whole number values, which have probabilities rather than densities.
    # parameters is a list of (name, default), and aliases gives other names each can be typed as.
    # make(values) gives the scipy.stats distribution for a dict of parameter values, raising
    # ValueError if they don't make sense
    def __init__(self, name, discrete, parameters, make, aliases=None):
        self.name = name
        self.discrete = discrete
        self.parameters = parameters
        self.make = make
        self.aliases = aliases or {}
        self.column = re.sub(r"\W+", "_", name.lower())

    def default_text(self):
        return ", ".join(f"{name} = {default:g}" for name, default in self.parameters)

    # Each name a setting can be typed as, in lower case -> the setting it means
    def names(self):
        names = {key: key for key in [name for name, default in self.parameters] + EXTRAS}
        for name, others in self.aliases.items():
            names.update({other.lower(): name for other in others})
        return names


# A setting that must be a whole number of at least low
def whole(values, name, low):
    value = values[name]
    if value != int(value) or value < low:
        raise ValueError(f"{name} must be a whole number of at least {low}")
    return int(value)


# Reads the panel's line of text into a dict of setting -> number, with defaults for the parameters
# not typed. Raises ValueError, saying what to fix, if the text or a value can't be used
def parse(distribution, text):
    names = distribution.names()
    values = dict(distribution.parameters)

    for part in re.split(r"[,;]", text):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Write each setting as name = value, e.g. {distribution.default_text()}")
        key, value = (s.strip() for s in part.split("=", 1))
        name = names.get(key.lower())
        if name is None:
            allowed = ", ".join([name for name, default in distribution.parameters] + EXTRAS)
            raise ValueError(f"{key} isn't a setting of the {distribution.name.lower()}: use {allowed}")
        try:
            values[name] = float(value)
        except ValueError:
            raise ValueError(f"{key} must be a number, not {value or 'blank'}") from None
        if not math.isfinite(values[name]):
            raise ValueError(f"{key} must be a finite number")

    if "size" in values and not 1 <= whole(values, "size", 1) <= MAX_SIZE:
        raise ValueError(f"size can be at most {MAX_SIZE:,}")
    if "seed" in values:
        whole(values, "seed", 0)
    if "q" in values and not 0 < values["q"] < 1:
        raise ValueError("q must be a probability between 0 and 1")
    if "x" in values and distribution.discrete and values["x"] != int(values["x"]):
        raise ValueError(f"x must be a whole number for the {distribution.name.lower()}")
    return values


# The distribution's probabilities across its likely values: k, probability and cdf for a discrete
# one, or x, density and cdf at evenly spaced points for a continuous one. cdf is P(X <= value)
def table(distribution, dist):
    # A distribution with only one possible value, like the geometric with p = 1, divides by zero here
    with np.errstate(divide="ignore"):
        low, high = dist.ppf(TAIL), dist.isf(TAIL)
    if distribution.discrete:
        low = max(low, dist.support()[0])
        high = max(high, low)
        k = np.arange(int(low), int(high) + 1)
        return pd.DataFrame({"k": k, "probability": dist.pmf(k), "cdf": dist.cdf(k)})
    # Heavy tails, like the t with 1 df, would stretch the table far beyond where the values bunch up
    median, spread = dist.median(), dist.ppf(0.75) - dist.ppf(0.25)
    low, high = max(low, median - MAX_SPREAD * spread), min(high, median + MAX_SPREAD * spread)
    x = np.linspace(low, high, CURVE_POINTS)
    return pd.DataFrame({"x": x, "density": dist.pdf(x), "cdf": dist.cdf(x)})


# size random draws as one column named after the distribution, the same draws every time if seed is set
def sample(distribution, dist, values):
    size = int(values.get("size", DEFAULT_SIZE))
    seed = int(values["seed"]) if "seed" in values else None
    draws = dist.rvs(size=size, random_state=np.random.default_rng(seed))
    return pd.DataFrame({distribution.column: draws})


# Reads the settings and gives the table or the sample, whichever the mode asks for
def make_output(distribution, settings):
    values = parse(distribution, settings.get("text", distribution.default_text()))
    dist = distribution.make(values)
    if settings.get("mode", "Table") == "Sample":
        return values, dist, sample(distribution, dist, values)
    return values, dist, table(distribution, dist)


def p_text(p):
    return "< 0.0001" if 0 < p < 0.0001 else f"{p:.4f}"


# A mean or sd, which some distributions don't have (the t with 1 df) or have as infinite (the t with 2 df)
def moment_text(value):
    if np.isnan(value):
        return "undefined"
    return "infinite" if np.isinf(value) else f"{value:.4g}"


# The mean and sd, then the probabilities for x and the value for q if they were typed
def describe(distribution, dist, values, mode):
    # scipy works out higher moments on the way, which can divide by zero when only one value is possible
    with np.errstate(all="ignore"):
        mean, sd = dist.mean(), dist.std()
    text =f"Mean = {moment_text(mean)}, sd = {moment_text(sd)}. "

    if "x" in values:
        x = values["x"]
        if distribution.discrete:
            k = int(x)
            text += f"P(X = {k}) = {p_text(dist.pmf(k))}, P(X <= {k}) = {p_text(dist.cdf(k))}, P(X >= {k}) = {p_text(dist.sf(k - 1))}. "
        else:
            text += f"P(X <= {x:g}) = {p_text(dist.cdf(x))}, P(X > {x:g}) = {p_text(dist.sf(x))}. "
        if np.isfinite(mean) and np.isfinite(sd) and sd > 0:
            text += f"x is {(x - mean) / sd:.3g} sd from the mean. "
    if "q" in values:
        q = values["q"]
        if distribution.discrete:
            text += f"The smallest k with P(X <= k) >= {q:g} is {dist.ppf(q):.0f}. "
        else:
            text += f"The x with P(X <= x) = {q:g} is {dist.ppf(q):.4g}. "
    if "x" not in values and "q" not in values:
        text += "Add x = a value for its probabilities, or q = a probability for its value. "

    if mode == "Sample" and "seed" not in values:
        text += "Each run draws a new sample; add seed = a whole number to draw the same one every run."
    return text.strip()


# The bars of a histogram of the draws between low and high, as (start, end, height). Heights are
# shares of all the draws for a discrete distribution and densities for a continuous one, so they
# line up with its curve. Draws outside the range, far out in a heavy tail, aren't drawn
def sample_bars(distribution, draws, low, high):
    if distribution.discrete:
        shares = pd.Series(draws).value_counts(normalize=True).sort_index()
        return [(k - BAR_WIDTH / 2, k + BAR_WIDTH / 2, share) for k, share in shares.items() if low <= k <= high]
    inside = draws[(draws >= low) & (draws <= high)]
    if inside.empty:
        return []
    edges = np.histogram_bin_edges(inside, bins="auto")
    if len(edges) - 1 > MAX_BINS:
        edges = np.histogram_bin_edges(inside, bins=MAX_BINS)
    counts, edges = np.histogram(inside, bins=edges)
    heights = counts / (len(draws) * np.diff(edges))
    return list(zip(edges[:-1], edges[1:], heights))


# Draws the distribution inside area: bars for each value of a discrete one, or a curve for a
# continuous one, with the area up to x filled in and a line at x if x was typed.
# With draws, a histogram of them is drawn under the distribution's curve instead
def draw_chart(screen, area, font, distribution, dist, x=None, draws=None):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    points = table(distribution, dist)
    xs, ys = points.iloc[:, 0].to_numpy(float), points.iloc[:, 1].to_numpy(float)
    x_low, x_high = xs[0], xs[-1]
    if distribution.discrete:
        x_low, x_high = x_low - 0.5, x_high + 0.5
    if x_low == x_high:
        x_low, x_high = x_low - 0.5, x_high + 0.5
    bars = sample_bars(distribution, draws, x_low, x_high) if draws is not None else []

    heights = np.array([*ys, *(height for start, end, height in bars)])
    y_high = min(heights.max(), MAX_PEAK * np.percentile(heights, 95)) * 1.08

    # Converts a point on the chart to a position on screen. Heights far above the top are brought
    # down to just above it, as a density near an end can be too big to draw
    def to_screen(x_value, y_value):
        y_value = min(y_value, y_high * 2)
        return plot.left + (x_value - x_low) / (x_high - x_low) * plot.width, plot.bottom - y_value / y_high * plot.height

    # Draws a bar from start to end along the x axis, up to height
    def draw_bar(start, end, height, colour):
        left, top = to_screen(start, height)
        right = to_screen(end, 0)[0]
        pygame.draw.rect(screen, colour, pygame.Rect(round(left), round(top), max(1, round(right) - round(left)), plot.bottom - round(top)))

    draw_axes(screen, plot, font, x_low, x_high, 0, y_high)
    screen.set_clip(plot)

    if bars:
        for start, end, height in bars:
            draw_bar(start, end, height, colours.CHART_POINT)
        curve = [to_screen(a, b) for a, b in zip(xs, ys)]
        if len(curve) > 1:
            pygame.draw.lines(screen, colours.CHART_LINE, False, curve, 3)
        if distribution.discrete and len(curve) <= plot.width / 3:
            for point in curve:
                pygame.draw.circle(screen, colours.CHART_LINE, point, 4)
    elif distribution.discrete and len(xs) <= plot.width / 3:
        for k, p in zip(xs, ys):
            colour = colours.CHART_LINE if x is not None and k <= x else colours.CHART_POINT
            draw_bar(k - BAR_WIDTH / 2, k + BAR_WIDTH / 2, p, colour)
    else:
        if x is not None and x > xs[0]:
            under = [to_screen(xs[0], 0)] + [to_screen(a, b) for a, b in zip(xs, ys) if a <= x]
            under += [to_screen(x, np.interp(x, xs, ys)), to_screen(x, 0)]
            pygame.draw.polygon(screen, colours.CHART_POINT, under)
        pygame.draw.lines(screen, colours.CHART_LINE, False, [to_screen(a, b) for a, b in zip(xs, ys)], 3)

    if x is not None:
        line_x = to_screen(x, 0)[0]
        pygame.draw.line(screen, colours.TEXT, (line_x, plot.top), (line_x, plot.bottom), 2)
    screen.set_clip(None)

    y_label = "Probability" if distribution.discrete else "Density"
    if bars:
        y_label = "Share of draws" if distribution.discrete else "Density of draws"
    draw_labels(screen, plot, font, "k" if distribution.discrete else "x", y_label)


# Opens a panel with buttons for the mode (a table of probabilities or a random sample) and the view
# (a chart or the data), and a line of text for the parameters
def open_panel(node, distribution):
    panel = Panel(node, distribution.name)
    panel.distribution = distribution
    panel.view = "Chart"
    for label in MODES:
        panel.add_button(label, choose_mode).selected_colour = "TAB_SELECTED"
    for label in VIEWS:
        panel.add_button(label, choose_view).selected_colour = "TAB_SELECTED"
    text = node.settings.get("text", distribution.default_text())
    panel.add_text_field("Set", text, change_text, blocked="", fill=True)
    panel.refresh = show_distribution
    show_distribution(panel)
    return panel


def choose_mode(panel, button):
    panel.node.settings["mode"] = button.text
    show_distribution(panel)


def choose_view(panel, button):
    panel.view = button.text
    show_distribution(panel)


def change_text(panel, text):
    panel.node.settings["text"] = text
    show_distribution(panel)


# Shows the parameters and mode, then the chart or the data, with the probabilities in the note.
# A problem with the settings is shown in the note instead
def show_distribution(panel):
    distribution = panel.distribution
    settings = panel.node.settings
    mode = settings.get("mode", "Table")
    for button in panel.buttons:
        button.selected = button.text in (mode, panel.view)
    panel.preview_title = panel.view
    panel.preview = None
    panel.chart = None
    panel.lines = [f"Mode: {mode}"]

    try:
        values, dist, data = make_output(distribution, settings)
    except ValueError as error:
        panel.note = str(error)
        panel.message = "Fix the settings to see the distribution"
        return

    shape = ", ".join(f"{name} = {values[name]:g}" for name, default in distribution.parameters)
    detail = f"{len(data):,} random draws" if mode == "Sample" else f"{len(data):,} rows of probabilities"
    panel.lines = [f"{distribution.name}({shape})    Mode: {mode}, {detail}"]
    panel.note = describe(distribution, dist, values, mode)
    panel.message = ""

    if panel.view == "Data":
        panel.preview = data
        return

    draws = data.iloc[:, 0] if mode == "Sample" else None

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_chart(screen, area, panel.table_font, distribution, dist, values.get("x"), draws)

    panel.chart = draw


def run(distribution, settings):
    return [make_output(distribution, settings)[2]]
