import math

import pygame

from ui import colours
from ui.panel import fit_text


TICKS = 5
LEFT_MARGIN = 70
BOTTOM_MARGIN = 50
POINT_RADIUS = 3
# A correlogram's lags come in twelves, so four ticks fall on whole lags
CORRELOGRAM_TICKS = 4
DASH = 8
# Bars fill this share of their slot, leaving a gap between neighbours
BAR_SHARE = 0.7
# Category names narrower than this can't be read, so names are skipped until each has this much room
MIN_LABEL_WIDTH = 60


# Returns the lowest and highest value with a little space added either side
def padded_range(values):
    low, high = min(values), max(values)
    if low == high:
        return low - 1, high + 1
    space = (high - low) * 0.05
    return low - space, high + space


# Shows large numbers like years in full, and everything else to 3 significant figures
def tick_text(value):
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.3g}"


# Draws a scatter chart of y against x inside area, with an optional fitted line
# given as (intercept, slope)
def draw_scatter(screen, area, font, x_values, y_values, x_label, y_label, line=None):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    x_low, x_high = padded_range(x_values)
    y_low, y_high = padded_range(y_values)

    # Converts a data point to a position on screen
    def to_screen(x, y):
        screen_x = plot.left + (x - x_low) / (x_high - x_low) * plot.width
        screen_y = plot.bottom - (y - y_low) / (y_high - y_low) * plot.height
        return screen_x, screen_y

    draw_axes(screen, plot, font, x_low, x_high, y_low, y_high)

    screen.set_clip(plot)
    for x, y in zip(x_values, y_values):
        pygame.draw.circle(screen, colours.CHART_POINT, to_screen(x, y), POINT_RADIUS)

    if line:
        intercept, slope = line
        start = to_screen(x_low, intercept + slope * x_low)
        end = to_screen(x_high, intercept + slope * x_high)
        pygame.draw.line(screen, colours.CHART_LINE, start, end, 3)
    screen.set_clip(None)

    draw_labels(screen, plot, font, x_label, y_label)


# Draws a correlogram: a bar from zero for each lag's correlation, on a scale from -1 to 1, with
# dashed lines at plus and minus bound, outside which a correlation is significant
def draw_correlogram(screen, area, font, values, bound, label):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    lags = len(values)

    # Converts a lag and a correlation to a position on screen
    def to_screen(lag, value):
        return plot.left + lag / lags * plot.width, plot.bottom - (value + 1) / 2 * plot.height

    draw_axes(screen, plot, font, 0, lags, -1, 1, CORRELOGRAM_TICKS)
    pygame.draw.line(screen, colours.TEXT, to_screen(0, 0), to_screen(lags, 0))

    for level in [bound, -bound]:
        for start in range(0, plot.width, DASH * 2):
            y = to_screen(0, level)[1]
            pygame.draw.line(screen, colours.CHART_LINE, (plot.left + start, y), (plot.left + min(start + DASH, plot.width), y), 2)

    width = max(2, int(plot.width / lags * 0.4))
    for lag, value in enumerate(values, start=1):
        x, top = to_screen(lag, max(value, 0))
        bottom = to_screen(lag, min(value, 0))[1]
        pygame.draw.rect(screen, colours.CHART_POINT, (x - width / 2, top, width, max(1, bottom - top)))

    draw_labels(screen, plot, font, "Lag", label)


# Draws a bar for each category's value, from zero, with the category names under the bars.
# Names that don't fit their slot are shortened, and if the slots are too narrow to read
# only every few names are written
def draw_bars(screen, area, font, names, values, x_label, y_label):
    plot = pygame.Rect(area.left + LEFT_MARGIN, area.top + 10, area.width - LEFT_MARGIN - 10, area.height - BOTTOM_MARGIN - 10)
    # The axis always includes zero, with space added only on the side the bars grow towards
    low, high = padded_range(values)
    y_low = low if min(values) < 0 else 0
    y_high = high if max(values) > 0 else 0
    if y_low == y_high:
        y_high = 1
    slot = plot.width / len(values)

    # Converts a value to a height on screen
    def to_y(value):
        return plot.bottom - (value - y_low) / (y_high - y_low) * plot.height

    draw_axes(screen, plot, font, 0, 1, y_low, y_high, x_ticks=False)
    pygame.draw.line(screen, colours.TEXT, (plot.left, to_y(0)), (plot.right, to_y(0)))

    width = max(1, slot * BAR_SHARE)
    step = math.ceil(MIN_LABEL_WIDTH / slot)
    for i, (name, value) in enumerate(zip(names, values)):
        centre = plot.left + (i + 0.5) * slot
        top, bottom = to_y(max(value, 0)), to_y(min(value, 0))
        pygame.draw.rect(screen, colours.CHART_POINT, (centre - width / 2, top, width, max(1, bottom - top)))
        if i % step == 0:
            text = font.render(fit_text(font, str(name), slot * step - 4), True, colours.TEXT)
            screen.blit(text, text.get_rect(midtop=(centre, plot.bottom + 12)))

    draw_labels(screen, plot, font, x_label, y_label)


# Draws the plot border, grid lines and evenly spaced tick values on both axes,
# or only up the side when the x axis holds categories rather than numbers
def draw_axes(screen, plot, font, x_low, x_high, y_low, y_high, ticks=TICKS, x_ticks=True):
    for i in range(ticks + 1):
        share = i / ticks

        if x_ticks:
            x = plot.left + share * plot.width
            pygame.draw.line(screen, colours.GRID, (x, plot.top), (x, plot.bottom))
            value = font.render(tick_text(x_low + share * (x_high - x_low)), True, colours.TEXT)
            screen.blit(value, value.get_rect(midtop=(x, plot.bottom + 12)))

        y = plot.bottom - share * plot.height
        pygame.draw.line(screen, colours.GRID, (plot.left, y), (plot.right, y))
        value = font.render(tick_text(y_low + share * (y_high - y_low)), True, colours.TEXT)
        screen.blit(value, value.get_rect(midright=(plot.left - 6, y)))

    pygame.draw.rect(screen, colours.TOOLBAR_LINE, plot, width=2)


# Writes the variable names under the x axis and up the side of the y axis
def draw_labels(screen, plot, font, x_label, y_label):
    x_text = font.render(x_label, True, colours.TEXT)
    screen.blit(x_text, x_text.get_rect(midtop=(plot.centerx, plot.bottom + 32)))

    y_text = pygame.transform.rotate(font.render(y_label, True, colours.TEXT), 90)
    screen.blit(y_text, y_text.get_rect(midright=(plot.left - 44, plot.centery)))
