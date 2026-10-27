import math

from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, lognormal, of random draws"),
]


def make(values):
    if values["sigma"] <= 0:
        raise ValueError("sigma must be above 0")
    if abs(values["mu"]) > 700:
        raise ValueError("mu must be between -700 and 700, as e to the power mu is too big beyond that")
    return stats.lognorm(values["sigma"], scale=math.exp(values["mu"]))


DISTRIBUTION = common.Distribution(
    "Lognormal", False, [("mu", 0), ("sigma", 1)], make,
)


# Opens a panel to set the mean and sd of the log, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
