from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, gamma, of random draws"),
]


def make(values):
    if values["shape"] <= 0 or values["scale"] <= 0:
        raise ValueError("shape and scale must be above 0")
    return stats.gamma(values["shape"], scale=values["scale"])


DISTRIBUTION = common.Distribution(
    "Gamma", False, [("shape", 2), ("scale", 1)], make,
    aliases={"shape": ["k", "alpha"], "scale": ["theta"]},
)


# Opens a panel to set the shape and scale, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
