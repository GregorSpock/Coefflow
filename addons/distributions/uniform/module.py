from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, uniform, of random draws"),
]


def make(values):
    if values["low"] >= values["high"]:
        raise ValueError("low must be below high")
    return stats.uniform(values["low"], values["high"] - values["low"])


DISTRIBUTION = common.Distribution(
    "Uniform", False, [("low", 0), ("high", 1)], make,
    aliases={"low": ["min", "a"], "high": ["max", "b"]},
)


# Opens a panel to set the lowest and highest values, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
