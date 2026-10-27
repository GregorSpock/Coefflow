from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, exponential, of random draws"),
]


def make(values):
    if values["rate"] <= 0:
        raise ValueError("rate must be above 0")
    return stats.expon(scale=1 / values["rate"])


DISTRIBUTION = common.Distribution(
    "Exponential", False, [("rate", 1)], make,
    aliases={"rate": ["lambda", "lam", "λ"]},
)


# Opens a panel to set the rate, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
