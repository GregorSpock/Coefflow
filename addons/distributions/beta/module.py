from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, beta, of random draws"),
]


def make(values):
    if values["a"] <= 0 or values["b"] <= 0:
        raise ValueError("a and b must be above 0")
    return stats.beta(values["a"], values["b"])


DISTRIBUTION = common.Distribution(
    "Beta", False, [("a", 2), ("b", 5)], make,
    aliases={"a": ["alpha"], "b": ["beta"]},
)


# Opens a panel to set the two shape parameters, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
