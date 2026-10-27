from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode k, probability (P(X = k)) and cdf (P(X <= k)) for the likely numbers of trials; in Sample mode one column, geometric, of random draws"),
]


def make(values):
    if not 0 < values["p"] < 1:
        raise ValueError("p must be a probability between 0 and 1")
    return stats.geom(values["p"])


DISTRIBUTION = common.Distribution(
    "Geometric", True, [("p", 0.3)], make,
)


# Opens a panel to set the chance of success on each trial, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
