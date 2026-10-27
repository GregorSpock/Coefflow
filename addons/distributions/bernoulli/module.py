from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode k, probability (P(X = k)) and cdf (P(X <= k)) for the likely outcomes, 0 and 1; in Sample mode one column, bernoulli, of random draws"),
]


def make(values):
    if not 0 <= values["p"] <= 1:
        raise ValueError("p must be a probability between 0 and 1")
    return stats.bernoulli(values["p"])


DISTRIBUTION = common.Distribution(
    "Bernoulli", True, [("p", 0.5)], make,
)


# Opens a panel to set the chance of a 1, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
