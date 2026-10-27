from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode k, probability (P(X = k)) and cdf (P(X <= k)) for the likely numbers of successes; in Sample mode one column, binomial, of random draws"),
]


def make(values):
    n = common.whole(values, "n", 1)
    if not 0 <= values["p"] <= 1:
        raise ValueError("p must be a probability between 0 and 1")
    return stats.binom(n, values["p"])


DISTRIBUTION = common.Distribution(
    "Binomial", True, [("n", 10), ("p", 0.5)], make,
    aliases={"n": ["trials"]},
)


# Opens a panel to set the number of trials and the chance of success, and choose between
# a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
