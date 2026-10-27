from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode k, probability (P(X = k)) and cdf (P(X <= k)) for the likely counts; in Sample mode one column, poisson, of random draws"),
]


def make(values):
    if values["lambda"] <= 0:
        raise ValueError("lambda must be above 0")
    return stats.poisson(values["lambda"])


DISTRIBUTION = common.Distribution(
    "Poisson", True, [("lambda", 3)], make,
    aliases={"lambda": ["lam", "mean", "rate", "λ"]},
)


# Opens a panel to set the mean count (lambda), and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
