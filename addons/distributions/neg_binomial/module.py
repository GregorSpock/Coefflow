from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode k, probability (P(X = k)) and cdf (P(X <= k)) for the likely counts; in Sample mode one column, neg_binomial, of random draws"),
]


def make(values):
    if values["mean"] <= 0 or values["alpha"] <= 0:
        raise ValueError("mean and alpha must be above 0")
    # scipy counts failures before the r-th success, so the mean and alpha are turned into r and p
    r = 1 / values["alpha"]
    return stats.nbinom(r, r / (r + values["mean"]))


DISTRIBUTION = common.Distribution(
    "Neg Binomial", True, [("mean", 3), ("alpha", 0.5)], make,
    aliases={"mean": ["mu"], "alpha": ["dispersion"]},
)


# Opens a panel to set the mean count and the overdispersion (alpha), and choose between a table of probabilities
# and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
