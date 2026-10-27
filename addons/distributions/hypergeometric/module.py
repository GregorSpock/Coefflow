from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode k, probability (P(X = k)) and cdf (P(X <= k)) for the likely numbers of successes drawn; in Sample mode one column, hypergeometric, of random draws"),
]


def make(values):
    population = common.whole(values, "population", 1)
    successes = common.whole(values, "successes", 0)
    draws = common.whole(values, "draws", 1)
    if successes > population or draws > population:
        raise ValueError("successes and draws can't be more than the population")
    return stats.hypergeom(population, successes, draws)


DISTRIBUTION = common.Distribution(
    "Hypergeometric", True, [("population", 50), ("successes", 10), ("draws", 5)], make,
    aliases={"population": ["pop", "total"], "successes": ["good"], "draws": ["n", "sample"]},
)


# Opens a panel to set the population, how many of it are successes and how many are drawn, and choose
# between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
