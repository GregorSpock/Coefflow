from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, normal, of random draws"),
]


def make(values):
    if values["sd"] <= 0:
        raise ValueError("sd must be above 0")
    return stats.norm(values["mean"], values["sd"])


DISTRIBUTION = common.Distribution(
    "Normal", False, [("mean", 0), ("sd", 1)], make,
    aliases={"mean": ["mu"], "sd": ["sigma"]},
)


# Opens a panel to set the mean and sd, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
