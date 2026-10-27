from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, chi_square, of random draws"),
]


def make(values):
    if values["df"] <= 0:
        raise ValueError("df must be above 0")
    return stats.chi2(values["df"])


DISTRIBUTION = common.Distribution(
    "Chi-Square", False, [("df", 3)], make,
    aliases={"df": ["k", "degrees"]},
)


# Opens a panel to set the degrees of freedom, and choose between a table of probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
