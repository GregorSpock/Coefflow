from scipy import stats

from .. import common


INPUTS = 0
OUTPUTS = 1
INPUT_SPECS = []
OUTPUT_SPECS = [
    ("Distribution", "in Table mode x, density and cdf (P(X <= x)) across the likely values; in Sample mode one column, f, of random draws"),
]


def make(values):
    if values["df1"] <= 0 or values["df2"] <= 0:
        raise ValueError("df1 and df2 must be above 0")
    return stats.f(values["df1"], values["df2"])


DISTRIBUTION = common.Distribution(
    "F", False, [("df1", 3), ("df2", 30)], make,
    aliases={"df1": ["d1"], "df2": ["d2"]},
)


# Opens a panel to set the numerator and denominator degrees of freedom, and choose between a table of
# probabilities and a random sample
def on_click(node):
    return common.open_panel(node, DISTRIBUTION)


def run(inputs, settings):
    return common.run(DISTRIBUTION, settings)
