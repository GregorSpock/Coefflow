import statsmodels.api as sm

from modules.logit import module as logit


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with a 0/1 Y column, numeric X and C columns, and optionally a field to cluster by"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with coefficient, std_error, z_stat, p_value, stars, marginal_effect and marginal_p"),
    ("Fitted data", "the rows used, plus probability (that Y is 1) and predicted (0/1) columns"),
]

FIELD_SETTINGS = logit.FIELD_SETTINGS


# Probit uses the same panel and checks as logit, with the normal distribution as its link.
# Coefficients are not odds ratios, so the marginal effects are the easier numbers to read
def on_click(node):
    return logit.open_panel(node, "Probit")


def run(inputs, settings):
    return logit.fit_binary(sm.Probit, inputs, settings, odds_ratios=False)
