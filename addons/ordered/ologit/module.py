from .. import common


INPUTS = 1
OUTPUTS = 3
INPUT_SPECS = [
    ("Data", "a table with an ordered numeric Y column (e.g. a 1-5 rating), numeric or text X and C columns (text ones enter as dummies), and optionally a field to cluster by"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with coefficient, std_error, z_stat, p_value, stars and odds_ratio, then one row per cut point"),
    ("Fitted data", "the rows used, plus the probability of each category (prob_<category>) and the predicted category"),
    ("Marginal effects", "the average effect of each term on the probability of each category, with std_error, p_value and stars"),
]

FIELD_SETTINGS = common.FIELD_SETTINGS


# Opens a panel to give each variable a role: Y (an ordered outcome), X (independent) or C (control),
# with buttons for robust and clustered standard errors
def on_click(node):
    return common.open_panel(node, "Ologit", common.show_model, robust=True)


def run(inputs, settings):
    return common.fit_ordered(inputs, settings, "logit")
