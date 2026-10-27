from engine import panel_data
from ui import panel_models
from ui.panel_models import FIELD_SETTINGS


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a panel table: a numeric column for Y, numeric or text columns for X and any controls (C), a field naming each entity (E) and optionally a time field (T). Rows missing any of them are dropped"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, coefficient, std_error, t_stat, p_value and stars. The intercept is the average entity effect. R squared within, between and overall are shown above"),
    ("Fitted data", "the rows used, plus fitted, residual and effect (each entity's effect, measured from the intercept) columns"),
]


def on_click(node):
    return panel_models.open_panel(node, "Fixed Effects")


# Fits fixed effects of Y on the X and C variables, comparing each entity with itself
# (see engine/panel_data.py)
def run(inputs, settings):
    return panel_data.fit_fixed(inputs[0], settings)
