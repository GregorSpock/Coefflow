import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from ui.chart import draw_scatter
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 0
INPUT_SPECS = [
    ("Data", "a table with at least two numeric columns for X and Y"),
]
OUTPUT_SPECS = []

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["x", "y"]


# Opens a panel to pick the X and Y variables, showing the scatter and a heteroscedasticity note
def on_click(node):
    panel = Panel(node, "Scatter")
    panel.preview_title = "Scatter"
    panel.refresh = show_scatter
    show_scatter(panel)
    return panel


# Sets the clicked variable as X or Y, or clears it if it already was.
# A variable can't be on both axes, so it leaves the other axis if needed
def choose_axis(panel, button):
    settings = panel.node.settings
    axis = button.text.lower()
    other = "y" if axis == "x" else "x"

    if settings.get(axis) == button.value:
        settings[axis] = None
    else:
        settings[axis] = button.value
        if settings.get(other) == button.value:
            settings[other] = None

    show_scatter(panel)


# Regresses y on x and runs a Breusch-Pagan test on the residuals.
# Returns the intercept, slope, test statistic and p value
def test_heteroscedasticity(x_values, y_values):
    x = sm.add_constant(x_values)
    model = sm.OLS(y_values, x).fit()
    statistic, p_value, f_statistic, f_p_value = het_breuschpagan(model.resid, x)
    return model.params.iloc[0], model.params.iloc[1], statistic, p_value


# Writes a short plain-English note about the test result
def describe_test(x_name, y_name, statistic, p_value):
    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    result = f"Breusch-Pagan test: LM = {statistic:.1f}, p {p_text}. "

    if p_value < 0.05:
        return result + (
            f"The spread of {y_name} changes with {x_name}, so regressing {y_name} on {x_name} "
            "gives heteroscedastic errors. Turn on robust standard errors in OLS."
        )
    return result + (
        "There is no significant evidence of heteroscedasticity at the 5% level, "
        "so ordinary standard errors should be fine."
    )


# Lists the numeric variables with X and Y buttons, then draws the scatter and the test note
def show_scatter(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.chart = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the variables"]
        panel.message = ""
        return

    x_name = settings.get("x")
    y_name = settings.get("y")
    panel.lines = [f"X axis: {x_name or 'none'}    Y axis: {y_name or 'none'}"]

    data = node.last_inputs[0]
    for column in data.select_dtypes("number").columns:
        panel.add_option(column, [
            ("X", choose_axis, column == x_name),
            ("Y", choose_axis, column == y_name),
        ])

    if not x_name or not y_name or x_name not in data or y_name not in data:
        panel.message = "Choose an X and a Y variable"
        return

    pair = data[[x_name, y_name]].dropna()
    line = None
    try:
        intercept, slope, statistic, p_value = test_heteroscedasticity(pair[x_name], pair[y_name])
        line = (intercept, slope)
        panel.note = describe_test(x_name, y_name, statistic, p_value)
    except Exception as error:
        panel.note = f"Could not run the heteroscedasticity test: {error}"

    # Draws the chart into the space the panel gives it
    def draw(screen, area):
        draw_scatter(screen, area, panel.table_font, pair[x_name], pair[y_name], x_name, y_name, line)

    panel.chart = draw


# Nothing to calculate for the workflow itself; just checks both variables are chosen
def run(inputs, settings):
    data = inputs[0]
    for axis in ["x", "y"]:
        if not settings.get(axis):
            raise ValueError("Choose an X and a Y variable")
        if settings[axis] not in data.columns:
            raise ValueError(f"Column not in the data: {settings[axis]}")
    return []
