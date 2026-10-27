from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Correlations", "the Pearson correlation between each pair of numeric columns"),
]


# Opens a panel showing the correlations from the last run, if there are any
def on_click(node):
    panel = Panel(node, "Pearson")
    panel.refresh = show_correlations
    show_correlations(panel)
    return panel


def show_correlations(panel):
    if panel.node.last_outputs:
        panel.preview_title = "Correlations"
        panel.preview = panel.node.last_outputs[0]
        panel.message = ""
    else:
        panel.message = "Run the workflow to see the correlations"


# A square table of correlations, with each row's field name in the first column
def run(inputs, settings):
    data = inputs[0].select_dtypes("number")
    if data.empty:
        raise ValueError("The data has no numeric columns")
    correlations = data.corr().round(4).reset_index()
    return [correlations.rename(columns={"index": "field"})]
