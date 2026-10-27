from ui.panel import Panel


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "any table"),
]
OUTPUT_SPECS = [
    ("Summary", "count, mean, standard deviation, min, quartiles and max of each numeric column"),
]


# Opens a panel showing the summary from the last run, if there is one
def on_click(node):
    panel = Panel(node, "Summary Stats")
    panel.refresh = show_summary
    show_summary(panel)
    return panel


def show_summary(panel):
    if panel.node.last_outputs:
        panel.preview_title = "Summary"
        panel.preview = panel.node.last_outputs[0]
        panel.message = ""
    else:
        panel.message = "Run the workflow to see the summary"


# One row per statistic and one column per numeric field, with the statistic's name in the first column
def run(inputs, settings):
    data = inputs[0].select_dtypes("number")
    if data.empty:
        raise ValueError("The data has no numeric columns")
    summary = data.describe().reset_index()
    return [summary.rename(columns={"index": "statistic"})]
