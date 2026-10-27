import pandas as pd

from engine import clusters
from ui.panel import Panel


# The Cluster button for a regression's panel and the panel it opens, where the user picks the
# field to cluster the standard errors by. The choice is kept in settings["cluster"], so a
# module using it lists "cluster" in FIELD_SETTINGS and passes the settings to clusters.covariance


def cluster_label(settings):
    return "Cluster: On" if settings.get("cluster") else "Cluster: Off"


# Adds the Cluster button to a regression panel's top row
def add_cluster_button(panel):
    panel.add_button(cluster_label(panel.node.settings), open_cluster_panel)


def open_cluster_panel(panel, button):
    panel.next_panel = cluster_panel(panel.node)


# Opens the panel for choosing the cluster field. Back returns to the regression's own panel
def cluster_panel(node):
    panel = Panel(node, "Cluster")
    panel.add_button("Back", go_back)
    panel.add_button("No clustering", clear_cluster)
    panel.preview_title = "Cluster Fields"
    panel.refresh = show_clusters
    show_clusters(panel)
    return panel


def go_back(panel, button):
    panel.next_panel = panel.node.on_click()


def clear_cluster(panel, button):
    panel.node.settings["cluster"] = None
    show_clusters(panel)


# Clusters by the clicked field, or stops clustering if it was already chosen
def choose_cluster(panel, button):
    settings = panel.node.settings
    settings["cluster"] = None if settings.get("cluster") == button.value else button.value
    show_clusters(panel)


# The number of coefficients the model has, counting the intercept
def parameter_count(settings):
    return 1 + len(settings.get("independent", [])) + len(settings.get("controls", []))


# A table of every field that could be clustered by: how many clusters it makes,
# how many they count as given their sizes, and why it is suggested
def cluster_table(data, settings, suggested):
    rows = []
    for column in data.columns:
        if column in clusters.model_variables(settings):
            continue
        summary = clusters.summarise(data[column])
        rows.append({
            "field": column,
            "suggested": suggested.get(column, ""),
            "clusters": summary["clusters"],
            "effective": round(summary["effective"], 1),
            "smallest": summary["smallest"],
            "largest": summary["largest"],
        })
    return pd.DataFrame(rows)


# Lists the fields with a Use button each, the suggestions, a table of what each field gives,
# and warnings about the chosen field. Counts use the rows the model would use
def show_clusters(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.note = ""

    if not node.last_inputs or node.last_inputs[0] is None:
        panel.lines = ["Run the workflow to load the fields"]
        panel.message = ""
        return

    data = clusters.model_rows(node.last_inputs[0], settings)
    suggested = dict(clusters.suggestions(node.last_inputs[0], settings))
    chosen = settings.get("cluster")

    if chosen and chosen in data.columns:
        summary = clusters.summarise(data[chosen])
        panel.lines = [
            f"Cluster by: {chosen} ({summary['clusters']} clusters, about {summary['effective']:.0f} effective)"
        ]
        panel.note = " ".join(clusters.warnings_for(summary, chosen, parameter_count(settings)))
    else:
        panel.lines = [f"Cluster by: {chosen or 'none'}"]
        if chosen:
            panel.note = f"{chosen} is not in the data, so the model will not run until another field is chosen."

    panel.lines.append(f"Suggested: {', '.join(suggested) or 'none found'}")
    panel.lines.append("Tip: cluster where X was set, else by the coarsest grouping with enough clusters")

    for column in data.columns:
        if column in clusters.model_variables(settings):
            continue
        count = data[column].nunique()
        label = f"{column} ({count}){' - suggested' if column in suggested else ''}"
        panel.add_option(label, [("Use", choose_cluster, column == chosen)])
        panel.options[-1][1][0].value = column

    panel.preview = cluster_table(data, settings, suggested)
    if panel.preview.empty:
        panel.preview = None
        panel.message = "Every field is a model variable. Add a field that groups the rows to cluster by"
