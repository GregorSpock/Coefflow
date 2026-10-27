import numpy as np
import pandas as pd
from scipy import stats

from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with an ordered numeric Y column, and numeric or text X and C columns (text ones enter as dummies)"),
]
OUTPUT_SPECS = [
    ("Test", "the likelihood ratio statistic, df and p_value, with the number of groups and rows"),
    ("Groups", "for each group of rows with similar predicted scores, the observed and expected count of each category"),
]

FIELD_SETTINGS = common.FIELD_SETTINGS
GROUP_CHOICES = [6, 8, 10, 12]
DEFAULT_GROUPS = 10
LINKS = ["logit", "probit"]


# The Lipsitz, Fitzmaurice and Molenberghs (1996) goodness of fit test for an ordered logit or
# probit, the ordered version of the Hosmer-Lemeshow test. Rows are split into groups by their
# predicted score (the expected category number), and the model is refitted with a dummy for each
# group. If the model fits, the dummies add nothing, so a small p value means a poor fit
def on_click(node):
    panel = common.open_panel(node, "Lipsitz Test", show_test)
    panel.add_button(link_label(node.settings), next_link)
    panel.add_button(groups_label(node.settings), next_groups)
    return panel


def link_label(settings):
    return f"Link: {settings.get('link', 'logit').title()}"


def groups_label(settings):
    return f"Groups: {settings.get('groups', DEFAULT_GROUPS)}"


# Switches between testing an ordered logit and an ordered probit
def next_link(panel, button):
    settings = panel.node.settings
    current = settings.get("link", "logit")
    settings["link"] = LINKS[(LINKS.index(current) + 1) % len(LINKS)]
    button.text = link_label(settings)
    panel.refresh(panel)


# Steps through the number of groups the rows are split into
def next_groups(panel, button):
    settings = panel.node.settings
    current = settings.get("groups", DEFAULT_GROUPS)
    position = GROUP_CHOICES.index(current) if current in GROUP_CHOICES else -1
    settings["groups"] = GROUP_CHOICES[(position + 1) % len(GROUP_CHOICES)]
    button.text = groups_label(settings)
    panel.refresh(panel)


def show_test(panel):
    if not common.show_roles(panel):
        return
    node = panel.node
    if not node.last_outputs:
        panel.message = "Choose an ordered Y and the X variables, then run the workflow"
        panel.lines.append(common.Y_HINT)
        return

    results = node.last_outputs[0]
    test = results.iloc[0]
    verdict = "poor fit" if test["p_value"] < 0.05 else "no evidence of poor fit"
    panel.lines.append(
        f"LR {test['lr_statistic']:.2f} on {test['df']} df, p {test['p_value']:.3g} ({verdict} at 5%)    "
        f"Groups: {test['groups']}    N: {test['observations']}"
    )
    panel.note = " ".join(results.attrs["warnings"])
    panel.preview = node.last_outputs[1]
    panel.preview_title = "Observed and expected counts"


def run(inputs, settings):
    notes = []
    link = settings.get("link", "logit")
    groups = settings.get("groups", DEFAULT_GROUPS)
    used, y, x, roles, categories = common.prepare(inputs, settings, notes)

    base = common.fit(y, x, link, notes)
    probabilities = common.probability_columns(base, x, categories)
    score = probabilities.to_numpy() @ np.arange(1, len(categories) + 1)

    # Equal sized groups by score, breaking ties by row order so every group gets rows
    group = pd.Series(pd.qcut(pd.Series(score).rank(method="first"), groups, labels=False).to_numpy(), index=x.index)
    if len(used) / (5 * len(categories)) < groups:
        notes.append(
            f"With {len(used)} rows and {len(categories)} categories, Lipsitz et al. suggest at most "
            f"{int(len(used) / (5 * len(categories)))} groups, so the p value may be unreliable."
        )

    dummies = pd.get_dummies(group, prefix="group", drop_first=True, dtype=float)
    extended = common.fit(y, pd.concat([x, dummies], axis=1), link, notes)
    statistic = max(0.0, 2 * (extended.llf - base.llf))
    df = groups - 1
    results = pd.DataFrame([{
        "test": f"Lipsitz ({'ordered ' + link})",
        "lr_statistic": statistic,
        "df": df,
        "p_value": stats.chi2.sf(statistic, df),
        "groups": groups,
        "observations": len(used),
    }])
    results.attrs = {"warnings": notes}

    rows = []
    for g in range(groups):
        members = (group == g).to_numpy()
        row = {"group": g + 1, "rows": int(members.sum()), "mean_score": score[members].mean()}
        for c, category in enumerate(categories):
            label = common.category_text(category)
            row[f"observed_{label}"] = int((y.to_numpy()[members] == category).sum())
            row[f"expected_{label}"] = probabilities.to_numpy()[members, c].sum()
        rows.append(row)
    return [results, pd.DataFrame(rows)]
