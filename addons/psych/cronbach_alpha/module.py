import numpy as np
import pandas as pd
from scipy import stats

from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with one row per person and a numeric column for each item of the scale"),
]
OUTPUT_SPECS = [
    ("Reliability", "Cronbach's alpha with its confidence interval, the standardised alpha, the mean correlation between items and the scale's mean and standard deviation"),
    ("Items", "each item's mean and standard deviation, its correlation with the rest of the scale, and what alpha would be without it"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["items", "reversed"]
# An item correlating with the rest of the scale below this adds little to it
WEAK_ITEM = 0.3


# Opens a panel to choose the items and mark the reversed ones, showing the reliability of the last run
def on_click(node):
    panel = common.open_panel(node, "Cronbach Alpha", show_reliability, "Reliability")
    common.add_choice(panel, "Show", "show", ["Scale", "Items"])
    show_reliability(panel)
    return panel


def show_reliability(panel):
    settings = panel.node.settings
    if common.list_fields(panel, common.ITEM_ROLES, common.choose_item) is None:
        return
    panel.lines = [f"Items: {common.names(settings, 'items')}", f"Reverse-scored (Rev): {common.names(settings, 'reversed')}"]
    items = settings.get("show") == "Items"
    panel.preview_title = "Item Statistics" if items else "Reliability"
    common.show_results(panel, "Click Item on each item of the scale, and Rev on those worded the other way, then run the workflow", output=1 if items else 0)


# Cronbach's alpha of a table of people by items: how much of the total score's variance is
# shared between the items, scaled so that perfectly consistent items give 1
def alpha(scores):
    items = scores.shape[1]
    total_variance = scores.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    return items / (items - 1) * (1 - scores.var(axis=0, ddof=1).sum() / total_variance)


# What the size of alpha is usually taken to mean (George and Mallery's rules of thumb)
def verdict(value):
    for least, word in [(0.9, "excellent"), (0.8, "good"), (0.7, "acceptable"), (0.6, "questionable"), (0.5, "poor")]:
        if value >= least:
            return word
    return "unacceptable"


# Works out the internal consistency of a scale from its items, on the people who answered them all
def run(inputs, settings):
    data = common.clean(inputs[0])
    frame, low, high = common.item_scores(data, settings)
    frame = frame.dropna()
    scores = frame.to_numpy()
    people, items = scores.shape
    if people < 3:
        raise ValueError("Fewer than 3 rows have an answer to every item")
    if (scores.var(axis=0, ddof=1) == 0).any():
        raise ValueError(f"These items have the same answer on every row: {', '.join(frame.columns[scores.var(axis=0, ddof=1) == 0])}")

    value = alpha(scores)
    correlations = np.corrcoef(scores, rowvar=False)
    mean_r = correlations[np.triu_indices(items, k=1)].mean()
    # Feldt's interval: (1 - alpha) follows an F distribution scaled by 1 - the true alpha
    df_1, df_2 = people - 1, (people - 1) * (items - 1)
    summary = pd.DataFrame([{
        "alpha": value,
        "ci_low": 1 - (1 - value) * stats.f.ppf(1 - common.LEVEL / 2, df_1, df_2),
        "ci_high": 1 - (1 - value) * stats.f.ppf(common.LEVEL / 2, df_1, df_2),
        "standardised_alpha": items * mean_r / (1 + (items - 1) * mean_r),
        "items": items, "n": people, "mean_inter_item_r": mean_r,
        "scale_mean": scores.sum(axis=1).mean(), "scale_sd": scores.sum(axis=1).std(ddof=1),
    }])

    reversed_items = settings.get("reversed") or []
    rows = []
    for i, name in enumerate(frame.columns):
        rest = np.delete(scores, i, axis=1)
        rows.append({
            "item": name, "reversed": "yes" if name in reversed_items else "",
            "mean": scores[:, i].mean(), "sd": scores[:, i].std(ddof=1),
            "item_rest_r": np.corrcoef(scores[:, i], rest.sum(axis=1))[0, 1],
            "alpha_if_deleted": alpha(rest) if items > 2 else np.nan,
        })
    item_table = pd.DataFrame(rows)

    notes = [
        f"Cronbach's alpha = {common.short(value)} (95% CI {common.short(summary['ci_low'][0])} to {common.short(summary['ci_high'][0])}) "
        f"for {items} items and {people} people: {verdict(value)} internal consistency."
    ]
    negative = list(item_table.loc[item_table["item_rest_r"] < 0, "item"])
    weak = list(item_table.loc[(item_table["item_rest_r"] >= 0) & (item_table["item_rest_r"] < WEAK_ITEM), "item"])
    better = list(item_table.loc[item_table["alpha_if_deleted"] > value + 0.005, "item"])
    if negative:
        notes.append(f"{', '.join(negative)} run against the rest of the scale: should they be reverse-scored (Rev)?")
    if weak:
        notes.append(f"{', '.join(weak)} correlate below {WEAK_ITEM} with the rest of the scale.")
    if better:
        notes.append(f"Alpha would be higher without {', '.join(better)}.")
    notes.append(common.reversal_text(settings, low, high))
    if people < len(data):
        notes.append(f"Left out {len(data) - people} rows missing an item.")

    summary.attrs = {"note": " ".join(note for note in notes if note)}
    return [summary, item_table]
