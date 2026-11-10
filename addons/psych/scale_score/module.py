import math

from .. import common


INPUTS = 1
OUTPUTS = 1
INPUT_SPECS = [
    ("Data", "a table with one row per person and a numeric column for each item of the scale"),
]
OUTPUT_SPECS = [
    ("Data", "the same table with a new field holding each person's scale score"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["items", "reversed"]
SCORES = ["Mean", "Sum"]
DEFAULT_NAME = "score"
# A person needs answers to at least this share of the items to get a score
MIN_ANSWERED = 0.5


# Opens a panel to choose the items, mark the reversed ones and name the new field
def on_click(node):
    panel = common.open_panel(node, "Scale Score", show_score, "Data Preview")
    common.add_choice(panel, "Score", "score", SCORES)
    panel.add_text_field("New field:", node.settings.get("name", DEFAULT_NAME), change_name, blocked="")
    show_score(panel)
    return panel


# Keeps the new field's name as it is typed
def change_name(panel, text):
    panel.node.settings["name"] = text.strip()


def show_score(panel):
    settings = panel.node.settings
    if common.list_fields(panel, common.ITEM_ROLES, common.choose_item) is None:
        return
    panel.lines = [f"Items: {common.names(settings, 'items')}", f"Reverse-scored (Rev): {common.names(settings, 'reversed')}"]
    common.show_results(panel, "Click Item on each item of the scale, and Rev on those worded the other way, then run the workflow")


# Adds each person's score on the scale: the mean of their answers, or the sum. Reversed items are
# turned round first. Someone who skipped items is scored on those they answered, as long as they
# answered at least half; a sum is then scaled up to stand for every item, so that it compares
# with other people's
def run(inputs, settings):
    data = inputs[0]
    name = settings.get("name", DEFAULT_NAME) or DEFAULT_NAME
    if name in data.columns:
        raise ValueError(f"The data already has a field called {name}. Type another name for the score")
    scores, low, high = common.item_scores(common.clean(data), settings)
    items = scores.shape[1]

    needed = math.ceil(items * MIN_ANSWERED)
    answered = scores.notna().sum(axis=1)
    score = scores.mean(axis=1).where(answered >= needed)
    if settings.get("score", SCORES[0]) == "Sum":
        score = score * items

    result = data.copy()
    result[name] = score
    notes = [f"{name} is the {settings.get('score', SCORES[0]).lower()} of {items} items."]
    notes.append(common.reversal_text(settings, low, high))
    partial = int(((answered < items) & (answered >= needed)).sum())
    unscored = int((answered < needed).sum())
    if partial:
        notes.append(f"{partial} rows skipped some items and are scored on the ones they answered.")
    if unscored:
        notes.append(f"{unscored} rows answered fewer than {needed} items and have no score.")
    result.attrs = {"note": " ".join(note for note in notes if note)}
    return [result]
