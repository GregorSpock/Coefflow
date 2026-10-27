from types import SimpleNamespace

import pandas as pd

from engine import significance
from ui.panel import Panel


# The Configure panel, opened from Settings, for choices that apply to the whole app rather than one
# module. It sets the p value levels for the significance stars every regression module shows.
# Each level has an Edit button that opens a box to type it in. A level is saved as soon as all
# three make sense together, and until then the note says what is wrong


# The panel isn't for a node on the canvas, so it gets a stand-in with nothing in it
def configure_panel():
    node = SimpleNamespace(id=None, name="Configure", error=None, module=None, settings={}, last_inputs=[], last_outputs=[])
    panel = Panel(node, "Configure")
    panel.option_columns = 1
    panel.add_button("Reset", reset_levels)
    panel.draft = [f"{level:g}" for level in significance.levels]
    panel.editing = None
    panel.preview_title = "Significance Stars"
    panel.refresh = show_configure
    show_configure(panel)
    return panel


def show_configure(panel):
    panel.clear_options()
    panel.lines = [
        "A coefficient gets one star for each level its p value is below",
        "Applies to every regression module. Run the workflow again to update their results",
    ]
    for stars, text in enumerate(panel.draft, start=1):
        label = f"{'*' * stars}  p < {text or '?'}"
        panel.add_option(label, [("Edit", toggle_edit, panel.editing == stars - 1)])
        panel.options[-1][1][0].value = stars - 1

    panel.preview = pd.DataFrame({
        "level": ["*" * stars for stars in range(1, len(significance.levels) + 1)],
        "p_value_below": significance.levels,
    })
    panel.message = ""


# Opens the box for the clicked level, or closes it if that level was already being edited
def toggle_edit(panel, button):
    if panel.editing == button.value:
        panel.editing = None
        panel.text_field = None
    else:
        panel.editing = button.value
        stars = "*" * (button.value + 1)
        panel.add_text_field(f"{stars} p below:", panel.draft[button.value], change_level, blocked="")
        panel.text_field.focused = True
    show_configure(panel)


# Keeps the typed level and saves all three once they are numbers that make sense together
def change_level(panel, text):
    panel.draft[panel.editing] = text.strip()
    panel.note = ""
    try:
        levels = [float(level) for level in panel.draft]
    except ValueError:
        panel.note = "Each level must be a number, like 0.05"
    else:
        try:
            significance.save(levels)
        except ValueError as error:
            panel.note = str(error)
        except OSError as error:
            panel.note = f"Could not save the levels: {error}"
    show_configure(panel)


def reset_levels(panel, button):
    significance.save(significance.DEFAULT_LEVELS)
    panel.draft = [f"{level:g}" for level in significance.levels]
    panel.editing = None
    panel.text_field = None
    panel.note = ""
    show_configure(panel)
