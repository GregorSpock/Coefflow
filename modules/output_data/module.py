from pathlib import Path

from ui.dialogs import choose_folder
from ui.panel import Panel


INPUTS = 1
OUTPUTS = 0
INPUT_SPECS = [
    ("Data", "any table, saved as it arrives"),
]
OUTPUT_SPECS = []
FILE_TYPES = ["csv", "xlsx"]


# Fills in a default file name and type the first time the node is used
def set_defaults(settings):
    settings.setdefault("name", "output")
    settings.setdefault("file_type", "csv")


# Opens a panel to choose the folder, file name and file type, with a preview of the data
def on_click(node):
    set_defaults(node.settings)

    panel = Panel(node, "Output Data")
    panel.add_button("Folder", select_folder)
    panel.add_button(node.settings["file_type"].upper(), switch_file_type)
    panel.add_text_field("File name", node.settings["name"], rename)
    panel.refresh = show_output
    show_output(panel)
    return panel


# Asks for a folder and remembers it in the node's settings
def select_folder(panel, button):
    folder = choose_folder()
    if folder:
        panel.node.settings["folder"] = folder
        show_output(panel)


# Switches between saving as CSV and Excel
def switch_file_type(panel, button):
    settings = panel.node.settings
    index = FILE_TYPES.index(settings["file_type"])
    settings["file_type"] = FILE_TYPES[(index + 1) % len(FILE_TYPES)]
    button.text = settings["file_type"].upper()
    show_output(panel)


# Keeps the file name in the settings up to date as the user types
def rename(panel, text):
    panel.node.settings["name"] = text
    show_output(panel)


# Shows where the file will be saved, and the data that arrived on the last run
def show_output(panel):
    settings = panel.node.settings
    folder = settings.get("folder", "not chosen yet")
    panel.lines = [
        f"Folder: {folder}",
        f"Saves as: {settings['name']}.{settings['file_type']}",
    ]

    if panel.node.last_inputs:
        panel.preview = panel.node.last_inputs[0]
    else:
        panel.preview = None
        panel.message = "Run the workflow to preview the data being saved"


# Saves the incoming data to the chosen folder, or just prints it if no folder is chosen
def run(inputs, settings):
    set_defaults(settings)
    data = inputs[0]

    if "folder" not in settings or not settings["name"]:
        print(data)
        return []

    path = Path(settings["folder"]) / f"{settings['name']}.{settings['file_type']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if settings["file_type"] == "xlsx":
        data.to_excel(path, index=False)
    else:
        data.to_csv(path, index=False)

    print(f"Saved {len(data)} rows to {path}")
    return []
