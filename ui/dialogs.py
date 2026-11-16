from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QColorDialog, QFileDialog, QMessageBox

from engine import dialogs as questions
from engine import file_formats


# The file choosers and message boxes. The ones a module can ask for (engine/dialogs.py) are
# given their answers here, and the rest are for the app itself

WORKFLOW_TYPES = "Coefflow workflow (*.confw)"
# Workflows saved before the rename to Coefflow use .ectyx, so they can still be opened
OPEN_TYPES = "Coefflow workflow (*.confw *.ectyx)"


# The window the dialogs open in front of
def window():
    return QApplication.activeWindow()


# Asks the user to pick a data file of any kind Coefflow reads (engine/file_formats.py).
# Returns "" if they cancel
def choose_file():
    return QFileDialog.getOpenFileName(window(), "Select a data file", "", file_formats.file_filter())[0]


# Asks the user to pick a folder. Returns "" if they cancel
def choose_folder():
    return QFileDialog.getExistingDirectory(window(), "Select a folder to save into")


# Asks the user to pick an add-on pack's folder to load. Returns "" if they cancel
def choose_addon_folder():
    return QFileDialog.getExistingDirectory(window(), "Select an add-on pack folder to load")


# Asks the user to pick a workflow file to open. Returns "" if they cancel
def choose_workflow():
    return QFileDialog.getOpenFileName(window(), "Open a workflow", "", OPEN_TYPES)[0]


# Asks where to save, adding the ending if it is left off. Returns "" if they cancel
def choose_save(title, types, ending):
    path = QFileDialog.getSaveFileName(window(), title, "", types)[0]
    if path and not path.lower().endswith(ending):
        path += ending
    return path


def choose_save_path():
    return choose_save("Save the workflow", WORKFLOW_TYPES, ".confw")


def choose_xlsx_path():
    return choose_save("Export the table", "Excel (*.xlsx)", ".xlsx")


def choose_tex_path():
    return choose_save("Export the table as LaTeX", "LaTeX (*.tex)", ".tex")


def choose_png_path():
    return choose_save("Export the chart", "PNG picture (*.png)", ".png")


# Asks the user to pick a colour, starting from the one given as (red, green, blue).
# Returns the colour picked as [red, green, blue], or None if they cancel
def choose_colour(colour):
    picked = QColorDialog.getColor(QColor(*colour), window(), "Choose a colour")
    return [picked.red(), picked.green(), picked.blue()] if picked.isValid() else None


# Asks whether to save changes first. Returns True for yes, False for no and None for cancel
def ask_save_changes(name="the current workflow"):
    buttons = QMessageBox.StandardButton
    answer = QMessageBox.question(
        window(), "Unsaved changes", f"Save changes to {name} first?",
        buttons.Yes | buttons.No | buttons.Cancel, buttons.Yes,
    )
    if answer == buttons.Cancel:
        return None
    return answer == buttons.Yes


# Asks a yes or no question. Returns True for yes
def ask_yes_no(title, text):
    buttons = QMessageBox.StandardButton
    return QMessageBox.question(window(), title, text, buttons.Yes | buttons.No) == buttons.Yes


def show_message(title, text):
    QMessageBox.warning(window(), title, text)


# Gives the modules' questions their answers
def install():
    questions.answers.update({
        "choose_file": choose_file,
        "choose_folder": choose_folder,
        "choose_xlsx_path": choose_xlsx_path,
        "choose_tex_path": choose_tex_path,
        "choose_png_path": choose_png_path,
        "show_message": show_message,
    })
