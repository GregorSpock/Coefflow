# The questions a module can put to the user: which file, which folder, where to save. A module
# asks here and the window answers (ui/dialogs.py puts its own answers in when the app starts).
# With no window, as in a test or a run from the command line, every question is answered as if
# the user had cancelled, so nothing ever waits for a click that can't come


# Asks the user to pick a data file. Returns "" if they cancel
def choose_file():
    return answer("choose_file", "")


# Asks the user to pick a folder to save into. Returns "" if they cancel
def choose_folder():
    return answer("choose_folder", "")


# Asks where to save a table as an Excel file. Returns "" if they cancel
def choose_xlsx_path():
    return answer("choose_xlsx_path", "")


# Asks where to save a table as a LaTeX file. Returns "" if they cancel
def choose_tex_path():
    return answer("choose_tex_path", "")


# Asks where to save a chart as a PNG picture. Returns "" if they cancel
def choose_png_path():
    return answer("choose_png_path", "")


# Shows a message with a title
def show_message(title, text):
    return answer("show_message", None, title, text)


# The answers the window gives, by the name of the question
answers = {}


def answer(question, cancelled, *details):
    if question in answers:
        return answers[question](*details)
    return cancelled
