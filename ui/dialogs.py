import tkinter as tk
from tkinter import filedialog, messagebox


WORKFLOW_TYPES = [("Coefflow workflow", "*.confw")]
# Workflows saved before the rename to Coefflow use .ectyx, so they can still be opened
OPEN_TYPES = [("Coefflow workflow", "*.confw *.ectyx")]


# Makes a hidden window that keeps the file dialogs in front of the full screen app
def hidden_window():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


# Asks the user to pick a CSV or Excel file. Returns "" if they cancel
def choose_file():
    root = hidden_window()
    path = filedialog.askopenfilename(
        parent=root,
        title="Select a data file",
        filetypes=[("Data files", "*.csv *.xlsx"), ("CSV", "*.csv"), ("Excel", "*.xlsx")],
    )
    root.destroy()
    return path


# Asks the user to pick a folder. Returns "" if they cancel
def choose_folder():
    root = hidden_window()
    path = filedialog.askdirectory(parent=root, title="Select a folder to save into")
    root.destroy()
    return path


# Asks the user to pick an add-on pack's folder to load. Returns "" if they cancel
def choose_addon_folder():
    root = hidden_window()
    path = filedialog.askdirectory(parent=root, title="Select an add-on pack folder to load")
    root.destroy()
    return path


# Asks the user to pick a workflow file to open. Returns "" if they cancel
def choose_workflow():
    root = hidden_window()
    path = filedialog.askopenfilename(parent=root, title="Open a workflow", filetypes=OPEN_TYPES)
    root.destroy()
    return path


# Asks the user where to save a workflow, adding .confw if they leave it off. Returns "" if they cancel
def choose_save_path():
    root = hidden_window()
    path = filedialog.asksaveasfilename(
        parent=root, title="Save the workflow", filetypes=WORKFLOW_TYPES, defaultextension=".confw"
    )
    root.destroy()
    return path


# Asks where to save a table as an Excel file, adding .xlsx if they leave it off. Returns "" if they cancel
def choose_xlsx_path():
    root = hidden_window()
    path = filedialog.asksaveasfilename(
        parent=root, title="Export the table", filetypes=[("Excel", "*.xlsx")], defaultextension=".xlsx"
    )
    root.destroy()
    return path


# Asks whether to save changes first. Returns True for yes, False for no and None for cancel
def ask_save_changes():
    root = hidden_window()
    answer = messagebox.askyesnocancel("Unsaved changes", "Save changes to the current workflow first?", parent=root)
    root.destroy()
    return answer


# Asks a yes or no question. Returns True for yes
def ask_yes_no(title, text):
    root = hidden_window()
    answer = messagebox.askyesno(title, text, parent=root)
    root.destroy()
    return answer


def show_message(title, text):
    root = hidden_window()
    messagebox.showwarning(title, text, parent=root)
    root.destroy()


# The text on the clipboard, or "" if it holds none
def paste_text():
    root = hidden_window()
    try:
        return root.clipboard_get()
    except tk.TclError:
        return ""
    finally:
        root.destroy()
