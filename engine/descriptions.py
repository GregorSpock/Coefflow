from pathlib import Path

from engine import addons


# Each module's help text, shown by the ? on the module, is kept in a descriptions.txt:
# modules/descriptions.txt for the core modules, and one in the main folder of each add-on pack
# for its own modules. In the file each module starts with its name in square brackets, as it is
# shown on the toolbar, followed by its description. A blank line starts a new paragraph and lines
# starting with # are left out, so the file can have notes of its own. For example:
#   [OLS]
#   Fits a linear regression of Y on X.
# The files are read each time help is asked for, so edits show up without restarting the app
FILE_NAME = "descriptions.txt"
CORE_FILE = Path(__file__).parent.parent / "modules" / FILE_NAME


# Reads a descriptions file into a dict of lower case module name -> list of paragraphs.
# A missing or unreadable file gives an empty dict
def read(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}

    descriptions = {}
    paragraphs = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            paragraphs = descriptions.setdefault(line[1:-1].strip().lower(), [""])
        elif paragraphs is None:
            continue
        elif not line:
            if paragraphs[-1]:
                paragraphs.append("")
        else:
            paragraphs[-1] = f"{paragraphs[-1]} {line}".strip()

    return {name: [p for p in paragraphs if p] for name, paragraphs in descriptions.items()}


# The paragraphs describing a module, looked for in the core file and then each loaded pack's file.
# Returns an empty list if no file describes it
def describe(name):
    files = [CORE_FILE] + [addons.ADDONS_FOLDER / pack / FILE_NAME for pack in addons.packs]
    for path in files:
        paragraphs = read(path).get(name.lower())
        if paragraphs:
            return paragraphs
    return []
