from pathlib import Path


# The program's version number is kept on the first line of version.txt in the main folder,
# the same way add-on packs keep theirs. To release a new version, change that line
VERSION_FILE = Path(__file__).parent.parent / "version.txt"


# The version number, e.g. "0.1.1", or "" if the file is missing or empty
def read():
    try:
        lines = VERSION_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return lines[0].strip() if lines else ""


# A version number as a tuple of whole numbers to compare, e.g. "0.1.10" becomes (0, 1, 10).
# Any part that isn't a whole number counts as 0
def parts(number):
    return tuple(int(part) if part.isdigit() else 0 for part in number.strip().split("."))


# True if version number a is later than version number b
def newer(a, b):
    return parts(a) > parts(b)
