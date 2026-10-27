import importlib
import re
import shutil
import sys
from pathlib import Path


# Add-ons are packs of extra modules kept apart from the core modules in modules/.
# A pack is a folder holding a pack.py and one folder per module, laid out like modules/
# (<pack>/<module folder>/module.py). pack.py gives:
#   NAME         the name shown in Settings > Add-ons
#   DESCRIPTION  one line on what the pack adds
#   TABS         toolbar tab name -> list of module names, e.g. {"Clean": ["Summary Stats"]}
# A pack can also hold a descriptions.txt next to pack.py, with the help text for each of its
# modules (see engine/descriptions.py for how it is laid out), and a version.txt holding the pack's
# version number on its first line (e.g. 1.2.0), shown under its name in Settings > Add-ons.
# A tab that already exists gets the pack's modules added to its end, otherwise a new tab is made.
# The addons/ folder holds only the packs that are loaded: adding a pack copies its folder in from
# elsewhere on the computer, and removing one deletes its copy. Every pack's pack.py in addons/ is
# loaded when the app starts and whenever a pack is added or removed. Its modules are only imported
# when they are needed (on the menu's loading screen or when first placed), as they can be slow to import.
# Core modules are always loaded and can't be removed, and a pack can't replace one
ADDONS_FOLDER = Path(__file__).parent.parent / "addons"

# The loaded packs: pack folder name -> {"name", "description", "version", "tabs"}. Filled by load_all
packs = {}


# The version number from a pack's version.txt, or "" if it has none
def read_version(folder):
    try:
        lines = (folder / "version.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return lines[0].strip() if lines else ""


# Folder name a pack is copied in under, e.g. "My Pack-2" becomes "my_pack_2", so it can be imported
def pack_folder_name(name):
    name = re.sub(r"\W", "_", name.lower())
    if not name or name[0].isdigit():
        name = f"pack_{name}"
    return name


# Forgets a pack's code that Python already imported, so a removed or replaced pack isn't reused
def forget_imports(pack):
    prefix = f"addons.{pack}"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


# Loads the pack.py of every pack in addons/, leaving its modules to be imported when needed.
# A pack whose pack.py fails to load is left out. Returns a list of problems worth telling the user about
def load_all():
    packs.clear()
    problems = []
    if not ADDONS_FOLDER.exists():
        return problems
    importlib.invalidate_caches()

    for folder in sorted(ADDONS_FOLDER.iterdir()):
        if not (folder / "pack.py").exists():
            continue
        try:
            manifest = importlib.import_module(f"addons.{folder.name}.pack")
        except Exception as error:
            problems.append(f"The {folder.name} add-on could not be loaded ({error}).")
            continue
        packs[folder.name] = {
            "name": getattr(manifest, "NAME", folder.name),
            "description": getattr(manifest, "DESCRIPTION", ""),
            "version": read_version(folder),
            "tabs": getattr(manifest, "TABS", {}),
        }
    return problems


# Imports each module of a loaded pack, to check a newly added pack works. Returns a list of
# problems worth telling the user about
def load_modules(pack):
    problems = []
    for module_folder in sorted((ADDONS_FOLDER / pack).iterdir()):
        if not (module_folder / "module.py").exists():
            continue
        try:
            importlib.import_module(f"addons.{pack}.{module_folder.name}.module")
        except Exception as error:
            problems.append(f"{module_folder.name} in the {packs[pack]['name']} add-on could not be loaded ({error}).")
    return problems


# Where a pack chosen from elsewhere would be copied to, or raises ValueError if the
# folder isn't a pack (it needs a pack.py) or is already inside addons/
def destination(source):
    source = Path(source).resolve()
    if not (source / "pack.py").exists():
        raise ValueError(f"{source.name} is not an add-on pack: it has no pack.py")
    if ADDONS_FOLDER.resolve() in [source, *source.parents]:
        raise ValueError("That pack is already in the add-ons folder")
    return ADDONS_FOLDER / pack_folder_name(source.name)


# Copies a pack folder into addons/, replacing any loaded pack with the same folder name
def add(source):
    target = destination(source)
    if target.exists():
        remove(target.name)
    ADDONS_FOLDER.mkdir(exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))


# Deletes a pack's folder from addons/ and forgets its code
def remove(pack):
    forget_imports(pack)
    shutil.rmtree(ADDONS_FOLDER / pack)
    packs.pop(pack, None)


# The loaded pack that holds a module folder, or None if no pack has it
def pack_with_module(folder):
    for pack in packs:
        if (ADDONS_FOLDER / pack / folder / "module.py").exists():
            return pack
    return None


# The toolbar tabs from the loaded packs, merged into one dict of tab name -> module names
def added_tabs():
    tabs = {}
    for settings in packs.values():
        for tab, names in settings["tabs"].items():
            tabs.setdefault(tab, []).extend(names)
    return tabs
