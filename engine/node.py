import importlib

from engine import addons


# Folder name for a module, e.g. "Durbin-Watson" becomes "durbin_watson"
def folder_name(name):
    name = name.lower()
    for character in [" ", "-", ".", "/"]:
        name = name.replace(character, "_")
    return name


# Loads modules/<folder>/module.py for a module name, or the module from an added add-on pack
# if it isn't a core module. Returns None if there isn't one. Core modules come first, so a pack
# can never replace one. Python keeps each module once loaded, so only the first call is slow.
# The core modules are imported by name rather than by looking for a module.py on disk, so this
# works both when running from source and from a compiled build where they live inside the binary
# with no source file alongside the executable. A core module that exists but fails to import still
# raises, so a real error in one is not mistaken for the module simply being absent.
def load_module(name):
    folder = folder_name(name)
    try:
        return importlib.import_module(f"modules.{folder}.module")
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing not in ("modules", f"modules.{folder}", f"modules.{folder}.module"):
            raise
    pack = addons.pack_with_module(folder)
    if pack:
        return importlib.import_module(f"addons.{pack}.{folder}.module")
    return None


class Node:
    # A module placed in a workflow, with its position on the canvas (its centre), its settings
    # and what it was last given and gave back. The number of inputs and outputs comes from the
    # module's own folder.
    # status is None until the node runs, then "queued" and "running" while a run is under way,
    # and "done" or "error" once it has finished. A node whose settings or inputs have changed
    # since it ran goes back to None, as its results are out of date.
    # ran_as is what the node was run on the last time it worked (see engine/runner.py), which
    # is how a run tells that its results can be kept rather than worked out again
    def __init__(self, name, x, y, node_id=None):
        self.id = node_id
        self.name = name
        self.x = x
        self.y = y
        self.settings = {}
        self.status = None
        self.last_inputs = []
        self.last_outputs = []
        self.error = None
        self.ran_as = None

        # A module with GROWING_INPUTS = True takes as many inputs as are connected to it: it starts
        # with one, and the workflow keeps one empty input after the connected ones (see
        # Workflow.tidy_inputs), so INPUTS is ignored for it
        self.module = load_module(name)
        self.growing = getattr(self.module, "GROWING_INPUTS", False)
        self.inputs = 1 if self.growing else getattr(self.module, "INPUTS", 0)
        self.outputs = getattr(self.module, "OUTPUTS", 0)

    # Runs the module's on_click, which may return a settings panel to show.
    # Does nothing if the module has no folder
    def on_click(self):
        if self.module and hasattr(self.module, "on_click"):
            return self.module.on_click(self)
        return None

    # Forgets what the node last gave, as after its settings are put back by undo
    def clear_results(self):
        self.status = None
        self.last_inputs = []
        self.last_outputs = []
        self.error = None
        self.ran_as = None
