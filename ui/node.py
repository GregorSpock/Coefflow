import importlib
from pathlib import Path

import pygame

from engine import addons
from ui import colours, help


NODE_WIDTH = 120
NODE_HEIGHT = 60
PORT_RADIUS = 7
# The least room each port gets down the side of a node, so a node with many ports grows taller
PORT_GAP = 15
MODULES_FOLDER = Path(__file__).parent.parent / "modules"


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
    # A module placed on the canvas, stored by its centre in canvas coordinates.
    # The number of inputs and outputs comes from the module's own folder.
    # Status is None until the workflow runs, then "done" or "error"
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

        # A module with GROWING_INPUTS = True takes as many inputs as are connected to it: it starts
        # with one, and the canvas keeps one empty input after the connected ones (see
        # Canvas.tidy_inputs), so INPUTS is ignored for it
        self.module = load_module(name)
        self.growing = getattr(self.module, "GROWING_INPUTS", False)
        self.inputs = 1 if self.growing else getattr(self.module, "INPUTS", 0)
        self.outputs = getattr(self.module, "OUTPUTS", 0)

    # Runs the module's on_click, which may return a settings panel to show.
    # Does nothing if the module has no folder
    def on_click(self):
        if self.module:
            return self.module.on_click(self)
        return None

    # Returns the node's rectangle on screen at the current zoom
    def screen_rect(self, canvas):
        width = NODE_WIDTH * canvas.zoom
        height = max(NODE_HEIGHT, PORT_GAP * (max(self.inputs, self.outputs) + 1)) * canvas.zoom
        rect = pygame.Rect(0, 0, width, height)
        rect.center = canvas.to_screen(self.x, self.y)
        return rect

    # Returns True if a screen position is on the node's ? help badge
    def help_clicked(self, canvas, screen_x, screen_y):
        return help.badge_rect(self.screen_rect(canvas), canvas.zoom).collidepoint(screen_x, screen_y)

    # Spaces a number of ports evenly down one side of the node
    def port_positions(self, rect, count, x):
        positions = []
        for i in range(count):
            y = rect.top + rect.height * (i + 1) / (count + 1)
            positions.append((x, y))
        return positions

    # Screen positions of the input ports (left side) and output ports (right side)
    def input_positions(self, canvas):
        rect = self.screen_rect(canvas)
        return self.port_positions(rect, self.inputs, rect.left)

    def output_positions(self, canvas):
        rect = self.screen_rect(canvas)
        return self.port_positions(rect, self.outputs, rect.right)

    # Draws the node and its ports, scaled to the canvas zoom
    def draw(self, screen, canvas):
        rect = self.screen_rect(canvas)

        border = colours.NODE_BORDER
        if self.status == "done":
            border = colours.NODE_DONE
        elif self.status == "error":
            border = colours.NODE_ERROR

        pygame.draw.rect(screen, colours.NODE, rect, border_radius=colours.NODE_RADIUS)
        pygame.draw.rect(screen, border, rect, width=3, border_radius=colours.NODE_RADIUS)

        label = canvas.font.render(self.name, True, colours.TEXT)
        screen.blit(label, label.get_rect(center=rect.center))
        help.draw_badge(screen, rect, canvas.zoom)

        radius = max(3, PORT_RADIUS * canvas.zoom)
        for position in self.input_positions(canvas) + self.output_positions(canvas):
            pygame.draw.circle(screen, colours.PORT, position, radius)
            pygame.draw.circle(screen, colours.NODE_BORDER, position, radius, width=2)
