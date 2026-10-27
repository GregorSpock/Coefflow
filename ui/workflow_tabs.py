import os
import shutil
from pathlib import Path

import pygame

from ui import colours, display, workspace
from ui.canvas import Canvas


# Untitled workflows are kept as files in temp/ while they are open, so they always have a file
# behind them. The folder is emptied when the app starts and when it closes
TEMP_FOLDER = Path(__file__).parent.parent / "temp"
UNTITLED_NAME = "UntitledWorkflow"

STRIP_HEIGHT = 22
STRIP_FONT_SIZE = 20
TAB_PADDING = 10
MAX_TAB_WIDTH = 220
TOGGLE_SIZE = 20


# Deletes everything in the temp folder, making the folder if it is missing
def clear_temp():
    if TEMP_FOLDER.exists():
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
    TEMP_FOLDER.mkdir(exist_ok=True)


class OpenWorkflow:
    # One open workflow: its canvas, the file it is saved to (None until it is saved somewhere),
    # its temp file while untitled, and its state at the last save to tell if it has changed
    def __init__(self, canvas, path=None, temp_path=None):
        self.canvas = canvas
        self.path = path
        self.temp_path = temp_path
        self.saved_state = workspace.snapshot(canvas)
        self.temp_state = self.saved_state

    def has_changes(self):
        return workspace.snapshot(self.canvas) != self.saved_state

    # An untitled workflow that has never been touched, which can be replaced by an opened file
    def is_blank(self):
        return self.path is None and not self.canvas.nodes and not self.has_changes()

    def name(self):
        file = self.path or self.temp_path
        return Path(file).stem

    # The name with an asterisk added when there are unsaved changes
    def label(self):
        return self.name() + ("*" if self.has_changes() else "")

    # Rewrites an untitled workflow's temp file if it has changed since it was last written
    def update_temp(self):
        if not self.temp_path:
            return
        state = workspace.snapshot(self.canvas)
        if state != self.temp_state:
            try:
                workspace.save(self.canvas, self.temp_path)
                self.temp_state = state
            except OSError:
                pass

    # Deletes the temp file, once the workflow is saved somewhere real or closed
    def remove_temp(self):
        if self.temp_path:
            try:
                os.remove(self.temp_path)
            except OSError:
                pass
            self.temp_path = None


# Makes a new untitled workflow saved into the temp folder as UntitledWorkflow_<number>
def new_untitled(rect, number):
    canvas = Canvas(rect)
    temp_path = TEMP_FOLDER / f"{UNTITLED_NAME}_{number}.confw"
    open_workflow = OpenWorkflow(canvas, temp_path=str(temp_path))
    try:
        TEMP_FOLDER.mkdir(exist_ok=True)
        workspace.save(canvas, temp_path)
    except OSError:
        pass
    return open_workflow


class WorkflowTabs:
    # A thin strip under the toolbar with a small tab for each open workflow, each with a box
    # to close it, and a small box at the right of the canvas that shows or hides the strip
    def __init__(self, font_size=STRIP_FONT_SIZE):
        self.font = pygame.font.Font(None, font_size)
        self.expanded = True
        self.top = 0
        self.width = 0
        self.tab_rects = []
        self.close_rects = []
        self.toggle_rect = pygame.Rect(0, 0, TOGGLE_SIZE, TOGGLE_SIZE)

    # Height the strip takes up, which is nothing while it is hidden
    def height(self):
        return STRIP_HEIGHT if self.expanded else 0

    # Places the strip just under the toolbar, and the toggle box at the top right of the canvas
    def place(self, top, width):
        self.top = top
        self.width = width
        self.toggle_rect.topright = (width - 4, top + self.height() + 4)

    # Works out each tab's box and close box. Tabs share the width if there are too many to fit
    def layout(self, workflows):
        self.tab_rects = []
        self.close_rects = []
        if not workflows:
            return
        fit_width = self.width // len(workflows)
        x = 0
        for open_workflow in workflows:
            text_width = self.font.size(open_workflow.label())[0]
            width = min(text_width + TAB_PADDING * 2 + STRIP_HEIGHT, MAX_TAB_WIDTH, fit_width)
            rect = pygame.Rect(x, self.top, width, STRIP_HEIGHT)
            close = pygame.Rect(rect.right - STRIP_HEIGHT + 3, rect.top + 3, STRIP_HEIGHT - 6, STRIP_HEIGHT - 6)
            self.tab_rects.append(rect)
            self.close_rects.append(close)
            x += width

    # Returns ("toggle", None), ("close", index), ("select", index) or None for a click,
    # and whether the event landed on the strip or the toggle box
    def handle_event(self, event, workflows):
        if not hasattr(event, "pos"):
            return None, False
        on_toggle = self.toggle_rect.collidepoint(event.pos)
        on_strip = self.expanded and self.top <= event.pos[1] < self.top + STRIP_HEIGHT
        if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
            return None, False
        if on_toggle:
            return ("toggle", None), True
        if not on_strip:
            return None, False

        self.layout(workflows)
        for index, (rect, close) in enumerate(zip(self.tab_rects, self.close_rects)):
            if close.collidepoint(event.pos):
                return ("close", index), True
            if rect.collidepoint(event.pos):
                return ("select", index), True
        return None, True

    # Draws the strip (when shown) with the active tab highlighted, then the toggle box
    def draw(self, screen, workflows, active):
        mouse = display.mouse_pos()
        if self.expanded:
            self.layout(workflows)
            pygame.draw.rect(screen, colours.TOOLBAR, (0, self.top, self.width, STRIP_HEIGHT))
            pygame.draw.line(
                screen, colours.TOOLBAR_LINE, (0, self.top + STRIP_HEIGHT - 1), (self.width, self.top + STRIP_HEIGHT - 1)
            )
            for index, open_workflow in enumerate(workflows):
                rect, close = self.tab_rects[index], self.close_rects[index]
                if open_workflow is active:
                    colour = colours.TAB_SELECTED
                elif rect.collidepoint(mouse):
                    colour = colours.TAB_HOVER
                else:
                    colour = colours.TAB
                pygame.draw.rect(screen, colour, rect.inflate(-2, -2))

                screen.set_clip(pygame.Rect(rect.left, rect.top, close.left - rect.left - 2, rect.height))
                label = self.font.render(open_workflow.label(), True, colours.TEXT)
                screen.blit(label, label.get_rect(midleft=(rect.left + TAB_PADDING, rect.centery)))
                screen.set_clip(None)

                close_colour = colours.EXIT_HOVER if close.collidepoint(mouse) else colours.TAB_HOVER
                pygame.draw.rect(screen, close_colour, close)
                cross = self.font.render("x", True, colours.TEXT)
                screen.blit(cross, cross.get_rect(center=close.center))

        toggle_colour = colours.ZOOM_BUTTON_HOVER if self.toggle_rect.collidepoint(mouse) else colours.ZOOM_BUTTON
        pygame.draw.rect(screen, toggle_colour, self.toggle_rect)
        pygame.draw.rect(screen, colours.TOOLBAR_LINE, self.toggle_rect, 1)
        # An upward arrow hides the strip, a downward one shows it
        x, y = self.toggle_rect.center
        if self.expanded:
            points = [(x - 5, y + 3), (x + 5, y + 3), (x, y - 3)]
        else:
            points = [(x - 5, y - 3), (x + 5, y - 3), (x, y + 3)]
        pygame.draw.polygon(screen, colours.TEXT, points)
