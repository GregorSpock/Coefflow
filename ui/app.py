import ctypes
import os
import sys

import pygame

from engine import addons, workflow
from ui import colours, dialogs, display, workspace
from ui.button import Button
from ui.canvas import Canvas
from ui.configure import configure_panel
from ui.help import HelpPopup
from ui.menu import Menu
from ui.settings import SettingsMenu
from ui.toolbar import Toolbar, ROW_HEIGHT
from ui.workflow_tabs import WorkflowTabs, OpenWorkflow, clear_temp, new_untitled


FPS = 60
# How often untitled workflows are written to their temp files, in milliseconds
TEMP_SAVE_INTERVAL = 2000
CLICK_DISTANCE = 4
PANEL_MARGIN = 30
# Windowed mode opens at this fraction of the screen size
WINDOW_SCALE = 0.8

# Stops the full screen window minimising when a file dialog opens in front of it
os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.png")

# On Windows, gives the app its own taskbar entry (with its own icon) rather than grouping it under Python
if sys.platform == "win32":
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Coefflow.App")
    except (AttributeError, OSError):
        pass


class App:
    # Sets up the full screen window, toolbar, canvas and buttons
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Coefflow")
        if os.path.exists(ICON_PATH):
            pygame.display.set_icon(pygame.image.load(ICON_PATH))

        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()
        self.fullscreen = True

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.zoom_font = pygame.font.Font(None, 40)
        self.running = True
        self.dragging = None
        self.moving_node = None
        self.grab_x = 0
        self.grab_y = 0
        self.press_pos = (0, 0)
        self.connecting = None
        self.panel = None
        self.help_popup = None

        self.toolbar = Toolbar(self.width, self.font)
        self.tabs = WorkflowTabs()
        self.workflows = []
        self.active = None
        self.untitled_count = 0
        self.last_temp_save = 0
        clear_temp()

        self.exit_button = Button(0, 4, 100, ROW_HEIGHT - 8, "Exit", self.font, "EXIT", "EXIT_HOVER")
        self.run_button = Button(0, 4, 100, ROW_HEIGHT - 8, "Run", self.font, "RUN", "RUN_HOVER")
        self.settings_button = Button(
            0, 4, 100, ROW_HEIGHT - 8, "Settings", self.font, "TAB", "TAB_HOVER", "TAB_SELECTED"
        )
        self.zoom_in_button = Button(0, 0, 50, 50, "+", self.zoom_font, "ZOOM_BUTTON", "ZOOM_BUTTON_HOVER")
        self.zoom_out_button = Button(0, 0, 50, 50, "-", self.zoom_font, "ZOOM_BUTTON", "ZOOM_BUTTON_HOVER")
        self.settings_menu = SettingsMenu(self.font)
        self.new_workflow()
        self.layout()

    # The canvas of the workflow on show
    @property
    def canvas(self):
        return self.active.canvas

    # Places the toolbar, canvas and buttons to fit the current window size
    def layout(self):
        self.width, self.height = self.screen.get_size()
        self.toolbar.width = self.width
        self.tabs.place(self.toolbar.height, self.width)
        for open_workflow in self.workflows:
            open_workflow.canvas.rect = self.canvas_rect()

        self.exit_button.rect.x = self.width - 110
        self.run_button.rect.x = self.width - 220
        self.settings_button.rect.x = self.width - 330
        self.toolbar.tab_room = self.settings_button.rect.x - 6
        self.zoom_in_button.rect.topleft = (self.width - 60, self.height - 110)
        self.zoom_out_button.rect.topleft = (self.width - 60, self.height - 58)

        if self.panel:
            self.open_panel(self.panel)

    # The area below the toolbar and the open workflows strip
    def canvas_rect(self):
        top = self.toolbar.height + self.tabs.height()
        return pygame.Rect(0, top, self.width, self.height - top)

    # Switches between full screen and a resizable window. Everything is still drawn at
    # full screen size, then scaled down to fit the window, so the layout never changes
    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        display.windowed = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        else:
            size = (int(self.width * WINDOW_SCALE), int(self.height * WINDOW_SCALE))
            pygame.display.set_mode(size, pygame.RESIZABLE)
            self.screen = pygame.Surface((self.width, self.height))

    # Loads the add-on packs, then shows the menu, then the main loop: handle input, draw,
    # then wait for the next frame
    def run(self):
        self.reload_addons()
        self.running = Menu(self.screen, self.font).run()

        while self.running:
            self.handle_events()
            self.save_temp()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        clear_temp()

    # Every few seconds, writes any changed untitled workflows to their temp files
    def save_temp(self):
        now = pygame.time.get_ticks()
        if now - self.last_temp_save < TEMP_SAVE_INTERVAL:
            return
        self.last_temp_save = now
        for open_workflow in self.workflows:
            open_workflow.update_temp()

    # Passes each event to the buttons, toolbar and canvas
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = not self.can_close_all()
            if hasattr(event, "pos"):
                event.pos = display.to_screen(event.pos)
            if self.help_popup and self.help_popup.closes_on(event):
                self.help_popup = None
                continue
            if self.exit_button.is_clicked(event):
                self.running = not self.can_close_all()
                continue
            if self.settings_button.is_clicked(event):
                self.settings_menu.toggle(self.settings_button.rect.right, ROW_HEIGHT, self.fullscreen)
                self.settings_button.selected = self.settings_menu.open
                continue
            if self.handle_settings(event):
                continue
            if self.panel and self.handle_panel(event):
                continue
            if self.handle_help(event):
                continue
            if self.handle_tabs(event):
                continue
            if self.handle_shortcut(event):
                continue
            if self.run_button.is_clicked(event):
                self.run_workflow()
                continue
            if self.zoom_in_button.is_clicked(event):
                self.canvas.zoom_in()
                continue
            if self.zoom_out_button.is_clicked(event):
                self.canvas.zoom_out()
                continue

            action = self.toolbar.action_clicked(event)
            if action == "New":
                self.new_workflow()
            elif action == "Open":
                self.open_workflow()
            elif action == "Save":
                self.save_workflow()
            elif action == "Save As":
                self.save_workflow_as()

            # Opening or closing a tab shows or hides the toolbar's second row, so the rest moves to fit
            toolbar_height = self.toolbar.height
            self.toolbar.handle_event(event)
            if self.toolbar.height != toolbar_height:
                self.layout()
            self.handle_drag(event)
            self.handle_connect(event)
            self.handle_move(event)
            self.canvas.handle_event(event)

    # Opens the help popup for a module whose ? was clicked, on the toolbar or the canvas.
    # Returns True if a ? was clicked
    def handle_help(self, event):
        name = self.toolbar.help_clicked(event)
        if not name and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.canvas.rect.collidepoint(event.pos):
                node = self.canvas.help_at(event.pos[0], event.pos[1])
                name = node.name if node else None
        if not name:
            return False
        self.help_popup = HelpPopup(name, event.pos, (self.width, self.height), self.font)
        return True

    # Passes events to the settings menu while it is open. Returns True if the menu used the event
    def handle_settings(self, event):
        choice, used = self.settings_menu.handle_event(event)
        self.settings_button.selected = self.settings_menu.open
        if choice == "Theme":
            colours.next_theme()
        elif choice == "Fullscreen":
            self.toggle_fullscreen()
        elif choice == "Configure":
            self.open_panel(configure_panel())
        elif choice == "Add pack":
            self.add_pack()
        elif choice == "Remove pack":
            self.remove_pack(self.settings_menu.pack)
        return used

    # Loads every pack in the add-ons folder and remakes the toolbar tabs to match.
    # Any pack or module that fails to load is reported
    def reload_addons(self):
        problems = addons.load_all()
        self.toolbar.refresh()
        self.layout()
        if self.settings_menu.open:
            self.settings_menu.show_addons()
        if problems:
            dialogs.show_message("Add-on problems", "\n".join(problems))

    # Copies a pack folder chosen from elsewhere on the computer into the add-ons folder, then
    # reloads the add-ons. A pack with the same folder name is replaced, after asking
    def add_pack(self):
        source = dialogs.choose_addon_folder()
        if not source:
            return
        target = None
        try:
            target = addons.destination(source)
            if target.exists() and not dialogs.ask_yes_no(
                "Replace add-on", f"An add-on called {target.name} is already loaded. Replace it?"
            ):
                return
            addons.add(source)
        except (OSError, ValueError) as error:
            dialogs.show_message("Could not add the pack", str(error))
        self.reload_addons()
        if target and target.name in addons.packs:
            problems = addons.load_modules(target.name)
            if problems:
                dialogs.show_message("Add-on problems", "\n".join(problems))

    # Deletes a pack from the add-ons folder after asking, then reloads the add-ons.
    # Its modules already on the canvas keep working until the workflow is opened again
    def remove_pack(self, pack):
        name = addons.packs[pack]["name"]
        if not dialogs.ask_yes_no("Remove add-on", f"Remove {name}? Its modules will be deleted from the add-ons folder."):
            return
        try:
            addons.remove(pack)
        except OSError as error:
            dialogs.show_message("Could not remove the pack", str(error))
        self.reload_addons()

    # Ctrl+S saves, Ctrl+Shift+S saves as, Ctrl+O opens, Ctrl+T opens a new workflow and
    # Ctrl+W closes the one on show. Returns True if a shortcut was used
    def handle_shortcut(self, event):
        if event.type != pygame.KEYDOWN or not event.mod & pygame.KMOD_CTRL:
            return False
        if event.key == pygame.K_s and event.mod & pygame.KMOD_SHIFT:
            self.save_workflow_as()
        elif event.key == pygame.K_s:
            self.save_workflow()
        elif event.key == pygame.K_o:
            self.open_workflow()
        elif event.key == pygame.K_t:
            self.new_workflow()
        elif event.key == pygame.K_w:
            self.close_workflow(self.active)
        else:
            return False
        return True

    # Shows the open workflow's name in the window title
    def update_title(self):
        pygame.display.set_caption(f"Coefflow - {self.active.name()}")

    # Passes clicks to the open workflows strip and its show/hide box.
    # Returns True if the event landed on either
    def handle_tabs(self, event):
        click, used = self.tabs.handle_event(event, self.workflows)
        if click:
            kind, index = click
            if kind == "toggle":
                self.tabs.expanded = not self.tabs.expanded
                self.layout()
            elif kind == "select":
                self.switch_to(self.workflows[index])
            elif kind == "close":
                self.close_workflow(self.workflows[index])
        return used

    # Shows another open workflow, dropping anything half done on the one being left
    def switch_to(self, open_workflow):
        self.active = open_workflow
        self.panel = None
        self.connecting = None
        self.moving_node = None
        self.dragging = None
        self.update_title()

    # Offers to save a workflow with changes before it is closed, showing it first.
    # Returns False if the user cancels, or chose to save and then cancelled the save
    def can_discard(self, open_workflow):
        if not open_workflow.has_changes():
            return True
        self.switch_to(open_workflow)
        answer = dialogs.ask_save_changes()
        if answer is None:
            return False
        return self.save_workflow() if answer else True

    # Before the app closes, offers to save each workflow with changes in turn
    def can_close_all(self):
        return all(self.can_discard(open_workflow) for open_workflow in list(self.workflows))

    # Closes a workflow's tab after offering to save it, deleting its temp file.
    # Closing the last one leaves a fresh untitled workflow open
    def close_workflow(self, open_workflow):
        shown = self.active
        if not self.can_discard(open_workflow):
            return
        index = self.workflows.index(open_workflow)
        self.workflows.remove(open_workflow)
        open_workflow.remove_temp()

        if not self.workflows:
            self.new_workflow()
        elif shown is open_workflow:
            self.switch_to(self.workflows[min(index, len(self.workflows) - 1)])
        else:
            self.switch_to(shown)

    # Saves to the workflow's file, or asks for one if it has never been saved.
    # Once saved somewhere real its temp file is no longer needed. Returns True if it was saved
    def save_workflow(self):
        if not self.active.path:
            return self.save_workflow_as()
        try:
            workspace.save(self.canvas, self.active.path)
        except OSError as error:
            dialogs.show_message("Could not save", str(error))
            return False
        self.active.saved_state = workspace.snapshot(self.canvas)
        self.active.remove_temp()
        self.update_title()
        return True

    def save_workflow_as(self):
        path = dialogs.choose_save_path()
        if not path:
            return False
        previous = self.active.path
        self.active.path = path
        if self.save_workflow():
            return True
        self.active.path = previous
        return False

    # Opens a fresh untitled workflow in a new tab, kept in the temp folder until it is saved
    def new_workflow(self):
        self.untitled_count += 1
        open_workflow = new_untitled(self.canvas_rect(), self.untitled_count)
        self.workflows.append(open_workflow)
        self.switch_to(open_workflow)

    # Opens a workflow from a file in a new tab, or shows it if it is already open.
    # An untouched untitled workflow on show is replaced rather than left behind.
    # Results are not stored in the file, so Run the workflow to see them again
    def open_workflow(self):
        path = dialogs.choose_workflow()
        if not path:
            return
        for open_workflow in self.workflows:
            if open_workflow.path and os.path.samefile(open_workflow.path, path):
                self.switch_to(open_workflow)
                return

        canvas = Canvas(self.canvas_rect())
        try:
            problems = workspace.load(canvas, path)
        except (OSError, ValueError) as error:
            dialogs.show_message("Could not open", str(error))
            return

        opened = OpenWorkflow(canvas, path)
        if self.active.is_blank():
            self.active.remove_temp()
            self.workflows[self.workflows.index(self.active)] = opened
        else:
            self.workflows.append(opened)
        self.switch_to(opened)
        if problems:
            dialogs.show_message("Opened with problems", "\n".join(problems))

    # Picks up a module from the toolbar and drops it onto the canvas
    def handle_drag(self, event):
        module = self.toolbar.module_clicked(event)
        if module:
            self.dragging = module

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            if self.canvas.rect.collidepoint(event.pos):
                self.canvas.add_node(self.dragging, event.pos[0], event.pos[1])
            self.dragging = None

    # Drags a line from an output port to an input port.
    # Grabbing an input that is already connected unplugs that line so it can be moved
    def handle_connect(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            port = self.canvas.port_at(event.pos[0], event.pos[1])
            if port and self.canvas.rect.collidepoint(event.pos):
                node, side, index = port
                if side == "output":
                    self.connecting = (node, index)
                else:
                    existing = self.canvas.connection_into(node, index)
                    if existing:
                        self.canvas.disconnect(existing)
                        self.connecting = (existing.source, existing.output_index)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.connecting:
            port = self.canvas.port_at(event.pos[0], event.pos[1])
            if port and port[1] == "input":
                source, output_index = self.connecting
                self.canvas.connect(source, output_index, port[0], port[2])
            self.connecting = None

    # Passes events to the open panel, forgets the panel once it is closed,
    # and swaps it for another if one of its buttons opened one
    def handle_panel(self, event):
        used = self.panel.handle_event(event)
        if self.panel.closed:
            self.panel = None
        elif self.panel.next_panel:
            self.open_panel(self.panel.next_panel)
        return used

    # Shows a node's settings panel on the right of the canvas
    def open_panel(self, panel):
        panel.canvas = self.canvas
        x = self.width - panel.rect.width - PANEL_MARGIN
        y = self.canvas.rect.top + PANEL_MARGIN
        height = self.zoom_in_button.rect.top - PANEL_MARGIN - y
        panel.place(x, y, height)
        self.panel = panel

    # Runs the workflow, then marks each node green if it worked or red if it failed
    def run_workflow(self):
        results, errors = workflow.run(self.canvas.nodes, self.canvas.connections)

        for node in self.canvas.nodes:
            node.error = errors.get(node.id)
            if node.id in errors:
                node.status = "error"
                print(f"{node.name} (#{node.id}): {errors[node.id]}")
            else:
                node.status = "done"

        if self.panel and self.panel.refresh:
            self.panel.refresh(self.panel)

    # Clicks or moves a placed node, and removes it if dropped on the toolbar
    def handle_move(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            on_port = self.canvas.port_at(event.pos[0], event.pos[1])
            if self.canvas.rect.collidepoint(event.pos) and not on_port:
                node = self.canvas.node_at(event.pos[0], event.pos[1])
                if node:
                    mouse_x, mouse_y = self.canvas.to_canvas(event.pos[0], event.pos[1])
                    self.grab_x = node.x - mouse_x
                    self.grab_y = node.y - mouse_y
                    self.press_pos = event.pos
                    self.moving_node = node
                    self.canvas.bring_to_front(node)

        if event.type == pygame.MOUSEMOTION and self.moving_node:
            mouse_x, mouse_y = self.canvas.to_canvas(event.pos[0], event.pos[1])
            self.moving_node.x = mouse_x + self.grab_x
            self.moving_node.y = mouse_y + self.grab_y

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.moving_node:
            moved_x = abs(event.pos[0] - self.press_pos[0])
            moved_y = abs(event.pos[1] - self.press_pos[1])

            if event.pos[1] < self.toolbar.height:
                self.canvas.remove_node(self.moving_node)
                if self.panel and self.panel.node is self.moving_node:
                    self.panel = None
            elif moved_x <= CLICK_DISTANCE and moved_y <= CLICK_DISTANCE:
                panel = self.moving_node.on_click()
                if panel:
                    self.open_panel(panel)

            self.moving_node = None

    # Draws the canvas first, then everything that sits on top of it
    def draw(self):
        self.screen.fill(colours.BACKGROUND)
        self.canvas.draw(self.screen)
        self.toolbar.draw(self.screen)
        self.tabs.draw(self.screen, self.workflows, self.active)

        self.exit_button.draw(self.screen)
        self.run_button.draw(self.screen)
        self.settings_button.draw(self.screen)
        self.zoom_in_button.draw(self.screen)
        self.zoom_out_button.draw(self.screen)

        if self.panel:
            self.panel.draw(self.screen)

        zoom_label = self.font.render(f"{round(self.canvas.zoom * 100)}%", True, colours.TEXT)
        zoom_rect = zoom_label.get_rect(midright=(self.width - 70, self.height - 58))
        self.screen.blit(zoom_label, zoom_rect)

        if self.connecting:
            node, index = self.connecting
            start = node.output_positions(self.canvas)[index]
            end = display.mouse_pos()
            pygame.draw.line(self.screen, colours.CONNECTION, start, end, self.canvas.line_width())

        if self.moving_node:
            self.moving_node.draw(self.screen, self.canvas)

        if self.dragging:
            mouse_x, mouse_y = display.mouse_pos()
            self.canvas.draw_preview(self.screen, self.dragging, mouse_x, mouse_y)

        self.settings_menu.draw(self.screen)
        if self.help_popup:
            self.help_popup.draw(self.screen)

        display.present(self.screen)
