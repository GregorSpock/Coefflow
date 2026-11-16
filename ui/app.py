import ctypes
import os
import shutil
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, QSplitter, QStackedWidget,
    QTabBar, QVBoxLayout, QWidget,
)

from engine import addons, catalogue, colours, runner, workspace
from engine.configure import configure_panel
from engine.history import History
from engine.node import load_module
from engine.workflow import Workflow
from ui import dialogs, theme
from ui.canvas import Canvas
from ui.help import HelpPopup
from ui.menu import StartMenu
from ui.panel_view import PANEL_WIDTH, PanelView
from ui.table_viewer import TableViewer
from ui.toolbar import Toolbar


# How often untitled workflows are written to their temp files, and how often the app looks
# for changes to keep in the undo history, both in milliseconds
TEMP_SAVE_INTERVAL = 2000
WATCH_INTERVAL = 300
# How long a message stays in the status bar, in milliseconds
MESSAGE_TIME = 6000

ROOT = Path(__file__).parent.parent
ICON_PATH = ROOT / "icon.png"
# Untitled workflows are kept as files in temp/ while they are open, so they always have a file
# behind them. The folder is emptied when the app starts and when it closes
TEMP_FOLDER = ROOT / "temp"
UNTITLED_NAME = "UntitledWorkflow"

# On Windows, gives the app its own taskbar entry (with its own icon) rather than grouping it under Python
if sys.platform == "win32":
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Coefflow.App")
    except (AttributeError, OSError):
        pass


# Deletes everything in the temp folder, making the folder if it is missing
def clear_temp():
    if TEMP_FOLDER.exists():
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
    TEMP_FOLDER.mkdir(exist_ok=True)


class OpenWorkflow:
    # One open workflow: the workflow itself, the canvas showing it, its undo history, the file
    # it is saved to (None until it is saved somewhere), its temp file while untitled, and its
    # state at the last save to tell if it has changed
    def __init__(self, workflow, path=None, temp_path=None):
        self.workflow = workflow
        self.canvas = Canvas(workflow)
        self.history = History(workflow)
        self.path = path
        self.temp_path = temp_path
        self.saved_state = workspace.snapshot(workflow)
        self.temp_state = self.saved_state

    def has_changes(self):
        return workspace.snapshot(self.workflow) != self.saved_state

    # An untitled workflow that has never been touched, which can be replaced by an opened file
    def is_blank(self):
        return self.path is None and not self.workflow.nodes and not self.has_changes()

    def name(self):
        return Path(self.path or self.temp_path).stem

    # The name with an asterisk added when there are unsaved changes
    def label(self):
        return self.name() + ("*" if self.has_changes() else "")

    # Rewrites an untitled workflow's temp file if it has changed since it was last written
    def update_temp(self):
        if not self.temp_path:
            return
        state = workspace.snapshot(self.workflow)
        if state != self.temp_state:
            try:
                workspace.save(self.workflow, self.temp_path, self.canvas.view())
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


class RunThread(QThread):
    # Does the work of a run away from the window, so the window stays in use meanwhile.
    # status is sent as each node starts and finishes
    status = Signal(object, str)

    def __init__(self, run):
        super().__init__()
        self.run_plan = run
        self.failure = None

    # A module that fails is the run's business, and is reported on its node. Anything else
    # going wrong is kept in failure, so the window can say so rather than the run just ending
    def run(self):
        try:
            self.run_plan.execute(self.status.emit)
        except Exception as error:
            traceback.print_exc()
            self.failure = f"{type(error).__name__}: {error}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coefflow")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.workflows = []
        self.active = None
        self.untitled_count = 0
        self.run_thread = None
        self.run_plan = None
        self.help_popup = None
        clear_temp()

        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.menu = StartMenu()
        self.menu.started.connect(self.start)
        self.menu.exit_asked.connect(self.close)
        self.menu.theme_asked.connect(self.next_theme)
        self.pages.addWidget(self.menu)
        self.pages.addWidget(self.build_main())

        self.make_shortcuts()
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.save_temp)
        self.temp_timer.start(TEMP_SAVE_INTERVAL)
        self.watch_timer = QTimer(self)
        self.watch_timer.timeout.connect(self.watch)
        self.watch_timer.start(WATCH_INTERVAL)

        self.reload_addons()
        self.new_workflow()

    # ----- Building the window -----

    # The toolbar, the open workflows strip, the canvas with the panel beside it, and the status bar
    def build_main(self):
        page = QWidget()
        rows = QVBoxLayout(page)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)

        self.toolbar = Toolbar()
        self.toolbar.action.connect(self.do_action)
        self.toolbar.module_placed.connect(lambda name: self.canvas.add_node(name))
        self.toolbar.help_asked.connect(self.show_help)
        self.toolbar.undo_button.clicked.connect(self.undo)
        self.toolbar.redo_button.clicked.connect(self.redo)
        self.toolbar.run_button.clicked.connect(self.run_clicked)
        self.toolbar.exit_button.clicked.connect(self.close)
        self.toolbar.settings_button.clicked.connect(self.show_settings)
        rows.addWidget(self.toolbar)

        self.tabs = QTabBar()
        self.tabs.setTabsClosable(True)
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tabs.currentChanged.connect(self.tab_chosen)
        self.tabs.tabCloseRequested.connect(lambda index: self.close_workflow(self.workflows[index]))
        rows.addWidget(self.tabs)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)
        self.canvases = QStackedWidget()
        self.panel_view = PanelView()
        self.panel_view.closed.connect(self.close_panel)
        self.panel_view.edited.connect(self.panel_edited)
        self.panel_view.hide()
        self.splitter.addWidget(self.canvases)
        self.splitter.addWidget(self.panel_view)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        rows.addWidget(self.splitter, 1)

        status = QWidget()
        status.setObjectName("status")
        status.setFixedHeight(34)
        status_row = QHBoxLayout(status)
        status_row.setContentsMargins(10, 0, 8, 0)
        self.status_label = QLabel()
        self.status_label.setObjectName("small")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("small")
        status_row.addWidget(self.status_label, 1)
        for text, tip, action in [
            ("Fit", "Fit the whole workflow in the window", lambda: self.canvas.zoom_to_fit()),
            ("-", "Zoom out", lambda: self.canvas.zoom_out()),
        ]:
            status_row.addWidget(self.status_button(text, tip, action))
        status_row.addWidget(self.zoom_label)
        status_row.addWidget(self.status_button("+", "Zoom in", lambda: self.canvas.zoom_in()))
        rows.addWidget(status)

        self.message_timer = QTimer(self)
        self.message_timer.setSingleShot(True)
        self.message_timer.timeout.connect(lambda: self.status_label.setText(""))
        return page

    def status_button(self, text, tip, action):
        button = QPushButton(text)
        button.setObjectName("small")
        button.setFixedHeight(24)
        button.setMinimumWidth(30)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setToolTip(tip)
        button.clicked.connect(action)
        return button

    # Ctrl+S saves, Ctrl+Shift+S saves as, Ctrl+O opens, Ctrl+T or Ctrl+N opens a new workflow,
    # Ctrl+W closes the one on show, Ctrl+Z undoes and Ctrl+Y or Ctrl+Shift+Z redoes. F5 runs
    # what has changed, Shift+F5 runs everything, Escape stops a run or closes the panel, and
    # F11 switches full screen. Copy, paste and delete belong to the canvas (see ui/canvas.py)
    def make_shortcuts(self):
        shortcuts = [
            ("Ctrl+S", self.save_workflow), ("Ctrl+Shift+S", self.save_workflow_as), ("Ctrl+O", self.open_workflow),
            ("Ctrl+T", self.new_workflow), ("Ctrl+N", self.new_workflow), ("Ctrl+W", lambda: self.close_workflow(self.active)),
            ("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo),
            ("F5", self.run_workflow), ("Shift+F5", lambda: self.run_workflow(force=True)),
            ("Escape", self.escape), ("F11", self.toggle_fullscreen),
        ]
        for keys, function in shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(keys))
            action.triggered.connect(lambda checked=False, function=function: self.shortcut(function))
            self.addAction(action)

    # Shortcuts do nothing on the opening screen
    def shortcut(self, function):
        if self.pages.currentIndex() == 1:
            function()

    # ----- The opening screen -----

    # Leaves the opening screen for the canvas, loading every module first if asked to
    def start(self, load_all):
        if load_all:
            self.load_all()
        self.pages.setCurrentIndex(1)
        self.canvas.setFocus()

    # Loads every toolbar module in turn, showing which one is loading, then lists any that failed
    def load_all(self):
        names = catalogue.all_modules()
        problems = []
        for count, name in enumerate(names):
            self.menu.show_loading(name, count, len(names))
            QApplication.processEvents()
            try:
                load_module(name)
            except Exception as error:
                problems.append(f"{name} could not be loaded ({error}).")
        self.menu.show_loading(None)
        if problems:
            dialogs.show_message("Module problems", "\n".join(problems))

    # ----- Settings -----

    # The drop-down under the Settings button: theme, full screen, Configure and the add-on packs
    def show_settings(self):
        menu = QMenu(self)
        menu.addAction(f"Theme: {colours.current_theme}", self.next_theme)
        menu.addAction("Windowed" if self.isFullScreen() else "Fullscreen", self.toggle_fullscreen)
        menu.addAction("Configure", lambda: self.open_panel(configure_panel()))
        packs = menu.addMenu("Add-ons")
        packs.addAction("Core modules: Always on").setEnabled(False)
        for pack, details in addons.packs.items():
            label = f"{details['name']}: Remove"
            if details["version"]:
                label += f"   (v{details['version']})"
            packs.addAction(label, lambda pack=pack: self.remove_pack(pack))
        packs.addSeparator()
        packs.addAction("Add pack...", self.add_pack)
        button = self.toolbar.settings_button
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def next_theme(self):
        colours.next_theme()
        self.apply_theme()

    # Gives everything the current theme's look
    def apply_theme(self):
        QApplication.instance().setStyleSheet(theme.style_sheet())
        self.menu.apply_theme()
        for open_workflow in self.workflows:
            open_workflow.canvas.refresh()
        if self.panel_view.panel:
            self.panel_view.sync()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    # Loads every pack in the add-ons folder and remakes the toolbar tabs to match.
    # Any pack or module that fails to load is reported
    def reload_addons(self):
        problems = addons.load_all()
        self.toolbar.refresh()
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

    # ----- Workflows and their tabs -----

    # The canvas of the workflow on show
    @property
    def canvas(self):
        return self.active.canvas

    def do_action(self, action):
        if self.running():
            return
        {"New": self.new_workflow, "Open": self.open_workflow, "Save": self.save_workflow, "Save As": self.save_workflow_as}[action]()

    # Adds a workflow's canvas and tab, and shows it
    def add_workflow(self, open_workflow, replacing=None):
        canvas = open_workflow.canvas
        canvas.node_clicked.connect(self.node_clicked)
        canvas.help_asked.connect(self.show_help)
        canvas.run_to_here.connect(lambda node: self.run_workflow(only=[node]))
        canvas.view_asked.connect(self.view_data)
        canvas.changed.connect(self.watch)
        canvas.message.connect(self.say)
        canvas.zoomed.connect(self.show_zoom)
        self.canvases.addWidget(canvas)

        self.tabs.blockSignals(True)
        if replacing:
            index = self.workflows.index(replacing)
            self.workflows[index] = open_workflow
            self.canvases.removeWidget(replacing.canvas)
            replacing.canvas.deleteLater()
            self.tabs.setTabText(index, open_workflow.label())
        else:
            self.workflows.append(open_workflow)
            self.tabs.addTab(open_workflow.label())
        self.tabs.blockSignals(False)
        self.switch_to(open_workflow)

    # Shows another open workflow, closing the panel of the one being left
    def switch_to(self, open_workflow):
        self.active = open_workflow
        self.close_panel()
        self.canvases.setCurrentWidget(open_workflow.canvas)
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(self.workflows.index(open_workflow))
        self.tabs.blockSignals(False)
        self.show_zoom()
        self.update_title()

    def tab_chosen(self, index):
        if 0 <= index < len(self.workflows) and self.workflows[index] is not self.active:
            if self.running():
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(self.workflows.index(self.active))
                self.tabs.blockSignals(False)
                self.say("Wait for the run to finish, or stop it, before changing workflow")
            else:
                self.switch_to(self.workflows[index])

    # Shows the open workflow's name in the window title, each tab's name with an asterisk if it
    # has unsaved changes, and whether there is anything to undo or redo
    def update_title(self):
        if not self.active:
            return
        self.setWindowTitle(f"Coefflow - {self.active.name()}")
        for index, open_workflow in enumerate(self.workflows):
            if self.tabs.tabText(index) != open_workflow.label():
                self.tabs.setTabText(index, open_workflow.label())
        self.toolbar.undo_button.setEnabled(self.active.history.can_undo() and not self.running())
        self.toolbar.redo_button.setEnabled(self.active.history.can_redo() and not self.running())

    # Opens a fresh untitled workflow in a new tab, kept in the temp folder until it is saved
    def new_workflow(self):
        if self.running():
            return
        self.untitled_count += 1
        temp_path = TEMP_FOLDER / f"{UNTITLED_NAME}_{self.untitled_count}.confw"
        open_workflow = OpenWorkflow(Workflow(), temp_path=str(temp_path))
        try:
            TEMP_FOLDER.mkdir(exist_ok=True)
            workspace.save(open_workflow.workflow, temp_path)
        except OSError:
            pass
        self.add_workflow(open_workflow)

    # Opens a workflow from a file in a new tab, or shows it if it is already open.
    # An untouched untitled workflow on show is replaced rather than left behind.
    # Results are not stored in the file, so Run the workflow to see them again
    def open_workflow(self, path=None):
        if self.running():
            return
        path = path or dialogs.choose_workflow()
        if not path:
            return
        for open_workflow in self.workflows:
            if open_workflow.path and os.path.exists(open_workflow.path) and os.path.samefile(open_workflow.path, path):
                self.switch_to(open_workflow)
                return

        try:
            workflow, view, problems = workspace.load(path)
        except (OSError, ValueError) as error:
            dialogs.show_message("Could not open", str(error))
            return

        opened = OpenWorkflow(workflow, path)
        blank = self.active if self.active and self.active.is_blank() else None
        if blank:
            blank.remove_temp()
        self.add_workflow(opened, replacing=blank)
        opened.canvas.set_view(view)
        if problems:
            dialogs.show_message("Opened with problems", "\n".join(problems))

    # Offers to save a workflow with changes before it is closed, showing it first.
    # Returns False if the user cancels, or chose to save and then cancelled the save
    def can_discard(self, open_workflow):
        if not open_workflow.has_changes():
            return True
        self.switch_to(open_workflow)
        answer = dialogs.ask_save_changes(open_workflow.name())
        if answer is None:
            return False
        return self.save_workflow() if answer else True

    # Closes a workflow's tab after offering to save it, deleting its temp file.
    # Closing the last one leaves a fresh untitled workflow open
    def close_workflow(self, open_workflow):
        if self.running():
            self.say("Wait for the run to finish, or stop it, before closing a workflow")
            return
        shown = self.active
        if not self.can_discard(open_workflow):
            return
        index = self.workflows.index(open_workflow)
        self.workflows.remove(open_workflow)
        open_workflow.remove_temp()
        self.tabs.blockSignals(True)
        self.tabs.removeTab(index)
        self.tabs.blockSignals(False)
        self.canvases.removeWidget(open_workflow.canvas)
        open_workflow.canvas.deleteLater()

        if not self.workflows:
            self.active = None
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
            workspace.save(self.active.workflow, self.active.path, self.canvas.view())
        except OSError as error:
            dialogs.show_message("Could not save", str(error))
            return False
        self.active.saved_state = workspace.snapshot(self.active.workflow)
        self.active.remove_temp()
        self.update_title()
        self.say(f"Saved {self.active.path}")
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

    # Every few seconds, writes any changed untitled workflows to their temp files
    def save_temp(self):
        for open_workflow in self.workflows:
            open_workflow.update_temp()

    # ----- Undo and redo -----

    # Looks for changes to the workflow on show: a change is kept in its undo history, and any
    # node whose results no longer match its settings or inputs goes back to not run.
    # Nothing is kept while the mouse is down, so a drag is one change rather than many
    def watch(self):
        if not self.active or self.running() or QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton:
            return
        self.active.history.record()
        if runner.mark_stale(self.active.workflow):
            self.canvas.refresh()
        self.update_title()

    def undo(self):
        self.step_history(self.active.history.undo, "Nothing to undo")

    def redo(self):
        self.step_history(self.active.history.redo, "Nothing to redo")

    # Undoes or redoes, then shows the workflow as it now is. The open panel is opened again
    # from its node, as its settings may be among what changed, or closed if its node has gone
    def step_history(self, step, nothing):
        if self.running():
            return
        if not step():
            self.say(nothing)
            return
        self.canvas.rebuild()
        runner.mark_stale(self.active.workflow)
        panel = self.panel_view.panel if self.panel_view.isVisible() else None
        if panel and panel.node.id is not None:
            node = self.active.workflow.node_by_id(panel.node.id)
            if node:
                self.node_clicked(node)
            else:
                self.close_panel()
        self.update_title()

    # ----- Panels and help -----

    def node_clicked(self, node):
        try:
            panel = node.on_click()
        except Exception as error:
            dialogs.show_message("Could not open the settings", f"{type(error).__name__}: {error}")
            return
        if panel:
            self.open_panel(panel)

    # Shows a settings panel beside the canvas
    def open_panel(self, panel):
        first = not self.panel_view.isVisible()
        self.panel_view.show_panel(panel, self.active.workflow)
        self.panel_view.show()
        self.panel_view.setEnabled(not self.running())
        if first:
            total = self.splitter.width()
            self.splitter.setSizes([max(200, total - PANEL_WIDTH), PANEL_WIDTH])

    def close_panel(self):
        self.panel_view.hide()
        self.panel_view.panel = None

    # A change made in the panel may add or remove ports and lines, so the canvas is shown again
    def panel_edited(self):
        for node in self.active.workflow.nodes:
            self.active.workflow.tidy_inputs(node)
        self.canvas.rebuild()

    def show_help(self, name, position):
        self.help_popup = HelpPopup(name, position, self)
        self.help_popup.show()

    def escape(self):
        if self.running():
            self.stop_run()
        elif self.panel_view.isVisible():
            self.close_panel()

    # Shows a message in the status bar for a few seconds
    def say(self, text):
        self.status_label.setText(text)
        self.message_timer.start(MESSAGE_TIME)

    def show_zoom(self):
        if self.active:
            self.zoom_label.setText(f"{round(self.canvas.zoom * 100)}%")

    # ----- Running -----

    def running(self):
        return self.run_thread is not None

    # The Run button runs what has changed, or everything with Shift held. While a run is under
    # way it is the Stop button
    def run_clicked(self):
        if self.running():
            self.stop_run()
        else:
            shift = QGuiApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier
            self.run_workflow(force=bool(shift))

    # Runs the workflow on show on another thread, so the window stays in use: only the nodes
    # that have changed since they last ran, and those below them, unless force is set.
    # only limits the run to the given nodes and the ones they depend on.
    # The workflow can't be edited until the run finishes
    # Opens a window showing every table the node was given and gave back on the last run
    def view_data(self, node):
        viewer = TableViewer(node, self)
        viewer.show()
        return viewer

    def run_workflow(self, only=None, force=False):
        if self.running():
            return
        self.active.history.record()
        try:
            self.run_plan = runner.Run(self.active.workflow, only, force)
        except ValueError as error:
            dialogs.show_message("Could not run", str(error))
            return

        self.set_locked(True)
        self.say("Running...")
        self.run_thread = RunThread(self.run_plan)
        self.run_thread.status.connect(self.node_status)
        self.run_thread.finished.connect(self.run_finished)
        self.run_thread.start()

    def stop_run(self):
        if self.run_plan:
            self.run_plan.cancel()
            self.say("Stopping once the module that is running has finished...")

    # Shows a node as running, done or failed as the run reaches it
    def node_status(self, node, status):
        node.status = status
        if status == "running":
            self.status_label.setText(f"Running {node.name} #{node.id}...")
        self.canvas.refresh()

    # Writes the run's results onto the nodes, then marks each node green if it worked or red if it failed
    def run_finished(self):
        run, self.run_plan = self.run_plan, None
        self.run_thread.wait()
        failure = self.run_thread.failure
        self.run_thread = None
        if failure:
            dialogs.show_message("The run stopped unexpectedly", failure)
        errors = run.apply()
        for node_id, error in errors.items():
            node = self.active.workflow.node_by_id(node_id)
            print(f"{node.name} (#{node_id}): {error}")

        self.set_locked(False)
        self.canvas.refresh()
        if self.panel_view.isVisible():
            self.panel_view.refresh()

        ran, kept = run.counts()
        parts = [f"Ran {ran} module{'s' if ran != 1 else ''}"]
        if kept:
            parts.append(f"kept the results of {kept} that had not changed")
        if errors:
            parts.append(f"{len(errors)} failed")
        if run.stop:
            parts.append("stopped early")
        self.say(", ".join(parts))
        self.update_title()

    # Stops the workflow being edited while it runs. The canvas can still be moved and zoomed
    def set_locked(self, locked):
        self.canvas.locked = locked
        self.panel_view.setEnabled(not locked)
        self.toolbar.modules.setEnabled(not locked)
        self.toolbar.run_button.setText("Stop" if locked else "Run")
        self.toolbar.run_button.setObjectName("exit" if locked else "run")
        self.toolbar.run_button.style().unpolish(self.toolbar.run_button)
        self.toolbar.run_button.style().polish(self.toolbar.run_button)
        self.update_title()

    # ----- Closing -----

    # Before the app closes, stops any run and offers to save each workflow with changes in turn
    def closeEvent(self, event):
        if self.running():
            self.run_plan.cancel()
            self.run_thread.wait()
            self.run_finished()
        if all(self.can_discard(open_workflow) for open_workflow in list(self.workflows)):
            clear_temp()
            event.accept()
        else:
            event.ignore()


# Starts the app: the opening screen first, then the canvas
def run():
    app = QApplication(sys.argv)
    app.setApplicationName("Coefflow")
    app.setStyleSheet(theme.style_sheet())
    dialogs.install()
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
