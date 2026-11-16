from PySide6.QtCore import QEvent, QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from engine import catalogue
from ui.canvas import MODULE_MIME


ROW_HEIGHT = 50
MODULE_ROW_HEIGHT = 46
TAB_WIDTH = 132
BUTTON_HEIGHT = 36
BADGE_SIZE = 16
# Toolbar buttons that do something when clicked, rather than being dragged onto the canvas
ACTIONS = ["New", "Open", "Save", "Save As"]
FILE_TAB = "File"


class ModuleButton(QPushButton):
    # A module on the toolbar. Dragging it onto the canvas places it, a double click places it
    # in the middle of the canvas, and the ? in its corner opens its help
    help_asked = Signal(str, QPoint)
    placed = Signal(str)

    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.press = None
        self.setObjectName("module")
        self.setFixedSize(TAB_WIDTH, BUTTON_HEIGHT - 2)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("Drag onto the canvas, or double-click to add it")
        self.badge = QPushButton("?", self)
        self.badge.setObjectName("toggle")
        self.badge.setFixedSize(BADGE_SIZE, BADGE_SIZE)
        self.badge.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.badge.setStyleSheet(f"border-radius: {BADGE_SIZE // 2}px; font-size: 11px;")
        self.badge.move(TAB_WIDTH - BADGE_SIZE - 4, (BUTTON_HEIGHT - 2 - BADGE_SIZE) // 2)
        self.badge.setToolTip("What this module does")
        self.badge.clicked.connect(lambda: self.help_asked.emit(self.name, self.badge.mapToGlobal(QPoint(0, BADGE_SIZE))))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.press = event.position().toPoint()
        super().mousePressEvent(event)

    # Picks the module up once the mouse has moved far enough to be a drag
    def mouseMoveEvent(self, event):
        if self.press is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return
        if (event.position().toPoint() - self.press).manhattanLength() < QApplication.startDragDistance():
            return
        self.press = None
        self.setDown(False)
        data = QMimeData()
        data.setData(MODULE_MIME, self.name.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(data)
        picture = self.grab()
        drag.setPixmap(picture)
        drag.setHotSpot(QPoint(picture.width() // 2, picture.height() // 2))
        drag.exec(Qt.DropAction.CopyAction)

    def mouseDoubleClickEvent(self, event):
        self.placed.emit(self.name)


class Row(QScrollArea):
    # A row of buttons that scrolls sideways when they don't all fit: with the mouse wheel, or
    # with a two finger swipe on a trackpad, either sideways or up and down
    def __init__(self, name, height=ROW_HEIGHT):
        super().__init__()
        self.setObjectName(name)
        self.setFixedHeight(height)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        self.buttons = QHBoxLayout(holder)
        self.buttons.setContentsMargins(8, 0, 8, 0)
        self.buttons.setSpacing(6)
        self.buttons.addStretch()
        self.setWidget(holder)
        holder.installEventFilter(self)

    # Moves the row by whichever way the wheel or swipe went furthest. A trackpad gives its
    # movement in pixels where it can, and a swipe sideways arrives as x rather than y
    def wheelEvent(self, event):
        delta = event.pixelDelta() if not event.pixelDelta().isNull() else event.angleDelta()
        move = delta.x() if abs(delta.x()) > abs(delta.y()) else delta.y()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - move)
        event.accept()

    # Scrolls the row for a wheel or swipe over any of its buttons, rather than leaving it to
    # be passed up from the button, which a trackpad's swipes do not always manage
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and watched is not self.viewport():
            self.wheelEvent(event)
            return True
        return super().eventFilter(watched, event)

    def clear(self):
        while self.buttons.count() > 1:
            item = self.buttons.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()

    def add(self, widget):
        self.buttons.insertWidget(self.buttons.count() - 1, widget)
        for part in [widget] + widget.findChildren(QWidget):
            part.installEventFilter(self)


class Toolbar(QWidget):
    # Two rows: category tabs on top, with the app's own buttons at the right, and the selected
    # tab's modules underneath. With no tab open, as when the app starts, the second row is hidden
    action = Signal(str)
    module_placed = Signal(str)
    help_asked = Signal(str, QPoint)

    def __init__(self):
        super().__init__()
        rows = QVBoxLayout(self)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)

        top = QWidget()
        top.setObjectName("toolbar")
        top.setFixedHeight(ROW_HEIGHT)
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(0, 0, 8, 0)
        top_row.setSpacing(0)
        self.tabs = Row("tabs", ROW_HEIGHT - 1)
        top_row.addWidget(self.tabs, 1)

        # The app's own buttons sit apart from the tabs, behind a divider, so the tabs can never
        # run into them however many packs add tabs
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedSize(1, ROW_HEIGHT - 18)
        top_row.addSpacing(8)
        top_row.addWidget(divider)
        top_row.addSpacing(12)

        self.undo_button = self.app_button("Undo", "Undo the last change (Ctrl+Z)")
        self.redo_button = self.app_button("Redo", "Redo the change that was undone (Ctrl+Y)")
        self.settings_button = self.app_button("Settings", "Theme, full screen, significance stars and add-ons")
        self.run_button = self.app_button("Run", "Run what has changed (F5). Shift + click runs everything again")
        self.run_button.setObjectName("run")
        self.exit_button = self.app_button("Exit", "Close Coefflow")
        self.exit_button.setObjectName("exit")
        # Undo and Redo, then Settings, then Run and Exit, each pair a little apart
        groups = [[self.undo_button, self.redo_button], [self.settings_button], [self.run_button, self.exit_button]]
        for number, group in enumerate(groups):
            if number:
                top_row.addSpacing(14)
            for index, button in enumerate(group):
                if index:
                    top_row.addSpacing(4)
                top_row.addWidget(button)
        rows.addWidget(top)

        # The open tab's modules, on a shade of their own with the tab's name at the start, so
        # the row reads as belonging to that tab rather than as more of the top row
        self.modules_bar = QWidget()
        self.modules_bar.setObjectName("modules")
        self.modules_bar.setFixedHeight(MODULE_ROW_HEIGHT)
        bar_row = QHBoxLayout(self.modules_bar)
        bar_row.setContentsMargins(14, 0, 0, 0)
        bar_row.setSpacing(0)
        self.modules_label = QLabel()
        self.modules_label.setObjectName("rowlabel")
        bar_row.addWidget(self.modules_label)
        self.modules = Row("modulerow", MODULE_ROW_HEIGHT)
        bar_row.addWidget(self.modules, 1)
        self.modules_bar.hide()
        rows.addWidget(self.modules_bar)

        self.tab_buttons = {}
        self.selected = None
        self.refresh()

    def app_button(self, text, tip):
        button = QPushButton(text)
        button.setFixedHeight(BUTTON_HEIGHT)
        button.setMinimumWidth(84)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setToolTip(tip)
        return button

    # Remakes the tab buttons from the core modules and the added packs, closing any open tab.
    # Called again whenever a pack is added or removed
    def refresh(self):
        self.select(None)
        self.tabs.clear()
        self.tab_buttons = {}
        for name in [FILE_TAB] + list(catalogue.all_tabs()):
            button = QPushButton(name)
            button.setObjectName("tab")
            button.setCheckable(True)
            button.setFixedSize(TAB_WIDTH, BUTTON_HEIGHT)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda checked=False, name=name: self.select(None if self.selected == name else name))
            self.tabs.add(button)
            self.tab_buttons[name] = button

    # Highlights a tab and fills the second row with its modules, or hides the row for None
    def select(self, name):
        self.selected = name
        for tab, button in getattr(self, "tab_buttons", {}).items():
            button.setChecked(tab == name)
        self.modules.clear()
        self.modules_bar.setVisible(name is not None)
        if name is None:
            return
        self.modules_label.setText(name)

        if name == FILE_TAB:
            for action in ACTIONS:
                button = QPushButton(action)
                button.setObjectName("module")
                button.setFixedSize(TAB_WIDTH, BUTTON_HEIGHT - 2)
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                button.clicked.connect(lambda checked=False, action=action: self.action.emit(action))
                self.modules.add(button)
            return
        for module in catalogue.all_tabs().get(name, []):
            button = ModuleButton(module)
            button.help_asked.connect(self.help_asked.emit)
            button.placed.connect(self.module_placed.emit)
            self.modules.add(button)
