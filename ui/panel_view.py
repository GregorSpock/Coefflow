import traceback

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QTableView, QVBoxLayout, QWidget,
)

from engine import chart, significance
from ui import dialogs
from ui.chart_settings import ChartSettings
from ui.painter import ChartWidget, export_png


# Shows a module's settings panel (engine/panel.py) beside the canvas. The panel says what it
# holds and this draws it with real buttons, a text box, a table that scrolls and sorts, and
# the chart. After every click and keystroke the module's function runs and the view is brought
# back in step with whatever the panel now holds

PANEL_WIDTH = 720
BUTTON_COLUMNS = 5
OPTION_HEIGHT = 30
# The most rows a preview table shows at once. Sorting and scrolling work on all of them
COLUMN_WIDTH = 110


# Turns a table value into short text, keeping decimals tidy
def cell_text(value):
    if isinstance(value, float):
        return "" if pd.isna(value) else f"{value:.4g}"
    if value is None:
        return ""
    return str(value)


class TableModel(QAbstractTableModel):
    # Shows a pandas table. Clicking a column's heading sorts the rows shown by it, which never
    # changes the data itself
    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return cell_text(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section])
        return str(section + 1)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if column < 0 or column >= len(self.frame.columns):
            return
        try:
            ordered = self.frame.iloc[:, column].sort_values(
                ascending=order == Qt.SortOrder.AscendingOrder, kind="stable", na_position="last"
            )
        except TypeError:
            return
        self.layoutAboutToBeChanged.emit()
        self.frame = self.frame.loc[ordered.index]
        self.layoutChanged.emit()


class ElidedLabel(QLabel):
    # A label that shortens its text with "..." to fit, and shows it in full when hovered over
    def __init__(self, text):
        super().__init__()
        self.full = text
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        text = QFontMetrics(self.font()).elidedText(self.full, Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)


class FieldBox(QLineEdit):
    # The panel's text box. It tells the panel's field when it is no longer the one being typed in
    def __init__(self):
        super().__init__()
        self.field = None

    def focusOutEvent(self, event):
        if self.field:
            self.field.focused = False
        super().focusOutEvent(event)


def small_button(text, name="small"):
    button = QPushButton(text)
    button.setObjectName(name)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button


def heading(text):
    label = QLabel(text)
    label.setObjectName("heading")
    return label


# A small box with an arrow that shows or hides a section: down to show it, up to hide it
def toggle_button():
    button = QPushButton()
    button.setObjectName("toggle")
    button.setFixedSize(20, 20)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button


class PanelView(QFrame):
    # closed is sent when the panel's X is clicked, and edited after anything that may have
    # changed the node's settings.
    # The inputs & outputs section starts hidden, and showing or hiding it, or the options when
    # they have a heading, applies to every panel
    closed = Signal()
    edited = Signal()
    ports_expanded = False
    options_expanded = True

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        self.setMinimumWidth(420)
        self.panel = None
        self.workflow = None
        self.shown_field = None
        self.shown_preview = None
        self.settings_open = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # A title bar of its own shade across the top, so the panel reads as a separate pane
        # rather than carrying on from the toolbar above it
        head = QWidget()
        head.setObjectName("panelhead")
        head.setFixedHeight(48)
        top = QHBoxLayout(head)
        top.setContentsMargins(14, 0, 8, 0)
        self.title = QLabel()
        self.title.setObjectName("paneltitle")
        close = QPushButton("X")
        close.setObjectName("close")
        close.setFixedSize(32, 30)
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.setToolTip("Close the panel (Esc)")
        close.clicked.connect(self.closed.emit)
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(close)
        outer.addWidget(head)

        rows = QVBoxLayout()
        rows.setContentsMargins(14, 12, 12, 10)
        rows.setSpacing(8)
        outer.addLayout(rows, 1)

        self.buttons = QGridLayout()
        self.buttons.setSpacing(6)
        rows.addLayout(self.buttons)

        self.field_row = QWidget()
        field_layout = QHBoxLayout(self.field_row)
        field_layout.setContentsMargins(0, 0, 0, 0)
        self.field_label = QLabel()
        self.field_box = FieldBox()
        self.field_box.textEdited.connect(self.typed)
        self.field_box.cursorPositionChanged.connect(self.cursor_moved)
        field_layout.addWidget(self.field_label)
        field_layout.addWidget(self.field_box, 1)
        rows.addWidget(self.field_row)

        self.lines = QLabel()
        self.lines.setWordWrap(True)
        self.lines.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rows.addWidget(self.lines)
        self.error = QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rows.addWidget(self.error)

        self.options_heading = QWidget()
        options_head = QHBoxLayout(self.options_heading)
        options_head.setContentsMargins(0, 0, 0, 0)
        self.options_title = heading("")
        self.options_toggle = toggle_button()
        self.options_toggle.clicked.connect(self.toggle_options)
        options_head.addWidget(self.options_title)
        options_head.addWidget(self.options_toggle)
        options_head.addStretch()
        rows.addWidget(self.options_heading)
        self.options = QScrollArea()
        self.options.setWidgetResizable(True)
        self.options.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows.addWidget(self.options)

        self.ports_heading = QWidget()
        ports_head = QHBoxLayout(self.ports_heading)
        ports_head.setContentsMargins(0, 0, 0, 0)
        self.ports_toggle = toggle_button()
        self.ports_toggle.clicked.connect(self.toggle_ports)
        ports_head.addWidget(heading("Inputs & Outputs"))
        ports_head.addWidget(self.ports_toggle)
        ports_head.addStretch()
        rows.addWidget(self.ports_heading)
        self.ports = QLabel()
        self.ports.setObjectName("small")
        self.ports.setWordWrap(True)
        self.ports.setTextFormat(Qt.TextFormat.RichText)
        rows.addWidget(self.ports)

        preview_head = QHBoxLayout()
        self.preview_title = heading("")
        self.size_label = QLabel()
        self.size_label.setObjectName("hint")
        self.settings_button = small_button("Settings")
        self.settings_button.setCheckable(True)
        self.settings_button.clicked.connect(self.toggle_chart_settings)
        self.export_button = small_button("Export PNG")
        self.export_button.clicked.connect(self.export_chart)
        self.view_button = small_button("View all")
        self.view_button.setToolTip("Open every table this module was given and gave back in a window, to sort, search, filter and summarise")
        self.view_button.clicked.connect(self.view_all)
        preview_head.addWidget(self.preview_title)
        preview_head.addWidget(self.size_label)
        preview_head.addStretch()
        preview_head.addWidget(self.settings_button)
        preview_head.addWidget(self.export_button)
        preview_head.addWidget(self.view_button)
        rows.addLayout(preview_head)

        self.legend = QLabel()
        self.legend.setObjectName("small")
        rows.addWidget(self.legend)

        self.chart_settings = ChartSettings()
        self.chart_settings.changed.connect(self.style_changed)
        rows.addWidget(self.chart_settings)

        self.stack = QStackedWidget()
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setDefaultSectionSize(COLUMN_WIDTH)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.chart = ChartWidget()
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.stack.addWidget(self.table)
        self.stack.addWidget(self.chart)
        self.stack.addWidget(self.message)
        rows.addWidget(self.stack, 1)

        self.note = QLabel()
        self.note.setObjectName("note")
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rows.addWidget(self.note)

    # ----- Showing a panel -----

    # Shows a panel in place of whatever was shown. workflow is the one its node is in
    def show_panel(self, panel, workflow):
        self.panel = panel
        self.workflow = workflow
        panel.canvas = workflow
        self.settings_open = False
        self.shown_preview = None
        self.shown_field = None
        self.sync()

    # Runs the panel's own refresh, as after the workflow has run, and shows the result
    def refresh(self):
        if not self.panel:
            return
        if self.panel.refresh:
            self.act(lambda: self.panel.refresh(self.panel))
        else:
            self.sync()

    # Runs one of the module's functions, then brings the view back in step with the panel.
    # A function that fails says why in a message rather than stopping the app
    def act(self, function):
        panel = self.panel
        try:
            function()
        except Exception as error:
            traceback.print_exc()
            dialogs.show_message("Something went wrong", f"{type(error).__name__}: {error}")

        if panel.closed:
            self.closed.emit()
            return
        if panel.next_panel:
            following, panel.next_panel = panel.next_panel, None
            self.show_panel(following, self.workflow)
        else:
            self.sync()
        self.edited.emit()

    def clicked(self, button):
        action = self.panel.actions.get(button)
        if action:
            self.act(lambda: action(self.panel, button))

    # Typing in the text box: what was typed goes to the panel's field, less any characters it
    # doesn't allow, and its on_change runs
    def typed(self, text):
        field = self.panel.text_field
        if not field:
            return
        allowed = field.allowed(text)
        cursor = self.field_box.cursorPosition() - (len(text) - len(allowed))
        field.text = allowed
        field.cursor = max(0, min(cursor, len(allowed)))
        field.focused = True
        self.act(lambda: field.on_change(self.panel, field.text))

    def cursor_moved(self, old, new):
        field = self.panel.text_field if self.panel else None
        if field and self.field_box.text() == field.text:
            field.cursor = new

    def toggle_options(self):
        PanelView.options_expanded = not PanelView.options_expanded
        self.sync()

    def toggle_ports(self):
        PanelView.ports_expanded = not PanelView.ports_expanded
        self.sync()

    def toggle_chart_settings(self):
        self.settings_open = not self.settings_open
        if self.settings_open:
            self.chart_settings.show_for(self.panel.node, self.panel.chart_axes)
        self.sync()

    def style_changed(self):
        self.chart.show_chart(self.panel.chart, self.panel.node.settings.get(chart.STYLE_KEY))
        self.edited.emit()

    # Asks where to save the chart, then saves it as a PNG picture in its style
    def export_chart(self):
        if not self.panel.chart:
            return
        path = dialogs.choose_png_path()
        if not path:
            return
        try:
            export_png(self.panel.chart, self.panel.node.settings.get(chart.STYLE_KEY), path)
        except Exception as error:
            dialogs.show_message("Could not export the chart", str(error))

    # ----- Bringing the view in step with the panel -----

    def sync(self):
        panel = self.panel
        if not panel:
            return
        node = panel.node
        number = f"  #{node.id}" if node.id is not None else ""
        self.title.setText(f"{panel.title}{number}")

        self.sync_buttons()
        self.sync_field()

        self.lines.setText("\n".join(panel.lines))
        self.lines.setVisible(bool(panel.lines))
        self.error.setText(f"Last run error: {node.error}" if node.error else "")
        self.error.setVisible(bool(node.error))

        self.sync_options()
        self.sync_ports()

        self.preview_title.setText(panel.preview_title)
        has_chart = panel.preview is None and panel.chart is not None
        self.settings_button.setVisible(has_chart)
        self.settings_button.setChecked(self.settings_open and has_chart)
        self.export_button.setVisible(has_chart)
        self.chart_settings.setVisible(self.settings_open and has_chart)

        # The chart's settings take the place of the options, so the chart keeps its room
        if self.settings_open and has_chart:
            for section in [self.options_heading, self.options, self.ports_heading, self.ports]:
                section.setVisible(False)

        self.view_button.setVisible(bool(node.last_outputs or node.last_inputs))

        stars = panel.preview is not None and "stars" in panel.preview.columns
        self.legend.setText(f"Stars: {significance.legend()}" if stars else "")
        self.legend.setVisible(stars)

        if panel.preview is not None:
            self.sync_table()
            self.stack.setCurrentWidget(self.table)
        elif panel.chart:
            self.size_label.setText("")
            self.chart.show_chart(panel.chart, node.settings.get(chart.STYLE_KEY))
            self.stack.setCurrentWidget(self.chart)
        else:
            self.size_label.setText("")
            self.message.setText(panel.message)
            self.stack.setCurrentWidget(self.message)

        self.note.setText(panel.note)
        self.note.setVisible(bool(panel.note))

    # Opens the table viewer (ui/table_viewer.py) on this panel's node
    def view_all(self):
        from ui.table_viewer import TableViewer
        self.viewer = TableViewer(self.panel.node, self.window())
        self.viewer.show()

    def sync_buttons(self):
        clear_layout(self.buttons)
        for index, button in enumerate(self.panel.buttons):
            widget = small_button(button.text, "")
            widget.setCheckable(button.selected)
            widget.setChecked(button.selected)
            widget.clicked.connect(lambda checked=False, button=button: self.clicked(button))
            self.buttons.addWidget(widget, index // BUTTON_COLUMNS, index % BUTTON_COLUMNS)
        for column in range(BUTTON_COLUMNS):
            self.buttons.setColumnStretch(column, 1)

    # The text box is kept from one sync to the next while the panel has the same field, so
    # typing isn't interrupted. Its text and cursor follow the field when the module changes them
    def sync_field(self):
        field = self.panel.text_field
        self.field_row.setVisible(field is not None)
        self.field_box.field = field
        if field is None:
            self.shown_field = None
            return
        self.field_label.setText(field.label)
        if self.field_box.text() != field.text:
            self.field_box.setText(field.text)
        if self.field_box.cursorPosition() != field.cursor:
            self.field_box.setCursorPosition(field.cursor)
        if field is not self.shown_field or field.focused:
            if field.focused and not self.field_box.hasFocus():
                self.field_box.setFocus()
        self.shown_field = field

    def sync_options(self):
        panel = self.panel
        titled = bool(panel.options_title)
        self.options_heading.setVisible(titled)
        self.options_title.setText(panel.options_title or "")
        self.options_toggle.setText("▲" if PanelView.options_expanded else "▼")
        shown = PanelView.options_expanded if titled else bool(panel.options)
        self.options.setVisible(shown)
        if not shown:
            return

        scroll = self.options.verticalScrollBar().value()
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 6, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        columns = max(1, panel.option_columns)
        for index, (label, buttons) in enumerate(panel.options):
            cell = QHBoxLayout()
            cell.setSpacing(4)
            for button in buttons:
                widget = small_button(button.text, "option")
                widget.setCheckable(True)
                widget.setChecked(bool(button.selected))
                widget.clicked.connect(lambda checked=False, button=button: self.clicked(button))
                cell.addWidget(widget)
            cell.addWidget(ElidedLabel(label), 1)
            grid.addLayout(cell, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        grid.setRowStretch(len(panel.options) // columns + 1, 1)
        self.options.setWidget(holder)

        rows = (len(panel.options) + columns - 1) // columns
        visible_rows = panel.max_option_rows if titled else min(rows, panel.max_option_rows)
        self.options.setFixedHeight(max(1, visible_rows) * OPTION_HEIGHT + 4)
        QTimer.singleShot(0, lambda: self.options.verticalScrollBar().setValue(scroll))

    # Lists what the module takes in on each input port and gives out on each output port,
    # from its INPUT_SPECS and OUTPUT_SPECS: one (name, description) per port, in port order
    def sync_ports(self):
        node = self.panel.node
        module = getattr(node, "module", None)
        ports = [(f"In {i + 1}", spec) for i, spec in enumerate(getattr(module, "INPUT_SPECS", []))]
        if getattr(node, "growing", False):
            ports = [("In 1+", spec) for label, spec in ports]
        ports += [(f"Out {i + 1}", spec) for i, spec in enumerate(getattr(module, "OUTPUT_SPECS", []))]

        self.ports_heading.setVisible(bool(ports))
        self.ports_toggle.setText("▲" if PanelView.ports_expanded else "▼")
        self.ports.setVisible(bool(ports) and PanelView.ports_expanded)
        rows = "".join(
            f"<tr><td style='padding-right:10px'><b>{label}</b></td><td>{name}: {description}</td></tr>"
            for label, (name, description) in ports
        )
        self.ports.setText(f"<table>{rows}</table>")

    # The table keeps its place and its sorting while it is showing the same data
    def sync_table(self):
        preview = self.panel.preview
        rows, columns = preview.shape
        self.size_label.setText(f"{rows:,} rows, {columns} columns")
        if preview is self.shown_preview:
            return
        self.shown_preview = preview
        self.table.setModel(TableModel(preview))
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.horizontalHeader().setResizeContentsPrecision(30)
        self.table.resizeColumnsToContents()
        for column in range(columns):
            width = self.table.columnWidth(column)
            self.table.setColumnWidth(column, max(60, min(width + 8, 260)))


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())
