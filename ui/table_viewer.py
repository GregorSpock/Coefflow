from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSplitter, QTableView, QVBoxLayout, QWidget,
)

from engine import file_formats, table_view
from ui.panel_view import TableModel


# A window for looking through every table a module was given or gave back: all the rows, sorted
# by clicking a heading, narrowed by a search and a condition, with a summary of every column and
# the most common values of the one chosen. Any number can be open at once, and each stays open
# while the workflow is edited, showing what the module held when it was opened or last refreshed.
# What it shows is worked out in engine/table_view.py

COLUMN_WIDTH = 110
# Searching waits this long after the last keystroke, so typing in a big table stays smooth
SEARCH_DELAY_MS = 250


class TableViewer(QDialog):
    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self.tables = []
        self.shown = None
        self.setWindowTitle(f"View data: {node.name} (#{node.id})")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1200, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        rows = QVBoxLayout(self)
        top = QHBoxLayout()
        self.picker = QComboBox()
        self.picker.currentIndexChanged.connect(self.update_view)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search every column")
        self.search.setClearButtonEnabled(True)
        self.condition = QLineEdit()
        self.condition.setPlaceholderText("Condition, e.g. year >= 2000 and region == 'London'")
        self.condition.setClearButtonEnabled(True)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(SEARCH_DELAY_MS)
        self.timer.timeout.connect(self.update_view)
        self.search.textChanged.connect(lambda text: self.timer.start())
        self.condition.returnPressed.connect(self.update_view)
        self.condition.textChanged.connect(lambda text: self.condition_error.setText("") if not text else None)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.load)
        copy = QPushButton("Copy")
        copy.setToolTip("Copy the rows shown, tab separated, to paste into a spreadsheet")
        copy.clicked.connect(self.copy_rows)
        save = QPushButton("Save rows...")
        save.clicked.connect(self.save_rows)
        top.addWidget(QLabel("Table:"))
        top.addWidget(self.picker, 2)
        top.addWidget(self.search, 2)
        top.addWidget(self.condition, 3)
        top.addWidget(refresh)
        top.addWidget(copy)
        top.addWidget(save)
        rows.addLayout(top)

        self.size_label = QLabel()
        self.size_label.setObjectName("hint")
        self.condition_error = QLabel()
        self.condition_error.setObjectName("note")
        info = QHBoxLayout()
        info.addWidget(self.size_label)
        info.addStretch()
        info.addWidget(self.condition_error)
        rows.addLayout(info)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = self.make_table(sortable=True)
        self.table.horizontalHeader().sectionClicked.connect(self.column_chosen)
        side = QWidget()
        side_rows = QVBoxLayout(side)
        side_rows.setContentsMargins(0, 0, 0, 0)
        side_rows.addWidget(QLabel("Columns (click one to see its values)"))
        self.summary = self.make_table(sortable=True)
        self.summary.clicked.connect(lambda index: self.show_frequencies(self.summary.model().frame.iat[index.row(), 0]))
        side_rows.addWidget(self.summary, 3)
        self.frequency_title = QLabel("Most common values")
        side_rows.addWidget(self.frequency_title)
        self.frequencies = self.make_table(sortable=True)
        side_rows.addWidget(self.frequencies, 2)
        splitter.addWidget(self.table)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        rows.addWidget(splitter, 1)
        self.load()

    def make_table(self, sortable):
        table = QTableView()
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(sortable)
        table.setWordWrap(False)
        table.horizontalHeader().setDefaultSectionSize(COLUMN_WIDTH)
        table.verticalHeader().setDefaultSectionSize(24)
        return table

    # Reads the node's tables again, keeping the one being looked at if it is still there
    def load(self):
        current = self.picker.currentText()
        self.tables = table_view.tables_of(self.node)
        self.picker.blockSignals(True)
        self.picker.clear()
        for label, table in self.tables:
            self.picker.addItem(f"{label} ({len(table):,} rows)", label)
        labels = [label for label, table in self.tables]
        wanted = current.split(" (")[0]
        self.picker.setCurrentIndex(labels.index(wanted) if wanted in labels else 0)
        self.picker.blockSignals(False)
        self.update_view()

    def current_table(self):
        index = self.picker.currentIndex()
        return self.tables[index][1] if 0 <= index < len(self.tables) else None

    def update_view(self):
        data = self.current_table()
        if data is None:
            self.size_label.setText("Run the workflow first: this module has no tables to show yet")
            self.table.setModel(None)
            self.summary.setModel(None)
            self.frequencies.setModel(None)
            return
        try:
            rows = table_view.narrow(data, self.search.text(), self.condition.text())
            self.condition_error.setText("")
        except Exception as error:
            rows = table_view.narrow(data, self.search.text())
            self.condition_error.setText(f"Condition not used: {error}")
        self.shown = rows
        self.size_label.setText(f"Showing {len(rows):,} of {len(data):,} rows, {len(data.columns)} columns")
        self.fill(self.table, rows)
        self.fill(self.summary, table_view.column_summary(rows))
        if len(rows.columns):
            self.show_frequencies(str(rows.columns[0]))

    def fill(self, view, frame):
        view.setModel(TableModel(frame))
        view.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        view.horizontalHeader().setResizeContentsPrecision(30)
        view.resizeColumnsToContents()
        for column in range(len(frame.columns)):
            view.setColumnWidth(column, max(60, min(view.columnWidth(column) + 8, 260)))

    def column_chosen(self, section):
        if self.shown is not None and 0 <= section < len(self.shown.columns):
            self.show_frequencies(str(self.shown.columns[section]))

    def show_frequencies(self, name):
        if self.shown is None:
            return
        columns = [c for c in self.shown.columns if str(c) == name]
        if not columns:
            return
        self.frequency_title.setText(f"Most common values of {name}")
        self.fill(self.frequencies, table_view.frequencies(self.shown[columns[0]]))

    def copy_rows(self):
        if self.shown is not None:
            QApplication.clipboard().setText(self.shown.to_csv(sep="\t", index=False))

    def save_rows(self):
        if self.shown is None:
            return
        kinds = {f"{name} (*{ending})": (kind, ending) for kind, (ending, name) in file_formats.WRITE_FORMATS.items()}
        path, chosen = QFileDialog.getSaveFileName(self, "Save the rows shown", "", ";;".join(kinds))
        if not path:
            return
        kind, ending = kinds.get(chosen, ("csv", ".csv"))
        if not path.lower().endswith(ending):
            path += ending
        try:
            file_formats.write_table(self.shown, path, kind, title=self.node.name)
        except Exception as error:
            self.condition_error.setText(f"Could not save: {error}")
