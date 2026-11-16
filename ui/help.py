from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from engine import descriptions


POPUP_WIDTH = 460
NO_DESCRIPTION = "No description has been written for this module yet."


class HelpPopup(QFrame):
    # A box showing a module's description from its descriptions.txt, opened by clicking its ?.
    # It sits beside the point clicked, kept on screen, and any click or Escape closes it
    def __init__(self, name, position, parent):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("help")
        self.setFixedWidth(POPUP_WIDTH)
        rows = QVBoxLayout(self)
        rows.setContentsMargins(14, 12, 14, 14)
        rows.setSpacing(8)

        title = QLabel(name)
        title.setObjectName("heading")
        rows.addWidget(title)
        for paragraph in descriptions.describe(name) or [NO_DESCRIPTION]:
            text = QLabel(paragraph)
            text.setWordWrap(True)
            rows.addWidget(text)

        self.adjustSize()
        screen = QGuiApplication.screenAt(position) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(position.x() + 12, area.right() - self.width())
        y = min(position.y() + 12, area.bottom() - self.height())
        self.move(QPoint(max(area.left(), x), max(area.top(), y)))

    def mousePressEvent(self, event):
        self.close()
