from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QPushButton, QWidget

from engine import colours, version
from ui import theme


BUTTON_WIDTH = 220
BUTTON_HEIGHT = 60
BUTTON_GAP = 20
TOGGLE_WIDTH = 320
THEME_WIDTH = 200
THEME_HEIGHT = 40
BAR_WIDTH = 500
BAR_HEIGHT = 24
GRID_SIZE = 40
SCANLINE_GAP = 3
SCANLINE_ALPHA = 90
SUBTITLE = "Econometric workflows"


class Scanlines(QWidget):
    # Dark horizontal lines laid over the whole menu for the retro look. Clicks pass through it
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, event):
        if not colours.SCANLINES:
            return
        painter = QPainter(self)
        painter.setPen(QColor(0, 0, 0, SCANLINE_ALPHA))
        for y in range(0, self.height(), SCANLINE_GAP):
            painter.drawLine(0, y, self.width(), y)


class StartMenu(QWidget):
    # The opening screen, with the title above a Start and an Exit button, a toggle for
    # loading every module before the app opens, and a smaller button below it to change theme.
    # started says whether every module should be loaded first
    started = Signal(bool)
    exit_asked = Signal()
    theme_asked = Signal()

    def __init__(self):
        super().__init__()
        self.loading = None
        self.start_button = QPushButton("Start", self)
        self.start_button.setObjectName("run")
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setObjectName("exit")
        self.load_toggle = QPushButton(self)
        self.load_toggle.setCheckable(True)
        self.load_toggle.setChecked(True)
        self.theme_button = QPushButton(self)
        self.buttons = [self.start_button, self.exit_button, self.load_toggle, self.theme_button]

        self.start_button.clicked.connect(lambda: self.started.emit(self.load_toggle.isChecked()))
        self.exit_button.clicked.connect(self.exit_asked.emit)
        self.load_toggle.clicked.connect(self.apply_theme)
        self.theme_button.clicked.connect(self.theme_asked.emit)

        self.scanlines = Scanlines(self)
        self.blink = QTimer(self)
        self.blink.timeout.connect(self.tick)
        self.blink.start(500)
        self.cursor_on = True
        self.apply_theme()

    # Retro blinks a block cursor after the title, on for half a second, off for half
    def tick(self):
        self.cursor_on = not self.cursor_on
        self.update()

    # Sets the labels and fonts for the current theme
    def apply_theme(self):
        self.load_toggle.setText(f"Load all modules: {'On' if self.load_toggle.isChecked() else 'Off'}")
        self.theme_button.setText(f"Theme: {colours.current_theme}")
        for button in [self.start_button, self.exit_button, self.load_toggle]:
            button.setFont(theme.font(20, bold=True))
            button.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.theme_button.setStyleSheet("font-size: 15px;")
        self.update()

    def resizeEvent(self, event):
        width, height = self.width(), self.height()
        x = width // 2 - BUTTON_WIDTH // 2
        self.start_button.setGeometry(x, height // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.exit_button.setGeometry(x, height // 2 + BUTTON_HEIGHT + BUTTON_GAP, BUTTON_WIDTH, BUTTON_HEIGHT)
        theme_top = height - THEME_HEIGHT - 40
        self.load_toggle.setGeometry(width // 2 - TOGGLE_WIDTH // 2, theme_top - BUTTON_HEIGHT - BUTTON_GAP, TOGGLE_WIDTH, BUTTON_HEIGHT)
        self.theme_button.setGeometry(width // 2 - THEME_WIDTH // 2, theme_top, THEME_WIDTH, THEME_HEIGHT)
        self.scanlines.setGeometry(self.rect())
        self.scanlines.raise_()

    # Swaps the buttons for the loading screen: the module being loaded and a progress bar.
    # None goes back to the buttons
    def show_loading(self, name=None, count=0, total=1):
        self.loading = None if name is None else (name, count, total)
        for button in self.buttons:
            button.setVisible(self.loading is None)
        self.repaint()

    # Draws the background and the title with its subtitle, both styled by the current theme
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), theme.colour("BACKGROUND"))

        if not colours.SCANLINES:
            painter.setPen(theme.colour("GRID"))
            for x in range(0, width, GRID_SIZE):
                painter.drawLine(x, 0, x, height)
            for y in range(0, height, GRID_SIZE):
                painter.drawLine(0, y, width, y)

        title_font = theme.font(84, bold=True)
        metrics = QFontMetrics(title_font)
        title_width = metrics.horizontalAdvance(colours.TITLE)
        title = QRect(width // 2 - title_width // 2, height // 2 - 90 - metrics.height(), title_width, metrics.height())
        painter.setFont(title_font)
        painter.setPen(theme.colour("TEXT"))
        painter.drawText(title, Qt.AlignmentFlag.AlignCenter, colours.TITLE)

        if colours.TITLE_CURSOR and self.cursor_on:
            painter.fillRect(title.right() + 14, title.top() + 18, 40, title.height() - 34, theme.colour("TEXT"))

        painter.setFont(theme.font(24))
        painter.setPen(theme.colour("SUBTITLE"))
        painter.drawText(QRect(0, title.bottom() + 5, width, 36), Qt.AlignmentFlag.AlignHCenter, SUBTITLE)

        number = version.read()
        if number:
            painter.setFont(theme.font(16))
            painter.drawText(20, height - 20, f"v{number}")

        if self.loading:
            name, count, total = self.loading
            painter.setFont(theme.font(20, bold=True))
            painter.setPen(theme.colour("TEXT"))
            painter.drawText(QRect(0, height // 2, width, 36), Qt.AlignmentFlag.AlignHCenter, f"Loading {name}... ({count + 1} of {total})")
            bar = QRect(width // 2 - BAR_WIDTH // 2, height // 2 + 50, BAR_WIDTH, BAR_HEIGHT)
            painter.fillRect(bar, theme.colour("TAB"))
            painter.fillRect(bar.left(), bar.top(), int(BAR_WIDTH * count / total), BAR_HEIGHT, theme.colour("TAB_SELECTED"))
