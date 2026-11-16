from PySide6.QtGui import QColor, QFont

from engine import colours


# Turns the current theme (engine/themes.py) into what Qt needs: colours, fonts and the style
# sheet that gives every button, box and table the theme's look. The app sets the sheet again
# whenever the theme changes


def colour(name):
    return QColor(*getattr(colours, name))


def css(name):
    return "rgb({}, {}, {})".format(*getattr(colours, name))


# The first font named in the theme's list, e.g. "consolas,couriernew,monospace" gives Consolas
def family():
    families = {"consolas": "Consolas", "couriernew": "Courier New", "segoeui": "Segoe UI", "arial": "Arial"}
    first = colours.MENU_FONT.split(",")[0]
    return families.get(first, first)


# A font in the theme's family, size pixels tall
def font(size, bold=False):
    made = QFont(family())
    made.setPixelSize(size)
    made.setBold(bold)
    if "mono" in colours.MENU_FONT:
        made.setStyleHint(QFont.StyleHint.Monospace)
    return made


def style_sheet():
    radius = colours.BUTTON_RADIUS
    return f"""
    QWidget {{ color: {css("TEXT")}; font-family: "{family()}"; font-size: 14px; }}
    QMainWindow, QDialog, QMessageBox {{ background: {css("BACKGROUND")}; }}
    QToolTip {{ background: {css("TOOLBAR")}; color: {css("TEXT")}; border: 1px solid {css("TOOLBAR_LINE")}; padding: 4px; }}
    #toolbar, #tabs, #panel, #status, #help {{ background: {css("TOOLBAR")}; }}
    #toolbar {{ border-bottom: 1px solid {css("TOOLBAR_LINE")}; }}
    #modules {{ background: {css("BACKGROUND")}; border-bottom: 1px solid {css("TOOLBAR_LINE")}; }}
    #modulerow {{ background: transparent; }}
    QLabel#rowlabel {{ color: {css("SUBTITLE")}; font-size: 12px; padding-right: 12px; border-right: 1px solid {css("TOOLBAR_LINE")}; }}
    QFrame#divider {{ background: {css("TOOLBAR_LINE")}; border: none; }}
    #panel {{ border: none; border-top: 1px solid {css("TOOLBAR_LINE")}; }}
    #panelhead {{ background: {css("TAB")}; border-bottom: 1px solid {css("TOOLBAR_LINE")}; }}
    QLabel#paneltitle {{ font-size: 17px; font-weight: bold; }}
    #help {{ border: 2px solid {css("TOOLBAR_LINE")}; border-radius: {radius}px; }}
    #status {{ border-top: 1px solid {css("TOOLBAR_LINE")}; }}
    QScrollArea, QScrollArea > QWidget > QWidget, #holder, QLabel, QStackedWidget {{ background: transparent; }}

    QPushButton {{
        background: {css("TAB")}; color: {css("TEXT")}; border: none;
        border-radius: {radius}px; padding: 6px 12px;
    }}
    QPushButton:hover {{ background: {css("TAB_HOVER")}; }}
    QPushButton:checked {{ background: {css("TAB_SELECTED")}; }}
    QPushButton#tab {{ border-bottom: 3px solid transparent; padding-top: 9px; }}
    QPushButton#tab:checked {{ background: {css("TAB_SELECTED")}; border-bottom: 3px solid {css("TEXT")}; }}
    QPushButton#module {{ background: {css("TOOLBAR")}; border: 1px solid {css("TOOLBAR_LINE")}; }}
    QPushButton#module:hover {{ background: {css("TAB_HOVER")}; }}
    QPushButton#close {{ background: transparent; border: 1px solid {css("TOOLBAR_LINE")}; padding: 0px; font-weight: bold; }}
    QPushButton#close:hover {{ background: {css("EXIT_HOVER")}; border-color: {css("EXIT_HOVER")}; }}
    QPushButton:disabled {{ color: {css("SUBTITLE")}; }}
    QPushButton#run {{ background: {css("RUN")}; }}
    QPushButton#run:hover {{ background: {css("RUN_HOVER")}; }}
    QPushButton#exit {{ background: {css("EXIT")}; }}
    QPushButton#exit:hover {{ background: {css("EXIT_HOVER")}; }}
    QPushButton#option {{ padding: 3px 8px; font-size: 13px; }}
    QPushButton#small {{ padding: 3px 10px; font-size: 13px; }}
    QPushButton#toggle {{ padding: 0px; background: {css("ZOOM_BUTTON")}; border: 1px solid {css("TOOLBAR_LINE")}; }}
    QPushButton#toggle:hover {{ background: {css("ZOOM_BUTTON_HOVER")}; }}
    QPushButton::menu-indicator {{ image: none; }}

    QLabel#heading {{ font-size: 18px; text-decoration: underline; }}
    QLabel#title {{ font-size: 15px; }}
    QLabel#error {{ color: {css("NODE_ERROR")}; }}
    QLabel#hint {{ color: {css("SUBTITLE")}; font-size: 12px; }}
    QLabel#small {{ font-size: 12px; }}
    QLabel#note {{ border-top: 1px solid {css("TOOLBAR_LINE")}; padding-top: 8px; }}

    QLineEdit {{
        background: {css("BACKGROUND")}; color: {css("TEXT")};
        border: 2px solid {css("TOOLBAR_LINE")}; border-radius: {radius}px; padding: 4px 8px;
        selection-background-color: {css("TAB_SELECTED")}; selection-color: {css("TEXT")};
    }}
    QLineEdit:focus {{ border-color: {css("TAB_SELECTED")}; }}
    QComboBox {{
        background: {css("BACKGROUND")}; color: {css("TEXT")};
        border: 2px solid {css("TOOLBAR_LINE")}; border-radius: {radius}px; padding: 4px 8px;
    }}
    QComboBox:focus, QComboBox:on {{ border-color: {css("TAB_SELECTED")}; }}
    QComboBox QAbstractItemView {{
        background: {css("TOOLBAR")}; color: {css("TEXT")}; border: 1px solid {css("TOOLBAR_LINE")};
        selection-background-color: {css("TAB_SELECTED")}; selection-color: {css("TEXT")};
    }}

    QTableView {{
        background: {css("TOOLBAR")}; alternate-background-color: {css("TAB")};
        gridline-color: {css("GRID")}; border: 1px solid {css("TOOLBAR_LINE")};
        selection-background-color: {css("TAB_SELECTED")}; selection-color: {css("TEXT")}; font-size: 13px;
    }}
    QHeaderView::section {{
        background: {css("TAB")}; color: {css("TEXT")}; border: none;
        border-right: 1px solid {css("TOOLBAR_LINE")}; border-bottom: 1px solid {css("TOOLBAR_LINE")};
        padding: 4px 6px; font-size: 13px;
    }}
    QHeaderView {{ background: {css("TOOLBAR")}; }}
    QTableCornerButton::section {{ background: {css("TAB")}; border: none; }}

    QScrollArea {{ border: none; }}
    QScrollBar:vertical {{ background: {css("TAB")}; width: 10px; margin: 0; }}
    QScrollBar:horizontal {{ background: {css("TAB")}; height: 10px; margin: 0; }}
    QScrollBar::handle {{ background: {css("TAB_SELECTED")}; border-radius: {min(radius, 4)}px; min-height: 20px; min-width: 20px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QMenu {{ background: {css("TOOLBAR")}; border: 1px solid {css("TOOLBAR_LINE")}; padding: 4px; }}
    QMenu::item {{ padding: 7px 28px 7px 16px; border-radius: {radius}px; }}
    QMenu::item:selected {{ background: {css("TAB_HOVER")}; }}
    QMenu::item:disabled {{ color: {css("SUBTITLE")}; }}
    QMenu::separator {{ height: 1px; background: {css("TOOLBAR_LINE")}; margin: 4px 6px; }}

    QTabBar {{ background: {css("TOOLBAR")}; }}
    QTabBar::tab {{
        background: {css("TAB")}; color: {css("TEXT")}; padding: 4px 10px; margin: 1px;
        border: none; font-size: 13px; max-width: 220px;
    }}
    QTabBar::tab:hover {{ background: {css("TAB_HOVER")}; }}
    QTabBar::tab:selected {{ background: {css("TAB_SELECTED")}; }}
    QTabBar::close-button {{ subcontrol-position: right; }}

    QSplitter::handle {{ background: {css("TOOLBAR_LINE")}; }}
    QGraphicsView {{ border: none; }}
    QProgressBar {{
        background: {css("TAB")}; border: none; border-radius: {radius}px; text-align: center; height: 20px;
    }}
    QProgressBar::chunk {{ background: {css("TAB_SELECTED")}; border-radius: {radius}px; }}
    """
