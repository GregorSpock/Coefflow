import pygame

from engine import addons
from ui import colours, display, help
from ui.button import Button
from ui.panel import draw_scrollbar


ROW_HEIGHT = 50
TAB_WIDTH = 140
# How far one notch of the mouse wheel scrolls a row that doesn't fit, and how tall its scrollbar is
SCROLL_STEP = 70
SCROLLBAR_HEIGHT = 3

# Each tab and the core modules it shows in the row below. Add-on packs add to these (see engine/addons.py)
MODULES = {
    "File": ["New", "Open", "Save", "Save As"],
    "Input/Output": ["Input Data", "Output Data"],
    "Clean": ["Join", "Split", "Filter", "Select", "Sort", "Convert", "Gen"],
    "Base Reg.": ["OLS", "Auto OLS", "Logit", "Probit"],
    "Time Series": ["Durbin-Watson", "Forecast", "ARIMAX", "Auto ARIMAX", "ACF/PACF", "ADF"],
    "Cross Sect.": ["Fixed Effects", "Random Effects", "Hausman"],
    "Graph": ["Scatter", "Bar Chart", "Line Chart"],
}

# Toolbar buttons that do something when clicked, rather than being dragged onto the canvas
ACTIONS = ["New", "Open", "Save", "Save As"]


# The core tabs with the added packs' modules joined on: to the end of a tab that already
# exists, or as new tabs after the core ones. A name already in a tab isn't shown twice
def all_tabs():
    tabs = {tab: list(names) for tab, names in MODULES.items()}
    for tab, names in addons.added_tabs().items():
        shown = tabs.setdefault(tab, [])
        for name in names:
            if name not in shown:
                shown.append(name)
    return tabs


class Toolbar:
    # Two rows: category tabs on top, the selected tab's modules underneath.
    # tab_room is how much of the top row the tabs can use, as the Settings, Run and Exit buttons
    # sit at its right end; the app sets it when it places them. A row with more buttons than fit
    # scrolls sideways with the mouse wheel, tab_scroll and module_scroll being how far in pixels
    def __init__(self, width, font):
        self.width = width
        self.tab_room = width
        self.font = font
        self.height = ROW_HEIGHT * 2
        self.selected_tab = None
        self.module_buttons = []
        self.tabs = {}
        self.tab_buttons = []
        self.tab_scroll = 0
        self.module_scroll = 0
        self.refresh()

    # Remakes the tab buttons from the core modules and the added packs, closing any open tab.
    # Called again whenever a pack is added or removed
    def refresh(self):
        self.select_tab(None)
        self.tabs = all_tabs()
        self.tab_buttons = [
            Button(0, 4, TAB_WIDTH - 8, ROW_HEIGHT - 8, name, self.font, "TAB", "TAB_HOVER", "TAB_SELECTED")
            for name in self.tabs
        ]
        self.tab_scroll = 0
        self.place_buttons()

    # The part of each row its buttons are shown in
    def tab_area(self):
        return pygame.Rect(0, 0, self.tab_room, ROW_HEIGHT)

    def module_area(self):
        return pygame.Rect(0, ROW_HEIGHT, self.width, ROW_HEIGHT)

    # How far a row of buttons can scroll before its last button reaches the right of its area
    @staticmethod
    def max_scroll(buttons, area):
        return max(0, len(buttons) * TAB_WIDTH - area.width)

    # Lines each row's buttons up from the left, moved along by how far the row is scrolled.
    # Scrolling is kept within each row's ends, as the window or the rows may have changed size
    def place_buttons(self):
        self.tab_scroll = min(self.tab_scroll, self.max_scroll(self.tab_buttons, self.tab_area()))
        self.module_scroll = min(self.module_scroll, self.max_scroll(self.module_buttons, self.module_area()))
        for buttons, scroll in [(self.tab_buttons, self.tab_scroll), (self.module_buttons, self.module_scroll)]:
            for i, button in enumerate(buttons):
                button.rect.x = i * TAB_WIDTH + 4 - scroll

    # True if the event is a click on the part of a button shown in its row, not one scrolled out of view
    def shown_click(self, button, event):
        area = self.tab_area() if button in self.tab_buttons else self.module_area()
        return button.is_clicked(event) and area.collidepoint(event.pos)

    # The mouse wheel over a row scrolls it sideways; a sideways swipe works too.
    # Otherwise opens the clicked tab, or closes it if it was already open
    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            step = (event.x - event.y) * SCROLL_STEP
            position = display.mouse_pos()
            if self.tab_area().collidepoint(position):
                self.tab_scroll = max(0, self.tab_scroll + step)
            elif self.module_area().collidepoint(position):
                self.module_scroll = max(0, self.module_scroll + step)
            self.place_buttons()
            return

        for tab in self.tab_buttons:
            if self.shown_click(tab, event):
                if tab == self.selected_tab:
                    self.select_tab(None)
                else:
                    self.select_tab(tab)

    # Highlights the tab and fills the second row with its modules
    def select_tab(self, tab):
        if self.selected_tab:
            self.selected_tab.selected = False

        self.selected_tab = tab
        self.module_buttons = []
        if tab is None:
            return

        tab.selected = True
        self.module_buttons = [
            Button(0, ROW_HEIGHT + 4, TAB_WIDTH - 8, ROW_HEIGHT - 8, name, self.font, "TAB", "TAB_HOVER")
            for name in self.tabs[tab.text]
        ]
        self.module_scroll = 0
        self.place_buttons()

    # Returns the name of the module clicked on, if any. A click on its ? help badge doesn't count
    def module_clicked(self, event):
        for button in self.module_buttons:
            if self.shown_click(button, event) and button.text not in ACTIONS and not self.on_badge(button, event):
                return button.text
        return None

    # Returns the name of the module whose ? help badge was clicked, if any
    def help_clicked(self, event):
        for button in self.module_buttons:
            if self.shown_click(button, event) and button.text not in ACTIONS and self.on_badge(button, event):
                return button.text
        return None

    def on_badge(self, button, event):
        return help.badge_rect(button.rect).collidepoint(event.pos)

    # Returns the name of the action button clicked on, if any
    def action_clicked(self, event):
        for button in self.module_buttons:
            if self.shown_click(button, event) and button.text in ACTIONS:
                return button.text
        return None

    # Draws the toolbar background, the line between rows, then the buttons
    def draw(self, screen):
        pygame.draw.rect(screen, colours.TOOLBAR, (0, 0, self.width, self.height))
        pygame.draw.line(screen, colours.TOOLBAR_LINE, (0, ROW_HEIGHT), (self.width, ROW_HEIGHT))
        pygame.draw.line(screen, colours.TOOLBAR_LINE, (0, self.height - 1), (self.width, self.height - 1))

        self.place_buttons()
        self.draw_row(screen, self.tab_buttons, self.tab_area(), self.tab_scroll)
        self.draw_row(screen, self.module_buttons, self.module_area(), self.module_scroll)

    # Draws a row's buttons inside its area, cutting off any scrolled partly out of view.
    # If they don't all fit, a scrollbar along the bottom of the row shows which part is in view
    def draw_row(self, screen, buttons, area, scroll):
        clip = screen.get_clip()
        screen.set_clip(area)
        for button in buttons:
            button.draw(screen)
            if button.text not in ACTIONS and button not in self.tab_buttons:
                help.draw_badge(screen, button.rect)
        screen.set_clip(clip)

        total = len(buttons) * TAB_WIDTH
        if total > area.width:
            track = pygame.Rect(area.left + 4, area.bottom - SCROLLBAR_HEIGHT - 1, area.width - 8, SCROLLBAR_HEIGHT)
            draw_scrollbar(screen, track, scroll, area.width, total)
