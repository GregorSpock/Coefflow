import pygame

from ui import colours, dialogs
from ui.button import Button
from ui.node import load_module
from ui.toolbar import ACTIONS, all_tabs


FPS = 60
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


class Menu:
    # The opening screen, with the title above a Start and an Exit button, a toggle for
    # loading every module before the app opens, and a smaller button below it to change theme
    def __init__(self, screen, font):
        self.screen = screen
        self.width, self.height = screen.get_size()

        x = self.width // 2 - BUTTON_WIDTH // 2
        y = self.height // 2
        self.start_button = Button(
            x, y, BUTTON_WIDTH, BUTTON_HEIGHT, "Start", font, "RUN", "RUN_HOVER"
        )
        self.exit_button = Button(
            x, y + BUTTON_HEIGHT + BUTTON_GAP, BUTTON_WIDTH, BUTTON_HEIGHT, "Exit",
            font, "EXIT", "EXIT_HOVER"
        )

        theme_top = self.height - THEME_HEIGHT - 40
        self.load_toggle = Button(
            self.width // 2 - TOGGLE_WIDTH // 2, theme_top - BUTTON_HEIGHT - BUTTON_GAP,
            TOGGLE_WIDTH, BUTTON_HEIGHT, "", font, "TAB", "TAB_HOVER", "TAB_SELECTED"
        )
        self.theme_button = Button(
            self.width // 2 - THEME_WIDTH // 2, theme_top, THEME_WIDTH, THEME_HEIGHT, "",
            font, "TAB", "TAB_HOVER"
        )

        self.scanlines = self.make_scanlines()
        self.set_toggle(True)
        self.apply_theme()

    # Turns the load toggle on or off and updates its label to match
    def set_toggle(self, on):
        self.load_toggle.selected = on
        if on:
            self.load_toggle.text = "Load all modules: On"
        else:
            self.load_toggle.text = "Load all modules: Off"

    # Remakes the fonts for the current theme and gives them to the buttons
    def apply_theme(self):
        self.title_font = pygame.font.SysFont(colours.TITLE_FONT, 90, bold=True)
        self.subtitle_font = pygame.font.SysFont(colours.MENU_FONT, 28)
        self.font = pygame.font.SysFont(colours.MENU_FONT, 24, bold=True)
        self.small_font = pygame.font.SysFont(colours.MENU_FONT, 20)

        for button in [self.start_button, self.exit_button, self.load_toggle]:
            button.font = self.font
        self.theme_button.font = self.small_font
        self.theme_button.text = f"Theme: {colours.current_theme}"

    # A see-through layer of dark horizontal lines, laid over the screen for the retro look
    def make_scanlines(self):
        layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(0, self.height, SCANLINE_GAP):
            pygame.draw.line(layer, (0, 0, 0, SCANLINE_ALPHA), (0, y), (self.width, y))
        return layer

    # Shows the menu until a button is pressed. Returns True for Start and False for Exit
    def run(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or self.exit_button.is_clicked(event):
                    return False
                if self.load_toggle.is_clicked(event):
                    self.set_toggle(not self.load_toggle.selected)
                if self.theme_button.is_clicked(event):
                    colours.next_theme()
                    self.apply_theme()
                if self.start_button.is_clicked(event):
                    if self.load_toggle.selected:
                        self.load_all()
                    return True

            self.draw()
            clock.tick(FPS)

    # Loads every toolbar module in turn, showing which one is loading, then lists any that failed
    def load_all(self):
        names = []
        for tab in all_tabs().values():
            for name in tab:
                if name not in ACTIONS:
                    names.append(name)

        problems = []
        for count, name in enumerate(names):
            self.draw_loading(name, count, len(names))
            pygame.event.pump()
            try:
                load_module(name)
            except Exception as error:
                problems.append(f"{name} could not be loaded ({error}).")

        if problems:
            dialogs.show_message("Module problems", "\n".join(problems))

    # Draws the background and the title with its subtitle, both styled by the current theme
    def draw_backdrop(self):
        self.screen.fill(colours.BACKGROUND)

        if not colours.SCANLINES:
            for x in range(0, self.width, GRID_SIZE):
                pygame.draw.line(self.screen, colours.GRID, (x, 0), (x, self.height))
            for y in range(0, self.height, GRID_SIZE):
                pygame.draw.line(self.screen, colours.GRID, (0, y), (self.width, y))

        title = self.title_font.render(colours.TITLE, True, colours.TEXT)
        title_rect = title.get_rect(midbottom=(self.width // 2, self.height // 2 - 90))
        self.screen.blit(title, title_rect)

        # Retro blinks a block cursor after the title, on for half a second, off for half
        if colours.TITLE_CURSOR and pygame.time.get_ticks() // 500 % 2 == 0:
            cursor = pygame.Rect(title_rect.right + 10, title_rect.top + 20, 40, title_rect.height - 35)
            pygame.draw.rect(self.screen, colours.TEXT, cursor)

        subtitle = self.subtitle_font.render(SUBTITLE, True, colours.SUBTITLE)
        self.screen.blit(subtitle, subtitle.get_rect(midtop=(self.width // 2, title_rect.bottom + 5)))

    # Lays the scanlines over everything if the theme uses them, then shows the frame
    def finish_frame(self):
        if colours.SCANLINES:
            self.screen.blit(self.scanlines, (0, 0))
        pygame.display.flip()

    # Draws the loading screen: the module being loaded and a progress bar
    def draw_loading(self, name, count, total):
        self.draw_backdrop()

        label = self.font.render(f"Loading {name}... ({count + 1} of {total})", True, colours.TEXT)
        self.screen.blit(label, label.get_rect(midtop=(self.width // 2, self.height // 2)))

        bar = pygame.Rect(0, 0, BAR_WIDTH, BAR_HEIGHT)
        bar.midtop = (self.width // 2, self.height // 2 + 50)
        filled = bar.copy()
        filled.width = BAR_WIDTH * count / total

        pygame.draw.rect(self.screen, colours.TAB, bar, border_radius=colours.BUTTON_RADIUS)
        pygame.draw.rect(self.screen, colours.TAB_SELECTED, filled, border_radius=colours.BUTTON_RADIUS)

        self.finish_frame()

    # Draws the title, the two buttons in the middle of the screen, and the toggle and theme button at the bottom
    def draw(self):
        self.draw_backdrop()

        self.start_button.draw(self.screen)
        self.exit_button.draw(self.screen)
        self.load_toggle.draw(self.screen)
        self.theme_button.draw(self.screen)

        self.finish_frame()
