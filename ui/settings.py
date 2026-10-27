import pygame

from engine import addons
from ui import colours
from ui.button import Button


ITEM_WIDTH = 200
ADDON_WIDTH = 320
ITEM_HEIGHT = 40
PADDING = 4


class SettingsMenu:
    # A drop-down list of settings that opens under the Settings button.
    # Add-ons swaps the list for one row per loaded add-on pack, each removing its pack, under a
    # row for the core modules, which are always on, and above a row to add a pack
    def __init__(self, font):
        self.font = font
        self.open = False
        self.rect = pygame.Rect(0, 0, ITEM_WIDTH + PADDING * 2, 0)
        self.items = []
        self.choices = []
        self.right = 0
        self.top = 0
        self.fullscreen = True
        self.pack = None

    # Opens the menu with its top right corner at (right, top), or closes it if already open
    def toggle(self, right, top, fullscreen):
        if self.open:
            self.open = False
            return

        self.right = right
        self.top = top
        self.fullscreen = fullscreen
        self.show_main()
        self.open = True

    # The first page: theme, full screen, Configure and the way into Add-ons
    def show_main(self):
        rows = [
            (f"Theme: {colours.current_theme}", "Theme", False),
            ("Windowed" if self.fullscreen else "Fullscreen", "Fullscreen", False),
            ("Configure", "Configure", False),
            ("Add-ons", "Open Add-ons", False),
        ]
        self.build(rows, ITEM_WIDTH)

    # The Add-ons page: the core modules, one row per loaded pack, then a row to add a pack.
    # A pack's row shows its version underneath, if it has one.
    # A pack's choice is its folder name, so a click can remove it
    def show_addons(self):
        rows = [("Core modules: Always on", None, True)]
        for pack, settings in addons.packs.items():
            label = f"{settings['name']}: Remove"
            if settings["version"]:
                label += f"\nv{settings['version']}"
            rows.append((label, pack, True))
        rows.append(("Add pack...", "Add pack", False))
        rows.append(("Back", "Back", False))
        self.build(rows, ADDON_WIDTH)

    # Makes one button per row of (label, choice, selected), stacked under the Settings button.
    # A row is a line taller for each extra line in its label
    def build(self, rows, width):
        heights = [ITEM_HEIGHT + label.count("\n") * self.font.get_linesize() for label, _, _ in rows]
        self.rect.width = width + PADDING * 2
        self.rect.height = sum(heights) + len(rows) * PADDING + PADDING
        self.rect.topright = (self.right, self.top)

        self.items = []
        self.choices = []
        y = self.rect.top + PADDING
        for (label, choice, selected), height in zip(rows, heights):
            item = Button(
                self.rect.left + PADDING, y, width, height, label, self.font,
                "TAB", "TAB_HOVER", "TAB_SELECTED"
            )
            item.selected = selected
            self.items.append(item)
            self.choices.append(choice)
            y += height + PADDING

    # Returns "Theme", "Fullscreen" or "Configure" if one was clicked, "Add pack" for the add row, or
    # "Remove pack" for a pack's row, with the pack's folder name left in self.pack.
    # Moving between pages and the Add-ons choices keep the menu open.
    # A click anywhere else closes the menu. The second value is True if the menu used the event
    def handle_event(self, event):
        if not self.open or event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None, False

        for item, choice in zip(self.items, self.choices):
            if not item.is_clicked(event):
                continue
            if choice == "Open Add-ons":
                self.show_addons()
                return None, True
            if choice == "Back":
                self.show_main()
                return None, True
            if choice is None:
                return None, True
            if choice in ["Theme", "Fullscreen", "Configure"]:
                self.open = False
                return choice, True
            if choice == "Add pack":
                return choice, True
            self.pack = choice
            return "Remove pack", True

        self.open = False
        return None, self.rect.collidepoint(event.pos)

    def draw(self, screen):
        if not self.open:
            return
        pygame.draw.rect(screen, colours.TOOLBAR, self.rect, border_radius=colours.BUTTON_RADIUS)
        pygame.draw.rect(screen, colours.TOOLBAR_LINE, self.rect, 1, border_radius=colours.BUTTON_RADIUS)
        for item in self.items:
            item.draw(screen)
