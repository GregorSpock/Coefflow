import pygame

from ui import colours, display


class Button:
    # A clickable rectangle with a text label, which can be marked as selected.
    # A label with "\n" in it is drawn as lines stacked in the middle of the button.
    # Colours are given by name (e.g. "TAB") so they follow the current theme
    def __init__(self, x, y, width, height, text, font, colour, hover_colour, selected_colour=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.colour = colour
        self.hover_colour = hover_colour
        self.selected_colour = selected_colour
        self.selected = False

    # Draws the button: highlighted if selected, lighter when the mouse is over it
    def draw(self, screen):
        mouse_pos = display.mouse_pos()
        if self.selected and self.selected_colour:
            colour = self.selected_colour
        elif self.rect.collidepoint(mouse_pos):
            colour = self.hover_colour
        else:
            colour = self.colour

        pygame.draw.rect(screen, getattr(colours, colour), self.rect, border_radius=colours.BUTTON_RADIUS)

        lines = self.text.split("\n")
        line_height = self.font.get_linesize()
        y = self.rect.centery - line_height * (len(lines) - 1) / 2
        for line in lines:
            label = self.font.render(line, True, colours.TEXT)
            screen.blit(label, label.get_rect(center=(self.rect.centerx, y)))
            y += line_height

    # Returns True if this event is a left click on the button
    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False
