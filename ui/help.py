import pygame

from engine import descriptions
from ui import colours
from ui.panel import wrap_text


BADGE_RADIUS = 8
BADGE_MARGIN = 4
POPUP_WIDTH = 460
PADDING = 12
LINE_HEIGHT = 24
PARAGRAPH_GAP = 8
NO_DESCRIPTION = "No description has been written for this module yet."

# Fonts for the ? at each size, made once when first needed
badge_fonts = {}


# The circle for the ? in the top right corner of a module's rectangle, scaled by zoom
def badge_rect(rect, zoom=1):
    radius = max(5, BADGE_RADIUS * zoom)
    margin = BADGE_MARGIN * zoom
    badge = pygame.Rect(0, 0, radius * 2, radius * 2)
    badge.topright = (rect.right - margin, rect.top + margin)
    return badge


# Draws the ? in the top right corner of a module's rectangle
def draw_badge(screen, rect, zoom=1):
    badge = badge_rect(rect, zoom)
    size = max(10, round(BADGE_RADIUS * 2.2 * zoom))
    if size not in badge_fonts:
        badge_fonts[size] = pygame.font.Font(None, size)

    pygame.draw.circle(screen, colours.TOOLBAR, badge.center, badge.width / 2)
    pygame.draw.circle(screen, colours.TEXT, badge.center, badge.width / 2, width=1)
    label = badge_fonts[size].render("?", True, colours.TEXT)
    screen.blit(label, label.get_rect(center=badge.center))


class HelpPopup:
    # A box showing a module's description from its descriptions.txt, opened by clicking its ?.
    # It sits beside the point clicked, kept on screen, and any click or Escape closes it
    def __init__(self, name, pos, screen_size, font):
        self.name = name
        self.font = font
        self.title_font = pygame.font.Font(None, 32)
        paragraphs = descriptions.describe(name) or [NO_DESCRIPTION]

        self.lines = []
        for paragraph in paragraphs:
            if self.lines:
                self.lines.append(None)
            self.lines += wrap_text(font, paragraph, POPUP_WIDTH - PADDING * 2)

        text_height = sum(PARAGRAPH_GAP if line is None else LINE_HEIGHT for line in self.lines)
        height = PADDING * 3 + self.title_font.get_height() + text_height
        self.rect = pygame.Rect(pos[0] + PADDING, pos[1] + PADDING, POPUP_WIDTH, height)
        self.rect.clamp_ip(pygame.Rect((0, 0), screen_size))

    # Returns True if the event closes the popup
    def closes_on(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            return True
        return event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE

    def draw(self, screen):
        pygame.draw.rect(screen, colours.TOOLBAR, self.rect, border_radius=colours.BUTTON_RADIUS)
        pygame.draw.rect(screen, colours.TOOLBAR_LINE, self.rect, width=2, border_radius=colours.BUTTON_RADIUS)

        x = self.rect.left + PADDING
        y = self.rect.top + PADDING
        screen.blit(self.title_font.render(self.name, True, colours.TEXT), (x, y))
        y += self.title_font.get_height() + PADDING

        for line in self.lines:
            if line is None:
                y += PARAGRAPH_GAP
                continue
            screen.blit(self.font.render(line, True, colours.TEXT), (x, y))
            y += LINE_HEIGHT
