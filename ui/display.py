import pygame


# The app is always drawn at full screen size. In windowed mode that picture is shrunk to fit
# the window, so mouse positions are turned back into positions on the full size picture
windowed = False
scale = 1
offset_x = 0
offset_y = 0


# Turns a position in the window into a position on the full size picture
def to_screen(pos):
    return (int((pos[0] - offset_x) / scale), int((pos[1] - offset_y) / scale))


def mouse_pos():
    return to_screen(pygame.mouse.get_pos())


# Shows the finished picture: as it is in full screen, or scaled to fit the window
# (keeping its shape, with black bars filling any space left over)
def present(screen):
    global scale, offset_x, offset_y
    if windowed:
        window = pygame.display.get_surface()
        window_width, window_height = window.get_size()
        width, height = screen.get_size()
        scale = min(window_width / width, window_height / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        offset_x = (window_width - size[0]) // 2
        offset_y = (window_height - size[1]) // 2

        window.fill((0, 0, 0))
        window.blit(pygame.transform.smoothscale(screen, size), (offset_x, offset_y))
    else:
        scale = 1
        offset_x = offset_y = 0
    pygame.display.flip()
