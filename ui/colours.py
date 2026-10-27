from ui.themes import THEMES


current_theme = "RETRO"


# Copies a theme's values onto this file, so colours.TEXT etc. always match the current theme
def set_theme(name):
    global current_theme
    current_theme = name
    globals().update(THEMES[name])


# Moves on to the next theme in the list, going back to the first after the last
def next_theme():
    names = list(THEMES)
    index = names.index(current_theme)
    set_theme(names[(index + 1) % len(names)])


set_theme("Retro")
