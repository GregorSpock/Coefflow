# Each theme gives a value for every name in ui/colours.py

# Green-screen terminal look with square corners
RETRO = {
    "BACKGROUND": (8, 12, 8),
    "GRID": (20, 45, 20),
    "TEXT": (80, 255, 120),

    "TOOLBAR": (10, 20, 10),
    "TOOLBAR_LINE": (40, 120, 50),
    "TAB": (15, 40, 18),
    "TAB_HOVER": (25, 70, 30),
    "TAB_SELECTED": (30, 100, 40),

    "EXIT": (120, 30, 20),
    "EXIT_HOVER": (160, 45, 30),
    "RUN": (20, 90, 35),
    "RUN_HOVER": (30, 120, 45),

    "ZOOM_BUTTON": (15, 40, 18),
    "ZOOM_BUTTON_HOVER": (25, 70, 30),

    "NODE": (10, 25, 12),
    "NODE_BORDER": (80, 255, 120),
    "PORT": (8, 12, 8),
    "NODE_DONE": (255, 200, 60),
    "NODE_ERROR": (255, 70, 50),
    "CONNECTION": (80, 255, 120),
    "CHART_POINT": (80, 255, 120),
    "CHART_LINE": (255, 200, 60),
    "CHART_SERIES": [(80, 255, 120), (255, 200, 60), (90, 200, 255), (255, 90, 70), (230, 130, 255), (240, 240, 240)],

    "BUTTON_RADIUS": 0,
    "NODE_RADIUS": 0,

    "TITLE": "> COEFFLOW",
    "TITLE_FONT": "consolas,couriernew,monospace",
    "MENU_FONT": "consolas,couriernew,monospace",
    "SUBTITLE": (40, 140, 60),
    "SCANLINES": True,
    "SCANLINE": (0, 0, 0),
    "TITLE_CURSOR": True,
}

# Classic dark: charcoal greys with blue accents
DARK = {
    "BACKGROUND": (32, 33, 36),
    "GRID": (44, 46, 50),
    "TEXT": (232, 234, 237),

    "TOOLBAR": (41, 42, 46),
    "TOOLBAR_LINE": (70, 72, 78),
    "TAB": (54, 56, 61),
    "TAB_HOVER": (72, 75, 82),
    "TAB_SELECTED": (50, 110, 210),

    "EXIT": (176, 52, 52),
    "EXIT_HOVER": (204, 68, 68),
    "RUN": (46, 125, 70),
    "RUN_HOVER": (60, 150, 88),

    "ZOOM_BUTTON": (54, 56, 61),
    "ZOOM_BUTTON_HOVER": (72, 75, 82),

    "NODE": (48, 50, 55),
    "NODE_BORDER": (138, 180, 248),
    "PORT": (232, 234, 237),
    "NODE_DONE": (129, 201, 149),
    "NODE_ERROR": (242, 139, 130),
    "CONNECTION": (154, 160, 166),
    "CHART_POINT": (138, 180, 248),
    "CHART_LINE": (253, 214, 99),
    "CHART_SERIES": [(138, 180, 248), (253, 214, 99), (129, 201, 149), (242, 139, 130), (197, 138, 249), (232, 234, 237)],

    "BUTTON_RADIUS": 4,
    "NODE_RADIUS": 6,

    "TITLE": "Coefflow",
    "TITLE_FONT": "segoeui,arial",
    "MENU_FONT": "segoeui,arial",
    "SUBTITLE": (154, 160, 166),
    "SCANLINES": False,
    "SCANLINE": (0, 0, 0),
    "TITLE_CURSOR": False,
}

# Classic light: white panels on pale grey with blue accents
LIGHT = {
    "BACKGROUND": (248, 249, 250),
    "GRID": (228, 231, 235),
    "TEXT": (32, 33, 36),

    "TOOLBAR": (255, 255, 255),
    "TOOLBAR_LINE": (200, 205, 212),
    "TAB": (236, 238, 241),
    "TAB_HOVER": (220, 224, 230),
    "TAB_SELECTED": (160, 195, 250),

    "EXIT": (240, 150, 140),
    "EXIT_HOVER": (245, 175, 165),
    "RUN": (140, 205, 155),
    "RUN_HOVER": (165, 220, 178),

    "ZOOM_BUTTON": (236, 238, 241),
    "ZOOM_BUTTON_HOVER": (220, 224, 230),

    "NODE": (255, 255, 255),
    "NODE_BORDER": (26, 115, 232),
    "PORT": (255, 255, 255),
    "NODE_DONE": (30, 142, 62),
    "NODE_ERROR": (217, 48, 37),
    "CONNECTION": (95, 99, 104),
    "CHART_POINT": (26, 115, 232),
    "CHART_LINE": (227, 116, 0),
    "CHART_SERIES": [(26, 115, 232), (227, 116, 0), (30, 142, 62), (217, 48, 37), (132, 48, 206), (32, 33, 36)],

    "BUTTON_RADIUS": 4,
    "NODE_RADIUS": 6,

    "TITLE": "Coefflow",
    "TITLE_FONT": "segoeui,arial",
    "MENU_FONT": "segoeui,arial",
    "SUBTITLE": (95, 99, 104),
    "SCANLINES": False,
    "SCANLINE": (0, 0, 0),
    "TITLE_CURSOR": False,
}

THEMES = {"Retro": RETRO, "Dark": DARK, "Light": LIGHT}
