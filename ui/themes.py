# Each theme gives a value for every name in ui/colours.py

NORMAL = {
    "BACKGROUND": (30, 32, 36),
    "GRID": (45, 48, 54),
    "TEXT": (240, 240, 240),

    "TOOLBAR": (40, 43, 49),
    "TOOLBAR_LINE": (65, 69, 78),
    "TAB": (55, 59, 67),
    "TAB_HOVER": (75, 80, 90),
    "TAB_SELECTED": (70, 120, 200),

    "EXIT": (180, 50, 50),
    "EXIT_HOVER": (210, 70, 70),
    "RUN": (50, 140, 80),
    "RUN_HOVER": (70, 170, 100),

    "ZOOM_BUTTON": (60, 64, 72),
    "ZOOM_BUTTON_HOVER": (85, 90, 100),

    "NODE": (70, 120, 200),
    "NODE_BORDER": (150, 180, 230),
    "PORT": (230, 235, 245),
    "NODE_DONE": (90, 210, 120),
    "NODE_ERROR": (230, 80, 80),
    "CONNECTION": (200, 205, 215),
    "CHART_POINT": (110, 160, 230),
    "CHART_LINE": (240, 170, 60),
    "CHART_SERIES": [(110, 160, 230), (240, 170, 60), (90, 210, 120), (230, 80, 80), (190, 130, 230), (240, 240, 240)],

    "BUTTON_RADIUS": 4,
    "NODE_RADIUS": 6,

    "TITLE": "Coefflow",
    "TITLE_FONT": "segoeui,arial",
    "MENU_FONT": "segoeui,arial",
    "SUBTITLE": (130, 140, 160),
    "SCANLINES": False,
    "SCANLINE": (0, 0, 0),
    "TITLE_CURSOR": False,
}

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

# Early-2000s desktop: silver windows on a bright blue screen, with hot pink, orange and purple accents
Y2K = {
    "BACKGROUND": (60, 160, 240),
    "GRID": (95, 185, 250),
    "TEXT": (20, 25, 75),

    "TOOLBAR": (226, 230, 240),
    "TOOLBAR_LINE": (120, 130, 160),
    "TAB": (200, 206, 222),
    "TAB_HOVER": (175, 225, 250),
    "TAB_SELECTED": (255, 130, 200),

    "EXIT": (255, 140, 90),
    "EXIT_HOVER": (255, 170, 120),
    "RUN": (130, 225, 140),
    "RUN_HOVER": (165, 240, 170),

    "ZOOM_BUTTON": (226, 230, 240),
    "ZOOM_BUTTON_HOVER": (175, 225, 250),

    "NODE": (250, 250, 255),
    "NODE_BORDER": (235, 60, 170),
    "PORT": (255, 200, 90),
    "NODE_DONE": (60, 200, 110),
    "NODE_ERROR": (240, 90, 40),
    "CONNECTION": (130, 60, 210),
    "CHART_POINT": (30, 110, 230),
    "CHART_LINE": (235, 60, 170),
    "CHART_SERIES": [(20, 25, 75), (235, 60, 170), (255, 220, 60), (240, 90, 40), (130, 60, 210), (255, 255, 255)],

    "BUTTON_RADIUS": 2,
    "NODE_RADIUS": 2,

    "TITLE": "Coefflow.exe",
    "TITLE_FONT": "tahoma,verdana,arial",
    "MENU_FONT": "tahoma,verdana,arial",
    "SUBTITLE": (110, 40, 170),
    "SCANLINES": False,
    "SCANLINE": (0, 0, 0),
    "TITLE_CURSOR": False,
}

THEMES = {"Normal": NORMAL, "Retro": RETRO, "Y2K": Y2K}
