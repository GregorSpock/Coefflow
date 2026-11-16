# Kept so add-on packs written before the modules were separated from the window still load:
# this used to be where they imported from. New code should import from engine.chart
from engine.chart import *  # noqa: F401,F403
from engine.chart import STYLE_KEY, colour, colour_in, series_colour  # noqa: F401
