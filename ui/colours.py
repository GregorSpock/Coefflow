# Kept so add-on packs written before the modules were separated from the window still load.
# New code should import engine.colours. The module itself is handed over, rather than its
# names copied, so the colours read through here follow the theme as it changes
import sys

from engine import colours

sys.modules[__name__] = colours
