import os
import sys
from pathlib import Path

# Qt draws to nothing rather than opening windows, so the tests run anywhere, with or without a screen
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest  # noqa: E402


# No test is ever left waiting on a file chooser or message box: the questions a module can ask
# (engine/dialogs.py) go unanswered, as if the user had cancelled
@pytest.fixture(autouse=True)
def no_dialogs():
    from engine import dialogs
    dialogs.answers.clear()
    yield
    dialogs.answers.clear()
