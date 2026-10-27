import json
from pathlib import Path

import pandas as pd


# The stars every regression module puts next to its coefficients: one star for each level the
# p value is below, so with the default levels p < 0.1 gets *, p < 0.05 ** and p < 0.001 ***.
# The levels are set in Settings > Configure and kept in preferences.json in the main folder,
# so they last between sessions. Changing them shows in a model's results the next time it runs
DEFAULT_LEVELS = [0.1, 0.05, 0.001]
PREFERENCES = Path(__file__).parent.parent / "preferences.json"
KEY = "star_levels"


# Returns a reason the levels can't be used, or None if they are fine: three numbers between
# 0 and 1, each smaller than the one before, as more stars must mean more significant
def problem_with(levels):
    if len(levels) != len(DEFAULT_LEVELS):
        return f"There must be {len(DEFAULT_LEVELS)} levels"
    for stars, level in enumerate(levels, start=1):
        if not 0 < level < 1:
            return f"The level for {'*' * stars} must be between 0 and 1"
    for stars in range(1, len(levels)):
        if levels[stars] >= levels[stars - 1]:
            return f"The level for {'*' * (stars + 1)} must be smaller than the one for {'*' * stars}"
    return None


def read_preferences():
    try:
        preferences = json.loads(PREFERENCES.read_text(encoding="utf-8"))
        return preferences if isinstance(preferences, dict) else {}
    except (OSError, ValueError):
        return {}


# The saved levels, or the defaults if none are saved or the saved ones can't be used
def load():
    saved = read_preferences().get(KEY)
    try:
        levels = [float(level) for level in saved]
    except (TypeError, ValueError):
        return list(DEFAULT_LEVELS)
    return levels if problem_with(levels) is None else list(DEFAULT_LEVELS)


levels = load()


# Uses new levels from now on and saves them, keeping any other preferences in the file
def save(new_levels):
    global levels
    reason = problem_with(new_levels)
    if reason:
        raise ValueError(reason)
    levels = list(new_levels)
    preferences = read_preferences()
    preferences[KEY] = levels
    PREFERENCES.write_text(json.dumps(preferences, indent=2), encoding="utf-8")


def stars(p_value):
    if pd.isna(p_value):
        return ""
    return "*" * sum(p_value < level for level in levels)


# What each number of stars means, e.g. "* p < 0.1    ** p < 0.05    *** p < 0.001"
def legend():
    return "    ".join(f"{'*' * stars} p < {level:g}" for stars, level in enumerate(levels, start=1))


# Puts a stars column, worked out from the p value column, straight after the estimate it marks,
# and returns the table
def add_stars(results, p_column="p_value", after="coefficient"):
    results.insert(results.columns.get_loc(after) + 1, "stars", results[p_column].map(stars))
    return results
