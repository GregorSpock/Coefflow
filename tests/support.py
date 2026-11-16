import json
from pathlib import Path

import numpy as np
import pandas as pd


# Helpers shared by the tests. The golden files in tests/golden hold a summary of what each sample
# workflow and each add-on module gave when the summaries were last made (see make_golden.py), so
# a change to the app that changes any result shows up as a failing test
ROOT = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "golden"
SAMPLE_DATA = ROOT / "modules" / "input_data" / "sample_data.csv"


# A value from a table's attrs as something that can be written to JSON, or None if it can't be
def scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, str):
        return value
    return None


# The shape, column names, column totals and plain attrs of a table
def summarise(table):
    numeric = table.select_dtypes("number")
    return {
        "shape": list(table.shape),
        "columns": [str(column) for column in table.columns],
        "sums": {str(column): scalar(numeric[column].astype(float).sum()) for column in numeric.columns},
        "attrs": {key: scalar(value) for key, value in table.attrs.items() if scalar(value) is not None},
    }


def summarise_outputs(outputs):
    return [summarise(output) if isinstance(output, pd.DataFrame) else None for output in outputs]


# True if two summaries match, with numbers allowed to differ by a rounding error
def same(a, b, tolerance=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key], tolerance) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
    return a == b


def read_golden(name):
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


def write_golden(name, summary):
    GOLDEN.mkdir(exist_ok=True)
    (GOLDEN / f"{name}.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
