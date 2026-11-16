import numpy as np
import pandas as pd

from engine import conditions


# What the table viewer shows, worked out without a window: the tables a module was given and
# gave back, rows narrowed by a search and a condition, and a summary of every column. The window
# (ui/table_viewer.py) only lays these out, so all of it can be tested


# Every table a node has to look at, as (label, table): its outputs from the last run, then its
# inputs, so a module that gives no table (Output Data, a chart) still shows what it was handed
def tables_of(node):
    found = []
    specs = getattr(node.module, "OUTPUT_SPECS", [])
    for index, table in enumerate(node.last_outputs or []):
        if isinstance(table, pd.DataFrame):
            name = specs[index][0] if index < len(specs) else "Output"
            found.append((f"Out {index + 1}: {name}", table))
    specs = getattr(node.module, "INPUT_SPECS", [])
    for index, table in enumerate(node.last_inputs or []):
        if isinstance(table, pd.DataFrame):
            name = specs[min(index, len(specs) - 1)][0] if specs else "Input"
            found.append((f"In {index + 1}: {name}", table))
    return found


# The rows with the search text anywhere in them, ignoring case, and matching the condition
# (as Filter takes them, e.g. year >= 2000 and region == 'London'). Either can be blank.
# Raises ValueError for a condition that can't be used
def narrow(data, search="", condition=""):
    rows = data
    if condition.strip():
        mask = conditions.evaluate(rows, condition)
        if not isinstance(mask, pd.Series) or mask.dtype != bool:
            mask = pd.Series(mask, index=rows.index).fillna(False).astype(bool)
        rows = rows[mask.fillna(False)]
    if search.strip():
        text = search.strip().lower()
        found = pd.Series(False, index=rows.index)
        for column in rows.columns:
            found |= rows[column].astype(str).str.lower().str.contains(text, regex=False, na=False)
        found |= pd.Series(rows.index.astype(str).str.lower().str.contains(text, regex=False), index=rows.index)
        rows = rows[found]
    return rows


def kind_of(values):
    if pd.api.types.is_bool_dtype(values):
        return "yes/no"
    if pd.api.types.is_numeric_dtype(values):
        return "whole number" if pd.api.types.is_integer_dtype(values) else "number"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "date"
    return "text"


# One row per column: its kind, how many values are there and missing, how many differ, the
# usual statistics for numbers and dates, and the most common value with how often it comes up
def column_summary(data):
    rows = []
    for column in data.columns:
        values = data[column]
        present = values.dropna()
        row = {
            "field": str(column),
            "kind": kind_of(values),
            "values": int(len(present)),
            "missing": int(values.isna().sum()),
            "distinct": int(present.nunique()),
            "mean": np.nan, "sd": np.nan, "min": None, "median": np.nan, "max": None,
            "most_common": "", "count": np.nan,
        }
        if row["kind"] in ("number", "whole number") and len(present):
            numbers = present.astype(float)
            row.update(mean=float(numbers.mean()), sd=float(numbers.std()) if len(numbers) > 1 else np.nan,
                       min=float(numbers.min()), median=float(numbers.median()), max=float(numbers.max()))
        elif row["kind"] == "date" and len(present):
            row.update(min=str(present.min().date()), max=str(present.max().date()))
        if len(present):
            counts = present.astype(str).value_counts()
            row.update(most_common=str(counts.index[0]), count=int(counts.iloc[0]))
        rows.append(row)
    return pd.DataFrame(rows, columns=["field", "kind", "values", "missing", "distinct", "mean", "sd", "min", "median", "max", "most_common", "count"])


# The most common values of one column with their counts and shares, for the viewer's side pane
def frequencies(values, limit=20):
    counts = values.astype(object).where(values.notna(), "(missing)").astype(str).value_counts()
    table = counts.head(limit).rename_axis("value").reset_index(name="count")
    table["share"] = table["count"] / max(1, len(values))
    return table
