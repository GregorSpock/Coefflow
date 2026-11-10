import numpy as np
import pandas as pd
from scipy import stats

from engine import significance
from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with two columns of categories, one for the rows (R) and one for the columns (C) of the cross table"),
]
OUTPUT_SPECS = [
    ("Test", "the chi-square test of independence with its df and p_value, Cramer's V, and for a 2 x 2 table the continuity-corrected and Fisher's exact p values and the odds ratio"),
    ("Cells", "one row per cell with its observed and expected count, row and column percentages and adjusted residual"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["row", "column"]
ROLES = [("R", "row", False, False), ("C", "column", False, False)]
# Chi-square is unreliable when more than this share of cells expect fewer than SMALL_COUNT rows
SMALL_COUNT = 5
SMALL_SHARE = 0.2


# Opens a panel to choose the two columns, showing the test or the cells of the last run
def on_click(node):
    panel = common.open_panel(node, "Crosstab", show_table, "Test")
    common.add_choice(panel, "Show", "show", ["Test", "Cells"])
    show_table(panel)
    return panel


def show_table(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [f"Rows (R): {common.names(settings, 'row')}    Columns (C): {common.names(settings, 'column')}"]
    cells = settings.get("show") == "Cells"
    panel.preview_title = "Cells" if cells else "Test"
    common.show_results(panel, "Choose a column for the rows (R) and one for the columns (C), then run the workflow", output=1 if cells else 0)


# How strong an association Cramer's V shows, by Cohen's rules of thumb for the table's smaller side
def strength(v, smaller_side):
    small, medium, large = [size / np.sqrt(smaller_side - 1) for size in (0.1, 0.3, 0.5)]
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


# One row per cell. The adjusted residual is how far the count is from what independence
# expects, in standard errors, so beyond about 2 either way marks a cell that stands out
def cell_table(counts, expected, row, column):
    total = counts.to_numpy().sum()
    row_share = counts.sum(axis=1) / total
    column_share = counts.sum(axis=0) / total
    cells = []
    for r in counts.index:
        for c in counts.columns:
            observed, expect = counts.loc[r, c], expected.loc[r, c]
            cells.append({
                row: common.group_text(r), column: common.group_text(c), "observed": observed, "expected": expect,
                "row_percent": 100 * observed / counts.loc[r].sum(), "column_percent": 100 * observed / counts[c].sum(),
                "adjusted_residual": (observed - expect) / np.sqrt(expect * (1 - row_share[r]) * (1 - column_share[c])),
            })
    return pd.DataFrame(cells)


# Cross-tabulates two columns and tests whether they are independent
def run(inputs, settings):
    data = inputs[0]
    row, column = settings.get("row"), settings.get("column")
    if not row or not column:
        raise ValueError("Choose a column for the rows (R) and one for the columns (C)")
    common.check_fields(data, [row, column])

    pairs = data[[row, column]].dropna()
    counts = pd.crosstab(pairs[row], pairs[column])
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError(f"{row} and {column} must each have at least 2 categories in the rows used")
    if max(counts.shape) > common.MAX_GROUPS:
        raise ValueError(f"A column has {max(counts.shape)} categories, which is too many for a cross table")

    chi_square, p_value, df, expected = stats.chi2_contingency(counts.to_numpy(), correction=False)
    expected = pd.DataFrame(expected, index=counts.index, columns=counts.columns)
    total = int(counts.to_numpy().sum())
    smaller_side = min(counts.shape)
    v = np.sqrt(chi_square / (total * (smaller_side - 1)))
    small_cells = int((expected.to_numpy() < SMALL_COUNT).sum())

    result = {
        "chi_square": chi_square, "df": df, "p_value": p_value, "cramers_v": v, "n": total,
        "min_expected": expected.to_numpy().min(), "cells_below_5": small_cells,
        "corrected_p": np.nan, "fisher_p": np.nan, "odds_ratio": np.nan,
    }
    notes = [
        f"chi-sq({df}, N = {total}) = {chi_square:.2f}, {common.p_text(p_value)}, Cramer's V = {common.short(v)} ({strength(v, smaller_side)}): "
        + (f"{row} and {column} are related." if p_value < common.LEVEL else f"no significant evidence that {row} and {column} are related.")
    ]
    if counts.shape == (2, 2):
        result["corrected_p"] = stats.chi2_contingency(counts.to_numpy(), correction=True)[1]
        fisher = stats.fisher_exact(counts.to_numpy())
        result["fisher_p"], result["odds_ratio"] = fisher.pvalue, fisher.statistic
    if small_cells / expected.size > SMALL_SHARE:
        advice = "use Fisher's exact p value" if counts.shape == (2, 2) else "consider merging categories"
        notes.append(f"{small_cells} of {expected.size} cells expect fewer than {SMALL_COUNT} rows, so the chi-square p value is unreliable: {advice}.")
    if len(pairs) < len(data):
        notes.append(f"Left out {len(data) - len(pairs)} rows missing {row} or {column}.")

    table = significance.add_stars(pd.DataFrame([result]), after="chi_square")
    table.attrs = {"note": " ".join(notes)}
    return [table, cell_table(counts, expected, row, column)]
