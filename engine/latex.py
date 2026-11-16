import math

import numpy as np
import pandas as pd


# Tables as LaTeX, laid out with the booktabs package the way journals print them: a heavy rule
# at the top and bottom, a light one under the headings. Output Data saves any table this way,
# and Reg Table saves its side-by-side regression table, with the significance note underneath.
# The result is a complete table environment that can be \input into a paper; it needs
# \usepackage{booktabs} in the preamble, and \usepackage{threeparttable} when there is a note

SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


# Text with LaTeX's special characters made safe, so a field called gdp_growth or 50% prints as it reads
def escape(text):
    return "".join(SPECIAL.get(character, character) for character in str(text))


# A cell as LaTeX text: numbers to the given decimal places, missing values blank, and the stars
# after a coefficient raised as a superscript so the numbers in a column still line up
def cell(value, decimals=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return escape(str(value))
        if value != 0 and (abs(value) >= 1e6 or abs(value) < 10 ** -decimals):
            mantissa, power = f"{value:.{decimals}e}".split("e")
            return f"${mantissa} \\times 10^{{{int(power)}}}$"
        return f"{value:.{decimals}f}"
    if isinstance(value, pd.Timestamp):
        return escape(value.date().isoformat() if value == value.normalize() else value.isoformat())
    return stars_up(escape(value))


# Moves the stars at the end of a coefficient such as 0.123** into a superscript: 0.123$^{**}$
def stars_up(text):
    stripped = text.rstrip("*")
    stars = len(text) - len(stripped)
    if stars and stripped and (stripped[-1].isdigit() or stripped[-1] == ")"):
        return f"{stripped}$^{{{'*' * stars}}}$"
    return text


# l for text columns and r for numbers
def alignment(data):
    return "".join("r" if pd.api.types.is_numeric_dtype(data[column]) else "l" for column in data.columns)


# A table as a LaTeX table environment. caption and label are left out when empty. note goes in
# a footnote under the table. rule_before lists row positions to draw a light rule above, e.g. to
# set the fit statistics apart from the coefficients
def table(data, caption="", label="", note="", decimals=3, rule_before=(), header=True, align=None, header_rule=True):
    columns = align or alignment(data)
    lines = ["\\begin{table}[htbp]", "\\centering"]
    if note:
        lines.append("\\begin{threeparttable}")
    if caption:
        lines.append(f"\\caption{{{escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{columns}}}")
    lines.append("\\toprule")
    if header:
        lines.append(" & ".join(escape(column) for column in data.columns) + " \\\\")
        if header_rule:
            lines.append("\\midrule")
    for position, row in enumerate(data.itertuples(index=False)):
        if position in rule_before:
            lines.append("\\midrule")
        lines.append(" & ".join(cell(value, decimals) for value in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append("\\begin{tablenotes}\\footnotesize")
        lines.append(f"\\item {escape(note)}")
        lines.append("\\end{tablenotes}")
        lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# A side-by-side regression table built by Reg Table, whose attrs say how many header,
# coefficient and statistic rows it has: the numbered headings, then the dependent variables and
# model names, ruled off from the coefficients, which are ruled off from the fit statistics
def regression_table(data, label=""):
    attrs = data.attrs
    header = attrs.get("header_rows", 0)
    coefficients = attrs.get("coefficient_rows", len(data) - header)
    rules = [header] if header else []
    if attrs.get("statistic_rows"):
        rules.append(header + coefficients)
    return table(
        data,
        caption=attrs.get("title", ""),
        label=label,
        note=attrs.get("note", ""),
        rule_before=rules,
        align="l" + "c" * (len(data.columns) - 1),
        header_rule=not header,
    )
