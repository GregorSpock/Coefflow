import ast
import keyword
import re

import numpy as np
import pandas as pd


# Makes new fields from lines of Python such as gdp_pc = rgdpe / pop, one new field per line.
# Fields are written by name, or in backticks if the name isn't a plain word (`GDP per head`).
# A line can use the fields made by the lines above it, and a name already in the data is replaced.
# Expressions are ordinary Python with a few changes so they work on whole columns at once:
#   ^ means power, like **
#   and, or and not combine conditions row by row, as do chains like 0 < x < 1
#   a if condition else b picks row by row
#   a condition gives 1 where true and 0 where false, and missing where a field it compares is missing
#   results that can't be worked out, like x / 0 or log(0), are missing rather than infinite
# Only the functions in FUNCTIONS can be called, attributes like x.to_csv can't be used, and names
# starting with _ can't be used, so a line can't reach anything outside the data


# Turns a field to group by, or a list of them, into what pandas groupby takes
def group_keys(by):
    if isinstance(by, (list, tuple)):
        return [pd.Series(b) for b in by]
    return pd.Series(by)


def grouped(x, by):
    x = pd.Series(x)
    return x if by is None else x.groupby(group_keys(by))


# The value n rows back, within each group of by if given. Rows should be sorted by time first
def lag(x, n=1, by=None):
    return grouped(x, by).shift(n)


def lead(x, n=1, by=None):
    return grouped(x, by).shift(-n)


def diff(x, n=1, by=None):
    return pd.Series(x) - lag(x, n, by)


# The change on n rows back as a share, e.g. 0.05 for 5% growth
def growth(x, n=1, by=None):
    return pd.Series(x) / lag(x, n, by) - 1


# A statistic of the whole column, or of each group of by, given on every row
def statistic(name):
    def calculate(x, by=None):
        x = pd.Series(x)
        if by is None:
            return pd.Series(getattr(x, name)(), index=x.index)
        return x.groupby(group_keys(by)).transform(name)
    return calculate


# Logs and square roots of values they don't work for, like log(0), give missing values
def positive_only(function, allow_zero=False):
    def calculate(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce") if not np.isscalar(x) else x
        valid = (x >= 0) if allow_zero else (x > 0)
        return function(x.where(valid)) if not np.isscalar(x) else (function(x) if valid else np.nan)
    return calculate


def where(condition, yes, no):
    condition = pd.Series(condition)
    picked = pd.Series(np.where(condition.fillna(False).astype(bool), yes, no), index=condition.index)
    return picked.where(condition.notna())


def missing(x):
    return pd.Series(x).isna()


def to_number(x):
    return pd.to_numeric(pd.Series(x), errors="coerce")


# Text of each value, keeping missing values missing
def to_text(x):
    x = pd.Series(x)
    return x.astype(str).where(x.notna())


FUNCTIONS = {
    "log": positive_only(np.log),
    "ln": positive_only(np.log),
    "log10": positive_only(np.log10),
    "sqrt": positive_only(np.sqrt, allow_zero=True),
    "exp": np.exp,
    "abs": np.abs,
    "int": np.trunc,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "where": where,
    "missing": missing,
    "float": to_number,
    "str": to_text,
    "lag": lag,
    "lead": lead,
    "diff": diff,
    "growth": growth,
    "mean": statistic("mean"),
    "median": statistic("median"),
    "sum": statistic("sum"),
    "std": statistic("std"),
    "min": statistic("min"),
    "max": statistic("max"),
    "count": statistic("count"),
    "cumsum": lambda x, by=None: grouped(x, by).cumsum(),
}
CONSTANTS = {"pi": np.pi, "e": np.e, "nan": np.nan}

# The node types a line can be built from. Anything else, like lambda or a loop, is refused
ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword,
    ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.Slice, ast.List, ast.Tuple,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)
BACKTICKS = re.compile(r"`([^`]*)`")
FIELD_PREFIX = "__field"
FUNCTION_PREFIX = "__function_"
COMPARE = "__compare"


# Swaps each `backticked name` for a stand-in word Python can read. Returns the text and the names
def replace_backticks(text):
    names = []

    def stand_in(match):
        names.append(match.group(1))
        return f"{FIELD_PREFIX}{len(names) - 1}"

    return BACKTICKS.sub(stand_in, text), names


# How a field's name is written in a line: bare if it is a plain word, else in backticks
def written(name):
    if name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_") and name not in FUNCTIONS:
        return name
    return f"`{name}`"


# Splits a line into the name written before = and the expression after it.
# Returns (None, line) if the line doesn't start with a name and =
def split_line(line):
    match = re.match(r"\s*(`[^`]*`|[^\s=`]+)\s*=(?!=)", line)
    if not match:
        return None, line
    return match.group(1), line[match.end():]


# The new field's name from a line, e.g. gdp_pc from "gdp_pc = rgdpe / pop", or None if it has none yet
def target_name(line):
    name, _ = split_line(line)
    if name is None:
        return None
    if name.startswith("`"):
        return name[1:-1] or None
    return name if name.isidentifier() and not keyword.iskeyword(name) else None


# Gives a field a new name everywhere an expression uses it, bare or in backticks, but not in
# keywords like by= or function names like log(
def rename_in(expression, old, new):
    pattern = f"`{re.escape(old)}`"
    if old.isidentifier():
        pattern += rf"|(?<![\w`.]){re.escape(old)}(?![\w`])(?!\s*=(?!=))(?!\s*\()"
    return re.sub(pattern, lambda match: written(new), expression)


# Renames a field in the expression part of a line, leaving the new field's own name alone
def rename_line(line, old, new):
    name, expression = split_line(line)
    if name is None:
        return rename_in(line, old, new)
    return line[:len(line) - len(expression)] + rename_in(expression, old, new)


class Rewrite(ast.NodeTransformer):
    # Changes a checked expression so it works on whole columns (see the notes at the top)
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.BitXor):
            node.op = ast.Pow()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        combined = node.values[0]
        for value in node.values[1:]:
            combined = ast.BinOp(left=combined, op=op, right=value)
        return combined

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            node.op = ast.Invert()
        return node

    # a < b < c becomes compare(a, "<", b) & compare(b, "<", c)
    def visit_Compare(self, node):
        self.generic_visit(node)
        parts = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            symbol = ast.Constant(type(op).__name__)
            parts.append(ast.Call(func=ast.Name(COMPARE, ast.Load()), args=[left, symbol, right], keywords=[]))
            left = right
        combined = parts[0]
        for part in parts[1:]:
            combined = ast.BinOp(left=combined, op=ast.BitAnd(), right=part)
        return combined

    def visit_IfExp(self, node):
        self.generic_visit(node)
        return ast.Call(func=ast.Name(f"{FUNCTION_PREFIX}where", ast.Load()), args=[node.test, node.body, node.orelse], keywords=[])

    # Function calls look in FUNCTIONS, so a field called log doesn't hide log()
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and not node.func.id.startswith("__"):
            node.func.id = FUNCTION_PREFIX + node.func.id
        return node


COMPARISONS = {
    "Eq": lambda a, b: a == b, "NotEq": lambda a, b: a != b,
    "Lt": lambda a, b: a < b, "LtE": lambda a, b: a <= b,
    "Gt": lambda a, b: a > b, "GtE": lambda a, b: a >= b,
    "In": lambda a, b: pd.Series(a).isin(b), "NotIn": lambda a, b: ~pd.Series(a).isin(b),
}


# Compares row by row, giving missing where either side is missing so a condition never
# quietly counts a missing value as false
def compare(left, op, right):
    result = COMPARISONS[op](left, right)
    if np.isscalar(result):
        return result
    result = pd.Series(result).astype("boolean")
    sides = [left] if op in ("In", "NotIn") else [left, right]
    for side in sides:
        if isinstance(side, pd.Series):
            result = result.mask(side.isna())
    return result


# Checks every part of an expression is allowed and every name is a field, function or constant
def check(tree, data, fields):
    called = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ValueError(f"{type(node).__name__} can't be used here")
        if any(node is function for function in called):
            if node.id not in FUNCTIONS:
                raise ValueError(f"There is no function called {node.id}")
        elif isinstance(node, ast.Name) and not node.id.startswith(FIELD_PREFIX):
            if node.id.startswith("_"):
                raise ValueError(f"Write `{node.id}` in backticks to use a field starting with _")
            if node.id not in data.columns and node.id not in CONSTANTS:
                raise ValueError(f"There is no field called {node.id}")
    for name in fields:
        if name not in data.columns:
            raise ValueError(f"There is no field called {name}")


# The fields an expression uses, in the order it uses them
def fields_used(expression, data):
    text, backticked = replace_backticks(expression)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return []
    called = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = []
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and not any(node is f for f in called)]
    for node in sorted(found, key=lambda node: node.col_offset):
        if node.id.startswith(FIELD_PREFIX) and node.id[len(FIELD_PREFIX):].isdigit():
            names.append(backticked[int(node.id[len(FIELD_PREFIX):])])
        elif node.id in data.columns:
            names.append(node.id)
    return [n for n in dict.fromkeys(names) if n in data.columns]


# Works out one line on the data. Returns the new field's name and its values
def evaluate(line, data):
    name = target_name(line)
    written_name, expression = split_line(line)
    if written_name is None:
        raise ValueError("Start with the new field's name and =, e.g. gdp_pc = rgdpe / pop")
    if name is None:
        raise ValueError(f"{written_name} can't be a field name. Use letters, digits and _, or put it in backticks")
    if not expression.strip():
        raise ValueError(f"{name} has no expression after = yet")

    text, backticked = replace_backticks(expression)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"This isn't valid Python yet ({error.msg})")
    check(tree, data, backticked)

    tree = ast.fix_missing_locations(Rewrite().visit(tree))
    namespace = {c: data[c] for c in data.columns if isinstance(c, str)}
    namespace.update({k: v for k, v in CONSTANTS.items() if k not in namespace})
    namespace.update({f"{FIELD_PREFIX}{i}": data[n] for i, n in enumerate(backticked)})
    namespace.update({FUNCTION_PREFIX + k: v for k, v in FUNCTIONS.items()})
    namespace[COMPARE] = compare

    try:
        with np.errstate(all="ignore"):
            values = eval(compile(tree, "<gen>", "eval"), {"__builtins__": {}}, namespace)
    except Exception as error:
        raise ValueError(f"Could not work out {name}: {error}")
    return name, column_of(values, data.index)


# Makes a result into a column the length of the data. Conditions become 1, 0 or missing
def column_of(values, index):
    if isinstance(values, pd.Series):
        values = values.reindex(index) if not values.index.equals(index) else values
    elif np.ndim(values) == 0:
        values = pd.Series(values, index=index)
    else:
        values = pd.Series(np.asarray(values), index=index) if len(values) == len(index) else None
        if values is None:
            raise ValueError("The result isn't one value per row")
    if pd.api.types.is_bool_dtype(values):
        values = values.astype("Float64").astype(float)
    if pd.api.types.is_float_dtype(values):
        values = values.replace([np.inf, -np.inf], np.nan)
    return values


# Adds or replaces a field for each line, in order. A line that fails stops with a message naming it
def generate(data, lines):
    data = data.copy()
    for number, line in enumerate(lines, start=1):
        try:
            name, values = evaluate(line, data)
        except ValueError as error:
            raise ValueError(f"Line {number}: {error}")
        data[name] = values
    return data
