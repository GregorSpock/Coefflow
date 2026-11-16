import ast
import re

import numpy as np
from scipy import stats

from engine.generate import FIELD_PREFIX, replace_backticks


# Tests of linear restrictions on a fitted model's coefficients, such as
#   gdp_growth = 0                 one coefficient is zero
#   gdp_growth = 0, inflation = 0  several at once (a joint test)
#   gdp_growth = inflation         two coefficients are equal
#   2*a + b = 1                    any linear combination equals a number
# Restrictions are separated by commas or semicolons, and one written without = means "= 0".
# Coefficient names are those in the model's variable column; a name with spaces or symbols, such as
# a text field's dummy region[T.London], is written in backticks. Intercept names the constant.
#
# The Wald test uses the model's own covariance, so it respects robust, clustered or HAC errors.
# The likelihood ratio (LR) test compares the log-likelihoods of a model and one nested in it


# Splits the restrictions text into one restriction each, leaving commas inside backticks alone
def split(text):
    parts, current, quoted = [], "", False
    for character in text:
        if character == "`":
            quoted = not quoted
        if character in ",;" and not quoted:
            parts.append(current)
            current = ""
        else:
            current += character
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


# A linear expression as a dict of coefficient name -> weight, and a number on its own.
# Raises ValueError for anything that isn't linear in the coefficients
def linear(node, names, known):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {}, float(node.value)
    if isinstance(node, ast.Name):
        name = node.id
        if name.startswith(FIELD_PREFIX):
            name = names[int(name[len(FIELD_PREFIX):])]
        match = match_name(name, known)
        return {match: 1.0}, 0.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        weights, constant = linear(node.operand, names, known)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return {k: sign * v for k, v in weights.items()}, sign * constant
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, left_constant = linear(node.left, names, known)
        right, right_constant = linear(node.right, names, known)
        sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
        for key, value in right.items():
            left[key] = left.get(key, 0.0) + sign * value
        return left, left_constant + sign * right_constant
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        left, left_constant = linear(node.left, names, known)
        right, right_constant = linear(node.right, names, known)
        if isinstance(node.op, ast.Div):
            if right:
                raise ValueError("A restriction can't divide by a coefficient")
            return {k: v / right_constant for k, v in left.items()}, left_constant / right_constant
        if left and right:
            raise ValueError("A restriction can't multiply two coefficients together")
        if right:
            left, left_constant, right, right_constant = right, right_constant, left, left_constant
        return {k: v * right_constant for k, v in left.items()}, left_constant * right_constant
    raise ValueError("Restrictions can only add, subtract and scale coefficients by numbers")


# The coefficient a written name stands for: an exact match, or one ignoring case, with "const"
# and "Constant" standing for the intercept
def match_name(name, known):
    if name in known:
        return name
    if name.lower() in ("const", "constant", "intercept", "_cons"):
        for candidate in known:
            if candidate.lower() in ("intercept", "const", "constant"):
                return candidate
    lowered = {k.lower(): k for k in known}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise ValueError(f"The model has no coefficient called {name}. It has: {', '.join(known)}")


# The restrictions as R and q in R b = q, with each restriction's text tidied for showing
def parse(text, known):
    rows, constants, shown = [], [], []
    for part in split(text):
        stood_in, names = replace_backticks(part)
        sides = re.split(r"(?<![=!<>])=(?!=)", stood_in.replace("==", "="))
        if len(sides) > 2:
            raise ValueError(f"{part} has more than one =")
        left_text = sides[0]
        right_text = sides[1] if len(sides) == 2 else "0"
        try:
            left = ast.parse(left_text.strip(), mode="eval").body
            right = ast.parse(right_text.strip() or "0", mode="eval").body
        except SyntaxError:
            raise ValueError(f"{part} isn't a restriction Coefflow can read")
        left_weights, left_constant = linear(left, names, known)
        right_weights, right_constant = linear(right, names, known)
        weights = dict(left_weights)
        for key, value in right_weights.items():
            weights[key] = weights.get(key, 0.0) - value
        if not any(abs(v) > 0 for v in weights.values()):
            raise ValueError(f"{part} doesn't involve any coefficient")
        rows.append([weights.get(name, 0.0) for name in known])
        constants.append(right_constant - left_constant)
        shown.append(part if len(sides) == 2 else f"{part} = 0")
    if not rows:
        raise ValueError("Type at least one restriction, e.g. gdp_growth = 0")
    matrix = np.array(rows, dtype=float)
    if np.linalg.matrix_rank(matrix) < len(rows):
        raise ValueError("Some restrictions repeat or follow from the others, so they can't be tested together")
    return matrix, np.array(constants, dtype=float), shown


# The Wald statistic of R b = q given the coefficients b and their covariance. Returns the
# statistic and p value: an F statistic with (restrictions, residual df) degrees of freedom when
# df_resid is given, as for a linear model, otherwise a chi-squared one
def wald(R, q, coefficients, covariance, df_resid=None):
    gap = R @ coefficients - q
    middle = R @ covariance @ R.T
    statistic = float(gap @ np.linalg.solve(middle, gap))
    restrictions = len(q)
    if df_resid:
        f = statistic / restrictions
        return f, float(stats.f.sf(f, restrictions, df_resid)), f"F({restrictions}, {int(df_resid)})"
    return statistic, float(stats.chi2.sf(statistic, restrictions)), f"chi2({restrictions})"


# The likelihood ratio test of a restricted model nested in a fuller one
def likelihood_ratio(full, restricted):
    statistic = 2 * (full["log_likelihood"] - restricted["log_likelihood"])
    df = full["parameters"] - restricted["parameters"]
    if df <= 0:
        raise ValueError("The restricted model (In 2) needs fewer coefficients than the full model (In 1)")
    if full.get("observations") != restricted.get("observations"):
        raise ValueError(
            f"The two models were fitted on different rows ({full.get('observations')} and {restricted.get('observations')}), "
            "so an LR test can't compare them"
        )
    return float(statistic), float(stats.chi2.sf(max(statistic, 0.0), df)), f"chi2({df})"
