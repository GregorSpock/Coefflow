import ast

from engine.generate import replace_backticks


# Checks a Filter or Split condition only compares fields and values, e.g. year >= 2000 and
# region == 'London'. pandas runs conditions, and would also let one call methods such as
# .to_csv() or reach Python variables with @, so a workflow from someone else could write files
def check(condition):
    text, _ = replace_backticks(condition)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"This isn't a valid condition ({error.msg})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError(f"Conditions can't use .{node.attr}")
        if isinstance(node, ast.MatMult):
            raise ValueError("Conditions can't use @")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("Conditions can only call plain functions, such as abs(x)")


def query(data, condition):
    check(condition)
    return data.query(condition)


def evaluate(data, condition):
    check(condition)
    return data.eval(condition)
