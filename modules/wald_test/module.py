import numpy as np
import pandas as pd

from engine import restrictions
from engine.generate import written
from engine.panel import Panel


INPUTS = 2
REQUIRED_INPUTS = 1
OUTPUTS = 1
INPUT_TYPES = ["model", "model"]
INPUT_SPECS = [
    ("Model", "the coefficient table (Out 1) of a regression: OLS, Logit, Probit, Count, Tobit, Heckman, Quantile Reg, GARCH, "
              "VAR, VECM, Dynamic Panel, or a model from an add-on. In systems such as VAR a coefficient is named like gdp: L1.inflation"),
    ("Restricted model", "optional: the same model fitted without some of its variables, on the same rows, for a likelihood ratio (LR) test"),
]
OUTPUT_SPECS = [
    ("Test", "the joint Wald test of every restriction, then each restriction on its own, with the statistic, its distribution, the p value "
             "and the estimated value of each restricted combination. With In 2 connected, an LR test row comes first"),
]
LEVEL = 0.05
# Linear models are tested with F, using their residual degrees of freedom
LINEAR = ["linear", "quantile"]


# The names coefficients are written with: the variable, with its equation in front in a system
# of equations such as a VAR, e.g. gdp: L1.inflation
def names_of(model):
    names = model["variable"].astype(str)
    if "equation" in model:
        names = model["equation"].astype(str) + ": " + names
    return [str(name) for name in names]


def set_defaults(settings):
    settings.setdefault("restrictions", "")


# Opens a panel to type the restrictions, with a button per coefficient to put its name in
def on_click(node):
    set_defaults(node.settings)
    panel = Panel(node, "Wald Test")
    panel.add_button("Clear", clear_text)
    panel.add_button("All = 0", all_zero)
    panel.add_text_field("Restrictions, e.g. a = 0, b = c", node.settings["restrictions"], type_restrictions, blocked="", fill=True)
    panel.options_title = "Coefficients (click to add)"
    panel.preview_title = "Test"
    panel.refresh = show_test
    show_test(panel)
    return panel


def type_restrictions(panel, text):
    panel.node.settings["restrictions"] = text
    show_test(panel)


def clear_text(panel, button):
    panel.node.settings["restrictions"] = ""
    panel.text_field.set_text("")
    show_test(panel)


# The names the model's coefficients are written with in a restriction
def coefficient_names(panel):
    model = panel.node.last_inputs[0] if panel.node.last_inputs else None
    if model is None or "variable" not in model.columns:
        return []
    return names_of(model)


# Fills in the test that every coefficient but the intercept is zero
def all_zero(panel, button):
    names = [n for n in coefficient_names(panel) if n.lower() not in ("intercept", "const", "constant")]
    text = ", ".join(f"{written(name)} = 0" for name in names)
    panel.node.settings["restrictions"] = text
    panel.text_field.set_text(text)
    show_test(panel)


# Puts a coefficient's name in at the cursor
def insert_name(panel, button):
    panel.text_field.insert(written(button.value))
    panel.text_field.focused = True
    panel.node.settings["restrictions"] = panel.text_field.text
    show_test(panel)


def show_test(panel):
    node = panel.node
    settings = node.settings
    set_defaults(settings)
    panel.clear_options()
    panel.preview = None
    panel.note = ""
    panel.lines = []

    names = coefficient_names(panel)
    if not names:
        panel.message = "Connect a model (Out 1 of a regression) and run the workflow"
        return
    for name in names:
        panel.add_option(name, [("Add", insert_name, False)])
        panel.options[-1][1][0].value = name

    try:
        table = run(node.last_inputs, settings)[0]
    except Exception as error:
        panel.message = str(error)
        return
    panel.preview = table
    panel.message = ""
    joint = table[table["test"].isin(["Wald (joint)", "LR"])]
    panel.note = " ".join(f"{t}: {r}" for t, r in zip(joint["test"], joint["result"]))


def result_text(p_value, what):
    if p_value < LEVEL:
        return f"Rejected at the 5% level (p = {p_value:.3g}): {what} doesn't hold."
    return f"Not rejected at the 5% level (p = {p_value:.3g}): the data are consistent with {what}."


# Tests the typed restrictions on the model's coefficients with a Wald test using the model's own
# covariance (so robust, clustered or HAC errors carry through), jointly and one at a time.
# With a restricted model on In 2, an LR test of it against the model on In 1 comes first
def run(inputs, settings):
    set_defaults(settings)
    model = inputs[0]
    restricted = inputs[1] if len(inputs) > 1 else None
    fit = model.attrs.get("fit") or {}
    rows = []

    if restricted is not None:
        statistic, p_value, distribution = restrictions.likelihood_ratio(fit, restricted.attrs.get("fit") or {})
        rows.append({
            "test": "LR", "restriction": "In 2 is nested in In 1", "estimate": np.nan, "statistic": statistic,
            "distribution": distribution, "p_value": p_value,
            "result": result_text(p_value, "the restricted model (the left-out coefficients are all zero)"),
        })

    text = settings["restrictions"].strip()
    if text:
        covariance = fit.get("covariance")
        if covariance is None:
            raise ValueError("This model doesn't carry the covariance of its coefficients, so it can't be Wald tested. Run its workflow again")
        known = names_of(model)
        coefficients = model["coefficient"].to_numpy(float)
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (len(known), len(known)):
            raise ValueError("The model's coefficients don't match its covariance; connect its Out 1 directly")
        df_resid = fit.get("degrees_of_freedom") if fit.get("kind") in LINEAR else None

        R, q, shown = restrictions.parse(text, known)
        if fit.get("diagonal_covariance") and np.count_nonzero(np.any(R != 0, axis=0)) > 1:
            raise ValueError(f"{fit.get('model', 'This model')} keeps only each coefficient's own standard error, not how they "
                             "move together, so only restrictions on one coefficient at a time can be tested")
        statistic, p_value, distribution = restrictions.wald(R, q, coefficients, covariance, df_resid)
        rows.append({
            "test": "Wald (joint)", "restriction": ", ".join(shown), "estimate": np.nan, "statistic": statistic,
            "distribution": distribution, "p_value": p_value,
            "result": result_text(p_value, "every restriction together"),
        })
        if len(shown) > 1:
            for index, part in enumerate(shown):
                one = R[index:index + 1]
                statistic, p_value, distribution = restrictions.wald(one, q[index:index + 1], coefficients, covariance, df_resid)
                rows.append({
                    "test": "Wald", "restriction": part, "estimate": float((one @ coefficients)[0] - q[index]),
                    "statistic": statistic, "distribution": distribution, "p_value": p_value,
                    "result": result_text(p_value, part),
                })
        elif len(shown) == 1:
            rows[-1]["estimate"] = float(R[0] @ coefficients - q[0])

    if not rows:
        raise ValueError("Type the restrictions to test, e.g. gdp_growth = 0, or connect a restricted model to In 2")
    return [pd.DataFrame(rows, columns=["test", "restriction", "estimate", "statistic", "distribution", "p_value", "result"])]
