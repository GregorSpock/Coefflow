import numpy as np
import pandas as pd
from scipy import stats

from ui.panel import Panel


INPUTS = 2
OUTPUTS = 1
INPUT_SPECS = [
    ("Fixed effects", "Out 1 (the coefficient table) of Fixed Effects, connected directly"),
    ("Random effects", "Out 1 (the coefficient table) of Random Effects for the same Y and X, connected directly"),
]
OUTPUT_SPECS = [
    ("Test", "one row per coefficient both models share, with the fixed and random effects estimates, their difference and its standard error. The chi-squared statistic, degrees of freedom and p value are shown above"),
]

# Eigenvalues of the covariance difference smaller than this share of the largest count as zero
TOLERANCE = 1e-8
LEVEL = 0.05


# Opens a panel showing the result of the last run. It has no settings
def on_click(node):
    panel = Panel(node, "Hausman")
    panel.preview_title = "Test"
    panel.refresh = show_test
    show_test(panel)
    return panel


def show_test(panel):
    node = panel.node
    panel.preview = None
    panel.note = ""
    if not node.last_outputs:
        panel.lines = []
        panel.message = "Connect Out 1 of Fixed Effects to In 1 and Out 1 of Random Effects to In 2, then run the workflow"
        return

    table = node.last_outputs[0]
    test = table.attrs
    panel.message = ""
    panel.lines = [
        f"Chi squared: {test['chi_squared']:.3f}    Degrees of freedom: {test['df']}    p: {test['p_value']:.3g}",
        test["verdict"],
    ]
    panel.note = " ".join(test["warnings"])
    panel.preview = table


# The coefficients and covariance a panel model recorded, by variable name, leaving out the intercept
def estimates(table, model, position):
    fit = table.attrs.get("fit", {})
    if "variable" not in table or "coefficient" not in table:
        raise ValueError(f"In {position} needs a coefficient table, like Out 1 of {model}")
    if fit.get("model") != model:
        raise ValueError(f"In {position} needs Out 1 of {model}, connected directly")
    covariance = fit.get("covariance")
    if covariance is None or covariance.shape != (len(table), len(table)):
        raise ValueError(f"The {model} table has lost its record of the fit (e.g. it was saved and loaded again, or rows were removed), so connect Out 1 of {model} directly")
    names = list(table["variable"].astype(str))
    beta = pd.Series(table["coefficient"].to_numpy(float), index=names)
    covariance = pd.DataFrame(covariance, index=names, columns=names)
    return beta.drop("Intercept"), covariance.drop(index="Intercept", columns="Intercept"), fit


# Warnings about the two fits being comparable, and about the test's assumptions
def check_fits(fixed, random, fixed_fit, random_fit):
    warnings = []
    if fixed_fit["dependent"] != random_fit["dependent"]:
        raise ValueError(f"The models have different Y fields ({fixed_fit['dependent']} and {random_fit['dependent']})")
    if fixed.attrs.get("observations") != random.attrs.get("observations"):
        warnings.append(
            f"The models use different rows ({fixed.attrs.get('observations')} and {random.attrs.get('observations')}), so the test is not valid. "
            "Give both the same data and variables."
        )
    if fixed.attrs.get("cluster") or random.attrs.get("cluster"):
        warnings.append(
            "The test assumes ordinary standard errors, as it relies on random effects being the efficient estimator. "
            "With robust or clustered errors it is not valid: switch Robust and Cluster off on both models for this test."
        )
    return warnings


# The Hausman test of whether random effects is consistent. Fixed effects is consistent either way,
# while random effects is only consistent if the entity effects are unrelated to the X variables, but
# is then more efficient. Under that assumption the difference d between their shared coefficients
# has covariance V_fe - V_re, and H = d' (V_fe - V_re)^-1 d is chi-squared with as many degrees of
# freedom as coefficients. A small p value says the coefficients differ by more than chance, so use
# fixed effects. As in Stata's hausman with sigmamore, both covariances use the error variance of
# random effects, the efficient model, as using each model's own estimate often leaves V_fe - V_re
# not positive definite. If it still isn't, a generalised inverse is used over the directions where
# it is positive and the degrees of freedom are their number
def run(inputs, settings):
    fixed, random = inputs
    fixed_beta, fixed_cov, fixed_fit = estimates(fixed, "Fixed Effects", 1)
    random_beta, random_cov, random_fit = estimates(random, "Random Effects", 2)
    warnings = check_fits(fixed, random, fixed_fit, random_fit)

    shared = [name for name in fixed_beta.index if name in random_beta.index]
    if not shared:
        raise ValueError("The models share no coefficients other than the intercept")
    left_out = [name for name in random_beta.index if name not in shared]
    if left_out:
        warnings.append(f"Not compared, as fixed effects can't estimate them: {', '.join(left_out)}.")

    difference = (fixed_beta[shared] - random_beta[shared]).to_numpy()
    rescale = random.attrs["error_variance"] / fixed.attrs["error_variance"]
    covariance = (rescale * fixed_cov.loc[shared, shared] - random_cov.loc[shared, shared]).to_numpy()
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    positive = values > TOLERANCE * max(values.max(), 0.0) if values.max() > 0 else np.zeros(len(values), bool)
    if not positive.any():
        raise ValueError("The fixed effects covariance is no larger than the random effects one in any direction, so the test can't be computed")
    if not positive.all():
        warnings.append(
            "The difference of the covariances is not positive definite, which happens in some samples, "
            f"so the test uses the {int(positive.sum())} directions where it is."
        )
    projected = vectors[:, positive].T @ difference
    chi_squared = float((projected ** 2 / values[positive]).sum())
    df = int(positive.sum())
    p_value = float(stats.chi2.sf(chi_squared, df))

    if p_value < LEVEL:
        verdict = f"p < {LEVEL:g}: the coefficients differ by more than chance, so random effects is inconsistent. Use fixed effects."
    else:
        verdict = f"p >= {LEVEL:g}: no evidence the coefficients differ, so random effects, being more efficient, can be used."

    variance = np.diag(covariance)
    table = pd.DataFrame({
        "variable": shared,
        "fixed_effects": fixed_beta[shared].to_numpy(),
        "random_effects": random_beta[shared].to_numpy(),
        "difference": difference,
        "std_error": np.sqrt(np.where(variance > 0, variance, np.nan)),
    })
    table.attrs = {
        "chi_squared": chi_squared,
        "df": df,
        "p_value": p_value,
        "verdict": verdict,
        "warnings": warnings,
    }
    return [table]
