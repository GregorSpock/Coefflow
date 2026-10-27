import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from .. import common


INPUTS = 1
OUTPUTS = 2
INPUT_SPECS = [
    ("Data", "a table with an ordered numeric Y column with at least 3 categories, and numeric or text X and C columns (text ones enter as dummies)"),
]
OUTPUT_SPECS = [
    ("Test", "one row for all terms together and one per term, with chi2, df and p_value"),
    ("Cut logits", "each term's coefficient in the binary logit of Y above each cut, which the test compares"),
]

FIELD_SETTINGS = common.FIELD_SETTINGS


# The Brant (1990) test of the proportional odds (parallel lines) assumption behind Ologit and
# Oprobit: that each X shifts the odds of being above every cut by the same amount. It fits one
# binary logit of Y above each cut and tests whether their slopes are equal. A small p value
# means the assumption fails for that term, so a single coefficient hides effects that differ by cut
def on_click(node):
    return common.open_panel(node, "Brant Test", show_test)


def show_test(panel):
    if not common.show_roles(panel):
        return
    node = panel.node
    if not node.last_outputs:
        panel.message = "Choose an ordered Y with at least 3 categories and the X variables, then run the workflow"
        panel.lines.append(common.Y_HINT)
        return

    results = node.last_outputs[0]
    overall = results.iloc[0]
    verdict = "rejected" if overall["p_value"] < 0.05 else "not rejected"
    panel.lines.append(
        f"All terms: chi2 {overall['chi2']:.2f} on {overall['df']} df, p {overall['p_value']:.3g}    "
        f"Proportional odds {verdict} at 5%"
    )
    panel.note = " ".join(results.attrs["warnings"])
    panel.preview = results


# The covariance between the coefficients of the binary logits above cuts j and m (with j at or
# below m), from Brant (1990): (X'W_j X)^-1 X'W_jm X (X'W_m X)^-1, where W_j holds p_j (1 - p_j)
# and W_jm holds p_m - p_j p_m, p_j being each row's fitted probability of being above cut j
def cross_covariance(x, p_j, p_m, inverse_j, inverse_m):
    middle = x.T @ (x * (p_m - p_j * p_m)[:, None])
    return inverse_j @ middle @ inverse_m


# A Wald test that the slopes picked by index are equal across the cut logits.
# slopes is (cuts x terms) and covariance covers the stacked rows of slopes
def wald_equal(slopes, covariance, index):
    cuts, terms = slopes.shape
    chosen = np.concatenate([np.arange(terms)[index] + c * terms for c in range(cuts)])
    b = slopes[:, index].ravel()
    v = covariance[np.ix_(chosen, chosen)]
    size = len(b) // cuts
    # Each later cut's slopes minus the first cut's
    contrast = np.hstack([np.tile(np.eye(size), (cuts - 1, 1)), -np.eye(size * (cuts - 1))])
    difference = contrast @ b
    statistic = float(difference @ np.linalg.pinv(contrast @ v @ contrast.T) @ difference)
    df = size * (cuts - 1)
    return statistic, df, stats.chi2.sf(statistic, df)


def run(inputs, settings):
    notes = []
    used, y, x, roles, categories = common.prepare(inputs, settings, notes, min_categories=3)
    design = sm.add_constant(x, has_constant="add")
    values = design.to_numpy(dtype=float)
    cuts = categories[:-1]

    fits, probabilities, inverses = [], [], []
    for cut in cuts:
        above = (y > cut).astype(float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = sm.Logit(above, design).fit(disp=0, maxiter=200)
            except np.linalg.LinAlgError:
                raise ValueError("A cut logit could not be fitted; an X or C variable may be a combination of the others")
        if any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
            notes.append(f"Some variables predict Y above {common.category_text(cut)} perfectly, so the test is unreliable.")
        p = model.predict(design).to_numpy()
        fits.append(model)
        probabilities.append(p)
        inverses.append(np.linalg.pinv(values.T @ (values * (p * (1 - p))[:, None])))

    # Stack the covariances of every pair of cut logits, then keep only the slopes
    size = values.shape[1]
    full = np.zeros((len(cuts) * size, len(cuts) * size))
    for j in range(len(cuts)):
        for m in range(j, len(cuts)):
            block = cross_covariance(values, probabilities[j], probabilities[m], inverses[j], inverses[m])
            full[j * size:(j + 1) * size, m * size:(m + 1) * size] = block
            full[m * size:(m + 1) * size, j * size:(j + 1) * size] = block.T
    keep = [c * size + t for c in range(len(cuts)) for t in range(1, size)]
    covariance = full[np.ix_(keep, keep)]
    slopes = np.array([np.asarray(model.params)[1:] for model in fits])

    terms = x.shape[1]
    rows = []
    statistic, df, p = wald_equal(slopes, covariance, np.arange(terms))
    rows.append({"role": "All", "variable": "All terms", "chi2": statistic, "df": df, "p_value": p})
    for t, term in enumerate(x.columns):
        statistic, df, p = wald_equal(slopes, covariance, np.array([t]))
        rows.append({"role": roles[t], "variable": term, "chi2": statistic, "df": df, "p_value": p})
    results = pd.DataFrame(rows)

    small = [c for c in categories if (y == c).sum() < 10]
    if small:
        notes.append("Some categories have fewer than 10 rows, so the test may reject too often.")
    results.attrs = {"warnings": notes, "observations": len(used)}

    logits = pd.DataFrame({"role": ["Intercept"] + roles, "variable": ["Intercept"] + list(x.columns)})
    for cut, model in zip(cuts, fits):
        logits[f"Y>{common.category_text(cut)}"] = np.asarray(model.params)
    return [results, logits]
