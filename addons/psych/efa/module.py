import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.factor import Factor

from .. import common


INPUTS = 1
OUTPUTS = 3
INPUT_SPECS = [
    ("Data", "a table with one row per person and three or more numeric variables or items (V)"),
]
OUTPUT_SPECS = [
    ("Loadings", "each variable's loading on each factor, with its communality (the share of its variance the factors explain) and uniqueness"),
    ("Variance", "the variance each factor explains after rotation, as a sum of squared loadings and as a share of the total"),
    ("Eigenvalues", "the eigenvalues of the correlation matrix beside those expected from random data, which decide how many factors to keep"),
]

# Settings that hold field names, kept up to date when a field is renamed or dropped upstream
FIELD_SETTINGS = ["variables"]
ROLES = [("V", "variables", True, True)]
FACTORS = ["Auto"] + [str(count) for count in range(1, 9)]
ROTATIONS = ["Varimax", "Oblimin", "Promax", "None"]
VIEWS = ["Loadings", "Variance", "Eigenvalues"]
# Parallel analysis compares the eigenvalues with those of this many sets of random data of the
# same size, keeping the factors that beat this percentile of them. The seed keeps results the same every run
RANDOM_SETS = 200
PERCENTILE = 95
SEED = 1
# A loading at least this big either way is usually read as the variable belonging to the factor
CLEAR_LOADING = 0.4


# Opens a panel to choose the variables, the number of factors and the rotation, showing the results of the last run
def on_click(node):
    panel = common.open_panel(node, "EFA", show_factors, "Loadings")
    common.add_choice(panel, "Factors", "factors", FACTORS)
    common.add_choice(panel, "Rotate", "rotation", ROTATIONS)
    common.add_choice(panel, "Show", "show", VIEWS)
    show_factors(panel)
    return panel


def show_factors(panel):
    settings = panel.node.settings
    if common.list_fields(panel, ROLES) is None:
        return
    panel.lines = [f"Variables (V): {common.names(settings, 'variables')}"]
    view = settings.get("show", VIEWS[0])
    panel.preview_title = view
    common.show_results(panel, "Click V on three or more variables, then run the workflow", output=VIEWS.index(view) if view in VIEWS else 0)


# The Kaiser-Meyer-Olkin measure of whether the variables share enough variance to factor: the
# correlations' share of the correlations and partial correlations together, from 0 to 1
def kmo(correlations):
    inverse = np.linalg.inv(correlations)
    scale = np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
    partial = -inverse / scale
    off = ~np.eye(len(correlations), dtype=bool)
    return (correlations[off] ** 2).sum() / ((correlations[off] ** 2).sum() + (partial[off] ** 2).sum())


# Bartlett's test that the variables are uncorrelated, in which case there is nothing to factor.
# Returns the chi-square, its df and p value
def bartlett(correlations, people):
    variables = len(correlations)
    chi_square = -(people - 1 - (2 * variables + 5) / 6) * np.log(np.linalg.det(correlations))
    df = variables * (variables - 1) / 2
    return chi_square, df, float(stats.chi2.sf(chi_square, df))


# The eigenvalues that random, unrelated data of the same size would give: the PERCENTILE of
# each across RANDOM_SETS sets of normal draws
def random_eigenvalues(people, variables):
    rng = np.random.default_rng(SEED)
    found = np.empty((RANDOM_SETS, variables))
    for i in range(RANDOM_SETS):
        draws = rng.standard_normal((people, variables))
        found[i] = np.linalg.eigvalsh(np.corrcoef(draws, rowvar=False))[::-1]
    return np.percentile(found, PERCENTILE, axis=0)


# Fits an exploratory factor analysis by principal axis factoring, with the chosen rotation
def run(inputs, settings):
    data = common.clean(inputs[0])
    variables = settings.get("variables") or []
    if len(variables) < 3:
        raise ValueError("Choose at least 3 variables (V)")
    common.check_fields(data, variables, numeric=variables)

    rows = data[variables].dropna().astype(float)
    people = len(rows)
    if people <= len(variables):
        raise ValueError(f"Only {people} rows have every variable, which is too few for {len(variables)} variables")
    still = [name for name in variables if rows[name].std() == 0]
    if still:
        raise ValueError(f"These variables don't vary: {', '.join(still)}")
    correlations = np.corrcoef(rows.to_numpy(), rowvar=False)
    if np.linalg.matrix_rank(correlations) < len(variables):
        raise ValueError("Some variables are exact combinations of others, so the factors can't be found. Drop one of them")

    eigenvalues = np.linalg.eigvalsh(correlations)[::-1]
    random = random_eigenvalues(people, len(variables))
    # Factors are kept from the first until one fails to beat random data
    suggested = int(np.argmin(eigenvalues > random)) if not (eigenvalues > random).all() else len(variables)
    suggested = max(1, suggested)

    chosen = settings.get("factors", FACTORS[0])
    most = max(1, (len(variables) - 1) // 2)
    count = min(suggested if chosen == "Auto" else int(chosen), most)

    rotation = settings.get("rotation", ROTATIONS[0])
    try:
        fit = Factor(rows.to_numpy(), n_factor=count, method="pa").fit()
        if count > 1 and rotation != "None":
            fit.rotate(rotation.lower())
    except Exception as error:
        raise ValueError(f"The factors could not be found ({error}). Try fewer factors")

    # A factor's sign is arbitrary, so each is turned to load positively on balance, then they
    # are put in order of the variance they explain
    loadings = np.asarray(fit.loadings)
    loadings = loadings * np.where(loadings.sum(axis=0) < 0, -1, 1)
    loadings = loadings[:, np.argsort(-(loadings ** 2).sum(axis=0))]
    names = [f"factor_{i + 1}" for i in range(count)]

    table = pd.DataFrame(loadings, columns=names).round(4)
    table.insert(0, "variable", variables)
    # Rotating doesn't change how much of a variable the factors explain between them
    communality = (np.asarray(fit.loadings_no_rot) ** 2).sum(axis=1)
    table["communality"] = communality.round(4)
    table["uniqueness"] = (1 - communality).round(4)

    explained = (loadings ** 2).sum(axis=0)
    variance = pd.DataFrame({
        "factor": names, "ss_loadings": explained, "share": explained / len(variables),
        "cumulative": np.cumsum(explained) / len(variables),
    })
    scree = pd.DataFrame({
        "component": np.arange(1, len(variables) + 1), "eigenvalue": eigenvalues, "random_data": random,
        "keep": np.where(np.arange(len(variables)) < suggested, "yes", ""),
    })

    measure = kmo(correlations)
    chi_square, df, p_value = bartlett(correlations, people)
    notes = [
        f"{count} factor{'s' if count > 1 else ''} by principal axis factoring" + (f" with {rotation.lower()} rotation" if count > 1 and rotation != "None" else "")
        + f", explaining {100 * explained.sum() / len(variables):.0f}% of the variance.",
        f"Parallel analysis suggests {suggested}; {int((eigenvalues > 1).sum())} eigenvalues are above 1.",
        f"KMO = {common.short(measure)} ({'adequate' if measure >= 0.6 else 'too low for factoring to work well'}).",
        f"Bartlett's test: chi-sq({df:.0f}) = {chi_square:.1f}, {common.p_text(p_value)}.",
    ]
    if chosen != "Auto" and int(chosen) > most:
        notes.append(f"{len(variables)} variables can support at most {most} factors.")
    lost = [name for name, row in zip(variables, loadings) if np.abs(row).max() < CLEAR_LOADING]
    if lost:
        notes.append(f"No loading of {CLEAR_LOADING} or more: {', '.join(lost)}.")
    if rotation in ("Oblimin", "Promax") and count > 1:
        notes.append("This rotation lets the factors correlate, so the loadings are pattern coefficients and the shares of variance overlap.")
    if people < len(data):
        notes.append(f"Left out {len(data) - people} rows missing a variable.")

    table.attrs = {"note": " ".join(notes)}
    return [table, variance, scree]
