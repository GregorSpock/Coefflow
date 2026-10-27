import re

import numpy as np
import pandas as pd
from scipy import stats

from engine import clusters, dummies, panel_data
from ui import colours
from .. import common


INPUTS = 1
OUTPUTS = 3
INPUT_SPECS = [
    ("Data", "a table with a numeric Y, the numeric running variable (R) that sets treatment by crossing the cutoff, optionally a 0/1 treatment (D) for a fuzzy design, and any controls (C)"),
]
OUTPUT_SPECS = [
    ("Estimate", "the local regression around the cutoff: one row per term, the Effect row being the jump at the cutoff, with std_error, t_stat, p_value and stars"),
    ("Checks", "the estimate at other bandwidths and polynomial orders, placebo cutoffs, covariate balance and the McCrary density test, each with its p value and verdict"),
    ("Bins", "the mean of Y in evenly spaced bins of R on each side of the cutoff, as drawn on the chart"),
]

ROLES = [("Y", "dependent", False), ("R", "running", False), ("D", "treatment", False), ("C", "controls", True)]
FIELD_SETTINGS = ["dependent", "running", "treatment", "controls", "cluster"]
KERNELS = ["Triangular", "Epanechnikov", "Uniform"]
ORDERS = ["1", "2"]
SHOW = ["Estimate", "Checks", "Chart"]
DEFAULT_TEXT = "cutoff = 0, bandwidth = auto, bins = 20"
SETTINGS = {"cutoff": 0.0, "bandwidth": None, "bins": 20}

# Each kernel's weight for a row |R - cutoff| / h of the way to the edge of the bandwidth
WEIGHTS = {
    "Triangular": lambda u: np.clip(1 - u, 0, None),
    "Epanechnikov": lambda u: np.clip(1 - u ** 2, 0, None),
    "Uniform": lambda u: (u <= 1).astype(float),
}
# Imbens & Kalyanaraman's constant C_K for a local linear fit at a boundary with each kernel
IK_CONSTANTS = {"Triangular": 3.4375, "Epanechnikov": 3.1999, "Uniform": 2.7019}
BANDWIDTH_SHARES = [0.5, 0.75, 1, 1.5, 2]
MIN_SIDE = 10
MAX_BINS = 200
CURVE_POINTS = 60


# Opens a panel to give each variable a role: Y, R (running variable), D (treatment, for a fuzzy
# design) or C (control), with a line of text for the cutoff, bandwidth and number of bins
def on_click(node):
    panel = common.open_panel(node, "RDD", ROLES, show_model, robust_default=True)
    panel.add_text_field("Set", node.settings.get("text", DEFAULT_TEXT), change_text, blocked="", fill=True)
    return panel


def change_text(panel, text):
    panel.node.settings["text"] = text
    show_model(panel)


# Reads the line of text into the cutoff, bandwidth (None for automatic) and number of bins
def parse(text):
    values = dict(SETTINGS)
    for part in re.split(r"[,;]", text or ""):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Write each setting as name = value, e.g. {DEFAULT_TEXT}")
        name, value = [p.strip().lower() for p in part.split("=", 1)]
        if name not in values:
            raise ValueError(f"Unknown setting {name}; use cutoff, bandwidth or bins")
        if name == "bandwidth" and value in ("auto", ""):
            values[name] = None
            continue
        try:
            values[name] = float(value)
        except ValueError:
            raise ValueError(f"{name} must be a number, not {value}")
    if values["bandwidth"] is not None and values["bandwidth"] <= 0:
        raise ValueError("bandwidth must be above 0, or auto")
    if values["bins"] != int(values["bins"]) or not 2 <= values["bins"] <= MAX_BINS:
        raise ValueError(f"bins must be a whole number from 2 to {MAX_BINS}")
    values["bins"] = int(values["bins"])
    return values


def allowed(data, column, letter):
    if column not in dummies.model_columns(data):
        return False
    return dummies.is_numeric(data[column]) or letter == "C"


def show_model(panel):
    node = panel.node
    settings = node.settings
    panel.clear_options()
    panel.preview = None
    panel.chart = None
    panel.note = ""
    panel.message = ""
    design = "fuzzy" if settings.get("treatment") else "sharp"
    panel.lines = [
        f"Y: {common.names_text(settings, 'dependent')}    Running (R): {common.names_text(settings, 'running')}    "
        f"Treatment (D): {common.names_text(settings, 'treatment')} ({design})",
        f"Controls (C): {common.names_text(settings, 'controls')}",
    ]
    try:
        parse(settings.get("text", DEFAULT_TEXT))
    except ValueError as error:
        panel.note = str(error)
    common.add_choice(panel, "Kernel", "kernel", KERNELS, KERNELS[0])
    common.add_choice(panel, "Order", "order", ORDERS, ORDERS[0])
    common.add_choice(panel, "Show", "show", SHOW, SHOW[0])
    if not common.add_role_options(panel, ROLES, allowed):
        return

    if not node.last_outputs:
        panel.message = "Choose Y and the running variable (R), set the cutoff, then run the workflow"
        return
    results, checks, bins = node.last_outputs
    fit = results.attrs
    panel.lines.append(
        f"Effect: {fit['effect']:.4g} (p = {fit['p_value']:.3g})    Bandwidth: {fit['bandwidth']:.4g} ({fit['bandwidth_source']})    "
        f"N: {fit['left']} left, {fit['right']} right"
    )
    if fit["fuzzy"]:
        panel.lines.append(
            f"First stage jump in {settings.get('treatment')}: {fit['first_stage']:.3g} (F = {fit['first_stage_f']:.3g})    "
            f"Standard errors: {clusters.errors_text(fit)}"
        )
    else:
        panel.lines.append(f"Kernel: {fit['kernel']}, order {fit['order']}    Standard errors: {clusters.errors_text(fit)}")
    if not panel.note:
        panel.note = " ".join(fit["warnings"] + (fit["cluster"]["warnings"] if fit.get("cluster") else []))

    show = settings.get("show", SHOW[0])
    if show == "Chart":
        panel.preview_title = "RD Plot"
        panel.chart = lambda screen, area: draw_plot(screen, area, panel.table_font, fit, settings)
    else:
        panel.preview_title = "Results"
        panel.preview = results if show == "Estimate" else checks


# Bin means as dots, the fitted curve on each side within the bandwidth and a dashed line at the cutoff
def draw_plot(screen, area, font, fit, settings):
    plot = fit["plot"]
    common.draw_lines(
        screen, area, font,
        [(plot["left_x"], plot["left_y"], colours.CHART_LINE, None), (plot["right_x"], plot["right_y"], colours.CHART_LINE, None)],
        settings.get("running"), settings.get("dependent"),
        vertical=fit["cutoff"], points=(plot["bin_x"], plot["bin_y"]),
    )


# The regressors of a local polynomial: the jump (1 above the cutoff), then each power of R - cutoff
# on its own and times the jump, so each side has its own curve
def polynomial(r, order):
    above = (r >= 0).astype(float)
    columns, names = [above], ["Jump at cutoff"]
    for power in range(1, order + 1):
        label = "(R - cutoff)" if power == 1 else f"(R - cutoff)^{power}"
        columns += [r ** power, above * r ** power]
        names += [label, f"above x {label}"]
    return np.column_stack(columns), names


# A weighted local polynomial regression around the cutoff. y is the outcome and d the treatment for
# a fuzzy design (None for sharp), in which case the jump in y is divided by the jump in d by 2SLS
# with being above the cutoff as the instrument. Returns a dict with the coefficients and their
# covariance, the rows used and, for fuzzy, the first stage. cluster_values is a one-column table of
# the field to cluster by, or None
def local_fit(r, y, h, kernel, order, extra=None, d=None, robust=True, cluster_values=None):
    weights = WEIGHTS[kernel](np.abs(r) / h)
    inside = (np.abs(r) <= h) & (weights > 0)
    left, right = int((inside & (r < 0)).sum()), int((inside & (r >= 0)).sum())
    if min(left, right) < order + 2:
        raise ValueError(f"Only {left} rows left and {right} right of the cutoff within the bandwidth {h:.4g}; widen it")
    root = np.sqrt(weights[inside])
    terms, names = polynomial(r[inside], order)
    z = np.column_stack([np.ones(inside.sum()), terms] + ([extra[inside]] if extra is not None else []))
    common.check_rank(z, "local regressors and controls")
    z = z * root[:, None]
    target = y[inside] * root
    n, parameters = z.shape
    df_resid = n - parameters

    groups, cluster, df_t = None, None, df_resid
    if cluster_values is not None:
        groups, cluster, df_t = common.error_groups(cluster_values[inside], cluster_values.columns[0], parameters, df_resid)
    result = {"left": left, "right": right, "names": names, "inside": inside, "cluster": cluster, "df": df_t}
    if d is None:
        beta, cov, resid = common.least_squares(z, target, df_resid, robust, groups)
    else:
        x = z.copy()
        x[:, 1] = d[inside] * root
        beta, resid, xhat = common.two_stage(target, x, z)
        cov = common.covariance(xhat, resid, df_resid, robust, groups)
        first, first_cov, first_resid = common.least_squares(z, d[inside] * root, df_resid, robust, groups)
        result["first_stage"] = first[1]
        result["first_stage_f"] = first[1] ** 2 / first_cov[1, 1]
    result.update(beta=beta, cov=cov, effect=beta[1], se=float(np.sqrt(cov[1, 1])))
    result["p_value"] = float(2 * stats.t.sf(abs(result["effect"] / result["se"]), df_t))
    return result


# Imbens & Kalyanaraman's (2012) bandwidth for a local linear fit at the cutoff, which balances the
# bias from the curvature of Y near the cutoff against the variance from using few rows
def ik_bandwidth(r, y, kernel):
    n = len(r)
    h1 = 1.84 * r.std(ddof=1) * n ** -0.2
    left, right = (r < 0) & (r >= -h1), (r >= 0) & (r <= h1)
    if left.sum() < 2 or right.sum() < 2:
        raise ValueError("Too few rows near the cutoff to choose a bandwidth; type one, e.g. bandwidth = 5")
    density = (left.sum() + right.sum()) / (2 * n * h1)
    variance = (((y[left] - y[left].mean()) ** 2).sum() + ((y[right] - y[right].mean()) ** 2).sum()) / (left.sum() + right.sum())

    # The third derivative from a global cubic between the medians of R either side of the cutoff
    middle = (r >= np.median(r[r < 0])) & (r <= np.median(r[r >= 0]))
    rm = r[middle]
    cubic = np.column_stack([np.ones(middle.sum()), rm >= 0, rm, rm ** 2, rm ** 3])
    third = 6 * np.linalg.lstsq(cubic, y[middle], rcond=None)[0][4]
    if third == 0:
        third = 1e-12

    # The second derivative on each side, from a local quadratic within a pilot bandwidth
    curvature, regularise = [], 0.0
    for side in [r < 0, r >= 0]:
        pilot = 3.56 * (variance / (density * third ** 2)) ** (1 / 7) * side.sum() ** (-1 / 7)
        near = side & (np.abs(r) <= pilot)
        if near.sum() < 4:
            raise ValueError("Too few rows near the cutoff to choose a bandwidth; type one, e.g. bandwidth = 5")
        rn = r[near]
        quadratic = np.column_stack([np.ones(near.sum()), rn, rn ** 2])
        curvature.append(2 * np.linalg.lstsq(quadratic, y[near], rcond=None)[0][2])
        regularise += 2160 * variance / (near.sum() * pilot ** 4)
    bias = (curvature[1] - curvature[0]) ** 2 + regularise
    return float(IK_CONSTANTS[kernel] * (2 * variance / density / bias) ** 0.2 * n ** -0.2)


# McCrary's (2008) test for a jump in the density of R at the cutoff, which would suggest units
# sorted themselves to one side. R is put in narrow bins, a local linear fit to the bin heights on
# each side gives the density just either side, and theta is the log of their ratio
def mccrary(r):
    n = len(r)
    width = 2 * r.std(ddof=1) / np.sqrt(n)
    # When R is rounded, e.g. to 0.1, and the cutoff is on its grid, each bin holds a whole number of
    # grid points, else bins with one more point either side of the cutoff would look like a jump
    grid = np.diff(np.unique(r)).min() if len(np.unique(r)) > 1 else 0
    if grid > 0 and np.allclose(r / grid, np.round(r / grid)):
        points = max(1, round(width / grid))
        index = np.round(r / grid).astype(int) // points
        width = points * grid
    else:
        index = np.floor(r / width).astype(int)
    bins = np.arange(index.min(), index.max() + 1)
    heights = np.bincount(index - index.min()) / (n * width)
    mids = (bins + 0.5) * width
    if (mids < 0).sum() < 5 or (mids > 0).sum() < 5:
        raise ValueError("too few bins on one side")

    # Bandwidth from a quartic fitted to each side's bin heights
    sides = [mids < 0, mids > 0]
    bandwidths = []
    for side in sides:
        x, h = mids[side], heights[side]
        quartic = np.column_stack([x ** p for p in range(5)])
        beta, resid = panel_data.ols(quartic, h)
        second = 2 * beta[2] + 6 * beta[3] * x + 12 * beta[4] * x ** 2
        spread = np.abs(x).max() + width / 2
        bandwidths.append(3.348 * (resid @ resid / len(x) * spread / max((second ** 2).sum(), 1e-12)) ** 0.2)
    bandwidth = float(np.mean(bandwidths))

    densities = []
    for side in sides:
        near = side & (np.abs(mids) < bandwidth)
        if near.sum() < 3:
            raise ValueError("too few bins within the bandwidth")
        root = np.sqrt(1 - np.abs(mids[near]) / bandwidth)
        x = np.column_stack([np.ones(near.sum()), mids[near]]) * root[:, None]
        densities.append(np.linalg.lstsq(x, heights[near] * root, rcond=None)[0][0])
    if min(densities) <= 0:
        raise ValueError("the estimated density is not positive at the cutoff")
    theta = float(np.log(densities[1]) - np.log(densities[0]))
    se = float(np.sqrt(24 / 5 / (n * bandwidth) * (1 / densities[1] + 1 / densities[0])))
    return theta, se, float(2 * stats.norm.sf(abs(theta / se)))


def check_row(check, setting, estimate, se, p_value, rows, verdict):
    return {"check": check, "setting": setting, "estimate": estimate, "std_error": se, "p_value": p_value, "rows": rows, "verdict": verdict}


# The robustness checks: the estimate at other bandwidths and the other order, placebo cutoffs at the
# median of R on each side (where there should be no jump), each control as the outcome (it should
# not jump either, if units just either side are alike) and the density test
def robustness(r, y, d, extra, controls, used, values, h, kernel, order, robust, cluster_values):
    rows = []
    for share in BANDWIDTH_SHARES:
        try:
            fit = local_fit(r, y, h * share, kernel, order, extra, d, robust, cluster_values)
            rows.append(check_row("Bandwidth", f"h = {h * share:.4g} (x{share:g})", fit["effect"], fit["se"], fit["p_value"],
                                  fit["left"] + fit["right"], "Chosen bandwidth" if share == 1 else "Should be close to the chosen estimate"))
        except ValueError as error:
            rows.append(check_row("Bandwidth", f"h = {h * share:.4g} (x{share:g})", np.nan, np.nan, np.nan, 0, str(error)))
    other = 2 if order == 1 else 1
    try:
        fit = local_fit(r, y, h, kernel, other, extra, d, robust, cluster_values)
        rows.append(check_row("Order", f"order {other}", fit["effect"], fit["se"], fit["p_value"], fit["left"] + fit["right"], "Should be close to the chosen estimate"))
    except ValueError as error:
        rows.append(check_row("Order", f"order {other}", np.nan, np.nan, np.nan, 0, str(error)))

    for name, side in [("left", r < 0), ("right", r >= 0)]:
        placebo = float(np.median(r[side]))
        try:
            fit = local_fit(r[side] - placebo, y[side], h, kernel, order, robust=robust,
                            cluster_values=cluster_values[side] if cluster_values is not None else None)
            verdict = "Jump where there should be none: the effect may be a feature of Y's curve" if fit["p_value"] < common.LEVEL else "No jump, as expected"
            rows.append(check_row("Placebo cutoff", f"{name}: {placebo + values['cutoff']:.4g}", fit["effect"], fit["se"], fit["p_value"], fit["left"] + fit["right"], verdict))
        except ValueError as error:
            rows.append(check_row("Placebo cutoff", name, np.nan, np.nan, np.nan, 0, str(error)))

    for column in controls:
        if not dummies.is_numeric(used[column]):
            continue
        try:
            fit = local_fit(r, used[column].to_numpy(float), h, kernel, order, robust=robust, cluster_values=cluster_values)
            verdict = "Imbalanced: units either side differ, so the design may be invalid" if fit["p_value"] < common.LEVEL else "Balanced"
            rows.append(check_row("Covariate balance", column, fit["effect"], fit["se"], fit["p_value"], fit["left"] + fit["right"], verdict))
        except ValueError as error:
            rows.append(check_row("Covariate balance", column, np.nan, np.nan, np.nan, 0, str(error)))

    try:
        theta, se, p = mccrary(r)
        verdict = "Density jumps: units may have sorted around the cutoff" if p < common.LEVEL else "No jump in the density of R"
        rows.append(check_row("McCrary density", "log difference", theta, se, p, len(r), verdict))
    except ValueError as error:
        rows.append(check_row("McCrary density", "log difference", np.nan, np.nan, np.nan, len(r), f"Could not be computed: {error}"))
    return pd.DataFrame(rows)


# Evenly spaced bins of R on each side of the cutoff with the mean of Y in each, and the fitted curve
# on each side within the bandwidth (without controls) for the chart
def bin_means(r, y, h, kernel, order, values):
    rows = []
    for side, low, high in [("left", r.min(), 0.0), ("right", 0.0, r.max())]:
        edges = np.linspace(low, high, values["bins"] + 1)
        mask = r < 0 if side == "left" else r >= 0
        place = np.clip(np.searchsorted(edges, r[mask], side="right") - 1, 0, values["bins"] - 1)
        for b in range(values["bins"]):
            chosen = y[mask][place == b]
            rows.append({
                "side": side, "bin_low": edges[b] + values["cutoff"], "bin_high": edges[b + 1] + values["cutoff"],
                "bin_mid": (edges[b] + edges[b + 1]) / 2 + values["cutoff"], "rows": len(chosen),
                "mean_y": chosen.mean() if len(chosen) else np.nan,
            })
    bins = pd.DataFrame(rows)

    plot = {"bin_x": list(bins["bin_mid"]), "bin_y": list(bins["mean_y"])}
    for side, low, high in [("left", max(-h, r.min()), 0.0), ("right", 0.0, min(h, r.max()))]:
        mask = (r < 0) if side == "left" else (r >= 0)
        mask = mask & (np.abs(r) <= h)
        root = np.sqrt(WEIGHTS[kernel](np.abs(r[mask]) / h))
        powers = np.column_stack([r[mask] ** p for p in range(order + 1)])
        beta = np.linalg.lstsq(powers * root[:, None], y[mask] * root, rcond=None)[0]
        grid = np.linspace(low, high, CURVE_POINTS)
        plot[f"{side}_x"] = list(grid + values["cutoff"])
        plot[f"{side}_y"] = list(np.column_stack([grid ** p for p in range(order + 1)]) @ beta)
    return bins, plot


# Estimates the jump in Y at the cutoff of the running variable, by a local polynomial regression on
# each side within the bandwidth, weighted by the kernel. Sharp if no treatment is chosen; fuzzy (the
# jump in Y over the jump in the treatment) if one is
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    dependent, running, treatment = settings.get("dependent"), settings.get("running"), settings.get("treatment")
    controls = settings.get("controls", [])
    kernel = settings.get("kernel", KERNELS[0])
    order = int(settings.get("order", ORDERS[0]))
    robust = settings.get("robust", True)
    values = parse(settings.get("text", DEFAULT_TEXT))
    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")
    if not running:
        raise ValueError("Choose the running variable (R), whose value crossing the cutoff sets treatment")
    common.check_fields(data, settings, ["dependent", "running", "treatment", "controls"])
    if not dummies.is_numeric(data[running]):
        raise ValueError(f"The running variable {running} must be numeric")

    fields = [dependent, running] + ([treatment] if treatment else []) + controls
    used = data.dropna(subset=fields + clusters.cluster_columns(data, settings))
    notes = []
    if len(used) < len(data):
        notes.append(f"Dropped {len(data) - len(used)} rows with missing values.")
    r = used[running].to_numpy(float) - values["cutoff"]
    y = used[dependent].to_numpy(float)
    if (r < 0).sum() < MIN_SIDE or (r >= 0).sum() < MIN_SIDE:
        raise ValueError(f"{(r < 0).sum()} rows below the cutoff {values['cutoff']:g} and {(r >= 0).sum()} at or above it; each side needs at least {MIN_SIDE}")
    d = None
    if treatment:
        d = used[treatment].to_numpy(float)
        if not np.isin(d, [0, 1]).all():
            notes.append(f"{treatment} is not 0/1, so the effect is per unit of {treatment}.")
    extra = None
    control_names, control_roles = [], []
    if controls:
        frame, control_roles = dummies.design_matrix(used, [(controls, "C")])
        extra, control_names = frame.to_numpy(float), list(frame.columns)
    cluster_values = used[[settings["cluster"]]] if settings.get("cluster") else None

    if values["bandwidth"] is None:
        h, source = ik_bandwidth(r, y, kernel), "Imbens-Kalyanaraman"
        if order > 1:
            notes.append("The automatic bandwidth is chosen for order 1; check the estimate at other bandwidths.")
        if treatment:
            notes.append("The automatic bandwidth is chosen for the jump in Y, not the fuzzy ratio.")
    else:
        h, source = values["bandwidth"], "typed"
    fit = local_fit(r, y, h, kernel, order, extra, d, robust, cluster_values)

    names = ["Intercept"] + ([treatment] if treatment else fit["names"][:1]) + fit["names"][1:] + control_names
    roles = ["Intercept", "Effect"] + ["Poly"] * (len(fit["names"]) - 1) + control_roles
    results = panel_data.coefficient_table(names, roles, fit["beta"], fit["cov"], fit["df"])
    if min(fit["left"], fit["right"]) < 30:
        notes.append(f"Only {min(fit['left'], fit['right'])} rows on one side within the bandwidth, so the estimate is imprecise.")
    if treatment and fit["first_stage_f"] < common.STRONG_F:
        notes.append(f"The jump in {treatment} at the cutoff is weak (F below {common.STRONG_F}), so the fuzzy estimate is unreliable.")

    checks = robustness(r, y, d, extra, controls, used, values, h, kernel, order, robust, cluster_values)
    bins, plot = bin_means(r, y, h, kernel, order, values)
    results.attrs = {
        "effect": fit["effect"], "p_value": fit["p_value"], "bandwidth": h, "bandwidth_source": source,
        "kernel": kernel, "order": order, "cutoff": values["cutoff"], "left": fit["left"], "right": fit["right"],
        "observations": fit["left"] + fit["right"], "fuzzy": bool(treatment),
        "first_stage": fit.get("first_stage"), "first_stage_f": fit.get("first_stage_f"),
        "robust": robust, "cluster": fit["cluster"],
        "plot": plot, "warnings": notes,
    }
    return [results, checks, bins]
