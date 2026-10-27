import re

import numpy as np
import pandas as pd

from engine import clusters, dummies, panel_data
from ui import colours
from .. import common


INPUTS = 1
OUTPUTS = 3
INPUT_SPECS = [
    ("Data", "for 2x2, a numeric Y with 0/1 fields for the treated group (G) and the post period (P); for TWFE or Event study, panel data with Y, a 0/1 treatment (D) that is 1 in the periods a unit is treated, the entity (E) and time (T) fields. Any controls (C) can be added"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, coefficient, std_error, t_stat, p_value and stars. The Effect rows are the treatment effect (2x2 and TWFE) or the effect at each period relative to treatment (Event study)"),
    ("Means", "the mean of Y in each group and period: the 2x2 table with its differences, or by period for the ever-treated and never-treated units"),
    ("Fitted data", "the rows used, plus fitted and residual columns (residual only for TWFE and Event study, after the fixed effects)"),
]

DESIGNS = ["2x2", "TWFE", "Event study"]
SHOW = ["Coefficients", "Means", "Chart"]
ROLES = {
    "2x2": [("Y", "dependent", False), ("G", "group", False), ("P", "post", False), ("C", "controls", True)],
    "TWFE": [("Y", "dependent", False), ("D", "treatment", False), ("E", "entity", False), ("T", "time", False), ("C", "controls", True)],
}
ROLES["Event study"] = ROLES["TWFE"]
FIELD_SETTINGS = ["dependent", "group", "post", "treatment", "entity", "time", "controls", "cluster"]
DEFAULT_TEXT = "leads = 4, lags = 4"


# Opens a panel to choose the design and give each field a role. Robust is on by default: HC1 errors
# for 2x2, and errors clustered by entity for TWFE and Event study
def on_click(node):
    design = node.settings.get("design", DESIGNS[0])
    return common.open_panel(node, "DiD", ROLES[design], show_model, robust_default=True)


def change_text(panel, text):
    panel.node.settings["text"] = text
    show_model(panel)


# Reads the number of periods before (leads) and after (lags) treatment the event study shows
def parse(text):
    values = {"leads": 4, "lags": 4}
    for part in re.split(r"[,;]", text or ""):
        if not part.strip():
            continue
        name, _, value = [p.strip().lower() for p in part.partition("=")]
        if name not in values:
            raise ValueError("Write the window as leads = 4, lags = 4")
        try:
            values[name] = int(value)
        except ValueError:
            raise ValueError(f"{name} must be a whole number, not {value}")
        if values[name] < 0:
            raise ValueError(f"{name} can't be below 0")
    if values["leads"] < 2:
        raise ValueError("leads must be at least 2, so there is a period before treatment to test besides the reference period")
    return values


def show_model(panel):
    node = panel.node
    settings = node.settings
    design = settings.get("design", DESIGNS[0])
    panel.roles = ROLES[design]
    panel.clear_options()
    panel.preview = None
    panel.chart = None
    panel.note = ""
    panel.message = ""
    if design == "Event study":
        if not panel.text_field:
            panel.add_text_field("Window", settings.get("text", DEFAULT_TEXT), change_text, blocked="", fill=True)
        try:
            parse(settings.get("text", DEFAULT_TEXT))
        except ValueError as error:
            panel.note = str(error)
    else:
        panel.text_field = None

    if design == "2x2":
        panel.lines = [
            f"Y: {common.names_text(settings, 'dependent')}    Treated group (G): {common.names_text(settings, 'group')}    "
            f"Post period (P): {common.names_text(settings, 'post')}",
        ]
    else:
        panel.lines = [
            f"Y: {common.names_text(settings, 'dependent')}    Treatment (D): {common.names_text(settings, 'treatment')}    "
            f"Entity (E): {common.names_text(settings, 'entity')}    Time (T): {common.names_text(settings, 'time')}",
        ]
    panel.lines[0] += f"    C: {common.names_text(settings, 'controls')}"
    common.add_choice(panel, "Design", "design", DESIGNS, DESIGNS[0])
    common.add_choice(panel, "Show", "show", SHOW, SHOW[0])
    if not common.add_role_options(panel, panel.roles, common.default_allowed(["C", "E", "T"])):
        return

    if not node.last_outputs:
        panel.message = "Choose the design and the fields for it, then run the workflow"
        return
    results, means, fitted = node.last_outputs
    fit = results.attrs
    if fit["design"] != design:
        panel.message = f"The last run was {fit['design']}; run the workflow again for {design}"
        return
    panel.lines += fit["lines"] + [f"Standard errors: {clusters.errors_text(fit)}"]
    if not panel.note:
        panel.note = " ".join(fit["warnings"] + (fit["cluster"]["warnings"] if fit.get("cluster") else []))

    show = settings.get("show", SHOW[0])
    if show == "Chart":
        panel.preview_title = "Event Study" if design == "Event study" else "Mean of Y by Period"
        panel.chart = lambda screen, area: draw_chart(screen, area, panel.table_font, results, means, settings)
    else:
        panel.preview_title = "Results"
        panel.preview = results if show == "Coefficients" else means


# The event study's coefficients with 95% confidence intervals, or the mean of Y over time for each group
def draw_chart(screen, area, font, results, means, settings):
    fit = results.attrs
    if fit["design"] == "Event study":
        plot = fit["plot"]
        common.draw_lines(
            screen, area, font, [(plot["time"], plot["estimate"], colours.CHART_LINE, "Effect")],
            "Periods relative to treatment", settings.get("dependent"),
            intervals=(plot["time"], plot["low"], plot["high"]), vertical=-0.5, horizontal=0.0,
        )
        return
    if fit["design"] == "2x2":
        series = []
        for group, label, colour in [(1.0, "Treated", colours.CHART_LINE), (0.0, "Control", colours.CHART_POINT)]:
            rows = means[(means["group"] == group)]
            series.append((list(rows["period"]), list(rows["mean_y"]), colour, label))
        common.draw_lines(screen, area, font, series, "Post period (P)", settings.get("dependent"), vertical=0.5)
        return
    series = [
        (fit["periods"], list(means["ever_treated"]), colours.CHART_LINE, "Ever treated"),
        (fit["periods"], list(means["never_treated"]), colours.CHART_POINT, "Never treated"),
    ]
    common.draw_lines(screen, area, font, series, settings.get("time"), settings.get("dependent"))


# Difference in differences with two groups and two periods: OLS of Y on G, P and G x P, whose
# coefficient is the effect. Robust gives HC1 errors; clustering by the unit that was treated, such
# as a state, is usually better
def two_by_two(data, settings, notes):
    dependent, group, post = settings.get("dependent"), settings.get("group"), settings.get("post")
    controls = settings.get("controls", [])
    if not group or not post:
        raise ValueError("Choose the 0/1 field marking the treated group (G) and the one marking the post period (P)")
    used = data.dropna(subset=[dependent, group, post] + controls + clusters.cluster_columns(data, settings))
    if len(used) < len(data):
        notes.append(f"Dropped {len(data) - len(used)} rows with missing values.")
    g = common.binary(used[group], group, "treated group (G)").to_numpy()
    p = common.binary(used[post], post, "post period (P)").to_numpy()
    y = used[dependent].to_numpy(float)

    cells = pd.crosstab(g, p)
    if cells.shape != (2, 2) or (cells.to_numpy() == 0).any():
        raise ValueError("Each of the four group and period combinations needs rows")
    frame, control_roles = dummies.design_matrix(used, [(controls, "C")])
    x = np.column_stack([np.ones(len(used)), g, p, g * p, frame.to_numpy(float)])
    common.check_rank(x, "regressors and controls")
    df_resid = len(used) - x.shape[1]
    groups, cluster, df_t = common.error_groups(used, settings.get("cluster"), x.shape[1], df_resid)
    beta, cov, resid = common.least_squares(x, y, df_resid, settings.get("robust", True), groups)
    names = ["Intercept", group, post, f"{group} x {post}"] + list(frame.columns)
    results = panel_data.coefficient_table(names, ["Intercept", "G", "P", "Effect"] + control_roles, beta, cov, df_t)

    rows = []
    for g_value in [0.0, 1.0]:
        for p_value in [0.0, 1.0]:
            chosen = y[(g == g_value) & (p == p_value)]
            rows.append({"group": g_value, "period": p_value, "mean_y": chosen.mean(), "rows": len(chosen)})
    means = pd.DataFrame(rows)
    m = means.set_index(["group", "period"])["mean_y"]
    change_treated, change_control = m[1, 1] - m[1, 0], m[0, 1] - m[0, 0]
    means = pd.concat([means, pd.DataFrame([
        {"group": "treated change", "period": "post - pre", "mean_y": change_treated, "rows": np.nan},
        {"group": "control change", "period": "post - pre", "mean_y": change_control, "rows": np.nan},
        {"group": "difference in differences", "period": "", "mean_y": change_treated - change_control, "rows": np.nan},
    ])], ignore_index=True)

    effect = results.iloc[3]
    lines = [f"Effect: {effect['coefficient']:.4g} (p = {effect['p_value']:.3g})    N: {len(used)}    "
             f"Treated change: {change_treated:.4g}    Control change: {change_control:.4g}"]
    fitted = used.copy()
    fitted["fitted"] = y - resid
    fitted["residual"] = resid
    return results, means, fitted, cluster, lines


# Checks and prepares panel data for TWFE and the event study: the rows used, each row's entity code,
# its period's position, the treatment and the first period each entity is treated (NaN if never)
def panel_rows(data, settings, notes):
    dependent, treatment, entity, time = (settings.get(k) for k in ["dependent", "treatment", "entity", "time"])
    controls = settings.get("controls", [])
    if not treatment or not entity or not time:
        raise ValueError("Choose the 0/1 treatment (D), the entity field (E) and the time field (T)")
    used = data.dropna(subset=[dependent, treatment, entity, time] + controls + clusters.cluster_columns(data, settings))
    if len(used) < len(data):
        notes.append(f"Dropped {len(data) - len(used)} rows with missing values.")
    if used.duplicated([entity, time]).any():
        row = used[used.duplicated([entity, time])].iloc[0]
        raise ValueError(f"{entity} {row[entity]} has more than one row for {time} {row[time]}. Each entity should have one row per period")
    used = used.sort_values([entity, time])
    d = common.binary(used[treatment], treatment, "treatment (D)")

    periods = np.sort(used[time].unique())
    position = pd.Series(np.searchsorted(periods, used[time]), index=used.index)
    first = position[d == 1].groupby(used.loc[d == 1, entity]).min()
    start = used[entity].map(first)
    switched_off = ((d == 0) & (position > start)).groupby(used[entity]).any()
    if d.sum() == 0:
        raise ValueError(f"{treatment} is never 1, so there is no treatment to estimate")
    if d.sum() == len(d):
        raise ValueError(f"{treatment} is always 1, so there is no comparison group")
    return used, d.to_numpy(), position.to_numpy(), start.to_numpy(float), periods, switched_off


# Two-way fixed effects: entity averages are taken away from Y, the treatment, the controls and a dummy
# for each period but the first, which by Frisch-Waugh-Lovell gives the same slopes as entity and
# period dummies even with unbalanced panels. Returns the slopes, their covariance, the residuals,
# the degrees of freedom for t tests and the cluster summary
def twfe(used, settings, y, regressors, position, periods, codes):
    time_dummies = (position[:, None] == np.arange(1, len(periods))[None, :]).astype(float)
    x = common.demean(np.column_stack([regressors, time_dummies]), codes)
    yd = common.demean(y, codes)
    kept = panel_data.independent_columns(x, panel_data.rms(np.column_stack([regressors, time_dummies])))
    x = x[:, kept]
    df_resid = len(y) - len(kept) - (codes.max() + 1)
    if df_resid <= 0:
        raise ValueError("Too few rows for the entity and period effects")
    column = settings.get("cluster") or (settings["entity"] if settings.get("robust", True) else None)
    groups, cluster, df_t = common.error_groups(used, column, len(kept), df_resid)
    beta, cov, resid = common.least_squares(x, yd, df_resid, False, groups)
    return kept, beta, cov, resid, df_t, cluster


# The share of the treated rows that TWFE gives a negative weight (de Chaisemartin & D'Haultfoeuille
# 2020). TWFE's estimate is a weighted sum of each treated row's effect, the weights being the treatment
# left over after the entity and period effects. With staggered timing some can be negative, so the
# estimate can even have the opposite sign to every row's effect
def negative_weights(d, position, periods, codes):
    time_dummies = (position[:, None] == np.arange(1, len(periods))[None, :]).astype(float)
    x = common.demean(time_dummies, codes)
    left = common.demean(d, codes)
    left = left - x @ np.linalg.lstsq(x, left, rcond=None)[0]
    weights = left[d == 1] / left[d == 1].sum()
    return int((weights < 0).sum()), int((d == 1).sum()), float(weights[weights < 0].sum())


# The mean of Y in each period for the units ever treated and those never treated
def trend_means(used, y, start, periods, position):
    ever = ~np.isnan(start)
    rows = []
    for p, period in enumerate(periods):
        at = position == p
        rows.append({
            "period": period,
            "ever_treated": y[at & ever].mean() if (at & ever).any() else np.nan,
            "never_treated": y[at & ~ever].mean() if (at & ~ever).any() else np.nan,
            "treated_rows": int((at & ever).sum()),
            "never_rows": int((at & ~ever).sum()),
        })
    return pd.DataFrame(rows)


# Periods as numbers for the chart: the time values if numeric, else their positions
def period_axis(periods):
    return [float(p) for p in periods] if pd.api.types.is_numeric_dtype(periods) else list(range(len(periods)))


def timing_notes(start, switched_off, notes):
    cohorts = np.unique(start[~np.isnan(start)])
    if switched_off.any():
        notes.append(f"{int(switched_off.sum())} entities stop being treated after starting; TWFE then mixes the effects of starting and stopping.")
    if np.isnan(start).sum() == 0:
        notes.append("Every entity is treated at some point, so later-treated units are the only comparison.")
    return cohorts


def twfe_design(data, settings, notes):
    used, d, position, start, periods, switched_off = panel_rows(data, settings, notes)
    codes, counts = panel_data.entity_codes(used, settings["entity"])
    y = used[settings["dependent"]].to_numpy(float)
    frame, control_roles = dummies.design_matrix(used, [(settings.get("controls", []), "C")])
    regressors = np.column_stack([d, frame.to_numpy(float)])
    kept, beta, cov, resid, df_t, cluster = twfe(used, settings, y, regressors, position, periods, codes)
    if 0 not in kept:
        raise ValueError("The treatment doesn't vary within entities and periods, so its effect can't be estimated")
    shown = [i for i, k in enumerate(kept) if k < regressors.shape[1]]
    names = [settings["treatment"]] + list(frame.columns)
    roles = ["Effect"] + control_roles
    results = panel_data.coefficient_table(
        [names[kept[i]] for i in shown], [roles[kept[i]] for i in shown], beta[shown], cov[np.ix_(shown, shown)], df_t
    )

    cohorts = timing_notes(start, switched_off, notes)
    lines = [f"Effect: {beta[0]:.4g} (p = {results.iloc[0]['p_value']:.3g})    N: {len(used)}    "
             f"Entities: {codes.max() + 1}    Periods: {len(periods)}    Treatment start times: {len(cohorts)}"]
    if len(cohorts) > 1:
        negative, treated, total = negative_weights(d, position, periods, codes)
        lines.append(f"Treated rows with a negative TWFE weight: {negative} of {treated} (sum {total:.3g})")
        notes.append(
            "Treatment starts at different times, so TWFE compares later-treated units with earlier-treated ones and "
            "is biased if effects change over time. Check the Event study, or use Callaway-Sant'Anna or Sun-Abraham estimators."
        )
    means = trend_means(used, y, start, periods, position)
    fitted = used.copy()
    fitted["residual"] = resid
    attrs = {"periods": period_axis(periods)}
    return results, means, fitted, cluster, lines, attrs


# The event study: TWFE with a dummy for each period relative to when an entity's treatment starts,
# from leads before to lags after, the ends binned (so -leads means that many periods before or
# earlier). The period just before treatment (-1) is left out as the reference, and never-treated
# units have every dummy 0. Leads should be near 0 if treated and control units were on parallel
# trends before treatment, which the joint test of the leads checks
def event_study(data, settings, notes):
    window = parse(settings.get("text", DEFAULT_TEXT))
    used, d, position, start, periods, switched_off = panel_rows(data, settings, notes)
    if switched_off.any():
        raise ValueError("The event study needs treatment to stay on once it starts, but some entities stop being treated")
    codes, counts = panel_data.entity_codes(used, settings["entity"])
    y = used[settings["dependent"]].to_numpy(float)
    relative = np.clip(position - start, -window["leads"], window["lags"])
    times = [k for k in range(-window["leads"], window["lags"] + 1) if k != -1]
    events = np.column_stack([(relative == k).astype(float) for k in times])
    frame, control_roles = dummies.design_matrix(used, [(settings.get("controls", []), "C")])
    regressors = np.column_stack([events, frame.to_numpy(float)])
    kept, beta, cov, resid, df_t, cluster = twfe(used, settings, y, regressors, position, periods, codes)

    names = [f"t{k:+d}" if k else "t0" for k in times] + list(frame.columns)
    roles = ["Lead" if k < 0 else "Effect" for k in times] + control_roles
    dropped = [names[i] for i in range(len(times)) if i not in kept]
    if dropped:
        notes.append(f"Left out {', '.join(dropped)}: no rows, or collinear with the period effects (e.g. with no never-treated units).")
    shown = [i for i, k in enumerate(kept) if k < regressors.shape[1]]
    results = panel_data.coefficient_table(
        [names[kept[i]] for i in shown], [roles[kept[i]] for i in shown], beta[shown], cov[np.ix_(shown, shown)], df_t
    )
    results.insert(2, "relative_time", [times[kept[i]] if kept[i] < len(times) else np.nan for i in shown])

    leads = [i for i, k in enumerate(kept) if k < len(times) and times[k] < -1]
    lines = [f"N: {len(used)}    Entities: {codes.max() + 1}    Periods: {len(periods)}    Window: {window['leads']} leads, {window['lags']} lags (ends binned)"]
    if leads:
        f, p = common.wald(beta, cov, leads, df_t)
        verdict = "rejected, so the trends before treatment differ" if p < common.LEVEL else "not rejected"
        lines.append(f"Pre-trends (all leads = 0): F = {f:.3g}, p = {p:.3g}, {verdict}")
    posts = [i for i, k in enumerate(kept) if k < len(times) and times[k] >= 0]
    if posts:
        average = float(np.mean(beta[posts]))
        lines[-1] += f"    Average effect after treatment: {average:.4g}"

    cohorts = timing_notes(start, switched_off, notes)
    if len(cohorts) > 1:
        notes.append(
            "Treatment starts at different times, so these TWFE event study estimates can mix in effects from other "
            "periods (Sun & Abraham 2021). Treat them with caution if effects differ between start times."
        )
    estimates = results[results["relative_time"].notna()]
    critical = 1.96
    plot_times = list(estimates["relative_time"]) + [-1]
    order = np.argsort(plot_times)
    plot = {
        key: [values[i] for i in order] for key, values in {
            "time": plot_times,
            "estimate": list(estimates["coefficient"]) + [0.0],
            "low": list(estimates["coefficient"] - critical * estimates["std_error"]) + [0.0],
            "high": list(estimates["coefficient"] + critical * estimates["std_error"]) + [0.0],
        }.items()
    }
    means = trend_means(used, y, start, periods, position)
    fitted = used.copy()
    fitted["relative_time"] = np.where(np.isnan(start), np.nan, position - start)
    fitted["residual"] = resid
    return results, means, fitted, cluster, lines, {"plot": plot, "periods": period_axis(periods)}


# Difference in differences in the chosen design: the classic 2x2, two-way fixed effects for panel
# data, or an event study of the effect in each period around the start of treatment
def run(inputs, settings):
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    design = settings.get("design", DESIGNS[0])
    if not settings.get("dependent"):
        raise ValueError("Choose a dependent variable (Y)")
    keys = [key for letter, key, many in ROLES[design]]
    common.check_fields(data, settings, keys)
    notes = []
    attrs = {}
    if design == "2x2":
        results, means, fitted, cluster, lines = two_by_two(data, settings, notes)
    elif design == "TWFE":
        results, means, fitted, cluster, lines, attrs = twfe_design(data, settings, notes)
    else:
        results, means, fitted, cluster, lines, attrs = event_study(data, settings, notes)
    results.attrs = dict(
        attrs, design=design, lines=lines, warnings=notes, cluster=cluster,
        robust=settings.get("robust", True), observations=len(fitted),
    )
    return [results, means, fitted]
