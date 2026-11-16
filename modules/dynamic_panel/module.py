import numpy as np
import pandas as pd
from scipy import stats

from engine import estimates, roles
from modules.arimax.module import read_time


INPUTS = 1
OUTPUTS = 2
OUTPUT_TYPES = ["model", "table"]
INPUT_SPECS = [
    ("Data", "a panel in long form: one row per entity (E) and period (T), a numeric Y, numeric X fields taken as exogenous, "
             "and numeric N fields that may be endogenous (affected by this period's shock)"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable (L1.y is last period's Y), coefficient, std_error (robust; Windmeijer-"
                     "corrected for two-step), z_stat, p_value and stars"),
    ("Tests", "the Arellano-Bond tests for first and second order autocorrelation in the differenced errors, Hansen's (and for "
              "difference GMM Sargan's) test of the overidentifying restrictions, a Wald test of all coefficients, and counts "
              "of entities, rows and instruments, each with a plain-words result"),
]

ESTIMATORS = [("difference", "Difference (Arellano-Bond)"), ("system", "System (Blundell-Bond)")]
STEPS = [("two", "Two-step"), ("one", "One-step")]
SWITCH = [(False, "Off"), (True, "On")]
LEVEL = 0.05

SPEC = roles.Spec(
    "Dynamic Panel",
    [
        roles.Role("Y", "dependent"),
        roles.Role("X", "independent", many=True),
        roles.Role("N", "endogenous", many=True),
        roles.Role("E", "entity", kinds="any"),
        roles.Role("T", "time", kinds="any"),
    ],
    choices=[roles.Choice("estimator", "GMM", ESTIMATORS), roles.Choice("steps", "Steps", STEPS),
             roles.Choice("collapse", "Collapse", SWITCH), roles.Choice("time_effects", "Time effects", SWITCH, default=True)],
    numbers=[roles.Number("lags", "Lags of Y", range(1, 4), 1), roles.Number("max_lag", "Deepest instrument lag (0 = all)", [0, 2, 3, 4, 5, 6, 8], 0)],
    hint="Choose Y, the entity (E) and time (T) fields and any X or N fields, then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings()


def on_click(node):
    return roles.open_panel(node, SPEC)


class Entity:
    # One entity's rows for the GMM: regressors, outcome, instruments and the one-step weight,
    # plus which rows belong to the differenced equation and their periods
    def __init__(self, x, y, z, h, is_diff, periods):
        self.x, self.y, self.z, self.h, self.is_diff, self.periods = x, y, z, h, is_diff, periods


# Puts each field on a grid of entity by period, with NaN where a row is missing
def to_grid(data, settings, fields):
    entity, time = settings["entity"], settings["time"]
    times = read_time(data[time])
    if times.isna().any():
        data, times = data[times.notna()], times[times.notna()]
    periods = np.sort(times.unique())
    period_code = pd.Series(np.searchsorted(periods, times), index=data.index)
    entity_names, entity_code = np.unique(data[entity].astype(str), return_inverse=True)
    duplicated = pd.DataFrame({"e": entity_code, "t": period_code.to_numpy()}).duplicated()
    if duplicated.any():
        raise ValueError(f"Some {entity} and {time} pairs repeat; each entity needs one row per period")
    grids = {}
    for field in fields:
        grid = np.full((len(entity_names), len(periods)), np.nan)
        grid[entity_code, period_code.to_numpy()] = data[field].to_numpy(float)
        grids[field] = grid
    return grids, entity_names, periods


def lagged(grid, lag):
    if lag == 0:
        return grid
    out = np.full_like(grid, np.nan)
    out[:, lag:] = grid[:, :-lag]
    return out


# Builds every entity's stacked rows: the differenced equation for each usable period and, for
# system GMM, the levels equation too, with their instruments. Missing instruments are zero
def build(grids, settings, periods):
    y = grids[settings["dependent"]]
    exogenous = settings.get("independent") or []
    endogenous = settings.get("endogenous") or []
    lags = settings["lags"]
    system = settings["estimator"] == "system"
    collapse = bool(settings["collapse"])
    deepest = settings["max_lag"] or None
    n_entities, n_periods = y.shape

    # Regressors in levels, and the outcome
    level_regressors = [lagged(y, lag) for lag in range(1, lags + 1)] + [grids[f] for f in exogenous + endogenous]
    names = [f"L{lag}.{settings['dependent']}" for lag in range(1, lags + 1)] + exogenous + endogenous
    term_roles = ["Lag"] * lags + ["X"] * len(exogenous) + ["N"] * len(endogenous)
    dummies = []
    if settings["time_effects"]:
        dummies = list(range(n_periods))
    # Which rows can enter each equation: Y and its lags, and every regressor and its last value
    diff_ok = np.isfinite(y) & np.isfinite(lagged(y, 1))
    for lag in range(1, lags + 1):
        diff_ok &= np.isfinite(lagged(y, lag + 1))
    for field in exogenous + endogenous:
        diff_ok &= np.isfinite(grids[field]) & np.isfinite(lagged(grids[field], 1))
    level_ok = np.isfinite(y)
    for regressor in level_regressors:
        level_ok &= np.isfinite(regressor)
    if system:
        # A levels row's instrument is last period's change in Y, so it needs Y two periods back
        level_ok &= np.isfinite(lagged(y, 2))
    else:
        level_ok[:] = False

    # The instrument columns: GMM-style ones for Y and each N field (levels at t-2 and before in
    # the differenced equation; the change at t-1 in the levels equation), then IV-style ones
    gmm_fields = [y] + [grids[f] for f in endogenous]
    deepest_lag = deepest or n_periods
    diff_periods = [t for t in range(n_periods) if diff_ok[:, t].any()]
    level_periods = [t for t in range(n_periods) if level_ok[:, t].any()]
    columns = {}

    def column(key):
        if key not in columns:
            columns[key] = len(columns)
        return columns[key]

    for g in range(len(gmm_fields)):
        for t in diff_periods:
            for s in range(max(0, t - deepest_lag), t - 1):
                column(("d", g, t - s) if collapse else ("d", g, t, s))
        for t in level_periods:
            column(("l", g) if collapse else ("l", g, t))
    iv_names = exogenous + [f"time{t}" for t in dummies] + (["const"] if system else [])
    for name in iv_names:
        column(("iv", name))

    entities = []
    for i in range(n_entities):
        rows_x, rows_y, rows_z, is_diff, row_periods = [], [], [], [], []
        for equation, ok_periods in (("d", [t for t in diff_periods if diff_ok[i, t]]), ("l", [t for t in level_periods if level_ok[i, t]])):
            for t in ok_periods:
                instruments = np.zeros(len(columns))
                for g, field in enumerate(gmm_fields):
                    if equation == "d":
                        for s in range(max(0, t - deepest_lag), t - 1):
                            value = field[i, s]
                            if np.isfinite(value):
                                instruments[columns[("d", g, t - s) if collapse else ("d", g, t, s)]] = value
                    else:
                        change = field[i, t - 1] - field[i, t - 2] if t >= 2 else np.nan
                        if np.isfinite(change):
                            instruments[columns[("l", g) if collapse else ("l", g, t)]] = change
                if equation == "d":
                    x_row = [r[i, t] - r[i, t - 1] for r in level_regressors]
                    x_row += [float(t == d) - float(t - 1 == d) for d in dummies]
                    x_row += [0.0] if system else []
                    for name in exogenous:
                        instruments[columns[("iv", name)]] = grids[name][i, t] - grids[name][i, t - 1]
                    for d in dummies:
                        instruments[columns[("iv", f"time{d}")]] = float(t == d) - float(t - 1 == d)
                    target = y[i, t] - y[i, t - 1]
                else:
                    x_row = [r[i, t] for r in level_regressors] + [float(t == d) for d in dummies] + [1.0]
                    for name in exogenous:
                        instruments[columns[("iv", name)]] = grids[name][i, t]
                    for d in dummies:
                        instruments[columns[("iv", f"time{d}")]] = float(t == d)
                    instruments[columns[("iv", "const")]] = 1.0
                    target = y[i, t]
                rows_x.append(x_row); rows_y.append(target); rows_z.append(instruments)
                is_diff.append(equation == "d"); row_periods.append(t)
        if not rows_y:
            continue
        is_diff = np.array(is_diff); row_periods = np.array(row_periods)
        h = one_step_weight(is_diff, row_periods)
        entities.append(Entity(np.array(rows_x, dtype=float), np.array(rows_y, dtype=float), np.array(rows_z, dtype=float),
                               h, is_diff, row_periods))

    names = names + [f"{settings['time']}={periods[d]}" for d in dummies] + (["Intercept"] if system else [])
    term_roles = term_roles + ["Time"] * len(dummies) + (["Intercept"] if system else [])
    return entities, names, term_roles, len(columns)


# The covariance of the errors, up to sigma squared, when they are independent: 2 on the diagonal
# and -1 between neighbouring periods for the differenced rows, 1 for levels rows, and between a
# differenced row at t and a levels row at s, 1 if s = t and -1 if s = t - 1 (xtabond2's h(3))
def one_step_weight(is_diff, periods):
    gap = periods[:, None] - periods[None, :]
    both_diff = is_diff[:, None] & is_diff[None, :]
    both_level = ~is_diff[:, None] & ~is_diff[None, :]
    h = np.where(both_diff, np.where(gap == 0, 2.0, np.where(np.abs(gap) == 1, -1.0, 0.0)), 0.0)
    h = np.where(both_level & (gap == 0), 1.0, h)
    cross = is_diff[:, None] & ~is_diff[None, :]
    h = np.where(cross, np.where(gap == 0, 1.0, np.where(gap == 1, -1.0, 0.0)), h)
    cross_t = ~is_diff[:, None] & is_diff[None, :]
    h = np.where(cross_t, np.where(gap == 0, 1.0, np.where(gap == -1, -1.0, 0.0)), h)
    return h


# Drops regressors that are all zero or a combination of the others, such as the time dummies of
# periods that can't be used. Returns the columns kept
def kept_columns(entities, names):
    x = np.vstack([e.x for e in entities])
    keep = []
    for j in range(x.shape[1]):
        trial = keep + [j]
        if np.any(x[:, j] != 0) and np.linalg.matrix_rank(x[:, trial]) == len(trial):
            keep.append(j)
    return keep


def gmm(entities, weight):
    zx = sum(e.z.T @ e.x for e in entities)
    zy = sum(e.z.T @ e.y for e in entities)
    a = np.linalg.pinv(zx.T @ weight @ zx)
    beta = a @ zx.T @ weight @ zy
    return beta, a, zx


# The Arellano-Bond test of order m: the m-th order autocorrelation of the differenced residuals,
# corrected for the estimates' own uncertainty. Independent errors make the first order one
# negative, so only the second order test matters for the model's validity
def ar_test(entities, residuals, m, a, zx, weight, covariance):
    total, squared = 0.0, 0.0
    wx = np.zeros(entities[0].x.shape[1])
    zuw = np.zeros(entities[0].z.shape[1])
    for e, u in zip(entities, residuals):
        w = np.zeros(len(u))
        diff = np.flatnonzero(e.is_diff)
        by_period = {e.periods[r]: u[r] for r in diff}
        for r in diff:
            w[r] = by_period.get(e.periods[r] - m, 0.0)
        u_d = np.where(e.is_diff, u, 0.0)
        value = w @ u_d
        total += value
        squared += value ** 2
        wx += w @ e.x
        zuw += e.z.T @ u * value
    variance = squared - 2 * wx @ a @ zx.T @ weight @ zuw + wx @ covariance @ wx
    if variance <= 0:
        return np.nan, np.nan
    z = total / np.sqrt(variance)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def test_row(test, null, statistic, df, p_value, result):
    return {"test": test, "null": null, "statistic": statistic, "df": df, "p_value": p_value, "result": result}


# Fits a dynamic panel model, Y on its own past values and X and N fields with an entity effect,
# by GMM. Fixed effects would be biased with a lagged Y (Nickell bias), so the differenced equation
# is instrumented with deeper lags of Y and N (Arellano and Bond 1991); system GMM adds the levels
# equation instrumented with past changes (Blundell and Bond 1998), which helps when Y is persistent
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y", "E", "T"))
    dependent = settings["dependent"]
    fields = list(dict.fromkeys([dependent] + (settings.get("independent") or []) + (settings.get("endogenous") or [])))
    overlap = set(settings.get("independent") or []) & set(settings.get("endogenous") or [])
    if overlap:
        raise ValueError(f"A field can be X or N, not both: {', '.join(overlap)}")
    notes = []
    grids, entity_names, periods = to_grid(data, settings, fields)
    if len(periods) < settings["lags"] + 2:
        raise ValueError(f"Dynamic panel GMM needs at least {settings['lags'] + 2} periods with {settings['lags']} lag(s) of Y")

    entities, names, term_roles, n_instruments = build(grids, settings, periods)
    if not entities:
        raise ValueError("No entity has enough consecutive periods to use")
    keep = kept_columns(entities, names)
    for e in entities:
        e.x = e.x[:, keep]
    names = [names[j] for j in keep]
    term_roles = [term_roles[j] for j in keep]
    k = len(names)
    used_instruments = np.vstack([e.z for e in entities]).any(axis=0)
    for e in entities:
        e.z = e.z[:, used_instruments]
    n_instruments = int(used_instruments.sum())
    if n_instruments < k:
        raise ValueError(f"{n_instruments} instruments for {k} coefficients; the model isn't identified. Use more periods or fewer terms")

    one_weight = np.linalg.pinv(sum(e.z.T @ e.h @ e.z for e in entities))
    beta1, a1, zx = gmm(entities, one_weight)
    residuals1 = [e.y - e.x @ beta1 for e in entities]
    omega = sum(np.outer(e.z.T @ u, e.z.T @ u) for e, u in zip(entities, residuals1))
    robust1 = a1 @ zx.T @ one_weight @ omega @ one_weight @ zx @ a1
    two_weight = np.linalg.pinv(omega)

    beta2, a2, _ = gmm(entities, two_weight)
    residuals2 = [e.y - e.x @ beta2 for e in entities]
    if settings["steps"] == "two":
        beta, a, residuals = beta2, a2, residuals2
        zu = sum(e.z.T @ u for e, u in zip(entities, residuals))
        # Windmeijer's (2005) correction for the two-step weight being estimated from beta1
        d = np.zeros((k, k))
        for j in range(k):
            change = sum(e.z.T @ (np.outer(e.x[:, j], u1) + np.outer(u1, e.x[:, j])) @ e.z for e, u1 in zip(entities, residuals1))
            d[:, j] = a @ zx.T @ two_weight @ change @ two_weight @ zu
        covariance = a + d @ a + a @ d.T + d @ robust1 @ d.T
        weight = two_weight
    else:
        beta, a, covariance, residuals, weight = beta1, a1, robust1, residuals1, one_weight

    table = estimates.coefficient_table(names, term_roles, beta, estimates.standard_errors(covariance))

    rows = []
    for m in (1, 2):
        z, p_value = ar_test(entities, residuals, m, a, zx, weight, covariance)
        if m == 1:
            result = ("Negative first order autocorrelation, as expected of differenced errors" if p_value < LEVEL
                      else "No first order autocorrelation found (expected some; not a problem on its own)")
        else:
            result = ("Second order autocorrelation: deeper lags are not valid instruments. Raise the lags of Y or start instruments deeper"
                      if p_value < LEVEL else "No second order autocorrelation: the instruments' timing is valid")
        rows.append(test_row(f"AR({m})", f"no order {m} autocorrelation in the differenced errors", z, "N(0, 1)", p_value, result))
    overid = n_instruments - k
    # Hansen's test always uses the two-step estimates, which is what it is built on
    zu2 = sum(e.z.T @ u for e, u in zip(entities, residuals2))
    if overid > 0:
        hansen = float(zu2 @ two_weight @ zu2)
        p_value = float(stats.chi2.sf(hansen, overid))
        result = "Instruments look valid" if p_value >= LEVEL else "Instruments rejected: some are correlated with the errors"
        if p_value > 0.99:
            result = "p is close to 1, a sign of too many instruments weakening the test. Collapse them or limit their lags"
        rows.append(test_row("Hansen J", "the instruments are valid (overidentifying restrictions hold)", hansen, f"chi2({overid})", p_value, result))
        if settings["estimator"] == "difference":
            zu1 = sum(e.z.T @ u for e, u in zip(entities, residuals1))
            sigma2 = sum(float(u @ u) for u in residuals1) / (2 * max(sum(len(u) for u in residuals1) - k, 1))
            sargan = float(zu1 @ one_weight @ zu1) / sigma2
            p_value = float(stats.chi2.sf(sargan, overid))
            rows.append(test_row("Sargan", "the instruments are valid (assumes homoscedastic errors)", sargan, f"chi2({overid})", p_value,
                                 "Instruments look valid" if p_value >= LEVEL else "Instruments rejected (not robust to heteroscedasticity; prefer Hansen)"))
    tested = [j for j, r in enumerate(term_roles) if r not in ("Intercept", "Time")]
    if tested:
        sub = beta[tested]
        wald = float(sub @ np.linalg.pinv(covariance[np.ix_(tested, tested)]) @ sub)
        rows.append(test_row("Wald", "every coefficient other than the intercept and time effects is 0", wald, f"chi2({len(tested)})",
                             float(stats.chi2.sf(wald, len(tested))), ""))
    n_rows = int(sum(e.is_diff.sum() for e in entities))
    rows.append(test_row("Counts", "entities, differenced rows, instruments", float(len(entities)), f"{n_rows} rows", np.nan,
                         f"{n_instruments} instruments for {len(entities)} entities"))
    tests = pd.DataFrame(rows)

    if n_instruments > len(entities):
        notes.append(f"{n_instruments} instruments is more than the {len(entities)} entities, which weakens the Hansen test and can bias "
                     "the estimates. Switch Collapse on or set the deepest instrument lag.")
    if settings["steps"] == "two" and len(entities) < 50:
        notes.append("Few entities: two-step estimates can be unreliable, even with Windmeijer's correction.")
    lag_rows = [j for j, r in enumerate(term_roles) if r == "Lag"]
    persistence = float(np.sum(beta[lag_rows]))
    if settings["estimator"] == "difference" and persistence > 0.8:
        notes.append(f"Y is very persistent (lags sum to {persistence:.2f}), which makes difference GMM's instruments weak; try system GMM.")
    ar2 = tests.loc[tests["test"] == "AR(2)", "p_value"].iloc[0]
    hansen_p = tests.loc[tests["test"] == "Hansen J", "p_value"].iloc[0] if (tests["test"] == "Hansen J").any() else np.nan
    summary = [
        f"{dict(ESTIMATORS)[settings['estimator']]}, {dict(STEPS)[settings['steps']].lower()}    Entities: {len(entities)}    "
        f"Rows: {n_rows}    Instruments: {n_instruments}",
        f"AR(2) p: {estimates.number_text(ar2)}    Hansen p: {estimates.number_text(hansen_p)}    "
        f"Errors: {'Windmeijer-corrected' if settings['steps'] == 'two' else 'robust'}",
    ]
    table.attrs = {"entities": len(entities), "observations": n_rows, "instruments": n_instruments,
                   "ar2_p": float(ar2) if np.isfinite(ar2) else np.nan, "summary": summary, "notes": notes}
    estimates.record(table, "dpanel", "Dynamic Panel", dependent, data, fields[1:], covariance, None, observations=n_rows)
    tests.attrs = {"summary": summary, "notes": notes}
    return [table, tests]
