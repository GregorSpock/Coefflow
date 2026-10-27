import numpy as np
import pandas as pd
from scipy import stats

from engine import clusters, dummies, fits, significance


# The estimators shared by the Fixed Effects and Random Effects modules, for panel data: several
# entities (e.g. countries) each seen over several periods. Each entity has its own effect u_i,
# a part of Y that doesn't change over time. Fixed effects removes it by comparing each entity with
# its own average (the within transform); random effects treats it as random and removes only a
# share theta of each entity's average (quasi-demeaning). Both follow Stata's xtreg: fixed effects
# reports an intercept equal to the average entity effect, and random effects uses the Swamy-Arora
# variance components. Settings are kept as:
#   dependent, independent, controls   Y, X and C, as in OLS
#   entity                             the field naming each entity (E), required
#   time                               the field naming each period (T), needed for time effects
#   time_effects                       True to add a dummy for each period (two-way effects)
#   robust                             True for errors clustered by entity
#   cluster                            a field to cluster the errors by instead
# Column scale below which a regressor counts as not varying, relative to its size
TOLERANCE = 1e-10


def model_variables(settings):
    return clusters.model_variables(settings)


# Checks the fields are chosen and usable, and returns the rows used (complete for every field)
def model_rows(data, settings):
    dependent = settings.get("dependent")
    entity = settings.get("entity")
    time = settings.get("time")
    if not dependent:
        raise ValueError("Choose a dependent variable (Y)")
    if not settings.get("independent"):
        raise ValueError("Choose at least one independent variable (X)")
    if not entity:
        raise ValueError("Choose the field naming each entity (E), such as a country or firm ID")
    if settings.get("time_effects") and not time:
        raise ValueError("Choose the time field (T) to add time effects")

    fields = model_variables(settings) + [c for c in [entity, time] if c]
    missing = [c for c in fields if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {', '.join(missing)}")
    if not dummies.is_numeric(data[dependent]):
        raise ValueError(f"{dependent} is not numeric, so it cannot be the dependent variable (Y)")
    roles = [entity, time] if time else [entity]
    if dependent in roles or set(roles) & set(settings.get("independent", []) + settings.get("controls", [])):
        raise ValueError("The entity (E) and time (T) fields can't also be Y, X or C")

    used = data.dropna(subset=fields + clusters.cluster_columns(data, settings))
    if time:
        repeated = used.duplicated([entity, time])
        if repeated.any():
            row = used[repeated].iloc[0]
            raise ValueError(
                f"{entity} {row[entity]} has more than one row for {time} {row[time]}. "
                "Each entity should have one row per period"
            )
    return used


# The regressors (not counting the intercept) and the role of each: X and C fields, text ones as
# dummies, then a dummy for each period but the first if time effects are on
def regressors(used, settings):
    groups = [(settings.get("independent", []), "X"), (settings.get("controls", []), "C")]
    terms, roles = dummies.design_matrix(used, groups)
    if settings.get("time_effects"):
        periods = dummies.terms(used[settings["time"]].astype(str), settings["time"])
        terms = pd.concat([terms, periods], axis=1)
        roles = roles + ["Time"] * periods.shape[1]
    return terms, roles


# Keeps the columns of a matrix that add something, in order: a column that doesn't vary, or that
# the columns before it already explain, is left out. Columns are compared in the units they had
# before transforming (scale, their root mean square by default), so rounding error left by removing
# the means doesn't count as variation. Returns the positions of the columns kept
def independent_columns(matrix, scale=None):
    if scale is None:
        scale = rms(matrix)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = matrix / scale
    kept = []
    for position in range(scaled.shape[1]):
        column = scaled[:, position]
        if kept:
            basis = scaled[:, kept]
            column = column - basis @ np.linalg.lstsq(basis, column, rcond=None)[0]
        if np.sqrt(column @ column) > np.sqrt(TOLERANCE) * np.sqrt(len(column)):
            kept.append(position)
    return kept


def rms(matrix):
    return np.sqrt((matrix ** 2).mean(axis=0))


# The entity each row belongs to as 0, 1, 2, ..., and the number of rows of each entity
def entity_codes(used, entity):
    codes = pd.factorize(used[entity])[0]
    return codes, np.bincount(codes)


# The average of each column within each entity, repeated on every row of that entity
def entity_means(values, codes):
    sums = np.zeros((codes.max() + 1,) + values.shape[1:])
    np.add.at(sums, codes, values)
    counts = np.bincount(codes).reshape((-1,) + (1,) * (values.ndim - 1))
    return (sums / counts)[codes]


def ols(x, y):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta, y - x @ beta


# The covariance of the coefficients from x (the regressors the fit used) and its residuals.
# Ordinary errors divide the squared residuals by df_resid. Clustered errors use the usual
# sandwich with Stata's small-sample correction G / (G - 1) * (N - 1) / (N - K)
def covariance(x, resid, df_resid, groups=None):
    bread = np.linalg.inv(x.T @ x)
    if groups is None:
        return bread * float(resid @ resid) / df_resid
    scores = np.zeros((groups.max() + 1, x.shape[1]))
    np.add.at(scores, groups, x * resid[:, None])
    count, rows, parameters = scores.shape[0], x.shape[0], x.shape[1]
    correction = count / (count - 1) * (rows - 1) / (rows - parameters)
    return correction * bread @ (scores.T @ scores) @ bread


# The groups to cluster by, the cluster summary shown in the panel and the degrees of freedom for
# the t tests. Robust on clusters by entity, as Stata's xtreg does: heteroscedasticity-robust (HC1)
# errors are not consistent for fixed effects with few periods, while clustering by entity is
# also robust to the errors of each entity being correlated over time
def error_groups(used, settings, parameters, df_resid):
    column = settings.get("cluster") or (settings["entity"] if settings.get("robust") else None)
    if not column:
        return None, None, df_resid
    summary = clusters.summarise(used[column])
    if summary["clusters"] < 2:
        raise ValueError(f"{column} has fewer than 2 values in the rows used, so it cannot be clustered by")
    info = dict(summary, column=column, warnings=clusters.warnings_for(summary, column, parameters))
    return pd.factorize(used[column])[0], info, summary["clusters"] - 1


def squared_correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1] ** 2)


# Stata's three R squared, from the slope part of the fit xb: within (changes over time inside each
# entity), between (differences between the entities' averages) and overall
def r_squared(y, xb, codes):
    first = np.unique(codes, return_index=True)[1]
    y_mean, xb_mean = entity_means(y, codes), entity_means(xb, codes)
    return {
        "r_squared_within": squared_correlation(y - y_mean, xb - xb_mean),
        "r_squared_between": squared_correlation(y_mean[first], xb_mean[first]),
        "r_squared_overall": squared_correlation(y, xb),
    }


# The number of entities and how many periods each has
def shape_of(counts):
    return {
        "entities": len(counts),
        "periods_min": int(counts.min()),
        "periods_mean": float(counts.mean()),
        "periods_max": int(counts.max()),
        "balanced": bool(counts.min() == counts.max()),
    }


# The coefficient table, with stars, in the same layout as OLS
def coefficient_table(names, roles, beta, cov, df_resid):
    std_error = np.sqrt(np.diag(cov))
    t_stat = beta / std_error
    results = pd.DataFrame({
        "role": roles,
        "variable": names,
        "coefficient": beta,
        "std_error": std_error,
        "t_stat": t_stat,
        "p_value": 2 * stats.t.sf(np.abs(t_stat), df_resid),
    })
    return significance.add_stars(results)


# Notes on regressors left out of a fit, and the names and roles of those kept
def kept_terms(terms, roles, kept, notes, reason):
    dropped = [name for position, name in enumerate(terms.columns) if position not in kept]
    if dropped:
        notes.append(f"Left out {', '.join(dropped)}: {reason}.")
    if not any(roles[position] == "X" for position in kept):
        raise ValueError(f"None of the X variables can be estimated: {reason}")
    return [terms.columns[p] for p in kept], [roles[p] for p in kept]


# Stata's fixed effects (xtreg, fe). Each variable's entity average is taken away and the overall
# average added back, so the slopes come only from changes within entities and the intercept is
# the average entity effect. Returns the coefficient table and the rows used with fitted values,
# residuals and each entity's effect u_i (measured from the intercept)
def fit_fixed(data, settings):
    used = model_rows(data, settings)
    terms, roles = regressors(used, settings)
    codes, counts = entity_codes(used, settings["entity"])
    rows = len(used)
    notes = []

    y = used[settings["dependent"]].to_numpy(float)
    x = terms.to_numpy(float)
    y_within = y - entity_means(y, codes)
    x_within = x - entity_means(x, codes)
    kept = independent_columns(x_within, rms(x))
    names, roles = kept_terms(terms, roles, kept, notes, "they don't change within entities, or other variables explain them, so the entity effects absorb them")
    x, x_within = x[:, kept], x_within[:, kept]

    slopes = len(names)
    df_resid = rows - len(counts) - slopes
    if df_resid <= 0:
        raise ValueError(f"{rows} rows are too few for {len(counts)} entities and {slopes} variables: each entity needs more periods")

    design = np.column_stack([np.ones(rows), x_within + x.mean(axis=0)])
    beta, resid = ols(design, y_within + y.mean())
    groups, cluster, df_test = error_groups(used, settings, design.shape[1], df_resid)
    cov = covariance(design, resid, df_resid, groups)

    xb = x @ beta[1:]
    effects = entity_means(y - xb, codes) - beta[0]
    sigma_e = np.sqrt(resid @ resid / df_resid)
    first = np.unique(codes, return_index=True)[1]
    sigma_u = float(np.std(effects[first], ddof=1)) if len(first) > 1 else 0.0

    # F test that every entity effect is zero: pooled OLS against fixed effects
    pooled = np.column_stack([np.ones(rows), x])
    pooled_resid = ols(pooled, y)[1]
    numerator = (pooled_resid @ pooled_resid - resid @ resid) / (len(counts) - 1)
    f_stat = numerator / (resid @ resid / df_resid) if len(counts) > 1 else np.nan

    results = coefficient_table(["Intercept"] + names, ["Intercept"] + roles, beta, cov, df_test)
    results.attrs = {
        "model": "Fixed Effects",
        "observations": rows,
        **shape_of(counts),
        **r_squared(y, xb, codes),
        "sigma_u": sigma_u,
        "sigma_e": float(sigma_e),
        "error_variance": float(sigma_e ** 2),
        "rho": sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2),
        "f_effects": float(f_stat),
        "f_effects_p": float(stats.f.sf(f_stat, len(counts) - 1, df_resid)),
        "robust": settings.get("robust", False),
        "cluster": cluster,
        "time_effects": bool(settings.get("time_effects")),
        "notes": notes,
    }
    record(results, "Fixed Effects", settings, used, cov, sigma_u ** 2 + sigma_e ** 2, df_test)

    fitted = used.copy()
    fitted["fitted"] = beta[0] + xb + effects
    fitted["residual"] = y - fitted["fitted"].to_numpy()
    fitted["effect"] = effects
    return [results, fitted]


# Stata's random effects (xtreg, re) with the Swamy-Arora variance components. The variance of the
# idiosyncratic error comes from the within (fixed effects) regression and that of the entity effects
# from the between regression of the entities' averages. Each row then has the share theta_i of its
# entity's average taken away, where theta_i = 1 - sqrt(sigma_e^2 / (T_i sigma_u^2 + sigma_e^2)), and
# OLS on the result is the GLS estimate. theta is 0 when there are no entity effects, giving pooled
# OLS, and nears 1 as they grow, giving fixed effects
def fit_random(data, settings):
    used = model_rows(data, settings)
    terms, roles = regressors(used, settings)
    codes, counts = entity_codes(used, settings["entity"])
    rows, entities = len(used), len(counts)
    notes = []

    y = used[settings["dependent"]].to_numpy(float)
    x = terms.to_numpy(float)
    kept = independent_columns(np.column_stack([np.ones(rows), x]))
    if kept[0] != 0:
        raise ValueError("The intercept can't be estimated")
    names, roles = kept_terms(terms, roles, [p - 1 for p in kept[1:]], notes, "other variables explain them")
    x = x[:, [p - 1 for p in kept[1:]]]

    # Within regression for sigma_e. Variables that don't change within entities drop out of it
    y_mean, x_mean = entity_means(y, codes), entity_means(x, codes)
    x_within = x - x_mean
    varying = independent_columns(x_within, rms(x))
    df_within = rows - entities - len(varying)
    if df_within <= 0:
        raise ValueError(f"{rows} rows are too few for {entities} entities and {len(varying)} variables that change over time")
    within_resid = ols(x_within[:, varying], y - y_mean)[1] if varying else y - y_mean
    sigma2_e = within_resid @ within_resid / df_within

    # Between regression of the entities' averages for sigma_u, with T-bar the harmonic mean of T_i
    first = np.unique(codes, return_index=True)[1]
    between = np.column_stack([np.ones(entities), x_mean[first]])
    df_between = entities - len(independent_columns(between))
    if df_between <= 0:
        raise ValueError(f"{entities} entities are too few to estimate the entity effects with {x.shape[1]} variables")
    between_resid = ols(between, y_mean[first])[1]
    t_bar = entities / (1.0 / counts).sum()
    sigma2_u = max(0.0, between_resid @ between_resid / df_between - sigma2_e / t_bar)
    if sigma2_u == 0:
        notes.append("The entity effects' variance is estimated as 0, so random effects is the same as pooled OLS.")

    theta = 1 - np.sqrt(sigma2_e / (counts * sigma2_u + sigma2_e))
    row_theta = theta[codes]
    design = np.column_stack([np.ones(rows), x]) - row_theta[:, None] * np.column_stack([np.ones(rows), x_mean])
    beta, resid = ols(design, y - row_theta * y_mean)
    df_resid = rows - design.shape[1]
    groups, cluster, df_test = error_groups(used, settings, design.shape[1], df_resid)
    cov = covariance(design, resid, df_resid, groups)

    xb = x @ beta[1:]
    shrink = counts * sigma2_u / (counts * sigma2_u + sigma2_e)
    effects = shrink[codes] * entity_means(y - beta[0] - xb, codes)
    pooled_resid = ols(np.column_stack([np.ones(rows), x]), y)[1]
    lm, lm_p = breusch_pagan_effects(pooled_resid, codes, counts)

    results = coefficient_table(["Intercept"] + names, ["Intercept"] + roles, beta, cov, df_test)
    results.attrs = {
        "model": "Random Effects",
        "observations": rows,
        **shape_of(counts),
        **r_squared(y, xb, codes),
        "sigma_u": float(np.sqrt(sigma2_u)),
        "sigma_e": float(np.sqrt(sigma2_e)),
        "error_variance": float(resid @ resid / df_resid),
        "rho": sigma2_u / (sigma2_u + sigma2_e),
        "theta_min": float(theta.min()),
        "theta_median": float(np.median(theta)),
        "theta_max": float(theta.max()),
        "lm_effects": lm,
        "lm_effects_p": lm_p,
        "robust": settings.get("robust", False),
        "cluster": cluster,
        "time_effects": bool(settings.get("time_effects")),
        "notes": notes,
    }
    record(results, "Random Effects", settings, used, cov, sigma2_u + sigma2_e, df_test)

    fitted = used.copy()
    fitted["fitted"] = beta[0] + xb + effects
    fitted["residual"] = y - fitted["fitted"].to_numpy()
    fitted["effect"] = effects
    return [results, fitted]


# The Breusch-Pagan Lagrange multiplier test that the entity effects have no variance, so pooled OLS
# would do, from the pooled OLS residuals (Baltagi and Li's version for unbalanced panels).
# Returns the statistic and its chi-squared (1) p value, or missing values if no entity has 2 periods
def breusch_pagan_effects(resid, codes, counts):
    pairs = float((counts * (counts - 1)).sum())
    if pairs == 0:
        return np.nan, np.nan
    sums = np.bincount(codes, weights=resid)
    lm = len(resid) ** 2 / (2 * pairs) * ((sums ** 2).sum() / (resid @ resid) - 1) ** 2
    return float(lm), float(stats.chi2.sf(lm, 1))


# Records the fit so Forecast can use it and Hausman can compare it. A forecast is for an entity
# whose own effect isn't known, so its prediction interval allows for the spread of the entity
# effects as well as the idiosyncratic error. Time dummies only forecast periods that were in the fit
def record(results, model_name, settings, used, cov, residual_variance, df_resid):
    fields = settings.get("independent", []) + settings.get("controls", [])
    if settings.get("time_effects"):
        used = used.assign(**{settings["time"]: used[settings["time"]].astype(str)})
        fields = fields + [settings["time"]]
    fits.record(
        results, "linear", model_name, settings["dependent"], used, fields,
        covariance=cov,
        residual_variance=float(residual_variance),
        degrees_of_freedom=float(df_resid),
    )
