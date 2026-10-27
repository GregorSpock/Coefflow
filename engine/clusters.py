import numpy as np
import pandas as pd


# Clustered standard errors for the regression modules. The field to cluster by is kept in
# settings["cluster"]. Cluster-robust errors are only reliable with enough clusters of
# reasonably even size, so the checks here turn the usual rules of thumb into warnings.
# With few clusters the errors come out too small and tests reject too often
# (Cameron & Miller 2015 put the danger zone below roughly 20-50 clusters)
TOO_FEW_CLUSTERS = 10
FEW_CLUSTERS = 30
UNEVEN_SHARE = 0.75
MAX_SUGGESTIONS = 3

# Words in a field name that hint it identifies a group the rows belong to
GROUP_WORDS = [
    "id", "group", "cluster", "state", "country", "region", "county", "province", "city",
    "district", "firm", "company", "school", "class", "village", "household", "family",
    "person", "individual", "industry", "sector", "market", "store", "hospital", "team",
]


# The model's variables held in a module's settings, e.g. Y, X and C
def model_variables(settings):
    names = [settings.get("dependent")] + settings.get("independent", []) + settings.get("controls", [])
    return [n for n in names if n]


# The rows a model would use: those complete for its variables that are in the data
def model_rows(data, settings):
    return data.dropna(subset=[c for c in model_variables(settings) if c in data.columns])


# How many clusters a field makes and how evenly the rows spread over them.
# The effective number of clusters, N squared over the sum of squared cluster sizes, equals
# the real number when every cluster is the same size and falls as their sizes drift apart
def summarise(values):
    sizes = values.dropna().value_counts()
    rows = int(sizes.sum())
    return {
        "clusters": len(sizes),
        "effective": rows ** 2 / float((sizes ** 2).sum()) if rows else 0.0,
        "smallest": int(sizes.min()) if rows else 0,
        "largest": int(sizes.max()) if rows else 0,
        "singletons": int((sizes == 1).sum()),
        "rows": rows,
        "missing": int(values.isna().sum()),
    }


# True if a field looks like a measurement rather than a label, e.g. 3.71 or 0.25
def is_continuous(values):
    values = values.dropna()
    if not pd.api.types.is_float_dtype(values):
        return False
    return not np.allclose(values, values.round())


# True if every X variable takes one value within each cluster. Then X was set at the
# cluster level (e.g. a policy by state), which is the strongest reason to cluster there
def x_set_by(data, column, independent):
    independent = [c for c in independent if c in data.columns and c != column]
    if not independent:
        return False
    within = data.dropna(subset=[column]).groupby(column)[independent].nunique()
    return bool(len(within)) and bool((within <= 1).all().all())


def has_group_word(column):
    words = str(column).lower().replace("-", "_").replace(" ", "_").split("_")
    return any(word in GROUP_WORDS or word.endswith("id") for word in words)


# A short reason a field would make a good cluster, or None if it would not.
# A field with too few clusters is only suggested if X is set at its level, with a warning
def suggestion_reason(data, column, settings):
    values = data[column]
    summary = summarise(values)
    if summary["clusters"] < 2 or summary["clusters"] > summary["rows"] / 2 or is_continuous(values):
        return None

    reasons = []
    if x_set_by(data, column, settings.get("independent", [])):
        reasons.append("X is set at this level")
    elif summary["clusters"] < TOO_FEW_CLUSTERS:
        return None
    if has_group_word(column):
        reasons.append("name looks like a group")
    if summary["clusters"] >= FEW_CLUSTERS:
        reasons.append("enough clusters")
    elif summary["clusters"] < TOO_FEW_CLUSTERS:
        reasons.append("but too few clusters")
    return ", ".join(reasons) or "groups the rows"


# The fields worth clustering by, best first. A field X is set by comes first, since errors
# should be clustered at the level the treatment varies. After that the coarsest grouping
# that still has enough clusters is preferred, as finer clusters can miss correlation between them
def suggestions(data, settings):
    data = model_rows(data, settings)
    ranked = []
    for column in data.columns:
        if column in model_variables(settings):
            continue
        reason = suggestion_reason(data, column, settings)
        if reason is None:
            continue
        clusters = data[column].nunique()
        ranked.append((
            not x_set_by(data, column, settings.get("independent", [])),
            clusters < FEW_CLUSTERS,
            not has_group_word(column),
            clusters if clusters >= FEW_CLUSTERS else -clusters,
            column,
            reason,
        ))
    ranked.sort(key=lambda r: r[:4])
    return [(r[4], r[5]) for r in ranked[:MAX_SUGGESTIONS]]


# Warnings about whether clustering by a field gives trustworthy standard errors.
# parameters is the number of coefficients including the intercept
def warnings_for(summary, column, parameters):
    clusters = summary["clusters"]
    effective = summary["effective"]
    notes = []

    if clusters < 2:
        return [f"{column} has fewer than 2 values in the rows used, so it cannot be clustered by."]
    if clusters < TOO_FEW_CLUSTERS:
        notes.append(
            f"Only {clusters} clusters: clustered standard errors will be far too small and p values too low. "
            "Cluster by a field with more clusters if it still groups the related rows, or use a wild cluster bootstrap."
        )
    elif clusters < FEW_CLUSTERS:
        notes.append(
            f"Only {clusters} clusters (fewer than about {FEW_CLUSTERS}): standard errors are likely biased down, "
            "so treat p values near the cut-off with caution."
        )
    if effective < FEW_CLUSTERS and effective < clusters * UNEVEN_SHARE:
        notes.append(
            f"Cluster sizes are uneven (largest {summary['largest']} rows, smallest {summary['smallest']}), "
            f"so the {clusters} clusters only count as about {effective:.0f}. Standard errors may be biased down."
        )
    if clusters <= parameters:
        notes.append(
            f"{clusters} clusters for {parameters} coefficients: tests of several coefficients at once are unreliable."
        )
    if summary["singletons"] > clusters / 2:
        notes.append(
            f"{summary['singletons']} of {clusters} clusters have a single row, so clustering adds little over robust errors."
        )
    if summary["missing"]:
        notes.append(f"{summary['missing']} rows with no {column} value are dropped.")
    return notes


# The covariance settings for a statsmodels fit, with the clusters' summary and warnings.
# used must already be complete for the cluster field. Returns (cov_type, cov_kwds, info)
def covariance(used, settings, parameters):
    column = settings.get("cluster")
    robust = settings.get("robust", False)
    if not column:
        return ("HC1" if robust else "nonrobust"), None, None

    summary = summarise(used[column])
    if summary["clusters"] < 2:
        raise ValueError(f"{column} has fewer than 2 values in the rows used, so it cannot be clustered by")
    groups = pd.factorize(used[column])[0]
    info = dict(summary, column=column, warnings=warnings_for(summary, column, parameters))
    return "cluster", {"groups": groups}, info


# Checks the cluster field is in the data. Returns the columns that rows need to be complete for
def cluster_columns(data, settings):
    column = settings.get("cluster")
    if not column:
        return []
    if column not in data.columns:
        raise ValueError(f"The cluster field {column} is not in the data")
    return [column]


# Describes the standard errors used in a fit, for the panel's info lines
def errors_text(fit):
    cluster = fit.get("cluster")
    if cluster:
        return f"clustered by {cluster['column']} ({cluster['clusters']} clusters)"
    return "robust (HC1)" if fit.get("robust") else "ordinary"
