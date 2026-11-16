import copy
import warnings

import numpy as np
import pandas as pd

from engine import roles, significance
from engine.node import load_module
from modules.arimax.module import read_time


INPUTS = 2
OUTPUTS = 2
INPUT_TYPES = ["model", "table"]
INPUT_SPECS = [
    ("Model", "Out 1 of a regression straight from the module that fitted it (OLS, Logit, Count, Quantile Reg, Fixed Effects, "
              "ARIMAX, an add-on model...), so Bootstrap knows which model to fit again and with what settings"),
    ("Data", "the same table that went into that regression"),
]
OUTPUT_SPECS = [
    ("Bootstrap results", "one row per term with the original coefficient, its usual std_error, the bootstrap boot_std_error, a "
                          "percentile interval (lower, upper), the bootstrap p_value and stars. Permutation gives each "
                          "coefficient's permutation p_value instead"),
    ("Draws", "one row per replication with each term's coefficient, for histograms or your own summaries"),
]

METHODS = [("pairs", "Pairs"), ("cluster", "Cluster"), ("block", "Block"), ("permutation", "Permutation")]
REPLICATIONS = [(199, "199"), (499, "499"), (999, "999"), (1999, "1999")]
LEVELS = [(0.95, "95%"), (0.9, "90%"), (0.99, "99%")]
SEED = 20240611

SPEC = roles.Spec(
    "Bootstrap",
    [
        roles.Role("G", "group", kinds="any"),
        roles.Role("T", "time", kinds="any"),
    ],
    choices=[roles.Choice("method", "Method", METHODS), roles.Choice("replications", "Draws", REPLICATIONS, default=499),
             roles.Choice("level", "Interval", LEVELS)],
    numbers=[roles.Number("block", "Block length (0 = automatic)", [0, 2, 4, 5, 8, 10, 12, 20], 0)],
    data_input=1,
    hint="Connect a model and its data, choose the method, then run the workflow. Cluster needs G; Block uses T for the order",
)
FIELD_SETTINGS = SPEC.field_settings()


def on_click(node):
    return roles.open_panel(node, SPEC)


# The names a model's rows go by: the variable, with the equation in front for systems like VAR,
# and the quantile for tables with several
def term_names(table):
    names = table["variable"].astype(str)
    if "equation" in table:
        names = table["equation"].astype(str) + ": " + names
    return list(names)


def coefficients_of(table):
    if not isinstance(table, pd.DataFrame) or "variable" not in table or "coefficient" not in table:
        raise ValueError("The model's Out 1 needs variable and coefficient columns")
    return pd.Series(table["coefficient"].to_numpy(float), index=term_names(table))


# One resampled copy of the data
def resample(data, settings, rng, n_groups=None):
    method = settings["method"]
    n = len(data)
    if method == "pairs":
        return data.iloc[rng.integers(0, n, n)].reset_index(drop=True)
    if method == "cluster":
        group = settings["group"]
        labels = data[group]
        names = labels.dropna().unique()
        picked = names[rng.integers(0, len(names), len(names))]
        parts = []
        numeric = pd.api.types.is_numeric_dtype(labels)
        for k, name in enumerate(picked):
            part = data[labels == name].copy()
            # A cluster drawn twice counts as two clusters, so each copy gets its own label
            part[group] = k if numeric else f"{name}#{k}"
            parts.append(part)
        return pd.concat(parts, ignore_index=True)
    # Moving blocks of consecutive rows, laid end to end, keep the series' short-run dependence.
    # The time field is set back to the original periods so the copy is an ordinary series
    length = settings["block"] or max(2, int(round(n ** (1 / 3))))
    length = min(length, n)
    starts = rng.integers(0, n - length + 1, int(np.ceil(n / length)))
    positions = np.concatenate([np.arange(s, s + length) for s in starts])[:n]
    copy_ = data.iloc[positions].reset_index(drop=True)
    if settings.get("time"):
        copy_[settings["time"]] = data[settings["time"]].to_numpy()
    return copy_


def check_source(model):
    source = model.attrs.get("source")
    if not source:
        raise ValueError("Bootstrap needs to know which module fitted the model. Connect Out 1 straight from the regression and run the workflow")
    module = load_module(source["module"])
    if module is None or not hasattr(module, "run"):
        raise ValueError(f"The module that fitted the model ({source['module']}) can't be found")
    if getattr(module, "INPUTS", 1) != 1 or getattr(module, "GROWING_INPUTS", False):
        raise ValueError(f"{source['module']} takes more than one input, so Bootstrap can't refit it from one table")
    return source, module


# Refits the model that made In 1 on many resampled copies of In 2 and summarises how its
# coefficients vary: bootstrap standard errors, percentile intervals and p values that don't rest
# on the usual formulas. Pairs resamples rows, Cluster whole groups, Block runs of consecutive
# periods for time series. Permutation instead shuffles Y to see how large each coefficient gets
# when Y has nothing to do with the fields
def run(inputs, settings):
    SPEC.defaults(settings)
    model, data = inputs[0], inputs[1]
    if data is None:
        raise ValueError("Connect the data the model was fitted on to In 2")
    source, module = check_source(model)
    model_settings = copy.deepcopy(source["settings"])
    method = settings["method"]
    roles.check(data, settings, SPEC.roles)
    if method == "cluster" and not settings.get("group"):
        raise ValueError("Cluster resampling needs the field naming each group (G)")
    if method == "block" and settings.get("time"):
        order = read_time(data[settings["time"]]).sort_values(kind="stable").index
        data = data.loc[order].reset_index(drop=True)
    dependent = (model.attrs.get("fit") or {}).get("dependent") or model_settings.get("dependent")
    if method == "permutation" and (not isinstance(dependent, str) or dependent not in data):
        raise ValueError("Permutation needs the model's Y field in the data")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_table = module.run([data], copy.deepcopy(model_settings))[0]
    original = coefficients_of(original_table)
    stated = coefficients_of(model)
    if list(original.index) != list(stated.index) or not np.allclose(original.to_numpy(), stated.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True):
        raise ValueError("Fitting the model again on In 2 doesn't give In 1's coefficients. Connect the same data that went into the regression")
    errors = pd.Series(original_table["std_error"].to_numpy(float), index=original.index) if "std_error" in original_table else None

    rng = np.random.default_rng(SEED)
    replications = int(settings["replications"])
    draws, failed = [], 0
    for _ in range(replications):
        if method == "permutation":
            sample = data.copy()
            sample[dependent] = rng.permutation(sample[dependent].to_numpy())
        else:
            sample = resample(data, settings, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                table = module.run([sample], copy.deepcopy(model_settings))[0]
            draws.append(coefficients_of(table).reindex(original.index))
        except Exception:
            failed += 1
    if len(draws) < max(20, replications // 2):
        raise ValueError(f"{failed} of {replications} refits failed, too many for a bootstrap. Check the model fits on smaller samples")
    draws = pd.DataFrame(draws).reset_index(drop=True)

    level = float(settings["level"])
    values = draws.to_numpy(float)
    counts = np.sum(np.isfinite(values), axis=0)
    centre = original.to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spread = np.nanstd(values, axis=0, ddof=1)
        lower = np.nanquantile(values, (1 - level) / 2, axis=0)
        upper = np.nanquantile(values, (1 + level) / 2, axis=0)
        if method == "permutation":
            beyond = np.sum(np.abs(values) >= np.abs(centre) - 1e-12, axis=0)
        else:
            # Symmetric bootstrap p value: how often the draws stray from the estimate by as much as the estimate is from 0
            beyond = np.sum(np.abs(values - centre) >= np.abs(centre), axis=0)
    p_values = (beyond + 1) / (counts + 1)

    results = pd.DataFrame({"variable": list(original.index), "coefficient": centre})
    if errors is not None:
        results["std_error"] = errors.to_numpy()
    if method == "permutation":
        results["permutation_mean"] = np.nanmean(values, axis=0)
    else:
        results["boot_std_error"] = spread
        results["lower"] = lower
        results["upper"] = upper
    results["p_value"] = p_values
    results["draws"] = counts
    significance.add_stars(results)

    name = dict(METHODS)[method]
    summary = [f"{name} {'test' if method == 'permutation' else 'bootstrap'} of {source['module']}    "
               f"Draws: {len(draws)} of {replications}" + (f" ({failed} failed)" if failed else "")]
    if method != "permutation" and errors is not None:
        ratio = np.nanmedian(spread / errors.to_numpy())
        summary.append(f"Bootstrap standard errors are {ratio:.2f} times the usual ones (median)")
    else:
        summary.append("p value: share of shuffled fits with a coefficient at least as large as the real one")
    notes = []
    if method == "pairs" and model_settings.get("time"):
        notes.append("Pairs resampling breaks up a time series; Block keeps runs of periods together.")
    if method == "pairs" and (model_settings.get("cluster") or model_settings.get("entity")):
        notes.append("The model has groups; Cluster resampling keeps each group's rows together.")
    if replications < 999:
        notes.append("Use 999 or more draws for p values and 99% intervals you can rely on.")
    results.attrs = {"replications": len(draws), "failed": failed, "summary": summary, "notes": notes}
    draws.insert(0, "draw", np.arange(1, len(draws) + 1))
    return [results, draws]
