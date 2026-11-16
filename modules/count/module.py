import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson

from engine import estimates, roles


INPUTS = 1
OUTPUTS = 2
OUTPUT_TYPES = ["model", "table"]
INPUT_SPECS = [
    ("Data", "a table with a count for Y (whole numbers, 0 or more), numeric or text X and C fields, optionally an exposure field (E) "
             "such as population or months observed, and for zero-inflated models the fields (Z) that predict an excess zero"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, coefficient, std_error, z_stat, p_value, stars and irr (the incidence "
                     "rate ratio, exp of the coefficient). Negative binomial models add alpha; zero-inflated ones add the Inflate rows"),
    ("Fitted data", "the rows used, plus predicted (the expected count) and residual"),
]

MODELS = [("poisson", "Poisson"), ("negbin", "Neg. binomial"), ("zip", "ZI Poisson"), ("zinb", "ZI neg. binomial")]
ERRORS = [("ordinary", "Ordinary"), ("robust", "Robust")]
SPEC = roles.Spec(
    "Count",
    [
        roles.Role("Y", "dependent"),
        roles.Role("X", "independent", many=True, kinds="model"),
        roles.Role("C", "controls", many=True, kinds="model"),
        roles.Role("E", "exposure"),
        roles.Role("Z", "inflate", many=True, kinds="model"),
    ],
    choices=[roles.Choice("model", "Model", MODELS), roles.Choice("errors", "Errors", ERRORS)],
    hint="Choose a count for Y and at least one X, then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings()
NAMES = dict(MODELS)


def on_click(node):
    return roles.open_panel(node, SPEC)


# Overdispersion test of Cameron and Trivedi (1990): regresses ((y - mu)^2 - y) / mu on mu with
# no constant. A significantly positive slope says the variance grows faster than the mean, so
# a negative binomial model fits better than Poisson
def overdispersion(y, mu):
    left = ((y - mu) ** 2 - y) / mu
    fit = sm.OLS(left, mu).fit()
    return float(fit.params[0]), float(stats.t.sf(fit.tvalues[0], fit.df_resid))


def fit_model(kind, y, x, exposure, inflate, robust):
    cov_type = "HC0" if robust else "nonrobust"
    offset = None if exposure is None else np.log(exposure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "poisson":
            model = sm.Poisson(y, x, offset=offset)
            return model.fit(disp=0, cov_type=cov_type, maxiter=200)
        if kind == "negbin":
            model = sm.NegativeBinomial(y, x, loglike_method="nb2", offset=offset)
            return model.fit(disp=0, cov_type=cov_type, maxiter=500)
        model_class = ZeroInflatedPoisson if kind == "zip" else ZeroInflatedNegativeBinomialP
        extra = {} if kind == "zip" else {"p": 2}
        model = model_class(y, x, exog_infl=inflate, offset=offset, inflation="logit", **extra)
        try:
            fit = model.fit(disp=0, cov_type=cov_type, maxiter=1000, method="bfgs")
        except np.linalg.LinAlgError:
            fit = model.fit(disp=0, cov_type=cov_type, maxiter=2000, method="nm")
        return fit


def describe_parameter(name, roles_by_column):
    if name.startswith("inflate_"):
        inner = name[len("inflate_"):]
        return "Inflate", f"inflate: {'Intercept' if inner == 'Intercept' else inner}"
    if name == "alpha":
        return "Alpha", "alpha"
    return roles_by_column.get(name, ""), name


# Fits a count model of Y on X and C by maximum likelihood: Poisson, negative binomial (NB2, whose
# variance is mu + alpha mu^2), or their zero-inflated forms, where a logit on the Z fields (or a
# constant) gives the chance of an excess zero. An exposure field enters as log(E) with its
# coefficient fixed at 1, so the coefficients describe rates per unit of exposure
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y", "X"))
    kind = settings["model"]
    dependent = settings["dependent"]
    exposure_field = settings.get("exposure")
    fields = settings["independent"] + settings["controls"]
    inflate_fields = settings.get("inflate") or []
    needed = [dependent] + fields + inflate_fields + ([exposure_field] if exposure_field else [])
    used = data.dropna(subset=list(dict.fromkeys(needed)))
    notes = []

    y = used[dependent].astype(float)
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError(f"{dependent} must be a count: whole numbers of 0 or more")
    exposure = None
    if exposure_field:
        exposure = used[exposure_field].astype(float)
        if (exposure <= 0).any():
            raise ValueError(f"The exposure {exposure_field} must be above 0 in every row")

    x, term_roles = estimates.design(used, settings)
    roles_by_column = dict(zip(x.columns, term_roles))
    inflate = None
    if kind in ("zip", "zinb"):
        inflate, _ = estimates.design(used, {"independent": inflate_fields}, groups=(("independent", "Z"),))
        if not inflate_fields:
            notes.append("No Z fields are chosen, so the chance of an excess zero is the same for every row.")
    elif inflate_fields:
        notes.append("Z fields are only used by the zero-inflated models.")
    if len(used) < x.shape[1] + 5:
        raise ValueError(f"{len(used)} rows is too few for {x.shape[1]} coefficients")

    fit = fit_model(kind, y, x, exposure, inflate, settings["errors"] == "robust")
    if not fit.mle_retvals.get("converged", True):
        notes.append("The optimiser did not converge, so treat the estimates with caution.")

    described = [describe_parameter(str(name), roles_by_column) for name in fit.params.index]
    table = estimates.coefficient_table(
        [v for r, v in described], [r for r, v in described], fit.params.values, fit.bse.values,
        extra={"irr": np.where([r not in ("Alpha", "Inflate") for r, v in described], np.exp(fit.params.values), np.nan)},
    )

    mu = np.asarray(fit.predict(), dtype=float) if kind in ("poisson", "negbin") else np.asarray(fit.predict(which="mean"), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        null_llf = getattr(fit, "llnull", np.nan)
    criteria = estimates.information_criteria(fit.llf, len(fit.params), len(used))
    summary = [
        f"{NAMES[kind]}    N: {len(used)}    Log-likelihood: {fit.llf:.2f}    AIC: {criteria['aic']:.1f}    BIC: {criteria['bic']:.1f}",
    ]
    attrs = {"observations": len(used), "log_likelihood": float(fit.llf), **criteria, "model_type": NAMES[kind]}
    if np.isfinite(null_llf):
        attrs["pseudo_r_squared"] = float(1 - fit.llf / null_llf)
        summary[0] += f"    Pseudo R squared: {attrs['pseudo_r_squared']:.3f}"
    if kind == "poisson":
        slope, p_value = overdispersion(y.to_numpy(), mu)
        attrs["overdispersion_p"] = p_value
        summary.append(f"Overdispersion test (Cameron-Trivedi): alpha = {slope:.3g}, p = {estimates.number_text(p_value)}")
        if p_value < 0.05:
            notes.append("The variance grows faster than the mean (overdispersion), so the Poisson errors are too small: "
                         "use robust errors or a negative binomial model.")
    if kind == "negbin":
        poisson = fit_model("poisson", y, x, exposure, None, False)
        statistic = max(2 * (fit.llf - poisson.llf), 0.0)
        # alpha = 0 is on the edge of what it can be, so the p value is half the chi-squared one
        p_value = 0.5 * stats.chi2.sf(statistic, 1)
        attrs["alpha_lr_p"] = float(p_value)
        summary.append(f"LR test of alpha = 0 (Poisson against negative binomial): chi2 = {statistic:.2f}, p = {estimates.number_text(p_value)}")
    if exposure_field:
        summary.append(f"Exposure: log({exposure_field}) enters with its coefficient fixed at 1")
    zeros = float((y == 0).mean())
    predicted_zeros = float(np.mean(np.exp(-mu))) if kind == "poisson" else np.nan
    if kind == "poisson":
        summary.append(f"Zeros: {zeros:.1%} of rows, {predicted_zeros:.1%} predicted")
        if zeros > predicted_zeros + 0.05:
            notes.append("There are more zeros than a Poisson model expects; a zero-inflated model may fit better.")
    attrs.update(summary=summary, notes=notes, robust=settings["errors"] == "robust")
    table.attrs = attrs
    estimates.record(table, "count", NAMES[kind], dependent, used, fields, fit.cov_params(), fit.llf,
                     observations=len(used), count_model=kind, exposure=exposure_field)

    fitted = used.copy()
    fitted["predicted"] = mu
    fitted["residual"] = y - mu
    return [table, fitted]
