import numpy as np
from scipy import stats
from scipy.special import log_ndtr

from engine import estimates, roles


INPUTS = 1
OUTPUTS = 2
OUTPUT_TYPES = ["model", "table"]
INPUT_SPECS = [
    ("Data", "a table with a numeric Y that piles up at a limit (such as hours worked, with many at 0), and numeric or text X and C fields"),
]
OUTPUT_SPECS = [
    ("Coefficients", "one row per term with role, variable, coefficient (the effect on the uncensored, latent Y), std_error, z_stat, "
                     "p_value, stars and marginal_effect (the average effect on the expected observed Y), then sigma"),
    ("Fitted data", "the rows used, plus xb (the latent prediction), expected (E[Y], allowing for the limits), "
                    "prob_uncensored and residual"),
]

ERRORS = [("ordinary", "Ordinary"), ("robust", "Robust")]
SPEC = roles.Spec(
    "Tobit",
    [
        roles.Role("Y", "dependent"),
        roles.Role("X", "independent", many=True, kinds="model"),
        roles.Role("C", "controls", many=True, kinds="model"),
    ],
    choices=[roles.Choice("errors", "Errors", ERRORS)],
    text=("limits", "Limits: lower, upper (blank for none), e.g. 0, or 0, 100", "0,"),
    hint="Choose Y and at least one X, type the limits, then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings()


def on_click(node):
    return roles.open_panel(node, SPEC)


# Reads "lower, upper" with either left blank, e.g. "0," or ", 100". "min" and "max" stand for the
# lowest and highest Y in the data
def parse_limits(text, y):
    parts = [p.strip() for p in str(text or "").replace(";", ",").split(",")]
    parts += [""] * (2 - len(parts))
    if len(parts) > 2:
        raise ValueError("Type at most two limits: lower, upper")
    limits = []
    for part, extreme in zip(parts, [y.min(), y.max()]):
        if part == "":
            limits.append(None)
        elif part.lower() in ("min", "max"):
            limits.append(float(extreme))
        else:
            try:
                limits.append(float(part))
            except ValueError:
                raise ValueError(f"{part} isn't a number. Type limits like 0, or 0, 100")
    lower, upper = limits
    if lower is None and upper is None:
        raise ValueError("Type at least one limit, e.g. 0, for Y censored below at 0")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("The lower limit must be below the upper one")
    return lower, upper


# Each row's log-likelihood: the normal density for values between the limits, and the chance of
# being at or beyond a limit for values there. The last parameter is log(sigma)
def loglike(params, y, x, low, high):
    beta, sigma = params[:-1], np.exp(params[-1])
    xb = x @ beta
    z = (y - xb) / sigma
    values = stats.norm.logpdf(z) - np.log(sigma)
    if low is not None:
        values = np.where(y <= low, log_ndtr((low - xb) / sigma), values)
    if high is not None:
        values = np.where(y >= high, log_ndtr((xb - high) / sigma), values)
    return values


# E[Y] allowing for the limits, the chance of being between them, and the derivative of E[Y]
# with respect to xb, which is that chance (so a marginal effect is beta times it)
def expected(xb, sigma, low, high):
    a = (low - xb) / sigma if low is not None else np.full_like(xb, -np.inf)
    b = (high - xb) / sigma if high is not None else np.full_like(xb, np.inf)
    pa, pb = stats.norm.cdf(a), stats.norm.cdf(b)
    inside = pb - pa
    value = xb * inside + sigma * (stats.norm.pdf(a) - stats.norm.pdf(b))
    if low is not None:
        value = value + low * pa
    if high is not None:
        value = value + high * (1 - pb)
    return value, inside


# Fits a Tobit (censored normal) regression by maximum likelihood: Y is a latent linear model in
# X and C, seen only between the limits, with values at or past a limit recorded at it
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y", "X"))
    dependent = settings["dependent"]
    fields = settings["independent"] + settings["controls"]
    used = data.dropna(subset=list(dict.fromkeys([dependent] + fields)))
    y = used[dependent].to_numpy(float)
    low, high = parse_limits(settings.get("limits"), used[dependent])
    x_frame, term_roles = estimates.design(used, settings)
    x = x_frame.to_numpy(float)
    notes = []

    below = int((y <= low).sum()) if low is not None else 0
    above = int((y >= high).sum()) if high is not None else 0
    if low is not None and (y < low).any():
        notes.append(f"{int((y < low).sum())} rows have Y below the lower limit, so they are taken as censored at it.")
    if high is not None and (y > high).any():
        notes.append(f"{int((y > high).sum())} rows have Y above the upper limit, so they are taken as censored at it.")
    if below + above == 0:
        notes.append("No rows are at a limit, so this is the same as OLS.")
    if len(y) - below - above < x.shape[1] + 2:
        raise ValueError("Too few rows are between the limits to fit the model")

    start_beta = np.linalg.lstsq(x, y, rcond=None)[0]
    start = np.append(start_beta, np.log(np.std(y - x @ start_beta) or 1.0))
    params, llf, covariance, converged = estimates.maximise(lambda p: loglike(p, y, x, low, high), start,
                                                            robust=settings["errors"] == "robust")
    if not converged:
        notes.append("The optimiser did not fully converge, so treat the estimates with caution.")

    sigma = float(np.exp(params[-1]))
    # sigma's standard error by the delta method from log(sigma)'s
    jacobian = np.eye(len(params))
    jacobian[-1, -1] = sigma
    covariance = jacobian @ covariance @ jacobian.T
    values = np.append(params[:-1], sigma)

    xb = x @ params[:-1]
    mean, inside = expected(xb, sigma, low, high)
    effects = params[:-1] * inside.mean()
    effects = np.where(np.array(term_roles) == "Intercept", np.nan, effects)
    table = estimates.coefficient_table(
        list(x_frame.columns) + ["sigma"], term_roles + ["Sigma"], values, estimates.standard_errors(covariance),
        extra={"marginal_effect": np.append(effects, np.nan)},
    )
    criteria = estimates.information_criteria(llf, len(params), len(y))
    limits_text = ", ".join(t for t in [f"left at {low:g}" if low is not None else "", f"right at {high:g}" if high is not None else ""] if t)
    summary = [
        f"N: {len(y)}    Censored {limits_text}: {below} below, {above} above, {len(y) - below - above} between",
        f"Log-likelihood: {llf:.2f}    AIC: {criteria['aic']:.1f}    BIC: {criteria['bic']:.1f}    sigma: {sigma:.4g}",
        "Coefficients are effects on the latent Y; marginal_effect is the average effect on the observed Y",
    ]
    table.attrs = {"observations": len(y), "log_likelihood": llf, "sigma": sigma, "censored_below": below, "censored_above": above,
                   **criteria, "summary": summary, "notes": notes}
    estimates.record(table, "tobit", "Tobit", dependent, used, fields, covariance, llf, lower=low, upper=high, sigma=sigma)

    fitted = used.copy()
    fitted["xb"] = xb
    fitted["expected"] = mean
    fitted["prob_uncensored"] = inside
    fitted["residual"] = y - mean
    return [table, fitted]
