import warnings

import numpy as np
import pandas as pd
from scipy import optimize, signal, special
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from engine import draw, estimates, roles
from engine.chart import draw_lines


INPUTS = 1
OUTPUTS = 3
OUTPUT_TYPES = ["model", "table", "table"]
INPUT_SPECS = [
    ("Data", "a time series table with a numeric Y, usually returns or growth rates in percent (e.g. 100 times the change "
             "in log prices), optionally X fields for the mean, a time field (T), and a group field (G) for several series"),
]
OUTPUT_SPECS = [
    ("Coefficients", "the mean equation's terms, then omega (the constant in the variance), alpha (ARCH: how much last period's "
                     "shock moves the variance), gamma (GJR or EGARCH: the extra effect of bad news), beta (GARCH: how long "
                     "volatility lasts) and nu (Student t degrees of freedom), with robust std_error, z_stat, p_value and stars"),
    ("Fitted data", "one row per period with the time, actual, mean, residual, volatility (the conditional standard deviation) "
                    "and std_residual"),
    ("Volatility forecast", "one row per period ahead with the expected variance and volatility"),
]

MODELS = [("garch", "GARCH"), ("gjr", "GJR-GARCH"), ("egarch", "EGARCH")]
MEANS = [("constant", "Constant"), ("zero", "Zero"), ("ar1", "AR(1)")]
DISTRIBUTIONS = [("normal", "Normal"), ("t", "Student t")]
FORECAST_PATHS = 2000


def volatility_chart(node):
    if len(node.last_outputs) < 2:
        return None
    fitted = node.last_outputs[1]
    x = np.arange(len(fitted), dtype=float)
    time = fitted.columns[0]
    if pd.api.types.is_numeric_dtype(fitted[time]):
        x = fitted[time].to_numpy(float)
    series = [("volatility", x, fitted["volatility"].to_numpy(float)), ("|residual|", x, fitted["residual"].abs().to_numpy(float))]

    def chart(painter, area):
        draw_lines(painter, area, draw.font(22), series, time)
    return chart


SPEC = roles.Spec(
    "GARCH",
    [
        roles.Role("Y", "dependent"),
        roles.Role("X", "independent", many=True),
        roles.Role("T", "time", kinds="any"),
        roles.Role("G", "group", kinds="any"),
    ],
    choices=[roles.Choice("model", "Model", MODELS), roles.Choice("mean", "Mean", MEANS),
             roles.Choice("distribution", "Errors", DISTRIBUTIONS)],
    numbers=[roles.Number("arch", "ARCH order (alpha)", range(1, 4), 1), roles.Number("garch", "GARCH order (beta)", range(0, 4), 1),
             roles.Number("horizon", "Forecast periods", [5, 10, 20, 50], 10)],
    chart=volatility_chart,
    series=True,
    hint="Choose Y (e.g. returns in percent) and the time field (T), then run the workflow",
)
FIELD_SETTINGS = SPEC.field_settings()


def on_click(node):
    return roles.open_panel(node, SPEC)


class Layout:
    # Where each group of parameters sits in the parameter vector
    def __init__(self, mean_names, q, p, model, student):
        self.mean_names = mean_names
        self.q, self.p, self.model, self.student = q, p, model, student
        self.k = len(mean_names)
        self.leverage = model in ("gjr", "egarch")
        names = list(mean_names) + ["omega"] + [f"alpha{i + 1}" for i in range(q)]
        if self.leverage:
            names += [f"gamma{i + 1}" for i in range(q)]
        names += [f"beta{j + 1}" for j in range(p)]
        if student:
            names.append("nu")
        self.names = names

    def split(self, params):
        k, q, p = self.k, self.q, self.p
        at = k
        omega = params[at]; at += 1
        alpha = params[at:at + q]; at += q
        gamma = params[at:at + q] if self.leverage else np.zeros(q)
        at += q if self.leverage else 0
        beta = params[at:at + p]; at += p
        nu = params[at] if self.student else None
        return params[:k], omega, alpha, gamma, beta, nu


# The conditional variance of each period. GARCH and GJR are linear in the past squared shocks, so
# they run through a filter; EGARCH works on the log variance one period at a time. Shocks before
# the sample are taken at their average size (the backcast)
def variances(layout, omega, alpha, gamma, beta, e, backcast, nu):
    n, q, p = len(e), layout.q, layout.p
    if layout.model == "egarch":
        mean_abs = np.sqrt(2 / np.pi) if nu is None else expected_abs_t(nu)
        log_h = np.empty(n)
        past_log = [np.log(backcast)] * max(p, 1)
        past_z = [0.0] * q
        for t in range(n):
            value = omega + sum(alpha[i] * (abs(past_z[i]) - mean_abs) + gamma[i] * past_z[i] for i in range(q))
            value += sum(beta[j] * past_log[j] for j in range(p))
            value = min(max(value, -50.0), 50.0)
            log_h[t] = value
            past_log = [value] + past_log[:-1]
            past_z = [e[t] / np.exp(value / 2)] + past_z[:-1]
        return np.exp(log_h)
    squared = np.concatenate([np.full(q, backcast), e ** 2])
    negative = np.concatenate([np.full(q, backcast / 2), (e < 0) * e ** 2])
    drive = np.full(n, omega)
    for i in range(q):
        drive += alpha[i] * squared[q - 1 - i:q - 1 - i + n] + gamma[i] * negative[q - 1 - i:q - 1 - i + n]
    if p == 0:
        return drive
    a = np.concatenate([[1.0], -beta])
    initial = signal.lfiltic([1.0], a, y=np.full(p, backcast))
    return signal.lfilter([1.0], a, drive, zi=initial)[0]


def expected_abs_t(nu):
    return np.sqrt(nu - 2) * special.gamma((nu - 1) / 2) / (np.sqrt(np.pi) * special.gamma(nu / 2))


def log_density(e, h, nu):
    if nu is None:
        return -0.5 * (np.log(2 * np.pi) + np.log(h) + e ** 2 / h)
    return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(h)
            - (nu + 1) / 2 * np.log1p(e ** 2 / (h * (nu - 2))))


# The size of the shocks before the sample starts: an exponentially weighted average of the
# first squared residuals (decay 0.94, up to 75 periods), as the arch package and RiskMetrics use
def backcast_of(e):
    first = e[:min(75, len(e))] ** 2
    weights = 0.94 ** np.arange(len(first))
    return float(np.sum(weights * first) / np.sum(weights))


def residuals_of(mean_params, y, x):
    return y - x @ mean_params if x.shape[1] else y.copy()


def persistence(layout, alpha, gamma, beta):
    if layout.model == "egarch":
        return float(np.sum(beta))
    return float(np.sum(alpha) + np.sum(gamma) / 2 + np.sum(beta))


def make_loglike(layout, y, x, backcast):
    def loglike(params):
        mean_params, omega, alpha, gamma, beta, nu = layout.split(params)
        if layout.model != "egarch" and persistence(layout, alpha, gamma, beta) >= 0.9999:
            return np.full(len(y), -1e10 / len(y))
        e = residuals_of(mean_params, y, x)
        h = variances(layout, omega, alpha, gamma, beta, e, backcast, nu)
        if not np.all(np.isfinite(h)) or np.any(h <= 0):
            return np.full(len(y), -1e10 / len(y))
        return log_density(e, h, nu)
    return loglike


def starting_values(layout, y, x, variance):
    mean = np.linalg.lstsq(x, y, rcond=None)[0] if x.shape[1] else np.array([])
    q, p = layout.q, layout.p
    if layout.model == "egarch":
        values = [np.log(variance) * (1 - 0.9 if p else 1)] + [0.1] * q + ([-0.05] * q) + [0.9 / max(p, 1)] * p
    else:
        alpha = [0.08 / q] * q
        gamma = [0.04 / q] * q if layout.leverage else []
        beta = [0.85 / p] * p if p else []
        values = [variance * (1 - sum(alpha) - sum(gamma) / 2 - sum(beta))] + alpha + gamma + beta
    if layout.student:
        values.append(8.0)
    return np.concatenate([mean, values])


def bounds_for(layout, variance):
    bounds = [(None, None)] * layout.k
    if layout.model == "egarch":
        bounds += [(None, None)] + [(None, None)] * layout.q * 2 + [(-0.9999, 0.9999)] * layout.p
    else:
        bounds += [(variance * 1e-8, variance * 10)] + [(0.0, 1.0)] * layout.q
        if layout.leverage:
            bounds += [(-1.0, 2.0)] * layout.q
        bounds += [(0.0, 1.0)] * layout.p
    if layout.student:
        bounds.append((2.05, 300.0))
    return bounds


# Simulates the variance forward from the end of the sample, averaging many paths, which works
# the same way for every model
def forecast_variance(layout, params, e, h, horizon, seed=12345):
    mean_params, omega, alpha, gamma, beta, nu = layout.split(params)
    rng = np.random.default_rng(seed)
    q, p = layout.q, layout.p
    paths = FORECAST_PATHS
    keep = max(p, q, 1)
    past_e = np.tile(e[::-1][:q], (paths, 1)) if q else np.zeros((paths, 0))
    past_h = np.tile(h[::-1][:keep], (paths, 1))
    mean_abs = np.sqrt(2 / np.pi) if nu is None else expected_abs_t(nu)
    output = []
    for _ in range(horizon):
        if layout.model == "egarch":
            z_past = past_e / np.sqrt(past_h[:, :q])
            log_next = (omega + (alpha * (np.abs(z_past) - mean_abs) + gamma * z_past).sum(axis=1)
                        + (beta * np.log(past_h[:, :p])).sum(axis=1))
            next_h = np.exp(np.clip(log_next, -50, 50))
        else:
            next_h = (omega + (alpha * past_e ** 2 + gamma * (past_e < 0) * past_e ** 2).sum(axis=1)
                      + (beta * past_h[:, :p]).sum(axis=1))
        output.append(float(next_h.mean()))
        z = rng.standard_normal(paths) if nu is None else rng.standard_t(nu, paths) * np.sqrt((nu - 2) / nu)
        shock = np.sqrt(next_h) * z
        if q:
            past_e = np.column_stack([shock, past_e[:, :-1]])
        past_h = np.column_stack([next_h, past_h[:, :-1]])
    return np.array(output)


# Fits a GARCH-family model by maximum likelihood: Y's mean (a constant, zero or an AR(1), plus
# any X fields) and its conditional variance, which rises after large shocks and fades back.
# GJR adds an extra effect of negative shocks, and EGARCH models the log of the variance so it
# needs no limits. Standard errors are Bollerslev-Wooldridge robust ones, valid even if the
# errors aren't the distribution assumed
def run(inputs, settings):
    SPEC.defaults(settings)
    data = inputs[0].replace([np.inf, -np.inf], np.nan)
    roles.check(data, settings, SPEC.roles, required=("Y",))
    notes = []
    data, times = estimates.time_order(data, settings, notes)
    dependent = settings["dependent"]
    fields = settings.get("independent") or []
    frame = data[[dependent] + fields].astype(float)
    mean_names = []
    columns = []
    if settings["mean"] in ("constant", "ar1"):
        mean_names.append("mu")
        columns.append(np.ones(len(frame)))
    if settings["mean"] == "ar1":
        mean_names.append("AR(1)")
        columns.append(frame[dependent].shift(1).to_numpy())
    for field in fields:
        mean_names.append(field)
        columns.append(frame[field].to_numpy())
    x_full = np.column_stack(columns) if columns else np.empty((len(frame), 0))
    complete = frame[dependent].notna().to_numpy() & ~np.isnan(x_full).any(axis=1)
    if (~complete).any():
        positions = np.flatnonzero(complete)
        inside = ~complete[positions[0]:positions[-1] + 1] if len(positions) else []
        if np.any(inside):
            notes.append(f"{int(np.sum(inside))} periods with missing values inside the sample are dropped, which joins the periods either side.")
    y = frame[dependent].to_numpy()[complete]
    x = x_full[complete]
    times = times[complete]
    if len(y) < 50:
        raise ValueError(f"{len(y)} periods is too few for a GARCH model; it needs at least 50 and works best with hundreds")
    scale = np.std(y)
    if scale > 0 and (scale < 0.01 or scale > 1000):
        notes.append(f"Y's standard deviation is {scale:.3g}; GARCH fits best with returns in percent (standard deviation around 1).")

    layout = Layout(mean_names, settings["arch"], settings["garch"], settings["model"], settings["distribution"] == "t")
    start = starting_values(layout, y, x, np.var(residuals_of(starting_values(layout, y, x, 1.0)[:layout.k], y, x)))
    e0 = residuals_of(start[:layout.k], y, x)
    variance = float(np.var(e0))
    backcast = backcast_of(e0)
    loglike = make_loglike(layout, y, x, backcast)

    def negative(params):
        value = -np.sum(loglike(params))
        return value if np.isfinite(value) else 1e20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize.minimize(negative, start, method="L-BFGS-B", bounds=bounds_for(layout, variance),
                                   options={"maxiter": 3000})
        again = optimize.minimize(negative, result.x, method="SLSQP", bounds=bounds_for(layout, variance), options={"maxiter": 1000})
        if again.fun < result.fun:
            result = again
        params = result.x
        covariance = estimates.covariance_of(loglike, params, robust=True)
    llf = -float(result.fun)
    if not result.success:
        notes.append("The optimiser reported a problem; check the estimates look sensible.")

    mean_params, omega, alpha, gamma, beta, nu = layout.split(params)
    e = residuals_of(mean_params, y, x)
    h = variances(layout, omega, alpha, gamma, beta, e, backcast, nu)
    term_roles = (["Mean"] * layout.k + ["Omega"] + ["ARCH"] * layout.q + (["Leverage"] * layout.q if layout.leverage else [])
                  + ["GARCH"] * layout.p + (["Nu"] if layout.student else []))
    table = estimates.coefficient_table(layout.names, term_roles, params, estimates.standard_errors(covariance))

    persist = persistence(layout, alpha, gamma, beta)
    criteria = estimates.information_criteria(llf, len(params), len(y))
    standardised = e / np.sqrt(h)
    lb_lag = min(10, len(y) // 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = float(acorr_ljungbox(standardised, lags=[lb_lag])["lb_pvalue"].iloc[0])
        lb2 = float(acorr_ljungbox(standardised ** 2, lags=[lb_lag])["lb_pvalue"].iloc[0])
        arch_p = float(het_arch(standardised, nlags=5)[1])
    name = dict(MODELS)[settings["model"]]
    summary = [
        f"{name}({layout.p}, {layout.q})    N: {len(y)}    Log-likelihood: {llf:.2f}    AIC: {criteria['aic']:.2f}    BIC: {criteria['bic']:.2f}",
        f"Persistence: {persist:.4f}" + (f"    Half-life of a shock: {np.log(0.5) / np.log(persist):.1f} periods" if 0 < persist < 1 else ""),
        f"Standardised residuals: Ljung-Box p (lag {lb_lag}) {lb:.3g}, on squares {lb2:.3g}, ARCH-LM p {arch_p:.3g}",
    ]
    if settings["model"] != "egarch" and persist < 1:
        summary.append(f"Long-run volatility: {np.sqrt(omega / (1 - persist)):.4g} per period")
    if lb2 < 0.05 or arch_p < 0.05:
        notes.append("Some ARCH effects are left in the standardised residuals; try higher orders or another model.")
    if persist > 0.99:
        notes.append("Persistence is close to 1, so shocks to volatility barely fade (an integrated GARCH).")
    table.attrs = {"observations": len(y), "log_likelihood": llf, "persistence": persist, **criteria,
                   "summary": summary, "notes": notes}
    estimates.record(table, "garch", name, dependent, data, fields, covariance, llf, observations=len(y))

    time_name = settings.get("time") or "row"
    fitted = pd.DataFrame({
        time_name: times.to_numpy() if hasattr(times, "to_numpy") else times,
        "actual": y, "mean": y - e, "residual": e, "volatility": np.sqrt(h), "std_residual": standardised,
    })
    if settings.get("group"):
        fitted.insert(0, settings["group"], roles.chosen_series(inputs[0], settings))
    horizon = settings["horizon"]
    ahead = forecast_variance(layout, params, e, h, horizon)
    forecast = pd.DataFrame({"periods_ahead": np.arange(1, horizon + 1), "variance": ahead, "volatility": np.sqrt(ahead)})
    for output in (fitted, forecast):
        output.attrs = {"summary": summary, "notes": notes}
    return [table, fitted, forecast]
