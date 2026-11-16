import numpy as np

from engine import dummies


# What a regression module records about its fit, so Forecast can apply its coefficients to new
# data. It goes in the coefficient table's attrs under "fit", which pandas carries along when the
# table is filtered or copied, as a dict of:
#   kind       one of KINDS
#   model      the module's name, e.g. "Auto OLS"
#   dependent  the Y field
#   levels     text field -> every category it had when fitting, in order, the first being the
#              baseline the dummies are measured against. A category not in the list is new
# plus what each kind needs on top (see the record functions)
KINDS = {
    "linear": "Linear (OLS)",
    "logit": "Logit",
    "probit": "Probit",
    "ologit": "Ordered logit",
    "oprobit": "Ordered probit",
    "arimax": "ARIMAX",
    "count": "Count (Poisson or negative binomial)",
    "tobit": "Tobit",
    "quantile": "Quantile regression",
}

# Models that record their fit for Wald tests and Bootstrap, but that Forecast can't carry on
# to new rows: their forecasts need more than the coefficients (a selection equation, past
# volatility, a system of equations or an entity's history)
OTHER_KINDS = {
    "heckman": "Heckman selection",
    "garch": "GARCH",
    "var": "VAR",
    "vecm": "VECM",
    "dpanel": "Dynamic panel GMM",
}


# Every category of each text field in columns, in the order the dummies take them
def levels(used, columns):
    return {
        column: sorted(used[column].astype(str).unique())
        for column in columns
        if column in used and not dummies.is_numeric(used[column])
    }


def record(results, kind, model, dependent, used, columns, **extra):
    results.attrs["fit"] = {
        "kind": kind,
        "model": model,
        "dependent": dependent,
        "levels": levels(used, columns),
        **extra,
    }


# A linear model also keeps the covariance of its coefficients (in the table's row order), its
# residual variance and degrees of freedom, so Forecast can give prediction intervals. columns are
# the fields it was fitted on (fields), so Diagnostics can rebuild its regressors from the fitted data, and
# the log-likelihood lets an LR test compare it with a model nested in it
def record_linear(results, model_name, fit, dependent, used, columns):
    record(
        results, "linear", model_name, dependent, used, columns,
        covariance=np.asarray(fit.cov_params(), dtype=float),
        residual_variance=float(fit.scale),
        degrees_of_freedom=float(fit.df_resid),
        fields=list(columns),
        log_likelihood=float(fit.llf),
        parameters=int(len(fit.params)),
        observations=int(fit.nobs),
    )


# A logit or probit keeps the covariance of its coefficients too, so Wald tests can use it
def record_binary(results, kind, model_name, fit, dependent, used, columns):
    record(
        results, kind, model_name, dependent, used, columns,
        covariance=np.asarray(fit.cov_params(), dtype=float),
        fields=list(columns),
        log_likelihood=float(fit.llf),
        parameters=int(len(fit.params)),
        observations=int(fit.nobs),
    )
