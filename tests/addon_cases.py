import importlib

import pandas as pd

from tests.support import ROOT, SAMPLE_DATA


# One case per add-on module: the data it is given and its settings. run_case runs it straight
# through the module's run function, the same way the workflow engine does
CAUSAL = ROOT / "addons" / "causal" / "sample_data"
PSYCH = ROOT / "addons" / "psych" / "sample_data" / "psych_survey.csv"
ESTEEM = [f"esteem_{i}" for i in range(1, 7)]
SOCIAL = [f"social_{i}" for i in range(1, 5)]
ORDERED = {"dependent": "esteem_1", "independent": ["stress"], "controls": ["sleep_hours"]}
DISTRIBUTIONS = [
    "bernoulli", "beta", "binomial", "chi_square", "exponential", "f", "gamma", "geometric",
    "hypergeometric", "lognormal", "neg_binomial", "normal", "poisson", "student_t", "uniform",
]

# name -> (pack, module folder, data file or None, settings)
CASES = {
    "did_2x2": ("causal", "did", CAUSAL / "did_states.csv",
                {"dependent": "employment", "group": "treated_state", "post": "post", "design": "2x2"}),
    "did_twfe": ("causal", "did", CAUSAL / "did_states.csv",
                 {"dependent": "employment", "treatment": "policy", "entity": "state", "time": "year", "design": "TWFE"}),
    "iv_2sls": ("causal", "iv_2sls", CAUSAL / "iv_schooling.csv",
                {"dependent": "log_wage", "independent": ["educ"], "instruments": ["distance", "q1_birth"], "controls": ["female"]}),
    "rdd": ("causal", "rdd", CAUSAL / "rdd_scholarship.csv", {"dependent": "earnings", "running": "score", "text": "cutoff=70"}),
    "pearson": ("descriptives", "pearson", SAMPLE_DATA, {}),
    "summary_stats": ("descriptives", "summary_stats", SAMPLE_DATA, {}),
    "box_plot": ("graphs_", "box_plot", SAMPLE_DATA, {"value": "unemployment", "group": "region"}),
    "histogram": ("graphs_", "histogram", SAMPLE_DATA, {"column": "unemployment"}),
    "pie_chart": ("graphs_", "pie_chart", SAMPLE_DATA, {"category": "region", "value": "population_k"}),
    "ologit": ("ordered", "ologit", PSYCH, ORDERED),
    "oprobit": ("ordered", "oprobit", PSYCH, ORDERED),
    "brant_test": ("ordered", "brant_test", PSYCH, ORDERED),
    "lipsitz_test": ("ordered", "lipsitz_test", PSYCH, ORDERED),
    "ordinal_assoc": ("ordered", "ordinal_assoc", PSYCH, {"dependent": "esteem_1", "independent": ["esteem_2"]}),
    "anova": ("psych", "anova", PSYCH, {"dependent": "wellbeing", "factors": ["group", "gender"]}),
    "correlations": ("psych", "correlations", PSYCH, {"variables": ["stress", "sleep_hours", "wellbeing"]}),
    "cronbach_alpha": ("psych", "cronbach_alpha", PSYCH, {"items": ESTEEM}),
    "crosstab": ("psych", "crosstab", PSYCH, {"row": "group", "column": "gender"}),
    "efa": ("psych", "efa", PSYCH, {"variables": ESTEEM + SOCIAL}),
    "means_plot": ("psych", "means_plot", PSYCH, {"dependent": "wellbeing", "group": "group"}),
    "mediation": ("psych", "mediation", PSYCH, {"predictor": "stress", "mediator": "sleep_hours", "outcome": "wellbeing"}),
    "normality": ("psych", "normality", PSYCH, {"variables": ["stress", "wellbeing"]}),
    "post_hoc": ("psych", "post_hoc", PSYCH, {"dependent": "wellbeing", "group": "group"}),
    "rank_tests": ("psych", "rank_tests", PSYCH, {"dependent": "wellbeing", "group": "group"}),
    "rm_anova": ("psych", "rm_anova", PSYCH, {"within": ["anxiety_pre", "anxiety_mid", "anxiety_post"]}),
    "scale_score": ("psych", "scale_score", PSYCH, {"items": ESTEEM}),
    "t_test": ("psych", "t_test", PSYCH, {"y": ["wellbeing"], "group": "gender"}),
}
for name in DISTRIBUTIONS:
    CASES[f"dist_{name}"] = ("distributions", name, None, {})


def load(pack, folder):
    return importlib.import_module(f"addons.{pack}.{folder}.module")


# Runs a case and returns its outputs. Settings are copied, as a module may fill in defaults
def run_case(name):
    pack, folder, data, settings = CASES[name]
    inputs = [pd.read_csv(data)] if data else []
    return load(pack, folder).run(inputs, dict(settings))
