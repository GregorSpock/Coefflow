from engine import addons


# Each toolbar tab and the core modules it shows. Add-on packs add to these (see engine/addons.py)
MODULES = {
    "Input/Output": ["Input Data", "Output Data", "Database", "Online Data"],
    "Clean": ["Join", "Split", "Filter", "Select", "Sort", "Convert", "Gen", "Group By", "Pivot", "Concat", "Missing Values", "Dates"],
    "Base Reg.": ["OLS", "Auto OLS", "Logit", "Probit"],
    "More Reg.": ["Count", "Tobit", "Heckman", "Quantile Reg"],
    "Time Series": ["Durbin-Watson", "Forecast", "ARIMAX", "Auto ARIMAX", "ACF/PACF", "ADF",
                    "VAR", "VECM", "Cointegration", "GARCH", "TS Filter"],
    "Cross Sect.": ["Fixed Effects", "Random Effects", "Hausman", "Dynamic Panel"],
    "Tests": ["Diagnostics", "Wald Test", "Bootstrap"],
    "Graph": ["Scatter", "Bar Chart", "Line Chart"],
}


# The core tabs with the added packs' modules joined on: to the end of a tab that already
# exists, or as new tabs after the core ones. A name already in a tab isn't shown twice
def all_tabs():
    tabs = {tab: list(names) for tab, names in MODULES.items()}
    for tab, names in addons.added_tabs().items():
        shown = tabs.setdefault(tab, [])
        for name in names:
            if name not in shown:
                shown.append(name)
    return tabs


# The name of every module on the toolbar, core modules first
def all_modules():
    return [name for names in all_tabs().values() for name in names]
