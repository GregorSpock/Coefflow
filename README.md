# Coefflow

**Drag-and-drop econometrics.** Build an analysis by dropping modules onto a canvas and wiring them together: load data, clean it, fit models, test them and export publication-ready tables, without writing code.

<img width="1000" height="600" alt="The Coefflow screen: toolbar, workflows strip, canvas with Input Data wired to OLS, and the settings panel" src="https://github.com/user-attachments/assets/50aab429-e5ec-4365-b97e-00ecec34cdd1" />

## Features

- **Visual workflows.** Each module does one job. Connect an output to an input and the data flows through; click **Run** and every module turns green or red. Each workflow file notes the version of Coefflow that last saved it, and opening one saved by a newer version gives a warning.
- **Runs only what changed.** A run works on its own thread, so the window stays in use and **Stop** ends it early. Modules whose settings and inputs are the same as last time keep their results, so changing one setting reruns that module and the ones below it, not the whole workflow. **Shift + Run** works everything out again, and right-clicking a module gives **Run to here**.
- **Full editing.** Undo and redo for every change, copy, cut, paste and duplicate, and selecting several modules at once to move or delete them together.
- **Typed ports.** A port carries either a table or a fitted model. Model ports are drawn in their own colour, and a line that would hand a module something it can't use is refused with the reason.
- **Reads your data as it comes.** CSV and text, Excel and OpenDocument, Stata, SPSS, SAS, Parquet, Feather, JSON, gretl and SQLite files, plus SQL queries against SQLite or a database server, and series downloaded from FRED, the World Bank, the OECD and DBnomics. Database queries can only read, and a server's password is never saved in a workflow (server databases need `pip install sqlalchemy` and the database's driver). Save tables as CSV, Excel, Stata, Parquet, JSON or LaTeX.
- **Data cleaning.** Join, split, filter, sort and convert tables, choose and reorder fields, and make new fields with one line of Python each (logs, lags, growth rates, group means...). Group By summarises rows in groups, Pivot reshapes between long and wide (Stata-style stubs included), Concat stacks tables, Missing Values drops, fills or interpolates gaps within groups, and Dates reads dates in any common form, including 2020Q1 and gretl's 1990:1, and splits them into parts and trends.
- **See every row.** Right-click a module and choose **View data**, or click **View all** above a preview, to open its tables in a window of their own: sort, search, filter with a condition, and see a summary and the most common values of every column.
- **Regression.** OLS (with robust, clustered or Newey-West HAC errors), Auto OLS, Logit and Probit, with marginal effects and odds ratios. Count models (Poisson, negative binomial and their zero-inflated forms, with exposure), Tobit for censored outcomes, Heckman selection (two-step or maximum likelihood) and quantile regression across any set of quantiles.
- **Diagnostics and tests.** One module runs Jarque-Bera, Breusch-Pagan, White, RESET, Breusch-Godfrey, ARCH-LM, Durbin-Watson, Chow and CUSUM tests, each with a plain-words verdict and what to do about it, plus VIFs, leverage, studentized residuals and Cook's distance, and residual, Q-Q and CUSUM plots. Wald tests of any linear restrictions (using the model's own robust, clustered or HAC errors, and across equations of a VAR) and LR tests of nested models. Bootstrap any model that takes one table, by pairs, clusters or blocks of periods, or run a permutation test.
- **Time series.** ARIMAX (with seasonal SARIMA terms) and Auto ARIMAX, ACF/PACF, ADF and KPSS stationarity tests, Durbin-Watson, and forecasting with prediction intervals. VARs with impulse responses, variance decompositions and Granger causality; Johansen and Engle-Granger cointegration tests and VECMs; GARCH, GJR-GARCH and EGARCH with volatility forecasts; and Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald and Hamilton filters and STL seasonal adjustment.
- **Panel data.** Fixed and random effects with time effects and clustered errors, the Hausman test, and dynamic panels by Arellano-Bond difference GMM or Blundell-Bond system GMM, with Windmeijer-corrected errors and the AR(2) and Hansen tests.
- **Forecasting and validation.** Hold back test rows with Split, then score any model out of sample with Forecast.
- **Charts.** Scatter plots with a heteroscedasticity check, bar charts by category, and line charts of several fields over time. Every chart has a **Settings** button to change its title, axis labels, scales and colours, and **Export PNG** to save it as a picture.
- **Add-on packs.** Extra modules that load from **Settings > Add-ons**, including causal inference (IV 2SLS, RDD, DiD), ordered logit/probit, probability distributions, extra charts, journal-style regression tables exported to Excel or LaTeX, and a Psych pack for psychologists (t-tests, ANOVA, post hoc and rank tests, Cronbach's alpha, factor analysis, mediation). You can also write your own.
- **Built-in help.** Every module has a **?** explaining what it does and what each of its inputs and outputs holds.
- **Sample data included,** so you can try everything straight away.
- **Three themes:** Retro, Dark and Light.

## Getting started

You need **Python 3.10 or newer**.

1. Download the code: click **Code > Download ZIP** above and unzip it, or clone it:

   ```bash
   git clone https://github.com/GregorSpock/Coefflow.git
   cd Coefflow
   ```

2. Install the packages it uses:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python main.py
   ```

**On Windows you can skip steps 2 and 3:** double-click **`Coefflow.exe`** in the Coefflow folder. It finds Python 3.10–3.13 (offering to install Python 3.13 for you if it is missing), installs the packages the first time (a download of about 250 MB), then opens the app. If the app can't open, it says why. Keep it in the same folder as `main.py`. After the first run, Coefflow appears in the Start menu, so you can just search for it from the taskbar (run `Coefflow.exe --unregister` to remove that entry).

**Installing and updating on Windows:** **`CoefflowInstaller.exe`** downloads Coefflow with git (offering to install git if you don't have it) and adds it to the Start menu. Run it again at any time to check GitHub for a newer version and update to it. It can also convert a copy you downloaded as a ZIP so it can be updated. Updates never overwrite your settings, workflows, outputs or the add-on packs you've added yourself, and if you have changed one of Coefflow's own files that an update replaces, your copy is kept in a `backup` folder first.

Then follow **[Building Your First Workflow](https://github.com/GregorSpock/Coefflow/wiki/Building-Your-First-Workflow)**, a ten-minute guide from an empty canvas to a saved regression.

## A first workflow

<img width="860" height="280" alt="Input Data wired to OLS, with the coefficients going to Output Data and the fitted data to Scatter" src="https://github.com/user-attachments/assets/94721230-4ecb-48f3-a7eb-2d9e9719a400" />

Load the sample data, regress unemployment on GDP growth with controls, save the coefficients to a CSV and plot the residuals: four modules and three lines.

## Documentation

The **[wiki](https://github.com/GregorSpock/Coefflow/wiki)** has a page for every module, covering its inputs, outputs, settings, how to read its results and worked examples.

- [Building Your First Workflow](https://github.com/GregorSpock/Coefflow/wiki/Building-Your-First-Workflow)
- [Core Modules](https://github.com/GregorSpock/Coefflow/wiki/Core-Modules)
- [Add-ons](https://github.com/GregorSpock/Coefflow/wiki/Add-ons)
- [Command Line](https://github.com/GregorSpock/Coefflow/wiki/Command-Line)
- [What's New](https://github.com/GregorSpock/Coefflow/wiki/Whats-New)
- [Writing an Add-on](https://github.com/GregorSpock/Coefflow/wiki/Writing-an-Add-on)

## Command line

Workflows also run without the window, from a terminal, a script, a scheduled task or CI:

```bash
python main.py run analysis.confw
```

```bash
python main.py run analysis.confw --export results --set 3.hac=true
```

`run` prints each module as it finishes and exits with 0 if every module worked, 1 if any failed and 2 if the workflow couldn't be opened. `--export FOLDER` saves every module's outputs (`--format` picks the file type), `--set ID.SETTING=VALUE` changes a setting for that run only, `--only ID` runs a module and what it needs, and `--show ID` prints a module's outputs. `python main.py check analysis.confw` lists the modules and their connections without running, and `python main.py modules` lists every module.

## Keyboard and mouse

| Do this | To |
|---|---|
| Drag a module from the toolbar, or double-click it | Place it on the canvas |
| Drag from an output port to an input port | Connect two modules. Drag a connected input away to unplug it |
| Click a module | Open its settings |
| Drag the background / scroll | Move around / zoom |
| Shift + drag the background, Ctrl + click, Ctrl + A | Select several modules |
| Delete | Remove the selected modules and lines |
| Ctrl + Z, Ctrl + Y | Undo, redo |
| Ctrl + C, Ctrl + X, Ctrl + V, Ctrl + D | Copy, cut, paste, duplicate |
| F5, Shift + F5, Esc | Run what changed, run everything, stop the run |
| Ctrl + S, Ctrl + Shift + S, Ctrl + O, Ctrl + T, Ctrl + W | Save, save as, open, new workflow, close workflow |
| F11 | Full screen |

## Modules

| Tab | Modules |
|---|---|
| Input/Output | Input Data, Output Data, Database, Online Data |
| Clean | Join, Split, Filter, Select, Sort, Convert, Gen, Group By, Pivot, Concat, Missing Values, Dates |
| Base Reg. | OLS, Auto OLS, Logit, Probit |
| More Reg. | Count, Tobit, Heckman, Quantile Reg |
| Time Series | Durbin-Watson, Forecast, ARIMAX, Auto ARIMAX, ACF/PACF, ADF, VAR, VECM, Cointegration, GARCH, TS Filter |
| Cross Sect. | Fixed Effects, Random Effects, Hausman, Dynamic Panel |
| Tests | Diagnostics, Wald Test, Bootstrap |
| Graph | Scatter, Bar Chart, Line Chart |

## Add-on packs

| Pack | Adds |
|---|---|
| Causal | IV 2SLS with weak-instrument and overidentification tests, regression discontinuity, difference in differences (2x2, TWFE, event study) |
| Descriptives | Summary statistics and correlations |
| Distributions | 15 continuous and discrete distributions: probabilities, critical values, tables and random samples |
| Graphs+ | Pie charts, histograms and box plots |
| Ordered | Ordered logit and probit, Brant and Lipsitz tests, ordinal association measures |
| Tables | Side-by-side regression tables in the style of economics journals, exported to Excel or LaTeX |

Load a pack with **Settings > Add-ons > Add pack...** and choose its folder.

> Add-on packs are Python code and run with your own permissions. Only load packs from sources you trust.

## Project layout

```
main.py            starts the app
engine/            everything that works without a window: running workflows, model fitting,
                   shared statistics, and what each module's panel and chart hold
modules/           one folder per core module (module.py), plus descriptions.txt for the ? help
ui/                the window: canvas, toolbar, panels, charts and themes (Qt, through PySide6)
addons/            add-on packs that have been loaded
tests/             the test suite
version.txt        the version number, shown on the opening menu
```

Modules never touch the window. A module's `run` takes tables and gives tables, its panel is described with `engine.panel.Panel`, and its chart is drawn through the painter in `engine/draw.py`, so every module can be loaded, run and tested with no screen.

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest tests
```

The tests check the estimators against NIST's certified Longley results and against the textbook formulas written out in numpy (including Newey-West errors, every diagnostic test, VIFs and influence measures, and Wald and LR tests), read back every file format Coefflow writes, compare every sample workflow and add-on module with stored results so a change in any number is caught, open every module's panel and draw every chart without a window, and drive the real window: drawing lines with the mouse, running, undoing, copying and pasting.

## Contributing

Ideas, bug reports and new modules are welcome. Please open an [issue](https://github.com/GregorSpock/Coefflow/issues) describing the bug or idea, or to share a module or add-on pack you've built. See [Writing an Add-on](https://github.com/GregorSpock/Coefflow/wiki/Writing-an-Add-on) for how packs are put together. Submitted packs are reviewed before they are added to the main project. By submitting a contribution, you agree to license it under the AGPLv3 with the add-on exception.

## Built with

[Qt](https://www.qt.io/) through [PySide6](https://doc.qt.io/qtforpython-6/) · [pandas](https://pandas.pydata.org/) · [statsmodels](https://www.statsmodels.org/) · [SciPy](https://scipy.org/) · [openpyxl](https://openpyxl.readthedocs.io/) · [pyreadstat](https://github.com/Roche/pyreadstat) · [PyArrow](https://arrow.apache.org/docs/python/)

## License

Coefflow is free software, licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3) with an [add-on exception](ADDON-EXCEPTION.md).

- You can use, study, modify and share Coefflow, including for commercial work.
- If you share a modified version of Coefflow, or let people use one over a network, you must release its source code under the AGPLv3.
- Add-on packs and modules are exempt. Anyone, including companies, can write them and keep them private or license them however they like, as long as they don't contain or change Coefflow's own code. [ADDON-EXCEPTION.md](ADDON-EXCEPTION.md) has the exact terms.
- Your workflows, data and results (tables, charts and exported files) are yours, and the licence does not cover them.

> Coefflow is provided as is, without warranty of any kind. Check your results before relying on them, especially for published work or decisions.

Copyright (C) 2026 Gregor Burroughs
