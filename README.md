# Coefflow

**Drag-and-drop econometrics.** Build an analysis by dropping modules onto a canvas and wiring them together: load data, clean it, fit models, test them and export publication-ready tables, without writing code.

<img width="1000" height="600" alt="The Coefflow screen: toolbar, workflows strip, canvas with Input Data wired to OLS, and the settings panel" src="https://github.com/user-attachments/assets/50aab429-e5ec-4365-b97e-00ecec34cdd1" />

## Features

- **Visual workflows.** Each module does one job. Connect an output to an input and the data flows through; click **Run** and every module turns green or red.
- **Data cleaning.** Join, split, filter, sort and convert tables, and make new fields with one line of Python each (logs, lags, growth rates, group means...).
- **Regression.** OLS (with robust or clustered errors), Auto OLS, Logit and Probit, with marginal effects and odds ratios.
- **Time series.** ARIMAX and Auto ARIMAX, ACF/PACF, ADF and KPSS stationarity tests, Durbin-Watson, and forecasting with prediction intervals.
- **Panel data.** Fixed and random effects with time effects and clustered errors, and the Hausman test.
- **Forecasting and validation.** Hold back test rows with Split, then score any model out of sample with Forecast.
- **Charts.** Scatter plots with a heteroscedasticity check, and bar charts by category.
- **Add-on packs.** Extra modules that load from **Settings > Add-ons**, including causal inference (IV 2SLS, RDD, DiD), ordered logit/probit, probability distributions, extra charts and journal-style regression tables exported to Excel. You can also write your own.
- **Built-in help.** Every module has a **?** explaining what it does and what each of its inputs and outputs holds.
- **Sample data included,** so you can try everything straight away.
- **Three themes:** Normal, Retro and Y2K.

## Getting started

You need **Python 3.10 to 3.13** (pygame, which draws the app, has no packages for newer versions yet).

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

On Linux you may also need Tk for the file dialogs, e.g. `sudo apt install python3-tk`.

**On Windows you can skip steps 2 and 3:** double-click **`Coefflow.exe`** in the Coefflow folder. It finds Python 3.10–3.13 (offering to install Python 3.13 for you if it is missing), installs the packages the first time, then opens the app. Keep it in the same folder as `main.py`. After the first run, Coefflow appears in the Start menu, so you can just search for it from the taskbar (run `Coefflow.exe --unregister` to remove that entry).

**Installing and updating on Windows:** **`CoefflowInstaller.exe`** downloads Coefflow with git (offering to install git if you don't have it) and adds it to the Start menu. Run it again at any time to check GitHub for a newer version and update to it. It can also convert a copy you downloaded as a ZIP so it can be updated. Updates never overwrite your settings, add-ons or outputs.

Then follow **[Building Your First Workflow](https://github.com/GregorSpock/Coefflow/wiki/Building-Your-First-Workflow)**, a ten-minute guide from an empty canvas to a saved regression.

## A first workflow

<img width="860" height="280" alt="Input Data wired to OLS, with the coefficients going to Output Data and the fitted data to Scatter" src="https://github.com/user-attachments/assets/94721230-4ecb-48f3-a7eb-2d9e9719a400" />

Load the sample data, regress unemployment on GDP growth with controls, save the coefficients to a CSV and plot the residuals: four modules and three lines.

## Documentation

The **[wiki](https://github.com/GregorSpock/Coefflow/wiki)** has a page for every module, covering its inputs, outputs, settings, how to read its results and worked examples.

- [Building Your First Workflow](https://github.com/GregorSpock/Coefflow/wiki/Building-Your-First-Workflow)
- [Core Modules](https://github.com/GregorSpock/Coefflow/wiki/Core-Modules)
- [Add-ons](https://github.com/GregorSpock/Coefflow/wiki/Add-ons)
- [Writing an Add-on](https://github.com/GregorSpock/Coefflow/wiki/Writing-an-Add-on)

## Modules

| Tab | Modules |
|---|---|
| Input/Output | Input Data, Output Data |
| Clean | Join, Split, Filter, Select, Sort, Convert, Gen |
| Base Reg. | OLS, Auto OLS, Logit, Probit |
| Time Series | Durbin-Watson, Forecast, ARIMAX, Auto ARIMAX, ACF/PACF, ADF |
| Cross Sect. | Fixed Effects, Random Effects, Hausman |
| Graph | Scatter, Bar Chart |

## Add-on packs

| Pack | Adds |
|---|---|
| Causal | IV 2SLS with weak-instrument and overidentification tests, regression discontinuity, difference in differences (2x2, TWFE, event study) |
| Descriptives | Summary statistics and correlations |
| Distributions | 15 continuous and discrete distributions: probabilities, critical values, tables and random samples |
| Graphs+ | Pie charts, histograms and box plots |
| Ordered | Ordered logit and probit, Brant and Lipsitz tests, ordinal association measures |
| Tables | Side-by-side regression tables in the style of economics journals, exported to Excel |

Load a pack with **Settings > Add-ons > Add pack...** and choose its folder.

> Add-on packs are Python code and run with your own permissions. Only load packs from sources you trust.

## Project layout

```
main.py            starts the app
engine/            workflow running, model fitting and shared statistics
modules/           one folder per core module (module.py), plus descriptions.txt for the ? help
ui/                the canvas, toolbar, panels, charts and themes (pygame)
addons/            add-on packs that have been loaded
```

## Contributing

Ideas, bug reports and new modules are welcome. Please open an [issue](https://github.com/GregorSpock/Coefflow/issues) describing the bug or idea, or to share a module or add-on pack you've built. See [Writing an Add-on](https://github.com/GregorSpock/Coefflow/wiki/Writing-an-Add-on) for how packs are put together. Submitted packs are reviewed before they are added to the main project. By submitting a contribution, you agree to license it under the AGPLv3 with the add-on exception.

## Built with

[pygame](https://www.pygame.org/) · [pandas](https://pandas.pydata.org/) · [statsmodels](https://www.statsmodels.org/) · [SciPy](https://scipy.org/) · [openpyxl](https://openpyxl.readthedocs.io/)

## License

Coefflow is free software, licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3) with an [add-on exception](ADDON-EXCEPTION.md).

- You can use, study, modify and share Coefflow, including for commercial work.
- If you share a modified version of Coefflow, or let people use one over a network, you must release its source code under the AGPLv3.
- Add-on packs and modules are exempt. Anyone, including companies, can write them and keep them private or license them however they like, as long as they don't contain or change Coefflow's own code. [ADDON-EXCEPTION.md](ADDON-EXCEPTION.md) has the exact terms.
- Your workflows, data and results (tables, charts and exported files) are yours, and the licence does not cover them.

> Coefflow is provided as is, without warranty of any kind. Check your results before relying on them, especially for published work or decisions.

Copyright (C) 2026 Gregor Burroughs
