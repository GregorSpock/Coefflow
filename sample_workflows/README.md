# Sample workflows

Eight workflows built on the sample data (20 UK regions, 1995 to 2024). They are numbered in the order to try them: each one adds new modules and ideas to the ones before. Open one with **Open**, click **Run**, then click any module to see its results.

Every Output Data module saves to the `output` folder next to this one.

| # | Workflow | What it shows | Modules |
|---|---|---|---|
| 01 | First regression | Unemployment on GDP growth with controls; the coefficients are saved and the residuals plotted against the fitted values | Input Data, OLS, Output Data, Scatter |
| 02 | Cleaning and charts | Dropping and renaming fields, sorting, making logs and growth rates within each region, filtering years, keeping a set of fields, turning region names into codes, then charting house prices against wages by region | Sort, Gen, Filter, Select, Convert, Bar Chart |
| 03 | Robust and Auto OLS | Wages with region dummies and errors clustered by region; unemployment with a lag of GDP growth and robust errors; Auto OLS choosing the controls for house prices | OLS (clustered, robust, text fields as dummies), Auto OLS |
| 04 | Logit, probit and holdout | Whether a region has high unemployment, by logit and probit, then a logit fitted on a random 80% of rows and scored on the other 20% | Logit, Probit, Split (shuffled share), Forecast |
| 05 | Time series diagnostics | London on its own: ADF and KPSS tests, ACF/PACF of unemployment in levels and differences, and a Durbin-Watson test on an OLS | Filter, ADF, ACF/PACF, Durbin-Watson |
| 06 | ARIMAX forecasting | ARIMAX and Auto ARIMAX fitted on 1995 to 2019 and forecast for 2020 to 2024 with 90% intervals, scored against what happened | Split (condition), ARIMAX, Auto ARIMAX, Forecast |
| 07 | Fixed and random effects | House prices across the regions by fixed and random effects, the Hausman test between them, a two-way fixed effects model with clustered errors, and each region's effect charted | Fixed Effects, Random Effects, Hausman |
| 08 | Full project pipeline | A 1995 baseline table joined back onto every year, new fields from it, a train/test split by year, OLS and two-way fixed effects, and out-of-sample forecasts exported to Excel | Join, every step above together |
