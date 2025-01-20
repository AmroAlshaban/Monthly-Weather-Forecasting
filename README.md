# Monthly-Weather-Forecasting

An analysis of monthly weather data from January 2001 to December 2023 was conducted to understand trends, seasonality, and residual patterns. The dataset was extracted, transformed and decomposed into its seasonal and residual components, with trend adjustments made as necessary. Stationarity was rigorously tested using methods like the Augmented Dickey-Fuller (ADF) test.

To identify correlations and lag patterns, Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots were analyzed. These insights guided the selection of parameters for fitting Autoregressive (AR) or Moving Average (MA) models. An Autoregressive Moving Average (ARMA) model was fitted to the residuals, with parameters $p = 1$ and $q = 1$ selected by minimizing the corrected Akaike Information Criterion (AICC). 

Using the finalized model, monthly temperature forecasts for the year 2024 were computed.

# Report

View the entire report [here](https://github.com/MathoVerse100/Monthly-Weather-Forecasting-in-Jordan/blob/main/report.md).

# Data

All data was scraped from [Weather Spark](https://weatherspark.com).
