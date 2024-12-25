The temperature data analyzed in this report was collected using a custom-built web scraper. This scraper extracted daily average temperature values
for the period spanning January 2001 to December 2023. 

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
dark_style = {
    'figure.facecolor': '#212946',
    'axes.facecolor': '#212946',
    'savefig.facecolor':'#212946',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': '#2A3459',
    'grid.linewidth': '1',
    'text.color': '0.9',
    'axes.labelcolor': '0.9',
    'xtick.color': '0.9',
    'ytick.color': '0.9',
    'font.size': 12 }
plt.rcParams.update(dark_style)

from pylab import rcParams
rcParams['figure.figsize'] = (18,7)


daily_data = pd.read_excel('/kaggle/input/daily-weather-temperature-2001-2024/Daily Weather 2001 - 2024.xlsx')
```

We check for any missing data:
```python
daily_data[daily_data['Temperature'].isna()]
```
| Date                |   Temperature |
|:--------------------|--------------:|
| 2013-05-30 00:00:00 |           nan |
| 2013-05-31 00:00:00 |           nan |

Of all the daily average temperature data, only two dates were missing (May 30, 2013, and May 31, 2013). These missing values were imputed using the 
mean of the yearly average temperatures for May from 2001 to 2023:

```python
daily_data.loc[daily_data["Temperature"].isna(), "Temperature"] = daily_data[daily_data["Date"].dt.month == 5]["Temperature"].mean().round(1)

plt.figure(figsize=(20, 6))
plt.plot(daily_data['Date'], daily_data['Temperature'], color='green')
plt.show()
```
![Daily Temperature Plot](https://i.ibb.co/7SmQDqW/results-3-0.png)

We aggregate the data to transform it into monthly averages, then split the data into two datasets, one containing data up to 2023 and the other contains the data from the start of 2024. A plot of the monthly data is shown:

```python
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
daily_data.set_index('Date', inplace=True)

monthly_data = daily_data.resample('M').mean()
monthly_data.reset_index(inplace=True)
monthly_data['Temperature'] = monthly_datanew['Temperature'].round().astype(int)

test_set = monthly_data.iloc[-4:]
monthly_data = monthly_data.iloc[:-4]
```
```python
plt.figure(figsize=(10, 6))
plt.plot(monthly_data['Date'], monthly_data['Temperature'], color='red')
plt.title('Monthly Average Temperatures (2001 - 2023)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
```
![Monthly Temperature Plot](https://i.ibb.co/GQVGTJP/download.png)

Summary statistics for the monthly average temperatures are computed using the ```.groupby()``` method as follows:

```python
monthly_data_copy = monthly_data.copy()
monthly_data_copy["Month"] = monthly_data_copy["Date"].dt.month
summary_statistics = monthly_data_copy[["Temperature", "Month"]].groupby("Month").describe().reset_index(drop=True).round(2).T
summary_statistics.rename(columns=dict(zip([i for i in range(12)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])), inplace=True)
summary_statistics.index = [item[1] for item in summary_statistics.index]
summary_statistics = summary_statistics[summary_statistics.index != 'count']
summary_statistics["All"] = monthly_data_copy["Temperature"].describe().round(2)[monthly_data_copy["Temperature"].describe().index != 'count']
```

|      |   January |   February |   March |   April |   May |   June |   July |   August |   September |   October |   November |   December |   All |
|:-----|----------:|-----------:|--------:|--------:|------:|-------:|-------:|---------:|------------:|----------:|-----------:|-----------:|------:|
| mean |      8.83 |      10.17 |   13.43 |   17.61 | 22.13 |  25.04 |  27.17 |    27.13 |       25    |     21.61 |      15.57 |      10.74 | 18.7  |
| std  |      1.3  |       1.3  |    2.04 |    1.62 |  1.25 |   1.11 |   1.03 |     1.14 |        1.41 |      1.41 |       1.56 |       1.42 |  6.71 |
| min  |      6    |       8    |    9    |   15    | 20    |  23    |  25    |    25    |       23    |     19    |      12    |       8    |  6    |
| 25%  |      8    |       9.5  |   12.5  |   16.5  | 21    |  24    |  27    |    26    |       24    |     20    |      15    |       9.5  | 12    |
| 50%  |      9    |      10    |   13    |   17    | 22    |  25    |  27    |    27    |       25    |     22    |      16    |      11    | 20    |
| 75%  |      9.5  |      11    |   15    |   19    | 23    |  26    |  27.5  |    28    |       26    |     22.5  |      16    |      12    | 25    |
| max  |     12    |      13    |   17    |   21    | 25    |  27    |  30    |    30    |       29    |     24    |      19    |      13    | 30    |

The following is a plot of the mean of the yearly average temperatures for every month:

```python
plt.figure(figsize=(8, 5))
plt.plot(range(1, 13), [summary_statistics[i].loc['mean'][0] for i in range(1, 13)], color='magenta')
plt.title("Mean Monthly Average Temperature per month from 2001 to 2023")
plt.xlabel("Month")
plt.ylabel("Monthly Average Temperature")
plt.show()
```
![Mean Monthly Temperature (2001–2023)](https://i.ibb.co/9pjcS1z/download.png)

We observe constant periodicity of 12 months, with no apparent trend and the amplitudes do not significantly vary in time, so we adopt an additive time series model. Indeed, we fit a linear regression model into our data, then perform a t-test to see if the slope is significantly different from zero. Our p-value is equal to 0.236 which is significantly greater than 0.05, indicating no statistical evidence that our slope varies from zero. The linear model summary is shown below (performed using ```statsmodels``` library):

```python
X = monthly_data_copy.index
y = monthly_data_copy['Temperature'].values

X = sm.add_constant(X)

trend_ = sm.OLS(y, X).fit()

print(trend_.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.005
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     1.411
Date:                Wed, 25 Dec 2024   Prob (F-statistic):              0.236
Time:                        10:03:23   Log-Likelihood:                -915.74
No. Observations:                 276   AIC:                             1835.
Df Residuals:                     274   BIC:                             1843.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         17.8757      0.805     22.211      0.000      16.291      19.460
x1             0.0060      0.005      1.188      0.236      -0.004       0.016
==============================================================================
Omnibus:                     1188.792   Durbin-Watson:                   0.328
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               23.928
Skew:                          -0.163   Prob(JB):                     6.37e-06
Kurtosis:                       1.595   Cond. No.                         317.
==============================================================================
```

Our data follows a no–trend seasonal additive model of the form: $Y_t = S_t + \epsilon_t$ where $Y_t$ is an observed value, $S_t$ is the seasonal component, and $\epsilon_t$ is the residual (or error), all at time $t$. We perform a seasonal decomposition on our data to separate the seasonal component and
residuals. Our results are visualized below:

```python
additive_monthly = seasonal_decompose(monthly_data['Temperature'], model='additive', period=12)

additive_monthly.plot().suptitle('Additive Model')
plt.tight_layout()
plt.tight_layout()
plt.show()
```
![Seasonal Decomposition Plot](https://i.ibb.co/RyX0Sbm/results-5.png)

Our residuals appear stationary, which we confirm by applying an Augmented Dickey Fuller test, resulting in a p-value of approximately 0.0003585, far below 0.05 indicating strong stationarity:

```python
print(adfuller(monthly_data['Temperature'] - additive_monthly.seasonal))

# Output: (-4.353415546369792, 0.000358545753143946, 11, 264, {'1%': -3.455365238788105, '5%': -2.8725510317187024, '10%': -2.5726375763314966}, 877.9825682189721)
# The p-value is given by 0.000358545753143946, less than the 5% significance level, indicating the data is stationary.
```

After achieving stationarity of residuals, our objective is to fit an appropriate Autoregressive Moving Average (ARMA) model to the residuals. We will then incorporate the seasonal component into our residual forecasts to predict the monthly average temperature for the year 2024. Our ARMA model will be chosen by minimizing the
corrected Akaike Information Criterion (AICC). This will be achieved using ```pmdarima``` library's ```autoarima()``` function which automatically finds the best ARMA model that minimizes the AICC.

Results are shown below:

```python
from pmdarima import auto_arima

auto_model = auto_arima(monthly_resids, test='adf', seasonal=False)
print(auto_model.summary())
```

```
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  276
Model:               SARIMAX(1, 0, 1)   Log Likelihood                -467.095
Date:                Sun, 22 Dec 2024   AIC                            942.189
Time:                        16:32:11   BIC                            956.671
Sample:                             0   HQIC                           948.000
                                - 276                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      4.8587      2.507      1.938      0.053      -0.055       9.773
ar.L1          0.7403      0.134      5.524      0.000       0.478       1.003
ma.L1         -0.5352      0.165     -3.245      0.001      -0.859      -0.212
sigma2         1.7311      0.138     12.507      0.000       1.460       2.002
===================================================================================
Ljung-Box (L1) (Q):                   0.17   Jarque-Bera (JB):                 4.59
Prob(Q):                              0.68   Prob(JB):                         0.10
Heteroskedasticity (H):               1.13   Skew:                             0.26
Prob(H) (two-sided):                  0.57   Kurtosis:                         3.36
===================================================================================
```

Diagnostics are plotted below (residuals, histograms, KDE, normal Q-Q, and correlogram plots):

```python
auto_model_results.plot_diagnostics()
plt.show()
```
![Diagnostics Plot](https://i.ibb.co/cgZ17s7/results-6.png)

We now use our model to find forecasts for all months in 2024. By calculating residual forecasts and then adding the seasonal components, we calculate our predictions which are plotted below:

```python
auto_model_results = ARIMA(monthly_resids, order=(1, 0, 1)).fit()

plt.plot(monthly_data['Date'], monthly_data['Temperature'], color='red', label='Actual')
plt.plot([datetime(year=2023, month=12, day=15)] + [datetime(year=2024, month=i, day=15) for i in range(1, 13)], [monthly_data['Temperature'].iloc[-1]] + (auto_model_results.forecast(steps=12).values + additive_monthly.seasonal[:12].values).tolist(), color='orange', label='Forecast')
plt.title("Actual Values and Forecasts")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
plt.show()
```

![Actual vs Forecast Plot](https://i.ibb.co/020jSQ6/results-7.png)

Our fitted model is given by $𝑌_𝑡 = 4.8587 + 0.7403 𝑌_{t-1} − 0.5352 𝑍_{t-1} + 𝑍_t$. From the diagnostics plot, the Q-Q plot indicates that the quantiles are very close to that of a normal distribution and the histogram/KDE plots show that the symmetry and tails of the data are close to a normal distribution, suggesting that our residuals could indeed be normal. Applying a Jarque Bera normality test computes a p-value of 0.1049, which implies no significant evidence that the residuals are not normally distributed. This provides enough evidence to regard our residuals as normally distributed.

The first 4 forecasts were compared with the true values, the results shown below:
