The temperature data analyzed in this report was collected using a custom-built web scraper. This scraper extracted daily average temperature values
for the period spanning January 2001 to December 2023. 

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from datetime import datetime
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

Summary statistics for the monthly average temperatures are computed using ```.groupby()``` method as follows:

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
