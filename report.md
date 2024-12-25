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
