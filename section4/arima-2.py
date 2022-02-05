import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Pre-processing
_df : pd.DataFrame = pd.read_csv('../data/s4_temperature_dataset.csv', index_col='DATE', parse_dates=True)
pd.DatetimeIndex(_df.index).freq = 'D'
_df.dropna(inplace=True)
_df = pd.DataFrame(_df['Temp'])

_train = _df.iloc[:510,0]
_test = _df.iloc[510:,0]


# Finding the Parameters (p, d, q) - With plotting ACF and PACF
'''
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(_train, lags=50)           # Number of backshifts we want to consider
plot_pacf(_train, lags=50)
plt.show()
'''


# Finding the Parameters (p, d, q) - Grid-Search (Recommended)
from pmdarima import auto_arima

auto_arima(_train, trace=True)  
# Result => p = 1, d = 1, q = 2
