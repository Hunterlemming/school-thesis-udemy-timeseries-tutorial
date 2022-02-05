import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.__config__ import show


# Pre-processing
_df : pd.DataFrame = pd.read_csv('../data/s4_temperature_dataset.csv', index_col='DATE', parse_dates=True)
pd.DatetimeIndex(_df.index).freq = 'D'
_df.dropna(inplace=True)

_train = _df.iloc[:510,0]
_test = _df.iloc[510:,0]

exo = _df.iloc[:,1:4]           # Exogenous variables
_exo_train = exo.iloc[:510]
_exo_test = exo.iloc[510:]


# Decomposing the dataset
'''
from statsmodels.tsa.seasonal import seasonal_decompose

Decomp_Results = seasonal_decompose(_df['Temp'])
Decomp_Results.plot()
plt.show()              # From the plot we made the assumption that the seasonality is roughly 7 days
'''


# Finding the Parameters (p, d, q), (P, D, Q, s) - Grid-Search (Recommended)
from pmdarima import auto_arima
print(auto_arima(_df['Temp'], exogenous=exo, m=7, trace=True, D=1).summary())
# Result => (2,0,2)(0,1,1)[7]
