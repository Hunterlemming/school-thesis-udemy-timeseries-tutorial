import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# Setting up data
_df : pd.DataFrame = pd.read_csv('../data/s4_temperature_dataset.csv', index_col='DATE', parse_dates=True)
pd.DatetimeIndex(_df.index).freq = 'D'      # Setting datetime frequency to Daily
_df.dropna(inplace=True)                    # Dropping null-rows
_df = pd.DataFrame(_df['Temp'])             # Cutting all columns besides 'Temp', since we only need temperature


# Splitting data to training and test sets
_train = _df.iloc[:510,0]
_test = _df.iloc[510:,0]


# Decomposition of data
Decomp_results = seasonal_decompose(_df)
Decomp_results.plot()
plt.show()
