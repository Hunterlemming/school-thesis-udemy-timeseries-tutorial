import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Pre-processing
_df : pd.DataFrame = pd.read_csv('../data/s4_temperature_dataset.csv', index_col='DATE', parse_dates=True)
pd.DatetimeIndex(_df.index).freq = 'D'
_df.dropna(inplace=True)

_train = _df.iloc[:510,0]
_test = _df.iloc[510:,0]

exo = _df.iloc[:,1:4]           # Exogenous variables
_exo_train = exo.iloc[:510]
_exo_test = exo.iloc[510:]


# Checking correlations with Exogenous (independent) variables
import seaborn as sn

sn.heatmap(_df.corr())
plt.show()
