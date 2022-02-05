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


# Developing ARIMA model
from statsmodels.tsa.arima.model import ARIMA

A_Model = ARIMA(_train, order=(1, 1, 2), freq='D')    # Check arima-2 Grid-Search for ARIMA parameters

predictor = A_Model.fit()
# print(predictor.summary())

Predicted_Results = predictor.predict(start=len(_train), end=len(_train)+len(_test)-1, typ='Levels')

plt.plot(_test, color='red', label="Actual Temp")
plt.plot(Predicted_Results, color='blue', label="Predicted Temp")
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()


import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(_test, Predicted_Results))
print(f"Difference between predicted and test values: {rmse}")
