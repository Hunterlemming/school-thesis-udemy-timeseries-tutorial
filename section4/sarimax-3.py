import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# Developing SARIMAX model -- (2,0,2)(0,1,1)[7]
from statsmodels.tsa.statespace.sarimax import SARIMAX

Model = SARIMAX(_train, _exo_train, order=(2, 0, 2), seasonal_order=(0, 1, 1, 7))

Model = Model.fit()

prediction = Model.predict(len(_train), len(_train)+len(_test)-1, exog=_exo_test, typ='Levels')

plt.plot(_test, color='red', label="Actual Temp")
plt.plot(prediction, color='blue', label="Predicted Temp")
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(_test, prediction))
print(f"Difference between predicted and test values: {rmse}")
