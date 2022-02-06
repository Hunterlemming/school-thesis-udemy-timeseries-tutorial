import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Pre-processing
df : pd.DataFrame = pd.read_csv('../data/s5_solar_dataset.csv')
df.dropna(inplace=True)

_train = df.iloc[:8712, 1:2].values     # We do not need the date-column
_test = df.iloc[8712:, 1:2].values      # .values turns the dataset into arrays


# Scaling the data (eg.: normalizing)
from sklearn.preprocessing import MinMaxScaler

sc : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))

_train_scaled = sc.fit_transform(_train)
_test_scaled = sc.fit_transform(_test)


# Creating lookback-window
x_train = []
y_train = []

window_size = 24    # Look at the seasonality of our dataset! Since we work with days divided into
                    # hours, our window size COULD be 24 representing that

for i in range(window_size, len(_train_scaled)):
    x_train.append(_train_scaled[i-window_size:i, 0:1])
    y_train.append(_train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
