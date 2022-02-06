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
