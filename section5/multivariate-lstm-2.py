import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_LOCATION = '../data/s5_electricity_consumption_dataset.csv'
MODEL_LOCATION = '../models/LSTM - Multivariate'


# Pre-processing
df : pd.DataFrame = pd.read_csv(DATA_LOCATION)
df.dropna(inplace=True)


'''
# Bonus: Check correlation between data
import seaborn as sn
sn.heatmap(df.corr())
plt.show()
'''


_train = df.iloc[:8712, 1:4].values
_test = df.iloc[8712:, 1:4].values


# Scaling the data
from sklearn.preprocessing import MinMaxScaler

sc : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))

_train_scaled = sc.fit_transform(_train)
_test_scaled = sc.fit_transform(_test)
_test_scaled = _test_scaled[:, 0:2]     # We don't need the target variables, just the predictors (other variables)


# Creating windows
x_train = []
y_train = []
window_size = 24

for i in range(window_size, len(_train_scaled)):
    x_train.append(_train_scaled[i-window_size:i, 0:3])     # All the independent and target variables
    y_train.append(_train_scaled[i, 2])                     # Just the target variable

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping, in order to make sure that x_train is in a proper shape before training
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))  # Number of features (dimension) = 3,
                                                                        # {Humidity, Temperature, Electricity}


# Developing LSTM Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def train_new_model() -> Sequential:
    _model = Sequential()

    _model.add(LSTM(units=70, return_sequences=True, input_shape=(x_train.shape[1], 3)))    # 3 -> same as dim above
    _model.add(Dropout(0.2))

    _model.add(LSTM(units=70, return_sequences=True))
    _model.add(Dropout(0.2))

    _model.add(LSTM(units=70, return_sequences=True))
    _model.add(Dropout(0.2))

    _model.add(LSTM(units=70))
    _model.add(Dropout(0.2))
    _model.add(Dense(units=1))

    _model.compile(optimizer='adam', loss='mean_squared_error')
    _model.fit(x_train, y_train, epochs=40, batch_size=32)
    _model.save(MODEL_LOCATION)

    return _model

    # Bonus: Getting a good epoch number
    plt.plot(range(len(_model.history.history['Loss'])), _model.history.history['Loss'])
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.show()              # 40 epochs are fine (first run was 80)


# Running model
from keras.models import load_model

try:
    _saved_model : Sequential = load_model(MODEL_LOCATION)
except OSError:
    _saved_model : Sequential = train_new_model()
