import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_LOCATION = '../data/s5_solar_dataset.csv'
MODEL_LOCATION = '../models/LSTM - Univariate'


# Pre-processing
df : pd.DataFrame = pd.read_csv(DATA_LOCATION)
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


# Developing LSTM Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def train_new_model():
    # Creating the model-base
    _model = Sequential()


    # Adding a new layer
    _model.add(LSTM(                
        units=60,                           # Number of neurons in this layer
        return_sequences=True,              # Whether to return the last output sequence or the full sequence
        input_shape=(x_train.shape[1], 1)   # The shape and dimension (features) of the input, used at weight creation
        ))

    # Avoiding overfitting
    _model.add(Dropout(0.2))                # Drop 20% of the neurons


    # Adding more layers
    _model.add(LSTM(units=60, return_sequences=True))   # We don't need to specify an input-shape anymore
    _model.add(Dropout(0.2))

    _model.add(LSTM(units=60, return_sequences=True))
    _model.add(Dropout(0.2))


    # Creating final layer
    _model.add(LSTM(units=60))
    _model.add(Dropout(0.2))

    _model.add(Dense(units= 1))     # We want 1 output (prediction)

    # Compiling our model
    _model.compile(optimizer='adam', loss='mean_squared_error')     # Choosing optimizer and the way loss is calculated

    # Fitting model
    _model.fit(x_train, y_train, epochs=15, batch_size=32)

    # Saving model
    _model.save(MODEL_LOCATION)

    return _model

    # Bonus: Getting a good epoch number
    plt.plot(range(len(_model.history.history['Loss'])), _model.history.history['Loss'])        # Check history loss
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.show()              # 15 is fine


# Running model
from keras.models import load_model

try:
    _saved_model : Sequential = load_model(MODEL_LOCATION)
except OSError:
    _saved_model : Sequential = train_new_model()
