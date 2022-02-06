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
        units=60,                           # Number of neurons in this layer (hyperparameter, adjustable)
        return_sequences=True,              # Whether to return the last output or the full sequence (True till the last)
        input_shape=(x_train.shape[1], 1)   # The shape and dimension (features) of the input, used at weight creation
        ))

    # Avoiding overfitting
    _model.add(Dropout(0.2))                # Drop 20% of the neurons (hyperparameter, adjustable)


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
    _model.fit(x_train, y_train, epochs=15, batch_size=32)          # epochs and batch_size are hyperparameters (adjustable)

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
    _trained_model : Sequential = load_model(MODEL_LOCATION)
except OSError:
    _trained_model : Sequential = train_new_model()

prediction_test = []

Batch_one = _train_scaled[-window_size:]                # Selecting last 24 hours
Batch_new = Batch_one.reshape((1, window_size, 1))

for i in range(len(_test_scaled)):                                      # Predicting 48 (len(test)) hours into the future
    First_Pred = _trained_model.predict(Batch_new)[0]
    prediction_test.append(First_Pred)
    Batch_new = np.append(Batch_new[:,1:,:], [[First_Pred]], axis=1)    # Moving window appending the predicted value

prediction_test = np.array(prediction_test)
predictions = sc.inverse_transform(prediction_test)     # Inverting the normalization transform we started with


# Evaluating the result
import math
from sklearn.metrics import mean_squared_error, r2_score

RMSE = math.sqrt(mean_squared_error(_test, predictions))
Rsquare = r2_score(_test, predictions)

print(f"Root-mean-square-deviation = {RMSE}\nCoefficient of determination = {Rsquare}")


# Visualizing the result
plt.plot(_test, color='red', label="Actual Values")
plt.plot(predictions, color='blue', label="Predicted Values")
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('Solar Irradiance')
plt.legend()
plt.show()
