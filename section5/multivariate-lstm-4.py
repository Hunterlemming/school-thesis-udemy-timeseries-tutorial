from cProfile import label
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
    _trained_model : Sequential = load_model(MODEL_LOCATION)
except OSError:
    _trained_model : Sequential = train_new_model()

prediction_test = []

Barch_one = _train_scaled[-window_size:]
Batch_new = Barch_one.reshape((1, window_size, 3))

for i in range(len(_test_scaled)):
    # Predicting next value
    First_Pred = _trained_model.predict(Batch_new)[0]
    prediction_test.append(First_Pred)

    # Creating a new row, to base our next prediction on
    New_var = _test_scaled[i,:]                                 # We already know the independent variables,
    New_var = New_var.reshape(1, 2)                             # so the only thing we have to do is creating
    New_test = np.insert(New_var, 2, [First_Pred], axis=1)      # a new row with these and the freshly predicted
    New_test = New_test.reshape(1, 1, 3)                        # target variable (First_Pred)

    Batch_new = np.append(Batch_new[:,1:,:], New_test, axis=1)  # We can update Batch_new with this new row

prediction_test = np.array(prediction_test)

# We can't simply use the sc to convert our result back to the original scale,
# since we used an array of 3 to normalize and now we only have 1 value
SI = MinMaxScaler(feature_range=(0,1))
y_scale = _train[:, 2:]
SI.fit(y_scale)             # However we can create a scaler for the last row

predictions = SI.inverse_transform(prediction_test)


# Evaluating the result
real_values = _test[:, 2]

import math
from sklearn.metrics import mean_squared_error, r2_score

RMSE = math.sqrt(mean_squared_error(real_values, predictions))
Rsquare = r2_score(real_values, predictions)

print(f"Root-mean-square-deviation = {RMSE}\nCoefficient of determination = {Rsquare}")

def mean_absolute_precentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

print(f"Mean-absolute-percentage-error = {mean_absolute_precentage_error(real_values, predictions)}")


# Visualizing the result
plt.plot(real_values, color='red', label="Actual Electrical Consumption")
plt.plot(predictions, color='blue', label="Predicted Electrical Consumption")
plt.title("Electrical Consumption Prediction")
plt.xlabel("Time (hr)")
plt.ylabel("Electrical Demand (MW)")
plt.legend()
plt.show()
