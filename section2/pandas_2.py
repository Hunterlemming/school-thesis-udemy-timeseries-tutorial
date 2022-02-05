import pandas as pd
import numpy as np


# Dataframes
_data = np.array([[20, 10, 8], [25, 8, 10], [27, 5, 3], [30, 9, 7]], dtype=np.int32)
print(_data)


_dataSet = pd.DataFrame(_data)                                      # Transforming a npArray to a DataFrame
print(_dataSet)

_dataSet = pd.DataFrame(_data, index=['S1', 'S2', 'S3', 'S4'])      # Assigning labels to rows
print(_dataSet)

_dataSet.columns = ['Age', 'Grade1', 'Grade2']                      # Assigning labels to columns
print(_dataSet)

_dataSet['Grade3'] = [9, 6, 7, 10]                                  # Adding another columns
print(_dataSet)
