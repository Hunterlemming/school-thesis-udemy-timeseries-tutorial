import numpy as np
import pandas as pd


_data = np.array([[20, 10, 8, 9], [25, 8, 10, 6], [27, 5, 3, 7], [30, 9, 7, 10]], dtype=np.int32)
_dataSet = pd.DataFrame(_data, index=['S1', 'S2', 'S3', 'S4'], columns=['Age', 'Grade1', 'Grade2', 'Grade3'])


print(_dataSet)
_dataSet = _dataSet.drop('Grade1', axis=1)              # Dropping column
print(_dataSet)

_dataSet = _dataSet.replace(10, 12)                     # Replacing values
print(_dataSet)

_dataSet = _dataSet.replace({12 : 10, 9 : 30})
print(_dataSet)

_dataSet.head(3)                                        # First 3 rows
_dataSet.tail(2)                                        # Last 2 rows


print(_dataSet.sort_values('Grade2', ascending=True))   # Sorting columns

print(_dataSet.sort_index(axis=0, ascending=False))     # Sorting rows


_csv = pd.read_csv('../data/wine.csv')                  # Loading csv files
print(_csv.head)
