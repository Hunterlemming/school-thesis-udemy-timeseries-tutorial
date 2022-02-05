import numpy as np
import pandas as pd


_data = np.array([[20, 10, 8, 9], [25, 8, 10, 6], [27, 5, 3, 7], [30, 9, 7, 10]], dtype=np.int32)
_dataSet = pd.DataFrame(_data, index=['S1', 'S2', 'S3', 'S4'], columns=['Age', 'Grade1', 'Grade2', 'Grade3'])


Selection1 = _dataSet.loc['S2']             # Accessing a group of rows and colums by label(s)
print(Selection1)

Selection2 = _dataSet.iloc[1, 3]            # Accessing data with the row and column indices
print(Selection2)

Selection3 = _dataSet.iloc[:, 0]            # All rows, First column
print(Selection3)

Selection4 = _dataSet.iloc[:, 1:3]          # All rows (0 <= x < 4 (= length)), Columns 2 and 3 (1 <= x < 3)
print(Selection4)