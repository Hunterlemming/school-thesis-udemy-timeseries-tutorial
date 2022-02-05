from typing import Callable
import pandas as pd


# Series

Age = pd.Series([10, 20, 30, 40], index=['age1', 'age2', 'age3', 'age4'], dtype=pd.Int32Dtype())

_age3 = Age.age3
print(_age3)

_filteredAge = Age[20 < Age]
print(_filteredAge)

_ageValues = Age.values
_ageIndices = Age.index

print(Age)
Age.index = ['A1', 'A2', 'A3', 'A4']        # Replacing indices
print(Age)