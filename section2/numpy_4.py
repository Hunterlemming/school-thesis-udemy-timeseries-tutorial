import numpy as np

NumP_Array = np.array([[1, 2, 3], [4, 6, 7]])

_NP1 = np.array([[1, 3], [4, 5]], dtype=np.int32)
_NP2 = np.array([[3, 4], [5, 7]], dtype=np.int32)

D = np.divide([12, 14, 16], 5)          # Dividing matrices
print(D)

D2 = np.floor_divide([12, 14, 16], 5)
print(D2)

Sq = np.sqrt(10)
print(Sq)
