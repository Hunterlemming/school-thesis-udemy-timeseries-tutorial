import numpy as np

NumP_Array = np.array([[1, 2, 3], [4, 6, 7]])

_NP1 = np.array([[1, 3], [4, 5]], dtype=np.int32)
_NP2 = np.array([[3, 4], [5, 7]], dtype=np.int32)

Sum1 = _NP1 + _NP2              # Adding matrices
print(Sum1)

Sub1 = _NP1 - _NP2              # Subtracting matrices
print(Sub1)

Sub2 = np.subtract(_NP1, _NP2)
print(Sub2)

El_Sum = np.sum(_NP1)           # Adding all the elements in a container (eg.: matrix)
print(El_Sum)

Broad_Nump = _NP1 + 3           # Automaticly transforming a constant to match matrix size
print(Broad_Nump)

NP3 = np.array([[3, 4]])
Broad_Nump2 = _NP1 + NP3
print(Broad_Nump2)
