import numpy as np

NumP_Array = np.array([[1, 2, 3], [4, 6, 7]])

_NP1 = np.array([[1, 3], [4, 5]], dtype=np.int32)
_NP2 = np.array([[3, 4], [5, 7]], dtype=np.int32)

MNP2 = _NP1 * _NP2          # Element-wise multiplication of matrices
print(MNP2)

NMP4 = np.multiply(_NP1, _NP2)
print(NMP4)
print(MNP2 == NMP4)

MNP = _NP1 @ _NP2           # Multiply matrices
print(MNP)

MNP3 = np.dot(_NP1, _NP2)
print(MNP3)
print(MNP == MNP3)
