import numpy as np

np.random.seed(1999)

_ND = np.random.standard_normal((3, 4))     # Generating random numbers in normal distribution
print(_ND)

_UD = np.random.uniform(1, 12, (3, 4))      # Generating random numbers (between 1 & 12) in uniform distribution
print(_UD)

_RA1 = np.random.rand()                     # Generating random float
_RA2 = np.random.randint(1, 50, (2, 5))     # Generating random int
print(_RA2)

_Z = np.zeros((3, 4))                       # Generating zeros
print(_Z)

_O = np.ones((3, 4))                        # Generating ones
print(_O)


# Filtering

_filter_for_RA2 = np.logical_and(30 < _RA2, _RA2 < 50)
print(_filter_for_RA2)

_filtered_RA2 = _RA2[_filter_for_RA2]
print(_filtered_RA2)
