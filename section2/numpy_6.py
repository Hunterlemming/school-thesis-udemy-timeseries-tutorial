import numpy as np


_data = np.array([1, 3, 4, 5, 7, 9])

Mean = lambda data : np.mean(data)                                  # Calculating mean of data
Median = lambda data : np.median(data)                              # Calculating median of data
Variance = lambda data : np.var(data)                               # Calculating variance of data                              
StandardDeviation = lambda data : np.sqrt(Variance(data))           # Calculating standard deviation of data
StandardDeviationNp = lambda data : np.std(data)

print(StandardDeviation(_data))


NumP_Array = np.array([[1, 2, 3], [4, 6, 7]])

Var_Nump = np.var(NumP_Array, axis=1)               # Calculating variance in matrix rows (axis = 1)
print(Var_Nump)

Var_Nump2 = np.var(NumP_Array, axis=0)               # Calculating variance in matrix columns (axis = 0)
print(Var_Nump2)
