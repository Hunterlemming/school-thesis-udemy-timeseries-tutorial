import numpy as np
import pandas as pd


Date1 = pd.date_range('jan 01 2021', periods=12, freq='M')
print(Date1)

Date2 = pd.date_range('jan 01 2021', periods=4, freq='3M')
print(Date2)

Date3 = pd.date_range('jan 01 2021', periods=8760, freq='H')
print(Date3)