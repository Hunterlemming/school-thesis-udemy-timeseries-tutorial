import numpy as np
import pandas as pd


'''
np.random.seed(1999)


Date3 = pd.date_range('jan 01 2021', periods=8760, freq='H')
Data_Set = np.random.randint(1, 1000, (8760, 2))


# Creating a time-series dataset
Data_Set_DF = pd.DataFrame(Data_Set)
Data_Set_DF.set_index(Date3, inplace=True)
print(Data_Set_DF)
'''


_df : pd.DataFrame = pd.read_csv('../data/s3_stock_dataset.csv', 
index_col='DATE',       # selecting a column as index instead of creating a new one
parse_dates=True)       # converting date fields to DateTime
print(_df.info())
