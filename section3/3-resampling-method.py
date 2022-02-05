import numpy as np
import pandas as pd


_df : pd.DataFrame = pd.read_csv('../data/s3_stock_dataset.csv', index_col='DATE', parse_dates=True)
# https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
RS_df = _df.resample(rule='D').mean()

print(RS_df)
