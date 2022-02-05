import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Year = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
Temp_I = [0.77, 0.61, 0.65, 0.68, 0.75, 0.90, 1.02, 0.93, 0.85, 0.99, 1.02]

plt.plot(Year, Temp_I)
plt.xlabel("Year")
plt.ylabel("Temp_Index")
plt.title("Global Warming", {'fontsize': 12, 'horizontalalignment': 'center'})

plt.show()
