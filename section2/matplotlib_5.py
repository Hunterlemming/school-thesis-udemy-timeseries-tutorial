import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Customer1 = [12, 13, 9, 8, 7, 8, 8, 7, 6, 5, 8, 10]
Customer2 = [14, 16, 11, 7, 6, 6, 7, 6, 5, 8, 9, 12]


plt.boxplot([Customer1, Customer2], patch_artist=True, 
boxprops={'facecolor': 'red', 'color': 'red'},
whiskerprops={'color': 'green'},
capprops={'color': 'blue'},
medianprops={'color': 'yellow'})
plt.show()
