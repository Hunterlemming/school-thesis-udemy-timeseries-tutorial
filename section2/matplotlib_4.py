import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Customer1 = [12, 13, 9, 8, 7, 8, 8, 7, 6, 5, 8, 10]
Customer2 = [14, 16, 11, 7, 6, 6, 7, 6, 5, 8, 9, 12]


'''
plt.bar(Month, Customer1, width=0.8, color='r', label="Customer1")
plt.bar(Month, Customer2, color='b', label="Customer2")
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Bar Chart")
plt.legend()
plt.show()
'''

bar_width = 0.4
Month_b = np.arange(12)

plt.bar(Month_b - bar_width / 2, Customer1, bar_width, color='blue', label="Customer1")
plt.bar(Month_b + bar_width / 2, Customer2, bar_width, color='red', label="Customer2")

plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.xticks(Month_b, Month)

plt.title("Bar Chart")
plt.legend()

plt.show()
