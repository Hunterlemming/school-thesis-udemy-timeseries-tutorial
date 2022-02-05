from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Customer1 = [12, 13, 9, 8, 7, 8, 8, 7, 6, 5, 8, 10]
Customer2 = [14, 16, 11, 7, 6, 6, 7, 6, 5, 8, 9, 12]

'''
plt.plot(Month, Customer1, color='red', label="Customer1", marker='o')
plt.plot(Month, Customer2, color='blue', label="Customer2", marker='^')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption")
plt.legend(loc='upper right')

plt.show()
'''

plt.subplot(1, 2, 1)                                                        # Rows, Colums, Part
plt.plot(Month, Customer1, color='red', label="Customer1", marker='o')
plt.xlabel("Month")
plt.ylabel("Electricity Consumption")
plt.title("Building Consumption of Customer1")

plt.subplot(1, 2, 2)
plt.plot(Month, Customer2, color='blue', label="Customer2", marker='^')
plt.xlabel("Month")
plt.title("Building Consumption of Customer2")
plt.show()
