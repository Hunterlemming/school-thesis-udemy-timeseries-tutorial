import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Customer1 = [12, 13, 9, 8, 7, 8, 8, 7, 6, 5, 8, 10]
Customer2 = [14, 16, 11, 7, 6, 6, 7, 6, 5, 8, 9, 12]


'''
plt.scatter(Month, Customer1, color='red', label="Customer1")
plt.scatter(Month, Customer2, color='blue', label="Customer2")
plt.xlabel("Months")
plt.ylabel("Electricity Consumption")
plt.title("Scatter Plot of Building Conumption")
plt.grid()
plt.legend(loc='best')

plt.show()
'''


plt.hist(Customer1, 10, color='green')
plt.ylabel("Eletricity Consumption")
plt.title("Histogram")
plt.show()


'''
plt.bar(Month, Customer1, width=0.8, color='b')
plt.show()
'''
