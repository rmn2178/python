# Importing necessary libraries
# This was based on the linear regression example from sklearn documentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Loading the dataset
df = pd.read_csv("india_per_capita_income.csv")

#plotting for the visualisation of the data
plt.xlabel('Area in square feet')
plt.ylabel('Price in US Dollars')
plt.scatter(df.year, df.pci, color='red', marker='*')
plt.show()

# Preparing the data for training
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df.pci)

# Making predictions
#num = int(input("Enter the area in square feet:"))
#print(reg.predict(pd.DataFrame([[num]], columns=['area'])))

# linear regression model is based on the idea of fitting a line to the data points in such a way that the sum of the squared differences between the observed values and the values predicted by the line is minimized. This method is known as least squares.
# y = mx + b  m=slope, b=intercept

# To get the slope (m) and intercept (b) of the fitted line, we can use the attributes of the trained model:
print("Slope (m):", reg.coef_)

# To get the value of the intercept (b):
print("Intercept (b):", reg.intercept_)

# Now testing the model with some test data
td = pd.read_csv("prediction.csv")
td.columns = td.columns.str.strip().str.lower()  # Normalize column names
p = reg.predict(td[['year']])
td['pci'] = p
print(td)

# To exporting the data to a csv file the file will have the area and the predicted price
td.to_csv("prediction.csv",index = False)

# IMPORTANT NOTE
# This works on linear regression so the sudden change in per-capita income leads to error
# This is only possible for gradual increase in the values