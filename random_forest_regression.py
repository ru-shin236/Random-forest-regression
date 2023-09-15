# Random forest regression

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Spliting the dataset into the Training and test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)"""

# Fitting Random forest regression to dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict(np.array(6.5).reshape(-1, 1))

# Visualising the Random Forest Regression results (higher resolution)
x_gird = np.arange(min(x), max(x), 0.01)
plt.scatter(x, y, color="red")
plt.plot(x_gird, regressor.predict(x_gird.reshape(-1, 1)), color="blue")
plt.title("Regression plot of position vs salary.")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
