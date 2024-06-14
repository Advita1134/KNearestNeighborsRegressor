"""
This code uses KNearestNeighborsRegression.

Link: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor

KNearestNeighborsRegressor does something similar to KNearestNeighborsClassifier.
It uses data that is already given to find answers for other data. (Supervised Learning)

Using the training data, it finds the closest x-values to a number to test on (X_test).
Then, it finds the y-value of those neighbors.
It averages the y-values so you can find out the y-value for your test data.
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Data
X_train = [[3], [5], [8], [9], [6], [2]] # It has to have rows.
y_train = [67, 79, 96, 100, 83, 60]
X_test = [[4], [7], [1]]

# Model
model = KNeighborsRegressor() # The default is 5 neighbors.

# Training
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

plt.scatter(X_train,y_train, c = "blue", label = "Training Data")
plt.scatter(X_test,y_pred, c = "red", marker = "x", s = 70, label = "Testing Data")
plt.legend()
plt.title("KNearestNeighborsRegression on Weight")
plt.xlabel("Grade")
plt.ylabel("Weight")
