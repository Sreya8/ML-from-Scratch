"""
Linear Regression Using Normal Equation
Write a Python function that performs linear regression using the normal equation. 
The function should take a matrix X (features) and a vector y (target) as input, and 
return the coefficients of the linear regression model. 
Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

Example:
Input:
X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
Output:
[0.0, 1.0]
Reasoning:
The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.
"""

import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	# ones = np.ones((len(X), 1))
	# X = np.concatenate((ones, X), 1)
	X = np.array(X)
	XT = X.transpose()
	inversed = np.linalg.inv(XT.dot(X))
	theta = inversed.dot(XT).dot(y)
	theta = np.round(theta, 1)
	return theta

print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))