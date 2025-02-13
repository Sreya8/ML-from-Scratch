import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import copy

class LinearRegressionOLS:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y

        # create a deep copy of X
        X = copy.deepcopy(X)
        dummy = np.ones((X.shape[0], 1))
        # print(X.shape)
        # print(dummy.shape)

        X = np.concatenate((dummy, X), 1)
        # print(X.shape)
        xT = X.transpose()

        inversed = np.linalg.inv(xT.dot(X))
        betas = inversed.dot(xT).dot(y)
        self.b = betas[0]
        self.w = betas[1:]
        return self

    def predict(self, X):
        return X.dot(self.w) + self.b

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegressionOLS()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = np.mean((y_test - y_pred) ** 2)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)