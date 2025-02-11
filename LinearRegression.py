import numpy as np
from GradientDescent import GradientDescent

class LinearRegression:
    def __init__(self, lr = 0.01, epochs = 10, method='batch', batch_size = 32):
        self.lr = lr
        self.epochs = epochs
        self.gd = GradientDescent(method=method, batch_size=batch_size, lr=lr)

    def fit(self, X, y):
        # Number of observations and features
        self.no_obs, self.no_features = X.shape

        # Initialize the weights and biases
        self.w = np.zeros(self.no_features)
        self.b = 0

        # Reading the data
        self.X = X
        self.y = y

        # Use gradient descent to update weights
        for _ in range(self.epochs):
            self.w, self.b = self.gd.update_weights(self.X, self.y, self.w, self.b)
        return self
    
    def predict(self, X):
        return X.dot(self.w) + self.b