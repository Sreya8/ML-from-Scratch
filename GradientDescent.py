import numpy as np

class GradientDescent:
    def __init__(self, method="batch", batch_size = 32, lr = 0.01):
        self.method = method
        self.batch_size = batch_size
        self.lr = lr

    def update_weights(self, X, y, w, b):
        if self.method == "batch":
            return self.batchGradientDescent(X, y, w, b)
        elif self.method == "mini-batch":
            return self.miniBatchGradientDescent(X, y, w, b)
        elif self.method == "stochastic":
            return self.stochasticGradientDescent(X, y, w, b)
        else:
            raise ValueError("Invalid Gradient Descent Method")
    
    def batchGradientDescent(self, X, y, w, b):
        no_obs = X.shape[0]

        # Get predictions
        y_pred = X.dot(w) + b

        # Calculate gradients for the complete dataset
        grad_w = -2 * np.dot(X.T, (y - y_pred)) / no_obs
        grad_b = -2 * np.mean(y - y_pred)

        # Update weights
        w -= self.lr * grad_w
        b -= self.lr * grad_b

        return w, b
    
    def stochasticGradientDescent(self, X, y, w, b):
        no_obs = X.shape[0]
        for i in range(no_obs):
            xi = X[i]
            yi = y[i]

            # Get the prediction for a single observation
            y_pred = np.dot(xi, w) + b

            # Calculate gradient for a single observation
            grad_w = -2 * (xi * (yi - y_pred))
            grad_b = -2 * (yi - y_pred)

            # Update weights during every iteration
            w -= self.lr * grad_w
            b -= self.lr * grad_b

        return w, b
    
    def miniBatchGradientDescent(self, X, y, w, b):
        no_obs = X.shape[0]
        indices = np.random.permutation(no_obs)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, no_obs, self.batch_size):
            X_batch = X_shuffled[i : i + self.batch_size]
            y_batch = y_shuffled[i : i + self.batch_size]

            y_pred = X_batch.dot(w) + b

            grad_w = -2 * (np.dot(X_batch.T, (y_batch - y_pred))) / len(y_pred)
            grad_b = -2 * np.mean(y_batch - y_pred)

            # grad_w = np.clip(grad_w, -1, 1)
            # grad_b = np.clip(grad_b, -1, 1)

            w -= self.lr - grad_w
            b -= self.lr - grad_b

        return w, b