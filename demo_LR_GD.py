from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler


X, y = datasets.load_diabetes(return_X_y=True)

if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    raise ValueError("Data contains NaN values")


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression(lr=0.01, epochs = 100, method='batch')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation Metrics
# Root mean squared error
mse = np.mean((y_test - predictions) ** 2)
print("Mean Squared Error:", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# mse_sk = mean_squared_error(y_test, predictions)
# print("Mean Squared error using sklearn", mse_sk)

