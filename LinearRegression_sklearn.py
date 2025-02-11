from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SGDRegressor(learning_rate='optimal', max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
