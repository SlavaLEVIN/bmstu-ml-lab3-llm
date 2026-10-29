from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]), {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="neg_root_mean_squared_error")
model.fit(X_train, y_train)
print(root_mean_squared_error(y_test, model.predict(X_test)))