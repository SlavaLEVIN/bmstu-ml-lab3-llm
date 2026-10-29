from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf'))])
param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 0.001, 0.01, 0.1]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print(accuracy_score(y_test, y_pred))