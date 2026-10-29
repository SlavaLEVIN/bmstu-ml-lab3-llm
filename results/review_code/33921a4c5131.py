from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best_k, best_score = 1, 0
for k in [1, 3, 5, 7, 9]:
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    score = cross_val_score(pipe, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_k, best_score = k, score

final_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=best_k))])
final_pipe.fit(X_train, y_train)
test_acc = final_pipe.score(X_test, y_test)
print(f"Best k: {best_k}")
print(f"Test accuracy: {test_acc:.4f}")