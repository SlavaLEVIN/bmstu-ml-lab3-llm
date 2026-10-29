from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe_with_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', LogisticRegression(max_iter=2000))
])

pipe_without_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

pipe_with_pca.fit(X_train, y_train)
pipe_without_pca.fit(X_train, y_train)

acc_with_pca = accuracy_score(y_test, pipe_with_pca.predict(X_test))
acc_without_pca = accuracy_score(y_test, pipe_without_pca.predict(X_test))

print(f"Accuracy with PCA: {acc_with_pca:.4f}")
print(f"Accuracy without PCA: {acc_without_pca:.4f}")