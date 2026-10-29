from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score

X, y = make_classification(n_samples=5000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f1_score(y_test, pred, pos_label=1), balanced_accuracy_score(y_test, pred))