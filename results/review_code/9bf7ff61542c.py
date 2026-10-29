```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
digits = load_digits()
X, y = digits.data, digits.target

# Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Модель с PCA
pipeline_with_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('logistic', LogisticRegression(max_iter=2000))
])

# Модель без PCA
pipeline_without_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=2000))
])

# Обучение и предсказание
y_pred_with_pca = pipeline_with_pca.fit(X_train, y_train).predict(X_test)
y_pred_without_pca = pipeline_without_pca.fit(X_train, y_train).predict(X_test)

# Сравнение accuracy
accuracy_with_pca = accuracy_score(y_test, y_pred_with_pca)
accuracy_without_pca = accuracy_score(y_test, y_pred_without_pca)

accuracy_with_pca, accuracy_without_pca
```