from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=3, random_state=42)
best_k, best_score = None, -1
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}, silhouette_score={score:.4f}")
    if score > best_score:
        best_k, best_score = k, score
print(f"Выбрано k={best_k}")