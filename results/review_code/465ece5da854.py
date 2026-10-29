from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=3, random_state=42)
scores = {}
for k in (2, 3, 4, 5):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    print(f"k={k}: silhouette={scores[k]:.4f}")
print(f"best_k={max(scores, key=scores.get)}")