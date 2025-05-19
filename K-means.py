'''

The K-means clustering algorithm assumes that the clusters are defined by the distance of the points
to their class centers only. The goal is to find those k mean vectors: c1, c2, ..., ck and provide
the cluster assignment yi ∈ {1, 2, ..., k} of each point xi in the set.

'''

from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import KMeans

class K_means:
    def __init__(self, k=2, max_iter=100, tolerance=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(self.k)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)



if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    kmeans = K_means(k=3, max_iter=100, tolerance=1e-4, random_state=42)
    k_means = KMeans(n_clusters=3, max_iter=100, random_state=42)
    print(kmeans.fit(X).predict(X)) # Self-defined
    print(y)
    k_means.fit(X)
    print(k_means.predict(X)) # The one in sklearn



