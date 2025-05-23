import numpy as np
from scipy.spatial.distance import cdist
import MDS # From MDS.py

# Use Floyd-Warshall algorithm to calculate the shortest distances of all node pairs
def FW(D):
    N = D.shape[0]
    d = D.copy()
    for k in range(N):
        for i in range(N):
            for j in range(N):
                d[i, j] = min(d[i, j], d[i, k] + d[j, k])

    return d

def Isomap(X, n_neighbors=5, n_components=2):
    n = X.shape[0]
    # 1. Calculate the adjacency graph
    dist_matrix = cdist(X, X)
    KNN_graph = np.full((n, n), np.inf)
    for i in range(n):
        idx = np.argsort(dist_matrix[i])[:n_neighbors + 1]
        for j in idx:
            KNN_graph[i, j] = dist_matrix[i, j]

    # 2. Shortest path graph
    SP = FW(KNN_graph)

    # 3. MDS usage
    return MDS.MDS(SP, n_components=n_components)
