import numpy as np
from scipy.spatial.distance import cdist

def LLE(X, n_neighbors=5, n_components=2, reg=1e-3):
    N, D = X.shape
    W = np.zeros((N, N))

    # 1. Calculating the nearest neighbor
    d = cdist(X, X)
    for i in range(N):
        idx = np.argsort(d[i])[1:n_neighbors + 1]
        Z = X[idx] - X[i]
        C = Z @ Z.T
        C += reg * np.eye(n_neighbors)
        w = np.linalg.solve(C, np.ones(n_neighbors))
        w /= np.sum(w)
        W[i, idx] = w

    # 2. Calculate the smallest eigenvector of (I - W) ^ T @ (I - W)
    M = (np.eye(N) - W).T @ (np.eye(N) - W)
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs[:, 1:n_components+1]