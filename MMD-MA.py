import numpy as np
from scipy.spatial.distance import cdist

def RBF(X, sigma=None):
    pairwise_sq_distances = cdist(X, X, 'sqeuclidean')
    if sigma is None:
        sigma = np.median(pairwise_sq_distances)
    K = np.exp(-pairwise_sq_distances / (2 * sigma ** 2))
    return K

def mmd_ma(X1, X2, n_components=2, L=1e-3):
    N1, D1 = X1.shape
    N2, D2 = X2.shape
    N = N1 + N2
    X = np.vstack([X1, X2])

    # RBF Kernel Matrix
    K = RBF(X)

    # MMD matrix
    E = np.zeros((N, 1))
    E[:N1] = 1.0 / N1
    E[N1:] = -1.0 / N2
    M = E @ E.T

    # Centralized Matrix
    H = np.eye(N) - np.ones((N, N)) / N

    # (K@M@K + lambda*I)@v = mu*K@H@K*v
    A = K @ M @ K + L * np.eye(N)
    B = K @ H @ K
    eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(B) @ A)
    idx = np.argsort(eigvals)
    return eigvecs[:, idx[:n_components]]