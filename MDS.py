import numpy as np

def MDS(D, n_components=2):
    '''

    :param D: Distance matrix, usually Euclidean distances
    :param n_components: Number of dimensions after dimension reduction

    '''
    N = D.shape[0]

    # 1. Transform the distance matrix into inner product matrix
    D_sq = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D_sq @ J

    # 2. Decomposition of features
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # 3. Select the first K dimensions
    L = np.diag(np.sort(eigvals[:n_components]))
    V = eigvecs[:, :n_components]
    return V @ L

