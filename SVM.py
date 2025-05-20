'''

Support Vector Machine (SVM) is a supervised ML algorithm which finds the
best hyperplane that separates data points into different classes by maximizing
the margin between the closest points of each class.

SVMs are effective in high-dimensional spaces, and SVMs can handle both linear and
non-linear data using kernel method.

'''
import numpy as np
from cvxopt import matrix, solvers

# Hard-margin
class hard_margin_SVM:
    def __init__(self, lr=1e-3, lambda_param=1e-2, iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.iters = iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape

        self.w = np.zeros(features)
        self.b = 0

        y_prime = np.where(y <= 0, -1, 1)

        for _ in range(self.iters):
            for idx, x in enumerate(X):
                cond = y_prime[idx] * (np.dot(x, self.w) + self.b) >= 1
                if cond:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param * self.w - y_prime[idx] * x
                    db = -y_prime[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# Soft-margin
class soft_margin_SVM:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        samples, features = X.shape
        y = y.astype(float)

        # Gram matrix
        K = np.dot(X, X.T)

        # QP
        P = matrix(np.outer(y, y) * K)
        Q = matrix(-np.ones(samples))
        G_std = -np.eye(samples)
        h_std = np.zeros(samples)
        G_slack = np.eye(samples)
        h_slack = self.C * np.ones(samples)
        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.vstack((h_std, h_slack)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        # Solution to Dual Problem
        sol = solvers.qp(P, Q, G, h, A, b)
        alphas = np.ravel(sol['x'])

        SV = alphas > 1e-5
        self.alpha = alphas[SV]
        self.svs = X[SV]
        self.sv_labels = y[SV]

        # Weight vector v
        self.w = np.sum(self.alpha[:, None] * self.sv_labels[:, None] * self.svs, axis=0)
        self.b = np.mean(self.sv_labels - np.dot(self.svs, self.w))

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

# Kernel
class KernelSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, deg=3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.deg = deg

    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)

        elif self.kernel == 'rbf':
            # e^(-gamma * ||x - x_prime||^2)
            squared_distances = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
            return np.exp(-self.gamma * squared_distances)

        elif self.kernel == 'poly':
            return (1 + np.dot(x1, x2.T)) ** self.deg

        else:
            raise AttributeError(f'Unknown kernel type: {self.kernel}')

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y = y.astype(float)
        samples = X.shape[0]

        # Kernel Matrix
        K = self._kernel(X, X)

        # QP
        P = matrix(np.outer(y, y) * K)
        Q = matrix(-np.ones(samples))
        G_std = -np.eye(samples)
        h_std = np.zeros(samples)
        G_slack = np.eye(samples)
        h_slack = self.C * np.ones(samples)
        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.vstack((h_std, h_slack)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        sol = solvers.qp(P, Q, G, h, A, b)
        alphas = np.ravel(sol['x'])

        SV = alphas > 1e-5
        self.alpha = alphas[SV]
        self.svs = X[SV]
        self.sv_labels = y[SV]

        # Bias b (estimated via SV)
        self.b = np.mean([
            y_k - np.sum(self.alpha * self.sv_labels *
                         self._kernel(np.array([x_k]), self.svs).flatten())
            for x_k, y_k in zip(self.svs, self.sv_labels)
        ])

    def project(self, X):
        K = self._kernel(X, self.svs)
        return np.dot(K, self.alpha * self.sv_labels) + self.b

    def predict(self, X):
        return np.sign(self.project(X))
