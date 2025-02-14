# Take breast cancer dataset as example, this file interprets how NMF works
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd

breast_cancer = load_breast_cancer()

V = breast_cancer.data # V: m*n
dataset = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
rows = ['sample {}'.format(i) for i in range(1, len(dataset) + 1)]
dataset.index = rows

# print(dataset)
np.random.seed(60)
r = 10
W = np.random.rand(V.shape[0], r) # W: m*r
H = np.random.rand(r, V.shape[1]) # H: r*n

max_iter = 20000
tol = 1e-3
prev_error = None

for iteration in range(1, max_iter+1):
    # Updating H, formula: H = h * W.T @ V / (W.T @ W @ H)
    numerator = W.T @ V
    denominator = W.T @ W @ H + 1e-10
    H *= numerator / denominator

    # Updating W, formula: W = W * V @ H.T / (W @ H @ H.T)
    numerator = V @ H.T
    denominator = W @ H @ H.T + 1e-10
    W *= numerator / denominator

    error = np.linalg.norm(V - W @ H, 'fro')
    # Error calculation
    if not prev_error or prev_error - error >= tol:
        print("Iteration {} complete, with error {}".format(iteration, error))
        prev_error = error

    else:
        print("Iteration {} complete, with error {}".format(iteration, error))
        print('Converged at iteration {}, final error is {}'.format(iteration, error))
        break


print(W)
print(H)
print(W @ H)
