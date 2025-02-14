import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Taking iris data as example

C = np.eye(150) - 1/150 * np.ones((150, 150))

iris = load_iris()['data']
target = load_iris()['target']
target_names = load_iris()['target_names']
colors = ['red', 'blue', 'green']
X = iris.T
covariance_matrix = X @ C @ X.T

eigvals, eigvecs = np.linalg.eig(covariance_matrix)
sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

variances = []

for i in range(len(eigvecs)):
    var = eigvecs[i].T @ covariance_matrix @ eigvecs[i]
    variances.append(var)

variances = np.array(variances)
print(eigvecs)
print(variances)

indices = np.argsort(variances)[::-1]

sorted_variances = variances[indices]
sorted_eigvecs = eigvecs[indices]

print(sorted_variances)
print(sorted_eigvecs)

PCs = sorted_eigvecs[0:2]

X_pca = X.T @ PCs.T

for i, color, label in zip(np.unique(target), colors, target_names):
    plt.scatter(X_pca[target == i, 0], X_pca[target == i, 1], c=color, label=label)

plt.legend()
plt.title("PCA Scatter Plot of Iris Dataset")
plt.xlabel("Principal Component I")
plt.ylabel("Principal Component II")
plt.grid(alpha=0.3)
plt.show()

