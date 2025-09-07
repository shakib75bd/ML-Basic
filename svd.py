import numpy as np
from scipy.linalg import svd

# Example matrix: 4 samples x 3 features
A = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 4.0, 6.0],
])

print("SVD - Singular Value Decomposition")
print("Original matrix shape:", A.shape)
print("Original matrix:")
print(A)

# Perform SVD: A = U × Σ × V^T
U, sigma, Vt = svd(A, full_matrices=False)

print("\nSVD Results:")
print("U shape:", U.shape)
print("Sigma (singular values):", sigma)
print("V^T shape:", Vt.shape)

# Reconstruction verification
A_reconstructed = U @ np.diag(sigma) @ Vt
print("\nReconstructed matrix:")
print(A_reconstructed)
print(f"Reconstruction error: {np.mean((A - A_reconstructed)**2):.10f}")

# Low-rank approximation (rank 2)
k = 2
A_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
print(f"\nRank-{k} approximation:")
print(A_k)
print(f"Approximation error: {np.mean((A - A_k)**2):.6f}")

# Variance analysis
variance_explained = sigma**2 / np.sum(sigma**2)
print(f"Variance explained by rank-{k}: {np.sum(variance_explained[:k]):.1%}")
