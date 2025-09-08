import numpy as np

# Example matrix: 4 samples x 3 features
A = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 4.0, 6.0],
])

print("SVD - Singular Value Decomposition (Manual Calculation)")
print("Original matrix shape:", A.shape)
print("Original matrix:")
print(A)

# Manual SVD calculation using eigendecomposition
# Step 1: Compute A^T A for V and eigenvalues
AtA = A.T @ A

# Step 2: Find eigenvalues and eigenvectors of A^T A
eigenvals_AtA, V = np.linalg.eigh(AtA)

# Sort eigenvalues in descending order
idx = np.argsort(eigenvals_AtA)[::-1]
eigenvals_AtA = eigenvals_AtA[idx]
V = V[:, idx]

# Step 3: Calculate singular values
sigma = np.sqrt(np.maximum(eigenvals_AtA, 0))  # Avoid negative due to numerical errors

# Step 4: Compute U matrix
# For non-zero singular values, U[:, i] = (1/sigma[i]) * A @ V[:, i]
U = np.zeros((A.shape[0], len(sigma)))
for i in range(len(sigma)):
    if sigma[i] > 1e-10:  # Only for non-zero singular values
        U[:, i] = (A @ V[:, i]) / sigma[i]

# Create V^T
Vt = V.T

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
