import numpy as np
from sklearn.decomposition import PCA

# Example matrix: 6 samples x 4 features
X = np.array([
    [1.0, 2.0, 1.0, 3.0],
    [2.0, 1.0, 3.0, 2.0],
    [3.0, 4.0, 2.0, 1.0],
    [4.0, 3.0, 4.0, 2.0],
    [5.0, 6.0, 3.0, 4.0],
    [6.0, 5.0, 5.0, 3.0],
])

print("PCA - Principal Component Analysis")
print("Original matrix shape:", X.shape)
print("Original matrix:")
print(X)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\nPCA Results:")
print("Reduced shape:", X_pca.shape)
print("Transformed data:")
print(X_pca)
print("Variance explained:", pca.explained_variance_ratio_)
print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.1%}")

# Reconstruction
X_reconstructed = pca.inverse_transform(X_pca)
error = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction error: {error:.6f}")
