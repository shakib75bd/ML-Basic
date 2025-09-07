import os
import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd

# Load cat images
cats = []
for f in os.listdir('Training Data/Cat')[:15]:  # 15 images for training
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))  # 16x16 = 256 features
            cats.append(img.flatten() / 255.0)  # Normalize to 0-1

# Load dog images
dogs = []
for f in os.listdir('Training Data/Dog')[:15]:  # 15 images for training
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Dog/{f}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))  # 16x16 = 256 features
            dogs.append(img.flatten() / 255.0)  # Normalize to 0-1

# Create training data
X_train = np.array(cats + dogs)
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=cat, 1=dog
labels = ['Cat']*len(cats) + ['Dog']*len(dogs)

print(f"Original data shape: {X_train.shape}")
print(f"Cats: {len(cats)}, Dogs: {len(dogs)}")

# === SINGULAR VALUE DECOMPOSITION ===

print("\nApplying SVD...")

# Method 1: Using NumPy's SVD (Full decomposition)
print("\n=== Full SVD using NumPy ===")
U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

print(f"U matrix shape (left singular vectors): {U.shape}")
print(f"S vector shape (singular values): {S.shape}")
print(f"Vt matrix shape (right singular vectors): {Vt.shape}")

# Calculate explained variance from singular values
total_variance = np.sum(S**2)
explained_variance_ratio = (S**2) / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"First 5 components explain: {cumulative_variance[4]:.3f} of variance")
print(f"First 10 components explain: {cumulative_variance[9]:.3f} of variance")
print(f"First 20 components explain: {cumulative_variance[19]:.3f} of variance")

# Method 2: Using Scikit-Learn's TruncatedSVD
print("\n=== Truncated SVD using Scikit-Learn ===")
svd_2d = TruncatedSVD(n_components=2, random_state=42)
X_svd_2d = svd_2d.fit_transform(X_train)

print(f"Reduced to 2D: {X_svd_2d.shape}")
print(f"2D SVD explains {sum(svd_2d.explained_variance_ratio_):.3f} of total variance")

# === SVD ANALYSIS ===

print(f"\nSVD Analysis:")
print(f"Rank of data matrix: {np.linalg.matrix_rank(X_train)}")
print(f"Number of non-zero singular values: {np.sum(S > 1e-10)}")
print(f"Largest singular value: {S[0]:.3f}")
print(f"Smallest singular value: {S[-1]:.6f}")
print(f"Condition number: {S[0]/S[-1]:.2e}")

# Show singular values
print(f"\nFirst 10 singular values: {S[:10]}")
print(f"Singular value ratios (S[i]/S[0]):")
for i in range(min(10, len(S))):
    print(f"  S[{i}]/S[0] = {S[i]/S[0]:.6f}")

# === DIMENSIONALITY REDUCTION WITH SVD ===

print(f"\nSVD Dimensionality Reduction Results:")
print("-" * 50)

max_components = min(X_train.shape[0], X_train.shape[1])
for n_components in [2, 5, 10, 20, 25]:
    if n_components <= max_components:
        # Manual reconstruction using first n components
        U_n = U[:, :n_components]
        S_n = S[:n_components]
        Vt_n = Vt[:n_components, :]

        # Calculate variance explained
        variance_explained = np.sum(S_n**2) / total_variance
        compression_ratio = X_train.shape[1] / n_components

        print(f"{n_components:2d} components: {n_components:3d} features, "
              f"{variance_explained:.3f} variance explained, "
              f"compression ratio: {compression_ratio:.1f}x")

# === APPLY SVD TO TEST IMAGES ===

print(f"\nSVD Transformation of Test Images:")
print("-" * 50)

for filename in sorted(os.listdir('TestData')):
    if filename.endswith('.jpg'):
        # Load test image
        img = cv2.imread(f'TestData/{filename}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))
            x_test = img.flatten() / 255.0

            # Project test image onto first 2 SVD components
            # x_test_svd = x_test @ Vt[:2, :].T
            x_test_svd = svd_2d.transform([x_test])

            print(f"{filename}: SVD1={x_test_svd[0][0]:6.3f}, SVD2={x_test_svd[0][1]:6.3f}")

# === RECONSTRUCTION ANALYSIS ===

print(f"\nSVD Reconstruction Analysis:")
print("-" * 50)

# Test reconstruction quality with different numbers of components
original_data = X_train.copy()

for n_components in [2, 5, 10, 20]:
    if n_components <= max_components:
        # Reconstruct using first n components
        U_n = U[:, :n_components]
        S_n = S[:n_components]
        Vt_n = Vt[:n_components, :]

        X_reconstructed = U_n @ np.diag(S_n) @ Vt_n

        # Calculate reconstruction error
        reconstruction_error = np.mean((original_data - X_reconstructed)**2)
        variance_retained = np.sum(S_n**2) / total_variance

        print(f"{n_components:2d} components: "
              f"reconstruction error = {reconstruction_error:.6f}, "
              f"variance retained = {variance_retained:.3f}")

# === LOW-RANK APPROXIMATION ===

print(f"\nLow-Rank Matrix Approximation:")
print("-" * 50)

# Show how well different ranks approximate the original matrix
print("Matrix approximation quality:")
for rank in [1, 2, 5, 10]:
    if rank <= max_components:
        # Create rank-k approximation
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vt_k = Vt[:rank, :]

        X_approx = U_k @ np.diag(S_k) @ Vt_k

        # Calculate Frobenius norm of error
        frobenius_error = np.linalg.norm(X_train - X_approx, 'fro')
        relative_error = frobenius_error / np.linalg.norm(X_train, 'fro')

        print(f"  Rank-{rank:2d}: Frobenius error = {frobenius_error:.4f}, "
              f"Relative error = {relative_error:.4f}")

# === 2D SVD VISUALIZATION DATA ===

print(f"\n2D SVD Visualization Data:")
print("-" * 50)
print("Cat images in 2D SVD space:")
for i, (x, y) in enumerate(X_svd_2d[:len(cats)]):
    print(f"  Cat {i+1:2d}: ({x:6.3f}, {y:6.3f})")

print("\nDog images in 2D SVD space:")
for i, (x, y) in enumerate(X_svd_2d[len(cats):]):
    print(f"  Dog {i+1:2d}: ({x:6.3f}, {y:6.3f})")

# === SVD COMPONENT ANALYSIS ===

print(f"\nSVD Component Analysis:")
print("-" * 50)

# Analyze the first few right singular vectors (feature patterns)
print("Right singular vectors (feature patterns):")
for i in range(min(3, Vt.shape[0])):
    max_val = np.max(np.abs(Vt[i, :]))
    max_idx = np.argmax(np.abs(Vt[i, :]))
    print(f"  Component {i+1}: max value = {max_val:.4f} at feature {max_idx}")
    print(f"    First 10 values: {Vt[i, :10]}")

# Analyze left singular vectors (sample patterns)
print("\nLeft singular vectors (sample patterns):")
for i in range(min(3, U.shape[1])):
    # Find which samples contribute most to this component
    max_val = np.max(np.abs(U[:, i]))
    max_idx = np.argmax(np.abs(U[:, i]))
    sample_type = "Cat" if max_idx < len(cats) else "Dog"
    print(f"  Component {i+1}: max contribution from {sample_type} sample {max_idx}")

# === SVD SUMMARY ===

print(f"\nSVD Summary:")
print("- SVD decomposes matrix into U, S, Vt components")
print("- U: Left singular vectors (row/sample patterns)")
print("- S: Singular values (importance weights)")
print("- Vt: Right singular vectors (column/feature patterns)")
print("- Provides optimal low-rank matrix approximation")
print(f"- Original: {X_train.shape[1]} features → 2D: {sum(svd_2d.explained_variance_ratio_):.1%} variance retained")

# === MATHEMATICAL PROPERTIES ===

print(f"\nSVD Mathematical Properties:")
print("-" * 50)

# Verify SVD decomposition
X_reconstructed_full = U @ np.diag(S) @ Vt
reconstruction_error_full = np.max(np.abs(X_train - X_reconstructed_full))
print(f"Full reconstruction error (should be ~0): {reconstruction_error_full:.2e}")

# Check orthogonality
U_orthogonal = np.max(np.abs(U.T @ U - np.eye(U.shape[1])))
Vt_orthogonal = np.max(np.abs(Vt @ Vt.T - np.eye(Vt.shape[0])))
print(f"U orthogonality error: {U_orthogonal:.2e}")
print(f"Vt orthogonality error: {Vt_orthogonal:.2e}")

# Nuclear norm (sum of singular values)
nuclear_norm = np.sum(S)
print(f"Nuclear norm (sum of singular values): {nuclear_norm:.3f}")
