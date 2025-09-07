import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# === PRINCIPAL COMPONENT ANALYSIS ===

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply PCA for dimensionality reduction
print("\nApplying PCA...")
pca = PCA()  # Keep all components initially
pca.fit(X_scaled)

# Get explained variance ratios
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"Total features: {X_train.shape[1]}")
print(f"First 5 components explain: {cumulative_variance[4]:.3f} of variance")
print(f"First 10 components explain: {cumulative_variance[9]:.3f} of variance")
print(f"First 20 components explain: {cumulative_variance[19]:.3f} of variance")

# Transform to reduced dimensions (using first 2 components for visualization)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"\nReduced to 2D: {X_pca_2d.shape}")
print(f"2D PCA explains {sum(pca_2d.explained_variance_ratio_):.3f} of total variance")

# === ANALYZE PCA COMPONENTS ===

print(f"\nPCA Analysis:")
print(f"Principal component 1 explains: {pca_2d.explained_variance_ratio_[0]:.3f} of variance")
print(f"Principal component 2 explains: {pca_2d.explained_variance_ratio_[1]:.3f} of variance")

# Show component loadings (first 10 features)
print(f"\nPC1 loadings (first 10 features): {pca_2d.components_[0][:10]}")
print(f"PC2 loadings (first 10 features): {pca_2d.components_[1][:10]}")

# === DIMENSIONALITY REDUCTION FOR DIFFERENT NUMBERS OF COMPONENTS ===

print(f"\nDimensionality Reduction Results:")
print("-" * 50)

max_components = min(X_train.shape[0], X_train.shape[1])  # Can't exceed min(samples, features)
for n_components in [2, 5, 10, 20, 25]:
    if n_components <= max_components:
        pca_n = PCA(n_components=n_components)
        X_reduced = pca_n.fit_transform(X_scaled)
        variance_explained = sum(pca_n.explained_variance_ratio_)

        print(f"{n_components:2d} components: {X_reduced.shape[1]:3d} features, "
              f"{variance_explained:.3f} variance explained, "
              f"compression ratio: {X_train.shape[1]/n_components:.1f}x")

# === APPLY PCA TO TEST IMAGES ===

print(f"\nPCA Transformation of Test Images:")
print("-" * 50)

for filename in sorted(os.listdir('TestData')):
    if filename.endswith('.jpg'):
        # Load test image
        img = cv2.imread(f'TestData/{filename}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))
            x_test = img.flatten() / 255.0

            # Apply same scaling and PCA transformation
            x_test_scaled = scaler.transform([x_test])
            x_test_pca = pca_2d.transform(x_test_scaled)

            print(f"{filename}: PC1={x_test_pca[0][0]:6.3f}, PC2={x_test_pca[0][1]:6.3f}")

# === RECONSTRUCTION ANALYSIS ===

print(f"\nReconstruction Analysis:")
print("-" * 50)

# Test reconstruction quality with different numbers of components
test_image_idx = 0  # First cat image
original_image = X_scaled[test_image_idx]

max_components = min(X_train.shape[0], X_train.shape[1])
for n_components in [2, 5, 10, 20]:
    if n_components <= max_components:
        # Reduce dimensionality
        pca_recon = PCA(n_components=n_components)
        X_reduced = pca_recon.fit_transform(X_scaled)

        # Reconstruct
        X_reconstructed = pca_recon.inverse_transform(X_reduced)

        # Calculate reconstruction error
        reconstruction_error = np.mean((X_scaled - X_reconstructed)**2)
        variance_retained = sum(pca_recon.explained_variance_ratio_)

        print(f"{n_components:2d} components: "
              f"reconstruction error = {reconstruction_error:.6f}, "
              f"variance retained = {variance_retained:.3f}")

# === DATA VISUALIZATION COORDINATES ===

print(f"\n2D PCA Visualization Data:")
print("-" * 50)
print("Cat images in 2D space:")
for i, (x, y) in enumerate(X_pca_2d[:len(cats)]):
    print(f"  Cat {i+1:2d}: ({x:6.3f}, {y:6.3f})")

print("\nDog images in 2D space:")
for i, (x, y) in enumerate(X_pca_2d[len(cats):]):
    print(f"  Dog {i+1:2d}: ({x:6.3f}, {y:6.3f})")

# === PCA SUMMARY ===

print(f"\nPCA Summary:")
print("- PCA finds directions of maximum variance in data")
print("- Principal components are orthogonal (uncorrelated)")
print("- Reduces dimensionality while preserving most information")
print("- Useful for visualization, noise reduction, and compression")
print(f"- Original: {X_train.shape[1]} features → 2D: {pca_2d.explained_variance_ratio_.sum():.1%} variance retained")

# === EIGENVALUE ANALYSIS ===

print(f"\nEigenvalue Analysis:")
print(f"Eigenvalues (first 10): {pca.explained_variance_[:10]}")
print(f"Eigenvalue ratio PC1/PC2: {pca.explained_variance_[0]/pca.explained_variance_[1]:.2f}")
