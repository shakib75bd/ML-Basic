# SVD Implementation - README

## What is SVD (Singular Value Decomposition)?

SVD is a fundamental **matrix factorization** technique used for **dimensionality reduction**, **data compression**, **noise reduction**, and **matrix approximation**. It decomposes any matrix into three components that reveal the underlying structure and patterns in the data.

### Key Concepts:

- **Matrix Factorization**: Decomposes X = U × S × Vᵀ
- **Singular Values**: Measure importance of each component (diagonal of S)
- **Left Singular Vectors (U)**: Row patterns (sample relationships)
- **Right Singular Vectors (Vᵀ)**: Column patterns (feature relationships)
- **Low-Rank Approximation**: Optimal compression using fewer components

## What is Needed for SVD?

### 1. **Input Matrix**

- Data matrix X: Any real-valued matrix (m × n)
- No labels needed (unsupervised technique)
- In our case: 28 images × 256 pixel features

### 2. **Mathematical Components**

- **U Matrix**: Left singular vectors (m × min(m,n))
- **S Vector**: Singular values (min(m,n) values, sorted descending)
- **Vᵀ Matrix**: Right singular vectors (min(m,n) × n)
- **Orthogonality**: U and Vᵀ have orthonormal columns/rows

### 3. **Decomposition Process**

- Uses NumPy's optimized SVD implementation
- Automatically computes all three matrices
- Provides exact factorization: X = U @ diag(S) @ Vᵀ

## How This SVD Code Works

### Step 1: Data Loading and Preprocessing

```python
# Load cat and dog images
cats = []
for f in os.listdir('Training Data/Cat')[:15]:
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))  # 16x16 = 256 features
            cats.append(img.flatten() / 255.0)  # Normalize to 0-1
```

### Step 2: Apply SVD Decomposition

```python
# Full SVD using NumPy
U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
```

### Step 3: Analyze Components

```python
# Calculate explained variance from singular values
total_variance = np.sum(S**2)
explained_variance_ratio = (S**2) / total_variance
```

### Step 4: Dimensionality Reduction

```python
# Use TruncatedSVD for reduced dimensions
svd_2d = TruncatedSVD(n_components=2)
X_reduced = svd_2d.fit_transform(X_train)
```

## Library Functions Used

### 1. **NumPy SVD Functions**

```python
import numpy as np

U, S, Vt = np.linalg.svd(X, full_matrices=False)  # Full SVD decomposition
rank = np.linalg.matrix_rank(X)                   # Matrix rank
condition_number = S[0] / S[-1]                   # Condition number
frobenius_norm = np.linalg.norm(X, 'fro')         # Frobenius norm
```

### 2. **Scikit-Learn SVD Functions**

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=k)               # Create truncated SVD
X_reduced = svd.fit_transform(X)                 # Fit and transform
svd.explained_variance_ratio_                    # Variance explained
svd.singular_values_                             # Singular values
```

### 3. **Matrix Operations**

```python
# Reconstruction from components
X_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Reconstruction error
error = np.linalg.norm(X - X_approx, 'fro')
```

## Key Advantages of SVD

### 1. **Optimal Low-Rank Approximation**

- **Eckart-Young Theorem**: SVD provides the best possible rank-k approximation
- **Minimal Error**: Minimizes Frobenius norm of approximation error
- **Guaranteed Optimality**: No other method can do better

### 2. **Mathematical Guarantees**

```python
# Perfect reconstruction with all components
X_reconstructed = U @ np.diag(S) @ Vt
# Reconstruction error ≈ 0 (machine precision)
```

### 3. **Versatile Applications**

- **Data compression**: Remove less important components
- **Noise reduction**: Keep only significant components
- **Visualization**: Project to 2D/3D space
- **Feature extraction**: Identify key patterns

### 4. **Numerical Stability**

- Robust algorithms available in NumPy/SciPy
- Handles rank-deficient matrices
- Stable for ill-conditioned problems

## Mathematical Foundation

### SVD Decomposition Theorem

For any real matrix X (m × n), there exist orthogonal matrices U and V such that:

```
X = U Σ Vᵀ
```

Where:

- **U**: m × min(m,n) orthogonal matrix (UᵀU = I)
- **Σ**: min(m,n) × min(m,n) diagonal matrix with σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
- **Vᵀ**: min(m,n) × n orthogonal matrix (VVᵀ = I)

### Singular Value Properties

```
σᵢ = sqrt(λᵢ)  where λᵢ are eigenvalues of XᵀX
```

### Rank-k Approximation

The optimal rank-k approximation is:

```
X_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ = U_k Σ_k V_k^T
```

This minimizes: `||X - X_k||_F`

## How SVD Creates Matrix Decomposition

### What Does Each Component Represent?

1. **U Matrix (Left Singular Vectors)**

   - Each column represents a "sample pattern"
   - Shows relationships between different images
   - Captures how images cluster together

2. **S Vector (Singular Values)**

   - Importance weights for each pattern
   - Larger values = more important patterns
   - Ordered from most to least important

3. **Vᵀ Matrix (Right Singular Vectors)**
   - Each row represents a "feature pattern"
   - Shows which pixels tend to vary together
   - Captures spatial patterns in images

### SVD Decomposition Process

During SVD computation:

```python
# Step 1: Compute XᵀX and XXᵀ
XTX = X.T @ X      # Feature covariance-like matrix
XXT = X @ X.T      # Sample covariance-like matrix

# Step 2: Find eigendecompositions
# V comes from eigenvectors of XᵀX
# U comes from eigenvectors of XXᵀ
# S comes from square roots of eigenvalues

# Step 3: Ensure proper ordering and signs
# Singular values in descending order
# Consistent sign conventions
```

### In Our Cat vs Dog Example

```python
# Results from our data:
print(f"Original matrix: {X_train.shape}")      # (28, 256)
print(f"U matrix: {U.shape}")                   # (28, 28) - sample patterns
print(f"S vector: {S.shape}")                   # (28,) - importance weights
print(f"Vᵀ matrix: {Vt.shape}")                 # (28, 256) - feature patterns
```

**Why these dimensions?**

- **28 samples**: Only 28 images, so rank ≤ 28
- **256 features**: Each pixel is a feature
- **Rank = 28**: All 28 components are meaningful

### SVD Compression Results

Our analysis shows excellent compression capabilities:

| Components | Variance Explained | Compression Ratio | Quality                     |
| ---------- | ------------------ | ----------------- | --------------------------- |
| 2          | 86.4%              | 128.0x            | Excellent for visualization |
| 5          | 91.2%              | 51.2x             | Very good compression       |
| 10         | 95.0%              | 25.6x             | High quality                |
| 20         | 98.8%              | 12.8x             | Near-perfect                |

**SVD vs PCA Comparison:**

- **SVD 2D**: 86.4% variance (much better than PCA's 35.4%)
- **SVD 5D**: 91.2% variance (vs PCA's 58.9%)
- **SVD advantages**: No centering required, works directly on data matrix

### How SVD Processes Images

When applying SVD to a test image:

```python
# Project new image onto SVD space
x_test_svd = svd_model.transform([x_test])
```

**Behind the scenes:**

1. **Direct Transformation**: Use learned Vᵀ matrix
2. **Projection**: `x_reduced = x_test @ Vᵀ[:k, :].T`
3. **No Preprocessing**: SVD works on original data (no centering required)

### Reconstruction Quality Analysis

SVD provides excellent reconstruction:

| Components | Reconstruction Error | Variance Retained |
| ---------- | -------------------- | ----------------- |
| 2          | 0.034                | 86.4%             |
| 5          | 0.022                | 91.2%             |
| 10         | 0.013                | 95.0%             |
| 20         | 0.003                | 98.8%             |

**Much better than PCA reconstruction!**

## Code Output Explanation

### SVD Decomposition Results

```
U matrix shape (left singular vectors): (28, 28)
S vector shape (singular values): (28,)
Vt matrix shape (right singular vectors): (28, 256)
```

- **U**: Sample relationships (which images are similar)
- **S**: Component importance (how much each pattern contributes)
- **Vᵀ**: Feature relationships (which pixels vary together)

### Singular Value Analysis

```
Largest singular value: 38.834
Smallest singular value: 1.165
Condition number: 3.33e+01
```

- **Large σ₁**: Strong first pattern dominates
- **Condition number**: Matrix is well-conditioned (not singular)
- **Decay rate**: How quickly importance drops off

### Test Image Transformations

```
0.jpg: SVD1= 6.429, SVD2=-0.073
1.jpg: SVD1= 8.743, SVD2=-0.859
```

- **SVD coordinates**: Position in reduced singular value space
- **Interpretation**: Similar to PCA but often better separated
- **Range**: Typically larger magnitudes than PCA

### Mathematical Verification

```
Full reconstruction error (should be ~0): 9.77e-15
U orthogonality error: 2.22e-15
Vt orthogonality error: 2.66e-15
```

- **Perfect reconstruction**: Confirms SVD correctness
- **Orthogonality**: U and Vᵀ have orthonormal columns/rows
- **Machine precision**: Errors at floating-point limit

## SVD vs PCA vs LDA vs SVM Comparison

| Aspect               | SVD                           | PCA                           | LDA                        | SVM                       |
| -------------------- | ----------------------------- | ----------------------------- | -------------------------- | ------------------------- |
| **Purpose**          | Matrix factorization          | Dimensionality reduction      | Classification + reduction | Classification            |
| **Supervision**      | Unsupervised                  | Unsupervised                  | Supervised                 | Supervised                |
| **Output**           | U, S, Vᵀ matrices             | Transformed coordinates       | Class probabilities        | Class predictions         |
| **Preprocessing**    | None required                 | Centering required            | Centering + scaling        | Scaling recommended       |
| **Optimality**       | Optimal low-rank approx       | Optimal variance preservation | Optimal class separation   | Optimal margin separation |
| **Interpretability** | High (clear components)       | High (principal directions)   | High (class differences)   | Medium (support vectors)  |
| **Compression**      | Excellent (86.4% in 2D)       | Good (35.4% in 2D)            | N/A                        | N/A                       |
| **Noise Handling**   | Excellent (natural filtering) | Good                          | Fair                       | Robust to outliers        |

## When to Use SVD

### **Use SVD when:**

- Need optimal matrix approximation
- Want to remove noise from data
- Require data compression
- Working with recommendation systems
- Need to analyze matrix structure
- Want interpretable decomposition

### **Don't use SVD when:**

- Working with very sparse matrices (use specialized methods)
- Need real-time processing (can be computationally expensive)
- Data has categorical features
- Only need classification (use SVM/LDA instead)

## Practical Applications of Our SVD Analysis

### 1. **Data Compression**

```
Original: 28 × 256 = 7,168 values
Rank-5 SVD: 28×5 + 5 + 5×256 = 1,425 values (5× smaller, 91.2% quality)
```

### 2. **Noise Reduction**

```
Keep top 10 components: 95.0% signal, remove 5% noise
Excellent for cleaning image data
```

### 3. **Feature Extraction**

```
256 pixel features → 10 SVD features
Much better than PCA for preserving information
```

### 4. **Data Analysis**

```
U matrix: Which images are similar?
Vᵀ matrix: Which pixels are important?
S vector: How important is each pattern?
```

## Advanced SVD Concepts

### 1. **Matrix Rank and Condition Number**

```python
rank = np.linalg.matrix_rank(X)                # Effective rank
condition_number = S[0] / S[-1]                # Numerical stability
```

**In our case:**

- **Rank = 28**: Matrix has full rank (28 independent samples)
- **Condition number = 33.3**: Well-conditioned (stable computation)

### 2. **Nuclear Norm and Sparsity**

```python
nuclear_norm = np.sum(S)                       # Sum of singular values
frobenius_norm = np.sqrt(np.sum(S**2))        # Energy of matrix
```

### 3. **Truncation Strategy**

```python
# Keep components that explain 95% of variance
cumulative_variance = np.cumsum(S**2) / np.sum(S**2)
k = np.argmax(cumulative_variance >= 0.95) + 1
```

## Summary

This SVD implementation demonstrates:

1. **Optimal matrix factorization** using fundamental linear algebra
2. **Superior compression** compared to PCA (86.4% vs 35.4% in 2D)
3. **Mathematical rigor** with perfect reconstruction verification
4. **Versatile analysis** of both sample and feature patterns
5. **Practical applications** for compression, denoising, and visualization

SVD provides the theoretical foundation for many machine learning techniques and offers unmatched optimality guarantees for matrix approximation!

## Key Takeaways

- **SVD is optimal**: Provides best possible low-rank matrix approximation
- **No preprocessing needed**: Works directly on data matrix (unlike PCA)
- **Three interpretable components**: U (samples), S (importance), Vᵀ (features)
- **Superior compression**: Much better information preservation than PCA
- **Mathematical foundation**: Underlies many advanced ML techniques
- **Versatile tool**: Useful for compression, denoising, analysis, and visualization
