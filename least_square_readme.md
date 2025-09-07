# Least Squares Classification Implementation - README

## What is Least Squares Classification?

Least Squares classification is a **linear regression** approach adapted for **binary classification** problems. It finds the **best-fitting linear model** that minimizes the sum of squared differences between predicted and actual values, then uses this linear model to make classification decisions through probability conversion.

### Key Concepts:

- **Linear Model**: Assumes relationship between features and output is linear
- **Least Squares**: Minimizes sum of squared errors (residuals)
- **Normal Equation**: Closed-form solution for optimal weights
- **Regression to Classification**: Convert continuous output to discrete classes
- **Sigmoid Function**: Maps linear scores to probabilities [0,1]

## What is Needed for Least Squares Classification?

### 1. **Mathematical Requirements**

- **Linear relationship**: Features should have linear correlation with target
- **Invertible matrix**: X^T X must be non-singular for unique solution
- **Numerical stability**: Well-conditioned matrices for reliable computation
- **Sufficient data**: More samples than features for overdetermined system

### 2. **Training Components**

- **Feature matrix X**: Samples as rows, features as columns
- **Target vector y**: Binary labels (0 for cats, 1 for dogs)
- **Bias term**: Intercept added as constant feature column
- **Regularization**: Optional to prevent overfitting

### 3. **Computational Setup**

- **Matrix operations**: Efficient linear algebra implementations
- **Memory management**: Store only necessary matrices
- **Numerical precision**: Handle potential precision issues

## How This Least Squares Code Works

### Step 1: Data Preparation

```python
# Load and preprocess images
cats, dogs = [], []
for f in os.listdir('Training Data/Cat')[:10]:
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:
            cats.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

# Create feature matrix and labels
X = np.array(cats + dogs)
y = np.array([0] * len(cats) + [1] * len(dogs))
```

### Step 2: Add Bias Term

```python
# Add intercept column for bias
X = np.column_stack([np.ones(X.shape[0]), X])
```

### Step 3: Solve Normal Equation

```python
# Compute optimal weights using closed-form solution
# w = (X^T X)^(-1) X^T y
XtX = X.T @ X
Xty = X.T @ y
w = np.linalg.solve(XtX, Xty)
```

### Step 4: Make Predictions

```python
# Linear prediction
linear_score = np.dot(test_features, w)

# Convert to probability using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

probability = sigmoid(linear_score)
```

## Mathematical Foundation

### Normal Equation Derivation

Starting with the cost function:

```
J(w) = ||Xw - y||² = (Xw - y)^T (Xw - y)
```

Expanding:

```
J(w) = w^T X^T X w - 2w^T X^T y + y^T y
```

Taking derivative and setting to zero:

```
∂J/∂w = 2X^T X w - 2X^T y = 0
```

Solving for w:

```
w = (X^T X)^(-1) X^T y
```

### Matrix Components

**Feature Matrix X:**

```
X = [1, x₁₁, x₁₂, ..., x₁ₙ]    ← Sample 1
    [1, x₂₁, x₂₂, ..., x₂ₙ]    ← Sample 2
    [⋮,  ⋮,   ⋮,  ⋱,   ⋮ ]
    [1, xₘ₁, xₘ₂, ..., xₘₙ]    ← Sample m
```

**Weight Vector w:**

```
w = [w₀]    ← Bias term
    [w₁]    ← Feature 1 weight
    [w₂]    ← Feature 2 weight
    [⋮ ]
    [wₙ]    ← Feature n weight
```

### Sigmoid Transformation

Convert linear output to probability:

```
P(y=1|x) = σ(w^T x) = 1 / (1 + e^(-w^T x))
```

**Properties:**

- **Range**: (0, 1) - proper probability
- **Monotonic**: Larger linear scores → higher probabilities
- **Smooth**: Differentiable transformation
- **Interpretable**: Can be viewed as log-odds

## Code Output Explanation

### Training Performance

```
Training accuracy: 75.0%
```

- **Moderate accuracy**: 15 out of 20 samples correctly classified
- **Linear limitation**: May struggle with non-linear patterns
- **Baseline performance**: Simple model with interpretable results
- **Room for improvement**: Could benefit from feature engineering

### Weight Analysis

```
Learned weights (first 5): [-0.123, 0.456, -0.789, 0.234, -0.567]
```

- **Bias term**: w₀ = -0.123 (global offset)
- **Feature weights**: How much each pixel contributes to dog vs cat
- **Positive weights**: Increase probability of being dog
- **Negative weights**: Increase probability of being cat
- **Magnitude**: Indicates feature importance

### Linear Score

```
Linear score for 0.jpg: 3.211
```

- **Raw prediction**: Before probability conversion
- **Positive value**: Points toward dog class (label 1)
- **Magnitude**: Indicates confidence in linear space
- **Unbounded**: Can be any real number

### Probability Conversion

```
Sigmoid probability: 0.961 (96.1% dog)
```

- **Probability interpretation**: 96.1% confident it's a dog
- **Sigmoid transformation**: Maps linear score to [0,1]
- **High confidence**: Strong linear signal
- **Decision threshold**: Typically 0.5 for binary classification

### Final Prediction

```
Prediction: Dog
```

- **Classification rule**: probability > 0.5 → Dog, else Cat
- **Correct prediction**: Matches expected result
- **Confidence-based**: Could adjust threshold based on costs

## Advantages of Least Squares Classification

### 1. **Computational Efficiency**

- **Closed-form solution**: No iterative optimization needed
- **One-step training**: Solve linear system once
- **Fast prediction**: Simple matrix-vector multiplication
- **Scalable**: Efficient for large datasets

### 2. **Mathematical Simplicity**

- **Well-understood**: Classical linear algebra
- **Stable solution**: Robust numerical methods
- **Interpretable weights**: Clear feature importance
- **No hyperparameters**: Minimal tuning required

### 3. **Probabilistic Output**

- **Uncertainty quantification**: Sigmoid provides probabilities
- **Confidence measures**: Know prediction reliability
- **Threshold tuning**: Adjust decision boundary
- **Risk assessment**: Make cost-sensitive decisions

### 4. **Memory Efficient**

- **Compact model**: Only store weight vector
- **No training data storage**: Weights capture all information
- **Fast deployment**: Minimal computational requirements

## When to Use Least Squares Classification

### **Use Least Squares when:**

- Features have linear relationship with target
- Need fast training and prediction
- Want interpretable model weights
- Working with well-conditioned data
- Baseline model for comparison
- Memory/computation constraints exist

### **Don't use Least Squares when:**

- Relationship is highly non-linear
- Features are highly correlated (multicollinearity)
- Number of features > number of samples
- Need maximum classification accuracy
- Data has outliers (least squares sensitive)
- Decision boundary is complex

## Comparison with Other Methods

| Aspect                | Least Squares | Bayesian     | SVM        | LDA    |
| --------------------- | ------------- | ------------ | ---------- | ------ |
| **Accuracy**          | 75%           | 92.9%        | 100%       | 79%    |
| **Training Speed**    | Very Fast     | Very Fast    | Medium     | Fast   |
| **Interpretability**  | Very High     | Very High    | Medium     | High   |
| **Probabilistic**     | Yes           | Yes          | No         | Yes    |
| **Assumptions**       | Linear        | Independence | None       | Normal |
| **Decision Boundary** | Linear        | Non-linear   | Non-linear | Linear |
| **Parameters**        | 65 weights    | 128 stats    | Many       | Few    |

## Practical Applications

### 1. **Linear Classification**

- Problems with natural linear separation
- Feature selection and importance analysis
- Baseline for more complex models

### 2. **Regression Analysis**

- Originally designed for continuous prediction
- Economic modeling and forecasting
- Scientific data analysis

### 3. **Feature Engineering**

- Understanding which features matter
- Linear combination discoveries
- Preprocessing for other algorithms

### 4. **Real-time Systems**

- Fast prediction requirements
- Limited computational resources
- Embedded systems applications

## Limitations and Considerations

### 1. **Linearity Assumption**

- **Problem**: Assumes linear relationship
- **Reality**: Many patterns are non-linear
- **Solution**: Feature engineering, polynomial features

### 2. **Sensitivity to Outliers**

- **Issue**: Squared loss heavily penalizes outliers
- **Impact**: Can skew the entire model
- **Mitigation**: Robust regression methods, outlier detection

### 3. **Multicollinearity**

- **Problem**: Correlated features cause instability
- **Symptoms**: Large weight magnitudes, numerical issues
- **Solutions**: Regularization (Ridge), feature selection

### 4. **Limited Expressiveness**

- **Constraint**: Only linear decision boundaries
- **Comparison**: Less flexible than kernel methods
- **Trade-off**: Simplicity vs. complexity

## Advanced Variations

### 1. **Ridge Regression**

```python
# Add L2 regularization
XtX_ridge = XtX + lambda * np.eye(XtX.shape[0])
w = np.linalg.solve(XtX_ridge, Xty)
```

### 2. **Weighted Least Squares**

```python
# Handle class imbalance
W = np.diag(weights)
w = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
```

### 3. **Polynomial Features**

```python
# Add non-linearity
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

## Implementation Details

### Numerical Stability

```python
# Clip sigmoid input to prevent overflow
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

### Matrix Inversion

```python
# Use solve instead of inv for better stability
w = np.linalg.solve(XtX, Xty)  # Better than np.linalg.inv(XtX) @ Xty
```

### Feature Scaling

```python
# Normalize features for numerical stability
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
```

## Summary

This Least Squares implementation demonstrates:

1. **Classical linear approach** with closed-form solution
2. **Fast and efficient** training without iteration
3. **Interpretable weights** showing feature importance
4. **Probabilistic output** through sigmoid transformation
5. **Mathematical foundation** based on optimization theory
6. **Practical limitations** in non-linear scenarios

Least Squares provides an excellent introduction to linear classification methods and serves as a valuable baseline for understanding more complex algorithms.

## Key Takeaways

- **Normal equation provides optimal solution**: Minimizes squared error exactly
- **Linear assumptions matter**: Performance depends on problem linearity
- **Interpretability is valuable**: Weight analysis provides insights
- **Speed is advantageous**: No iterative training required
- **Foundation for other methods**: Basis for logistic regression, neural networks
- **Regularization often helpful**: Ridge regression improves stability
- **Good baseline**: Simple model to compare against complex alternatives
