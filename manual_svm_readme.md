# Manual SVM Implementation - README

## What is SVM (Support Vector Machine)?

SVM is a powerful machine learning algorithm used for **classification** and **regression** tasks. It works by finding the best **hyperplane** (decision boundary) that separates different classes with the maximum margin.

### Key Concepts:

- **Hyperplane**: A line (in 2D) or plane (in higher dimensions) that separates classes
- **Support Vectors**: Data points closest to the hyperplane that define the margin
- **Margin**: The distance between the hyperplane and the nearest data points
- **Decision Boundary**: The hyperplane that classifies new data points

## What is Needed for SVM?

### 1. **Training Data**

- Input features (X): Matrix of data points
- Labels (y): Class labels for each data point
- In our case: Cat images (label = -1) and Dog images (label = +1)

### 2. **Mathematical Components**

- **Weight Vector (w)**: Defines the orientation of the hyperplane
- **Bias Term (b)**: Shifts the hyperplane away from origin
- **Decision Function**: f(x) = w^T × x + b

### 3. **Training Process**

- Initialize weights and bias
- Iteratively update parameters using training data
- Minimize classification errors while maximizing margin

## How This SVM Code Works

### Step 1: Data Loading

```python
# Load 10 cat images and 10 dog images
cats = []  # Load cat images from 'Training Data/Cat'
dogs = []  # Load dog images from 'Training Data/Dog'
```

### Step 2: Data Preprocessing

```python
# Resize images to 8x8 pixels (64 features)
img = cv2.resize(img, (8, 8))
# Normalize pixel values to 0-1 range
img = img.flatten() / 255.0
```

### Step 3: Create Training Matrix

```python
X = np.array(cats + dogs)          # Shape: (20, 64)
y = np.array([0]*10 + [1]*10)      # Labels: 0=cat, 1=dog
y = np.where(y == 0, -1, 1)        # Convert to -1, +1 for SVM
```

### Step 4: Manual SVM Training

```python
w = np.zeros(64)  # Initialize weight vector
b = 0.0           # Initialize bias

for epoch in range(100):
    for i in range(20):
        # Calculate decision: w^T × x + b
        decision = np.dot(w, X[i]) + b

        # Check if misclassified (wrong side of margin)
        if y[i] * decision < 1:
            # Update weights: w = w + learning_rate × y × x
            w = w + 0.01 * y[i] * X[i]
            # Update bias: b = b + learning_rate × y
            b = b + 0.01 * y[i]
```

### Step 5: Prediction

```python
# For each test image:
prediction = np.dot(w, x_test) + b
result = "Cat" if prediction < 0 else "Dog"
```

## Manual Calculations in This Code

### 1. **Matrix Multiplication (Dot Product)**

```python
decision = np.dot(w, X[i]) + b
```

- **Mathematical form**: w₁×x₁ + w₂×x₂ + ... + w₆₄×x₆₄ + b
- **Purpose**: Calculate how far a point is from the decision boundary
- **Result**: Positive = Dog, Negative = Cat

### 2. **Weight Update Rule**

```python
w = w + 0.01 * y[i] * X[i]
```

- **Mathematical form**: w_new = w_old + α × y × x
- **α (learning rate)**: 0.01 (controls step size)
- **y**: True label (-1 for cat, +1 for dog)
- **x**: Image feature vector (64 pixel values)

### 3. **Bias Update Rule**

```python
b = b + 0.01 * y[i]
```

- **Mathematical form**: b_new = b_old + α × y
- **Purpose**: Shifts the hyperplane to better separate classes

### 4. **Margin Condition**

```python
if y[i] * decision < 1:
```

- **Mathematical form**: y × (w^T × x + b) < 1
- **Purpose**: Check if point is misclassified or within margin
- **Action**: Update weights only when condition is true

## Library Used Calculations

### 1. **NumPy Operations**

```python
np.dot(w, X[i])        # Efficient matrix multiplication
np.zeros(64)           # Initialize zero vector
np.array(cats + dogs)  # Convert list to NumPy array
np.where(y == 0, -1, 1) # Conditional array operation
```

### 2. **OpenCV Operations**

```python
cv2.imread(path, 0)    # Load image as grayscale
cv2.resize(img, (8,8)) # Resize image to 8x8 pixels
```

### 3. **Built-in Python**

```python
os.listdir('folder')   # List files in directory
img.flatten()          # Convert 2D array to 1D
sorted(filenames)      # Sort filenames alphabetically
```

## Mathematical Foundation

### SVM Optimization Problem

The SVM tries to solve:

```
Minimize: ||w||² / 2
Subject to: yᵢ(w^T×xᵢ + b) ≥ 1 for all training points
```

### Our Simplified Approach

Instead of complex optimization, we use:

```
If yᵢ(w^T×xᵢ + b) < 1:  # Point is misclassified or in margin
    w = w + α×yᵢ×xᵢ      # Move hyperplane toward correct side
    b = b + α×yᵢ         # Adjust bias
```

## Code Output Explanation

### Training Output

```
Training data: (20, 64)
Training done! Accuracy: 18/20 = 0.90
```

- **20 images**: 10 cats + 10 dogs
- **64 features**: 8×8 pixel values per image
- **90% accuracy**: 18 out of 20 training images classified correctly

### Prediction Output

```
0.jpg: Dog (score: 0.35, confidence: 0.35)
1.jpg: Cat (score: -0.64, confidence: 0.64)
```

- **Score**: Raw output of w^T×x + b
- **Positive score**: Dog prediction
- **Negative score**: Cat prediction
- **Confidence**: Absolute value of score (higher = more confident)

### Learned Parameters

```
Learned weights: [-0.467, -0.537, 0.601, ...]
Bias: 0.890
```

- **Weights**: 64 numbers showing importance of each pixel
- **Bias**: Offset value that shifts the decision boundary

## Summary

This manual SVM implementation demonstrates:

1. **Core SVM mathematics** using basic matrix operations
2. **Gradient-based learning** without complex optimization libraries
3. **Binary classification** (Cat vs Dog) using image pixel features
4. **Real predictions** on test images with confidence scores

The code shows how SVMs work "under the hood" using only fundamental mathematical operations!
