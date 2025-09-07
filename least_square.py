import os
import cv2
import numpy as np

# Load training data
cats, dogs = [], []
for f in os.listdir('Training Data/Cat')[:10]:
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:
            cats.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

for f in os.listdir('Training Data/Dog')[:10]:
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Dog/{f}', 0)
        if img is not None:
            dogs.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

# Create training matrix and labels
X_train = np.array(cats + dogs)  # Training features
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=cat, 1=dog

# Add bias term (column of ones)
X_train_bias = np.column_stack([np.ones(len(X_train)), X_train])

# Load test image
img = cv2.imread('TestData/0.jpg', 0)
test_img = cv2.resize(img, (8, 8)).flatten() / 255.0
x_test_bias = np.concatenate([[1], test_img])  # Add bias term

# Least Squares Solution
# Solve: X * w = y using Normal Equation: w = (X^T * X)^(-1) * X^T * y
XTX = X_train_bias.T @ X_train_bias
XTy = X_train_bias.T @ y_train
weights = np.linalg.solve(XTX, XTy)

# Make prediction
prediction_score = x_test_bias @ weights
prediction_prob = 1 / (1 + np.exp(-prediction_score))  # Sigmoid for probability

# Calculate training accuracy
train_predictions = X_train_bias @ weights
train_probs = 1 / (1 + np.exp(-train_predictions))
train_classes = (train_probs > 0.5).astype(int)
training_accuracy = np.mean(train_classes == y_train)

print("Least Squares Classification for 0.jpg:")
print(f"Training accuracy: {training_accuracy:.3f}")
print(f"Linear score: {prediction_score:.3f}")
print(f"Probability (sigmoid): {prediction_prob:.3f}")
print(f"Cat probability: {1 - prediction_prob:.3f}")
print(f"Dog probability: {prediction_prob:.3f}")
print(f"Prediction: {'Cat' if prediction_prob < 0.5 else 'Dog'}")
print()
print("Least Squares Method:")
print("- Finds linear weights w that minimize ||Xw - y||²")
print("- Uses Normal Equation: w = (X^T X)^(-1) X^T y")
print("- Linear decision boundary: w₀ + w₁x₁ + w₂x₂ + ... = 0.5")
print("- Sigmoid converts linear score to probability")
