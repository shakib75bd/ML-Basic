import os
import cv2
import numpy as np

# Load cat images
cats = []
for i, f in enumerate(os.listdir('Training Data/Cat')[:10]):  # Only 10 images
    img = cv2.imread(f'Training Data/Cat/{f}', 0)
    img = cv2.resize(img, (8, 8))  # Very small: 8x8 = 64 pixels
    cats.append(img.flatten() / 255.0)  # Normalize to 0-1

# Load dog images
dogs = []
for i, f in enumerate(os.listdir('Training Data/Dog')[:10]):  # Only 10 images
    img = cv2.imread(f'Training Data/Dog/{f}', 0)
    img = cv2.resize(img, (8, 8))  # Very small: 8x8 = 64 pixels
    dogs.append(img.flatten() / 255.0)  # Normalize to 0-1

# Create training matrix
X = np.array(cats + dogs)  # 20 images x 64 pixels
y = np.array([0]*10 + [1]*10)  # 0=cat, 1=dog

print(f"Training data: {X.shape}")

# === MANUAL SVM MATRIX CALCULATION ===

# Convert labels to -1, +1
y = np.where(y == 0, -1, 1)

# Initialize weights and bias
w = np.zeros(64)  # 64 weights (one per pixel)
b = 0.0           # bias

# Simple training (more iterations for better learning)
for epoch in range(100):  # More training epochs
    for i in range(20):  # 20 training samples
        # Matrix calculation: decision = w^T * x + b
        decision = np.dot(w, X[i]) + b

        # If wrong prediction, update weights (smaller learning rate)
        if y[i] * decision < 1:
            w = w + 0.01 * y[i] * X[i]  # Smaller learning rate: 0.01
            b = b + 0.01 * y[i]         # Smaller learning rate: 0.01

# Check training accuracy
correct = 0
for i in range(20):
    decision = np.dot(w, X[i]) + b
    predicted = 1 if decision > 0 else -1
    if predicted == y[i]:
        correct += 1

print(f"Training done! Accuracy: {correct}/20 = {correct/20:.2f}")

# === PREDICT TEST IMAGES ===

print("\nPredictions for ALL test images:")
print("-" * 50)
for filename in sorted(os.listdir('TestData')):  # ALL images (removed [:8])
    if filename.endswith('.jpg'):
        # Load test image
        img = cv2.imread(f'TestData/{filename}', 0)
        img = cv2.resize(img, (8, 8))
        x_test = img.flatten() / 255.0

        # Matrix calculation: prediction = w^T * x + b
        prediction = np.dot(w, x_test) + b

        # Convert to class label (fixed logic)
        result = "Cat" if prediction < 0 else "Dog"
        confidence = abs(prediction)
        print(f"{filename}: {result} (score: {prediction:.2f}, confidence: {confidence:.2f})")

print(f"\nLearned weights: {w[:5]}... (showing first 5)")
print(f"Bias: {b:.3f}")
